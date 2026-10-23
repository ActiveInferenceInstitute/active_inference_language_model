"""HTTP server and API endpoint for the AILM Interactive Dashboard.

Serves the single-page HTML/JS/CSS frontend and exposes:
- POST /api/infer: Hierarchical active inference state filtering over raw text.
- GET /api/static_matrices: Full matrix dictionaries for deep inspection.

The numerical engine is not imported at module load: the default
`DashboardServices` adapters import it on the first computation request, so
static assets and configuration can be served before any numpy/jax cost.
This module is not an entry point — run ``python -m ailm.dashboard``, which
owns configuration validation and bind precedence.
"""

from __future__ import annotations

import hmac
import json
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingTCPServer
from threading import BoundedSemaphore, Lock
from typing import Any
from urllib.parse import urlsplit

from ailm.dashboard.config import (
    DashboardConfig,
    is_loopback_host,
    load_config,
    require_loopback_host,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
EXPLORER_PAGE = Path(__file__).resolve().parents[3] / "output" / "web" / "ailm.html"
AUTH_COOKIE = "ailm_dashboard_token"
GENERIC_INTERNAL_ERROR = "internal server error"

#: The configuration this process serves and enforces. Set by ``run_dashboard``;
#: the module-level shipped default keeps the handler usable for embedding and
#: direct construction. Environment parsing belongs at launch time so
#: ``ailm.dashboard.__main__`` can report a refused override as a clean exit 2
#: instead of raising while the package itself is still importing.
CONFIG: DashboardConfig = DashboardConfig()

#: Compatibility alias preserving this module's historical public surface
#: (the limit lived here before ``config.py``). It is a view onto the
#: configuration, not a second place the limit is written. No repository
#: caller or test uses it today; removal is a pending cutover decision, not
#: a dependency.
MAX_REQUEST_BYTES = CONFIG.max_request_bytes


def _infer_sentence_hierarchy(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from ailm.dashboard.engine import infer_sentence_hierarchy

    return infer_sentence_hierarchy(*args, **kwargs)


def _converse(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from ailm.dashboard.engine import converse

    return converse(*args, **kwargs)


def _get_dashboard_models() -> dict[str, Any]:
    from ailm.dashboard.engine import get_dashboard_models

    return get_dashboard_models()


@dataclass(frozen=True, slots=True)
class DashboardServices:
    """Computation boundary used by an HTTP handler.

    The default loads the production engine on the first computation request.
    Static assets and configuration need no numerical imports. An embedding may supply the same
    interface for another local engine, and tests can exercise transport error
    handling with a deliberately failing implementation without replacing
    module globals.
    """

    infer: Callable[..., dict[str, Any]] = _infer_sentence_hierarchy
    chat: Callable[..., dict[str, Any]] = _converse
    models: Callable[[], dict[str, Any]] = _get_dashboard_models


DEFAULT_SERVICES = DashboardServices()


def _as_int(value: Any, default: int) -> int:
    """Coerce a wire value to int, falling back rather than raising."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_history(value: Any, *, max_turns: int) -> list[dict[str, Any]]:
    """Coerce a wire transcript into turns, dropping anything malformed.

    A chat client can send anything; the engine replays whatever it is given, so
    the shape is normalized once here rather than trusted. Over-long transcripts
    are truncated at the source too — the engine caps as well, but a request
    carrying tens of thousands of turns should not be parsed in full first.
    """
    if not isinstance(value, list):
        return []
    turns: list[dict[str, Any]] = []
    for entry in value[-int(max_turns) :]:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        if role not in ("user", "model"):
            continue
        turns.append({"role": role, "text": str(entry.get("text", ""))})
    return turns


def _as_float(value: Any, default: float) -> float:
    """Coerce a wire value to float, falling back rather than raising."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return default if parsed != parsed else parsed  # reject NaN


class DashboardHandler(SimpleHTTPRequestHandler):
    """Custom request handler serving dashboard static assets and REST API."""

    def __init__(
        self,
        *args,
        services: DashboardServices = DEFAULT_SERVICES,
        config: DashboardConfig | None = None,
        explorer_page: Path = EXPLORER_PAGE,
        **kwargs,
    ):
        self.explorer_page = explorer_page
        self.services = services
        self.config = CONFIG if config is None else config
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def end_headers(self) -> None:
        """Forbid caching of the static tree.

        `SimpleHTTPRequestHandler` sends `Last-Modified` and honors
        `If-Modified-Since`, which is correct for a file server and wrong for
        this one: the browser then keeps a stylesheet or an ES module across
        edits, and a page verified after a change is a page running the code
        from before it. That failure is silent in both directions — a fix looks
        broken, and a break looks fixed — and no test can see it, because the
        stale copy lives in the browser and every test reads the file.

        Measured live: an edited `dashboard.css` was served fresh by this
        server (byte-identical to disk) while the loaded page's CSSOM still
        held the previous rules, through a reload and a cache-busted URL. The
        query string busts the document; it does not bust its subresources.

        There is nothing to trade off. The assets are local, tiny, and read
        once per launch.
        """
        self.send_header("Cache-Control", "no-store, must-revalidate")
        if getattr(self, "_issue_auth_cookie", False):
            self.send_header(
                "Set-Cookie",
                f"{AUTH_COOKIE}={self.config.api_token}; HttpOnly; SameSite=Strict; Path=/",
            )
        super().end_headers()

    def _presented_token(self) -> str | None:
        authorization = self.headers.get("Authorization", "")
        scheme, _, credential = authorization.partition(" ")
        if scheme.lower() == "bearer" and credential:
            return credential.strip()
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except Exception:
            return None
        morsel = cookie.get(AUTH_COOKIE)
        return morsel.value if morsel is not None else None

    def _origin_is_same_loopback_server(self) -> bool:
        """Accept no browser origin except the exact loopback request origin.

        Non-browser clients normally omit ``Origin`` and remain supported, but
        they must still authenticate. Browsers attach ``Origin`` to a
        cross-origin POST, including a CORS-simple ``text/plain`` request, so a
        hostile page cannot spend local inference resources as blind CSRF.
        """
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        parsed = urlsplit(origin)
        request_host = self.headers.get("Host", "").lower()
        return (
            parsed.scheme == "http"
            and parsed.netloc.lower() == request_host
            and parsed.hostname is not None
            and is_loopback_host(parsed.hostname)
        )

    def _authorize_api(self) -> bool:
        presented = self._presented_token()
        if presented is None or not hmac.compare_digest(presented, self.config.api_token):
            self._send_json({"error": "authentication required"}, HTTPStatus.UNAUTHORIZED)
            return False
        if not self._origin_is_same_loopback_server():
            self._send_json({"error": "cross-origin request refused"}, HTTPStatus.FORBIDDEN)
            return False
        server = self.server
        consume = getattr(server, "consume_rate_slot", None)
        if consume is not None and not consume(self.client_address[0]):
            self._send_json({"error": "request rate exceeded"}, HTTPStatus.TOO_MANY_REQUESTS)
            return False
        return True

    def do_POST(self) -> None:
        if self.path not in ("/api/infer", "/api/chat"):
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")
            return
        if not self._authorize_api():
            return
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            self._send_json(
                {"error": "Content-Type must be application/json"},
                status=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            )
            return

        # The Content-Length parse and the body read were both outside the
        # try: a non-numeric header raised straight out of the handler, and an
        # oversized one had the server allocate whatever the client claimed.
        try:
            declared = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            self._send_json({"error": "malformed Content-Length"}, status=HTTPStatus.BAD_REQUEST)
            return
        if declared < 0 or declared > self.config.max_request_bytes:
            self._send_json(
                {"error": f"request body exceeds {self.config.max_request_bytes} bytes"},
                status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return

        try:
            body = self.rfile.read(declared).decode("utf-8")
            data = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json({"error": "bad request: malformed JSON"}, status=HTTPStatus.BAD_REQUEST)
            return
        if not isinstance(data, dict):
            self._send_json(
                {"error": "bad request: body must be a JSON object"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        try:
            if self.path == "/api/chat":
                # narrative_phase is optional here on purpose: left out, the
                # engine derives it from the turn index, which is the whole
                # reason Level 6 means anything in a conversation.
                phase = data.get("narrative_phase")
                result = self.services.chat(
                    str(data.get("message", "")),
                    history=_as_history(
                        data.get("history"), max_turns=self.config.max_history_turns
                    ),
                    n_words=max(
                        0,
                        min(
                            _as_int(data.get("n_words"), self.config.default_reply_words),
                            self.config.max_continuation_words,
                        ),
                    ),
                    narrative_phase=None if phase is None else _as_int(phase, 0),
                    narrative_alpha=_as_float(
                        data.get("narrative_alpha"), self.config.default_narrative_alpha
                    ),
                    max_history_turns=self.config.max_history_turns,
                    max_continuation_words=self.config.max_continuation_words,
                )
            else:
                result = self.services.infer(
                    str(data.get("text", "")),
                    n_continuation_words=max(
                        0,
                        min(
                            _as_int(data.get("n_words"), self.config.default_reply_words),
                            self.config.max_continuation_words,
                        ),
                    ),
                    narrative_phase=_as_int(data.get("narrative_phase"), 0),
                    narrative_alpha=_as_float(
                        data.get("narrative_alpha"), self.config.default_narrative_alpha
                    ),
                    max_continuation_words=self.config.max_continuation_words,
                )
            self._send_json(result)
        except Exception as e:
            logger.exception("Inference error: %s", e)
            self._send_json(
                {"error": GENERIC_INTERNAL_ERROR}, status=HTTPStatus.INTERNAL_SERVER_ERROR
            )

    def do_OPTIONS(self) -> None:
        """Refuse cross-origin preflights; same-origin callers need none."""
        if not self.path.startswith("/api/"):
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")
            return
        if not self._authorize_api():
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _serve_explorer(self) -> None:
        """Serve only the generated explorer, never an arbitrary repository path."""
        try:
            page = self.explorer_page.read_bytes()
        except FileNotFoundError:
            self._send_json(
                {
                    "error": "Explorer is unavailable. From the checkout root, run "
                    ".venv/bin/python scripts/export_web_model.py then "
                    ".venv/bin/python scripts/build_web_ailm.py; retry this view."
                },
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return
        except OSError:
            logger.exception("Cannot read generated explorer")
            self._send_json(
                {"error": "Explorer could not be read; check the generated artifact permissions."},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def do_GET(self) -> None:
        if urlsplit(self.path).path == "/explorer/":
            self._serve_explorer()
            return
        if self.path.startswith("/api/") and not self._authorize_api():
            return
        if self.path == "/api/config":
            # The page reads its bounds from the object this server enforces
            # them with, so a changed ceiling cannot be honored in one place and
            # ignored in the other.
            self._send_json(self.config.to_public_dict())
            return
        if self.path == "/api/static_matrices":
            try:
                models = self.services.models()
                payload = {
                    "A_letter": models["A_letter"].tolist(),
                    "A_g2p": models["A_g2p"].tolist(),
                    "B_g2p": models["B_g2p"].tolist(),
                    "D_g2p": models["D_g2p"].tolist(),
                    "A_case": models["A_case"].tolist(),
                    "B_case": models["B_case"].tolist(),
                    "D_case": models["D_case"].tolist(),
                    "clause_prior": models["clause_models"]["type_prior"].tolist(),
                }
                self._send_json(payload)
            except Exception as e:
                logger.exception("Matrix dump error: %s", e)
                self._send_json(
                    {"error": GENERIC_INTERNAL_ERROR}, status=HTTPStatus.INTERNAL_SERVER_ERROR
                )
        else:
            # Fallback to serving static frontend
            self._issue_auth_cookie = True
            super().do_GET()

    def _send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        response_bytes = json.dumps(data, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)


class ReusableThreadingServer(ThreadingTCPServer):
    """Threaded local server with a hard concurrency and socket-time ceiling."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address,
        request_handler_class,
        *,
        max_workers: int | None = None,
        max_requests_per_minute: int | None = None,
        request_timeout: float | None = None,
    ):
        self.max_workers = CONFIG.max_concurrent_requests if max_workers is None else max_workers
        self.max_requests_per_minute = (
            CONFIG.max_requests_per_minute
            if max_requests_per_minute is None
            else max_requests_per_minute
        )
        self.request_timeout = (
            CONFIG.request_timeout_seconds if request_timeout is None else request_timeout
        )
        self._worker_slots = BoundedSemaphore(self.max_workers)
        self._rate_lock = Lock()
        self._request_times: dict[str, deque[float]] = defaultdict(deque)
        super().__init__(server_address, request_handler_class)

    def consume_rate_slot(self, client: str) -> bool:
        """Atomically spend one authenticated request from a rolling minute."""
        now = time.monotonic()
        cutoff = now - 60.0
        with self._rate_lock:
            timestamps = self._request_times[client]
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()
            if len(timestamps) >= self.max_requests_per_minute:
                return False
            timestamps.append(now)
            return True

    def get_request(self):
        request, client_address = super().get_request()
        request.settimeout(self.request_timeout)
        return request, client_address

    def process_request(self, request, client_address) -> None:
        self._worker_slots.acquire()
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._worker_slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._worker_slots.release()


def run_dashboard(
    host: str | None = None, port: int | None = None, config: DashboardConfig | None = None
) -> None:
    """Launch the AILM dashboard server.

    Args:
        host: Override the configured bind address.
        port: Override the configured port.
        config: Configuration to serve and enforce. Defaults to the environment.
    """
    global CONFIG
    CONFIG = config or load_config()
    host = host if host is not None else CONFIG.host
    port = port if port is not None else CONFIG.port
    require_loopback_host(host)
    server_address = (host, port)
    httpd = ReusableThreadingServer(
        server_address,
        partial(DashboardHandler, config=CONFIG),
        max_workers=CONFIG.max_concurrent_requests,
        max_requests_per_minute=CONFIG.max_requests_per_minute,
        request_timeout=CONFIG.request_timeout_seconds,
    )
    print("==================================================")
    print("AILM Multimodal Interactive Dashboard live at:")
    print(f"http://{host}:{port}/")
    print("==================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard server...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    # The launch surface is `python -m ailm.dashboard`, which validates the
    # configuration and resolves bind precedence before any socket opens.
    # This module was once runnable itself, on a hardcoded 8080 with the
    # environment discarded — the trap the entry point exists to prevent —
    # and a silent no-op would be its own trap, so it refuses instead.
    raise SystemExit(
        "ailm.dashboard.server is not an entry point; run `python -m ailm.dashboard`, "
        "which validates configuration and bind precedence before opening a socket"
    )
