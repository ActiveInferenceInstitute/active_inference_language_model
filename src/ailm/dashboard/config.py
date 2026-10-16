"""Every tunable the dashboard has, in one place, with one default each.

## Why this module exists

The limits were spread across three files and two languages. ``MAX_SPOKEN_CHARS``
lived in a ``<script>`` block, ``MAX_REQUEST_BYTES`` in the server, the rollout
ceiling in the engine, and the default port in ``__main__``. Two of them were
also written a second time as HTML attributes: the reply-length input carried
``max="200"`` and the alpha slider ``min="0" max="1"``, transcribed by hand from
constants they had no link to. A limit written twice is a limit that will
disagree with itself, and the browser copy is the one nobody tests.

So: one dataclass, one default per field, and ``GET /api/config`` serves it to
the page. The front end reads its bounds from the same object the server
enforces them with, which means a changed ceiling cannot be honored in one place
and ignored in the other.

## Overriding

Every field reads an environment variable named ``AILM_DASHBOARD_<FIELD>``,
uppercased. Values that do not parse are refused rather than silently defaulted,
because a typo in a limit is exactly the case where a quiet fallback hides the
mistake:

    AILM_DASHBOARD_PORT=9000 AILM_DASHBOARD_MAX_SPOKEN_CHARS=120 \\
        .venv/bin/python -m ailm.dashboard

``DashboardConfig`` is frozen. Nothing mutates a live configuration; a caller
that needs different settings builds a different object.
"""

from __future__ import annotations

import ipaddress
import math
import os
import secrets
from dataclasses import asdict, dataclass, field, fields
from typing import Any

__all__ = [
    "ENV_PREFIX",
    "DashboardConfig",
    "config_from_env",
    "is_loopback_host",
    "load_config",
    "require_loopback_host",
]

#: Environment-variable prefix. ``port`` reads ``AILM_DASHBOARD_PORT``.
ENV_PREFIX = "AILM_DASHBOARD_"


def is_loopback_host(host: str) -> bool:
    """Whether ``host`` is an explicit loopback name or address.

    No DNS lookup is performed.  Resolving an arbitrary hostname here would
    turn a local-only invariant into a statement about mutable DNS state and
    would admit rebinding between validation and ``bind(2)``.
    """
    candidate = host.strip().strip("[]").lower()
    if candidate == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def require_loopback_host(host: str) -> str:
    """Return a validated local bind host or raise ``ValueError``."""
    if not is_loopback_host(host):
        raise ValueError(
            f"dashboard host {host!r} is not loopback; this unauthenticated static "
            "application may bind only localhost, 127.0.0.0/8, or ::1"
        )
    return host


@dataclass(frozen=True)
class DashboardConfig:
    """Server, engine and interface limits for one dashboard process.

    Attributes:
        host: Interface to bind. Only explicit loopback names and addresses are
            accepted; the local browser session is authenticated with a
            per-process token and same-origin requests.
        port: TCP port to listen on.
        max_request_bytes: Largest accepted POST body. The inference payload is
            one sentence; reading a client-declared length without a ceiling
            lets one request allocate arbitrary memory.
        max_continuation_words: Ceiling on the top-down rollout. The rollout
            costs one lexicon scan per emitted character and runs inside the
            request handler, so an unclamped count off the wire is a stall.
        default_reply_words: Reply length the interface starts at.
        max_history_turns: Conversation turns retained. The transcript is
            replayed on every request, so an unbounded history is an unbounded
            per-request cost.
        max_spoken_chars: Longest utterance handed to speech synthesis. A long
            rollout is the utterance most likely to outlive the turn that
            started it.
        speech_watchdog_ms: How long after starting an utterance the interface
            force-clears the synthesis queue. Chrome can leave ``speaking`` true
            after a cancelled utterance, so this does not trust that flag.
        default_narrative_alpha: Starting Level 6 to Level 5 channel weight.
        alpha_grid: Weights the channel sweep reports. ``0.0`` must be present:
            it is the identity anchor every gain is measured against.
        max_concurrent_requests: Hard ceiling on request-handler threads.
        max_requests_per_minute: Authenticated API request ceiling per local
            client address. This limits accidental retry storms in addition to
            bounding simultaneous handlers.
        request_timeout_seconds: Socket read/write timeout for one client.
        api_token: Per-process bearer/cookie credential. Generated randomly by
            default and omitted from the public configuration response.
    """

    host: str = "127.0.0.1"
    port: int = 8080
    max_request_bytes: int = 64 * 1024
    max_continuation_words: int = 200
    default_reply_words: int = 8
    max_history_turns: int = 50
    max_spoken_chars: int = 400
    speech_watchdog_ms: int = 30_000
    default_narrative_alpha: float = 0.5
    alpha_grid: tuple[float, ...] = (0.0, 0.05, 0.1, 0.25, 0.5)
    max_concurrent_requests: int = 4
    max_requests_per_minute: int = 300
    request_timeout_seconds: float = 30.0
    api_token: str = field(default_factory=lambda: secrets.token_urlsafe(32), repr=False)

    def __post_init__(self) -> None:
        """Reject a configuration that cannot describe a working dashboard.

        Raises:
            ValueError: On any bound that is not usable. A limit of zero or a
                port outside the legal range is a typo, and defaulting it would
                hide the typo behind behavior that looks deliberate.
        """
        require_loopback_host(self.host)
        if not 1 <= self.port <= 65535:
            raise ValueError(f"port {self.port} outside 1-65535")
        for name in (
            "max_request_bytes",
            "max_continuation_words",
            "default_reply_words",
            "max_history_turns",
            "max_spoken_chars",
            "speech_watchdog_ms",
            "max_concurrent_requests",
            "max_requests_per_minute",
            "request_timeout_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.default_reply_words > self.max_continuation_words:
            raise ValueError(
                f"default_reply_words {self.default_reply_words} exceeds "
                f"max_continuation_words {self.max_continuation_words}; the interface "
                f"would open on a value the engine immediately clamps"
            )
        if not math.isfinite(self.default_narrative_alpha):
            raise ValueError(
                f"default_narrative_alpha must be finite, got {self.default_narrative_alpha}"
            )
        if not 0.0 <= self.default_narrative_alpha <= 1.0:
            raise ValueError(
                f"default_narrative_alpha {self.default_narrative_alpha} outside [0, 1]"
            )
        if any(not math.isfinite(value) for value in self.alpha_grid):
            raise ValueError(f"alpha_grid {self.alpha_grid} must contain only finite weights")
        if any(not 0.0 <= value <= 1.0 for value in self.alpha_grid):
            raise ValueError(f"alpha_grid {self.alpha_grid} contains a weight outside [0, 1]")
        if 0.0 not in self.alpha_grid:
            raise ValueError(
                f"alpha_grid {self.alpha_grid} lacks the 0.0 identity anchor; every "
                f"reported gain is measured against it"
            )
        if sorted(self.alpha_grid) != list(self.alpha_grid):
            raise ValueError(f"alpha_grid {self.alpha_grid} is not ascending")
        if len(self.api_token.encode("utf-8")) < 32:
            raise ValueError("api_token must contain at least 32 UTF-8 bytes")

    def to_public_dict(self) -> dict[str, Any]:
        """The subset the browser needs, as JSON-ready values.

        ``host`` and ``port`` are omitted: the page already knows where it was
        served from, and echoing a bind address into a response body tells a
        reader something about the machine they did not ask for.
        """
        payload = asdict(self)
        payload.pop("host", None)
        payload.pop("port", None)
        payload.pop("api_token", None)
        payload["alpha_grid"] = list(self.alpha_grid)
        return payload


def _parse(name: str, raw: str, current: Any) -> Any:
    """Coerce one environment string to the field's type, or refuse.

    Raises:
        ValueError: If ``raw`` does not parse as the field's type. Refusing is
            the point: a mistyped limit that silently keeps its default is a
            setting the operator believes they changed.
    """
    if isinstance(current, bool):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{ENV_PREFIX}{name.upper()}={raw!r} is not a boolean")
    if isinstance(current, tuple):
        try:
            return tuple(float(part) for part in raw.split(",") if part.strip())
        except ValueError as exc:
            raise ValueError(
                f"{ENV_PREFIX}{name.upper()}={raw!r} is not a comma-separated float list"
            ) from exc
    for kind in (int, float, str):
        if isinstance(current, kind):
            try:
                return kind(raw)
            except ValueError as exc:
                raise ValueError(
                    f"{ENV_PREFIX}{name.upper()}={raw!r} is not a {kind.__name__}"
                ) from exc
    raise ValueError(f"no parser for field {name}")


def config_from_env(
    base: DashboardConfig | None = None, environ: dict[str, str] | None = None
) -> DashboardConfig:
    """Apply ``AILM_DASHBOARD_*`` overrides on top of ``base``.

    Args:
        base: Configuration to start from. Defaults to the shipped defaults.
        environ: Environment to read. Defaults to the process environment.

    Returns:
        A new frozen configuration.

    Raises:
        ValueError: If any override fails to parse, or the result fails
            validation.
    """
    source = os.environ if environ is None else environ
    start = base or DashboardConfig()
    overrides: dict[str, Any] = {}
    for config_field in fields(start):
        raw = source.get(f"{ENV_PREFIX}{config_field.name.upper()}")
        if raw is not None and raw.strip():
            overrides[config_field.name] = _parse(
                config_field.name, raw, getattr(start, config_field.name)
            )
    return DashboardConfig(**{**asdict(start), **overrides})


def load_config() -> DashboardConfig:
    """The configuration this process should run with."""
    return config_from_env()
