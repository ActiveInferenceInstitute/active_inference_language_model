"""pytest fixtures for AILM Stage 0 tests."""

from __future__ import annotations

import json
import socket
import threading
from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from ailm.claims import ClaimsLedger
from ailm.gates import AUDITED_GATE_IDS, GateVerdict


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "publication_context: requires retained private publication history, raw inputs, or original-source proofs; mandatory in full",
    )
    config.addinivalue_line(
        "markers",
        "template_context: requires the configured renderer checkout or its staged manuscript auxiliaries; mandatory in full",
    )


def read_gate_audit_reports(root: Path) -> dict[str, dict[str, Any]]:
    """Read the complete committed audit inputs; absent evidence is a failure.

    Keep this reader independent of the audit writer. Report consumers must
    neither skip a missing report nor pass by iterating an empty or partial one.
    """
    reports = (
        ("gate_audit.json", {str(gate) for gate in AUDITED_GATE_IDS}),
        ("gate_audit_ablation.json", {"stage5_clause_case_only"}),
    )
    rows: dict[str, dict[str, Any]] = {}
    for name, expected in reports:
        path = root / "output" / "reports" / name
        assert path.is_file(), (
            f"Required gate audit report missing: {path}. "
            "Run .venv/bin/python scripts/audit_gates.py --full --json."
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, list), f"{name}: expected a list of gate rows"
        found: set[str] = set()
        for row in payload:
            assert isinstance(row, dict), f"{name}: expected an object per gate row"
            gate_id = row.get("gate_id")
            assert isinstance(gate_id, str), f"{name}: missing or invalid gate_id"
            assert gate_id not in found, f"{name}: duplicate gate_id {gate_id}"
            assert row.get("outcome") in ("PASS", "FAIL"), f"{name}: invalid outcome for {gate_id}"
            assert row.get("verdict") in tuple(GateVerdict), (
                f"{name}: invalid verdict for {gate_id}"
            )
            found.add(gate_id)
            rows[gate_id] = row
        assert found == expected, (
            f"{name}: gate inventory differs; missing={sorted(expected - found)}, "
            f"unexpected={sorted(found - expected)}"
        )
    return rows


@pytest.fixture(scope="session")
def gate_audit_report_reader():
    """Expose the real reader for damaged-report controls under tmp_path."""
    return read_gate_audit_reports


@pytest.fixture(scope="session")
def gate_audit_rows() -> dict[str, dict[str, Any]]:
    """Shared validated audit rows for cover, hierarchy, and finding checks."""
    return read_gate_audit_reports(Path(__file__).resolve().parents[1])


@dataclass(frozen=True)
class RecordedHTTPRequest:
    """One request observed by a real loopback HTTP server."""

    method: str
    path: str
    body: bytes
    headers: dict[str, str]


@dataclass(frozen=True)
class _QueuedHTTPResponse:
    status: int
    body: bytes
    content_type: str
    disconnect: bool = False
    headers: tuple[tuple[str, str], ...] = ()
    declared_length: int | str | None = None
    truncate_after: int | None = None
    omit_content_length: bool = False


class LocalHTTPServer:
    """Threaded loopback server with queued real responses and request evidence."""

    def __init__(self) -> None:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                owner._serve(self)

            def do_POST(self) -> None:
                owner._serve(self)

            def log_message(self, _format: str, *args: object) -> None:
                pass

        self._responses: dict[str, deque[_QueuedHTTPResponse]] = defaultdict(deque)
        self._requests: list[RecordedHTTPRequest] = []
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._closed = False
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def requests(self) -> tuple[RecordedHTTPRequest, ...]:
        with self._lock:
            return tuple(self._requests)

    def requests_for(self, path: str) -> tuple[RecordedHTTPRequest, ...]:
        return tuple(request for request in self.requests if request.path == path)

    def enqueue_json(self, path: str, payload: Any, *, status: int = 200, repeat: int = 1) -> None:
        body = json.dumps(payload).encode("utf-8")
        for _ in range(repeat):
            self._responses[path].append(_QueuedHTTPResponse(status, body, "application/json"))

    def enqueue_bytes(
        self,
        path: str,
        body: bytes,
        *,
        status: int = 200,
        content_type: str = "application/octet-stream",
        repeat: int = 1,
        headers: dict[str, str] | None = None,
        declared_length: int | str | None = None,
        truncate_after: int | None = None,
        omit_content_length: bool = False,
    ) -> None:
        for _ in range(repeat):
            self._responses[path].append(
                _QueuedHTTPResponse(
                    status,
                    body,
                    content_type,
                    headers=tuple((headers or {}).items()),
                    declared_length=declared_length,
                    truncate_after=truncate_after,
                    omit_content_length=omit_content_length,
                )
            )

    def enqueue_disconnect(self, path: str) -> None:
        self._responses[path].append(_QueuedHTTPResponse(200, b"", "text/plain", True))

    def _serve(self, handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length", "0"))
        body = handler.rfile.read(length)
        with self._lock:
            self._requests.append(
                RecordedHTTPRequest(
                    method=handler.command,
                    path=handler.path,
                    body=body,
                    headers={key.lower(): value for key, value in handler.headers.items()},
                )
            )
            response = (
                self._responses[handler.path].popleft()
                if self._responses[handler.path]
                else _QueuedHTTPResponse(
                    500,
                    json.dumps({"error": f"no response queued for {handler.path}"}).encode(),
                    "application/json",
                )
            )
        if response.disconnect:
            handler.close_connection = True
            try:
                handler.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            handler.connection.close()
            return
        handler.send_response(response.status)
        handler.send_header("Content-Type", response.content_type)
        if not response.omit_content_length:
            declared_length = (
                len(response.body) if response.declared_length is None else response.declared_length
            )
            handler.send_header("Content-Length", str(declared_length))
        for name, value in response.headers:
            handler.send_header(name, value)
        handler.end_headers()
        wire_body = (
            response.body
            if response.truncate_after is None
            else response.body[: response.truncate_after]
        )
        try:
            handler.wfile.write(wire_body)
            handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        if response.truncate_after is not None:
            handler.close_connection = True
            try:
                handler.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            handler.connection.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def local_http_server() -> Iterator[LocalHTTPServer]:
    server = LocalHTTPServer()
    try:
        yield server
    finally:
        server.close()


@pytest.fixture
def tiny_cmudict() -> dict[str, list[list[str]]]:
    """Minimal hard-coded CMUdict for offline tests.

    5 words, covering stops, fricatives, vowels. Stress digits included
    as they appear in the real file.
    """
    return {
        "THE": [["DH", "AH0"]],
        "CAT": [["K", "AE1", "T"]],
        "SAT": [["S", "AE1", "T"]],
        "ON": [["AO1", "N"]],
        "MAT": [["M", "AE1", "T"]],
    }


@pytest.fixture
def claims_ledger_tmp(tmp_path) -> ClaimsLedger:
    """A fresh ClaimsLedger backed by a temporary directory."""
    return ClaimsLedger(tmp_path / "data" / "claims.json")


@pytest.fixture
def two_factor_agent():
    """Real pymdp Agent with 2 factors and A_dependencies set."""
    from ailm.generative.smoke import make_two_factor_agent

    return make_two_factor_agent()
