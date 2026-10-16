"""Dashboard boundary contracts that the subprocess launch tests do not reach.

The configuration parser's refusals, the loopback-host predicate, the
preflight refusal for non-API paths, and the rate limiter's rolling window
are each exercised in-process on real objects: a handler with no socket, and
a real bound server whose request log is aged by hand.
"""

from __future__ import annotations

from email.message import Message
from http import HTTPStatus
from time import monotonic

import pytest

from ailm.dashboard import server as dashboard_server
from ailm.dashboard.config import (
    DashboardConfig,
    _parse,
    is_loopback_host,
    require_loopback_host,
)

# -- configuration parsing ---------------------------------------------------


@pytest.mark.parametrize("host", ["localhost", " LOCALHOST ", "127.0.0.1", "[::1]"])
def test_loopback_names_and_literals_are_accepted(host) -> None:
    assert is_loopback_host(host) is True
    assert require_loopback_host(host) == host


@pytest.mark.parametrize("host", ["example.org", "10.0.0.1", "0.0.0.0", ""])
def test_non_loopback_hosts_are_refused_without_a_lookup(host) -> None:
    assert is_loopback_host(host) is False
    with pytest.raises(ValueError, match="is not loopback"):
        require_loopback_host(host)


@pytest.mark.parametrize(("raw", "expected"), [("yes", True), (" On ", True), ("0", False)])
def test_boolean_fields_accept_the_usual_spellings(raw, expected) -> None:
    assert _parse("flag", raw, True) is expected


def test_boolean_fields_refuse_anything_else() -> None:
    with pytest.raises(ValueError, match="AILM_DASHBOARD_FLAG='maybe' is not a boolean"):
        _parse("flag", "maybe", False)


def test_float_list_fields_refuse_a_non_numeric_entry() -> None:
    with pytest.raises(ValueError, match="is not a comma-separated float list"):
        _parse("alpha_grid", "0.1,x", (0.0,))


def test_a_field_of_unknown_type_has_no_parser() -> None:
    with pytest.raises(ValueError, match="no parser for field mystery"):
        _parse("mystery", "1", None)


# -- the request handler, without a socket -----------------------------------


class _Probe(dashboard_server.DashboardHandler):
    """A handler with no socket that records the error it would have sent."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.headers = Message()
        self.request_version = "HTTP/1.1"
        self.errors: list[tuple[HTTPStatus, str]] = []
        self.config = DashboardConfig()

    def send_error(self, code, message=None, explain=None):
        self.errors.append((code, message))


def test_preflight_outside_the_api_is_not_found() -> None:
    probe = _Probe("/index.html")
    probe.do_OPTIONS()
    assert probe.errors == [(HTTPStatus.NOT_FOUND, "Endpoint not found")]


def test_presented_token_prefers_the_bearer_header_over_the_cookie() -> None:
    probe = _Probe("/api/config")
    probe.headers["Authorization"] = "Bearer header-token"
    probe.headers["Cookie"] = f"{dashboard_server.AUTH_COOKIE}=cookie-token"
    assert probe._presented_token() == "header-token"
    del probe.headers["Authorization"]
    assert probe._presented_token() == "cookie-token"


# -- the rate limiter on a real bound server ---------------------------------


@pytest.fixture
def bound_server():
    server = dashboard_server.ReusableThreadingServer(
        ("127.0.0.1", 0),
        dashboard_server.DashboardHandler,
        max_workers=2,
        max_requests_per_minute=2,
        request_timeout=1.0,
    )
    try:
        yield server
    finally:
        server.server_close()


def test_rate_limiter_forgets_requests_older_than_its_window(bound_server) -> None:
    client = "127.0.0.1"
    stale = monotonic() - 61.0
    bound_server._request_times[client].extend([stale, stale])
    assert bound_server.consume_rate_slot(client) is True
    assert bound_server.consume_rate_slot(client) is True
    assert bound_server.consume_rate_slot(client) is False
    assert all(stamp > stale for stamp in bound_server._request_times[client])
