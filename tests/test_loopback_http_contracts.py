"""Refusals of the proxy-free loopback JSON client, at the boundary.

Every endpoint and path the client accepts is a place a request could be
steered off the local machine. The tests below hit each refusal with the
narrowest malformed input and, where a response is involved, a real loopback
server sending the exact malformed header or oversized body.
"""

from __future__ import annotations

import pytest

from ailm.loopback_http import (
    LoopbackTransportError,
    _validated_loopback_target,
    loopback_json_request,
)

_ORIGIN = "http://127.0.0.1:9"


@pytest.mark.parametrize("endpoint", ["", "   ", None, 7])
def test_endpoint_must_be_a_non_empty_string(endpoint) -> None:
    with pytest.raises(ValueError, match="must be a non-empty string"):
        _validated_loopback_target(endpoint)


def test_endpoint_must_contain_a_host() -> None:
    with pytest.raises(ValueError, match="must contain a host"):
        _validated_loopback_target("http://")


def test_endpoint_must_carry_a_numeric_port() -> None:
    with pytest.raises(ValueError, match="invalid port"):
        _validated_loopback_target("http://127.0.0.1:abc")


def test_endpoint_literal_must_be_loopback() -> None:
    with pytest.raises(ValueError, match="is not loopback"):
        _validated_loopback_target("http://8.8.8.8:80")


def test_a_trailing_dot_localhost_is_either_resolved_to_loopback_or_refused() -> None:
    """The one name the allowlist admits besides a literal is a localhost spelling.

    Whether the resolver answers for the fully qualified spelling is a property
    of the host's DNS configuration, not of this module: on some systems it
    resolves to loopback and is accepted, on others it fails and must be refused
    with the resolver's error, never with a bare exception. Both outcomes are
    asserted so the test is honest on either kind of machine.
    """
    try:
        target = _validated_loopback_target("http://localhost.:80")
    except ValueError as refusal:
        assert "did not resolve" in str(refusal)
    else:
        assert target.host_header == "localhost.:80"
        assert target.connection_origin.startswith("http://")


def test_an_ipv6_loopback_literal_is_bracketed_for_the_host_header() -> None:
    target = _validated_loopback_target("http://[::1]:8080")
    assert target.host_header == "[::1]:8080"
    assert target.connection_origin == "http://[::1]:8080"


@pytest.mark.parametrize("path", ["api/tags", "//api/tags", 7])
def test_api_path_must_be_one_absolute_path(path) -> None:
    with pytest.raises(ValueError, match="one absolute path on the origin"):
        loopback_json_request(_ORIGIN, path, timeout=1)


@pytest.mark.parametrize("path", ["/api/tags?x=1", "/api/tags#top"])
def test_api_path_must_not_carry_a_query_or_fragment(path) -> None:
    with pytest.raises(ValueError, match="must not contain a query or fragment"):
        loopback_json_request(_ORIGIN, path, timeout=1)


@pytest.mark.parametrize("timeout", [0, -1, True, "1"])
def test_timeout_must_be_positive(timeout) -> None:
    with pytest.raises(ValueError, match="timeout must be positive"):
        loopback_json_request(_ORIGIN, "/api/tags", timeout=timeout)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_response_limit_must_be_a_positive_integer(limit) -> None:
    with pytest.raises(ValueError, match="max_response_bytes must be a positive integer"):
        loopback_json_request(_ORIGIN, "/api/tags", timeout=1, max_response_bytes=limit)


def test_an_unparseable_content_length_fails_closed(local_http_server) -> None:
    local_http_server.enqueue_bytes(
        "/bad-length", b"{}", content_type="application/json", declared_length="abc"
    )
    with pytest.raises(LoopbackTransportError, match="invalid Content-Length"):
        loopback_json_request(local_http_server.base_url, "/bad-length", timeout=1)


def test_an_oversized_body_without_a_content_length_is_still_bounded(local_http_server) -> None:
    """The byte cap must hold when the server declares no length at all."""
    body = b'{"payload": "' + b"x" * 128 + b'"}'
    local_http_server.enqueue_bytes(
        "/unbounded", body, content_type="application/json", omit_content_length=True
    )
    with pytest.raises(LoopbackTransportError, match="exceeded the 32-byte limit"):
        loopback_json_request(
            local_http_server.base_url, "/unbounded", timeout=1, max_response_bytes=32
        )
