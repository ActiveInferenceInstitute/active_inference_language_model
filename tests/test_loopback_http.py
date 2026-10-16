"""Real-network controls for the loopback-only model-service boundary."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

import pytest

from ailm.benchmarks.ollama import ollama_available
from ailm.loopback_http import (
    LoopbackTransportError,
    _validated_loopback_target,
    loopback_json_request,
    validate_loopback_endpoint,
)


def test_inherited_proxy_cannot_intercept_a_local_ollama_request(
    local_http_server, monkeypatch
) -> None:
    """NEGATIVE CONTROL reproducing the former prompt-exfiltration path."""
    target = "http://localhost:65530"
    absolute_proxy_path = f"{target}/api/tags"
    local_http_server.enqueue_json(
        absolute_proxy_path,
        {"models": [{"name": "victim:latest"}]},
    )
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.setenv(name, local_http_server.base_url)

    assert ollama_available("victim:latest", endpoint=target) is False
    assert local_http_server.requests == (), (
        "the proxy received a nominally local Ollama request; prompts and replies can leave loopback"
    )


@pytest.mark.parametrize(
    "endpoint",
    (
        "https://127.0.0.1:11434",
        "http://example.com:11434",
        "http://127.0.0.1:11434/api",
        "http://user@127.0.0.1:11434",
    ),
)
def test_non_loopback_or_non_origin_endpoints_are_refused(endpoint: str) -> None:
    with pytest.raises(ValueError):
        validate_loopback_endpoint(endpoint)


def test_loopback_ipv4_and_localhost_origins_are_accepted(local_http_server) -> None:
    assert validate_loopback_endpoint(local_http_server.base_url) == local_http_server.base_url
    port = local_http_server.base_url.rsplit(":", 1)[1]
    assert validate_loopback_endpoint(f"http://localhost:{port}") == f"http://localhost:{port}"


def test_localhost_connection_is_pinned_to_the_validated_literal(local_http_server) -> None:
    """NEGATIVE CONTROL: the request URL contains no hostname to re-resolve."""
    port = int(local_http_server.base_url.rsplit(":", 1)[1])
    target = _validated_loopback_target(f"http://localhost:{port}")
    connection_host = urlsplit(target.connection_origin).hostname
    assert connection_host is not None
    assert ipaddress.ip_address(connection_host).is_loopback
    assert connection_host != "localhost"
    assert target.host_header == f"localhost:{port}"

    local_http_server.enqueue_json("/api/generate", {"response": "local"})

    result = loopback_json_request(
        f"http://localhost:{port}",
        "/api/generate",
        payload={"prompt": "must stay local"},
        timeout=1,
    )

    assert result == {"response": "local"}
    request = local_http_server.requests_for("/api/generate")[0]
    assert request.headers["host"] == f"localhost:{port}"
    assert b"must stay local" in request.body


def test_redirects_are_refused_even_when_the_destination_is_loopback(local_http_server) -> None:
    local_http_server.enqueue_bytes(
        "/redirect",
        b"{}",
        status=302,
        content_type="application/json",
        headers={"Location": local_http_server.base_url + "/api/tags"},
    )
    local_http_server.enqueue_json("/api/tags", {"models": []})
    with pytest.raises(LoopbackTransportError, match="HTTP status 302"):
        loopback_json_request(local_http_server.base_url, "/redirect", timeout=1)
    assert not local_http_server.requests_for("/api/tags")


def test_response_type_and_size_are_bounded(local_http_server) -> None:
    local_http_server.enqueue_bytes("/wrong-type", b"{}", content_type="text/plain")
    with pytest.raises(LoopbackTransportError, match="expected 'application/json'"):
        loopback_json_request(local_http_server.base_url, "/wrong-type", timeout=1)

    local_http_server.enqueue_json("/too-large", {"payload": "x" * 128})
    with pytest.raises(LoopbackTransportError, match="limit"):
        loopback_json_request(
            local_http_server.base_url,
            "/too-large",
            timeout=1,
            max_response_bytes=32,
        )


def test_invalid_json_fails_closed(local_http_server) -> None:
    local_http_server.enqueue_bytes("/invalid-json", b"{not-json", content_type="application/json")
    with pytest.raises(LoopbackTransportError, match="invalid UTF-8 JSON"):
        loopback_json_request(local_http_server.base_url, "/invalid-json", timeout=1)
