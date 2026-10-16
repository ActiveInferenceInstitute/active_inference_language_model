"""Fail-closed JSON transport for services that must remain on loopback.

The standard urllib opener inherits ``HTTP_PROXY``/``HTTPS_PROXY`` and follows
redirects. Both behaviors are wrong for AILM's Ollama boundary: a caller that
names ``localhost`` intends prompts, corpus snippets, model manifests, and
replies to stay on the machine. This module owns that boundary once so every
benchmark, cache identity lookup, and preflight has the same restrictions.
"""

from __future__ import annotations

import ipaddress
import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

__all__ = ["LoopbackTransportError", "loopback_json_request", "validate_loopback_endpoint"]

DEFAULT_MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class LoopbackTransportError(RuntimeError):
    """A loopback response violated the local JSON transport contract."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Turn every redirect into an HTTP error instead of following it."""

    def redirect_request(
        self,
        _request: urllib.request.Request,
        _file_pointer: Any,
        _code: int,
        _message: str,
        _headers: Any,
        _new_url: str,
    ) -> None:
        return None


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


@dataclass(frozen=True, slots=True)
class _ValidatedLoopbackTarget:
    """Logical origin plus the already-resolved literal connection origin."""

    origin: str
    connection_origin: str
    host_header: str


def _loopback_addresses(host: str, port: int) -> tuple[str, ...]:
    """Resolve one host and require every returned address to be loopback."""
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        if host.rstrip(".").lower() != "localhost":
            raise ValueError(
                f"loopback endpoint host must be localhost or a loopback IP literal, got {host!r}"
            ) from None
        try:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ValueError(f"loopback endpoint host {host!r} did not resolve: {exc}") from exc
        addresses = tuple(sorted({str(info[4][0]) for info in infos}))
        if not addresses:
            raise ValueError(f"loopback endpoint host {host!r} resolved to no addresses") from None
        if not all(ipaddress.ip_address(address).is_loopback for address in addresses):
            raise ValueError(
                f"loopback endpoint host {host!r} resolved outside loopback: {addresses!r}"
            ) from None
        return addresses
    if not literal.is_loopback:
        raise ValueError(f"endpoint address {host!r} is not loopback")
    return (str(literal),)


def _validated_loopback_target(endpoint: str) -> _ValidatedLoopbackTarget:
    """Resolve one logical origin once and pin the connection to a literal.

    Paths, query strings, fragments, and user information are refused so callers
    cannot smuggle a second request target into an origin parameter. API paths
    are supplied separately to :func:`loopback_json_request`.
    """
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("loopback endpoint must be a non-empty string")
    parsed = urlsplit(endpoint.strip())
    if parsed.scheme.lower() != "http":
        raise ValueError(f"loopback endpoint must use http, got {parsed.scheme!r}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("loopback endpoint must not contain user information")
    if not parsed.hostname:
        raise ValueError("loopback endpoint must contain a host")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("loopback endpoint must be an origin without path, query, or fragment")
    try:
        port = parsed.port if parsed.port is not None else 80
    except ValueError as exc:
        raise ValueError(f"loopback endpoint has an invalid port: {exc}") from exc
    addresses = _loopback_addresses(parsed.hostname, port)
    host = parsed.hostname.lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{port}"
    address = addresses[0]
    connection_host = f"[{address}]" if ":" in address else address
    return _ValidatedLoopbackTarget(
        origin=urlunsplit(("http", netloc, "", "", "")),
        connection_origin=urlunsplit(("http", f"{connection_host}:{port}", "", "", "")),
        host_header=netloc,
    )


def validate_loopback_endpoint(endpoint: str) -> str:
    """Return the canonical logical HTTP origin after one loopback resolution."""

    return _validated_loopback_target(endpoint).origin


def loopback_json_request(
    endpoint: str,
    path: str,
    *,
    payload: Any | None = None,
    timeout: float,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
) -> Any:
    """Make one proxy-free, redirect-free JSON request to a loopback service."""
    target = _validated_loopback_target(endpoint)
    if not isinstance(path, str) or not path.startswith("/") or path.startswith("//"):
        raise ValueError(f"API path must be one absolute path on the origin, got {path!r}")
    if "?" in path or "#" in path:
        raise ValueError("API path must not contain a query or fragment")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout!r}")
    if (
        not isinstance(max_response_bytes, int)
        or isinstance(max_response_bytes, bool)
        or max_response_bytes <= 0
    ):
        raise ValueError(
            f"max_response_bytes must be a positive integer, got {max_response_bytes!r}"
        )

    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {"Accept": "application/json", "Host": target.host_header}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{target.connection_origin}{path}",
        data=data,
        headers=headers,
        method="POST" if data is not None else "GET",
    )
    try:
        with _OPENER.open(request, timeout=float(timeout)) as response:
            content_type = response.headers.get_content_type().lower()
            if content_type != "application/json":
                raise LoopbackTransportError(
                    f"loopback service returned {content_type!r}, expected 'application/json'"
                )
            length_header = response.headers.get("Content-Length")
            if length_header is not None:
                try:
                    declared_length = int(length_header)
                except ValueError as exc:
                    raise LoopbackTransportError(
                        f"loopback service returned invalid Content-Length {length_header!r}"
                    ) from exc
                if declared_length < 0 or declared_length > max_response_bytes:
                    raise LoopbackTransportError(
                        f"loopback response declares {declared_length} bytes; limit is "
                        f"{max_response_bytes}"
                    )
            body = response.read(max_response_bytes + 1)
    except urllib.error.HTTPError as exc:
        raise LoopbackTransportError(
            f"loopback service refused the request with HTTP status {exc.code}"
        ) from exc
    if len(body) > max_response_bytes:
        raise LoopbackTransportError(
            f"loopback response exceeded the {max_response_bytes}-byte limit"
        )
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LoopbackTransportError("loopback service returned invalid UTF-8 JSON") from exc
