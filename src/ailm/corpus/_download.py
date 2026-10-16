"""Bounded response authentication and durable raw-corpus publication.

The corpus-specific modules own network access.  This module deliberately does
not issue requests: it enforces the shared response boundary after a CMUdict or
UD loader has opened a streaming response, and it publishes already-verified
bytes through a sibling temporary file.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

import requests
from urllib3.exceptions import HTTPError as Urllib3HTTPError

_STREAM_CHUNK_BYTES = 64 * 1024


def _bounded_response_chunks(response: requests.Response, *, max_bytes: int) -> Iterator[bytes]:
    """Read no more than ``max_bytes`` from one identity-encoded response."""
    remaining = max_bytes
    while remaining:
        try:
            chunk = response.raw.read(
                min(_STREAM_CHUNK_BYTES, remaining),
                decode_content=False,
            )
        except Urllib3HTTPError as exc:
            raise requests.ConnectionError(f"raw corpus response was interrupted: {exc}") from exc
        if not chunk:
            return
        remaining -= len(chunk)
        yield chunk


def _authenticate_chunks(
    chunks: Iterable[bytes],
    *,
    expected_size_bytes: int,
    expected_sha256: str,
    source_name: str,
    integrity_error: type[ValueError],
) -> bytes:
    """Retain at most the expected body plus one rejecting control byte."""
    body = bytearray()
    digest = hashlib.sha256()
    for chunk in chunks:
        retained = chunk[: expected_size_bytes + 1 - len(body)]
        body.extend(retained)
        digest.update(retained)
        if len(body) > expected_size_bytes:
            raise integrity_error(
                f"{source_name} download exceeded the {expected_size_bytes}-byte limit"
            )

    measured_size = len(body)
    measured_sha256 = digest.hexdigest()
    if measured_size != expected_size_bytes or measured_sha256 != expected_sha256:
        raise integrity_error(
            f"{source_name} source does not match the declared revision: "
            f"bytes={measured_size} (expected {expected_size_bytes}), "
            f"sha256={measured_sha256} (expected {expected_sha256})"
        )
    return bytes(body)


def read_authenticated_response(
    response: requests.Response,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
    source_name: str,
    integrity_error: type[ValueError] = ValueError,
) -> bytes:
    """Read, bound, hash, and authenticate one non-redirecting response.

    The caller must open ``response`` with ``stream=True`` and
    ``allow_redirects=False``.  A single exact ``Content-Length`` is required,
    content and transfer encodings are refused, and at most
    ``expected_size_bytes + 1`` decoded bytes are retained.  The extra byte is
    solely the live negative control for an overlong response.
    """
    if 300 <= response.status_code < 400:
        location = response.headers.get("Location", "<missing>")
        raise integrity_error(
            f"{source_name} download refused HTTP redirect {response.status_code} to {location!r}"
        )
    response.raise_for_status()

    content_encoding = response.headers.get("Content-Encoding")
    if content_encoding is not None and content_encoding.strip().lower() != "identity":
        raise integrity_error(
            f"{source_name} download refused Content-Encoding {content_encoding!r}"
        )
    transfer_encoding = response.headers.get("Transfer-Encoding")
    if transfer_encoding is not None:
        raise integrity_error(
            f"{source_name} download refused Transfer-Encoding {transfer_encoding!r}"
        )

    length_values = tuple(str(value) for value in response.raw.headers.getlist("Content-Length"))
    if len(length_values) != 1:
        raise integrity_error(
            f"{source_name} download requires exactly one Content-Length; "
            f"received {length_values!r}"
        )
    length_text = length_values[0].strip()
    if not length_text.isascii() or not length_text.isdecimal():
        raise integrity_error(
            f"{source_name} download returned invalid Content-Length {length_values[0]!r}"
        )
    declared_size = int(length_text)
    if declared_size != expected_size_bytes:
        raise integrity_error(
            f"{source_name} download declared {declared_size} bytes, expected {expected_size_bytes}"
        )

    return _authenticate_chunks(
        _bounded_response_chunks(response, max_bytes=expected_size_bytes + 1),
        expected_size_bytes=expected_size_bytes,
        expected_sha256=expected_sha256,
        source_name=source_name,
        integrity_error=integrity_error,
    )


def atomic_publish_bytes(path: Path, content: bytes) -> None:
    """Fsync a unique sibling temporary, replace, then fsync the directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, scratch_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    scratch = Path(scratch_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(scratch, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if scratch.exists():
            scratch.unlink()
