"""Live controls for bounded raw-corpus response authentication."""

from __future__ import annotations

import hashlib

import pytest
import requests

from ailm.corpus._download import (
    _authenticate_chunks,
    _bounded_response_chunks,
    atomic_publish_bytes,
)
from ailm.corpus.cmudict import load_cmudict

PAYLOAD = b"word  W ER1 D\n"


def test_chunk_authenticator_positive_and_negative_controls() -> None:
    """Exact chunks pass; truncation, digest drift, and excess bytes fail."""
    digest = hashlib.sha256(PAYLOAD).hexdigest()
    assert (
        _authenticate_chunks(
            [PAYLOAD[:4], PAYLOAD[4:]],
            expected_size_bytes=len(PAYLOAD),
            expected_sha256=digest,
            source_name="fixture",
            integrity_error=ValueError,
        )
        == PAYLOAD
    )

    with pytest.raises(ValueError, match="bytes=.*expected"):
        _authenticate_chunks(
            [PAYLOAD[:-1]],
            expected_size_bytes=len(PAYLOAD),
            expected_sha256=digest,
            source_name="fixture",
            integrity_error=ValueError,
        )
    with pytest.raises(ValueError, match="sha256=.*expected"):
        _authenticate_chunks(
            [PAYLOAD],
            expected_size_bytes=len(PAYLOAD),
            expected_sha256="0" * 64,
            source_name="fixture",
            integrity_error=ValueError,
        )

    def oversized_chunks():
        yield PAYLOAD + b"x" * 1_000_000
        raise AssertionError("the reader continued after its rejecting control byte")

    with pytest.raises(ValueError, match="exceeded"):
        _authenticate_chunks(
            oversized_chunks(),
            expected_size_bytes=len(PAYLOAD),
            expected_sha256=digest,
            source_name="fixture",
            integrity_error=ValueError,
        )


def test_real_http_reader_stops_at_its_exact_byte_limit(local_http_server) -> None:
    """The low-level stream cannot advance beyond the caller's hard bound."""
    route = "/bounded-body"
    body = PAYLOAD + b"rejecting-control-byte"
    local_http_server.enqueue_bytes(route, body)

    with requests.get(
        local_http_server.base_url + route,
        stream=True,
        allow_redirects=False,
        headers={"Accept-Encoding": "identity"},
        timeout=(1, 1),
    ) as response:
        chunks = tuple(_bounded_response_chunks(response, max_bytes=len(PAYLOAD) + 1))

    assert b"".join(chunks) == body[: len(PAYLOAD) + 1]


@pytest.mark.parametrize(
    ("response_options", "message"),
    [
        ({"omit_content_length": True}, "exactly one Content-Length"),
        (
            {"headers": {"Content-Length": str(len(PAYLOAD))}},
            "Content-Length|multiple",
        ),
        ({"declared_length": "not-a-number"}, "Content-Length"),
        ({"headers": {"Content-Encoding": "gzip"}}, "Content-Encoding"),
        ({"headers": {"Transfer-Encoding": "chunked"}}, "Transfer-Encoding"),
    ],
)
def test_real_http_header_ambiguities_fail_closed(
    tmp_path,
    local_http_server,
    response_options,
    message,
) -> None:
    """Missing, duplicate, malformed, or encoded lengths never publish."""
    route = "/strict-cmudict.dict"
    destination = tmp_path / "cmudict.dict"
    local_http_server.enqueue_bytes(route, PAYLOAD, **response_options)

    with pytest.raises((ValueError, requests.RequestException), match=message):
        load_cmudict(
            path=destination,
            url=local_http_server.base_url + route,
            expected_sha256=hashlib.sha256(PAYLOAD).hexdigest(),
            expected_size_bytes=len(PAYLOAD),
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".cmudict.dict.*.tmp"))


def test_atomic_publication_cleans_its_scratch_on_replace_failure(tmp_path) -> None:
    """A real filesystem rejection exercises the writer's cleanup branch."""
    destination = tmp_path / "existing-directory"
    destination.mkdir()

    with pytest.raises(OSError):
        atomic_publish_bytes(destination, PAYLOAD)

    assert destination.is_dir()
    assert not list(tmp_path.glob(".existing-directory.*.tmp"))
