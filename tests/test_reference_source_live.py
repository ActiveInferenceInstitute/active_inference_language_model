"""Offline, real-HTTP controls for the opt-in Zenodo source verifier."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
import requests

from ailm.reference_sources import (
    MAX_METADATA_BYTES,
    ReferenceVerificationError,
    _expected_references,
    _load_snapshot,
    _measure_pdf_pages,
    _origin,
    _resolve_pdfinfo,
    _response_identity,
    _verify_expected,
    _verify_metadata,
    _verify_reference_payload_on_loopback,
    verify_reference_sources,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = PROJECT_ROOT / "data" / "reference_sources.json"
FIXTURE_ORIGIN = "http://127.0.0.1:8765"


def _minimal_pdf() -> bytes:
    """Construct one valid blank-page PDF with a deterministic xref table."""

    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 72 72] /Resources << >> /Contents 4 0 R >>",
        b"<< /Length 0 >>\nstream\n\nendstream",
    )
    body = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, payload in enumerate(objects, start=1):
        offsets.append(len(body))
        body.extend(f"{number} 0 obj\n".encode("ascii"))
        body.extend(payload)
        body.extend(b"\nendobj\n")
    xref = len(body)
    body.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    body.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        body.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    body.extend(
        (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").encode(
            "ascii"
        )
    )
    return bytes(body)


def _fixture_payload(origin: str, pdf: bytes | None = None, *, pdf_pages: int = 1) -> dict:
    content = _minimal_pdf() if pdf is None else pdf
    record_id = 42
    filename = "reference.pdf"
    return {
        "_schema": "ailm-reference-sources-v1",
        "sources": {
            "fixture": {
                "zenodo_record": {
                    "concept_doi": "10.5281/zenodo.41",
                    "record_doi": f"10.5281/zenodo.{record_id}",
                    "record_id": record_id,
                    "title": "Fixture reference",
                    "version": "v1",
                    "publication_date": "2026-01-01",
                },
                "deposited_file": {
                    "filename": filename,
                    "size_bytes": len(content),
                    "md5": hashlib.md5(content, usedforsecurity=False).hexdigest(),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "pdf_pages": pdf_pages,
                    "download_url": (f"{origin}/api/records/{record_id}/files/{filename}/content"),
                },
            }
        },
    }


def _metadata(payload: dict, *, title: str | None = None) -> dict:
    source = payload["sources"]["fixture"]
    record = source["zenodo_record"]
    deposited = source["deposited_file"]
    record_url = deposited["download_url"].split("/files/", 1)[0]
    return {
        "id": record["record_id"],
        "doi": record["record_doi"],
        "conceptdoi": record["concept_doi"],
        "links": {"self": record_url},
        "metadata": {
            "title": record["title"] if title is None else title,
            "version": record["version"],
            "publication_date": record["publication_date"],
        },
        "files": [
            {
                "key": deposited["filename"],
                "size": deposited["size_bytes"],
                "checksum": f"md5:{deposited['md5']}",
                "links": {"self": deposited["download_url"]},
            }
        ],
    }


def _routes(payload: dict) -> tuple[str, str]:
    source = payload["sources"]["fixture"]
    record_id = source["zenodo_record"]["record_id"]
    return (
        f"/api/records/{record_id}",
        source["deposited_file"]["download_url"].split(payload_origin(payload), 1)[1],
    )


def payload_origin(payload: dict) -> str:
    url = payload["sources"]["fixture"]["deposited_file"]["download_url"]
    return url.split("/api/", 1)[0]


def _expected_fixture(payload: dict, origin: str):
    return _expected_references(
        payload,
        origin=origin,
        allow_loopback_http=True,
    )[0]


def test_official_snapshot_targets_only_exact_https_zenodo_records() -> None:
    payload = _load_snapshot(SNAPSHOT_PATH)
    expected = _expected_references(
        payload,
        origin="https://zenodo.org",
        allow_loopback_http=False,
    )
    assert {entry.source_key for entry in expected} == {"cerebrum", "compositional_case"}
    assert all(entry.record_url.startswith("https://zenodo.org/api/records/") for entry in expected)


@pytest.mark.parametrize(
    "replacement",
    (
        "http://zenodo.org/api/records/15231156/files/CEREBRUM_1-4_April-16-2025.pdf/content",
        "https://example.invalid/api/records/15231156/files/CEREBRUM_1-4_April-16-2025.pdf/content",
    ),
)
def test_official_snapshot_rejects_protocol_or_host_drift(replacement: str) -> None:
    payload = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    payload["sources"]["cerebrum"]["deposited_file"]["download_url"] = replacement
    with pytest.raises(ReferenceVerificationError, match="host, protocol"):
        _expected_references(
            payload,
            origin="https://zenodo.org",
            allow_loopback_http=False,
        )


def test_duplicate_snapshot_key_is_rejected_instead_of_last_wins(tmp_path: Path) -> None:
    planted = tmp_path / "reference_sources.json"
    planted.write_text(
        '{"_schema":"ailm-reference-sources-v1","_schema":"attacker-selected","sources":{}}',
        encoding="utf-8",
    )
    with pytest.raises(ReferenceVerificationError, match="duplicate JSON key '_schema'"):
        _load_snapshot(planted)


@pytest.mark.parametrize(
    ("content", "message"),
    (
        (None, "not found"),
        (b"\xff", "not UTF-8"),
        (b"{", "invalid JSON"),
        (b"[]", "must be a JSON object"),
    ),
)
def test_snapshot_loader_fails_closed_on_invalid_files(
    tmp_path: Path, content: bytes | None, message: str
) -> None:
    path = tmp_path / "reference_sources.json"
    if content is not None:
        path.write_bytes(content)
    with pytest.raises(ReferenceVerificationError, match=message):
        _load_snapshot(path)


def test_public_verifier_loads_and_validates_before_network(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(ReferenceVerificationError, match="not found"):
        verify_reference_sources(missing)


@pytest.mark.parametrize(
    ("origin", "allow_loopback", "message"),
    (
        ("https://user@zenodo.org", False, "user information"),
        ("https://zenodo.org/api", False, "path, query, or fragment"),
        ("https://zenodo.org?record=1", False, "path, query, or fragment"),
        ("https://zenodo.org:443", False, "must not specify a port"),
        ("http://zenodo.org", False, "restricted"),
        ("https://example.invalid", False, "restricted"),
        ("http://127.0.0.1", True, "explicit loopback HTTP port"),
        ("https://127.0.0.1:8080", True, "explicit loopback HTTP port"),
        ("http://localhost:8080", True, "loopback IP literal"),
        ("http://127.0.0.1:not-a-port", True, "invalid port"),
    ),
)
def test_reference_origin_policy_rejects_ambiguous_targets(
    origin: str, allow_loopback: bool, message: str
) -> None:
    with pytest.raises(ReferenceVerificationError, match=message):
        _origin(origin, allow_loopback_http=allow_loopback)


def test_reference_origin_policy_normalizes_ipv6_loopback() -> None:
    assert _origin("http://[::1]:8765", allow_loopback_http=True) == "http://[::1]:8765"


def test_reference_schema_requires_a_nonempty_source_mapping() -> None:
    for payload, message in (
        ({"_schema": "wrong", "sources": {}}, "declare schema"),
        ({"_schema": "ailm-reference-sources-v1", "sources": None}, "JSON object"),
        ({"_schema": "ailm-reference-sources-v1", "sources": {}}, "has no sources"),
    ):
        with pytest.raises(ReferenceVerificationError, match=message):
            _expected_references(
                payload,
                origin=FIXTURE_ORIGIN,
                allow_loopback_http=True,
            )


def test_reference_schema_rejects_invalid_source_keys_and_objects() -> None:
    valid = _fixture_payload(FIXTURE_ORIGIN)
    source = valid["sources"]["fixture"]
    for sources, message in (
        ({"": source}, "source keys"),
        ({7: source}, "source keys"),
        ({"fixture": None}, "must be a JSON object"),
    ):
        payload = {"_schema": "ailm-reference-sources-v1", "sources": sources}
        with pytest.raises(ReferenceVerificationError, match=message):
            _expected_references(
                payload,
                origin=FIXTURE_ORIGIN,
                allow_loopback_http=True,
            )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("zenodo_record",), None, "zenodo_record must be a JSON object"),
        (("deposited_file",), None, "deposited_file must be a JSON object"),
        (("zenodo_record", "record_id"), True, "positive integer"),
        (("zenodo_record", "title"), "", "non-empty string"),
        (("zenodo_record", "publication_date"), "January", "ISO date"),
        (("zenodo_record", "record_doi"), "10.5281/zenodo.99", "does not identify"),
        (("zenodo_record", "concept_doi"), "doi:41", "not a Zenodo DOI"),
        (("zenodo_record", "concept_doi"), "10.5281/zenodo.42", "collapses"),
        (("deposited_file", "filename"), "reference.txt", "safe PDF basename"),
        (("deposited_file", "filename"), "../reference.pdf", "safe PDF basename"),
        (("deposited_file", "size_bytes"), 0, "positive integer"),
        (("deposited_file", "pdf_pages"), False, "positive integer"),
        (("deposited_file", "md5"), "F" * 32, "lowercase digest"),
        (("deposited_file", "sha256"), "0" * 63, "lowercase digest"),
    ),
)
def test_reference_schema_rejects_malformed_identity_fields(
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    payload = _fixture_payload(FIXTURE_ORIGIN)
    source = payload["sources"]["fixture"]
    cursor = source
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(ReferenceVerificationError, match=message):
        _expected_references(
            payload,
            origin=FIXTURE_ORIGIN,
            allow_loopback_http=True,
        )


def test_real_http_metadata_matches_the_exact_record(local_http_server) -> None:
    payload = _fixture_payload(local_http_server.base_url)
    record_route, _ = _routes(payload)
    local_http_server.enqueue_json(record_route, _metadata(payload))

    verified = _verify_reference_payload_on_loopback(
        payload,
        origin=local_http_server.base_url,
    )

    assert len(verified) == 1
    assert verified[0].metadata_verified is True
    assert verified[0].file_verified is False
    assert verified[0].to_dict()["source_key"] == "fixture"
    request = local_http_server.requests_for(record_route)[0]
    assert request.headers["accept"] == "application/json"
    assert request.headers["accept-encoding"] == "identity"


def test_real_http_metadata_drift_is_detected(local_http_server) -> None:
    payload = _fixture_payload(local_http_server.base_url)
    record_route, _ = _routes(payload)
    local_http_server.enqueue_json(record_route, _metadata(payload, title="Drifted title"))

    with pytest.raises(ReferenceVerificationError, match="metadata.title"):
        _verify_reference_payload_on_loopback(payload, origin=local_http_server.base_url)


@pytest.mark.parametrize(
    ("replacement", "value", "message"),
    (
        (("metadata",), None, "metadata must be a JSON object"),
        (("links",), None, "links must be a JSON object"),
        (("id",), 99, "metadata drift at id"),
        (("doi",), "wrong", "metadata drift at doi"),
        (("conceptdoi",), "wrong", "metadata drift at conceptdoi"),
        (("links", "self"), "http://elsewhere.invalid", "metadata drift at links.self"),
        (("metadata", "version"), "drift", "metadata.version"),
        (("metadata", "publication_date"), "drift", "metadata.publication_date"),
        (("files",), None, "files must be a JSON array"),
        (("files",), [None], "non-object entry"),
        (("files",), [], "inventory drifted"),
        (("files", "0", "links"), None, "links must be a JSON object"),
        (("files", "0", "size"), 1, r"files\[0\].size"),
        (("files", "0", "checksum"), "md5:wrong", r"files\[0\].checksum"),
        (
            ("files", "0", "links", "self"),
            "http://elsewhere.invalid",
            r"files\[0\].links.self",
        ),
    ),
)
def test_metadata_contract_rejects_each_identity_drift(
    replacement: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    payload = _fixture_payload(FIXTURE_ORIGIN)
    metadata = _metadata(payload)
    cursor = metadata
    for component in replacement[:-1]:
        cursor = cursor[int(component)] if isinstance(cursor, list) else cursor[component]
    component = replacement[-1]
    if isinstance(cursor, list):
        cursor[int(component)] = value
    else:
        cursor[component] = value
    with pytest.raises(ReferenceVerificationError, match=message):
        _verify_metadata(metadata, _expected_fixture(payload, FIXTURE_ORIGIN))


def test_real_http_redirect_drift_is_refused_without_contacting_target(
    local_http_server,
) -> None:
    payload = _fixture_payload(local_http_server.base_url)
    record_route, _ = _routes(payload)
    target_route = "/redirected-record"
    local_http_server.enqueue_bytes(
        record_route,
        b"",
        status=302,
        headers={"Location": local_http_server.base_url + target_route},
    )
    local_http_server.enqueue_json(target_route, _metadata(payload))

    with pytest.raises(ReferenceVerificationError, match="refused redirect"):
        _verify_reference_payload_on_loopback(payload, origin=local_http_server.base_url)

    assert len(local_http_server.requests_for(record_route)) == 1
    assert not local_http_server.requests_for(target_route)


@pytest.mark.parametrize(
    ("body", "options", "message"),
    (
        (b"{}", {"status": 503}, "HTTP status"),
        (b"{}", {"headers": {"Location": "/unexpected"}}, "unexpected Location"),
        (b"{}", {"headers": {"Content-Range": "bytes 0-1/2"}}, "Content-Range"),
        (b"{}", {"headers": {"Content-Encoding": "gzip"}}, "Content-Encoding"),
        (b"{}", {"headers": {"Transfer-Encoding": "chunked"}}, "Transfer-Encoding"),
        (b"{}", {"content_type": "text/plain"}, "application/json Content-Type"),
        (
            b"{}",
            {"headers": {"Content-Type": "application/json"}},
            "application/json Content-Type",
        ),
        (b"{}", {"omit_content_length": True}, "exactly one Content-Length"),
        (b"{}", {"declared_length": "not-a-number"}, "invalid Content-Length"),
        (b"", {"content_type": "application/json"}, "invalid or oversized"),
        (
            b"{}",
            {"declared_length": MAX_METADATA_BYTES + 1},
            "invalid or oversized",
        ),
        (b"\xff", {"content_type": "application/json"}, "not UTF-8"),
        (b"{", {"content_type": "application/json"}, "invalid JSON"),
        (b"[]", {"content_type": "application/json"}, "must be a JSON object"),
    ),
)
def test_real_http_metadata_response_framing_fails_closed(
    local_http_server,
    body: bytes,
    options: dict,
    message: str,
) -> None:
    payload = _fixture_payload(local_http_server.base_url)
    record_route, _ = _routes(payload)
    enqueue_options = {"content_type": "application/json", **options}
    local_http_server.enqueue_bytes(record_route, body, **enqueue_options)
    with pytest.raises(ReferenceVerificationError, match=message):
        _verify_reference_payload_on_loopback(payload, origin=local_http_server.base_url)


def test_real_http_truncated_metadata_body_fails_closed(local_http_server) -> None:
    payload = _fixture_payload(local_http_server.base_url)
    record_route, _ = _routes(payload)
    body = json.dumps(_metadata(payload)).encode("utf-8")
    local_http_server.enqueue_bytes(
        record_route,
        body,
        content_type="application/json",
        declared_length=len(body),
        truncate_after=len(body) - 1,
    )
    with pytest.raises(ReferenceVerificationError, match="interrupted|had .* bytes"):
        _verify_reference_payload_on_loopback(payload, origin=local_http_server.base_url)


def test_real_http_disconnect_is_reported_as_a_request_failure(local_http_server) -> None:
    payload = _fixture_payload(local_http_server.base_url)
    record_route, _ = _routes(payload)
    local_http_server.enqueue_disconnect(record_route)
    with pytest.raises(ReferenceVerificationError, match="request failed"):
        _verify_reference_payload_on_loopback(payload, origin=local_http_server.base_url)


def test_response_identity_rejects_history_and_final_url_drift(local_http_server) -> None:
    redirect_route = "/history"
    target_route = "/history-target"
    local_http_server.enqueue_bytes(
        redirect_route,
        b"",
        status=302,
        headers={"Location": local_http_server.base_url + target_route},
    )
    local_http_server.enqueue_json(target_route, {"ok": True})
    with (
        requests.get(
            local_http_server.base_url + redirect_route,
            allow_redirects=True,
            timeout=(1, 1),
        ) as response,
        pytest.raises(ReferenceVerificationError, match="redirect history"),
    ):
        _response_identity(
            response,
            expected_url=response.url,
            source_name="fixture",
        )

    direct_route = "/direct"
    local_http_server.enqueue_json(direct_route, {"ok": True})
    with (
        requests.get(
            local_http_server.base_url + direct_route,
            allow_redirects=False,
            timeout=(1, 1),
        ) as response,
        pytest.raises(ReferenceVerificationError, match="response URL drifted"),
    ):
        _response_identity(
            response,
            expected_url=local_http_server.base_url + "/different",
            source_name="fixture",
        )


def test_real_http_full_file_verifies_extent_and_both_digests(local_http_server) -> None:
    content = _minimal_pdf()
    payload = _fixture_payload(local_http_server.base_url, content)
    record_route, file_route = _routes(payload)
    local_http_server.enqueue_json(record_route, _metadata(payload))
    local_http_server.enqueue_bytes(file_route, content)

    verified = _verify_reference_payload_on_loopback(
        payload,
        origin=local_http_server.base_url,
        full_files=True,
    )

    result = verified[0]
    assert result.file_verified is True
    assert result.size_bytes == len(content)
    assert result.md5 == hashlib.md5(content, usedforsecurity=False).hexdigest()
    assert result.sha256 == hashlib.sha256(content).hexdigest()
    assert local_http_server.requests_for(file_route)[0].headers["accept"] == "*/*"
    if shutil.which("pdfinfo") is None:
        assert result.pdf_extent_status == "unavailable"
        assert result.measured_pdf_pages is None
    else:
        assert result.pdf_extent_status == "verified"
        assert result.measured_pdf_pages == 1


def test_real_http_truncated_file_body_fails_closed(local_http_server) -> None:
    content = _minimal_pdf()
    payload = _fixture_payload(local_http_server.base_url, content)
    record_route, file_route = _routes(payload)
    local_http_server.enqueue_json(record_route, _metadata(payload))
    local_http_server.enqueue_bytes(
        file_route,
        content,
        declared_length=len(content),
        truncate_after=len(content) - 1,
    )

    with pytest.raises(ReferenceVerificationError, match="interrupted|does not match"):
        _verify_reference_payload_on_loopback(
            payload,
            origin=local_http_server.base_url,
            full_files=True,
        )


def test_real_http_oversized_file_body_fails_before_hash_acceptance(local_http_server) -> None:
    content = _minimal_pdf()
    payload = _fixture_payload(local_http_server.base_url, content)
    record_route, file_route = _routes(payload)
    local_http_server.enqueue_json(record_route, _metadata(payload))
    local_http_server.enqueue_bytes(file_route, content + b"attacker-controlled-suffix")

    with pytest.raises(ReferenceVerificationError, match="declared .* expected|exceeded"):
        _verify_reference_payload_on_loopback(
            payload,
            origin=local_http_server.base_url,
            full_files=True,
        )


def test_real_http_md5_control_rejects_a_body_after_sha256_authentication(
    local_http_server,
) -> None:
    content = _minimal_pdf()
    payload = _fixture_payload(local_http_server.base_url, content)
    payload["sources"]["fixture"]["deposited_file"]["md5"] = "0" * 32
    record_route, file_route = _routes(payload)
    local_http_server.enqueue_json(record_route, _metadata(payload))
    local_http_server.enqueue_bytes(file_route, content)

    with pytest.raises(ReferenceVerificationError, match="MD5 drift"):
        _verify_reference_payload_on_loopback(
            payload,
            origin=local_http_server.base_url,
            full_files=True,
        )


def test_full_file_mode_can_report_unavailable_pdf_tool_without_claiming_extent(
    local_http_server,
) -> None:
    content = _minimal_pdf()
    payload = _fixture_payload(local_http_server.base_url, content)
    record_route, file_route = _routes(payload)
    local_http_server.enqueue_json(record_route, _metadata(payload))
    local_http_server.enqueue_bytes(file_route, content)

    result = _verify_reference_payload_on_loopback(
        payload,
        origin=local_http_server.base_url,
        full_files=True,
        pdfinfo_path="ailm-definitely-missing-pdfinfo",
    )[0]
    assert result.pdf_extent_status == "unavailable"
    assert result.measured_pdf_pages is None


@pytest.mark.parametrize("timeout", (True, 0, -1, float("inf"), float("nan"), "fast"))
def test_verifier_rejects_invalid_timeout_before_network(timeout: object) -> None:
    with pytest.raises(ReferenceVerificationError, match="finite and positive"):
        _verify_expected(
            (),
            full_files=False,
            timeout_seconds=timeout,  # type: ignore[arg-type]  # deliberate runtime control
            pdfinfo_path=None,
            require_pdfinfo=False,
        )


def test_pdfinfo_requirement_contract_fails_before_network() -> None:
    with pytest.raises(ReferenceVerificationError, match="only with full_files"):
        _verify_expected(
            (),
            full_files=False,
            timeout_seconds=1,
            pdfinfo_path=None,
            require_pdfinfo=True,
        )
    with pytest.raises(ReferenceVerificationError, match="no executable"):
        _verify_expected(
            (),
            full_files=True,
            timeout_seconds=1,
            pdfinfo_path="ailm-definitely-missing-pdfinfo",
            require_pdfinfo=True,
        )


def _executable(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o700)
    return path


def test_pdfinfo_resolution_and_execution_fail_closed(tmp_path: Path) -> None:
    assert _resolve_pdfinfo("ailm-definitely-missing-pdfinfo") is None

    fifo = tmp_path / "pdfinfo-fifo"
    os.mkfifo(fifo)
    fifo.chmod(0o700)
    with pytest.raises(ReferenceVerificationError, match="executable regular file"):
        _resolve_pdfinfo(fifo)

    invalid_executable = _executable(tmp_path / "invalid-executable", "not an executable image\n")
    with pytest.raises(ReferenceVerificationError, match="execution failed"):
        _measure_pdf_pages(
            _minimal_pdf(),
            pdfinfo=invalid_executable,
            source_name="fixture",
        )

    rejecting = _executable(
        tmp_path / "rejecting-pdfinfo",
        "#!/bin/sh\necho malformed PDF >&2\nexit 1\n",
    )
    with pytest.raises(ReferenceVerificationError, match="rejected.*malformed PDF"):
        _measure_pdf_pages(_minimal_pdf(), pdfinfo=rejecting, source_name="fixture")

    no_pages = _executable(
        tmp_path / "no-pages-pdfinfo",
        "#!/bin/sh\necho 'PDF version: 1.4'\n",
    )
    with pytest.raises(ReferenceVerificationError, match="omitted Pages"):
        _measure_pdf_pages(_minimal_pdf(), pdfinfo=no_pages, source_name="fixture")


def test_pdf_extent_reader_rejects_non_pdf_before_tool_execution(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist"
    tool = _executable(
        tmp_path / "marker-pdfinfo",
        f"#!/bin/sh\ntouch {marker}\necho 'Pages: 1'\n",
    )
    with pytest.raises(ReferenceVerificationError, match="lacks a PDF header"):
        _measure_pdf_pages(b"not a PDF", pdfinfo=tool, source_name="fixture")
    assert not marker.exists()


@pytest.mark.skipif(shutil.which("pdfinfo") is None, reason="Poppler pdfinfo is unavailable")
def test_real_pdf_extent_drift_is_rejected_by_local_poppler(local_http_server) -> None:
    content = _minimal_pdf()
    payload = _fixture_payload(local_http_server.base_url, content, pdf_pages=2)
    record_route, file_route = _routes(payload)
    local_http_server.enqueue_json(record_route, _metadata(payload))
    local_http_server.enqueue_bytes(file_route, content)

    with pytest.raises(ReferenceVerificationError, match="PDF extent drift"):
        _verify_reference_payload_on_loopback(
            payload,
            origin=local_http_server.base_url,
            full_files=True,
            require_pdfinfo=True,
        )


def test_non_loopback_integration_override_is_rejected_before_network() -> None:
    payload = _fixture_payload("http://192.0.2.1:8765")
    with pytest.raises(ReferenceVerificationError, match="not loopback"):
        _verify_reference_payload_on_loopback(payload, origin="http://192.0.2.1:8765")
