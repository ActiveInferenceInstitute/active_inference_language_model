"""Opt-in verification of AILM's exact deposited Zenodo references.

The committed reference snapshot remains the offline source of truth for normal
tests and rendering.  This module is the explicitly networked revalidation path:
it compares the snapshot with exact-version Zenodo API records and can additionally
authenticate the complete deposited PDFs.

The public entry point has no endpoint override.  It accepts only ``https`` URLs
on ``zenodo.org`` whose paths are derived from the recorded version identifier and
filename.  A private loopback-only integration seam exists so the transport's
failure modes can be exercised by a real local HTTP server without putting network
access in the default test suite.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import requests

from ailm.corpus._download import _bounded_response_chunks, read_authenticated_response

__all__ = [
    "ReferenceVerificationError",
    "VerifiedReference",
    "verify_reference_sources",
]

ZENODO_ORIGIN = "https://zenodo.org"
REFERENCE_SCHEMA = "ailm-reference-sources-v1"
MAX_METADATA_BYTES = 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 30.0

_MD5_RE = re.compile(r"[0-9a-f]{32}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_FILENAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_PAGES_RE = re.compile(r"^Pages:\s+([1-9]\d*)\s*$", re.MULTILINE)


class ReferenceVerificationError(ValueError):
    """A snapshot, Zenodo response, deposited file, or local PDF tool drifted."""


@dataclass(frozen=True, slots=True)
class _ExpectedReference:
    """Validated fields needed to re-check one exact Zenodo version."""

    source_key: str
    record_id: int
    concept_doi: str
    record_doi: str
    title: str
    version: str
    publication_date: str
    filename: str
    size_bytes: int
    md5: str
    sha256: str
    pdf_pages: int
    record_url: str
    download_url: str


@dataclass(frozen=True, slots=True)
class VerifiedReference:
    """Evidence returned for one source after live verification."""

    source_key: str
    record_id: int
    record_url: str
    metadata_verified: bool
    file_verified: bool
    filename: str
    size_bytes: int | None
    md5: str | None
    sha256: str | None
    expected_pdf_pages: int
    measured_pdf_pages: int | None
    pdf_extent_status: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable verification record."""

        return asdict(self)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while refusing last-key-wins parsing."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReferenceVerificationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_snapshot(path: Path) -> dict[str, Any]:
    """Load one UTF-8 reference snapshot without accepting duplicate keys."""

    if not path.is_file():
        raise ReferenceVerificationError(f"reference-source snapshot not found: {path}")
    try:
        decoded = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReferenceVerificationError(f"reference-source snapshot is not UTF-8: {path}") from exc
    try:
        payload = json.loads(decoded, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ReferenceVerificationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReferenceVerificationError("reference-source snapshot must be a JSON object")
    return payload


def _origin(origin: str, *, allow_loopback_http: bool) -> str:
    """Validate one origin; production permits only the exact Zenodo origin."""

    parsed = urlsplit(origin)
    if parsed.username is not None or parsed.password is not None:
        raise ReferenceVerificationError("reference origin must not contain user information")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ReferenceVerificationError(
            "reference origin must not contain a path, query, or fragment"
        )
    try:
        parsed_port = parsed.port
    except ValueError as exc:
        raise ReferenceVerificationError(f"reference origin has an invalid port: {exc}") from exc
    if parsed_port is not None:
        if not allow_loopback_http:
            raise ReferenceVerificationError("official Zenodo origin must not specify a port")
        port = parsed_port
    else:
        port = None
    hostname = (parsed.hostname or "").lower()
    if not allow_loopback_http:
        if parsed.scheme != "https" or hostname != "zenodo.org" or port is not None:
            raise ReferenceVerificationError(
                "live reference verification is restricted to https://zenodo.org"
            )
        return ZENODO_ORIGIN

    if parsed.scheme != "http" or port is None:
        raise ReferenceVerificationError(
            "test reference origin must be an explicit loopback HTTP port"
        )
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError as exc:
        raise ReferenceVerificationError(
            "test reference origin must use a loopback IP literal"
        ) from exc
    if not address.is_loopback:
        raise ReferenceVerificationError("test reference origin is not loopback")
    host = f"[{address}]" if address.version == 6 else str(address)
    return urlunsplit(("http", f"{host}:{port}", "", "", ""))


def _required_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReferenceVerificationError(f"{label} must be a JSON object")
    return value


def _required_string(mapping: dict[str, Any], key: str, *, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ReferenceVerificationError(f"{label}.{key} must be a non-empty string")
    return value


def _positive_integer(mapping: dict[str, Any], key: str, *, label: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReferenceVerificationError(f"{label}.{key} must be a positive integer")
    return value


def _expected_references(
    payload: dict[str, Any],
    *,
    origin: str,
    allow_loopback_http: bool,
) -> tuple[_ExpectedReference, ...]:
    """Validate the live-verification subset of the committed snapshot."""

    normalized_origin = _origin(origin, allow_loopback_http=allow_loopback_http)
    if payload.get("_schema") != REFERENCE_SCHEMA:
        raise ReferenceVerificationError(
            f"reference-source snapshot must declare schema {REFERENCE_SCHEMA!r}"
        )
    sources = _required_mapping(payload.get("sources"), label="sources")
    if not sources:
        raise ReferenceVerificationError("reference-source snapshot has no sources")
    if any(not isinstance(key, str) or not key for key in sources):
        raise ReferenceVerificationError("reference source keys must be non-empty strings")

    expected: list[_ExpectedReference] = []
    for source_key in sorted(sources):
        source = _required_mapping(sources[source_key], label=f"sources.{source_key}")
        record_label = f"sources.{source_key}.zenodo_record"
        file_label = f"sources.{source_key}.deposited_file"
        record = _required_mapping(source.get("zenodo_record"), label=record_label)
        deposited = _required_mapping(source.get("deposited_file"), label=file_label)

        record_id = _positive_integer(record, "record_id", label=record_label)
        concept_doi = _required_string(record, "concept_doi", label=record_label)
        record_doi = _required_string(record, "record_doi", label=record_label)
        title = _required_string(record, "title", label=record_label)
        version = _required_string(record, "version", label=record_label)
        publication_date = _required_string(record, "publication_date", label=record_label)
        if not _DATE_RE.fullmatch(publication_date):
            raise ReferenceVerificationError(
                f"{record_label}.publication_date must have ISO date form"
            )
        if record_doi != f"10.5281/zenodo.{record_id}":
            raise ReferenceVerificationError(
                f"{record_label}.record_doi does not identify record_id {record_id}"
            )
        if not re.fullmatch(r"10\.5281/zenodo\.[1-9]\d*", concept_doi):
            raise ReferenceVerificationError(f"{record_label}.concept_doi is not a Zenodo DOI")
        if concept_doi == record_doi:
            raise ReferenceVerificationError(
                f"{record_label} collapses the concept and exact-version DOI"
            )

        filename = _required_string(deposited, "filename", label=file_label)
        if not _FILENAME_RE.fullmatch(filename) or not filename.lower().endswith(".pdf"):
            raise ReferenceVerificationError(f"{file_label}.filename must be one safe PDF basename")
        size_bytes = _positive_integer(deposited, "size_bytes", label=file_label)
        pdf_pages = _positive_integer(deposited, "pdf_pages", label=file_label)
        md5 = _required_string(deposited, "md5", label=file_label)
        sha256 = _required_string(deposited, "sha256", label=file_label)
        if not _MD5_RE.fullmatch(md5):
            raise ReferenceVerificationError(f"{file_label}.md5 must be a lowercase digest")
        if not _SHA256_RE.fullmatch(sha256):
            raise ReferenceVerificationError(f"{file_label}.sha256 must be a lowercase digest")

        record_url = f"{normalized_origin}/api/records/{record_id}"
        encoded_filename = quote(filename, safe="-._~")
        canonical_download = f"{record_url}/files/{encoded_filename}/content"
        download_url = _required_string(deposited, "download_url", label=file_label)
        if download_url != canonical_download:
            raise ReferenceVerificationError(
                f"{file_label}.download_url drifted from the exact record, host, protocol, or filename"
            )

        expected.append(
            _ExpectedReference(
                source_key=source_key,
                record_id=record_id,
                concept_doi=concept_doi,
                record_doi=record_doi,
                title=title,
                version=version,
                publication_date=publication_date,
                filename=filename,
                size_bytes=size_bytes,
                md5=md5,
                sha256=sha256,
                pdf_pages=pdf_pages,
                record_url=record_url,
                download_url=download_url,
            )
        )
    return tuple(expected)


def _response_identity(response: requests.Response, *, expected_url: str, source_name: str) -> None:
    """Require the response to remain on the one exact request target."""

    if response.history:
        raise ReferenceVerificationError(f"{source_name} response contains redirect history")
    if response.url != expected_url:
        raise ReferenceVerificationError(
            f"{source_name} response URL drifted: {response.url!r} != {expected_url!r}"
        )
    if 300 <= response.status_code < 400:
        location = response.headers.get("Location", "<missing>")
        raise ReferenceVerificationError(
            f"{source_name} refused redirect {response.status_code} to {location!r}"
        )
    if response.status_code != 200:
        raise ReferenceVerificationError(
            f"{source_name} returned HTTP status {response.status_code}, expected 200"
        )
    if response.headers.get("Location") is not None:
        raise ReferenceVerificationError(f"{source_name} returned an unexpected Location header")
    if response.headers.get("Content-Range") is not None:
        raise ReferenceVerificationError(f"{source_name} returned a partial Content-Range")


def _one_content_length(response: requests.Response, *, source_name: str) -> int:
    values = tuple(str(value) for value in response.raw.headers.getlist("Content-Length"))
    if len(values) != 1:
        raise ReferenceVerificationError(
            f"{source_name} requires exactly one Content-Length; received {values!r}"
        )
    text = values[0].strip()
    if not text.isascii() or not text.isdecimal():
        raise ReferenceVerificationError(
            f"{source_name} returned invalid Content-Length {values[0]!r}"
        )
    return int(text)


def _read_metadata_response(
    response: requests.Response,
    *,
    expected_url: str,
    source_name: str,
) -> dict[str, Any]:
    """Read one bounded, identity-encoded JSON response without redirects."""

    _response_identity(response, expected_url=expected_url, source_name=source_name)
    content_encoding = response.headers.get("Content-Encoding")
    if content_encoding is not None and content_encoding.strip().lower() != "identity":
        raise ReferenceVerificationError(
            f"{source_name} refused Content-Encoding {content_encoding!r}"
        )
    if response.headers.get("Transfer-Encoding") is not None:
        raise ReferenceVerificationError(f"{source_name} refused Transfer-Encoding")
    content_types = tuple(str(value) for value in response.raw.headers.getlist("Content-Type"))
    if len(content_types) != 1 or content_types[0].split(";", 1)[0].strip().lower() != (
        "application/json"
    ):
        raise ReferenceVerificationError(
            f"{source_name} requires one application/json Content-Type; received {content_types!r}"
        )
    declared_size = _one_content_length(response, source_name=source_name)
    if declared_size <= 0 or declared_size > MAX_METADATA_BYTES:
        raise ReferenceVerificationError(
            f"{source_name} declared an invalid or oversized metadata body"
        )
    try:
        body = b"".join(
            _bounded_response_chunks(
                response, max_bytes=min(declared_size + 1, MAX_METADATA_BYTES + 1)
            )
        )
    except requests.RequestException as exc:
        raise ReferenceVerificationError(f"{source_name} metadata body was interrupted") from exc
    if len(body) != declared_size:
        raise ReferenceVerificationError(
            f"{source_name} metadata body had {len(body)} bytes, expected {declared_size}"
        )
    try:
        decoded = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReferenceVerificationError(f"{source_name} metadata is not UTF-8") from exc
    try:
        payload = json.loads(decoded, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ReferenceVerificationError(f"{source_name} returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReferenceVerificationError(f"{source_name} metadata must be a JSON object")
    return payload


def _same(actual: object, expected: object, *, field: str, source_name: str) -> None:
    if actual != expected:
        raise ReferenceVerificationError(
            f"{source_name} metadata drift at {field}: {actual!r} != {expected!r}"
        )


def _verify_metadata(payload: dict[str, Any], expected: _ExpectedReference) -> None:
    """Compare Zenodo's live exact-version record with the committed identity."""

    source_name = expected.source_key
    metadata = _required_mapping(payload.get("metadata"), label=f"{source_name}.metadata")
    links = _required_mapping(payload.get("links"), label=f"{source_name}.links")
    _same(payload.get("id"), expected.record_id, field="id", source_name=source_name)
    _same(payload.get("doi"), expected.record_doi, field="doi", source_name=source_name)
    _same(
        payload.get("conceptdoi"),
        expected.concept_doi,
        field="conceptdoi",
        source_name=source_name,
    )
    _same(links.get("self"), expected.record_url, field="links.self", source_name=source_name)
    _same(metadata.get("title"), expected.title, field="metadata.title", source_name=source_name)
    _same(
        metadata.get("version"),
        expected.version,
        field="metadata.version",
        source_name=source_name,
    )
    _same(
        metadata.get("publication_date"),
        expected.publication_date,
        field="metadata.publication_date",
        source_name=source_name,
    )

    files = payload.get("files")
    if not isinstance(files, list):
        raise ReferenceVerificationError(f"{source_name}.files must be a JSON array")
    file_entries = [entry for entry in files if isinstance(entry, dict)]
    if len(file_entries) != len(files):
        raise ReferenceVerificationError(f"{source_name}.files contains a non-object entry")
    keys = [entry.get("key") for entry in file_entries]
    if keys != [expected.filename]:
        raise ReferenceVerificationError(
            f"{source_name} deposited-file inventory drifted: {keys!r}"
        )
    entry = file_entries[0]
    file_links = _required_mapping(entry.get("links"), label=f"{source_name}.files[0].links")
    _same(entry.get("size"), expected.size_bytes, field="files[0].size", source_name=source_name)
    _same(
        entry.get("checksum"),
        f"md5:{expected.md5}",
        field="files[0].checksum",
        source_name=source_name,
    )
    _same(
        file_links.get("self"),
        expected.download_url,
        field="files[0].links.self",
        source_name=source_name,
    )


def _resolve_pdfinfo(pdfinfo_path: str | os.PathLike[str] | None) -> Path | None:
    """Resolve one executable PDF extent reader, if the host provides it."""

    candidate = (
        shutil.which("pdfinfo") if pdfinfo_path is None else shutil.which(os.fspath(pdfinfo_path))
    )
    if candidate is None:
        return None
    try:
        resolved = Path(candidate).resolve(strict=True)
        mode = resolved.stat().st_mode
    except OSError as exc:
        raise ReferenceVerificationError(
            f"cannot resolve pdfinfo executable {candidate!r}"
        ) from exc
    if not stat.S_ISREG(mode) or not os.access(resolved, os.X_OK):
        raise ReferenceVerificationError(
            f"pdfinfo path is not an executable regular file: {resolved}"
        )
    return resolved


def _measure_pdf_pages(content: bytes, *, pdfinfo: Path, source_name: str) -> int:
    """Invoke Poppler pdfinfo on already digest-authenticated bytes."""

    if not content.startswith(b"%PDF-"):
        raise ReferenceVerificationError(f"{source_name} deposited file lacks a PDF header")
    with tempfile.TemporaryDirectory(prefix="ailm-reference-pdf-") as directory_name:
        directory = Path(directory_name)
        pdf_path = directory / "deposited.pdf"
        with pdf_path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            completed = subprocess.run(
                [str(pdfinfo), str(pdf_path)],
                cwd=directory,
                env={"LANG": "C", "LC_ALL": "C"},
                capture_output=True,
                check=False,
                timeout=DEFAULT_TIMEOUT_SECONDS,
                text=True,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ReferenceVerificationError(
                f"{source_name} pdfinfo execution failed: {exc}"
            ) from exc
    if completed.returncode != 0:
        diagnostic = " ".join(completed.stderr.split())
        raise ReferenceVerificationError(
            f"{source_name} pdfinfo rejected the deposited file: {diagnostic}"
        )
    match = _PAGES_RE.search(completed.stdout)
    if match is None:
        raise ReferenceVerificationError(f"{source_name} pdfinfo output omitted Pages")
    return int(match.group(1))


def _request(
    session: requests.Session,
    url: str,
    *,
    accept: str,
    timeout_seconds: float,
) -> requests.Response:
    """Open one proxy-free, redirect-free, identity-encoded streaming request."""

    try:
        return session.get(
            url,
            stream=True,
            allow_redirects=False,
            timeout=(timeout_seconds, timeout_seconds),
            headers={
                "Accept": accept,
                "Accept-Encoding": "identity",
                "User-Agent": "ailm-reference-verifier/1",
            },
        )
    except requests.RequestException as exc:
        raise ReferenceVerificationError(f"request failed for {url}: {exc}") from exc


def _verify_expected(
    references: tuple[_ExpectedReference, ...],
    *,
    full_files: bool,
    timeout_seconds: float,
    pdfinfo_path: str | os.PathLike[str] | None,
    require_pdfinfo: bool,
) -> tuple[VerifiedReference, ...]:
    """Run metadata and optional full-file checks over validated targets."""

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise ReferenceVerificationError("timeout_seconds must be finite and positive")
    if require_pdfinfo and not full_files:
        raise ReferenceVerificationError("require_pdfinfo is meaningful only with full_files")
    pdfinfo = _resolve_pdfinfo(pdfinfo_path) if full_files else None
    if require_pdfinfo and pdfinfo is None:
        raise ReferenceVerificationError("pdfinfo is required but no executable was found")

    session = requests.Session()
    session.trust_env = False
    results: list[VerifiedReference] = []
    try:
        for expected in references:
            record_response = _request(
                session,
                expected.record_url,
                accept="application/json",
                timeout_seconds=float(timeout_seconds),
            )
            with record_response:
                metadata = _read_metadata_response(
                    record_response,
                    expected_url=expected.record_url,
                    source_name=f"{expected.source_key} record",
                )
            _verify_metadata(metadata, expected)

            if not full_files:
                results.append(
                    VerifiedReference(
                        source_key=expected.source_key,
                        record_id=expected.record_id,
                        record_url=expected.record_url,
                        metadata_verified=True,
                        file_verified=False,
                        filename=expected.filename,
                        size_bytes=None,
                        md5=None,
                        sha256=None,
                        expected_pdf_pages=expected.pdf_pages,
                        measured_pdf_pages=None,
                        pdf_extent_status="not-requested",
                    )
                )
                continue

            file_response = _request(
                session,
                expected.download_url,
                # Zenodo's content endpoint rejects media-specific Accept values
                # even though its successful response is an octet stream.
                accept="*/*",
                timeout_seconds=float(timeout_seconds),
            )
            with file_response:
                _response_identity(
                    file_response,
                    expected_url=expected.download_url,
                    source_name=f"{expected.source_key} deposited file",
                )
                try:
                    content = read_authenticated_response(
                        file_response,
                        expected_size_bytes=expected.size_bytes,
                        expected_sha256=expected.sha256,
                        source_name=f"{expected.source_key} deposited file",
                        integrity_error=ReferenceVerificationError,
                    )
                except requests.RequestException as exc:
                    raise ReferenceVerificationError(
                        f"{expected.source_key} deposited-file body was interrupted"
                    ) from exc

            measured_md5 = hashlib.md5(content, usedforsecurity=False).hexdigest()
            if measured_md5 != expected.md5:
                raise ReferenceVerificationError(
                    f"{expected.source_key} deposited-file MD5 drift: "
                    f"{measured_md5} != {expected.md5}"
                )
            measured_sha256 = hashlib.sha256(content).hexdigest()
            measured_pages = None
            extent_status = "unavailable"
            if pdfinfo is not None:
                measured_pages = _measure_pdf_pages(
                    content,
                    pdfinfo=pdfinfo,
                    source_name=expected.source_key,
                )
                if measured_pages != expected.pdf_pages:
                    raise ReferenceVerificationError(
                        f"{expected.source_key} PDF extent drift: "
                        f"{measured_pages} pages != {expected.pdf_pages}"
                    )
                extent_status = "verified"

            results.append(
                VerifiedReference(
                    source_key=expected.source_key,
                    record_id=expected.record_id,
                    record_url=expected.record_url,
                    metadata_verified=True,
                    file_verified=True,
                    filename=expected.filename,
                    size_bytes=len(content),
                    md5=measured_md5,
                    sha256=measured_sha256,
                    expected_pdf_pages=expected.pdf_pages,
                    measured_pdf_pages=measured_pages,
                    pdf_extent_status=extent_status,
                )
            )
    finally:
        session.close()
    return tuple(results)


def _verify_reference_payload_on_loopback(
    payload: dict[str, Any],
    *,
    origin: str,
    full_files: bool = False,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    pdfinfo_path: str | os.PathLike[str] | None = None,
    require_pdfinfo: bool = False,
) -> tuple[VerifiedReference, ...]:
    """Private real-HTTP integration seam restricted to loopback test servers."""

    references = _expected_references(
        payload,
        origin=origin,
        allow_loopback_http=True,
    )
    return _verify_expected(
        references,
        full_files=full_files,
        timeout_seconds=timeout_seconds,
        pdfinfo_path=pdfinfo_path,
        require_pdfinfo=require_pdfinfo,
    )


def verify_reference_sources(
    snapshot_path: str | os.PathLike[str],
    *,
    full_files: bool = False,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    pdfinfo_path: str | os.PathLike[str] | None = None,
    require_pdfinfo: bool = False,
) -> tuple[VerifiedReference, ...]:
    """Revalidate the committed snapshot against official exact-version records.

    Normal tests and manuscript rendering do not call this function.  Invoking it
    explicitly performs network I/O only to the exact ``zenodo.org`` record and
    file URLs derived from the snapshot.  Full-file mode authenticates bounded
    bytes against both recorded digests; when Poppler ``pdfinfo`` is available it
    also checks the recorded page extent.
    """

    payload = _load_snapshot(Path(snapshot_path))
    references = _expected_references(
        payload,
        origin=ZENODO_ORIGIN,
        allow_loopback_http=False,
    )
    return _verify_expected(
        references,
        full_files=full_files,
        timeout_seconds=timeout_seconds,
        pdfinfo_path=pdfinfo_path,
        require_pdfinfo=require_pdfinfo,
    )
