"""Stdlib-only presentation of canonical deposited sources and local protocol links.

This is the shared offline presenter used by manuscript and documentation
hydration. It performs no measurement, model import, network request or write.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ailm.value_formatting import ManuscriptVariableError
from ailm.value_formatting import format_ledger_value as _format_value


class InvalidReferenceSourceError(ManuscriptVariableError):
    """Raised when canonical deposited-source metadata cannot produce tokens."""

    __module__ = "ailm.manuscript_variables._exceptions"


class DuplicateInputKeyError(ManuscriptVariableError):
    """Raised when JSON or YAML repeats a key and would otherwise last-win."""

    __module__ = "ailm.manuscript_variables._exceptions"


_ISO_DATE_RE = re.compile(r"(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\Z")

# Stable local publication destinations. The documentation render hook validates
# these Markdown labels against the actual supporting HTML before publication.
PROTOCOL_REFERENCE_ANCHORS = {
    "PROTOCOL_P1_FIT_EVALUATION_URL": "protocol-p1-fit-evaluation",
    "PROTOCOL_P2_ALIGNED_CORRESPONDENCE_URL": "protocol-p2-aligned-correspondence",
    "PROTOCOL_P3_BRIDGE_URL": "protocol-p3-bridge",
}
PROTOCOL_SUPPORTING_HTML = "ailm-supporting-docs/docs__extension_protocols.html"


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Construct one JSON object while refusing duplicate member names."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateInputKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_reference_variables(project_root: Path) -> dict[str, str]:
    """Hydrate deposited-source identity and locators from one canonical snapshot.

    ``data/reference_sources.json`` is already the exact deposited-source
    inventory. Copying its values into the free-form literal registry made every
    version, date, and locator independently mutable while a parity test merely
    checked the current copies happened to agree. This loader reads the source
    snapshot directly and refuses duplicate JSON keys, absent fields, or locator
    shapes that do not mean what the token name says.
    """
    path = project_root / "data" / "reference_sources.json"
    if not path.is_file():
        raise InvalidReferenceSourceError(f"reference-source snapshot not found: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object
        )
    except json.JSONDecodeError as exc:
        raise InvalidReferenceSourceError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("_schema") != "ailm-reference-sources-v1":
        raise InvalidReferenceSourceError(f"{path} must declare schema 'ailm-reference-sources-v1'")

    def leaf(*keys: str) -> Any:
        value: Any = payload
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                raise InvalidReferenceSourceError(
                    f"{path} is missing canonical reference field {'.'.join(keys)!r}"
                )
            value = value[key]
        return value

    version_fields = {
        "CEREBRUM_ZENODO_VERSION": leaf("sources", "cerebrum", "zenodo_record", "version"),
        "CEREBRUM_PRINTED_VERSION": leaf("sources", "cerebrum", "printed_identity", "version"),
    }
    date_fields = {
        "CEREBRUM_ZENODO_DATE": leaf("sources", "cerebrum", "zenodo_record", "publication_date"),
        "CEREBRUM_PRINTED_DATE": leaf("sources", "cerebrum", "printed_identity", "date"),
    }
    page_fields = {
        "CEREBRUM_PDF_PAGE_COUNT": leaf("sources", "cerebrum", "deposited_file", "pdf_pages"),
        "COMPOSITIONAL_CASE_PDF_PAGE_COUNT": leaf(
            "sources", "compositional_case", "deposited_file", "pdf_pages"
        ),
    }
    for name, value in version_fields.items():
        if not isinstance(value, str) or re.fullmatch(r"\d+(?:\.\d+)+", value) is None:
            raise InvalidReferenceSourceError(
                f"canonical reference field for {name} must be a dotted numeric version, "
                f"got {value!r}"
            )
    for name, value in date_fields.items():
        if not isinstance(value, str) or _ISO_DATE_RE.fullmatch(value) is None:
            raise InvalidReferenceSourceError(
                f"canonical reference field for {name} must have ISO YYYY-MM-DD form, got {value!r}"
            )
    for name, value in page_fields.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise InvalidReferenceSourceError(
                f"canonical reference field for {name} must be a positive page count, got {value!r}"
            )

    locator_fields = {
        "CEREBRUM_GENERATIVE_SUPPLEMENT": (
            leaf(
                "sources",
                "cerebrum",
                "pdf_evidence",
                "novel_generative_case",
                "locator",
            ),
            r"Supplement ([1-9]\d*(?:\.\d+)+)",
        ),
        "CEREBRUM_CASE_SUPPLEMENT": (
            leaf(
                "sources",
                "cerebrum",
                "pdf_evidence",
                "core_case_formulations_locator",
            ),
            r"Supplement ([1-9]\d*)",
        ),
        "CEREBRUM_ACTIVE_INFERENCE_FIGURE": (
            leaf(
                "sources",
                "cerebrum",
                "pdf_evidence",
                "active_inference_figure",
                "locator",
            ),
            r"Figure ([1-9]\d*)",
        ),
        "CEREBRUM_POMDP_SUPPLEMENT": (
            leaf("sources", "cerebrum", "pdf_evidence", "pomdp_tuple", "locator"),
            r"Supplement ([1-9]\d*(?:\.\d+)+)",
        ),
    }
    for token, evidence_key in (
        ("CEREBRUM_TRANSFORMATION_SUPPLEMENT", "case_transition_free_energy"),
        ("CEREBRUM_EFE_TRANSITION_SUPPLEMENT", "expected_free_energy_transition"),
        ("CEREBRUM_CORE_FUNCTION_TABLE", "core_function_table"),
    ):
        locator_fields[token] = (
            leaf("sources", "cerebrum", "pdf_evidence", evidence_key, "locator"),
            r"Table ([1-9]\d*)" if token.endswith("TABLE") else r"Supplement ([1-9]\d*(?:\.\d+)+)",
        )
    locators: dict[str, str] = {}
    for name, (raw, pattern) in locator_fields.items():
        match = re.fullmatch(pattern, raw) if isinstance(raw, str) else None
        if match is None:
            raise InvalidReferenceSourceError(
                f"canonical reference locator for {name} has the wrong shape: {raw!r}"
            )
        locators[name] = match.group(1)

    record_id = leaf("sources", "cerebrum", "zenodo_record", "record_id")
    filename = leaf("sources", "cerebrum", "deposited_file", "filename")
    download_url = leaf("sources", "cerebrum", "deposited_file", "download_url")
    if (
        isinstance(record_id, bool)
        or not isinstance(record_id, int)
        or record_id <= 0
        or not isinstance(filename, str)
        or not filename.endswith(".pdf")
        or "/" in filename
        or download_url
        != f"https://zenodo.org/api/records/{record_id}/files/{quote(filename, safe='')}/content"
    ):
        raise InvalidReferenceSourceError(
            "canonical deposited PDF URL must match its record and filename"
        )
    page_locators = {
        "CEREBRUM_POMDP_SUPPLEMENT_URL": leaf(
            "sources", "cerebrum", "pdf_evidence", "pomdp_tuple", "physical_page"
        ),
        "CEREBRUM_GENERATIVE_SUPPLEMENT_URL": leaf(
            "sources", "cerebrum", "pdf_evidence", "novel_generative_case", "physical_pages"
        ),
        "CEREBRUM_CASE_SUPPLEMENT_URL": leaf(
            "sources", "cerebrum", "pdf_evidence", "core_case_formulations", "NOM", "physical_page"
        ),
        "CEREBRUM_ACTIVE_INFERENCE_FIGURE_URL": leaf(
            "sources", "cerebrum", "pdf_evidence", "active_inference_figure", "physical_page"
        ),
    }
    for token, evidence_key in (
        ("CEREBRUM_TRANSFORMATION_SUPPLEMENT", "case_transition_free_energy"),
        ("CEREBRUM_EFE_TRANSITION_SUPPLEMENT", "expected_free_energy_transition"),
        ("CEREBRUM_CORE_FUNCTION_TABLE", "core_function_table"),
    ):
        page_locators[f"{token}_URL"] = leaf(
            "sources", "cerebrum", "pdf_evidence", evidence_key, "physical_page"
        )
    page_locators["CEREBRUM_GENITIVE_CASE_URL"] = leaf(
        "sources", "cerebrum", "pdf_evidence", "core_case_formulations", "GEN", "physical_page"
    )
    physical_pages: dict[str, str] = {}
    links: dict[str, str] = {}
    for name, recorded in page_locators.items():
        pages = recorded if isinstance(recorded, list) else [recorded]
        if (
            not pages
            or any(
                isinstance(page, bool)
                or not isinstance(page, int)
                or not 1 <= page <= page_fields["CEREBRUM_PDF_PAGE_COUNT"]
                for page in pages
            )
            or pages != sorted(set(pages))
        ):
            raise InvalidReferenceSourceError(
                f"canonical physical page locator for {name} is invalid: {recorded!r}"
            )
        links[name] = f"{download_url}#page={pages[0]}"
        physical_pages[name.removesuffix("_URL") + "_PAGES"] = ", ".join(map(str, pages))

    identity_fields: dict[str, str] = {}
    for prefix, source_key in (
        ("CEREBRUM", "cerebrum"),
        ("COMPOSITIONAL_CASE", "compositional_case"),
    ):
        for suffix, keys in {
            "CONCEPT_DOI": ("zenodo_record", "concept_doi"),
            "RECORD_DOI": ("zenodo_record", "record_doi"),
            "RECORD_TITLE": ("zenodo_record", "title"),
            "PRINTED_TITLE": ("printed_identity", "title"),
            "PDF_FILENAME": ("deposited_file", "filename"),
            "PDF_MD5": ("deposited_file", "md5"),
            "PDF_SHA256": ("deposited_file", "sha256"),
        }.items():
            value = leaf("sources", source_key, *keys)
            if not isinstance(value, str) or not value.strip():
                raise InvalidReferenceSourceError(
                    f"canonical identity field for {prefix}_{suffix} must be nonempty text"
                )
            identity_fields[f"{prefix}_{suffix}"] = value
        size = leaf("sources", source_key, "deposited_file", "size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise InvalidReferenceSourceError(
                f"canonical PDF size for {prefix} must be a positive integer"
            )
        identity_fields[f"{prefix}_PDF_SIZE_BYTES"] = _format_value(size)
    for suffix, keys in {
        "ZENODO_VERSION": ("zenodo_record", "version"),
        "ZENODO_DATE": ("zenodo_record", "publication_date"),
        "PRINTED_DATE": ("printed_identity", "date"),
        "COMPANION_REVISION": ("optional_companion_repository", "revision"),
        "COMPANION_VERSION": ("optional_companion_repository", "manuscript_version"),
        "COMPANION_DATE": ("optional_companion_repository", "manuscript_date"),
    }.items():
        value = leaf("sources", "compositional_case", *keys)
        if not isinstance(value, str) or not value.strip():
            raise InvalidReferenceSourceError(
                f"canonical identity field for COMPOSITIONAL_CASE_{suffix} must be nonempty text"
            )
        identity_fields[f"COMPOSITIONAL_CASE_{suffix}"] = value
    protocol_links = {
        name: f"{PROTOCOL_SUPPORTING_HTML}#{anchor}"
        for name, anchor in PROTOCOL_REFERENCE_ANCHORS.items()
    }
    return {
        **links,
        **physical_pages,
        **identity_fields,
        **protocol_links,
        **{name: _format_value(value) for name, value in version_fields.items()},
        **{name: _format_value(value) for name, value in date_fields.items()},
        **{name: _format_value(value) for name, value in page_fields.items()},
        **locators,
    }
