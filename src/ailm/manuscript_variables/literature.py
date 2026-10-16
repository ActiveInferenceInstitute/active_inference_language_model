"""Project-file loaders: claims, config, literature, references, literals, vendor metadata."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ailm.claims import ClaimsLedger
from ailm.manuscript_contract import TOKEN_NAME_RE
from ailm.reference_presentation import (
    InvalidReferenceSourceError as InvalidReferenceSourceError,
)
from ailm.reference_presentation import (
    _load_reference_variables as _load_reference_variables,
)
from ailm.required_claims import REQUIRED_CLAIMS
from ailm.value_formatting import (
    ManuscriptVariableError,
)
from ailm.value_formatting import (
    format_ledger_value as _format_value,
)

from ._claims import _validate_required_claim_values
from ._exceptions import (
    InvalidLiteralVariableError,
    InvalidLiteratureSearchError,
    MissingClaimError,
)
from ._io import (
    DIGEST_SHORT_LENGTH,
    _load_unique_yaml,
    _unique_json_object,
)


def _load_claims(project_root: Path) -> dict[str, Any]:
    """Load and validate the claims ledger without repairing or rewriting it."""
    ledger_path = project_root / "data" / "claims.json"
    if not ledger_path.exists():
        raise MissingClaimError(
            f"Claims ledger not found at {ledger_path}. Run the stage recorders: "
            "scripts/record_stage0_claims.py, record_stage1_claims.py, "
            "record_stage2_claims.py"
        )
    try:
        payload = json.loads(
            ledger_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object
        )
    except json.JSONDecodeError as exc:
        raise MissingClaimError(f"Claims ledger at {ledger_path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise MissingClaimError(f"Claims ledger at {ledger_path} must be a JSON object")
    if payload.get("_schema") != "ailm-claims-v2":
        raise MissingClaimError(
            f"Claims ledger at {ledger_path} has unsupported schema {payload.get('_schema')!r}"
        )
    claims = payload.get("claims")
    if not isinstance(claims, dict):
        raise MissingClaimError(f"Claims ledger at {ledger_path} must contain a claims object")
    try:
        for label, entry in claims.items():
            ClaimsLedger._validate_entry(label, entry)
    except ValueError as exc:
        raise MissingClaimError(f"Invalid claim entry in {ledger_path}: {exc}") from exc

    missing = [label for label in REQUIRED_CLAIMS if label not in claims]
    if missing:
        raise MissingClaimError(
            f"The manuscript requires {len(missing)} claim(s) the ledger does not "
            f"have: {sorted(missing)}. Record them from the run that produced the "
            "value — never hand-write the number into prose."
        )
    _validate_required_claim_values(claims)
    return claims


def _load_config(project_root: Path) -> dict[str, Any]:
    config_path = project_root / "docs" / "manuscript" / "config.yaml"
    if not config_path.exists():
        raise ManuscriptVariableError(f"manuscript configuration not found: {config_path}")
    config = _load_unique_yaml(config_path)
    if not isinstance(config, dict):
        raise ManuscriptVariableError(f"{config_path} must contain a YAML mapping")

    paper = config.get("paper")
    if not isinstance(paper, dict):
        raise ManuscriptVariableError(f"{config_path} must contain a paper mapping")
    for field in ("title", "version"):
        value = paper.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ManuscriptVariableError(f"{config_path} paper.{field} must be non-empty text")

    authors = config.get("authors")
    if not isinstance(authors, list) or not authors:
        raise ManuscriptVariableError(f"{config_path} authors must be a non-empty list")
    for index, author in enumerate(authors):
        if (
            not isinstance(author, dict)
            or not isinstance(author.get("name"), str)
            or not author["name"].strip()
        ):
            raise ManuscriptVariableError(
                f"{config_path} authors[{index}] must contain a non-empty name"
            )

    keywords = config.get("keywords")
    if (
        not isinstance(keywords, list)
        or not keywords
        or any(not isinstance(keyword, str) or not keyword.strip() for keyword in keywords)
    ):
        raise ManuscriptVariableError(f"{config_path} keywords must be non-empty text entries")
    _rendering_variables(config)
    return config


_RENDERING_RANGES = {
    "body_font_pt": (9.0, 12.0),
    "body_baseline_pt": (10.0, 18.0),
    "toc_font_pt": (8.0, 12.0),
    "toc_baseline_pt": (9.0, 18.0),
    "margin_top_in": (0.25, 1.5),
    "margin_bottom_in": (0.25, 1.5),
    "margin_left_in": (0.25, 1.5),
    "margin_right_in": (0.25, 1.5),
}


def _rendering_variables(config: dict[str, Any]) -> dict[str, str]:
    rendering = config.get("rendering")
    if not isinstance(rendering, dict):
        raise ManuscriptVariableError("configuration rendering must be a mapping")
    values: dict[str, str] = {}
    for field, (lower, upper) in _RENDERING_RANGES.items():
        value = rendering.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not lower <= value <= upper
            or not math.isfinite(value)
        ):
            raise ManuscriptVariableError(
                f"rendering.{field} must be finite numeric in [{lower}, {upper}], got {value!r}"
            )
        values[f"META_{field.upper()}"] = str(value)
    for prefix in ("body", "toc"):
        if not 1.1 <= rendering[f"{prefix}_baseline_pt"] / rendering[f"{prefix}_font_pt"] <= 1.6:
            raise ManuscriptVariableError(f"rendering.{prefix} baseline must suit its font size")
    return values


def _load_rendering_variables(project_root: Path) -> dict[str, str]:
    """Read validated article typography without loading measured claims."""
    return _rendering_variables(_load_config(project_root))


_LITERAL_VARIABLE_NAMES = frozenset(
    {
        "DECISION_UNOBSERVED_EMISSION_LEARNING",
        "DECISION_CZECH_CASE_PROTOCOL",
        "DECISION_CONFIRMATORY_REPLICATION",
        "DECISION_REPLICATION_RETIRED",
        "DECISION_SENTENCE_STRIDE_SUPERSEDED",
        "DECISION_DOCUMENT_PARTITION_REPAIR",
        "DECISION_CLAUSE_WORD_NON_PASSING",
        "DECISION_LEXICAL_BACKOFF",
        "DECISION_PYMDP_CONJUGACY",
        "DECISION_NARRATIVE_REFERENCE",
        "DECISION_CASE_WORD_SWEEP",
        "DECISION_CLAUSE_WORD_HARM",
        "DECISION_ONLINE_VS_FROZEN",
    }
)
_DECISION_LITERAL_NAMES = _LITERAL_VARIABLE_NAMES
_MONTH_YEAR_RE = re.compile(
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December) (?:19|20)\d{2}\Z"
)
_ISO_DATE_RE = re.compile(r"(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\Z")


def _decision_ids(project_root: Path) -> frozenset[int]:
    """Decision numbers present in the canonical decision log."""
    path = project_root / "docs" / "decisions.md"
    if not path.is_file():
        raise InvalidLiteralVariableError(
            f"decision-backed literal variables require the decision log: {path}"
        )
    return frozenset(
        int(match.group(1))
        for match in re.finditer(r"^##\s+(\d+)(?:\s|—|\()", path.read_text(encoding="utf-8"), re.M)
    )


def _validate_literal_value(
    project_root: Path, name: str, value: Any, decisions: frozenset[int]
) -> str:
    """Validate one stable literal against its key-specific semantic type."""
    if name in _DECISION_LITERAL_NAMES:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise InvalidLiteralVariableError(
                f"literal variable {name!r} must be a positive decision number, got {value!r}"
            )
        if value not in decisions:
            raise InvalidLiteralVariableError(
                f"literal variable {name!r} points to absent decision {value} in "
                f"{project_root / 'docs' / 'decisions.md'}"
            )
    else:  # pragma: no cover - exact registry comparison guards this branch
        raise InvalidLiteralVariableError(f"literal variable {name!r} has no semantic schema")
    return _format_value(value)


def _load_literature_variables(project_root: Path) -> dict[str, str]:
    """Hydrate search metadata from its reviewable, fail-closed evidence record."""
    path = project_root / "data" / "literature_search.json"
    if not path.is_file():
        raise InvalidLiteratureSearchError(f"literature-search record not found: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object
        )
    except json.JSONDecodeError as exc:
        raise InvalidLiteratureSearchError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("_schema") != "ailm-literature-search-v1":
        raise InvalidLiteratureSearchError(
            f"{path} must declare schema 'ailm-literature-search-v1'"
        )

    conducted_on = payload.get("conducted_on")
    cutoff = payload.get("cutoff_display")
    if not isinstance(conducted_on, str) or _ISO_DATE_RE.fullmatch(conducted_on) is None:
        raise InvalidLiteratureSearchError(f"{path} conducted_on must have ISO YYYY-MM-DD form")
    try:
        expected_cutoff = datetime.fromisoformat(conducted_on).strftime("%B %Y")
    except ValueError as exc:
        raise InvalidLiteratureSearchError(
            f"{path} conducted_on is not a real calendar date: {conducted_on!r}"
        ) from exc
    if cutoff != expected_cutoff or not isinstance(cutoff, str):
        raise InvalidLiteratureSearchError(
            f"{path} cutoff_display must be derived from conducted_on as {expected_cutoff!r}"
        )

    criterion = payload.get("inclusion_criterion")
    expected_facets = {
        "discrete_active_inference_engine",
        "natural_language_corpus_fit",
        "heldout_language_model_metric",
    }
    if (
        not isinstance(criterion, dict)
        or not isinstance(criterion.get("description"), str)
        or len(criterion["description"].split()) < 12
        or set(criterion.get("required_facets", ())) != expected_facets
    ):
        raise InvalidLiteratureSearchError(f"{path} has an incomplete inclusion criterion")

    searches = payload.get("searches")
    if not isinstance(searches, list) or not searches:
        raise InvalidLiteratureSearchError(f"{path} searches must be a non-empty list")
    seen_sources: set[str] = set()
    incomplete = False
    for index, search in enumerate(searches):
        if not isinstance(search, dict):
            raise InvalidLiteratureSearchError(f"{path} searches[{index}] must be an object")
        source = search.get("source")
        endpoint = search.get("endpoint")
        query = search.get("query")
        status = search.get("status")
        returned = search.get("returned_records")
        total = search.get("reported_total")
        failure = search.get("failure")
        if not isinstance(source, str) or not source.strip() or source in seen_sources:
            raise InvalidLiteratureSearchError(
                f"{path} searches[{index}] needs a unique non-empty source"
            )
        seen_sources.add(source)
        if (
            not isinstance(endpoint, str)
            or not endpoint.startswith("https://")
            or not isinstance(query, str)
            or not query.strip()
        ):
            raise InvalidLiteratureSearchError(
                f"{path} searches[{index}] needs an HTTPS endpoint and explicit query"
            )
        if isinstance(returned, bool) or not isinstance(returned, int) or returned < 0:
            raise InvalidLiteratureSearchError(
                f"{path} searches[{index}] returned_records must be non-negative"
            )
        if total is not None and (
            isinstance(total, bool) or not isinstance(total, int) or total < returned
        ):
            raise InvalidLiteratureSearchError(
                f"{path} searches[{index}] reported_total cannot be below returned_records"
            )
        if status == "complete":
            if total != returned or failure is not None:
                raise InvalidLiteratureSearchError(
                    f"{path} searches[{index}] falsely declares completion"
                )
        elif isinstance(status, str) and status.startswith(("incomplete_", "blocked_")):
            incomplete = True
            if not isinstance(failure, str) or not failure.strip():
                raise InvalidLiteratureSearchError(
                    f"{path} searches[{index}] must explain its incomplete status"
                )
        else:
            raise InvalidLiteratureSearchError(
                f"{path} searches[{index}] has unsupported status {status!r}"
            )

    policy = payload.get("claim_policy")
    if not isinstance(policy, dict):
        raise InvalidLiteratureSearchError(f"{path} claim_policy must be an object")
    priority_allowed = policy.get("priority_claim_allowed")
    if not isinstance(priority_allowed, bool):
        raise InvalidLiteratureSearchError(f"{path} priority_claim_allowed must be a JSON boolean")
    if incomplete and priority_allowed:
        raise InvalidLiteratureSearchError(
            f"{path} cannot allow a priority claim while a declared search is incomplete"
        )
    for field in ("reason", "retirement_condition"):
        value = policy.get(field)
        if not isinstance(value, str) or len(value.split()) < 8:
            raise InvalidLiteratureSearchError(
                f"{path} claim_policy.{field} must give a substantive explanation"
            )

    candidates = payload.get("screened_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise InvalidLiteratureSearchError(f"{path} screened_candidates must be a non-empty list")
    candidate_ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise InvalidLiteratureSearchError(
                f"{path} screened_candidates[{index}] must be an object"
            )
        candidate_id = candidate.get("id")
        title = candidate.get("title")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in candidate_ids
            or not isinstance(title, str)
            or not title.strip()
        ):
            raise InvalidLiteratureSearchError(
                f"{path} screened_candidates[{index}] needs a unique id and title"
            )
        candidate_ids.add(candidate_id)
        facets = [candidate.get(name) for name in expected_facets]
        eligible = candidate.get("eligible")
        if any(not isinstance(value, bool) for value in (*facets, eligible)):
            raise InvalidLiteratureSearchError(
                f"{path} screened_candidates[{index}] facet values must be booleans"
            )
        if eligible != all(facets):
            raise InvalidLiteratureSearchError(
                f"{path} screened_candidates[{index}] eligibility contradicts its facets"
            )
        exclusion = candidate.get("exclusion")
        if eligible:
            raise InvalidLiteratureSearchError(
                f"{path} contains eligible counterexample {candidate_id}; revise the manuscript"
            )
        if not isinstance(exclusion, str) or len(exclusion.split()) < 8:
            raise InvalidLiteratureSearchError(
                f"{path} screened_candidates[{index}] needs a substantive exclusion"
            )

    return {
        "LITERATURE_SEARCH_DATE": conducted_on,
        "LITERATURE_SEARCH_CUTOFF": cutoff,
        "LITERATURE_PRIORITY_CLAIM_ALLOWED": _format_value(priority_allowed),
        "LITERATURE_SEARCH_SOURCE_COUNT": str(len(searches)),
        "LITERATURE_SCREENED_CANDIDATE_COUNT": str(len(candidates)),
    }


def _load_literal_variables(project_root: Path) -> dict[str, str]:
    """Load declared non-measurement variables used by the manuscript.

    Results belong in ``data/claims.json``. Decision-log locators have no
    measured run, but they are still variables
    that must have one reviewable source. Deposited-source metadata is excluded
    deliberately: :func:`_load_reference_variables` reads it directly from its
    canonical snapshot. This registry is strict and cannot serve as an
    unstructured exemption list.
    """
    path = project_root / "data" / "manuscript_literals.yaml"
    if not path.exists():
        raise InvalidLiteralVariableError(f"literal variable registry not found: {path}")
    payload = _load_unique_yaml(path)
    if not isinstance(payload, dict):
        raise InvalidLiteralVariableError(f"{path} must contain a YAML mapping")
    if payload.get("schema") != "ailm-manuscript-literals-v2":
        raise InvalidLiteralVariableError(
            f"{path} must declare schema 'ailm-manuscript-literals-v2'"
        )
    entries = payload.get("variables")
    if not isinstance(entries, dict) or not entries:
        raise InvalidLiteralVariableError(f"{path} must contain a non-empty variables mapping")
    names = set(entries)
    invalid_names = sorted(
        name for name in names if not isinstance(name, str) or TOKEN_NAME_RE.fullmatch(name) is None
    )
    if invalid_names:
        raise InvalidLiteralVariableError(
            f"invalid literal variable name(s) {invalid_names!r} in {path}"
        )
    if names != _LITERAL_VARIABLE_NAMES:
        raise InvalidLiteralVariableError(
            "literal variable registry must exactly match its typed schema; "
            f"missing={sorted(_LITERAL_VARIABLE_NAMES - names)}, "
            f"unknown={sorted(names - _LITERAL_VARIABLE_NAMES)}"
        )

    resolved: dict[str, str] = {}
    decisions = _decision_ids(project_root)
    for name, entry in entries.items():
        if not isinstance(entry, dict) or "value" not in entry:
            raise InvalidLiteralVariableError(f"literal variable {name!r} has no value")
        reason = entry.get("reason")
        if not isinstance(reason, str) or len(reason.split()) < 8:
            raise InvalidLiteralVariableError(
                f"literal variable {name!r} needs a substantive reason (at least eight words)"
            )
        value = _validate_literal_value(project_root, name, entry["value"], decisions)
        if not value.strip() or "{{" in value or "}}" in value:
            raise InvalidLiteralVariableError(
                f"literal variable {name!r} resolves to empty or recursive text"
            )
        resolved[name] = value
    return resolved


def _vendor_metadata(project_root: Path) -> dict[str, str]:
    """Size and checksum of each vendored corpus artifact.

    Recorded in the manuscript's reproducibility section so a reader can verify
    they have the same inputs, rather than trusting that "the Brown corpus" and
    "CMUdict" mean the same bytes everywhere.
    """
    out: dict[str, str] = {}
    vendor = project_root / "vendor"
    artifacts = (
        ("brown_counts.json", "META_BROWN_COUNTS"),
        ("cmudict_counts.json", "META_CMUDICT_COUNTS"),
        ("brown_word_counts.json", "META_BROWN_WORD_COUNTS"),
        ("ud_ewt_case_counts.json", "META_UD_CASE_COUNTS"),
    )
    missing = [name for name, _prefix in artifacts if not (vendor / name).is_file()]
    if missing:
        raise ManuscriptVariableError(
            f"required vendored manuscript artifact(s) are missing: {missing}"
        )

    for name, prefix in artifacts:
        path = vendor / name
        payload = path.read_bytes()
        out[f"{prefix}_SHA256"] = hashlib.sha256(payload).hexdigest()
        out[f"{prefix}_SHA256_SHORT"] = hashlib.sha256(payload).hexdigest()[:DIGEST_SHORT_LENGTH]
        # SI units throughout the paper: kB is exactly 1,000 bytes.  The old
        # divisor was 1,024 while the rendered label said KB, silently printing
        # KiB magnitudes under a metric unit.
        out[f"{prefix}_KB"] = str(round(len(payload) / 1000))
    return out
