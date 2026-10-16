"""Refusals of the manuscript-variable loaders and helpers, one input at a time.

The generator hydrates every number and locator the manuscript prints from a
handful of reviewable records. Each loader below is exercised with the
narrowest malformed record that reaches one specific refusal, using a copy of
the shipped record under a temporary project root, and the exact message is
matched so a refusal for the wrong reason cannot read as a pass.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

from ailm.manuscript_variables import (
    _DECISION_LITERAL_NAMES,
    _REQUIRED_BOOLEAN_CLAIMS,
    _REQUIRED_STRING_CLAIMS,
    _REQUIRED_STRING_LIST_CLAIMS,
    DuplicateManuscriptTokenError,
    InvalidLiteralVariableError,
    InvalidLiteratureSearchError,
    InvalidReferenceSourceError,
    ManuscriptVariableError,
    MissingClaimError,
    _build_timestamp,
    _claim_number,
    _count_word,
    _decision_ids,
    _g2p_bucket_variables,
    _integer_success_count,
    _load_claims,
    _load_config,
    _load_literal_variables,
    _load_literature_variables,
    _load_reference_variables,
    _ordinal_word,
    _require_paired_table,
    _StrictVariableMap,
    _validate_literal_value,
    _validate_required_claim_values,
    paired_difference_bound_from_counts,
)
from tests.test_manuscript_variables import _write_project

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# -- small pure helpers ------------------------------------------------------


def test_merging_an_owned_token_names_every_duplicate() -> None:
    tokens = _StrictVariableMap()
    tokens["ALPHA"] = "1"
    tokens["BETA"] = "2"
    with pytest.raises(DuplicateManuscriptTokenError, match=r"\['ALPHA', 'BETA'\]"):
        tokens.add_many({"ALPHA": "x", "BETA": "y", "GAMMA": "z"}, producer="fixture")
    assert "GAMMA" not in tokens


def test_build_stamp_refuses_an_epoch_outside_the_calendar(monkeypatch) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "99999999999999999999")
    with pytest.raises(ManuscriptVariableError, match="outside the supported timestamp range"):
        _build_timestamp()


def test_count_and_ordinal_words_refuse_values_outside_their_tables() -> None:
    with pytest.raises(ManuscriptVariableError, match="no manuscript count-word rendering"):
        _count_word(10**6)
    with pytest.raises(ManuscriptVariableError, match="no manuscript ordinal rendering"):
        _ordinal_word(0)


def test_claim_number_refuses_absent_non_numeric_and_non_finite_values() -> None:
    with pytest.raises(MissingClaimError, match="needs numeric 'absent'"):
        _claim_number({}, "absent")
    with pytest.raises(MissingClaimError, match="needs numeric 'flat'"):
        _claim_number({"flat": 3.0}, "flat")
    for bad in ("3", True, None):
        with pytest.raises(ManuscriptVariableError, match="needs numeric 'typed'"):
            _claim_number({"typed": {"value": bad}}, "typed")
    with pytest.raises(ManuscriptVariableError, match="needs finite 'nan'"):
        _claim_number({"nan": {"value": math.nan}}, "nan")


def test_success_count_must_come_from_an_integer() -> None:
    with pytest.raises(ManuscriptVariableError, match="cannot come from an integer success count"):
        _integer_success_count("rate", 0.3333, 10)


def test_paired_table_refuses_a_negative_cell() -> None:
    with pytest.raises(ManuscriptVariableError, match="negative paired-table cell"):
        _require_paired_table("t", first_successes=-1, second_successes=0, b=0, c=0, total=1)


def test_paired_bound_refuses_a_non_positive_population() -> None:
    with pytest.raises(ValueError, match="n must be positive"):
        paired_difference_bound_from_counts(1, 1, 0)


def test_g2p_buckets_must_have_valid_inclusive_ranges() -> None:
    with pytest.raises(ManuscriptVariableError, match="invalid inclusive range 3..2"):
        _g2p_bucket_variables((("short", 0, 1), ("mid", 3, 2), ("long", 4, 5)))


# -- required-claim value typing --------------------------------------------


def test_required_text_claim_must_be_non_empty_text() -> None:
    label = sorted(_REQUIRED_STRING_CLAIMS)[0]
    with pytest.raises(ManuscriptVariableError, match="must be non-empty text"):
        _validate_required_claim_values({label: {"value": "   "}})


def test_required_boolean_claim_must_be_boolean() -> None:
    label = sorted(_REQUIRED_BOOLEAN_CLAIMS)[0]
    with pytest.raises(ManuscriptVariableError, match="must be boolean"):
        _validate_required_claim_values({label: {"value": "yes"}})


@pytest.mark.parametrize("value", [["a", "a"], ["a", ""], "a,b", [1]])
def test_required_string_list_claim_must_be_a_duplicate_free_list(value) -> None:
    label = sorted(_REQUIRED_STRING_LIST_CLAIMS)[0]
    with pytest.raises(ManuscriptVariableError, match="duplicate-free list"):
        _validate_required_claim_values({label: {"value": value}})


# -- the ledger and configuration loaders -----------------------------------


def _root_with_claims_text(tmp_path: Path, text: str) -> Path:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "claims.json").write_text(text, encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("{not json", "is not valid JSON"),
        ("[]", "must be a JSON object"),
        ('{"_schema": "ailm-claims-v1", "claims": {}}', "unsupported schema"),
        ('{"_schema": "ailm-claims-v2", "claims": []}', "must contain a claims object"),
    ],
)
def test_ledger_loader_refuses_malformed_ledgers(tmp_path, text, message) -> None:
    with pytest.raises(MissingClaimError, match=message):
        _load_claims(_root_with_claims_text(tmp_path, text))


def test_config_loader_refuses_a_non_mapping(tmp_path) -> None:
    config = tmp_path / "docs" / "manuscript" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ManuscriptVariableError, match="must contain a YAML mapping"):
        _load_config(tmp_path)


# -- decision-backed literals ------------------------------------------------


def test_decision_ids_require_the_decision_log(tmp_path) -> None:
    with pytest.raises(InvalidLiteralVariableError, match="require the decision log"):
        _decision_ids(tmp_path)


@pytest.mark.parametrize("value", [0, -1, True, "77", 7.0])
def test_decision_literals_must_be_positive_decision_numbers(tmp_path, value) -> None:
    name = sorted(_DECISION_LITERAL_NAMES)[0]
    with pytest.raises(InvalidLiteralVariableError, match="must be a positive decision number"):
        _validate_literal_value(tmp_path, name, value, frozenset({77}))


def _literal_registry_rejection(tmp_path: Path, mutate, message: str) -> None:
    root = _write_project(tmp_path, {})
    registry = root / "data" / "manuscript_literals.yaml"
    payload = yaml.safe_load(registry.read_text(encoding="utf-8"))
    name = sorted(payload["variables"])[0]
    mutate(payload["variables"][name])
    registry.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(InvalidLiteralVariableError, match=f"{name!r} {message}"):
        _load_literal_variables(root)


def test_literal_entries_must_carry_a_value(tmp_path) -> None:
    _literal_registry_rejection(tmp_path, lambda entry: entry.pop("value"), "has no value")


def test_literal_entries_must_carry_a_substantive_reason(tmp_path) -> None:
    _literal_registry_rejection(
        tmp_path, lambda entry: entry.__setitem__("reason", "because"), "needs a substantive"
    )


# -- the literature-search evidence record ------------------------------------


def _literature_rejection(tmp_path: Path, mutate, message: str) -> None:
    root = _write_project(tmp_path, {})
    record = root / "data" / "literature_search.json"
    payload = json.loads(record.read_text(encoding="utf-8"))
    mutate(payload)
    record.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(InvalidLiteratureSearchError, match=message):
        _load_literature_variables(root)


def test_literature_record_must_be_valid_json(tmp_path) -> None:
    root = _write_project(tmp_path, {})
    (root / "data" / "literature_search.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(InvalidLiteratureSearchError, match="invalid JSON"):
        _load_literature_variables(root)


def _first_search(payload: dict) -> dict:
    return payload["searches"][0]


def _first_candidate(payload: dict) -> dict:
    return payload["screened_candidates"][0]


def _declare_incomplete_without_reason(payload: dict) -> None:
    search = _first_search(payload)
    search["status"] = "incomplete_rate_limited"
    search["failure"] = None
    payload["claim_policy"]["priority_claim_allowed"] = False


def _declare_complete_with_a_failure(payload: dict) -> None:
    search = _first_search(payload)
    search["status"] = "complete"
    search["reported_total"] = search["returned_records"]
    search["failure"] = "the endpoint closed the connection"


def _under_report_the_total(payload: dict) -> None:
    search = _first_search(payload)
    search["returned_records"] = 5
    search["reported_total"] = 4


def _duplicate_the_first_source(payload: dict) -> None:
    searches = payload["searches"]
    if len(searches) > 1:
        searches[1]["source"] = searches[0]["source"]
    else:
        searches[0]["source"] = ""


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.__setitem__("_schema", "ailm-literature-search-v0"), "must declare schema"),
        (lambda p: p.__setitem__("conducted_on", "20260101"), "ISO YYYY-MM-DD form"),
        (lambda p: p.__setitem__("conducted_on", "2026-02-30"), "not a real calendar date"),
        (lambda p: p.__setitem__("inclusion_criterion", {}), "incomplete inclusion criterion"),
        (lambda p: p.__setitem__("searches", []), "searches must be a non-empty list"),
        (lambda p: p.__setitem__("searches", ["pubmed"]), r"searches\[0\] must be an object"),
        (_duplicate_the_first_source, "unique non-empty source"),
        (
            lambda p: _first_search(p).__setitem__("endpoint", "http://example.org/search"),
            "HTTPS endpoint and explicit query",
        ),
        (
            lambda p: _first_search(p).__setitem__("returned_records", -1),
            "returned_records must be non-negative",
        ),
        (_under_report_the_total, "cannot be below returned_records"),
        (_declare_complete_with_a_failure, "falsely declares completion"),
        (_declare_incomplete_without_reason, "must explain its incomplete status"),
        (lambda p: _first_search(p).__setitem__("status", "paused"), "unsupported status"),
        (lambda p: p.__setitem__("claim_policy", []), "claim_policy must be an object"),
        (
            lambda p: p["claim_policy"].__setitem__("priority_claim_allowed", "no"),
            "must be a JSON boolean",
        ),
        (
            lambda p: p["claim_policy"].__setitem__("reason", "too short"),
            "claim_policy.reason must give a substantive explanation",
        ),
        (lambda p: p.__setitem__("screened_candidates", []), "must be a non-empty list"),
        (
            lambda p: p.__setitem__("screened_candidates", ["paper"]),
            r"screened_candidates\[0\] must be an object",
        ),
        (lambda p: _first_candidate(p).__setitem__("id", ""), "needs a unique id and title"),
        (
            lambda p: _first_candidate(p).__setitem__("eligible", "no"),
            "facet values must be booleans",
        ),
        (
            lambda p: _first_candidate(p).__setitem__("exclusion", "too short"),
            "needs a substantive exclusion",
        ),
    ],
    ids=(
        "schema",
        "date-form",
        "date-calendar",
        "criterion",
        "no-searches",
        "search-shape",
        "duplicate-source",
        "endpoint",
        "returned",
        "total",
        "false-completion",
        "incomplete-reason",
        "status",
        "policy-shape",
        "policy-flag",
        "policy-reason",
        "no-candidates",
        "candidate-shape",
        "candidate-id",
        "candidate-facet",
        "candidate-exclusion",
    ),
)
def test_literature_record_refuses_each_planted_defect(tmp_path, mutate, message) -> None:
    _literature_rejection(tmp_path, mutate, message)


# -- the reference-source snapshot -------------------------------------------


def _reference_rejection(tmp_path: Path, mutate, message: str) -> None:
    root = _write_project(tmp_path, {})
    snapshot = root / "data" / "reference_sources.json"
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    mutate(payload)
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(InvalidReferenceSourceError, match=message):
        _load_reference_variables(root)


def test_reference_snapshot_must_be_valid_json(tmp_path) -> None:
    root = _write_project(tmp_path, {})
    (root / "data" / "reference_sources.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(InvalidReferenceSourceError, match="invalid JSON"):
        _load_reference_variables(root)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.__setitem__("_schema", "ailm-reference-sources-v0"), "must declare schema"),
        (
            lambda p: p["sources"]["cerebrum"]["zenodo_record"].pop("version"),
            "missing canonical reference field",
        ),
        (
            lambda p: p["sources"]["cerebrum"]["zenodo_record"].__setitem__("version", "v1"),
            "must be a dotted numeric version",
        ),
        (
            lambda p: p["sources"]["cerebrum"]["printed_identity"].__setitem__(
                "date", "April 2025"
            ),
            "must have ISO YYYY-MM-DD form",
        ),
    ],
    ids=("schema", "missing-leaf", "version-form", "date-form"),
)
def test_reference_snapshot_refuses_each_planted_defect(tmp_path, mutate, message) -> None:
    _reference_rejection(tmp_path, mutate, message)
