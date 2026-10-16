"""Rejection contracts of the Czech case artifact and its pinned source shards.

The study-design validator decides whether the word "preregistered" may be
used at all; the artifact loader binds every population to an exact document
partition and source pin; the shard helpers refuse anything that is not the
byte-identical upstream file. Each test plants one defect and matches the
exact message, so a refusal for the wrong reason cannot read as a pass.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ailm.corpus.ud_czech import (
    UD_CZECH_PDTC_SHARDS,
    UDCzechSourceIntegrityError,
    read_verified_ud_czech_shard,
    validate_ud_czech_source_metadata,
    verified_ud_czech_sources,
    verify_ud_czech_shard,
    verify_ud_czech_shard_bytes,
)
from ailm.levels.case_czech import (
    CzechCaseArtifact,
    load_czech_case_artifact,
    validate_study_design_provenance,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = PROJECT_ROOT / "vendor" / "ud_czech_case_counts.json"
STAGING = PROJECT_ROOT / "vendor" / "staging_ud_czech"


def _prior_artifact() -> dict:
    return {
        "path": "docs/protocols/czech-case.md",
        "git_commit": "a" * 40,
        "sha256": "b" * 64,
        "committed_at": "2026-08-20T12:00:00+00:00",
    }


def _preregistered(**overrides) -> dict:
    prior = _prior_artifact()
    prior.update(overrides)
    return {
        "protocol_status": "preregistered",
        "independently_timestamped": True,
        "immutable_prior_artifact": prior,
    }


# -- study-design terminology ------------------------------------------------


@pytest.mark.parametrize("provenance", [None, [], {"protocol_status": "preregistered"}])
def test_study_design_must_carry_exactly_its_three_fields(provenance) -> None:
    with pytest.raises(ValueError, match="study_design must contain exactly"):
        validate_study_design_provenance(provenance)


def test_study_design_rejects_an_unsupported_status() -> None:
    planted = _preregistered()
    planted["protocol_status"] = "registered-report"
    with pytest.raises(ValueError, match="unsupported study-design protocol_status"):
        validate_study_design_provenance(planted)


def test_study_design_rejects_a_non_boolean_timestamp_flag() -> None:
    planted = _preregistered()
    planted["independently_timestamped"] = 1
    with pytest.raises(ValueError, match="independently_timestamped must be a boolean"):
        validate_study_design_provenance(planted)


@pytest.mark.parametrize("field", ["independently_timestamped", "immutable_prior_artifact"])
def test_a_prospective_specification_may_not_claim_prior_evidence(field) -> None:
    planted = {
        "protocol_status": "prospective-development-session-specification",
        "independently_timestamped": False,
        "immutable_prior_artifact": None,
    }
    planted[field] = True if field == "independently_timestamped" else _prior_artifact()
    with pytest.raises(ValueError, match="prospective development-session specification"):
        validate_study_design_provenance(planted)


@pytest.mark.parametrize("path", ["", "/abs/protocol.md", "../escape.md", 3])
def test_prior_artifact_path_must_be_repository_relative(path) -> None:
    with pytest.raises(ValueError, match="repository-relative path"):
        validate_study_design_provenance(_preregistered(path=path))


@pytest.mark.parametrize("commit", ["A" * 40, "a" * 39, 7])
def test_prior_artifact_commit_must_be_a_full_lowercase_hash(commit) -> None:
    with pytest.raises(ValueError, match="git_commit must be 40 lowercase hex"):
        validate_study_design_provenance(_preregistered(git_commit=commit))


@pytest.mark.parametrize("digest", ["B" * 64, "b" * 63, None])
def test_prior_artifact_digest_must_be_a_full_lowercase_hash(digest) -> None:
    with pytest.raises(ValueError, match="sha256 must be 64 lowercase hex"):
        validate_study_design_provenance(_preregistered(sha256=digest))


@pytest.mark.parametrize("committed_at", [20260820, "not a date"])
def test_prior_artifact_timestamp_must_be_iso_text(committed_at) -> None:
    with pytest.raises(ValueError, match="committed_at must be timezone-aware ISO text"):
        validate_study_design_provenance(_preregistered(committed_at=committed_at))


def test_prior_artifact_timestamp_must_carry_a_timezone() -> None:
    with pytest.raises(ValueError, match="committed_at must include a timezone"):
        validate_study_design_provenance(_preregistered(committed_at="2026-08-20T12:00:00"))


# -- the artifact ------------------------------------------------------------


@pytest.fixture
def payload() -> dict:
    if not ARTIFACT.exists():
        pytest.skip("vendor/ud_czech_case_counts.json absent")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _rejection(payload: dict, mutate, message: str) -> None:
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        CzechCaseArtifact(payload)


def test_artifact_rejects_a_non_object_raw_source(payload) -> None:
    _rejection(
        payload,
        lambda p: p["provenance"].__setitem__("raw_source", "pinned"),
        "lacks a raw_source provenance object",
    )


def test_artifact_rejects_a_nominal_alphabet_that_drifts_from_the_loader(payload) -> None:
    _rejection(
        payload,
        lambda p: p["provenance"].__setitem__("nominal_upos", ["NOUN"]),
        "nominal_upos provenance disagrees",
    )


@pytest.mark.parametrize("values", ["Nom,Gen", [1, 2]])
def test_artifact_rejects_case_values_that_are_not_a_string_list(payload, values) -> None:
    _rejection(
        payload,
        lambda p: p.__setitem__("ud_case_values", values),
        "ud_case_values must be a string list",
    )


def test_artifact_rejects_split_provenance_with_the_wrong_arms(payload) -> None:
    _rejection(
        payload,
        lambda p: p["provenance"]["splits"].pop("test"),
        "provenance.splits must contain exactly train and test",
    )


def test_artifact_rejects_documents_with_the_wrong_arms(payload) -> None:
    _rejection(
        payload,
        lambda p: p["documents"].pop("test"),
        "documents must contain exactly train and test",
    )


def test_artifact_rejects_an_empty_population_split(payload) -> None:
    _rejection(
        payload,
        lambda p: p["populations"].__setitem__("train", []),
        "train split must be a non-empty sequence list",
    )


def test_artifact_rejects_a_document_identity_no_sequence_uses(payload) -> None:
    def alias_every_sequence_to_the_first_document(p: dict) -> None:
        mapping = p["documents"]["test"]["sequence_document_indices"]
        p["documents"]["test"]["sequence_document_indices"] = [0] * len(mapping)

    _rejection(
        payload,
        alias_every_sequence_to_the_first_document,
        "contains an unused or absent document identity",
    )


def test_artifact_rejects_document_ids_out_of_first_use_order(payload) -> None:
    def reverse_first_use(p: dict) -> None:
        mapping = p["documents"]["test"]["sequence_document_indices"]
        p["documents"]["test"]["sequence_document_indices"] = mapping[::-1]

    _rejection(payload, reverse_first_use, "ids must follow first source use")


def test_artifact_rejects_an_observation_alphabet_wider_than_its_populations(payload) -> None:
    nominal = {"ADJ", "DET", "NOUN", "NUM", "PRON", "PROPN"}
    unused = sorted(nominal - set(payload["upos_tags"]))
    if not unused:
        # The shipped artifact observes every nominal tag, so no alphabet wider
        # than its populations can be built from it: state that precondition
        # instead of skipping, so a later artifact that stops observing a tag
        # re-arms the planted-tag control below.
        observed = {
            upos
            for split in ("train", "test")
            for sequence in getattr(CzechCaseArtifact(payload), split)
            for upos, _case in sequence
        }
        assert set(payload["upos_tags"]) == nominal == observed
        return

    def widen(p: dict) -> None:
        p["upos_tags"] = sorted([*p["upos_tags"], unused[0]])

    _rejection(payload, widen, "upos_tags must exactly equal the observations")


def test_loader_reports_an_absent_artifact(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="generate_czech_case_counts"):
        load_czech_case_artifact(tmp_path / "missing.json")


# -- pinned source shards ----------------------------------------------------


def test_shard_check_refuses_an_unknown_split() -> None:
    with pytest.raises(UDCzechSourceIntegrityError, match="unknown Czech artifact split 'dev'"):
        verify_ud_czech_shard_bytes(b"", "dev", filename="cs_pdtc-ud-dev.conllu")


def test_shard_check_refuses_the_wrong_filename() -> None:
    with pytest.raises(UDCzechSourceIntegrityError, match="wrong filename"):
        verify_ud_czech_shard_bytes(b"", "train", filename="cs_pdtc-ud-train.conllu")


def test_source_metadata_must_be_complete() -> None:
    with pytest.raises(UDCzechSourceIntegrityError, match="lacks pinned source metadata"):
        validate_ud_czech_source_metadata({})


def _staged_shards_match_the_pin() -> bool:
    for shard in UD_CZECH_PDTC_SHARDS.values():
        path = STAGING / shard.filename
        if not path.is_file():
            return False
        if hashlib.sha256(path.read_bytes()).hexdigest() != shard.sha256:
            return False
    return True


def test_staged_shards_are_read_and_resolved_only_when_byte_identical() -> None:
    if not _staged_shards_match_the_pin():
        pytest.skip("the pinned Czech shards are not staged on this machine")
    train = UD_CZECH_PDTC_SHARDS["train"]
    path = STAGING / train.filename
    content = read_verified_ud_czech_shard(path, "train")
    assert verify_ud_czech_shard_bytes(content, "train", filename=train.filename) == train.sha256
    assert verify_ud_czech_shard(path, "train") == path.resolve()
    resolved = verified_ud_czech_sources(STAGING)
    assert set(resolved) == set(UD_CZECH_PDTC_SHARDS)
    assert all(resolved[split].is_file() for split in resolved)
