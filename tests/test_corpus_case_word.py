"""Loader tests for the case-word (L4→L3) bridge artifact.

Every refusal here is a structural guard on the account protocol P3 froze:
the observation contract, the authenticated source identity, the stride
split, the declared label routes, the vocabulary intersection, and the
content digests. The payload factory builds a small valid artifact; each
planted mutation must fail the loader for its own reason.
"""

from __future__ import annotations

import copy
import json

import pytest

from ailm.corpus.case_word import (
    CASE_WORD_ARTIFACT_SCHEMA,
    CASE_WORD_OBSERVATION,
    CaseWordArtifact,
    load_case_word_artifact,
)
from ailm.corpus.ud import UD_EWT_RAW_SHA256, document_identity_sha256

VOCAB = ["the", "dog", "cat", "sed"]


def _payload() -> dict:
    """A minimal valid artifact: two documents under the stride rule.

    Document 0 is TEST (0 % 5 == 0), document 1 is TRAIN. The train document
    carries labeled and unlabeled rows; the sender map counts only
    in-vocabulary lowercase forms from train rows.
    """
    train_rows = [
        ["the", "DET", None, None],
        ["dog", "NOUN", "NOM", "deprel"],
        ["Dog", "NOUN", "ACC", "morphology"],
        ["sed", "VERB", "GEN", "deprel"],
        ["zebra", "NOUN", "VOC", "deprel"],
    ]
    test_rows = [
        ["the", "DET", None, None],
        ["cat", "NOUN", "NOM", "deprel"],
    ]
    payload = {
        "schema": CASE_WORD_ARTIFACT_SCHEMA,
        "provenance": {
            "corpus": "UD English EWT dev (en_ewt-ud-dev.conllu)",
            "license": "CC BY-SA 4.0",
            "source_sha256": UD_EWT_RAW_SHA256,
            "observation": CASE_WORD_OBSERVATION,
            "test_every": 5,
            "documents": 2,
            "train_documents": 1,
            "test_documents": 1,
            "document_identity_sha256": document_identity_sha256(
                ["case-word-fixture-0", "case-word-fixture-1"]
            ),
            "vocab_size": len(VOCAB),
            "dropped_oov_train_words": 1,
        },
        "documents": {
            "ids": ["case-word-fixture-0", "case-word-fixture-1"],
            "split_indices": {"train": [1], "test": [0]},
        },
        "vocab": list(VOCAB),
        "populations": {
            "train": [
                {"document_index": 1, "document_id": "case-word-fixture-1", "rows": train_rows}
            ],
            "test": [
                {"document_index": 0, "document_id": "case-word-fixture-0", "rows": test_rows}
            ],
        },
        "sender_map": {"NOM": {"dog": 1}, "ACC": {"cat": 2}, "GEN": {"sed": 1}},
        "route_stats": {
            route: {
                "selected_tokens": 0,
                "fired_tokens": 0,
                "agreements_with_selected": 0,
                "disagreements_with_selected": 0,
            }
            for route in ("morphology", "deprel", "adposition_token")
        },
    }
    payload["route_stats"]["morphology"]["selected_tokens"] = 1
    payload["route_stats"]["deprel"]["selected_tokens"] = 2
    payload["route_stats"]["deprel"]["fired_tokens"] = 3
    payload["route_stats"]["deprel"]["agreements_with_selected"] = 2
    return payload


def _digests(payload: dict) -> dict:
    return {
        name: __import__("hashlib")
        .sha256(json.dumps(fragment, sort_keys=True, ensure_ascii=True).encode("utf-8"))
        .hexdigest()
        for name, fragment in (
            ("populations", payload["populations"]),
            ("sender_map", payload["sender_map"]),
            ("vocab", payload["vocab"]),
        )
    }


def _valid() -> CaseWordArtifact:
    payload = _payload()
    payload["digests"] = _digests(payload)
    return CaseWordArtifact(copy.deepcopy(payload))


def test_the_valid_fixture_loads_and_exposes_the_declared_surfaces():
    artifact = _valid()
    assert artifact.vocab == VOCAB
    assert artifact.document_ids == ["case-word-fixture-0", "case-word-fixture-1"]
    assert [row.word_form for row in artifact.rows("train")] == [
        "the",
        "dog",
        "Dog",
        "sed",
        "zebra",
    ]
    assert artifact.sender_support == frozenset({"NOM", "ACC", "GEN"})


def test_the_observation_contract_is_enforced_verbatim():
    payload = _payload()
    payload["provenance"]["observation"] = "UPOS only; labels withheld"
    payload["digests"] = _digests(payload)
    with pytest.raises(ValueError, match="observation contract"):
        CaseWordArtifact(payload)


def test_an_overlap_or_partition_defect_in_the_splits_is_refused():
    """Planted split defects fire the loader's structural guards: duplicated
    or overlapping document indices break the partition check first."""
    payload = _payload()
    payload["documents"]["split_indices"] = {"train": [0, 1], "test": [1]}
    payload["digests"] = _digests(payload)
    with pytest.raises(ValueError, match="partition the documents"):
        CaseWordArtifact(payload)


def test_a_row_pairing_a_label_without_a_route_is_refused():
    payload = _payload()
    payload["populations"]["train"][0]["rows"][1] = ["dog", "NOUN", "NOM", None]
    payload["digests"] = _digests(payload)
    with pytest.raises(ValueError, match="route"):
        CaseWordArtifact(payload)


def test_an_unknown_oracle_route_is_refused():
    payload = _payload()
    payload["populations"]["train"][0]["rows"][1] = ["dog", "NOUN", "NOM", "head_transfer"]
    payload["digests"] = _digests(payload)
    with pytest.raises(ValueError, match="oracle route"):
        CaseWordArtifact(payload)


def test_a_non_canonical_upos_row_is_refused():
    payload = _payload()
    payload["populations"]["train"][0]["rows"][0] = ["the", "DET+X", None, None]
    payload["digests"] = _digests(payload)
    with pytest.raises(ValueError, match="non-canonical UPOS"):
        CaseWordArtifact(payload)


def test_a_sender_count_outside_the_vocabulary_is_refused():
    payload = _payload()
    payload["sender_map"]["NOM"]["bear"] = 1
    payload["digests"] = _digests(payload)
    with pytest.raises(ValueError, match="intersection"):
        CaseWordArtifact(payload)


def test_a_nonpositive_sender_count_is_refused():
    payload = _payload()
    payload["sender_map"]["NOM"]["dog"] = 0
    payload["digests"] = _digests(payload)
    with pytest.raises(ValueError, match="positive integer"):
        CaseWordArtifact(payload)


def test_an_uppercase_vocabulary_is_refused():
    payload = _payload()
    payload["vocab"] = ["The", "dog", "cat", "sed"]
    payload["digests"] = _digests(payload)
    with pytest.raises(ValueError, match="lowercase"):
        CaseWordArtifact(payload)


def test_a_tampered_content_digest_is_refused():
    payload = _payload()
    payload["digests"] = _digests(payload)
    payload["digests"]["sender_map"] = "0" * 64
    with pytest.raises(ValueError, match="sender_map digest"):
        CaseWordArtifact(payload)


def test_route_stats_missing_a_route_is_refused():
    payload = _payload()
    del payload["route_stats"]["adposition_token"]
    payload["digests"] = _digests(payload)
    with pytest.raises(ValueError, match="adposition_token"):
        CaseWordArtifact(payload)


def test_loader_refuses_an_absent_artifact_instead_of_falling_back(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_case_word_artifact(tmp_path / "absent.json")
