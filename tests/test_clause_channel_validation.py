"""Rejection contracts of the clause-word bridge and its clustered summary.

The bridge binds Stage 5's sentence rows to Stage 3's receiver vocabulary and
document partition. Each test below plants one defect in the real vendored
bridge and matches the exact message the loader raises for it, so the artifact
cannot be relabeled, re-partitioned, or re-typed without the loader noticing.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ailm.levels.clause_channel import (
    CLAUSE_TYPES,
    ClauseWordArtifact,
    build_clause_message,
    clustered_clause_word_summaries,
    load_clause_word_artifact,
    message_for_type,
)
from ailm.levels.word import WordArtifact, load_word_artifact

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRIDGE = PROJECT_ROOT / "vendor" / "brown_clause_word_counts.json"
WORD_ARTIFACT = PROJECT_ROOT / "vendor" / "brown_word_counts.json"


@pytest.fixture(scope="module")
def word_artifact() -> WordArtifact:
    return load_word_artifact(WORD_ARTIFACT)


@pytest.fixture(scope="module")
def bridge_payload() -> dict:
    if not BRIDGE.exists():
        pytest.skip("vendor/brown_clause_word_counts.json absent")
    return json.loads(BRIDGE.read_text(encoding="utf-8"))


def _validate(
    payload: dict,
    word_artifact: WordArtifact,
    *,
    train_indices: list[int] | None = None,
    heldout_indices: list[int] | None = None,
) -> ClauseWordArtifact:
    return ClauseWordArtifact(
        payload,
        expected_vocab=word_artifact.vocab,
        expected_vocab_sha256=hashlib.sha256(WORD_ARTIFACT.read_bytes()).hexdigest(),
        expected_document_ids=word_artifact.document_ids,
        expected_categories=word_artifact.document_categories,
        expected_train_indices=(
            word_artifact.train_document_indices if train_indices is None else train_indices
        ),
        expected_heldout_indices=(
            word_artifact.heldout_document_indices if heldout_indices is None else heldout_indices
        ),
    )


def test_the_shipped_bridge_is_accepted(bridge_payload, word_artifact) -> None:
    bridge = _validate(bridge_payload, word_artifact)
    assert bridge.document_ids == word_artifact.document_ids


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.__setitem__("schema", "ailm-clause-word-counts-v0"), "schema must be"),
        (lambda p: p["provenance"].__setitem__("stray", 1), "provenance keys disagree"),
        (
            lambda p: p["provenance"].__setitem__("sentence_types", list(CLAUSE_TYPES)[::-1]),
            "sentence_types disagree",
        ),
        (
            lambda p: p["provenance"].__setitem__("vocab_source_schema", "ailm-word-counts-v1"),
            "vocabulary source schema",
        ),
        (lambda p: p.__setitem__("vocab", p["vocab"][::-1]), "Stage 3 receiver vocabulary"),
        (
            lambda p: p["provenance"].__setitem__("vocab_size", p["provenance"]["vocab_size"] + 1),
            "vocab_size disagrees with vocab",
        ),
        (lambda p: p["documents"].__setitem__("stray", 1), "document keys disagree"),
        (
            lambda p: p["documents"].__setitem__("ids", p["documents"]["ids"][::-1]),
            "identities/categories disagree",
        ),
        (
            lambda p: p["documents"].__setitem__("identity_sha256", "0" * 64),
            "identity digest is inconsistent",
        ),
        (
            lambda p: p["documents"].__setitem__("partition_input_sha256", "0" * 64),
            "partition-input digest is inconsistent",
        ),
        (
            lambda p: p["documents"]["split_indices"].__setitem__("validation", []),
            "split_indices must contain exactly",
        ),
        (
            lambda p: p["documents"]["split_indices"].__setitem__("train", "0,1,2"),
            "split indices must be lists",
        ),
        (
            lambda p: p["documents"].__setitem__(
                "typed_sentence_counts", p["documents"]["typed_sentence_counts"][:-1]
            ),
            "per-document counts must align",
        ),
        (lambda p: p["splits"].__setitem__("validation", []), "splits must contain exactly"),
        (lambda p: p["splits"].__setitem__("train", []), "train split must be a non-empty"),
        (
            lambda p: p["splits"]["train"][0].__setitem__("stray", 1),
            "sentence rows must match the exact schema",
        ),
        (
            lambda p: p["splits"]["train"][0].__setitem__("type", "exclamative"),
            "unknown type",
        ),
        (
            lambda p: p["provenance"].__setitem__(
                "typed_sentences", p["provenance"]["typed_sentences"] + 1
            ),
            "provenance typed_sentences disagrees",
        ),
        (
            lambda p: p["provenance"].__setitem__("train_document_fraction", 0.5),
            "train_document_fraction disagrees with document counts",
        ),
    ],
    ids=(
        "schema",
        "provenance-key",
        "type-order",
        "vocab-schema",
        "vocab-order",
        "vocab-size",
        "documents-key",
        "document-order",
        "identity-digest",
        "partition-digest",
        "split-arm",
        "split-type",
        "count-alignment",
        "splits-key",
        "empty-split",
        "row-schema",
        "row-type",
        "provenance-count",
        "document-fraction",
    ),
)
def test_bridge_rejects_each_planted_defect(bridge_payload, word_artifact, mutate, message) -> None:
    planted = copy.deepcopy(bridge_payload)
    mutate(planted)
    with pytest.raises(ValueError, match=message):
        _validate(planted, word_artifact)


def test_bridge_rejects_a_split_the_receiver_accepts_but_the_rule_does_not(
    bridge_payload, word_artifact
) -> None:
    """The canonical rule is checked even when the receiver already agrees."""
    planted = copy.deepcopy(bridge_payload)
    train = planted["documents"]["split_indices"]["train"]
    heldout = planted["documents"]["split_indices"]["heldout"]
    train[0], heldout[0] = heldout[0], train[0]
    train.sort()
    heldout.sort()
    with pytest.raises(ValueError, match="disagrees with the canonical rule"):
        _validate(planted, word_artifact, train_indices=train, heldout_indices=heldout)


def test_bridge_rejects_a_split_missing_a_sentence_type(bridge_payload, word_artifact) -> None:
    planted = copy.deepcopy(bridge_payload)
    dropped = planted["splits"]["test"][0]["type"]
    planted["splits"]["test"] = [row for row in planted["splits"]["test"] if row["type"] != dropped]
    with pytest.raises(ValueError, match="test split must represent every sentence type"):
        _validate(planted, word_artifact)


# -- loader ------------------------------------------------------------------


def test_loader_reports_an_absent_bridge(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="generate_clause_word_counts"):
        load_clause_word_artifact(tmp_path / "missing.json")


def test_loader_reports_an_absent_word_artifact(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="generate_word_counts"):
        load_clause_word_artifact(BRIDGE, word_path=tmp_path / "missing.json")


def test_loader_rejects_a_bridge_that_is_not_an_object(tmp_path) -> None:
    planted = tmp_path / "bridge.json"
    planted.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_clause_word_artifact(planted, word_path=WORD_ARTIFACT)


# -- message estimation ------------------------------------------------------


_ROWS = [
    {"type": CLAUSE_TYPES[0], "words": ["cat", "sat"]},
    {"type": CLAUSE_TYPES[1], "words": ["sat"]},
]
_INDEX = {"cat": 0, "sat": 1, "mat": 2}


def test_message_rejects_a_nonpositive_pseudocount() -> None:
    with pytest.raises(ValueError, match="smoothing must be positive"):
        build_clause_message(_ROWS, _INDEX, smoothing=0.0)


def test_message_for_type_reads_the_typed_row_or_the_fallback() -> None:
    message = build_clause_message(_ROWS, _INDEX)
    typed = message_for_type(message, CLAUSE_TYPES[0])
    assert np.isclose(typed.sum(), 1.0)
    assert typed[0] > typed[2]
    fallback = message_for_type(message, "never-seen")
    assert np.array_equal(fallback, message.fallback)


# -- clustered summaries -----------------------------------------------------


def _sentences() -> list[dict]:
    return [
        {"document_index": 0, "words": ["cat", "sat"]},
        {"document_index": 1, "words": ["mat"]},
    ]


def test_clustered_summary_rejects_misaligned_streams() -> None:
    with pytest.raises(ValueError, match="streams must align"):
        clustered_clause_word_summaries(_sentences(), [0.0, 0.0, 0.0], [0.0, 0.0])


def test_clustered_summary_rejects_a_row_without_a_document_index() -> None:
    rows = _sentences()
    rows[0]["document_index"] = True
    with pytest.raises(ValueError, match="need a document index and word list"):
        clustered_clause_word_summaries(rows, [0.0] * 3, [0.0] * 3)


def test_clustered_summary_rejects_a_non_string_word() -> None:
    rows = _sentences()
    rows[0]["words"][0] = 7
    with pytest.raises(ValueError, match="non-string word"):
        clustered_clause_word_summaries(rows, [0.0] * 3, [0.0] * 3)


def test_clustered_summary_rejects_more_tokens_than_log_probabilities() -> None:
    with pytest.raises(ValueError, match="more tokens than the log-probability stream"):
        clustered_clause_word_summaries(_sentences(), [0.0, 0.0], [0.0, 0.0])


def test_clustered_summary_rejects_more_log_probabilities_than_tokens() -> None:
    with pytest.raises(ValueError, match="tokens absent from the sentence rows"):
        clustered_clause_word_summaries(_sentences(), [0.0] * 4, [0.0] * 4)


def test_clustered_summary_needs_two_source_documents() -> None:
    rows = [{"document_index": 0, "words": ["cat", "sat", "mat"]}]
    with pytest.raises(ValueError, match="at least two source documents"):
        clustered_clause_word_summaries(rows, [0.0] * 3, [0.0] * 3)
