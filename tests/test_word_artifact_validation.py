"""Rejection contracts of the Stage 3 word artifact and its held-out stream.

Every check below is a distinct way a malformed or relabeled artifact could
otherwise reach a published Stage 3 number. The payloads are either the
minimal valid fixture the Stage 3 gate already uses, a four-document variant
built here so that a split with more than one held-out document exists, or
the real vendored held-out stream with one field planted.

Each test plants exactly one defect and matches the exact message the loader
raises for it, so a rejection for the wrong reason cannot read as a pass.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ailm.corpus.brown import (
    brown_document_identity_sha256,
    brown_partition_input_sha256,
    category_stratified_document_indices,
)
from ailm.levels.word import (
    Stage3Miswiring,
    WordArtifact,
    load_word_artifact,
    validate_heldout_stream_artifact,
    validate_stage3_miswirings,
)
from tests.test_stage3_gate import _minimal_word_payload

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAM = PROJECT_ROOT / "vendor" / "brown_heldout_stream.json"


def _four_document_payload() -> dict:
    """A valid payload whose canonical split holds two held-out documents."""
    document_ids = ["doc-a", "doc-b", "doc-c", "doc-d"]
    categories = ["fiction", "fiction", "news", "news"]
    train, heldout = category_stratified_document_indices(document_ids, categories, 0.9)
    assert len(train) == 2 and len(heldout) == 2
    counts = [45 if index in train else 5 for index in range(len(document_ids))]
    payload = _minimal_word_payload()
    payload["provenance"].update(
        {
            "train_document_fraction": 0.5,
            "documents": 4,
            "train_documents": 2,
            "heldout_documents": 2,
            "categories": 2,
            "document_identity_sha256": brown_document_identity_sha256(document_ids),
            "document_partition_input_sha256": brown_partition_input_sha256(
                document_ids, categories
            ),
        }
    )
    payload["documents"] = {
        "ids": document_ids,
        "categories": categories,
        "alphabetic_token_counts": counts,
        "split_indices": {"train": list(train), "heldout": list(heldout)},
        "eval_token_document_indices": [heldout[0], heldout[1]],
    }
    return payload


def _custom_lexicon_payload() -> dict:
    payload = _minimal_word_payload()
    payload["provenance"]["lexicon_source"] = "custom:lexicon.txt"
    payload["provenance"]["lexicon_sha256"] = "a" * 64
    payload["provenance"]["lexicon_frequency_total"] = 90
    return payload


# -- the fixtures themselves must be accepted, or every rejection below is vacuous


def test_minimal_and_four_document_payloads_are_accepted() -> None:
    minimal = WordArtifact(_minimal_word_payload())
    assert minimal.observed_max_word_len == 3
    four = WordArtifact(_four_document_payload())
    assert len(four.heldout_document_indices) == 2
    custom = WordArtifact(_custom_lexicon_payload())
    assert custom.provenance["lexicon_source"] == "custom:lexicon.txt"


# -- top-level shape and cross-table contracts --------------------------------


def test_rejects_a_non_object_payload() -> None:
    with pytest.raises(ValueError, match="word artifact must be a JSON object"):
        WordArtifact([])


def test_rejects_a_non_array_evaluation_stream() -> None:
    payload = _minimal_word_payload()
    payload["eval_tokens"] = "cat sat"
    with pytest.raises(ValueError, match="eval_tokens must be a JSON array"):
        WordArtifact(payload)


def test_rejects_a_vocabulary_word_absent_from_the_bar_lexicon() -> None:
    payload = _minimal_word_payload()
    payload["bar_train_freq"] = {"cat": 60}
    with pytest.raises(ValueError, match="word_freq is not a subset of bar_train_freq"):
        WordArtifact(payload)


def test_rejects_a_vocabulary_word_longer_than_the_declared_horizon() -> None:
    payload = _minimal_word_payload()
    payload["provenance"]["max_word_len"] = 2
    with pytest.raises(ValueError, match="exceeds provenance max_word_len"):
        WordArtifact(payload)


def test_rejects_a_vocabulary_that_is_not_the_frequency_ranked_prefix() -> None:
    payload = _minimal_word_payload()
    payload["bar_train_freq"] = {"cat": 60, "sat": 30, "on": 10}
    payload["word_freq"] = {"cat": 60, "on": 10}
    payload["eval_tokens"] = ["cat", "on"]
    payload["provenance"]["train_types"] = 3
    payload["provenance"]["train_tokens"] = 100
    payload["provenance"]["total_alphabetic_tokens"] = 110
    payload["provenance"]["train_token_fraction"] = 100 / 110
    payload["documents"]["alphabetic_token_counts"] = [100, 10]
    with pytest.raises(ValueError, match="frequency-ranked eligible prefix"):
        WordArtifact(payload)


def test_rejects_a_bar_lexicon_whose_mass_disagrees_with_train_tokens() -> None:
    payload = _minimal_word_payload()
    payload["word_freq"] = {"cat": 61, "sat": 30}
    payload["bar_train_freq"] = {"cat": 61, "sat": 30}
    with pytest.raises(ValueError, match="bar_train_freq sums to 91 tokens"):
        WordArtifact(payload)


def test_rejects_a_custom_lexicon_whose_mass_disagrees_with_its_total() -> None:
    payload = _custom_lexicon_payload()
    payload["provenance"]["lexicon_frequency_total"] = 91
    with pytest.raises(ValueError, match="custom-lexicon counts"):
        WordArtifact(payload)


@pytest.mark.parametrize("bound", [None, "unbounded"])
def test_accepts_an_absent_or_unbounded_vocabulary_bound(bound) -> None:
    payload = _minimal_word_payload()
    payload["provenance"]["vocab_bound"] = bound
    assert len(WordArtifact(payload).vocab) == 2


@pytest.mark.parametrize("bound", [0, -1, True, 1.5, "2"])
def test_rejects_a_malformed_vocabulary_bound(bound) -> None:
    payload = _minimal_word_payload()
    payload["provenance"]["vocab_bound"] = bound
    with pytest.raises(ValueError, match="vocab_bound must be a positive integer"):
        WordArtifact(payload)


def test_rejects_a_vocabulary_above_its_bound() -> None:
    payload = _minimal_word_payload()
    payload["provenance"]["vocab_bound"] = 1
    with pytest.raises(ValueError, match="types above vocab_bound=1"):
        WordArtifact(payload)


def test_rejects_a_non_object_frequency_map() -> None:
    payload = _minimal_word_payload()
    payload["word_freq"] = []
    with pytest.raises(ValueError, match="word_freq must be a JSON object"):
        WordArtifact(payload)


# -- documents block ---------------------------------------------------------


def _documents_rejection(mutate, message: str, *, four: bool = False) -> None:
    payload = _four_document_payload() if four else _minimal_word_payload()
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        WordArtifact(payload)


def test_rejects_a_non_object_documents_block() -> None:
    _documents_rejection(lambda p: p.__setitem__("documents", []), "documents must be a JSON")


def test_rejects_a_documents_block_with_extra_or_missing_keys() -> None:
    _documents_rejection(
        lambda p: p["documents"].__setitem__("stray", 1), "documents fields do not match"
    )


def test_rejects_empty_document_identities() -> None:
    _documents_rejection(
        lambda p: p["documents"].__setitem__("ids", []), "documents.ids must be a non-empty"
    )


def test_rejects_misaligned_categories() -> None:
    _documents_rejection(
        lambda p: p["documents"].__setitem__("categories", ["fiction"]),
        "categories must align one-for-one",
    )


@pytest.mark.parametrize("category", ["", 3, "fic\0tion"])
def test_rejects_an_invalid_category_value(category) -> None:
    _documents_rejection(
        lambda p: p["documents"]["categories"].__setitem__(0, category),
        r"documents.categories\[0\] is invalid",
    )


def test_rejects_misaligned_token_counts() -> None:
    _documents_rejection(
        lambda p: p["documents"].__setitem__("alphabetic_token_counts", [100]),
        "alphabetic_token_counts must align",
    )


@pytest.mark.parametrize("count", [-1, True, 2.5])
def test_rejects_an_invalid_token_count(count) -> None:
    _documents_rejection(
        lambda p: p["documents"]["alphabetic_token_counts"].__setitem__(0, count),
        r"alphabetic_token_counts\[0\] must be a non-negative integer",
    )


def test_rejects_split_indices_with_unexpected_arms() -> None:
    _documents_rejection(
        lambda p: p["documents"]["split_indices"].__setitem__("validation", []),
        "must contain exactly train and heldout",
    )


def test_rejects_an_empty_split_arm() -> None:
    _documents_rejection(
        lambda p: p["documents"]["split_indices"].__setitem__("heldout", []),
        "split_indices.heldout must be a non-empty array",
    )


@pytest.mark.parametrize("index", [5, -1, True, "1"])
def test_rejects_an_out_of_range_split_index(index) -> None:
    _documents_rejection(
        lambda p: p["documents"]["split_indices"].__setitem__("heldout", [index]),
        "split_indices.heldout contains an invalid index",
    )


def test_rejects_an_unsorted_or_repeated_split_arm() -> None:
    def unsort(payload: dict) -> None:
        heldout = payload["documents"]["split_indices"]["heldout"]
        payload["documents"]["split_indices"]["heldout"] = heldout[::-1]

    _documents_rejection(unsort, "must be unique and corpus-ordered", four=True)


def test_rejects_a_non_exhaustive_document_split() -> None:
    def drop(payload: dict) -> None:
        split = payload["documents"]["split_indices"]
        split["train"] = split["train"][:1]
        split["heldout"] = split["heldout"][:1]

    _documents_rejection(drop, "document split is not exhaustive", four=True)


def test_rejects_provenance_that_disagrees_with_the_documents_block() -> None:
    _documents_rejection(
        lambda p: p["provenance"].__setitem__("categories", 2),
        "provenance categories=2 disagrees with documents metadata",
    )


def test_rejects_misaligned_evaluation_document_indices() -> None:
    _documents_rejection(
        lambda p: p["documents"].__setitem__("eval_token_document_indices", [1]),
        "eval_token_document_indices must align one-for-one with eval_tokens",
    )


def test_rejects_evaluation_document_indices_out_of_corpus_order() -> None:
    def reverse(payload: dict) -> None:
        indices = payload["documents"]["eval_token_document_indices"]
        payload["documents"]["eval_token_document_indices"] = indices[::-1]

    _documents_rejection(reverse, "must preserve corpus document order", four=True)


# -- provenance block --------------------------------------------------------


def test_provenance_validator_rejects_a_non_object() -> None:
    with pytest.raises(ValueError, match="provenance must be a JSON object"):
        WordArtifact._validate_provenance([])


def _provenance_rejection(mutate, message: str, *, custom: bool = False) -> None:
    payload = _custom_lexicon_payload() if custom else _minimal_word_payload()
    mutate(payload["provenance"])
    with pytest.raises(ValueError, match=message):
        WordArtifact(payload)


def test_rejects_brown_provenance_carrying_custom_only_fields() -> None:
    _provenance_rejection(
        lambda p: p.__setitem__("lexicon_sha256", "a" * 64),
        "Brown provenance carries custom-lexicon-only fields",
    )


@pytest.mark.parametrize("source", ["custom:", "custom:a/b", "custom:a\\b", 5, "other"])
def test_rejects_an_unrecognized_lexicon_source(source) -> None:
    _provenance_rejection(
        lambda p: p.__setitem__("lexicon_source", source),
        "lexicon_source must be 'brown-frequency-ranked'",
    )


def test_rejects_a_custom_lexicon_without_its_digest() -> None:
    _provenance_rejection(
        lambda p: p.pop("lexicon_sha256"),
        "custom lexicon provenance is missing",
        custom=True,
    )


@pytest.mark.parametrize("total", [0, -3, True, "90"])
def test_rejects_an_invalid_custom_lexicon_total(total) -> None:
    _provenance_rejection(
        lambda p: p.__setitem__("lexicon_frequency_total", total),
        "lexicon_frequency_total must be a positive integer",
        custom=True,
    )


@pytest.mark.parametrize("digest", ["xyz", "A" * 64, 64, "0" * 63])
def test_rejects_a_malformed_document_digest(digest) -> None:
    _provenance_rejection(
        lambda p: p.__setitem__("document_identity_sha256", digest),
        "document_identity_sha256 must be a lowercase SHA-256 hex digest",
    )


@pytest.mark.parametrize("field", ["vocab_size", "max_word_len", "documents", "categories"])
def test_rejects_a_non_positive_cardinality(field) -> None:
    _provenance_rejection(
        lambda p: p.__setitem__(field, 0), f"provenance {field} must be a positive integer"
    )


@pytest.mark.parametrize("dropped", [-1, True, "0"])
def test_rejects_an_invalid_dropped_type_count(dropped) -> None:
    _provenance_rejection(
        lambda p: p.__setitem__("types_dropped_for_length", dropped),
        "types_dropped_for_length must be a non-negative",
    )


def test_rejects_document_cardinalities_that_do_not_sum() -> None:
    _provenance_rejection(
        lambda p: p.__setitem__("documents", 3),
        "train_documents \\+ heldout_documents != documents",
    )


# -- the Stage 3 miswiring registry ------------------------------------------


def _arm(key: str, label: str | None) -> Stage3Miswiring:
    return Stage3Miswiring(key=key, display_name=key, claim_label=label)


def test_the_shipped_miswiring_registry_is_accepted_and_names_its_analytic_arm() -> None:
    arms = validate_stage3_miswirings()
    assert [arm.key for arm in arms if not arm.has_finite_score] == ["early_end_of_word"]


def test_miswiring_registry_rejects_an_empty_registry() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_stage3_miswirings(())


def test_miswiring_registry_rejects_duplicate_claim_labels() -> None:
    arms = (
        _arm("one", "stage3_miswire_shared"),
        _arm("two", "stage3_miswire_shared"),
        _arm("infinite", None),
    )
    with pytest.raises(ValueError, match="duplicate claim labels"):
        validate_stage3_miswirings(arms)


def test_miswiring_registry_rejects_a_label_outside_its_family() -> None:
    arms = (_arm("one", "stage4_miswire_one"), _arm("infinite", None))
    with pytest.raises(ValueError, match="wrong family"):
        validate_stage3_miswirings(arms)


# -- the held-out stream, planted one field at a time -------------------------


@pytest.fixture(scope="module")
def word_artifact() -> WordArtifact:
    return load_word_artifact()


@pytest.fixture(scope="module")
def stream() -> dict:
    if not STREAM.exists():
        pytest.skip("vendor/brown_heldout_stream.json absent")
    return json.loads(STREAM.read_text(encoding="utf-8"))


def _stream_rejection(stream: dict, word_artifact: WordArtifact, mutate, message: str) -> None:
    planted = copy.deepcopy(stream)
    mutate(planted)
    with pytest.raises(ValueError, match=message):
        validate_heldout_stream_artifact(planted, word_artifact)


def test_stream_rejects_a_non_object_payload(word_artifact) -> None:
    with pytest.raises(ValueError, match="held-out stream must be a JSON object"):
        validate_heldout_stream_artifact([], word_artifact)


def test_stream_rejects_unexpected_top_level_fields(stream, word_artifact) -> None:
    _stream_rejection(
        stream, word_artifact, lambda p: p.__setitem__("stray", 1), "fields do not match the v3"
    )


def test_stream_rejects_a_foreign_schema(stream, word_artifact) -> None:
    _stream_rejection(
        stream,
        word_artifact,
        lambda p: p.__setitem__("schema", "ailm-heldout-stream-v1"),
        "unexpected held-out stream schema",
    )


def test_stream_rejects_unexpected_provenance_fields(stream, word_artifact) -> None:
    _stream_rejection(
        stream,
        word_artifact,
        lambda p: p["provenance"].__setitem__("stray", 1),
        "provenance fields do not match the v3",
    )


def test_stream_rejects_a_relabeled_corpus(stream, word_artifact) -> None:
    _stream_rejection(
        stream,
        word_artifact,
        lambda p: p["provenance"].__setitem__("corpus", "mutable Brown"),
        "provenance corpus must be",
    )


def test_stream_rejects_a_length_bound_applied_to_the_denominator(stream, word_artifact) -> None:
    _stream_rejection(
        stream,
        word_artifact,
        lambda p: p["provenance"].__setitem__("max_word_len_applied", True),
        "must not apply the word-artifact length bound",
    )


def test_stream_rejects_a_horizon_that_disagrees_with_the_word_artifact(
    stream, word_artifact
) -> None:
    _stream_rejection(
        stream,
        word_artifact,
        lambda p: p["provenance"].__setitem__("max_word_len_in_word_artifact", 99),
        "disagree on max_word_len",
    )


def test_stream_rejects_a_non_object_documents_block(stream, word_artifact) -> None:
    _stream_rejection(
        stream,
        word_artifact,
        lambda p: p.__setitem__("documents", []),
        "documents must be a JSON object",
    )


def test_stream_rejects_unexpected_documents_fields(stream, word_artifact) -> None:
    _stream_rejection(
        stream,
        word_artifact,
        lambda p: p["documents"].__setitem__("stray", 1),
        "documents fields do not match the v3",
    )


def test_stream_rejects_an_empty_token_stream(stream, word_artifact) -> None:
    _stream_rejection(
        stream,
        word_artifact,
        lambda p: p.__setitem__("held_tokens", []),
        "held_tokens must be a non-empty",
    )


@pytest.mark.parametrize("token", ["Cat", "", "café", 7])
def test_stream_rejects_a_token_outside_the_alphabet(stream, word_artifact, token) -> None:
    _stream_rejection(
        stream,
        word_artifact,
        lambda p: p["held_tokens"].__setitem__(0, token),
        r"held_tokens\[0\] is not a lowercase ASCII-letter token",
    )


def test_stream_rejects_a_token_count_drift(stream, word_artifact) -> None:
    _stream_rejection(
        stream,
        word_artifact,
        lambda p: p["held_tokens"].pop(),
        "token count disagrees with provenance",
    )


def test_stream_rejects_a_type_count_drift(stream, word_artifact) -> None:
    def merge_a_singleton_into_the_commonest_type(payload: dict) -> None:
        tokens = payload["held_tokens"]
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        singleton = next(token for token in tokens if counts[token] == 1)
        commonest = max(counts, key=lambda token: (counts[token], token))
        tokens[tokens.index(singleton)] = commonest

    _stream_rejection(
        stream,
        word_artifact,
        merge_a_singleton_into_the_commonest_type,
        "type count disagrees with provenance",
    )


def test_stream_rejects_misaligned_document_indices(stream, word_artifact) -> None:
    _stream_rejection(
        stream,
        word_artifact,
        lambda p: p["documents"]["held_token_document_indices"].pop(),
        "must align one-for-one with held_tokens",
    )


def test_stream_rejects_document_indices_out_of_corpus_order(stream, word_artifact) -> None:
    def swap_across_a_document_boundary(payload: dict) -> None:
        indices = payload["documents"]["held_token_document_indices"]
        boundary = next(i for i in range(len(indices) - 1) if indices[i] < indices[i + 1])
        indices[boundary], indices[boundary + 1] = indices[boundary + 1], indices[boundary]

    _stream_rejection(
        stream,
        word_artifact,
        swap_across_a_document_boundary,
        "must preserve corpus document order",
    )
