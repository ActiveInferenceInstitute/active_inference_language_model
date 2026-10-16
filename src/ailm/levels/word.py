"""Word-level active inference agent for AILM Stage 3.

A two-factor, single-modality pymdp generative model over a bounded lexicon:

  - Hidden factor 0: "word identity"  — ``N_VOCAB`` states
  - Hidden factor 1: "position in word" — ``MAX_WORD_LEN + 1`` states
  - Observation modality: "current letter" — 27 observations (a-z plus END)
  - A (1, 27, N_VOCAB, n_pos): deterministic. ``A[0, k, w, p] = 1`` iff letter
    ``p`` of word ``w`` is symbol ``k``, or ``k`` is END and ``p >= len(w)``.
    Declared with ``A_dependencies = [[0, 1]]`` — one modality, both factors.
  - B[0] (1, N_VOCAB, N_VOCAB, 1): the **identity**. A word does not change
    into another word part-way through being spelled.
  - B[1] (1, n_pos, n_pos, 1): the +1 shift, absorbing at the last position.
  - D[0]: word unigram frequencies from the training split. **This is where the
    corpus statistics live at this level** — not in A, not in B.
  - D[1]: a point mass on position 0.

Why not a trie state space
--------------------------
The design document specifies words as paths through a letter-phoneme trie. A
dense pymdp ``B`` over trie nodes is claim:stage3_trie_dense_gb GB at the G2P
vocabulary and roughly claim:stage3_trie_dense_full_tb TB at full CMUdict, and
no sparse path is reachable through pymdp's public ``Agent`` API. The precision
matters, because the obvious reading of that sentence is
false: pymdp's JAX backend *does* contain sparse contraction machinery, and the
transition contraction this level uses is already sparse-capable — ``factor_dot``
carries a ``JAXSparse`` overload dispatching to ``spm_dot_sparse``. What blocks
the path is the ``Agent`` API, which rejects sparse arrays at construction and
does not preserve sparsity across the filtering loop. Decision 58 records the
exact boundary and ``scripts/probe_pymdp_sparse.py`` reproduces it, including the
result that a sparse-``B`` agent under two named patches is bit-identical to the
dense one. The trie is not discarded: after
observing a prefix the posterior over factor 0 is supported exactly on the words
sharing that prefix, which *is* the trie node. The factorization is the same
model in cheaper coordinates, and ``docs/decisions.md`` entry 13 records the
measured comparison.

Where the statistics live
-------------------------
At Levels 1 and 2 the learned sequence dynamics live in ``B``. At Level 3 both
transitions are structural constants and the learned quantity is the **word
prior** ``D[0]``. That difference is the level's whole content: a lexicon with
frequencies. It also relocates the mis-wiring risk — a wrong ``D`` here is what a
wrong ``B`` was there, and it is *not* caught by beating the accuracy bar.

Perplexity
----------
``P(path of w) = D[0][w]`` exactly, because every other factor is deterministic.
An emission is a letter or END, so a token of length ``L`` costs ``L + 1``
emissions and the level's score is ``exp(-sum log D[0][w] / sum (L_w + 1))``.

**Not comparable with Level 1.** That perplexity is over 30 running-text symbols;
this is over 27 word-internal symbols. Different alphabets, different position
sets, different quantities.

Closed vocabulary
-----------------
The model assigns **no** probability to a word outside its lexicon — not a small
one. Evaluation is restricted to in-vocabulary held-out tokens and the coverage
is reported as ``claim:stage3_coverage_pct``; the functions here raise rather
than smooth, because a fallback would report a number for a model that is
undefined on the input.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from ailm.config import DEFAULT_LEVEL_CONFIG, LevelConfig
from ailm.corpus.brown import (
    BROWN_HELDOUT_ARTIFACT_SCHEMA,
    BROWN_WORD_ARTIFACT_SCHEMA,
    brown_document_identity_sha256,
    brown_partition_input_sha256,
    category_stratified_document_indices,
    validate_brown_artifact_provenance,
    validate_canonical_brown_document_metadata,
    validate_canonical_brown_partition_target,
)
from ailm.levels._stochastic import column_stochastic
from ailm.resources import resolve_vendored_artifact

if TYPE_CHECKING:  # pragma: no cover
    from pymdp.agent import Agent as PymdpAgent

__all__ = [
    "END_INDEX",
    "LETTERS",
    "N_OBS",
    "STAGE3_MISWIRINGS",
    "Stage3Miswiring",
    "WordArtifact",
    "build_emission",
    "build_letter_bigram_bar",
    "build_position_transition",
    "build_word_agent",
    "build_word_prior",
    "build_word_transition",
    "load_word_artifact",
    "posterior_word_support",
    "validate_heldout_stream_artifact",
    "validate_stage3_miswirings",
    "word_perplexity",
]

logger = logging.getLogger(__name__)

#: Letter alphabet. Index 0-25; END is index 26.
LETTERS = "abcdefghijklmnopqrstuvwxyz"

#: End-of-word observation index.
END_INDEX = 26

#: Observation alphabet size: 26 letters + END.
N_OBS = 27


@dataclass(frozen=True, slots=True)
class Stage3Miswiring:
    """One canonical Stage 3 structural negative control.

    ``claim_label`` is absent only when the construction has analytically
    infinite perplexity and therefore cannot be serialized in the finite JSON
    claims ledger.  Keeping that arm in this registry prevents every consumer
    from reconstructing the total as ``number of recorded labels + 1``.
    """

    key: str
    display_name: str
    claim_label: str | None

    @property
    def has_finite_score(self) -> bool:
        """Whether the arm owns a finite, recorded perplexity claim."""
        return self.claim_label is not None


STAGE3_MISWIRINGS: tuple[Stage3Miswiring, ...] = (
    Stage3Miswiring(
        key="uniform_prior",
        display_name="uniform prior\n(frequencies dropped)",
        claim_label="stage3_miswire_uniform_prior",
    ),
    Stage3Miswiring(
        key="length_prior",
        display_name="prior from word length",
        claim_label="stage3_miswire_length_prior",
    ),
    Stage3Miswiring(
        key="reversed_prior",
        display_name="frequency ranking\nREVERSED",
        claim_label="stage3_miswire_reversed_prior",
    ),
    Stage3Miswiring(
        key="time_reversed",
        display_name="time-reversed emission",
        claim_label="stage3_miswire_time_reversed",
    ),
    Stage3Miswiring(
        key="early_end_of_word",
        display_name="end-of-word emitted early",
        claim_label=None,
    ),
)


def validate_stage3_miswirings(
    registry: Sequence[Stage3Miswiring] = STAGE3_MISWIRINGS,
) -> tuple[Stage3Miswiring, ...]:
    """Return a registry whose keys, claims, and score classes are unambiguous."""
    arms = tuple(registry)
    if not arms:
        raise ValueError("Stage 3 miswiring registry cannot be empty")
    keys = [arm.key for arm in arms]
    if len(set(keys)) != len(keys):
        raise ValueError("Stage 3 miswiring registry contains duplicate keys")
    labels = [arm.claim_label for arm in arms if arm.claim_label is not None]
    if len(set(labels)) != len(labels):
        raise ValueError("Stage 3 miswiring registry contains duplicate claim labels")
    invalid_labels = [label for label in labels if not label.startswith("stage3_miswire_")]
    if invalid_labels:
        raise ValueError(f"Stage 3 miswiring claim labels have the wrong family: {invalid_labels}")
    analytic = [arm for arm in arms if not arm.has_finite_score]
    if len(analytic) != 1 or analytic[0].key != "early_end_of_word":
        raise ValueError(
            "Stage 3 registry must contain exactly the analytic-infinite early_end_of_word arm"
        )
    return arms


#: Memory budget for the word factor's dense transition, in bytes. ``B`` over
#: word identity is ``N x N`` float32 even though it is the identity, because
#: no sparse path is reachable through pymdp's public Agent API (decision 58:
#: the sparse contraction machinery exists, the Agent API is where it is
#: blocked) -- so the vocabulary that can be *instantiated* is bounded by this,
#: and the bound is arithmetic rather than a guess:
#:
#:     N = 2,000   ->    16 MB        N =  16,000 ->   1.0 GB
#:     N = 8,000   ->   256 MB
#:     N = 126,000 ->    63 GB        (full CMUdict -- not instantiable)
#:
#: Above the bound the closed form stands in, which is exact rather than an
#: approximation: the path-probability identity means the model's score for a
#: word IS its prior mass, so ``word_perplexity`` needs no agent at all. What is
#: lost above the bound is the ability to *run pymdp*, not the ability to measure
#: the model -- and saying which of the two is lost is the point of the guard.
MAX_AGENT_VOCAB_BYTES: int = 1_200_000_000

DEFAULT_ARTIFACT = resolve_vendored_artifact("brown_word_counts.json")


class WordArtifact:
    """The vendored Stage 3 sufficient statistics, validated on load.

    Attributes:
        vocab: Vocabulary words in descending training frequency.
        word_to_index: Map from word to its index in ``vocab``.
        word_freq: Training frequency of each vocabulary word.
        bar_train_freq: Every training type's frequency, for the letter-bigram
            bar. Deliberately *not* restricted to the vocabulary — handicapping
            the comparator to the model's lexicon would make it crippled.
        eval_tokens: Held-out in-vocabulary token stream.
        provenance: The generator's recorded parameters.
    """

    def __init__(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            raise ValueError(f"word artifact must be a JSON object, got {type(payload).__name__}")
        required = {
            "schema",
            "provenance",
            "documents",
            "word_freq",
            "bar_train_freq",
            "eval_tokens",
        }
        missing = required - set(payload)
        extra = set(payload) - required
        if missing or extra:
            raise ValueError(
                "word artifact fields do not match the v2 schema: "
                f"missing={sorted(missing)}, unexpected={sorted(extra)}"
            )
        schema = payload.get("schema")
        if schema != BROWN_WORD_ARTIFACT_SCHEMA:
            raise ValueError(
                f"Unexpected artifact schema {schema!r}; regenerate with "
                f"scripts/generate_word_counts.py"
            )
        validate_brown_artifact_provenance(payload)

        self.word_freq = self._validate_frequency_map("word_freq", payload["word_freq"])
        self.bar_train_freq = self._validate_frequency_map(
            "bar_train_freq", payload["bar_train_freq"]
        )
        eval_tokens = payload["eval_tokens"]
        if not isinstance(eval_tokens, list):
            raise ValueError(f"eval_tokens must be a JSON array, got {type(eval_tokens).__name__}")
        self.eval_tokens = list(eval_tokens)
        self.vocab: list[str] = list(self.word_freq)
        self.word_to_index: dict[str, int] = {w: i for i, w in enumerate(self.vocab)}

        if not self.vocab:
            raise ValueError("artifact carries an empty vocabulary")
        if not self.eval_tokens:
            raise ValueError("artifact carries an empty evaluation stream")
        invalid_eval = next(
            (
                (index, token)
                for index, token in enumerate(self.eval_tokens)
                if not isinstance(token, str) or token not in self.word_to_index
            ),
            None,
        )
        if invalid_eval is not None:
            index, stray = invalid_eval
            raise ValueError(
                f"eval_tokens[{index}] is out-of-vocabulary or not a word: {stray!r}; the "
                f"artifact was not built by the shipped generator"
            )
        self.provenance = self._validate_provenance(payload["provenance"])
        (
            self.document_ids,
            self.document_categories,
            self.document_token_counts,
            split_document_indices,
            self.eval_token_document_indices,
        ) = self._validate_documents(payload["documents"])
        self.train_document_indices = split_document_indices["train"]
        self.heldout_document_indices = split_document_indices["heldout"]

        missing_from_bar = sorted(set(self.word_freq) - set(self.bar_train_freq))
        if missing_from_bar:
            raise ValueError(
                "word_freq is not a subset of bar_train_freq; missing training "
                f"types include {missing_from_bar[:3]!r}"
            )
        mismatched = next(
            (word for word, count in self.word_freq.items() if self.bar_train_freq[word] != count),
            None,
        )
        if mismatched is not None:
            raise ValueError(
                f"word_freq count for {mismatched!r} ({self.word_freq[mismatched]}) "
                "does not match bar_train_freq "
                f"({self.bar_train_freq[mismatched]})"
            )

        max_word_len = self.provenance["max_word_len"]
        too_long = next((word for word in self.vocab if len(word) > max_word_len), None)
        if too_long is not None:
            raise ValueError(
                f"vocabulary word {too_long!r} exceeds provenance max_word_len={max_word_len}"
            )
        eligible_prefix = [word for word in self.bar_train_freq if len(word) <= max_word_len][
            : len(self.vocab)
        ]
        if self.vocab != eligible_prefix:
            raise ValueError(
                "word_freq must be the frequency-ranked eligible prefix of "
                "bar_train_freq; the vocabulary order or membership drifted"
            )

        if self.provenance["vocab_size"] != len(self.vocab):
            raise ValueError(
                f"provenance vocab_size={self.provenance['vocab_size']} does not "
                f"match word_freq cardinality {len(self.vocab)}"
            )
        if self.provenance["train_types"] != len(self.bar_train_freq):
            raise ValueError(
                f"provenance train_types={self.provenance['train_types']} does not "
                f"match bar_train_freq cardinality {len(self.bar_train_freq)}"
            )
        if len(self.eval_tokens) > self.provenance["held_tokens"]:
            raise ValueError(
                f"evaluation stream has {len(self.eval_tokens)} tokens but provenance "
                f"declares only {self.provenance['held_tokens']} held tokens"
            )

        lexicon_source = self.provenance["lexicon_source"]
        if lexicon_source == "brown-frequency-ranked":
            dropped = sum(len(word) > max_word_len for word in self.bar_train_freq)
            if dropped != self.provenance["types_dropped_for_length"]:
                raise ValueError(
                    "provenance types_dropped_for_length does not match the "
                    f"training vocabulary: {self.provenance['types_dropped_for_length']} "
                    f"declared vs {dropped} measured"
                )
            train_count = sum(self.bar_train_freq.values())
            if train_count != self.provenance["train_tokens"]:
                raise ValueError(
                    f"bar_train_freq sums to {train_count} tokens but provenance "
                    f"declares train_tokens={self.provenance['train_tokens']}"
                )
        else:
            frequency_total = sum(self.bar_train_freq.values())
            if frequency_total != self.provenance["lexicon_frequency_total"]:
                raise ValueError(
                    f"bar_train_freq sums to {frequency_total} custom-lexicon counts but "
                    "provenance declares lexicon_frequency_total="
                    f"{self.provenance['lexicon_frequency_total']}"
                )

        vocab_bound = self.provenance.get("vocab_bound")
        if vocab_bound is not None:
            if vocab_bound == "unbounded":
                pass
            elif (
                isinstance(vocab_bound, bool) or not isinstance(vocab_bound, int) or vocab_bound < 1
            ):
                raise ValueError(
                    "provenance vocab_bound must be a positive integer or "
                    f"'unbounded', got {vocab_bound!r}"
                )
            elif len(self.vocab) > vocab_bound:
                raise ValueError(
                    f"word_freq has {len(self.vocab)} types above vocab_bound={vocab_bound}"
                )

    def _validate_documents(
        self, value: object
    ) -> tuple[list[str], list[str], list[int], dict[str, list[int]], list[int]]:
        """Validate source identity, whole-document partition, and token alignment."""
        if not isinstance(value, dict):
            raise ValueError(f"documents must be a JSON object, got {type(value).__name__}")
        required = {
            "ids",
            "categories",
            "alphabetic_token_counts",
            "split_indices",
            "eval_token_document_indices",
        }
        missing = required - set(value)
        extra = set(value) - required
        if missing or extra:
            raise ValueError(
                "documents fields do not match the v2 schema: "
                f"missing={sorted(missing)}, unexpected={sorted(extra)}"
            )

        document_ids = value["ids"]
        if not isinstance(document_ids, list) or not document_ids:
            raise ValueError("documents.ids must be a non-empty JSON array")
        try:
            identity = brown_document_identity_sha256(document_ids)
        except ValueError as exc:
            raise ValueError(f"invalid word-artifact document identities: {exc}") from exc
        if identity != self.provenance["document_identity_sha256"]:
            raise ValueError(
                "word artifact document identity digest disagrees with its ordered ids"
            )

        categories = value["categories"]
        if not isinstance(categories, list) or len(categories) != len(document_ids):
            raise ValueError("documents.categories must align one-for-one with ids")
        for index, category in enumerate(categories):
            if not isinstance(category, str) or not category or "\0" in category:
                raise ValueError(f"documents.categories[{index}] is invalid: {category!r}")
        partition_digest = brown_partition_input_sha256(document_ids, categories)
        if partition_digest != self.provenance["document_partition_input_sha256"]:
            raise ValueError(
                "word artifact partition-input digest disagrees with its ordered "
                "identity/category records"
            )

        counts = value["alphabetic_token_counts"]
        if not isinstance(counts, list) or len(counts) != len(document_ids):
            raise ValueError("documents.alphabetic_token_counts must align one-for-one with ids")
        for index, count in enumerate(counts):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(
                    "documents.alphabetic_token_counts"
                    f"[{index}] must be a non-negative integer, got {count!r}"
                )

        split_indices = value["split_indices"]
        if not isinstance(split_indices, dict) or set(split_indices) != {"train", "heldout"}:
            raise ValueError("documents.split_indices must contain exactly train and heldout")
        validated_splits: dict[str, list[int]] = {}
        for name in ("train", "heldout"):
            indices = split_indices[name]
            if not isinstance(indices, list) or not indices:
                raise ValueError(f"documents.split_indices.{name} must be a non-empty array")
            if any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(document_ids)
                for index in indices
            ):
                raise ValueError(f"documents.split_indices.{name} contains an invalid index")
            if indices != sorted(set(indices)):
                raise ValueError(
                    f"documents.split_indices.{name} must be unique and corpus-ordered"
                )
            validated_splits[name] = list(indices)

        train_indices = validated_splits["train"]
        heldout_indices = validated_splits["heldout"]
        if set(train_indices) & set(heldout_indices):
            raise ValueError("word artifact train and heldout document sets overlap")
        if set(train_indices) | set(heldout_indices) != set(range(len(document_ids))):
            raise ValueError("word artifact document split is not exhaustive")
        expected_train, expected_heldout = category_stratified_document_indices(
            document_ids,
            categories,
            self.provenance["target_train_document_fraction"],
        )
        if train_indices != list(expected_train) or heldout_indices != list(expected_heldout):
            raise ValueError(
                "word artifact split does not match the category-stratified "
                "revision-bound document rule"
            )

        train_tokens = sum(counts[index] for index in train_indices)
        heldout_tokens = sum(counts[index] for index in heldout_indices)
        expected_cardinalities = {
            "documents": len(document_ids),
            "train_documents": len(train_indices),
            "heldout_documents": len(heldout_indices),
            "categories": len(set(categories)),
            "total_alphabetic_tokens": train_tokens + heldout_tokens,
            "train_tokens": train_tokens,
            "held_tokens": heldout_tokens,
        }
        for field, expected in expected_cardinalities.items():
            if self.provenance[field] != expected:
                raise ValueError(
                    f"provenance {field}={self.provenance[field]} disagrees with "
                    f"documents metadata ({expected})"
                )

        eval_document_indices = value["eval_token_document_indices"]
        if not isinstance(eval_document_indices, list) or len(eval_document_indices) != len(
            self.eval_tokens
        ):
            raise ValueError(
                "documents.eval_token_document_indices must align one-for-one with eval_tokens"
            )
        heldout_set = set(heldout_indices)
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index not in heldout_set
            for index in eval_document_indices
        ):
            raise ValueError(
                "documents.eval_token_document_indices contains a non-heldout document"
            )
        if eval_document_indices != sorted(eval_document_indices):
            raise ValueError(
                "documents.eval_token_document_indices must preserve corpus document order"
            )
        per_document_eval: dict[int, int] = {}
        for index in eval_document_indices:
            per_document_eval[index] = per_document_eval.get(index, 0) + 1
        overfull = next(
            (index for index, count in per_document_eval.items() if count > counts[index]),
            None,
        )
        if overfull is not None:
            raise ValueError(
                f"document {overfull} maps {per_document_eval[overfull]} evaluation "
                f"tokens but contains only {counts[overfull]} normalized tokens"
            )
        return (
            list(document_ids),
            list(categories),
            list(counts),
            validated_splits,
            list(eval_document_indices),
        )

    @staticmethod
    def _validate_frequency_map(name: str, value: object) -> dict[str, int]:
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a JSON object, got {type(value).__name__}")
        letters = frozenset(LETTERS)
        validated: dict[str, int] = {}
        for word, count in value.items():
            if (
                not isinstance(word, str)
                or not word
                or word != word.lower()
                or not letters.issuperset(word)
            ):
                raise ValueError(
                    f"{name} key {word!r} must be a non-empty lowercase ASCII-letter word"
                )
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(
                    f"{name}[{word!r}] must be a positive integer count, got {count!r}"
                )
            validated[word] = count
        expected_order = sorted(validated.items(), key=lambda item: (-item[1], item[0]))
        if list(validated.items()) != expected_order:
            raise ValueError(f"{name} must be ordered by descending count then word")
        return validated

    @staticmethod
    def _validate_provenance(value: object) -> dict:
        if not isinstance(value, dict):
            raise ValueError(f"provenance must be a JSON object, got {type(value).__name__}")
        required = {
            "corpus",
            "raw_source",
            "normalization",
            "target_train_document_fraction",
            "train_document_fraction",
            "train_token_fraction",
            "split",
            "vocab_size",
            "vocab_bound",
            "lexicon_source",
            "max_word_len",
            "types_dropped_for_length",
            "total_alphabetic_tokens",
            "train_tokens",
            "held_tokens",
            "train_types",
            "documents",
            "train_documents",
            "heldout_documents",
            "categories",
            "document_identity_sha256",
            "document_partition_input_sha256",
        }
        custom_only = {"lexicon_sha256", "lexicon_frequency_total"}
        missing = required - set(value)
        extra = set(value) - required - custom_only
        if missing or extra:
            raise ValueError(
                "provenance fields do not match the v2 schema: "
                f"missing={sorted(missing)}, unexpected={sorted(extra)}"
            )

        provenance = dict(value)
        expected_text = {
            "corpus": "NLTK Brown",
            "normalization": "lowercased; tokens of ASCII letters only",
            "split": (
                "category-stratified revision-bound SHA-256 document split; "
                "not by word type (closed vocabulary)"
            ),
        }
        for field, expected in expected_text.items():
            if provenance[field] != expected:
                raise ValueError(
                    f"provenance {field} must be {expected!r}, got {provenance[field]!r}"
                )
        lexicon_source = provenance["lexicon_source"]
        if lexicon_source == "brown-frequency-ranked":
            unexpected_custom = custom_only & set(provenance)
            if unexpected_custom:
                raise ValueError(
                    "Brown provenance carries custom-lexicon-only fields "
                    f"{sorted(unexpected_custom)}"
                )
        elif not (
            isinstance(lexicon_source, str)
            and lexicon_source.startswith("custom:")
            and lexicon_source.removeprefix("custom:")
            and "/" not in lexicon_source.removeprefix("custom:")
            and "\\" not in lexicon_source.removeprefix("custom:")
        ):
            raise ValueError(
                "provenance lexicon_source must be 'brown-frequency-ranked' or "
                f"a basename-prefixed custom source, got {lexicon_source!r}"
            )
        else:
            missing_custom = custom_only - set(provenance)
            if missing_custom:
                raise ValueError(f"custom lexicon provenance is missing {sorted(missing_custom)}")
            digest = provenance["lexicon_sha256"]
            if not (
                isinstance(digest, str)
                and len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(
                    "provenance lexicon_sha256 must be a lowercase SHA-256 hex digest, "
                    f"got {digest!r}"
                )
            frequency_total = provenance["lexicon_frequency_total"]
            if (
                isinstance(frequency_total, bool)
                or not isinstance(frequency_total, int)
                or frequency_total < 1
            ):
                raise ValueError(
                    "provenance lexicon_frequency_total must be a positive integer, "
                    f"got {frequency_total!r}"
                )

        for field in (
            "target_train_document_fraction",
            "train_document_fraction",
            "train_token_fraction",
        ):
            fraction = provenance[field]
            if (
                isinstance(fraction, bool)
                or not isinstance(fraction, Real)
                or not math.isfinite(float(fraction))
                or not 0.0 < float(fraction) < 1.0
            ):
                raise ValueError(
                    f"provenance {field} must be finite and in (0, 1), got {fraction!r}"
                )
            provenance[field] = float(fraction)

        for field in ("document_identity_sha256", "document_partition_input_sha256"):
            digest = provenance[field]
            if not (
                isinstance(digest, str)
                and len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(
                    f"provenance {field} must be a lowercase SHA-256 hex digest, got {digest!r}"
                )

        positive_fields = (
            "vocab_size",
            "max_word_len",
            "total_alphabetic_tokens",
            "train_tokens",
            "held_tokens",
            "train_types",
            "documents",
            "train_documents",
            "heldout_documents",
            "categories",
        )
        for field in positive_fields:
            field_value = provenance[field]
            if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value < 1:
                raise ValueError(
                    f"provenance {field} must be a positive integer, got {field_value!r}"
                )
        dropped = provenance["types_dropped_for_length"]
        if isinstance(dropped, bool) or not isinstance(dropped, int) or dropped < 0:
            raise ValueError(
                "provenance types_dropped_for_length must be a non-negative "
                f"integer, got {dropped!r}"
            )
        total = provenance["total_alphabetic_tokens"]
        train = provenance["train_tokens"]
        held = provenance["held_tokens"]
        if train + held != total:
            raise ValueError(
                f"provenance token cardinalities disagree: train {train} + held "
                f"{held} != total {total}"
            )
        if not math.isclose(provenance["train_token_fraction"], train / total, abs_tol=1e-15):
            raise ValueError(
                "provenance train_token_fraction disagrees with train_tokens / "
                "total_alphabetic_tokens"
            )
        if (
            provenance["train_documents"] + provenance["heldout_documents"]
            != provenance["documents"]
        ):
            raise ValueError(
                "provenance document cardinalities disagree: train_documents + "
                "heldout_documents != documents"
            )
        if not math.isclose(
            provenance["train_document_fraction"],
            provenance["train_documents"] / provenance["documents"],
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "provenance train_document_fraction disagrees with train_documents / documents"
            )
        return provenance

    @property
    def max_word_len(self) -> int:
        """Configured vocabulary horizon, which sizes the position factor.

        This is the artifact's declared cutoff, not the longest word that happens
        to survive a frequency-ranked vocabulary bound. Structural dimensions
        must not drift when corpus composition changes which word ranks last.
        """
        return int(self.provenance["max_word_len"])

    @property
    def observed_max_word_len(self) -> int:
        """Longest word actually present in this artifact's bounded vocabulary."""
        return max(len(w) for w in self.vocab)

    @property
    def n_positions(self) -> int:
        """Position-factor size: one state per letter slot plus the END slot."""
        return self.max_word_len + 1

    @property
    def coverage_pct(self) -> float:
        """Share of held-out tokens the model is defined on."""
        return 100.0 * len(self.eval_tokens) / self.provenance["held_tokens"]


def load_word_artifact(path: Path | str | None = None) -> WordArtifact:
    """Load and validate the vendored word artifact.

    Args:
        path: Artifact location; defaults to ``vendor/brown_word_counts.json``.

    Returns:
        The validated artifact.

    Raises:
        FileNotFoundError: If the artifact is absent. Stage 3 has no fallback
            corpus — a lexicon assembled from stand-in prose would be a
            different language, silently.
        ValueError: If the artifact fails validation.
    """
    resolved = (
        DEFAULT_ARTIFACT
        if path is None
        else resolve_vendored_artifact("brown_word_counts.json", path=path)
    )
    if not resolved.is_file():
        raise FileNotFoundError(f"{resolved} is absent. Run scripts/generate_word_counts.py.")
    artifact = WordArtifact(json.loads(resolved.read_text(encoding="utf-8")))
    validate_canonical_brown_partition_target(artifact.provenance["target_train_document_fraction"])
    validate_canonical_brown_document_metadata(
        artifact.document_ids,
        artifact.document_categories,
    )
    heldout_stream = resolve_vendored_artifact("brown_heldout_stream.json")
    if not heldout_stream.is_file():
        raise FileNotFoundError(
            f"{heldout_stream} is absent. Run scripts/generate_heldout_stream.py "
            "before loading a canonical word artifact."
        )
    validate_heldout_stream_artifact(
        json.loads(heldout_stream.read_text(encoding="utf-8")),
        artifact,
    )
    return artifact


def validate_heldout_stream_artifact(
    payload: object,
    word_artifact: WordArtifact,
) -> list[str]:
    """Validate the unfiltered Stage 3 denominator against its scoring artifact.

    This check is deliberately raw-corpus independent: a committed artifact must
    prove its source-document projection and token alignment even in an offline
    installed environment. Raw-source regeneration is a separate, stronger gate.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"held-out stream must be a JSON object, got {type(payload).__name__}")
    required = {"schema", "provenance", "documents", "held_tokens"}
    missing = required - set(payload)
    extra = set(payload) - required
    if missing or extra:
        raise ValueError(
            "held-out stream fields do not match the v3 schema: "
            f"missing={sorted(missing)}, unexpected={sorted(extra)}"
        )
    if payload["schema"] != BROWN_HELDOUT_ARTIFACT_SCHEMA:
        raise ValueError(
            f"unexpected held-out stream schema {payload['schema']!r}; regenerate "
            "with scripts/generate_heldout_stream.py"
        )
    validate_brown_artifact_provenance(payload)

    provenance = payload["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("held-out stream provenance must be a JSON object")
    required_provenance = {
        "corpus",
        "raw_source",
        "normalization",
        "target_train_document_fraction",
        "train_document_fraction",
        "train_token_fraction",
        "split",
        "total_alphabetic_tokens",
        "train_tokens",
        "held_tokens",
        "held_types",
        "documents",
        "train_documents",
        "heldout_documents",
        "categories",
        "document_identity_sha256",
        "document_partition_input_sha256",
        "max_word_len_applied",
        "max_word_len_in_word_artifact",
    }
    missing = required_provenance - set(provenance)
    extra = set(provenance) - required_provenance
    if missing or extra:
        raise ValueError(
            "held-out stream provenance fields do not match the v3 schema: "
            f"missing={sorted(missing)}, unexpected={sorted(extra)}"
        )
    expected_text = {
        "corpus": "NLTK Brown",
        "normalization": "lowercased; tokens of ASCII letters only",
        "split": "category-stratified revision-bound SHA-256 document split",
    }
    for field, expected in expected_text.items():
        if provenance[field] != expected:
            raise ValueError(
                f"held-out stream provenance {field} must be {expected!r}, "
                f"got {provenance[field]!r}"
            )
    if provenance["max_word_len_applied"] is not False:
        raise ValueError("held-out stream must not apply the word-artifact length bound")
    if provenance["max_word_len_in_word_artifact"] != word_artifact.provenance["max_word_len"]:
        raise ValueError("held-out stream and word artifact disagree on max_word_len")

    shared_provenance = {
        "raw_source": "raw_source",
        "target_train_document_fraction": "target_train_document_fraction",
        "train_document_fraction": "train_document_fraction",
        "train_token_fraction": "train_token_fraction",
        "total_alphabetic_tokens": "total_alphabetic_tokens",
        "train_tokens": "train_tokens",
        "held_tokens": "held_tokens",
        "documents": "documents",
        "train_documents": "train_documents",
        "heldout_documents": "heldout_documents",
        "categories": "categories",
        "document_identity_sha256": "document_identity_sha256",
        "document_partition_input_sha256": "document_partition_input_sha256",
    }
    for stream_field, word_field in shared_provenance.items():
        if provenance[stream_field] != word_artifact.provenance[word_field]:
            raise ValueError(f"held-out stream {stream_field} disagrees with the word artifact")

    documents = payload["documents"]
    if not isinstance(documents, dict):
        raise ValueError("held-out stream documents must be a JSON object")
    required_documents = {
        "ids",
        "categories",
        "alphabetic_token_counts",
        "split_indices",
        "held_token_document_indices",
    }
    missing = required_documents - set(documents)
    extra = set(documents) - required_documents
    if missing or extra:
        raise ValueError(
            "held-out stream documents fields do not match the v3 schema: "
            f"missing={sorted(missing)}, unexpected={sorted(extra)}"
        )
    expected_documents: dict[str, object] = {
        "ids": word_artifact.document_ids,
        "categories": word_artifact.document_categories,
        "alphabetic_token_counts": word_artifact.document_token_counts,
        "split_indices": {
            "train": word_artifact.train_document_indices,
            "heldout": word_artifact.heldout_document_indices,
        },
    }
    for document_field, document_expected in expected_documents.items():
        if documents[document_field] != document_expected:
            raise ValueError(
                f"held-out stream documents.{document_field} disagrees with the word artifact"
            )

    held_tokens = payload["held_tokens"]
    if not isinstance(held_tokens, list) or not held_tokens:
        raise ValueError("held-out stream held_tokens must be a non-empty JSON array")
    invalid_token = next(
        (
            (index, token)
            for index, token in enumerate(held_tokens)
            if not isinstance(token, str)
            or not token
            or token != token.lower()
            or not frozenset(LETTERS).issuperset(token)
        ),
        None,
    )
    if invalid_token is not None:
        raise ValueError(
            f"held_tokens[{invalid_token[0]}] is not a lowercase ASCII-letter token: "
            f"{invalid_token[1]!r}"
        )
    if len(held_tokens) != provenance["held_tokens"]:
        raise ValueError("held-out stream token count disagrees with provenance")
    if len(set(held_tokens)) != provenance["held_types"]:
        raise ValueError("held-out stream type count disagrees with provenance")

    token_document_indices = documents["held_token_document_indices"]
    if not isinstance(token_document_indices, list) or len(token_document_indices) != len(
        held_tokens
    ):
        raise ValueError("held_token_document_indices must align one-for-one with held_tokens")
    heldout_set = set(word_artifact.heldout_document_indices)
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index not in heldout_set
        for index in token_document_indices
    ):
        raise ValueError("held_token_document_indices contains a non-heldout document")
    if token_document_indices != sorted(token_document_indices):
        raise ValueError("held_token_document_indices must preserve corpus document order")
    mapped_counts: dict[int, int] = {}
    for index in token_document_indices:
        mapped_counts[index] = mapped_counts.get(index, 0) + 1
    expected_counts = {
        index: word_artifact.document_token_counts[index]
        for index in word_artifact.heldout_document_indices
    }
    if mapped_counts != expected_counts:
        raise ValueError(
            "held_token_document_indices counts disagree with per-document token counts"
        )

    vocabulary = set(word_artifact.word_freq)
    covered = [
        (token, document_index)
        for token, document_index in zip(held_tokens, token_document_indices, strict=True)
        if token in vocabulary
    ]
    if [token for token, _index in covered] != word_artifact.eval_tokens:
        raise ValueError(
            "filtering the held-out stream does not reproduce eval_tokens element-for-element"
        )
    if [index for _token, index in covered] != word_artifact.eval_token_document_indices:
        raise ValueError(
            "filtering the held-out stream does not reproduce evaluation document alignment"
        )
    return list(held_tokens)


def build_word_prior(
    artifact: WordArtifact, *, config: LevelConfig = DEFAULT_LEVEL_CONFIG
) -> np.ndarray:
    """The learned quantity of this level: ``D[0]``, word unigram frequencies.

    Smoothing is applied over the *bounded vocabulary*, which is the whole
    support. It does not open the lexicon: an out-of-vocabulary word still gets
    no probability, because it has no state.

    Args:
        artifact: The vendored statistics.
        config: Supplies the add-k pseudocount.

    Returns:
        ``(n_vocab,)`` float64 array summing to 1.
    """
    counts = np.array([artifact.word_freq[w] for w in artifact.vocab], dtype=np.float64)
    smoothed = counts + config.smoothing
    return smoothed / smoothed.sum()


def build_emission(artifact: WordArtifact) -> np.ndarray:
    """``A[0]``: deterministic letter emission, shape ``(27, n_vocab, n_pos)``.

    Normalized over the **observation** axis for every ``(word, position)`` pair.
    Asserting the sum along any other axis would be a green check on a
    transposed matrix, which is why the axis is named here and pinned in
    ``tests/test_stage3_gate.py``.

    Args:
        artifact: The vendored statistics.

    Returns:
        ``(27, n_vocab, n_pos)`` float64 array, column-stochastic over axis 0.
    """
    n_pos = artifact.n_positions
    emission = np.zeros((N_OBS, len(artifact.vocab), n_pos), dtype=np.float64)
    for w_idx, word in enumerate(artifact.vocab):
        for pos in range(n_pos):
            obs = ord(word[pos]) - 97 if pos < len(word) else END_INDEX
            emission[obs, w_idx, pos] = 1.0
    return emission


def build_word_transition(artifact: WordArtifact) -> np.ndarray:
    """``B[0]``: the identity over word identity, shape ``(n_vocab, n_vocab)``.

    A word does not become a different word part-way through being spelled. Any
    other matrix here lets the model change its mind about which word it is
    emitting, which breaks the path-probability identity that pins the level.
    """
    return np.eye(len(artifact.vocab), dtype=np.float64)


def build_position_transition(artifact: WordArtifact) -> np.ndarray:
    """``B[1]``: the +1 shift over position, shape ``(n_pos, n_pos)``.

    ``B[next, current]``, absorbing at the final position so that a word past its
    last letter keeps emitting END. The orientation matters: the transpose is the
    time-reversed model that emits words back to front.
    """
    n_pos = artifact.n_positions
    transition = np.zeros((n_pos, n_pos), dtype=np.float64)
    for pos in range(n_pos):
        transition[min(pos + 1, n_pos - 1), pos] = 1.0
    return transition


def build_word_agent(
    artifact: WordArtifact,
    *,
    config: LevelConfig = DEFAULT_LEVEL_CONFIG,
    word_prior: np.ndarray | None = None,
    emission: np.ndarray | None = None,
) -> PymdpAgent:
    """Build the Stage 3 pymdp Agent.

    Args:
        artifact: The vendored statistics.
        config: Hyperparameters; supplies the prior's pseudocount.
        word_prior: Override for ``D[0]``. Present so negative controls can
            install a mis-wired prior through the shipped builder rather than a
            parallel one — a control that exercises different code proves
            nothing about the shipped path.
        emission: Override for ``A[0]``, for the same reason.

    Returns:
        A configured ``pymdp.agent.Agent``.

    Raises:
        ValueError: If an override has the wrong shape.
    """
    from pymdp.agent import Agent

    n_vocab = len(artifact.vocab)
    n_pos = artifact.n_positions

    # Fail with the arithmetic rather than with an allocation error. A vocabulary
    # too large to instantiate is a documented boundary of the pymdp path, not a
    # bug, and the closed form remains exact above it.
    required = n_vocab * n_vocab * 4
    if required > MAX_AGENT_VOCAB_BYTES:
        raise ValueError(
            f"A {n_vocab:,}-word vocabulary needs {required / 1e9:.1f} GB for the "
            f"word factor's dense transition, over the "
            f"{MAX_AGENT_VOCAB_BYTES / 1e9:.1f} GB budget. No sparse path is "
            f"reachable through pymdp's public Agent API, so the agent cannot be "
            f"instantiated at this size.\n\n"
            f"This does NOT block measurement: every factor except D is "
            f"deterministic, so word_perplexity() and the oracle compute the "
            f"model's scores in closed form at any vocabulary size, exactly. Use "
            f"those, and instantiate the agent at a smaller vocabulary to prove "
            f"the closed form still describes it (see docs/decisions.md entry 13)."
        )

    prior = (
        build_word_prior(artifact, config=config)
        if word_prior is None
        else np.asarray(word_prior, dtype=np.float64)
    )
    if prior.shape != (n_vocab,):
        raise ValueError(f"word_prior must be ({n_vocab},), got {prior.shape}")

    emit = build_emission(artifact) if emission is None else np.asarray(emission, dtype=np.float64)
    if emit.shape != (N_OBS, n_vocab, n_pos):
        raise ValueError(f"emission must be ({N_OBS}, {n_vocab}, {n_pos}), got {emit.shape}")

    position_start = np.zeros(n_pos, dtype=np.float64)
    position_start[0] = 1.0

    return Agent(
        A=[jnp.array(emit, dtype=jnp.float32)[jnp.newaxis]],
        B=[
            jnp.array(build_word_transition(artifact), dtype=jnp.float32)[
                jnp.newaxis, :, :, jnp.newaxis
            ],
            jnp.array(build_position_transition(artifact), dtype=jnp.float32)[
                jnp.newaxis, :, :, jnp.newaxis
            ],
        ],
        C=[jnp.zeros((1, N_OBS))],
        D=[
            jnp.array(prior, dtype=jnp.float32)[jnp.newaxis],
            jnp.array(position_start, dtype=jnp.float32)[jnp.newaxis],
        ],
        A_dependencies=[[0, 1]],
        B_dependencies=[[0], [1]],
    )


def posterior_word_support(
    agent: PymdpAgent, artifact: WordArtifact, prefix: str, *, tol: float = 1e-6
) -> list[str]:
    """Words the agent still considers possible after observing ``prefix``.

    This is the trie node, read out of the factored model. Used by
    ``tests/test_stage3_gate.py`` to check the induced-trie correspondence rather
    than assume it.

    Args:
        agent: A Stage 3 agent.
        artifact: The artifact it was built from.
        prefix: Letters observed so far.
        tol: Posterior mass below which a word counts as excluded.

    Returns:
        Vocabulary words with posterior mass above ``tol``, in vocabulary order.

    Raises:
        ValueError: If ``prefix`` is empty or not lowercase ASCII letters.
    """
    if not prefix or not all(ch in LETTERS for ch in prefix):
        raise ValueError(f"prefix must be non-empty lowercase letters, got {prefix!r}")

    n_pos = artifact.n_positions
    position_start = np.zeros(n_pos, dtype=np.float32)
    position_start[0] = 1.0
    prior = [
        jnp.array(np.asarray(agent.D[0]).reshape(1, -1)),
        jnp.array(position_start[np.newaxis]),
    ]
    qs = None
    for letter in prefix:
        qs = agent.infer_states([jnp.array([ord(letter) - 97])], prior)
        prior = agent.update_empirical_prior(jnp.array([[0, 0]]), qs)

    if qs is None:
        raise RuntimeError("no posterior was produced for the non-empty prefix")
    mass = np.asarray(qs[0]).ravel()
    return [w for i, w in enumerate(artifact.vocab) if mass[i] > tol]


def word_perplexity(
    artifact: WordArtifact,
    *,
    config: LevelConfig = DEFAULT_LEVEL_CONFIG,
    word_prior: np.ndarray | None = None,
    tokens: list[str] | None = None,
) -> tuple[float, int]:
    """Per-emission perplexity of the word level over the held-out stream.

    Computed in closed form. Every factor except ``D[0]`` is deterministic, so
    the path probability of a token *is* its prior mass; that identity is
    checked against a live pymdp walk in ``tests/test_stage3_gate.py`` before
    this function's output is trusted, which is the oracle-equivalence pattern
    Stage 2 established (``docs/gate_design.md``).

    Args:
        artifact: The vendored statistics.
        config: Hyperparameters.
        word_prior: Override for ``D[0]``, used by the negative controls.
        tokens: Evaluation stream; defaults to the artifact's held-out tokens.

    Returns:
        ``(perplexity, n_emissions)``.

    Raises:
        ValueError: If a token is out of vocabulary.
    """
    from ailm.verification.oracle import word_path_perplexity

    prior = (
        build_word_prior(artifact, config=config)
        if word_prior is None
        else np.asarray(word_prior, dtype=np.float64)
    )
    stream = artifact.eval_tokens if tokens is None else tokens
    return word_path_perplexity(stream, artifact.word_to_index, prior)


def build_letter_bigram_bar(
    artifact: WordArtifact, *, config: LevelConfig = DEFAULT_LEVEL_CONFIG
) -> tuple[np.ndarray, np.ndarray]:
    """The Stage 3 comparator: a letter-bigram model that does not know the lexicon.

    Trained on **every** training word type, not only the bounded vocabulary, so
    the bar is the strongest lexicon-free model rather than one handicapped to
    the level's own support (decision 2). Add-k smoothed, because unlike the word
    level it must assign mass to letter transitions it never saw.

    Args:
        artifact: The vendored statistics.
        config: Supplies the add-k pseudocount.

    Returns:
        ``(transition, start)`` — ``(27, 27)`` column-stochastic and ``(27,)``.
    """
    k = config.smoothing
    transition = np.full((N_OBS, N_OBS), k, dtype=np.float64)
    start = np.full(N_OBS, k, dtype=np.float64)

    for word, count in artifact.bar_train_freq.items():
        seq = [ord(ch) - 97 for ch in word] + [END_INDEX]
        start[seq[0]] += count
        for prev, nxt in pairwise(seq):
            transition[nxt, prev] += count

    return column_stochastic(transition), start / start.sum()


#: Vocabulary sizes the scaling sweep reports, smallest first. Frequency-ranked,
#: so each rung is a prefix of the next and the sweep is nested rather than a set
#: of unrelated lexicons. The top rung contains every length-eligible training
#: type in the artifact, whatever that count is, so the sweep does not stop at a
#: literal that would go stale when the corpus does.
SCALING_RUNGS: tuple[int, ...] = (500, 1000, 2000, 4000, 8000, 16000, 32000)


def _letter_bigram_perplexity(
    tokens: list[str],
    transition: np.ndarray,
    start: np.ndarray,
) -> tuple[float, int]:
    """Late-bound import of the oracle's bar, to keep this module import-light.

    The signature mirrors :func:`ailm.verification.oracle.letter_bigram_perplexity`
    exactly so mypy checks the call site against the real contract instead of an
    opaque ``*args: object`` passthrough.
    """
    from ailm.verification.oracle import letter_bigram_perplexity

    return letter_bigram_perplexity(tokens, transition, start)


def scaling_rungs(
    artifact: WordArtifact,
    *,
    sizes: Sequence[int] = SCALING_RUNGS,
    config: LevelConfig = DEFAULT_LEVEL_CONFIG,
) -> list[dict[str, float | int]]:
    """Coverage and perplexity at each vocabulary size, on one artifact.

    The word level's two numeric headline quantities both rise with lexicon
    size, but their utility directions oppose: higher coverage is desirable,
    while higher per-emission perplexity is worse. A bigger lexicon covers more
    of the held-out stream and faces a harder prediction problem. That trade is
    the finding; a single operating point hides it.

    Each rung takes the ``n`` most frequent types and restricts evaluation to the
    held-out tokens that lexicon covers, which is the same rule the shipped
    artifact follows — the model assigns *no* mass outside its vocabulary, so
    scoring a token it cannot represent would be scoring a fallback rather than
    the model. ``coverage_pct`` stays measured against the *original* held-out
    token count, so the rungs are comparable to each other and to the shipped
    number rather than each being a percentage of its own survivors.

    The sweep is cheap because every factor except the word prior is
    deterministic: a word's score simply is its prior mass, so no agent is
    instantiated and no rung is bounded by the ``pymdp`` memory cap that limits
    what can be *run*. That gap between what can be measured and what can be run
    is the point of [@sec:word_level]'s scaling paragraph.

    Args:
        artifact: The lexicon to sweep. Rungs at or above its vocabulary size
            collapse onto the set of all length-eligible training types in the
            artifact, which is reported once.
        sizes: Vocabulary sizes to report below the full size.
        config: Level configuration, passed through to :func:`word_perplexity`.

    Returns:
        One mapping per rung, smallest first, each with ``vocab_size``,
        ``coverage_pct``, ``perplexity``, ``bigram_baseline`` and ``positions``.
        The final entry always contains all length-eligible training types in
        the artifact.

    Raises:
        ValueError: If ``sizes`` contains a non-positive entry. A rung of zero
            types has no lexicon and would report a coverage of zero beside a
            perplexity computed over nothing.
    """
    if any(int(size) < 1 for size in sizes):
        raise ValueError(f"scaling rungs must be positive vocabulary sizes, got {list(sizes)}")

    held_tokens = int(artifact.provenance["held_tokens"])
    ranked = sorted(artifact.word_freq.items(), key=lambda item: (-item[1], item[0]))
    full_size = len(ranked)
    wanted = sorted({int(size) for size in sizes if int(size) < full_size} | {full_size})

    # Rebuild through the real validator rather than mutating the artifact:
    # a rung that skipped validation could carry an eval token its lexicon does
    # not contain, which is the one thing the artifact contract forbids and the
    # exact defect a truncated vocabulary invites.
    payload: dict[str, Any] = {
        "schema": BROWN_WORD_ARTIFACT_SCHEMA,
        "word_freq": artifact.word_freq,
        "bar_train_freq": artifact.bar_train_freq,
        "eval_tokens": artifact.eval_tokens,
        "provenance": artifact.provenance,
        "documents": {
            "ids": artifact.document_ids,
            "categories": artifact.document_categories,
            "alphabetic_token_counts": artifact.document_token_counts,
            "split_indices": {
                "train": artifact.train_document_indices,
                "heldout": artifact.heldout_document_indices,
            },
            "eval_token_document_indices": artifact.eval_token_document_indices,
        },
    }
    rungs: list[dict[str, float | int]] = []
    for size in wanted:
        lexicon = dict(ranked[:size])
        retained = [
            (token, document_index)
            for token, document_index in zip(
                artifact.eval_tokens,
                artifact.eval_token_document_indices,
                strict=True,
            )
            if token in lexicon
        ]
        tokens = [token for token, _document_index in retained]
        document_indices = [document_index for _token, document_index in retained]
        rung_artifact = WordArtifact(
            {
                **payload,
                "word_freq": lexicon,
                "eval_tokens": tokens,
                "documents": {
                    **payload["documents"],
                    "eval_token_document_indices": document_indices,
                },
                "provenance": {**artifact.provenance, "vocab_size": len(lexicon)},
            }
        )
        perplexity, positions = word_perplexity(rung_artifact, config=config)
        # The comparator is fit ONCE, on the full training frequencies, and
        # re-scored on each rung's surviving tokens. ``bar_train_freq`` is passed
        # through untouched above, so this call returns bit-identical matrices at
        # every rung — verified, not assumed. Saying "re-fit at every rung" would
        # credit the comparator's movement to a refit that does not happen; all
        # of its movement is the changing evaluation population.
        transition, start = build_letter_bigram_bar(rung_artifact)
        baseline, _ = _letter_bigram_perplexity(tokens, transition, start)
        rungs.append(
            {
                "vocab_size": len(rung_artifact.vocab),
                # Against the ORIGINAL held-out count, not this rung's survivors.
                "coverage_pct": round(100.0 * len(tokens) / held_tokens, 2),
                "perplexity": round(perplexity, 4),
                "bigram_baseline": round(baseline, 4),
                "positions": positions,
            }
        )

    # Two gains, because the per-emission one is not comparable across rungs and
    # reporting it alone inverts the finding. Perplexity at each rung is an
    # average over that rung's surviving positions, and a smaller lexicon
    # survives on an easier subset, so the per-emission gain is nearly flat and
    # slightly non-monotone — a property of the denominator. Weighting by the
    # positions each rung actually scores, against the top rung's count, puts
    # every rung on one denominator, which is the rule already enforced on the
    # coverage axis and for the same reason. On that scale the gain is strictly
    # monotone. Both are recorded so a reader can see which question each answers.
    top_positions = rungs[-1]["positions"]
    for rung in rungs:
        per_emission = math.log(float(rung["bigram_baseline"])) - math.log(
            float(rung["perplexity"])
        )
        rung["gain_nats"] = round(per_emission, 4)
        rung["gain_nats_shared_denominator"] = round(
            per_emission * float(rung["positions"]) / float(top_positions), 4
        )
    return rungs
