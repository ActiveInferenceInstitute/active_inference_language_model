"""Punctuation-type oracle -> L3: a typed word message into prior D[0].

The experiment this module exists to run
----------------------------------------
``docs/decisions.md`` entry 27 pre-registers what it called the **clause ->
word** channel as a
falsification test of the first clause of the novel-information criterion (entry
18; refined to two clauses by entries 25-26): a message helps iff it is **novel
to the receiver** and the receiver's likelihood is **sensitive to it**. The
prediction, written down before this module existed, is that this channel
**harms**. Decision 89 corrects the implemented sender to a privileged
punctuation-type oracle: this module consumes no L5 posterior.

The receiver is the word level (Stage 3, L3). Its learned quantity is the word
prior ``D[0] = P(word)``; its emission resolves word identity deterministically
from the letters, so L3 *already observes* each word it recognizes. A clause-
punctuation-type-conditioned statistic about the sentential context in which words appear is
therefore, at the receiving side, information the receiver already possesses —
it is lower-resolution (a distribution over words conditioned on sentence
type) than the exact word the receiver can read. Mixing it into ``D[0]``
substitutes a redundant statement for the empirical word prior.

The prediction: mixing a punctuation-type-conditioned word statistic into ``D[0]`` with
weight ``alpha`` yields a **non-positive** effect on the word-level
*predictive* — the held-out perplexity of the prior over in-vocabulary tokens
does not improve — and the sweep is a **monotone decline** (no interior
optimum). That is exactly the redundant-information signature of entry 9's
L1->L2B result, on a different receiver.

The corrected measurement does not confirm that prediction. On the canonical
Stage 3 whole-document partition, every tested positive weight improves the
declared perplexity and the best tested value is the high grid endpoint. The
sweep therefore establishes neither harm nor an interior optimum; it records a
small descriptive gain over the tested range. ``docs/decisions.md`` entries 27
and 88 own that chronology and interpretation.

Why the message is conditioned on sentence type, and the axis
--------------------------------------------------------------
The type variable is the clause level's own: sentence-final punctuation marks a
sentence declarative (``.``) or interrogative (``?``), exactly the two types
entry 26 and the Stage 5 clause models use. The message is ``P(word | type)``
estimated from **train** sentences and applied to **held-out** tokens, so it can
be wrong — the property that makes it a measurement rather than a lookup. The
vocabulary is the Stage 3 word vocabulary; all estimation and scoring happens
over it, and the closed-vocabulary bound of Stage 3 applies here too.

The meaningful axis is the predictive, not recognition: L3 word recognition is
deterministic given the letters, so accuracy is pinned and cannot register harm.
The score is the held-out perplexity of the (mixed) prior ``D[0]`` over
in-vocabulary tokens — the same axis decision 6 and ISC-28 govern.

Linear mixing, and the identity anchor
--------------------------------------
``D_alpha  <-  (1 - alpha) * D0  +  alpha * message_type``

At ``alpha = 0`` this is *exactly* the uncoupled prior ``D0`` — bit for bit —
which is the identity anchor that licenses trusting every other point of the
sweep (entry 9's lesson). Because both terms are distributions and the mix is a
convex combination, every ``D_alpha`` sums to 1 with no renormalization.

What this deliberately does not claim
------------------------------------
It is not an L5-output channel, evidence about learning, or a recommendation about
whether to *build* the channel — harm is the prediction, and a recorded harm
is a finding, not a verdict.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ailm.corpus.brown import (
    BROWN_TRAIN_DOCUMENT_FRACTION,
    brown_document_identity_sha256,
    brown_partition_input_sha256,
    category_stratified_document_indices,
    validate_brown_artifact_provenance,
    validate_canonical_brown_document_metadata,
    validate_canonical_brown_partition_target,
)
from ailm.resources import resolve_vendored_artifact

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    from ailm.clustering import ClusteredPairedSummary

__all__ = [
    "CLAUSE_TYPES",
    "CLAUSE_WORD_ARTIFACT_SCHEMA",
    "ClauseWordArtifact",
    "ClauseMessage",
    "build_clause_message",
    "clustered_clause_word_summaries",
    "load_clause_word_artifact",
    "message_for_type",
    "mix_prior",
    "scramble_word_distribution",
    "sweep_clause_alpha",
]

#: Tolerance on the sum-to-one check inside :func:`mix_prior`. Float32-sourced
#: distributions deviate from 1 by ~1e-7 at the 2,000-word vocabulary scale;
#: the tolerance admits rounding and refuses genuinely unnormalized operands.
MIX_NORMALIZATION_ATOL: float = 1e-6

logger = logging.getLogger(__name__)

CLAUSE_WORD_ARTIFACT_SCHEMA = "ailm-clause-word-counts-v3"
CLAUSE_TYPES: tuple[str, ...] = ("declarative", "interrogative")
CLAUSE_WORD_CORPUS = "NLTK Brown"
CLAUSE_WORD_SPLIT = "category-stratified document SHA-256 v1"
CLAUSE_WORD_TYPE_LABEL_SOURCE = (
    "sentence-final punctuation token: '.' declarative, '?' "
    "interrogative; '!' and unlabeled sentences excluded"
)
CLAUSE_WORD_VOCAB_SOURCE = "vendor/brown_word_counts.json (Stage 3)"


def _whole_artifact_number(name: str, value: object) -> int:
    """Return one non-negative JSON integer without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"clause-word artifact {name} must be a non-negative integer")
    return value


class ClauseWordArtifact:
    """Validated document-disjoint Brown bridge from clause type to words.

    Internal digests are insufficient on their own because a coordinated
    relabel can recompute them. Canonical loading also binds IDs/categories to
    the immutable Brown projection and binds the vocabulary, document arrays,
    and split indices to the separately validated Stage 3 receiver artifact.
    """

    _TOP_KEYS = frozenset({"schema", "provenance", "documents", "vocab", "splits"})
    _PROVENANCE_KEYS = frozenset(
        {
            "corpus",
            "raw_source",
            "type_label_source",
            "excluded_exclamative",
            "excluded_unpunctuated",
            "vocab_source",
            "vocab_source_schema",
            "vocab_source_sha256",
            "vocab_size",
            "words_dropped_out_of_vocabulary",
            "split",
            "target_train_document_fraction",
            "train_document_fraction",
            "documents",
            "train_documents",
            "heldout_documents",
            "document_overlap",
            "categories",
            "typed_sentences",
            "train_typed_sentences",
            "test_typed_sentences",
            "sentence_types",
        }
    )
    _DOCUMENT_KEYS = frozenset(
        {
            "ids",
            "categories",
            "identity_sha256",
            "partition_input_sha256",
            "split_indices",
            "typed_sentence_counts",
            "in_vocab_word_counts",
        }
    )
    _ENTRY_KEYS = frozenset({"document_index", "sentence_index", "type", "words"})

    def __init__(
        self,
        payload: Mapping[str, object],
        *,
        expected_vocab: Sequence[str],
        expected_vocab_sha256: str,
        expected_document_ids: Sequence[str],
        expected_categories: Sequence[str],
        expected_train_indices: Sequence[int],
        expected_heldout_indices: Sequence[int],
    ) -> None:
        if set(payload) != self._TOP_KEYS:
            raise ValueError(
                "clause-word artifact top-level keys disagree with schema: "
                f"missing={sorted(self._TOP_KEYS - set(payload))}, "
                f"unexpected={sorted(set(payload) - self._TOP_KEYS)}"
            )
        if payload.get("schema") != CLAUSE_WORD_ARTIFACT_SCHEMA:
            raise ValueError(f"clause-word artifact schema must be {CLAUSE_WORD_ARTIFACT_SCHEMA!r}")
        validate_brown_artifact_provenance(payload)

        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping) or set(provenance) != self._PROVENANCE_KEYS:
            keys = set(provenance) if isinstance(provenance, Mapping) else set()
            raise ValueError(
                "clause-word provenance keys disagree with schema: "
                f"missing={sorted(self._PROVENANCE_KEYS - keys)}, "
                f"unexpected={sorted(keys - self._PROVENANCE_KEYS)}"
            )
        validate_canonical_brown_partition_target(provenance.get("target_train_document_fraction"))
        exact_provenance_values = {
            "corpus": CLAUSE_WORD_CORPUS,
            "split": CLAUSE_WORD_SPLIT,
            "type_label_source": CLAUSE_WORD_TYPE_LABEL_SOURCE,
            "vocab_source": CLAUSE_WORD_VOCAB_SOURCE,
        }
        mismatches = {
            name: (expected, provenance.get(name))
            for name, expected in exact_provenance_values.items()
            if provenance.get(name) != expected
        }
        if mismatches:
            raise ValueError(
                f"clause-word artifact provenance constants disagree with the schema: {mismatches}"
            )
        if provenance.get("split") != CLAUSE_WORD_SPLIT:
            raise ValueError("clause-word artifact must declare the canonical document split")
        if provenance.get("sentence_types") != list(CLAUSE_TYPES):
            raise ValueError("clause-word artifact sentence_types disagree with the typed contract")
        if provenance.get("vocab_source_schema") != "ailm-word-counts-v2":
            raise ValueError("clause-word vocabulary source schema is not the Stage 3 artifact")
        if provenance.get("vocab_source_sha256") != expected_vocab_sha256:
            raise ValueError("clause-word vocabulary source digest disagrees with Stage 3")

        vocab = payload.get("vocab")
        if (
            not isinstance(vocab, list)
            or any(not isinstance(word, str) or not word for word in vocab)
            or len(vocab) != len(set(vocab))
        ):
            raise ValueError("clause-word artifact vocab must be unique non-empty strings")
        if tuple(vocab) != tuple(expected_vocab):
            raise ValueError(
                "clause-word vocabulary does not equal the Stage 3 receiver vocabulary"
            )
        if _whole_artifact_number("vocab_size", provenance.get("vocab_size")) != len(vocab):
            raise ValueError("clause-word provenance vocab_size disagrees with vocab")

        documents = payload.get("documents")
        if not isinstance(documents, Mapping) or set(documents) != self._DOCUMENT_KEYS:
            keys = set(documents) if isinstance(documents, Mapping) else set()
            raise ValueError(
                "clause-word document keys disagree with schema: "
                f"missing={sorted(self._DOCUMENT_KEYS - keys)}, "
                f"unexpected={sorted(keys - self._DOCUMENT_KEYS)}"
            )
        ids = documents.get("ids")
        categories = documents.get("categories")
        if (
            not isinstance(ids, list)
            or not isinstance(categories, list)
            or tuple(ids) != tuple(expected_document_ids)
            or tuple(categories) != tuple(expected_categories)
        ):
            raise ValueError("clause-word document identities/categories disagree with Stage 3")
        validate_canonical_brown_document_metadata(ids, categories)
        if documents.get("identity_sha256") != brown_document_identity_sha256(ids):
            raise ValueError("clause-word document identity digest is inconsistent")
        if documents.get("partition_input_sha256") != brown_partition_input_sha256(ids, categories):
            raise ValueError("clause-word document partition-input digest is inconsistent")

        split_indices = documents.get("split_indices")
        if not isinstance(split_indices, Mapping) or set(split_indices) != {"train", "heldout"}:
            raise ValueError("clause-word split_indices must contain exactly train/heldout")
        train_indices = split_indices.get("train")
        heldout_indices = split_indices.get("heldout")
        if not isinstance(train_indices, list) or not isinstance(heldout_indices, list):
            raise ValueError("clause-word split indices must be lists")
        if tuple(train_indices) != tuple(expected_train_indices) or tuple(heldout_indices) != tuple(
            expected_heldout_indices
        ):
            raise ValueError("clause-word document split disagrees with the Stage 3 receiver")
        canonical_train, canonical_heldout = category_stratified_document_indices(
            ids, categories, BROWN_TRAIN_DOCUMENT_FRACTION
        )
        if tuple(train_indices) != canonical_train or tuple(heldout_indices) != canonical_heldout:
            raise ValueError("clause-word document split disagrees with the canonical rule")
        train_set, heldout_set = set(train_indices), set(heldout_indices)
        if train_set & heldout_set or train_set | heldout_set != set(range(len(ids))):
            raise ValueError("clause-word document split must be exhaustive and disjoint")

        typed_counts = documents.get("typed_sentence_counts")
        word_counts = documents.get("in_vocab_word_counts")
        if (
            not isinstance(typed_counts, list)
            or not isinstance(word_counts, list)
            or len(typed_counts) != len(ids)
            or len(word_counts) != len(ids)
        ):
            raise ValueError("clause-word per-document counts must align with document identities")
        typed_counts = [
            _whole_artifact_number(f"typed_sentence_counts[{index}]", value)
            for index, value in enumerate(typed_counts)
        ]
        word_counts = [
            _whole_artifact_number(f"in_vocab_word_counts[{index}]", value)
            for index, value in enumerate(word_counts)
        ]
        typed_expected = [0] * len(ids)
        words_expected = [0] * len(ids)

        splits = payload.get("splits")
        if not isinstance(splits, Mapping) or set(splits) != {"train", "test"}:
            raise ValueError("clause-word splits must contain exactly train/test")
        checked: dict[str, list[dict[str, Any]]] = {}
        types_by_split: dict[str, set[str]] = {}
        seen_coordinates: set[tuple[int, int]] = set()
        vocab_set = set(vocab)
        for split_name, allowed in (("train", train_set), ("test", heldout_set)):
            entries = splits.get(split_name)
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"clause-word {split_name} split must be a non-empty list")
            previous_coordinate: tuple[int, int] | None = None
            checked_entries: list[dict[str, Any]] = []
            observed_types: set[str] = set()
            for entry in entries:
                if not isinstance(entry, Mapping) or set(entry) != self._ENTRY_KEYS:
                    raise ValueError("clause-word sentence rows must match the exact schema")
                document_index = entry.get("document_index")
                sentence_index = entry.get("sentence_index")
                if (
                    isinstance(document_index, bool)
                    or not isinstance(document_index, int)
                    or document_index not in allowed
                    or isinstance(sentence_index, bool)
                    or not isinstance(sentence_index, int)
                    or sentence_index < 0
                ):
                    raise ValueError(
                        f"clause-word {split_name} sentence is mapped outside its document arm"
                    )
                coordinate = (document_index, sentence_index)
                if coordinate in seen_coordinates or (
                    previous_coordinate is not None and coordinate <= previous_coordinate
                ):
                    raise ValueError("clause-word sentence coordinates must be unique and ordered")
                seen_coordinates.add(coordinate)
                previous_coordinate = coordinate
                sentence_type = entry.get("type")
                words = entry.get("words")
                if sentence_type not in CLAUSE_TYPES:
                    raise ValueError("clause-word sentence has an unknown type")
                observed_types.add(sentence_type)
                if not isinstance(words, list) or any(
                    not isinstance(word, str) or word not in vocab_set for word in words
                ):
                    raise ValueError("clause-word sentence contains a word outside Stage 3 vocab")
                typed_expected[document_index] += 1
                words_expected[document_index] += len(words)
                checked_entries.append(dict(entry))
            checked[split_name] = checked_entries
            types_by_split[split_name] = observed_types

        for split_name, observed_types in types_by_split.items():
            if observed_types != set(CLAUSE_TYPES):
                raise ValueError(
                    f"clause-word {split_name} split must represent every sentence type"
                )

        if typed_counts != typed_expected or word_counts != words_expected:
            raise ValueError("clause-word per-document counts disagree with sentence rows")

        source_documents = len(ids)
        train_documents = len(train_indices)
        heldout_documents = len(heldout_indices)
        expected_provenance_counts = {
            "documents": source_documents,
            "train_documents": train_documents,
            "heldout_documents": heldout_documents,
            "document_overlap": 0,
            "categories": len(set(categories)),
            "typed_sentences": sum(typed_expected),
            "train_typed_sentences": len(checked["train"]),
            "test_typed_sentences": len(checked["test"]),
        }
        for name, expected in expected_provenance_counts.items():
            if _whole_artifact_number(name, provenance.get(name)) != expected:
                raise ValueError(f"clause-word provenance {name} disagrees with artifact rows")
        actual_fraction = train_documents / source_documents
        declared_fraction = provenance.get("train_document_fraction")
        if (
            isinstance(declared_fraction, bool)
            or not isinstance(declared_fraction, (int, float))
            or not np.isfinite(float(declared_fraction))
            or float(declared_fraction) != actual_fraction
        ):
            raise ValueError("clause-word train_document_fraction disagrees with document counts")
        for name in (
            "excluded_exclamative",
            "excluded_unpunctuated",
            "words_dropped_out_of_vocabulary",
        ):
            _whole_artifact_number(name, provenance.get(name))

        self.provenance = dict(provenance)
        self.vocab = list(vocab)
        self.document_ids = list(ids)
        self.document_categories = list(categories)
        self.train_document_indices = list(train_indices)
        self.heldout_document_indices = list(heldout_indices)
        self.train = checked["train"]
        self.test = checked["test"]


def load_clause_word_artifact(
    path: Path | str | None = None,
    *,
    word_path: Path | str | None = None,
) -> ClauseWordArtifact:
    """Load the bridge and bind it to the separately validated Stage 3 artifact."""
    resolved = resolve_vendored_artifact("brown_clause_word_counts.json", path=path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"{resolved} is absent. Run scripts/generate_clause_word_counts.py."
        )
    word_resource = resolve_vendored_artifact("brown_word_counts.json", path=word_path)
    if not word_resource.is_file():
        raise FileNotFoundError(f"{word_resource} is absent. Run scripts/generate_word_counts.py.")

    from ailm.levels.word import load_word_artifact

    word_artifact = load_word_artifact(word_path)
    word_digest = hashlib.sha256(word_resource.read_bytes()).hexdigest()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("clause-word artifact must be a JSON object")
    return ClauseWordArtifact(
        payload,
        expected_vocab=word_artifact.vocab,
        expected_vocab_sha256=word_digest,
        expected_document_ids=word_artifact.document_ids,
        expected_categories=word_artifact.document_categories,
        expected_train_indices=word_artifact.train_document_indices,
        expected_heldout_indices=word_artifact.heldout_document_indices,
    )


class ClauseMessage:
    """``P(word | sentence type)`` tables estimated from train sentences.

    Attributes:
        by_type: ``type -> (n_vocab,)`` word distribution.
        types: The ordered type labels.
        fallback: The context-free (marginal) word distribution, used when a
            type in the message request has no estimated distribution.
        n_vocab: Vocabulary size.
    """

    def __init__(
        self,
        by_type: dict[str, NDArray[np.float64]],
        types: list[str],
        fallback: NDArray[np.float64],
        n_vocab: int,
    ) -> None:
        self.by_type = by_type
        self.types = list(types)
        self.fallback = fallback
        self.n_vocab = n_vocab

    def lookup(self, sentence_type: str) -> NDArray[np.float64]:
        """Distribution for ``sentence_type``, or the fallback when unseen."""
        return self.by_type.get(sentence_type, self.fallback)


def build_clause_message(
    train_sentences: list[dict],
    word_to_index: dict[str, int],
    *,
    smoothing: float = 1.0,
    types: list[str] | None = None,
) -> ClauseMessage:
    """Estimate ``P(word | type)`` from the training sentences.

    Args:
        train_sentences: ``[{"type": t, "words": [w, ...]}]`` from the bridge
            artifact's train split.
        word_to_index: Stage 3 vocabulary map.
        smoothing: Add-k pseudocount over the vocabulary.
        types: Ordered type labels; defaults to those seen in ``train_sentences``.

    Returns:
        The estimated message.

    Raises:
        ValueError: If ``train_sentences`` is empty or ``smoothing`` is not > 0.
    """
    if not train_sentences:
        raise ValueError("train_sentences is empty; there is nothing to estimate")
    if smoothing <= 0.0:
        raise ValueError(f"smoothing must be positive, got {smoothing}")

    n_vocab = len(word_to_index)
    counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_vocab, dtype=np.float64))
    seen: dict[str, int] = defaultdict(int)
    overall = np.zeros(n_vocab, dtype=np.float64)

    for sentence in train_sentences:
        sentence_type = sentence["type"]
        seen[sentence_type] += 1
        for word in sentence["words"]:
            slot = word_to_index[word]
            overall[slot] += 1.0
            counts[sentence_type][slot] += 1.0

    by_type = {}
    for sentence_type, row in counts.items():
        smoothed = row + smoothing
        by_type[sentence_type] = smoothed / smoothed.sum()
    fallback = (overall + smoothing) / (overall + smoothing).sum()

    ordered = list(types) if types is not None else sorted(by_type)
    return ClauseMessage(by_type, ordered, fallback, n_vocab)


def message_for_type(message: ClauseMessage, sentence_type: str) -> NDArray[np.float64]:
    """The ``P(word | type)`` message for ``sentence_type``."""
    return message.lookup(sentence_type)


def mix_prior(
    prior: NDArray[np.float64],
    message: NDArray[np.float64],
    alpha: float,
) -> NDArray[np.float64]:
    """Convex mix of the uncoupled prior and the clause message.

    At ``alpha = 0`` the result is bit-identical to ``prior`` (the identity
    anchor); at ``alpha = 1`` it is exactly the message. The mix of two
    distributions is itself a distribution, so no renormalization is applied.

    Both inputs are validated as finite, nonnegative distributions summing to
    one within :data:`MIX_NORMALIZATION_ATOL`: a message (or prior) whose
    masses do not sum to one was silently accepted before, and every
    downstream reading of the mixed vector — perplexities, log gains,
    per-token effective-α shrink — assumes a normalized operand. The returned
    expression is unchanged, so the alpha-zero bit-identity anchor and every
    valid caller's result are unchanged.

    Raises:
        ValueError: If ``alpha`` is outside ``[0, 1]``, the shapes mismatch,
            or an input is nonfinite, negative, or unnormalized.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1], got {alpha!r}")
    prior = np.asarray(prior, dtype=np.float64)
    message = np.asarray(message, dtype=np.float64)
    if prior.shape != message.shape:
        raise ValueError(
            f"prior and message must share shape, got {prior.shape} and {message.shape}"
        )
    for name, operand in (("prior", prior), ("message", message)):
        if not np.all(np.isfinite(operand)):
            raise ValueError(f"{name} must be finite, got a nonfinite entry")
        if np.any(operand < 0.0):
            raise ValueError(f"{name} must be nonnegative")
        total = float(operand.sum())
        if abs(total - 1.0) > MIX_NORMALIZATION_ATOL:
            raise ValueError(
                f"{name} must sum to 1 within {MIX_NORMALIZATION_ATOL}, got {total!r}; "
                f"the convex mix would silently rescale the other operand"
            )
    return (1.0 - alpha) * prior + alpha * message


def scramble_word_distribution(
    distribution: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Reverse the word axis — the entropy-preserving negative control.

    A fixed involutive permutation, so the scrambled message has identical
    entropy, peakedness and normalization to the real one; only its
    correspondence to the word inventory is destroyed. Replacing it with a
    uniform distribution instead would confound content with entropy (entry 9).
    """
    return np.asarray(distribution, dtype=np.float64)[::-1].copy()


def sweep_clause_alpha(
    test_sentences: list[dict],
    prior: NDArray[np.float64],
    message: ClauseMessage,
    word_to_index: dict[str, int],
    grid: tuple[float, ...],
    *,
    scrambled: bool = False,
) -> dict[float, list[float]]:
    """Per-token log-probability for every ``alpha`` in ``grid``.

    Returns a dict ``alpha -> [log P_alpha(token)]`` aligned position-for-
    position across alphas (each list indexes the same token stream), so any
    two weights can be paired. The token stream is the flattened held-out
    in-vocabulary words of ``test_sentences``.

    Raises:
        ValueError: If ``grid`` lacks ``0.0`` or fewer than three positive
            values — a collapsed sweep cannot support a "no alpha beats the
            anchor" finding (entry 9's grid guard).
    """
    positives = {a for a in grid if a > 0.0}
    if 0.0 not in grid or len(positives) < 3:
        raise ValueError(
            f"grid must contain 0.0 plus at least three distinct positive "
            f"alphas so a null cannot come from a collapsed sweep; got {grid!r}"
        )

    results: dict[float, list[float]] = {a: [] for a in grid}
    LOG_FLOOR = 1e-300
    for sentence in test_sentences:
        sentence_type = sentence["type"]
        message_dist = (
            message.lookup(sentence_type)
            if not scrambled
            else scramble_word_distribution(message.lookup(sentence_type))
        )
        for word in sentence["words"]:
            index = word_to_index[word]
            for alpha in grid:
                mixed = mix_prior(prior, message_dist, alpha)
                results[alpha].append(float(np.log(max(mixed[index], LOG_FLOOR))))
    return results


def clustered_clause_word_summaries(
    test_sentences: list[dict[str, Any]],
    selected_log_probs: Sequence[float],
    anchor_log_probs: Sequence[float],
) -> tuple[ClusteredPairedSummary, ClusteredPairedSummary]:
    """Whole-document uncertainty for the token direction and mean log gain.

    The bridge nests tokens inside Brown source documents. Treating those
    tokens as independent makes repeated within-document language patterns look
    like independent replication. This function preserves the two descriptive
    estimands while resampling and sign-flipping only whole source documents.

    The token summary uses ``+1/-1/0`` for improved/degraded/tied word
    probabilities. The log-gain summary expands each word's log-probability
    difference across its ``len(word) + 1`` emissions, so its observation-
    weighted mean is exactly ``sum(log q_selected - log q_anchor) / emissions``.
    """
    from ailm.clustering import summarize_clustered_difference

    if len(selected_log_probs) != len(anchor_log_probs):
        raise ValueError("selected and anchor log-probability streams must align")

    token_directions: list[float] = []
    token_documents: list[int] = []
    emission_gains: list[float] = []
    emission_documents: list[int] = []
    cursor = 0
    for sentence in test_sentences:
        document_index = sentence.get("document_index")
        words = sentence.get("words")
        if (
            isinstance(document_index, bool)
            or not isinstance(document_index, int)
            or not isinstance(words, list)
        ):
            raise ValueError("clustered clause-word rows need a document index and word list")
        for word in words:
            if not isinstance(word, str):
                raise ValueError("clustered clause-word rows contain a non-string word")
            if cursor >= len(selected_log_probs):
                raise ValueError(
                    "sentence rows contain more tokens than the log-probability stream"
                )
            difference = float(selected_log_probs[cursor] - anchor_log_probs[cursor])
            direction = float((difference > 0.0) - (difference < 0.0))
            token_directions.append(direction)
            token_documents.append(document_index)
            emissions = len(word) + 1
            per_emission = difference / emissions
            emission_gains.extend([per_emission] * emissions)
            emission_documents.extend([document_index] * emissions)
            cursor += 1
    if cursor != len(selected_log_probs):
        raise ValueError("log-probability stream contains tokens absent from the sentence rows")
    if len(set(token_documents)) < 2:
        raise ValueError("clustered clause-word inference needs at least two source documents")

    token_summary = summarize_clustered_difference(token_directions, token_documents)
    emission_summary = summarize_clustered_difference(emission_gains, emission_documents)
    return token_summary, emission_summary
