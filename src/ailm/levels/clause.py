"""Clause-level (Stage 5, L5) generative model: sentence type over case roles.

What this level is
------------------
A clause is to case roles what a word is to letters: a latent discrete variable
— the sentence type, declarative or interrogative, labeled by sentence-final
punctuation — conditioning a sequence over the level below's alphabet. The
model is a two-component mixture of case-role Markov chains, embedded as a
single-factor pymdp agent over the joint (type × case) state space:

  - 16 hidden states, type-major: state ``t * 8 + c`` is (type ``t``, case ``c``)
  - ``B`` block-diagonal: within type ``t`` the type-conditioned case bigram;
    zero cross-type mass — a sentence does not change type mid-way
  - ``A`` deterministic: the observed case role is the case component
  - ``D``: P(type) ⊗ P(first case | type)

Filtering a case sequence therefore accumulates evidence about the sentence
type, and the type posterior is the marginal over blocks.

What the measurement found — read this before extending the level
-----------------------------------------------------------------
The Stage 5 gate is **non-passing and unresolved** (``docs/decisions.md``
entry 83). On this UD EWT split at bigram order, the privileged
type-*oracle* reference has a small negative point estimate against the type-free
chain, and its document-clustered bootstrap intervals span zero in nats per
sentence and in the pooled per-position ratio; the mixture is slightly
*worse* than type-free on held-out data. That difference is consistent with
limited-sample estimation variance, but this design does not isolate variance
from smoothing, misspecification, or distribution shift. Classification recovers none of the held-out interrogatives
against a claim:stage5_majority_baseline_pct% majority baseline. The
instrument is not at fault: on a synthetic corpus whose types genuinely differ
the same harness classifies near-perfectly and the mixture beats type-free
decisively. That detectability control rules out a generically dead instrument;
it does not establish equivalence or negligible predictive value at the shipped
operating point. The machinery ships; this corpus and model order do not resolve
a predictive gain for the mixture.

The disjointness inherited from Stage 4
---------------------------------------
The type label's source is the sentence-final punctuation token; observations
are derived case roles; PUNCT tokens carry no case role, so the label source
cannot appear in the observation stream (asserted at artifact build time and
re-checked in ``tests/test_stage5_gate.py``).
"""

from __future__ import annotations

import json
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import rankdata

from ailm.cases import CaseRole
from ailm.config import DEFAULT_LEVEL_CONFIG, LevelConfig
from ailm.corpus.ud import document_identity_sha256, validate_ud_ewt_source_metadata
from ailm.levels._stochastic import column_stochastic
from ailm.resources import resolve_vendored_artifact

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pymdp.agent import Agent as PymdpAgent

__all__ = [
    "CLAUSE_TYPES",
    "CLAUSE_ARTIFACT_SCHEMA",
    "DEFAULT_ARTIFACT",
    "N_CASES",
    "N_JOINT",
    "N_TYPES",
    "ClauseArtifact",
    "ClusteredAucBootstrapSummary",
    "OracleGainBootstrapSummary",
    "STAGE5_BOOTSTRAP_CONFIDENCE",
    "STAGE5_BOOTSTRAP_RESAMPLES",
    "STAGE5_BOOTSTRAP_SEED",
    "build_clause_agent",
    "build_joint_emission",
    "build_joint_start",
    "build_joint_transition",
    "build_type_free_chain",
    "build_type_models",
    "binary_auc",
    "clustered_binary_auc_interval",
    "classify_sentences",
    "load_clause_artifact",
    "proper_scores",
    "summarize_oracle_gain",
    "sequence_log_evidence",
    "synthetic_control_corpus",
    "type_conditional_perplexities",
]

#: The two labeled sentence types, in artifact order. Exclamative and
#: unpunctuated sentences are excluded by the generator, with counts recorded.
CLAUSE_TYPES: tuple[str, ...] = ("declarative", "interrogative")
CLAUSE_ARTIFACT_SCHEMA = "ailm-clause-counts-v2"

N_TYPES: int = len(CLAUSE_TYPES)
N_CASES: int = len(CaseRole)
N_JOINT: int = N_TYPES * N_CASES

#: One resampling contract for every Stage 5 bootstrap quantity. These values
#: were first declared for the AUC interval; the oracle-gain interval uses the
#: same held-out documents and therefore shares the same draw count, seed and
#: coverage rather than introducing a second, visually indistinguishable
#: convention. Recorders persist all three as claims. The coverage is
#: deliberately not ``ailm.statistics.DIFFERENCE_CONFIDENCE``: these are
#: percentile intervals rather than the share interval, and decision 66 kept
#: them out of the coverage harmonization on that ground, so the divergence is
#: visible here at the constant rather than discovered between two recorders.
STAGE5_BOOTSTRAP_RESAMPLES: int = 10_000
STAGE5_BOOTSTRAP_SEED: int = 20260804
STAGE5_BOOTSTRAP_CONFIDENCE: float = 0.99

#: Bound the peak allocation without changing the RNG draw sequence. The
#: canonical evaluation has hundreds of sentences, but callers may use this
#: production helper on larger sentence sets.
_BOOTSTRAP_BLOCK: int = 256

#: Case-role name -> observation index, in CaseRole declaration order.
CASE_TO_IDX: dict[str, int] = {role.name: i for i, role in enumerate(CaseRole)}

DEFAULT_ARTIFACT = resolve_vendored_artifact("ud_ewt_clause_counts.json")


@dataclass(frozen=True, slots=True)
class OracleGainBootstrapSummary:
    """Document-clustered bound on the privileged-label reference gain.

    The paired observation is one held-out sentence's log evidence under the
    fitted chain selected by its reference type label minus its log evidence
    under the type-free chain. Two estimands are retained from the same paired
    draws:

    - ``mean_nats_per_sentence`` gives every sampled sentence equal weight;
    - ``pooled_nats_per_position`` divides the resampled sum of gains by the
      resampled sum of sentence lengths. That ratio matches the position-weighted
      perplexity comparison already published by :func:`type_conditional_perplexities`.

    Resampling individual sentences would still be anti-conservative on EWT:
    multiple sentences share one source document. Whole documents are the
    primary resampling unit while the effect remains expressed per sentence
    and per position.
    """

    n_sentences: int
    n_documents: int
    n_positions: int
    mean_nats_per_sentence: float
    sentence_ci_low: float
    sentence_ci_high: float
    pooled_nats_per_position: float
    position_ci_low: float
    position_ci_high: float
    confidence: float
    resamples: int
    seed: int


@dataclass(frozen=True, slots=True)
class ClusteredAucBootstrapSummary:
    """Whole-document percentile interval for a sentence-level binary AUC."""

    auc: float
    ci_low: float
    ci_high: float
    n_observations: int
    n_documents: int
    confidence: float
    resamples: int
    seed: int


class ClauseArtifact:
    """The vendored Stage 5 statistics, validated on load.

    Attributes:
        train: ``(type, [case indices])`` training sentences.
        test: Held-out sentences, same shape.
        provenance: The generator's recorded parameters.
    """

    def __init__(self, payload: dict) -> None:
        if payload.get("schema") != CLAUSE_ARTIFACT_SCHEMA:
            raise ValueError(
                f"Unexpected artifact schema {payload.get('schema')!r}; "
                f"regenerate with scripts/generate_clause_counts.py"
            )
        self.provenance: dict = payload["provenance"]
        self.train = self._validate_split(payload["splits"]["train"], "train")
        self.test = self._validate_split(payload["splits"]["test"], "test")
        (
            self.document_ids,
            split_document_indices,
            sequence_document_indices,
        ) = self._validate_documents(payload.get("documents"), payload.get("splits"))
        self.train_split_document_indices = split_document_indices["train"]
        self.test_split_document_indices = split_document_indices["test"]
        self.train_sequence_document_indices = sequence_document_indices["train"]
        self.test_sequence_document_indices = sequence_document_indices["test"]
        self.train_document_ids = [
            self.document_ids[index] for index in sorted(set(self.train_sequence_document_indices))
        ]
        self.test_document_ids = [
            self.document_ids[index] for index in sorted(set(self.test_sequence_document_indices))
        ]
        if self.provenance.get("document_identity_sha256") != document_identity_sha256(
            self.document_ids
        ):
            raise ValueError(
                "clause artifact document identity digest disagrees with its ordered ids"
            )
        if self.provenance.get("documents") != len(self.document_ids):
            raise ValueError("clause artifact provenance document count is inconsistent")
        if self.provenance.get("train_documents") != len(self.train_split_document_indices):
            raise ValueError("clause artifact provenance train-document count is inconsistent")
        if self.provenance.get("test_documents") != len(self.test_split_document_indices):
            raise ValueError("clause artifact provenance test-document count is inconsistent")
        train_types = {t for t, _ in self.train}
        if train_types != set(CLAUSE_TYPES):
            raise ValueError(
                f"training split carries types {sorted(train_types)}; both "
                f"{CLAUSE_TYPES} are required to estimate the mixture"
            )

    @staticmethod
    def _validate_split(entries: list, name: str) -> list[tuple[str, list[int]]]:
        if not entries:
            raise ValueError(f"{name} split is empty")
        out: list[tuple[str, list[int]]] = []
        for entry in entries:
            sentence_type = entry["type"]
            if sentence_type not in CLAUSE_TYPES:
                raise ValueError(f"unknown sentence type {sentence_type!r}")
            cases = entry["cases"]
            if len(cases) < 2:
                raise ValueError(f"{name} split carries a sequence with no transition")
            stray = next((c for c in cases if c not in CASE_TO_IDX), None)
            if stray is not None:
                # The structural guard of decision 16, one level up: any symbol
                # outside the 8 case roles could smuggle the label source in.
                raise ValueError(
                    f"observation {stray!r} is outside the case-role alphabet; "
                    f"the artifact was not built by the shipped generator"
                )
            out.append((sentence_type, [CASE_TO_IDX[c] for c in cases]))
        return out

    @property
    def n_test_documents(self) -> int:
        """Number of source documents contributing typed held-out sentences."""
        return len(self.test_document_ids)

    @staticmethod
    def _validate_documents(
        value: object, splits: object
    ) -> tuple[list[str], dict[str, list[int]], dict[str, list[int]]]:
        expected_keys = {"ids", "split_indices", "sequence_document_indices"}
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise ValueError(
                "clause artifact documents must contain ids, split_indices, and "
                "sequence_document_indices"
            )
        ids = value["ids"]
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(item, str) or not item or "\0" in item for item in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError("clause artifact document ids must be non-empty unique strings")
        if not isinstance(splits, dict) or set(splits) != {"train", "test"}:
            raise ValueError("clause artifact splits must name train and test")

        raw_split_indices = value["split_indices"]
        raw_sequence_indices = value["sequence_document_indices"]
        if (
            not isinstance(raw_split_indices, dict)
            or set(raw_split_indices) != {"train", "test"}
            or not isinstance(raw_sequence_indices, dict)
            or set(raw_sequence_indices) != {"train", "test"}
        ):
            raise ValueError("clause artifact document mappings must name train and test")
        checked_splits: dict[str, list[int]] = {}
        checked_sequences: dict[str, list[int]] = {}
        for split in ("train", "test"):
            document_indices = raw_split_indices[split]
            sequence_indices = raw_sequence_indices[split]
            if (
                not isinstance(document_indices, list)
                or not document_indices
                or any(
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < len(ids)
                    for index in document_indices
                )
                or len(document_indices) != len(set(document_indices))
            ):
                raise ValueError(f"clause artifact {split} document indices are invalid")
            allowed = set(document_indices)
            if (
                not isinstance(sequence_indices, list)
                or len(sequence_indices) != len(splits[split])
                or any(
                    isinstance(index, bool) or not isinstance(index, int) or index not in allowed
                    for index in sequence_indices
                )
            ):
                raise ValueError(f"clause artifact {split} sequence/document alignment is invalid")
            checked_splits[split] = list(document_indices)
            checked_sequences[split] = list(sequence_indices)
        if set(checked_splits["train"]) & set(checked_splits["test"]):
            raise ValueError("clause artifact train and test documents overlap")
        if set(checked_splits["train"]) | set(checked_splits["test"]) != set(range(len(ids))):
            raise ValueError("clause artifact document split does not partition its identities")
        return list(ids), checked_splits, checked_sequences


def load_clause_artifact(path: Path | str | None = None) -> ClauseArtifact:
    """Load and validate the vendored clause artifact.

    Raises:
        FileNotFoundError: If absent — Stage 5 has no fallback corpus.
        ValueError: On validation failure.
    """
    resolved = (
        DEFAULT_ARTIFACT
        if path is None
        else resolve_vendored_artifact("ud_ewt_clause_counts.json", path=path)
    )
    if not resolved.is_file():
        raise FileNotFoundError(f"{resolved} is absent. Run scripts/generate_clause_counts.py.")
    artifact = ClauseArtifact(json.loads(resolved.read_text(encoding="utf-8")))
    validate_ud_ewt_source_metadata(artifact.provenance)
    return artifact


# ── ESTIMATION ───────────────────────────────────────────────────────────────


def build_type_models(
    sentences: list[tuple[str, list[int]]],
    *,
    config: LevelConfig = DEFAULT_LEVEL_CONFIG,
    n_symbols: int = N_CASES,
) -> dict:
    """Per-type observation chains plus the type prior, add-k smoothed.

    Args:
        sentences: Typed observation-index sequences.
        config: Hyperparameters (add-k pseudocount).
        n_symbols: Observation alphabet size. Defaults to the shipped
            case-role alphabet, so every existing caller is byte-identical.
            The widened arm of decision 112 passes its own alphabet size here
            rather than transcribing this estimator a second time; a second
            copy is how the alphabet came to be written down three times
            before ``ailm.levels.widened`` collected it.

    Returns:
        ``transitions``: ``(N_TYPES, n_symbols, n_symbols)`` column-stochastic
        per type; ``starts``: ``(N_TYPES, n_symbols)``; ``type_prior``:
        ``(N_TYPES,)``.
    """
    k = float(config.smoothing)
    transitions = np.full((N_TYPES, n_symbols, n_symbols), k, dtype=np.float64)
    starts = np.full((N_TYPES, n_symbols), k, dtype=np.float64)
    type_counts = np.full(N_TYPES, k, dtype=np.float64)
    for sentence_type, sequence in sentences:
        t = CLAUSE_TYPES.index(sentence_type)
        type_counts[t] += 1.0
        starts[t, sequence[0]] += 1.0
        for prev, cur in pairwise(sequence):
            transitions[t, cur, prev] += 1.0
    transitions /= transitions.sum(axis=1, keepdims=True)
    starts /= starts.sum(axis=1, keepdims=True)
    return {
        "transitions": transitions,
        "starts": starts,
        "type_prior": type_counts / type_counts.sum(),
    }


def build_type_free_chain(
    sentences: list[tuple[str, list[int]]],
    *,
    config: LevelConfig = DEFAULT_LEVEL_CONFIG,
    n_symbols: int = N_CASES,
) -> tuple[np.ndarray, np.ndarray]:
    """The clause-blind comparator: one pooled chain over all types.

    This is decision 2 applied to the new variable: the honest bar is the
    strongest model that does not use *type*, not a uniform strawman. It must
    be refitted on whatever stream is being scored — a bar carried over from a
    different alphabet would credit the new alphabet with the refit, which is
    why ``n_symbols`` is a parameter here rather than a module constant.
    """
    k = float(config.smoothing)
    transition = np.full((n_symbols, n_symbols), k, dtype=np.float64)
    start = np.full(n_symbols, k, dtype=np.float64)
    for _sentence_type, sequence in sentences:
        start[sequence[0]] += 1.0
        for prev, cur in pairwise(sequence):
            transition[cur, prev] += 1.0
    return column_stochastic(transition), start / start.sum()


# ── THE JOINT pymdp AGENT ────────────────────────────────────────────────────


def build_joint_emission(n_symbols: int = N_CASES) -> np.ndarray:
    """``A``: ``(n_symbols, N_TYPES * n_symbols)`` — observed IS the symbol part.

    Args:
        n_symbols: Observation alphabet size, defaulting to the shipped
            case-role alphabet.
    """
    emission = np.zeros((n_symbols, N_TYPES * n_symbols), dtype=np.float64)
    for t in range(N_TYPES):
        for c in range(n_symbols):
            emission[c, t * n_symbols + c] = 1.0
    return emission


def build_joint_transition(models: dict) -> np.ndarray:
    """``B``: block-diagonal joint transition — type persists exactly.

    Zero cross-type mass is the structural claim of the level; any leakage
    lets the model change its mind about the sentence type mid-sentence,
    which is the clause-level analogue of a word drifting mid-spelling.

    The alphabet size is read off the fitted models rather than taken as a
    second argument: a block size that disagreed with the blocks being written
    into it is not a configuration this function should be able to express.
    """
    n_symbols = int(models["transitions"].shape[1])
    n_joint = N_TYPES * n_symbols
    joint = np.zeros((n_joint, n_joint), dtype=np.float64)
    for t in range(N_TYPES):
        block = slice(t * n_symbols, (t + 1) * n_symbols)
        joint[block, block] = models["transitions"][t]
    return joint


def build_joint_start(models: dict) -> np.ndarray:
    """``D``: ``P(type) ⊗ P(first symbol | type)``, length ``N_TYPES * n_symbols``."""
    return (models["type_prior"][:, None] * models["starts"]).reshape(-1)


def build_clause_agent(
    artifact: ClauseArtifact,
    *,
    config: LevelConfig = DEFAULT_LEVEL_CONFIG,
    transition: np.ndarray | None = None,
    start: np.ndarray | None = None,
) -> PymdpAgent:
    """Build the Stage 5 pymdp agent from the vendored statistics.

    Args:
        artifact: The vendored statistics.
        config: Hyperparameters (add-k pseudocount).
        transition: Override for the joint ``B`` — negative controls inject
            mis-wired matrices through the shipped builder (decision 15's
            discipline).
        start: Override for the joint ``D``.

    Raises:
        ValueError: If an override has the wrong shape.
    """
    import jax.numpy as jnp
    from pymdp.agent import Agent

    models = build_type_models(artifact.train, config=config)
    b = (
        build_joint_transition(models)
        if transition is None
        else np.asarray(transition, dtype=np.float64)
    )
    d = build_joint_start(models) if start is None else np.asarray(start, dtype=np.float64)
    if b.shape != (N_JOINT, N_JOINT):
        raise ValueError(f"transition must be ({N_JOINT}, {N_JOINT}), got {b.shape}")
    if d.shape != (N_JOINT,):
        raise ValueError(f"start must be ({N_JOINT},), got {d.shape}")

    return Agent(
        A=[jnp.array(build_joint_emission(), dtype=jnp.float32)[jnp.newaxis]],
        B=[jnp.array(b, dtype=jnp.float32)[jnp.newaxis, :, :, jnp.newaxis]],
        C=[jnp.zeros((1, N_CASES))],
        D=[jnp.array(d, dtype=jnp.float32)[jnp.newaxis]],
        num_controls=[1],
        batch_size=1,
    )


# ── SCORING, in closed form ──────────────────────────────────────────────────


def sequence_log_evidence(sequence: list[int], models: dict) -> tuple[float, np.ndarray]:
    """Mixture log evidence of one case sequence, plus the type posterior.

    ``log P(seq) = logsumexp_t [ log P(t) + log P(seq | t) ]``. This scorer is
    the exact closed-form mixture calculation. ``build_clause_agent`` stores the
    same starts and transitions in a live pymdp Agent; the tests verify the
    matrix mapping and an independent manual array walk, not live pymdp
    filtering equivalence for the headline score.
    """
    per_type = np.empty(N_TYPES, dtype=np.float64)
    for t in range(N_TYPES):
        ll = float(np.log(models["starts"][t, sequence[0]]))
        for prev, cur in pairwise(sequence):
            ll += float(np.log(models["transitions"][t, cur, prev]))
        per_type[t] = ll
    weighted = np.log(models["type_prior"]) + per_type
    top = float(np.max(weighted))
    log_evidence = top + float(np.log(np.exp(weighted - top).sum()))
    posterior = np.exp(weighted - log_evidence)
    return log_evidence, posterior


def type_conditional_perplexities(
    sentences: list[tuple[str, list[int]]], models: dict, comparator: tuple
) -> dict:
    """Per-position perplexity of mixture, type-free bar, and type oracle.

    The type oracle scores every sentence under the chain selected by its
    privileged reference label. It is a sensitivity probe for the fitted model,
    not a hard information bound: under misspecification, a mixture can assign
    more probability than the label-selected component.
    """
    free_transition, free_start = comparator
    mixture_ll = free_ll = oracle_ll = 0.0
    n_positions = 0
    for sentence_type, sequence in sentences:
        n_positions += len(sequence)
        log_evidence, _ = sequence_log_evidence(sequence, models)
        mixture_ll += log_evidence
        ll = float(np.log(free_start[sequence[0]]))
        for prev, cur in pairwise(sequence):
            ll += float(np.log(free_transition[cur, prev]))
        free_ll += ll
        t = CLAUSE_TYPES.index(sentence_type)
        ll = float(np.log(models["starts"][t, sequence[0]]))
        for prev, cur in pairwise(sequence):
            ll += float(np.log(models["transitions"][t, cur, prev]))
        oracle_ll += ll
    return {
        "mixture": float(np.exp(-mixture_ll / n_positions)),
        "type_free": float(np.exp(-free_ll / n_positions)),
        "type_oracle": float(np.exp(-oracle_ll / n_positions)),
        "positions": n_positions,
    }


def _validated_bootstrap_contract(
    *, confidence: float, resamples: int, seed: int
) -> tuple[float, int, int]:
    """Validate and normalize one deterministic bootstrap contract."""
    if isinstance(confidence, bool):
        raise ValueError("confidence must be a finite probability strictly between 0 and 1")
    try:
        level = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "confidence must be a finite probability strictly between 0 and 1"
        ) from exc
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError(
            f"confidence must be a finite probability strictly between 0 and 1, got {confidence!r}"
        )
    if isinstance(resamples, bool) or not isinstance(resamples, (int, np.integer)):
        raise ValueError(f"resamples must be a positive integer, got {resamples!r}")
    draws = int(resamples)
    if draws <= 0:
        raise ValueError(f"resamples must be a positive integer, got {resamples!r}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    normalized_seed = int(seed)
    if normalized_seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    return level, draws, normalized_seed


def _cluster_index(
    clusters: Sequence[Hashable] | None, n_observations: int
) -> tuple[list[Hashable], np.ndarray]:
    """Validate aligned cluster keys and return first-seen keys plus inverse indices."""
    raw: Sequence[Hashable] = range(n_observations) if clusters is None else clusters
    if len(raw) != n_observations:
        raise ValueError(
            f"cluster ids must align one-for-one with {n_observations} observations, got {len(raw)}"
        )
    keys: list[Hashable] = []
    index_by_key: dict[Hashable, int] = {}
    inverse = np.empty(n_observations, dtype=np.int64)
    for observation_index, key in enumerate(raw):
        if key is None or isinstance(key, (bool, np.bool_)):
            raise ValueError(f"cluster id at observation {observation_index} is invalid: {key!r}")
        try:
            cluster_index = index_by_key.get(key)
            if cluster_index is None:
                cluster_index = len(keys)
                index_by_key[key] = cluster_index
                keys.append(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"cluster id at observation {observation_index} is not hashable: {key!r}"
            ) from exc
        inverse[observation_index] = cluster_index
    if len(keys) < 2:
        raise ValueError("cluster bootstrap needs at least two independent documents")
    return keys, inverse


def _sequence_component_log_evidence(
    sequence: list[int],
    start: np.ndarray,
    transition: np.ndarray,
    *,
    name: str,
    n_symbols: int = N_CASES,
    symbol_kind: str = "case-role",
) -> float:
    """Log evidence under one fitted chain, refusing invalid probabilities.

    ``symbol_kind`` names the alphabet in the refusal messages. It is a
    parameter rather than a fixed noun so the widened arm's failures say what
    they are: an error that calls a ``VERB`` symbol a case role sends the
    reader to the wrong module.
    """
    if not sequence:
        raise ValueError(
            f"each evaluation sentence must contain at least one {symbol_kind} position"
        )
    if start.shape != (n_symbols,) or transition.shape != (n_symbols, n_symbols):
        raise ValueError(
            f"{name} must provide start {(n_symbols,)} and transition "
            f"{(n_symbols, n_symbols)}, got {start.shape} and {transition.shape}"
        )
    _validate_probability_chain(start, transition, name=name)
    invalid_role = next(
        (
            role
            for role in sequence
            if isinstance(role, (bool, np.bool_))
            or not isinstance(role, (int, np.integer))
            or not 0 <= int(role) < n_symbols
        ),
        None,
    )
    if invalid_role is not None:
        raise ValueError(f"{name} cannot score invalid {symbol_kind} index {invalid_role!r}")
    try:
        probabilities = [float(start[sequence[0]])]
        probabilities.extend(float(transition[cur, prev]) for prev, cur in pairwise(sequence))
    except (IndexError, TypeError) as exc:
        raise ValueError(f"{name} cannot score case-role sequence {sequence!r}") from exc
    values = np.asarray(probabilities, dtype=np.float64)
    if not bool(np.all(np.isfinite(values))) or bool(np.any(values <= 0.0)):
        raise ValueError(
            f"{name} assigned a non-finite or non-positive probability to sequence {sequence!r}"
        )
    return float(np.log(values).sum())


def _validate_probability_chain(
    start: np.ndarray,
    transition: np.ndarray,
    *,
    name: str,
) -> None:
    """Require one start vector and column-oriented transition simplex."""
    if not bool(np.all(np.isfinite(start))) or not bool(np.all(np.isfinite(transition))):
        raise ValueError(f"{name} probability arrays must contain only finite values")
    if bool(np.any(start < 0.0)) or bool(np.any(transition < 0.0)):
        raise ValueError(f"{name} probability arrays must be non-negative")
    start_mass = float(start.sum(dtype=np.float64))
    column_masses = transition.sum(axis=0, dtype=np.float64)
    if not np.isclose(start_mass, 1.0, rtol=0.0, atol=1e-10) or not bool(
        np.allclose(column_masses, 1.0, rtol=0.0, atol=1e-10)
    ):
        raise ValueError(
            f"{name} must be a probability simplex: start sum={start_mass}, "
            f"transition column sums={column_masses.tolist()}"
        )


def summarize_oracle_gain(
    sentences: list[tuple[str, list[int]]],
    models: dict,
    comparator: tuple[np.ndarray, np.ndarray],
    *,
    clusters: Sequence[Hashable] | None = None,
    confidence: float = STAGE5_BOOTSTRAP_CONFIDENCE,
    resamples: int = STAGE5_BOOTSTRAP_RESAMPLES,
    seed: int = STAGE5_BOOTSTRAP_SEED,
    n_symbols: int = N_CASES,
    symbol_kind: str = "case-role",
) -> OracleGainBootstrapSummary:
    """Bootstrap paired privileged-label gain over held-out documents.

    For sentence ``i`` the paired difference is

    ``log P(sequence_i | fitted reference type_i) - log P(sequence_i | type free)``.

    Whole document indices are drawn for each replicate. Every sentence and
    position belonging to a selected document moves together, preserving both
    within-sentence trajectories and cross-sentence document dependence. The
    same replicate supplies the paired gain and both denominators, so neither
    bound can silently resample arms or numerator and denominator independently.

    Args:
        sentences: Typed held-out case-role sequences.
        clusters: One source-document identity per sentence. ``None`` means
            one independent cluster per sentence and is intended for synthetic
            fixtures; the shipped EWT recorder always supplies artifact-backed
            document indices.
        models: Fitted type-conditioned models from :func:`build_type_models`.
        comparator: Fitted type-free transition and start arrays.
        confidence: Two-sided percentile coverage.
        resamples: Number of fixed-seed bootstrap replicates.
        seed: Non-negative NumPy RNG seed.
        n_symbols: Observation alphabet size the fitted arms are shaped over,
            defaulting to the shipped case-role alphabet.
        symbol_kind: The alphabet's name, used only in refusal messages.

    Raises:
        ValueError: If the resampling contract is invalid, the evaluation is
            smaller than two independent documents, a type label is unknown, a sequence is empty or out of range,
            or either fitted arm assigns an invalid probability.
    """
    level, draws, normalized_seed = _validated_bootstrap_contract(
        confidence=confidence, resamples=resamples, seed=seed
    )
    if len(sentences) < 2:
        raise ValueError("oracle-gain bootstrap needs at least two evaluation sentences")
    cluster_keys, cluster_inverse = _cluster_index(clusters, len(sentences))

    try:
        free_transition_raw, free_start_raw = comparator
    except (TypeError, ValueError) as exc:
        raise ValueError("comparator must be a (transition, start) pair") from exc
    try:
        free_transition = np.asarray(free_transition_raw, dtype=np.float64)
        free_start = np.asarray(free_start_raw, dtype=np.float64)
        type_starts = np.asarray(models["starts"], dtype=np.float64)
        type_transitions = np.asarray(models["transitions"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("models and comparator must carry numeric starts and transitions") from exc
    if type_starts.shape != (N_TYPES, n_symbols) or type_transitions.shape != (
        N_TYPES,
        n_symbols,
        n_symbols,
    ):
        raise ValueError(
            "models must provide starts and transitions with shapes "
            f"{(N_TYPES, n_symbols)} and {(N_TYPES, n_symbols, n_symbols)}, got "
            f"{type_starts.shape} and {type_transitions.shape}"
        )
    if free_start.shape != (n_symbols,) or free_transition.shape != (n_symbols, n_symbols):
        raise ValueError(
            f"type-free comparator must provide start {(n_symbols,)} and transition "
            f"{(n_symbols, n_symbols)}, got {free_start.shape} and {free_transition.shape}"
        )
    _validate_probability_chain(
        free_start,
        free_transition,
        name="type-free comparator",
    )
    for type_index, sentence_type in enumerate(CLAUSE_TYPES):
        _validate_probability_chain(
            type_starts[type_index],
            type_transitions[type_index],
            name=f"privileged-label reference {sentence_type}",
        )

    gains: list[float] = []
    lengths: list[int] = []
    for sentence_type, sequence in sentences:
        if sentence_type not in CLAUSE_TYPES:
            raise ValueError(f"unknown sentence type {sentence_type!r}")
        t = CLAUSE_TYPES.index(sentence_type)
        oracle_ll = _sequence_component_log_evidence(
            sequence,
            type_starts[t],
            type_transitions[t],
            name="privileged-label reference",
            n_symbols=n_symbols,
            symbol_kind=symbol_kind,
        )
        free_ll = _sequence_component_log_evidence(
            sequence,
            free_start,
            free_transition,
            name="type-free comparator",
            n_symbols=n_symbols,
            symbol_kind=symbol_kind,
        )
        gain = oracle_ll - free_ll
        if not np.isfinite(gain):
            raise ValueError(
                f"paired oracle gain is non-finite for sentence type {sentence_type!r}"
            )
        gains.append(float(gain))
        lengths.append(len(sequence))

    gain_values = np.asarray(gains, dtype=np.float64)
    length_values = np.asarray(lengths, dtype=np.float64)
    n_documents = len(cluster_keys)
    cluster_gain_sums = np.bincount(cluster_inverse, weights=gain_values, minlength=n_documents)
    cluster_sentence_counts = np.bincount(cluster_inverse, minlength=n_documents).astype(np.float64)
    cluster_position_counts = np.bincount(
        cluster_inverse, weights=length_values, minlength=n_documents
    )
    sentence_replicates = np.empty(draws, dtype=np.float64)
    position_replicates = np.empty(draws, dtype=np.float64)
    rng = np.random.default_rng(normalized_seed)
    n_sentences = len(gain_values)
    for first in range(0, draws, _BOOTSTRAP_BLOCK):
        stop = min(first + _BOOTSTRAP_BLOCK, draws)
        indices = rng.integers(0, n_documents, size=(stop - first, n_documents))
        sampled_gains = cluster_gain_sums[indices].sum(axis=1)
        sentence_replicates[first:stop] = sampled_gains / cluster_sentence_counts[indices].sum(
            axis=1
        )
        position_replicates[first:stop] = sampled_gains / cluster_position_counts[indices].sum(
            axis=1
        )

    tail_pct = (1.0 - level) * 50.0
    sentence_low, sentence_high = np.percentile(
        sentence_replicates, [tail_pct, 100.0 - tail_pct], method="linear"
    )
    position_low, position_high = np.percentile(
        position_replicates, [tail_pct, 100.0 - tail_pct], method="linear"
    )
    return OracleGainBootstrapSummary(
        n_sentences=n_sentences,
        n_documents=n_documents,
        n_positions=int(length_values.sum()),
        mean_nats_per_sentence=float(gain_values.mean()),
        sentence_ci_low=float(sentence_low),
        sentence_ci_high=float(sentence_high),
        pooled_nats_per_position=float(gain_values.sum() / length_values.sum()),
        position_ci_low=float(position_low),
        position_ci_high=float(position_high),
        confidence=level,
        resamples=draws,
        seed=normalized_seed,
    )


def binary_auc(scores: list[float] | np.ndarray, labels: list[int] | np.ndarray) -> float:
    """Return tie-aware binary ROC AUC from finite scores and binary labels.

    This is the Mann--Whitney rank statistic. Equal scores receive their
    average rank, which gives a tied positive/negative pair one-half credit and
    makes the result invariant to row order. Assigning ordinal ranks to ties
    makes AUC depend on whichever sample happened to arrive first -- a real
    failure mode here because many short clauses have identical posteriors.

    Args:
        scores: One finite real-valued ranking score per observation.
        labels: Corresponding binary labels encoded as zero or one.

    Returns:
        ROC AUC in ``[0, 1]``.

    Raises:
        ValueError: If inputs are not aligned, one-dimensional, finite, binary,
            non-empty, or do not contain both classes.
    """
    score_values = np.asarray(scores, dtype=np.float64)
    label_values = np.asarray(labels)
    if score_values.ndim != 1 or label_values.ndim != 1:
        raise ValueError("scores and labels must be one-dimensional")
    if score_values.shape != label_values.shape:
        raise ValueError(
            f"scores and labels must have identical shapes, got "
            f"{score_values.shape} and {label_values.shape}"
        )
    if score_values.size == 0:
        raise ValueError("AUC is undefined on an empty sample")
    if not np.all(np.isfinite(score_values)):
        raise ValueError("scores must be finite")
    if not np.all(np.isin(label_values, (0, 1))):
        raise ValueError("labels must contain only binary values 0 and 1")

    binary_labels = label_values.astype(np.int64, copy=False)
    n_pos = int(binary_labels.sum())
    n_neg = int(binary_labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC is undefined on a single-class sample")

    ranks = rankdata(score_values, method="average")
    auc = (ranks[binary_labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def clustered_binary_auc_interval(
    scores: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
    clusters: Sequence[Hashable],
    *,
    confidence: float = STAGE5_BOOTSTRAP_CONFIDENCE,
    resamples: int = STAGE5_BOOTSTRAP_RESAMPLES,
    seed: int = STAGE5_BOOTSTRAP_SEED,
) -> ClusteredAucBootstrapSummary:
    """Return a whole-document percentile interval for sentence-level ROC AUC.

    A bootstrap draw samples source documents with replacement and includes
    every sentence from each selected document. Replicates containing only one
    class are rejected because AUC is undefined there. Both classes must occur
    in at least two distinct documents, which makes the primary uncertainty
    genuinely document-level rather than one influential source masquerading
    as multiple independent sentences.
    """
    level, draws, normalized_seed = _validated_bootstrap_contract(
        confidence=confidence, resamples=resamples, seed=seed
    )
    score_values = np.asarray(scores, dtype=np.float64)
    label_values = np.asarray(labels)
    point_auc = binary_auc(score_values, label_values)
    cluster_keys, inverse = _cluster_index(clusters, len(score_values))
    binary_labels = label_values.astype(np.int64, copy=False)
    groups = [np.flatnonzero(inverse == index) for index in range(len(cluster_keys))]
    for label in (0, 1):
        label_documents = {int(inverse[index]) for index in np.flatnonzero(binary_labels == label)}
        if len(label_documents) < 2:
            raise ValueError(
                f"clustered AUC needs label {label} in at least two independent documents"
            )

    rng = np.random.default_rng(normalized_seed)
    replicates = np.empty(draws, dtype=np.float64)
    accepted = 0
    attempts = 0
    max_attempts = max(draws * 100, draws + 1_000)
    while accepted < draws and attempts < max_attempts:
        attempts += 1
        selected = rng.integers(0, len(groups), size=len(groups))
        sampled_indices = np.concatenate([groups[int(index)] for index in selected])
        sampled_labels = binary_labels[sampled_indices]
        if sampled_labels.min() == sampled_labels.max():
            continue
        replicates[accepted] = binary_auc(score_values[sampled_indices], sampled_labels)
        accepted += 1
    if accepted != draws:  # pragma: no cover - protected by the class/document guard
        raise ValueError(
            f"clustered AUC accepted only {accepted} of {draws} required replicates "
            f"after {attempts} attempts"
        )

    tail_pct = (1.0 - level) * 50.0
    low, high = np.percentile(replicates, [tail_pct, 100.0 - tail_pct], method="linear")
    return ClusteredAucBootstrapSummary(
        auc=point_auc,
        ci_low=float(low),
        ci_high=float(high),
        n_observations=len(score_values),
        n_documents=len(cluster_keys),
        confidence=level,
        resamples=draws,
        seed=normalized_seed,
    )


def classify_sentences(sentences: list[tuple[str, list[int]]], models: dict) -> dict:
    """MAP type classification against the always-majority baseline, paired.

    Returns per-sentence hit vectors (for McNemar), accuracy, per-type recall,
    and the rank statistic (AUC of the interrogative posterior) — the
    prior-free description of how much signal exists at all.
    """
    majority = CLAUSE_TYPES[0]
    model_hits: list[bool] = []
    majority_hits: list[bool] = []
    scores: list[float] = []
    labels: list[int] = []
    for sentence_type, sequence in sentences:
        _, posterior = sequence_log_evidence(sequence, models)
        predicted = CLAUSE_TYPES[int(np.argmax(posterior))]
        model_hits.append(predicted == sentence_type)
        majority_hits.append(majority == sentence_type)
        scores.append(float(posterior[1]))
        labels.append(int(sentence_type == CLAUSE_TYPES[1]))
    auc = binary_auc(scores, labels)
    recalls = {}
    for _t, name in enumerate(CLAUSE_TYPES):
        of_type = [h for h, (st, _) in zip(model_hits, sentences, strict=True) if st == name]
        recalls[name] = float(np.mean(of_type)) if of_type else float("nan")
    return {
        "model_hits": model_hits,
        "majority_hits": majority_hits,
        "accuracy": float(np.mean(model_hits)),
        "majority_accuracy": float(np.mean(majority_hits)),
        "recalls": recalls,
        "auc": float(auc),
        # Per-sentence rank inputs, exposed so a caller can attach honest
        # uncertainty to the AUC (e.g. a bootstrap interval) without
        # re-running inference.
        "scores": scores,
        "labels": labels,
    }


def proper_scores(scores: list[float], labels: list[int], prior: float) -> dict:
    """Threshold-free scores that AUC alone leaves out.

    AUC answers one question — how well the posterior *ranks* interrogatives
    above declaratives — and answers it independently of prevalence. That
    independence is why the level reports it, and also why it is not enough on
    its own. A rank statistic near 0.7 alongside near-zero minority recall reads
    as a contradiction only if AUC is mistaken for a statement about decisions
    or about probabilities. It is neither.

    Two more quantities close that gap, both computed from the same posteriors:

    **Average precision**, the area under the precision-recall curve. Unlike
    AUC, its chance level is the positive-class prevalence rather than 0.5, so
    ``average_precision`` is reported beside ``prevalence`` and is only
    interpretable against it. This is the statistic that reflects how hard the
    minority class actually is here.

    **Brier score**, the mean squared error of the posterior against the label —
    a strictly proper scoring rule, so it is minimized only by the true
    conditional probability and cannot be improved by miscalibrated confidence.
    Its reference is the constant predictor that always emits the training
    prevalence, which is the probabilistic counterpart of the majority-class
    baseline the accuracy comparison already uses. The skill score is the
    fractional improvement over that reference; a negative value means the
    posteriors are worse than knowing nothing but the base rate, which is a
    finding rather than an error.

    Args:
        scores: Posterior probability of the positive (interrogative) class.
        labels: Ground-truth indicators, 1 for the positive class.
        prior: Positive-class prevalence in the *training* split, used as the
            constant reference predictor. Taking it from training keeps the
            reference from peeking at the evaluation labels.

    Returns:
        Mapping with ``average_precision``, ``prevalence``, ``brier``,
        ``brier_reference``, and ``brier_skill``.

    Raises:
        ValueError: If the inputs disagree in length, are empty, contain a
            single class, or carry a posterior outside [0, 1].
    """
    if len(scores) != len(labels):
        raise ValueError(f"scores and labels differ in length: {len(scores)} vs {len(labels)}")
    if not labels:
        raise ValueError("no evaluation sentences")
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(p)) or p.min() < 0.0 or p.max() > 1.0:
        raise ValueError("scores must be finite probabilities in [0, 1]")
    n_pos = float(y.sum())
    if n_pos == 0.0 or n_pos == len(y):
        raise ValueError("average precision is undefined on a single-class sample")
    if not 0.0 < prior < 1.0:
        raise ValueError(f"prior must lie strictly in (0, 1), got {prior!r}")

    # Average precision as the step-wise sum over recall increments, which is
    # the definition that does not interpolate between operating points.
    #
    # Ties are grouped rather than merely stably sorted. Equal posteriors are
    # common here — many short sentences receive the same score — and a
    # stable sort makes the result depend on the order rows arrived in: the
    # same evaluation set, permuted, scored differently. Since a decision rule
    # cannot separate items it assigns the same probability, precision and
    # recall are evaluated only at the *end* of each tie group, which is both
    # the standard definition and order-invariant.
    order = np.argsort(-p, kind="stable")
    ranked_scores = p[order]
    hits = y[order]
    cumulative_hits = np.cumsum(hits)
    seen = np.arange(1, len(hits) + 1, dtype=np.float64)
    # Last index of each run of equal scores.
    group_ends = np.flatnonzero(np.diff(ranked_scores) != 0.0)
    group_ends = np.append(group_ends, len(ranked_scores) - 1)
    precision = cumulative_hits[group_ends] / seen[group_ends]
    recall_gain = np.diff(np.concatenate(([0.0], cumulative_hits[group_ends]))) / n_pos
    average_precision = float((precision * recall_gain).sum())

    brier = float(np.mean((p - y) ** 2))
    reference = float(np.mean((prior - y) ** 2))
    return {
        "average_precision": average_precision,
        "prevalence": float(n_pos / len(y)),
        "brier": brier,
        "brier_reference": reference,
        "brier_skill": float(1.0 - brier / reference) if reference else float("nan"),
    }


# ── THE DETECTABILITY CONTROL ────────────────────────────────────────────────


def synthetic_control_corpus(
    *,
    n_train_per_type: int = 400,
    n_test_per_type: int = 100,
    seed: int = 0,
    n_symbols: int = N_CASES,
) -> tuple[list[tuple[str, list[int]]], list[tuple[str, list[int]]]]:
    """A corpus whose two types genuinely differ — the positive control.

    Two random chains over the observation alphabet, drawn once from fixed
    seeds, generate the two types' sequences. On this corpus the identical
    harness must classify far above the majority baseline and the mixture must
    beat the type-free chain decisively. This proves that the instrument
    detects a large planted effect; it does not establish that the unresolved
    EWT gain is absent or negligible. A harness that could not detect the
    planted effect would make the non-passing result instrument-ambiguous.

    Args:
        n_train_per_type: Training sentences generated per type.
        n_test_per_type: Held-out sentences generated per type.
        seed: Fixed RNG seed for the corpus and both planted chains.
        n_symbols: Observation alphabet size. The widened arm plants its
            control at its own alphabet size so the control exercises the
            harness the measurement actually runs, not a narrower one.
    """
    rng = np.random.default_rng(seed)
    chain_rngs = [np.random.default_rng(seed + 1), np.random.default_rng(seed + 2)]
    chains = []
    for chain_rng in chain_rngs:
        transition = chain_rng.dirichlet(np.full(n_symbols, 0.3), size=n_symbols).T
        start = chain_rng.dirichlet(np.full(n_symbols, 0.5))
        chains.append((transition, start))

    def sample(t: int, length: int) -> list[int]:
        transition, start = chains[t]
        sequence = [int(rng.choice(n_symbols, p=start))]
        for _ in range(length - 1):
            sequence.append(int(rng.choice(n_symbols, p=transition[:, sequence[-1]])))
        return sequence

    def corpus(n_per_type: int) -> list[tuple[str, list[int]]]:
        out = []
        for t, name in enumerate(CLAUSE_TYPES):
            for _ in range(n_per_type):
                out.append((name, sample(t, 6)))
        return out

    return corpus(n_train_per_type), corpus(n_test_per_type)
