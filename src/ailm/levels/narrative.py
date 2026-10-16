"""Narrative level (Stage 6, L6): document phases over sentence types.

One step up from clauses: the document. The observation stream is the
sentence-type sequence — Level 5's latent variable, one level down — and the
hidden state is the narrative **phase**, operationalised deterministically as
the document-position tercile (opening / middle / closing). The phase is a
function of position, so every parameter is a count: no EM, no fitted
segmentation, nothing whose estimation could quietly overfit the claim.

Two things live here, measured separately (decisions 25–26):

1. **Recognition** — does document position inform sentence type out of
   sample? P(type | phase) against the type-unigram bar, per-sentence
   log-loss, paired over the held-out documents.
2. **The top-down channel (ISC-23)** — an independently measured prior
   intervention: the narrative prediction P(type | phase)
   blends into the clause level's type prior,
   ``D'(α) = (1−α)·P_clause + α·P(type | phase)``, re-formed into the joint
   (type × case) prior. α = 0 is the exact identity anchor; a phase-permuted
   scramble is the entropy-preserving control. Stage 5's
   ``stage5_privileged_label_gain_pct`` records sensitivity to a privileged type label;
   it is a reference value, not a hard bound on this intervention under model
   misspecification. This channel is not connected to the separate privileged
   punctuation-type-oracle→L3 or L3→L1g runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ailm.config import DEFAULT_LEVEL_CONFIG, LevelConfig
from ailm.corpus.ud import document_identity_sha256, validate_ud_ewt_source_metadata
from ailm.levels._stochastic import column_stochastic
from ailm.levels.clause import CLAUSE_TYPES, N_CASES, N_JOINT, N_TYPES
from ailm.resources import resolve_vendored_artifact

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pymdp.agent import Agent as PymdpAgent

__all__ = [
    "DEFAULT_ARTIFACT",
    "NARRATIVE_PHASES",
    "N_PHASES",
    "NarrativeArtifact",
    "build_narrative_agent",
    "build_phase_models",
    "build_phase_transition",
    "clause_prior_with_narrative",
    "heldout_type_logloss",
    "live_agent_type_logloss",
    "load_narrative_artifact",
    "tercile",
]

NARRATIVE_PHASES: tuple[str, ...] = ("opening", "middle", "closing")
N_PHASES: int = len(NARRATIVE_PHASES)
NARRATIVE_ARTIFACT_SCHEMA = "ailm-narrative-counts-v2"

DEFAULT_ARTIFACT = resolve_vendored_artifact("ud_ewt_narrative_counts.json")


def tercile(index: int, length: int) -> int:
    """Deterministic phase of sentence ``index`` in a document of ``length``."""
    if not 0 <= index < length:
        raise ValueError(f"index {index} outside document of length {length}")
    return min(N_PHASES - 1, (N_PHASES * index) // length)


class NarrativeArtifact:
    """The vendored Stage 6 statistics, validated on load."""

    def __init__(self, payload: dict) -> None:
        if not isinstance(payload, dict) or set(payload) != {
            "schema",
            "provenance",
            "splits",
            "documents",
        }:
            raise ValueError(
                "narrative artifact must contain schema, provenance, splits, documents"
            )
        if payload.get("schema") != NARRATIVE_ARTIFACT_SCHEMA:
            raise ValueError(
                f"Unexpected artifact schema {payload.get('schema')!r}; "
                f"regenerate with scripts/generate_narrative_counts.py"
            )
        if not isinstance(payload["provenance"], dict):
            raise ValueError("narrative artifact provenance must be an object")
        self.provenance: dict = payload["provenance"]
        splits = payload["splits"]
        if not isinstance(splits, dict) or set(splits) != {"train", "test"}:
            raise ValueError("narrative artifact splits must name train and test")
        train_doc_names, self.train = self._validate(splits["train"], "train")
        test_doc_names, self.test = self._validate(splits["test"], "test")
        (
            self.document_ids,
            split_document_indices,
            sequence_document_indices,
        ) = self._validate_documents(payload["documents"], splits)
        self.train_split_document_indices = split_document_indices["train"]
        self.test_split_document_indices = split_document_indices["test"]
        self.train_sequence_document_indices = sequence_document_indices["train"]
        self.test_sequence_document_indices = sequence_document_indices["test"]
        self.train_document_ids = [
            self.document_ids[index] for index in self.train_sequence_document_indices
        ]
        self.test_document_ids = [
            self.document_ids[index] for index in self.test_sequence_document_indices
        ]
        if train_doc_names != self.train_document_ids or test_doc_names != self.test_document_ids:
            raise ValueError("narrative split entry document ids disagree with their index mapping")
        if self.provenance.get("document_identity_sha256") != document_identity_sha256(
            self.document_ids
        ):
            raise ValueError(
                "narrative artifact document identity digest disagrees with its ordered ids"
            )
        expected_counts = {
            "documents_total": len(self.document_ids),
            "train_documents": len(self.train_split_document_indices),
            "test_documents": len(self.test_split_document_indices),
            "train_eligible_documents": len(self.train_document_ids),
            "test_eligible_documents": len(self.test_document_ids),
            "documents_excluded_short": len(self.document_ids)
            - len(self.train_document_ids)
            - len(self.test_document_ids),
        }
        for field, expected in expected_counts.items():
            if self.provenance.get(field) != expected:
                raise ValueError(
                    f"narrative artifact provenance {field} is inconsistent: "
                    f"{self.provenance.get(field)!r} != {expected}"
                )
        typed = self.provenance.get("documents_with_typed_sentences")
        eligible = len(self.train_document_ids) + len(self.test_document_ids)
        if (
            isinstance(typed, bool)
            or not isinstance(typed, int)
            or not eligible <= typed <= len(self.document_ids)
        ):
            raise ValueError(
                "narrative documents_with_typed_sentences must bound the eligible documents"
            )

    @staticmethod
    def _validate(entries: object, name: str) -> tuple[list[str], list[list[int]]]:
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{name} split is empty")
        doc_names: list[str] = []
        out: list[list[int]] = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"doc", "types"}:
                raise ValueError(f"{name} split has a malformed document entry")
            doc = entry["doc"]
            if not isinstance(doc, str) or not doc or "\0" in doc:
                raise ValueError(f"{name} split carries an invalid document id {doc!r}")
            types = entry["types"]
            if not isinstance(types, list):
                raise ValueError(f"{name} split document {doc!r} has no type sequence")
            if len(types) < 3:
                raise ValueError(f"{name} split carries a document with no arc")
            stray = next((t for t in types if t not in CLAUSE_TYPES), None)
            if stray is not None:
                raise ValueError(f"unknown sentence type {stray!r}")
            doc_names.append(doc)
            out.append([CLAUSE_TYPES.index(t) for t in types])
        if len(doc_names) != len(set(doc_names)):
            raise ValueError(f"{name} split repeats a document id")
        return doc_names, out

    @staticmethod
    def _validate_documents(
        value: object, splits: object
    ) -> tuple[list[str], dict[str, list[int]], dict[str, list[int]]]:
        expected_keys = {"ids", "split_indices", "sequence_document_indices"}
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise ValueError(
                "narrative artifact documents must contain ids, split_indices, and "
                "sequence_document_indices"
            )
        ids = value["ids"]
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(item, str) or not item or "\0" in item for item in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError("narrative artifact document ids must be non-empty unique strings")
        if not isinstance(splits, dict) or set(splits) != {"train", "test"}:
            raise ValueError("narrative artifact splits must name train and test")

        raw_split_indices = value["split_indices"]
        raw_sequence_indices = value["sequence_document_indices"]
        if (
            not isinstance(raw_split_indices, dict)
            or set(raw_split_indices) != {"train", "test"}
            or not isinstance(raw_sequence_indices, dict)
            or set(raw_sequence_indices) != {"train", "test"}
        ):
            raise ValueError("narrative artifact document mappings must name train and test")

        def validate_indices(raw: object, *, label: str) -> list[int]:
            if (
                not isinstance(raw, list)
                or any(
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < len(ids)
                    for index in raw
                )
                or len(raw) != len(set(raw))
            ):
                raise ValueError(f"narrative artifact {label} must contain unique valid indices")
            return list(raw)

        split_indices = {
            name: validate_indices(raw_split_indices[name], label=f"{name} split indices")
            for name in ("train", "test")
        }
        sequence_indices = {
            name: validate_indices(
                raw_sequence_indices[name], label=f"{name} sequence document indices"
            )
            for name in ("train", "test")
        }
        train_set = set(split_indices["train"])
        test_set = set(split_indices["test"])
        if train_set & test_set or train_set | test_set != set(range(len(ids))):
            raise ValueError(
                "narrative artifact train/test document partition must be disjoint and exhaustive"
            )
        for name, allowed in (("train", train_set), ("test", test_set)):
            if len(sequence_indices[name]) != len(splits[name]):
                raise ValueError(
                    f"narrative artifact {name} sequence/document mapping length is inconsistent"
                )
            if not set(sequence_indices[name]) <= allowed:
                raise ValueError(
                    f"narrative artifact {name} sequence maps outside its document split"
                )
        return list(ids), split_indices, sequence_indices


def load_narrative_artifact(path: Path | str | None = None) -> NarrativeArtifact:
    """Load and validate the vendored narrative artifact.

    Raises:
        FileNotFoundError: If absent — Stage 6 has no fallback corpus.
        ValueError: On validation failure.
    """
    resolved = (
        DEFAULT_ARTIFACT
        if path is None
        else resolve_vendored_artifact("ud_ewt_narrative_counts.json", path=path)
    )
    if not resolved.is_file():
        raise FileNotFoundError(f"{resolved} is absent. Run scripts/generate_narrative_counts.py.")
    artifact = NarrativeArtifact(json.loads(resolved.read_text(encoding="utf-8")))
    validate_ud_ewt_source_metadata(artifact.provenance)
    return artifact


# ── ESTIMATION ───────────────────────────────────────────────────────────────


def build_phase_models(
    documents: list[list[int]],
    *,
    config: LevelConfig = DEFAULT_LEVEL_CONFIG,
) -> dict:
    """``emission``: P(type | phase), ``unigram``: the type bar — all counts.

    Returns:
        ``emission``: ``(N_TYPES, N_PHASES)`` column-stochastic per phase;
        ``unigram``: ``(N_TYPES,)`` — the phase-blind comparator (decision 2:
        the honest bar is the strongest model that does not use *phase*).
    """
    k = float(config.smoothing)
    emission = np.full((N_TYPES, N_PHASES), k, dtype=np.float64)
    unigram = np.full(N_TYPES, k, dtype=np.float64)
    for types in documents:
        for index, sentence_type in enumerate(types):
            emission[sentence_type, tercile(index, len(types))] += 1.0
            unigram[sentence_type] += 1.0
    return {
        "emission": column_stochastic(emission),
        "unigram": unigram / unigram.sum(),
    }


def build_phase_transition(documents: list[list[int]]) -> np.ndarray:
    """``B``: P(phase' | phase) from consecutive-sentence tercile pairs.

    Left-to-right by construction — a tercile never decreases along a
    document. In ``B[next, current]`` orientation that puts every non-zero
    entry ON or BELOW the main diagonal (``next >= current``), so the
    strictly **upper** triangle is exactly zero. Stating that with the wrong
    triangle is precisely the orientation trap of decision 1, and the first
    draft of this docstring did: the recorded
    ``claim:stage6_transition_above_diagonal`` pins the correct invariant,
    and ``tests/test_stage6_gate.py`` asserts it rather than trusting prose.
    """
    counts = np.zeros((N_PHASES, N_PHASES), dtype=np.float64)
    for types in documents:
        length = len(types)
        for index in range(length - 1):
            counts[tercile(index + 1, length), tercile(index, length)] += 1.0
    return column_stochastic(counts)


def build_narrative_agent(
    artifact: NarrativeArtifact,
    *,
    config: LevelConfig = DEFAULT_LEVEL_CONFIG,
) -> PymdpAgent:
    """The Stage 6 pymdp agent: 3 phase states, 2 type observations."""
    import jax.numpy as jnp
    from pymdp.agent import Agent

    models = build_phase_models(artifact.train, config=config)
    transition = build_phase_transition(artifact.train)
    start = np.zeros(N_PHASES, dtype=np.float64)
    start[0] = 1.0  # tercile(0, n) is 0 for every n: documents open in "opening"

    return Agent(
        A=[jnp.array(models["emission"], dtype=jnp.float32)[jnp.newaxis]],
        B=[jnp.array(transition, dtype=jnp.float32)[jnp.newaxis, :, :, jnp.newaxis]],
        C=[jnp.zeros((1, N_TYPES))],
        D=[jnp.array(start, dtype=jnp.float32)[jnp.newaxis]],
        num_controls=[1],
        batch_size=1,
    )


# ── RECOGNITION: does position inform type, out of sample? ───────────────────


def heldout_type_logloss(documents: list[list[int]], models: dict) -> dict:
    """Observed-position association: phase-conditional vs unigram log loss.

    The phase column is selected from the known sentence index and document
    length. This is an observed-covariate score of ``P(type | position tercile)``;
    it is not filtering through :func:`build_narrative_agent` and therefore does
    not use that agent's ``B`` or ``D``. The live-agent counterpart is
    :func:`live_agent_type_logloss`; keeping the two names and ledger rows
    separate prevents an association measured with true position from being
    attributed to latent-state inference.

    Returns per-sentence deltas (positive = the phase model was better on that
    sentence) so the caller can run a paired test, plus the two mean losses.

    Raises:
        ValueError: If no sentence is scored — an empty document list, or
            documents containing no sentences. A mean over zero sentences is
            not a small number, it is undefined; the same fail-closed contract
            :func:`ailm.levels.case.case_agreement` uses for zero positions.
    """
    phase_losses: list[float] = []
    unigram_losses: list[float] = []
    for types in documents:
        length = len(types)
        for index, sentence_type in enumerate(types):
            phase = tercile(index, length)
            phase_losses.append(-float(np.log(models["emission"][sentence_type, phase])))
            unigram_losses.append(-float(np.log(models["unigram"][sentence_type])))
    deltas = [u - p for p, u in zip(phase_losses, unigram_losses, strict=True)]
    if not deltas:
        raise ValueError("no sentences scored; log loss is undefined")
    return {
        "phase_mean": float(np.mean(phase_losses)),
        "unigram_mean": float(np.mean(unigram_losses)),
        "mean_delta": float(np.mean(deltas)),
        "deltas": deltas,
        "n_sentences": len(deltas),
    }


def live_agent_type_logloss(
    documents: list[list[int]], agent: PymdpAgent, unigram: np.ndarray
) -> dict:
    """Sequential predictive log loss through the real Stage 6 pymdp agent.

    Each document resets to the agent's declared ``D``. Before observing a
    sentence type, ``A @ prior`` is the predictive type distribution and is what
    receives the log-loss charge. After the observation, ``infer_states`` and
    ``update_empirical_prior`` advance the within-level belief through ``B``.
    Scoring the posterior after showing it the answer would be leakage, while
    indexing the true tercile would be the separate observed-position estimand
    implemented by :func:`heldout_type_logloss`.

    Args:
        documents: Held-out type-index sequences, one sequence per document.
        agent: A real pymdp narrative agent.
        unigram: Phase-blind ``(N_TYPES,)`` comparator.

    Returns:
        Mean live-agent and unigram losses, their per-sentence deltas, and count.

    Raises:
        ValueError: If the comparator or agent matrices have incompatible shapes
            or contain invalid probabilities.
    """
    import jax.numpy as jnp

    comparator = np.asarray(unigram, dtype=np.float64)
    if comparator.shape != (N_TYPES,):
        raise ValueError(f"unigram must have shape ({N_TYPES},), got {comparator.shape}")
    if not np.isfinite(comparator).all() or (comparator <= 0.0).any():
        raise ValueError("unigram must contain finite positive probabilities")
    if not np.isclose(comparator.sum(), 1.0, atol=1e-9):
        raise ValueError(f"unigram sums to {comparator.sum()}, not 1")

    emission = np.asarray(agent.A[0], dtype=np.float64)
    start = np.asarray(agent.D[0], dtype=np.float64)
    if emission.shape != (1, N_TYPES, N_PHASES):
        raise ValueError(
            f"agent emission must have shape (1, {N_TYPES}, {N_PHASES}), got {emission.shape}"
        )
    if start.shape != (1, N_PHASES):
        raise ValueError(f"agent start prior must have shape (1, {N_PHASES}), got {start.shape}")

    live_losses: list[float] = []
    unigram_losses: list[float] = []
    action = jnp.zeros((1, 1), dtype=jnp.int32)
    for types in documents:
        prior = [jnp.asarray(start)]
        for sentence_type in types:
            predictive = emission[0] @ np.asarray(prior[0], dtype=np.float64).ravel()
            probability = float(predictive[sentence_type])
            if not np.isfinite(probability) or probability <= 0.0:
                raise ValueError(
                    f"agent assigned invalid predictive mass {probability!r} "
                    f"to sentence type {sentence_type}"
                )
            live_losses.append(-float(np.log(probability)))
            unigram_losses.append(-float(np.log(comparator[sentence_type])))
            posterior = agent.infer_states([jnp.array([sentence_type])], prior)
            prior = agent.update_empirical_prior(action, posterior)

    deltas = [u - live for live, u in zip(live_losses, unigram_losses, strict=True)]
    return {
        "live_mean": float(np.mean(live_losses)),
        "unigram_mean": float(np.mean(unigram_losses)),
        "mean_delta": float(np.mean(deltas)),
        "deltas": deltas,
        "n_sentences": len(deltas),
    }


# ── THE TOP-DOWN CHANNEL (ISC-23) ────────────────────────────────────────────


def clause_prior_with_narrative(
    clause_models: dict,
    phase: int,
    narrative_models: dict,
    *,
    alpha: float,
    scrambled: bool = False,
) -> np.ndarray:
    """The clause level's joint prior with the narrative message blended in.

    ``D'(α) = (1−α)·P_clause(type) + α·P(type | phase)``, re-formed into the
    joint (type × case) prior with the clause level's own per-type start
    distributions. α = 0 returns the unmodified prior **exactly** — the
    identity anchor the recorder refuses to proceed without. ``scrambled``
    reverses the phase axis (an entropy-preserving permutation): the control
    message that carries the same bits about nothing.

    Args:
        clause_models: ``build_type_models`` output for the clause level.
        phase: The narrative phase index at the receiving sentence.
        narrative_models: ``build_phase_models`` output.
        alpha: Channel weight in [0, 1].
        scrambled: Use the permuted-phase control message.

    Returns:
        ``(N_JOINT,)`` prior over the clause level's joint states.

    Raises:
        ValueError: If ``alpha`` is outside [0, 1] or ``phase`` is invalid.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not 0 <= phase < N_PHASES:
        raise ValueError(f"phase must be in [0, {N_PHASES}), got {phase}")
    effective_phase = (N_PHASES - 1 - phase) if scrambled else phase
    message = narrative_models["emission"][:, effective_phase]
    blended = (1.0 - alpha) * clause_models["type_prior"] + alpha * message
    return (blended[:, None] * clause_models["starts"]).reshape(N_JOINT)


def channel_shift_summary(
    clause_models: dict, narrative_models: dict, *, alpha: float = 1.0
) -> dict:
    """How far the channel moves P(interrogative), per phase.

    The mechanism metric of decision 25's P2: the α = 1 shift must be
    strictly positive for at least one phase; the α = 0 shift must be exactly
    zero for every phase; the scrambled shifts must differ from the real ones
    *for at least one phase*.

    Not for every phase, and the difference is structural rather than
    empirical: the scramble is the reversal ``phase -> N_PHASES - 1 - phase``,
    and with three phases that permutation fixes the middle one. The middle
    phase's scrambled shift is therefore identical to its real shift by
    construction, for any corpus. ``scripts/record_stage6_claims.py`` guards
    with ``all(...)`` — it fails only when the scramble is invisible
    everywhere — which is the strongest form this control can honestly take.
    Reading it as a per-phase requirement would make the guard permanently
    unsatisfiable.
    """
    interrogative = CLAUSE_TYPES.index("interrogative")
    block = slice(interrogative * N_CASES, (interrogative + 1) * N_CASES)
    static = clause_prior_with_narrative(clause_models, 0, narrative_models, alpha=0.0)[block].sum()
    shifts = {}
    scrambled_shifts = {}
    for phase, name in enumerate(NARRATIVE_PHASES):
        real = clause_prior_with_narrative(clause_models, phase, narrative_models, alpha=alpha)[
            block
        ].sum()
        control = clause_prior_with_narrative(
            clause_models, phase, narrative_models, alpha=alpha, scrambled=True
        )[block].sum()
        shifts[name] = float(real - static)
        scrambled_shifts[name] = float(control - static)
    return {
        "static_interrogative_mass": float(static),
        "shifts": shifts,
        "scrambled_shifts": scrambled_shifts,
    }
