"""Case-level active inference agent for AILM Stage 4.

A single-factor, single-modality pymdp model over grammatical case roles:

  - Hidden factor:        the token's case role — 8 states (``CaseRole``)
  - Observation modality: the token's UPOS tag — one state per tag seen
  - A: ``P(UPOS | case)``, learned from the training split
  - B: ``P(case_t | case_{t-1})`` along the sentence — the case-role transition
  - D: ``P(case)`` at the first labeled token of a sentence
  - C: flat zeros

As at Levels 1 and 2, the learned sequence dynamics live in the **transition**
and ``A`` is the emission. Structurally this is Level 2B one level up: two
different alphabets, observations of one and inference over the other.

What the model is NOT allowed to observe, and why it matters more here
----------------------------------------------------------------------
The observation is the **UPOS tag only**. Never ``deprel``, never
``feats[Case]``. Those are the fields the evaluation labels are *derived from*,
so a model observing them would be reading its own answer key: agreement would
approach perfection and would measure nothing at all. This is the sharpest trap at
this level, because nothing about such a model would look broken — it would
simply be excellent. ``docs/decisions.md`` entry 16 records it, and
``tests/test_stage4_gate.py::test_the_observation_never_carries_the_label_source``
enforces it.

Agreement, not accuracy
-----------------------
Only Nom/Acc/Gen are morphologically marked in English. Dat/Ins/Loc/Abl/Voc have
**no** morphological ground truth in English EWT, so their labels come from a
documented syntactic heuristic. A score against those labels is *agreement with
a research claim*, and this module's functions are named to make saying
otherwise awkward: :func:`case_agreement`, never ``case_accuracy``.

The heuristic agrees with real morphology on
claim:stage4_heuristic_gold_agreement_pct% of the tokens carrying both. That is
a validation result for the covered, morphologically marked subset only. It is
not a fidelity ceiling for the full derived population, and it contains no
token-level adposition-route examples (the measured count is
claim:stage4_heuristic_gold_adposition_token_tokens).

Two populations, never merged
-----------------------------
``gold`` is 3 of 8 cases and, after a held-out split, claim:stage4_gold_test_tokens
tokens — too few to gate on, and UPOS is nearly uninformative there because
English marks case almost only on pronouns. That is a measured null, reported
rather than hidden. ``derived`` is the population Stage 4's gate is stated over.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from ailm.cases import CaseRole
from ailm.config import DEFAULT_LEVEL_CONFIG, LevelConfig
from ailm.corpus.ud import (
    CASE_ORACLE_PATHS,
    UD_UPOS_TAGS,
    document_identity_sha256,
    validate_ud_ewt_source_metadata,
)
from ailm.levels._stochastic import column_stochastic
from ailm.resources import resolve_vendored_artifact

if TYPE_CHECKING:  # pragma: no cover
    from pymdp.agent import Agent as PymdpAgent

__all__ = [
    "CASES",
    "CASE_TO_IDX",
    "N_CASES",
    "CaseArtifact",
    "CASE_ARTIFACT_SCHEMA",
    "baseline_case_positions",
    "build_case_agent",
    "build_case_emission",
    "build_case_start",
    "build_case_transition",
    "case_agreement",
    "context_free_baseline",
    "decode_case_positions",
    "load_case_artifact",
    "ORACLE_PATHS",
    "oracle_path_strata",
]

logger = logging.getLogger(__name__)

#: The 8 case roles, in a fixed order that indexes every matrix here.
CASES: tuple[CaseRole, ...] = tuple(CaseRole)
CASE_TO_IDX: dict[CaseRole, int] = {case: i for i, case in enumerate(CASES)}
N_CASES: int = len(CASES)

DEFAULT_ARTIFACT = resolve_vendored_artifact("ud_ewt_case_counts.json")
CASE_ARTIFACT_SCHEMA = "ailm-case-counts-v2"

#: Fields a token must NOT expose to the model. Named here so the prohibition is
#: a constant the tests can bind to rather than a comment.
FORBIDDEN_OBSERVATIONS: frozenset[str] = frozenset({"deprel", "feats", "case"})


def _validated_case_sequences(value: object, *, context: str) -> list[list[tuple[str, str]]]:
    """Validate observations against UD's external UPOS alphabet and case labels."""

    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of token sequences")
    valid_cases = {role.name for role in CaseRole}
    sequences: list[list[tuple[str, str]]] = []
    for sequence_index, sequence in enumerate(value):
        if not isinstance(sequence, list):
            raise ValueError(f"{context}[{sequence_index}] must be a token sequence")
        checked: list[tuple[str, str]] = []
        for token_index, pair in enumerate(sequence):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"{context}[{sequence_index}][{token_index}] must be [UPOS, case]")
            upos, case = pair
            if upos not in UD_UPOS_TAGS:
                raise ValueError(
                    f"{context}[{sequence_index}][{token_index}] observation {upos!r} "
                    "is not a canonical UD UPOS tag"
                )
            if case not in valid_cases:
                raise ValueError(
                    f"{context}[{sequence_index}][{token_index}] label {case!r} is not a CaseRole"
                )
            checked.append((upos, case))
        sequences.append(checked)
    return sequences


def _nonnegative_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    """Read one sufficient count without accepting booleans or coercions."""
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context}.{key} must be a non-negative integer, got {value!r}")
    return value


def _validate_oracle_scope_audit(
    value: object, heuristic_vs_gold: Mapping[str, object]
) -> dict[str, object]:
    """Fail closed when the artifact's scope sufficient counts do not reconcile."""
    if not isinstance(value, dict):
        raise ValueError("case artifact lacks an oracle_scope_audit object")

    full_derived = _nonnegative_int(value, "full_derived_tokens", context="oracle_scope_audit")
    path_counts = value.get("derived_path_tokens")
    if not isinstance(path_counts, dict) or set(path_counts) != set(CASE_ORACLE_PATHS):
        raise ValueError(
            "oracle_scope_audit.derived_path_tokens must name exactly "
            f"{CASE_ORACLE_PATHS}, got {path_counts!r}"
        )
    measured_path_total = sum(
        _nonnegative_int(path_counts, path, context="oracle_scope_audit.derived_path_tokens")
        for path in CASE_ORACLE_PATHS
    )
    if measured_path_total != full_derived:
        raise ValueError(
            "oracle-scope path counts do not sum to full_derived_tokens: "
            f"{measured_path_total} != {full_derived}"
        )

    fidelity = value.get("heuristic_vs_gold")
    if not isinstance(fidelity, dict) or fidelity != heuristic_vs_gold:
        raise ValueError(
            "oracle_scope_audit.heuristic_vs_gold must equal the artifact's "
            "top-level heuristic_vs_gold record"
        )
    both = _nonnegative_int(fidelity, "tokens_with_both", context="heuristic_vs_gold")
    agreements = _nonnegative_int(fidelity, "agreements", context="heuristic_vs_gold")
    if agreements > both:
        raise ValueError("heuristic_vs_gold agreements exceed tokens_with_both")
    fidelity_paths = fidelity.get("source_path_tokens")
    expected_fidelity_paths = {"deprel", "adposition_token"}
    if not isinstance(fidelity_paths, dict) or set(fidelity_paths) != expected_fidelity_paths:
        raise ValueError(
            "heuristic_vs_gold.source_path_tokens must name deprel and adposition_token"
        )
    if (
        sum(
            _nonnegative_int(fidelity_paths, path, context="heuristic_vs_gold.source_path_tokens")
            for path in sorted(expected_fidelity_paths)
        )
        != both
    ):
        raise ValueError("heuristic fidelity path counts do not sum to tokens_with_both")

    adposition = value.get("adposition_token")
    if not isinstance(adposition, dict):
        raise ValueError("oracle_scope_audit lacks adposition_token sufficient counts")
    adposition_tokens = _nonnegative_int(adposition, "tokens", context="adposition_token")
    if adposition_tokens != path_counts["adposition_token"]:
        raise ValueError("adposition-token audit extent disagrees with the path composition")
    resolved = _nonnegative_int(adposition, "head_resolved_tokens", context="adposition_token")
    unresolved = _nonnegative_int(adposition, "head_unresolved_tokens", context="adposition_token")
    if resolved + unresolved != adposition_tokens:
        raise ValueError("resolved and unresolved adposition heads do not sum to the path extent")
    also_labeled = _nonnegative_int(
        adposition, "head_also_labeled_tokens", context="adposition_token"
    )
    head_agree = _nonnegative_int(adposition, "head_role_agreements", context="adposition_token")
    head_disagree = _nonnegative_int(
        adposition, "head_role_disagreements", context="adposition_token"
    )
    if also_labeled > resolved or head_agree + head_disagree != also_labeled:
        raise ValueError(
            "adposition head-role agreements and disagreements must partition the labeled heads"
        )
    _nonnegative_int(adposition, "morphological_case_tokens", context="adposition_token")
    return value


class CaseArtifact:
    """The vendored Stage 4 statistics for one population.

    Attributes:
        population: ``"gold"`` or ``"derived"``.
        train: Training sentences, each ``[(upos, case_name), ...]``.
        test: Held-out sentences, same shape.
        upos_tags: Sorted observation alphabet.
        upos_to_idx: Observation index map.
        provenance: The generator's recorded parameters.
        heuristic_vs_gold: Agreement of the syntactic heuristic with real
            morphology, on the tokens carrying both.
        oracle_scope_audit: Path composition and token/head sufficient counts
            that delimit what that agreement measurement can validate.
    """

    def __init__(self, payload: dict, population: str) -> None:
        schema = payload.get("schema")
        if schema != CASE_ARTIFACT_SCHEMA:
            raise ValueError(
                f"Unexpected artifact schema {schema!r}; regenerate with "
                f"scripts/generate_case_counts.py"
            )
        if population not in payload["populations"]:
            raise ValueError(
                f"Unknown population {population!r}; the artifact carries "
                f"{sorted(payload['populations'])}"
            )
        self.population = population
        self.provenance: dict = payload["provenance"]
        self.heuristic_vs_gold: dict = payload["heuristic_vs_gold"]
        self.oracle_scope_audit = _validate_oracle_scope_audit(
            payload.get("oracle_scope_audit"), self.heuristic_vs_gold
        )
        splits = payload["populations"][population]
        self.train = _validated_case_sequences(
            splits.get("train") if isinstance(splits, dict) else None,
            context=f"populations.{population}.train",
        )
        self.test = _validated_case_sequences(
            splits.get("test") if isinstance(splits, dict) else None,
            context=f"populations.{population}.test",
        )
        if not self.train or not self.test:
            raise ValueError(
                f"population {population!r} has an empty split "
                f"({len(self.train)} train, {len(self.test)} test)"
            )
        self.document_ids, split_document_indices, sequence_document_indices = (
            self._validate_documents(payload.get("documents"), payload["populations"])
        )
        self.train_split_document_indices = split_document_indices["train"]
        self.test_split_document_indices = split_document_indices["test"]
        population_documents = sequence_document_indices[population]
        self.train_sequence_document_indices = population_documents["train"]
        self.test_sequence_document_indices = population_documents["test"]
        self.train_document_ids = [
            self.document_ids[index] for index in sorted(set(self.train_sequence_document_indices))
        ]
        self.test_document_ids = [
            self.document_ids[index] for index in sorted(set(self.test_sequence_document_indices))
        ]
        provenance_digest = self.provenance.get("document_identity_sha256")
        measured_digest = document_identity_sha256(self.document_ids)
        if provenance_digest != measured_digest:
            raise ValueError(
                "case artifact document identity digest disagrees with its ordered ids"
            )
        if self.provenance.get("documents") != len(self.document_ids):
            raise ValueError("case artifact provenance document count is inconsistent")
        if self.provenance.get("train_documents") != len(self.train_split_document_indices):
            raise ValueError("case artifact provenance train-document count is inconsistent")
        if self.provenance.get("test_documents") != len(self.test_split_document_indices):
            raise ValueError("case artifact provenance test-document count is inconsistent")
        self.upos_tags: list[str] = sorted({u for seq in self.train + self.test for u, _ in seq})
        self.upos_to_idx: dict[str, int] = {u: i for i, u in enumerate(self.upos_tags)}

    @staticmethod
    def _validate_documents(
        value: object, populations: object
    ) -> tuple[list[str], dict[str, list[int]], dict[str, dict[str, list[int]]]]:
        """Validate exact document identity, partition, and sequence alignment."""
        expected_keys = {"ids", "split_indices", "sequence_document_indices"}
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise ValueError(
                "case artifact documents must contain ids, split_indices, and "
                "sequence_document_indices"
            )
        ids = value["ids"]
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(item, str) or not item or "\0" in item for item in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError("case artifact document ids must be non-empty unique strings")

        split_indices = value["split_indices"]
        if not isinstance(split_indices, dict) or set(split_indices) != {"train", "test"}:
            raise ValueError("case artifact split_indices must name train and test")
        checked_splits: dict[str, list[int]] = {}
        for split in ("train", "test"):
            indices = split_indices[split]
            if (
                not isinstance(indices, list)
                or not indices
                or any(
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < len(ids)
                    for index in indices
                )
                or len(indices) != len(set(indices))
            ):
                raise ValueError(f"case artifact {split} document indices are invalid")
            checked_splits[split] = list(indices)
        if set(checked_splits["train"]) & set(checked_splits["test"]):
            raise ValueError("case artifact train and test documents overlap")
        if set(checked_splits["train"]) | set(checked_splits["test"]) != set(range(len(ids))):
            raise ValueError("case artifact document split does not partition its identities")

        sequence_map = value["sequence_document_indices"]
        if (
            not isinstance(sequence_map, dict)
            or not isinstance(populations, dict)
            or set(sequence_map) != set(populations)
        ):
            raise ValueError("case artifact document mapping does not match its populations")
        checked_mapping: dict[str, dict[str, list[int]]] = {}
        for population, splits in populations.items():
            mapping = sequence_map[population]
            if (
                not isinstance(splits, dict)
                or not isinstance(mapping, dict)
                or set(mapping)
                != {
                    "train",
                    "test",
                }
            ):
                raise ValueError(f"case artifact {population} document mapping is malformed")
            checked_mapping[population] = {}
            for split in ("train", "test"):
                indices = mapping[split]
                sequences = splits.get(split)
                allowed = set(checked_splits[split])
                if (
                    not isinstance(indices, list)
                    or not isinstance(sequences, list)
                    or len(indices) != len(sequences)
                    or any(
                        isinstance(index, bool)
                        or not isinstance(index, int)
                        or index not in allowed
                        for index in indices
                    )
                ):
                    raise ValueError(
                        f"case artifact {population}.{split} sequence/document alignment is invalid"
                    )
                checked_mapping[population][split] = list(indices)
        return list(ids), checked_splits, checked_mapping

    @property
    def n_obs(self) -> int:
        """Observation alphabet size."""
        return len(self.upos_tags)

    @property
    def n_test_documents(self) -> int:
        """Number of source documents contributing held-out sequences."""
        return len(self.test_document_ids)

    @property
    def n_test_tokens(self) -> int:
        """Held-out labeled tokens — the *n* every reported score needs."""
        return sum(len(seq) for seq in self.test)

    @property
    def cases_present(self) -> set[str]:
        """Case names that actually occur in this population."""
        return {c for seq in self.train + self.test for _, c in seq}

    @property
    def heuristic_gold_agreement_pct(self) -> float:
        """How often the syntactic heuristic matches real morphology."""
        both = self.heuristic_vs_gold["tokens_with_both"]
        return 100.0 * self.heuristic_vs_gold["agreements"] / both


def load_case_artifact(population: str = "derived", path: Path | str | None = None) -> CaseArtifact:
    """Load and validate the vendored case artifact for one population.

    Args:
        population: ``"derived"`` (the gated population) or ``"gold"``.
        path: Artifact location; defaults to ``vendor/ud_ewt_case_counts.json``.

    Returns:
        The validated artifact.

    Raises:
        FileNotFoundError: If the artifact is absent.
        ValueError: If it fails validation.
    """
    resolved = (
        DEFAULT_ARTIFACT
        if path is None
        else resolve_vendored_artifact("ud_ewt_case_counts.json", path=path)
    )
    if not resolved.is_file():
        raise FileNotFoundError(f"{resolved} is absent. Run scripts/generate_case_counts.py.")
    artifact = CaseArtifact(json.loads(resolved.read_text(encoding="utf-8")), population)
    validate_ud_ewt_source_metadata(artifact.provenance)
    return artifact


def build_case_emission(
    artifact: CaseArtifact, *, config: LevelConfig = DEFAULT_LEVEL_CONFIG
) -> np.ndarray:
    """``A``: ``P(UPOS | case)``, shape ``(n_obs, 8)``, column-stochastic.

    Normalized over the **observation** axis for each case, so ``A[:, c]`` sums
    to 1. A matrix summing to 1 along the other axis is the transposed model and
    pymdp accepts it, which is the Stage 2 lesson (``docs/decisions.md`` entry 1).
    """
    smoothing = (
        config.emission_smoothing if config.emission_smoothing is not None else config.smoothing
    )
    emission = np.full((artifact.n_obs, N_CASES), smoothing, dtype=np.float64)
    for sequence in artifact.train:
        for upos, case in sequence:
            emission[artifact.upos_to_idx[upos], CASE_TO_IDX[CaseRole[case]]] += 1.0
    return column_stochastic(emission)


def build_case_transition(
    artifact: CaseArtifact, *, config: LevelConfig = DEFAULT_LEVEL_CONFIG
) -> np.ndarray:
    """``B``: ``P(case_t | case_{t-1})``, shape ``(8, 8)``, column-stochastic.

    This is the level's claim: that case roles follow a learnable order along a
    sentence (subject before object, and so on). It is the transition, not the
    emission — the placement decision decision 1 records.
    """
    transition = np.full((N_CASES, N_CASES), config.smoothing, dtype=np.float64)
    for sequence in artifact.train:
        for (_, previous), (_, current) in pairwise(sequence):
            transition[CASE_TO_IDX[CaseRole[current]], CASE_TO_IDX[CaseRole[previous]]] += 1.0
    return column_stochastic(transition)


def build_case_start(
    artifact: CaseArtifact, *, config: LevelConfig = DEFAULT_LEVEL_CONFIG
) -> np.ndarray:
    """``D``: the case distribution at a sentence's first labeled token."""
    start = np.full(N_CASES, config.smoothing, dtype=np.float64)
    for sequence in artifact.train:
        start[CASE_TO_IDX[CaseRole[sequence[0][1]]]] += 1.0
    return start / start.sum()


def build_case_agent(
    artifact: CaseArtifact,
    *,
    config: LevelConfig = DEFAULT_LEVEL_CONFIG,
    emission: np.ndarray | None = None,
    transition: np.ndarray | None = None,
    start: np.ndarray | None = None,
) -> PymdpAgent:
    """Build the Stage 4 pymdp Agent.

    Args:
        artifact: The vendored statistics.
        config: Hyperparameters.
        emission: Override for ``A``; present so the negative controls run
            through the shipped builder rather than a parallel one.
        transition: Override for ``B``.
        start: Override for ``D``.

    Returns:
        A configured ``pymdp.agent.Agent``.

    Raises:
        ValueError: If an override has the wrong shape.
    """
    from pymdp.agent import Agent

    a = (
        build_case_emission(artifact, config=config)
        if emission is None
        else np.asarray(emission, dtype=np.float64)
    )
    b = (
        build_case_transition(artifact, config=config)
        if transition is None
        else np.asarray(transition, dtype=np.float64)
    )
    d = (
        build_case_start(artifact, config=config)
        if start is None
        else np.asarray(start, dtype=np.float64)
    )

    if a.shape != (artifact.n_obs, N_CASES):
        raise ValueError(f"emission must be ({artifact.n_obs}, {N_CASES}), got {a.shape}")
    if b.shape != (N_CASES, N_CASES):
        raise ValueError(f"transition must be ({N_CASES}, {N_CASES}), got {b.shape}")
    if d.shape != (N_CASES,):
        raise ValueError(f"start must be ({N_CASES},), got {d.shape}")

    return Agent(
        A=[jnp.array(a, dtype=jnp.float32)[jnp.newaxis]],
        B=[jnp.array(b, dtype=jnp.float32)[jnp.newaxis, :, :, jnp.newaxis]],
        C=[jnp.zeros((1, artifact.n_obs))],
        D=[jnp.array(d, dtype=jnp.float32)[jnp.newaxis]],
    )


def decode_case_positions(
    artifact: CaseArtifact,
    emission: np.ndarray,
    transition: np.ndarray,
    start: np.ndarray,
) -> list[bool]:
    """Per-position hits of the forward-filtering decoder, in a fixed order.

    Filtering, not smoothing: position ``t`` sees observations ``1..t`` only
    (decision 8). The return is a position-aligned vector so it can be paired
    with :func:`baseline_case_positions` for a McNemar test — comparing two
    aggregate agreements throws away the pairing that makes the test powerful
    (decision 5).

    Returns:
        One boolean per held-out token, in sentence-then-position order.
    """
    hits: list[bool] = []
    for sequence in artifact.test:
        observations = [artifact.upos_to_idx[u] for u, _ in sequence]
        truth = [CASE_TO_IDX[CaseRole[c]] for _, c in sequence]
        belief = start * emission[observations[0], :]
        belief = belief / belief.sum()
        for position in range(len(observations)):
            if position > 0:
                belief = (transition @ belief) * emission[observations[position], :]
                belief = belief / belief.sum()
            hits.append(bool(int(np.argmax(belief)) == truth[position]))
    return hits


def context_free_baseline(artifact: CaseArtifact) -> dict[str, str]:
    """The comparator: ``argmax P(case | UPOS)``, with no sequence context.

    The strongest model that lacks the mechanism this level claims to add
    (decision 2). Uniform, or a majority-class constant, would be a crippled
    comparator.

    Ties are broken by sorted case name, the same rule
    :func:`ailm.levels.phoneme.per_letter_argmax_baseline` states for its own
    argmax bar. ``Counter.most_common`` breaks them by first-seen insertion
    order instead, which is a property of the artifact's row order rather than of
    the data: a regenerated artifact with identical counts but a different
    sentence order could move this comparator, and with it the Stage 4 margin.

    Returns:
        UPOS tag -> predicted case name. Tags unseen in training fall back to the
        overall majority case, under the same tie-break.
    """
    per_tag: dict[str, Counter] = defaultdict(Counter)
    overall: Counter = Counter()
    for sequence in artifact.train:
        for upos, case in sequence:
            per_tag[upos][case] += 1
            overall[case] += 1
    fallback = _argmax_by_name(overall)
    table = {upos: _argmax_by_name(counts) for upos, counts in per_tag.items()}
    return {upos: table.get(upos, fallback) for upos in artifact.upos_tags}


def _argmax_by_name(counts: Counter) -> str:
    """Most frequent key, ties broken by sorted name rather than insertion order.

    ``max()`` keeps the first maximum, so sorting the items by name first makes
    the winner a function of the counts alone.
    """
    if not counts:
        raise ValueError("no observations to take an argmax over")
    return str(max(sorted(counts.items()), key=lambda kv: kv[1])[0])


def baseline_case_positions(artifact: CaseArtifact) -> list[bool]:
    """Per-position hits of the context-free baseline, aligned with the decoder."""
    table = context_free_baseline(artifact)

    overall = Counter(c for seq in artifact.train for _, c in seq)
    fallback = _argmax_by_name(overall)
    return [
        table.get(upos, fallback) == case for sequence in artifact.test for upos, case in sequence
    ]


def case_agreement(hits: list[bool]) -> float:
    """Fraction of held-out positions matching the oracle labels.

    Named *agreement* rather than accuracy on purpose: five of the eight case
    roles have no morphological ground truth in English EWT, so on the derived
    population this compares two labelings, one of which is a documented
    heuristic. Calling it accuracy would assert the heuristic is correct.

    Raises:
        ValueError: If ``hits`` is empty — an agreement over no positions is not
            a small number, it is undefined.
    """
    if not hits:
        raise ValueError("no positions scored; agreement is undefined")
    return sum(hits) / len(hits)


#: Fillmore's subject-selection hierarchy, highest priority first. From "The
#: Case for Case": if there is an agentive it becomes the subject; otherwise, if
#: there is an instrumental it becomes the subject; otherwise the subject is the
#: objective. Mapped onto this project's inventory, the agentive role is NOM, the
#: instrumental INS, and the objective ACC — the neutral case Fillmore was at
#: pains to distinguish from the notion of direct object.
FILLMORE_SUBJECT_PRIORITY: tuple[CaseRole, ...] = (CaseRole.NOM, CaseRole.INS, CaseRole.ACC)


def fillmore_subject(roles: Sequence[CaseRole | None]) -> int | None:
    """Index of the token Fillmore's hierarchy selects as subject, or ``None``.

    A pure function of an assignment of case roles to a sentence's tokens, so it
    can be applied to whichever assignment is under test — the model's, the
    heuristic oracle's, or a control's — without the selection rule differing
    between them. Ties within a priority level go to the leftmost token, which
    is a stated convention rather than a claim about English.

    Args:
        roles: One entry per token, ``None`` where no role was assigned.

    Returns:
        The index of the selected token, or ``None`` when no token carries any
        role in the hierarchy. ``None`` is a refusal to predict rather than a
        default to token zero: a rule that always names some token cannot be
        distinguished from a positional baseline by any score.
    """
    for role in FILLMORE_SUBJECT_PRIORITY:
        for index, assigned in enumerate(roles):
            if assigned is role:
                return index
    return None


def subject_prediction_accuracy(
    sentences: Iterable[Sequence[tuple[CaseRole | None, str]]],
) -> dict[str, float | int]:
    """How often Fillmore's hierarchy, over given roles, names the UD subject.

    This is an *external* check on a case assignment: the target is the token
    whose dependency relation is ``nsubj``, a label the L4 model never observes
    (:data:`ailm.levels.case.FORBIDDEN_OBSERVATIONS`). Scoring the model's
    UPOS-only assignment against it therefore measures something the model was
    not given.

    The same call scored on the *derived* assignment is the circularity control,
    and it is expected to be near-perfect rather than good: this project's
    heuristic maps ``nsubj`` to NOM directly, so predicting ``nsubj`` from
    derived NOM is predicting a label from its own relabeling. Reporting both
    is what makes the model number interpretable — a model path that also scores
    near-perfectly means the observation alphabet leaked, not that the model is
    excellent. That failure mode has already occurred once in this level's
    history and produced a hundred-percent agreement that confirmed nothing
    ([@sec:case_level]).

    Args:
        sentences: One sequence per sentence of ``(assigned_role, deprel)``
            pairs, in token order.

    Returns:
        ``sentences`` scored, ``predicted`` (sentences where the rule named a
        token), ``correct``, ``accuracy`` over the predicted subset, and
        ``coverage`` — the share of sentences on which the rule ventured a
        prediction at all. Accuracy over the predicted subset and coverage are
        reported separately because a rule that refuses often can look accurate.
    """
    scored = predicted = correct = 0
    for tokens in sentences:
        gold = [index for index, (_role, deprel) in enumerate(tokens) if deprel == "nsubj"]
        if len(gold) != 1:
            # No subject, or more than one: neither is a case the hierarchy is
            # about, and scoring them would mix a selection question with a
            # detection question.
            continue
        scored += 1
        choice = fillmore_subject([role for role, _deprel in tokens])
        if choice is None:
            continue
        predicted += 1
        correct += int(choice == gold[0])
    return {
        "sentences": scored,
        "predicted": predicted,
        "correct": correct,
        "accuracy": round(correct / predicted, 4) if predicted else 0.0,
        "coverage": round(predicted / scored, 4) if scored else 0.0,
    }


#: The three paths the corpus oracle can take to assign a label, in the
#: precedence order it applies them. Named here because the *provenance* of a
#: derived label turns out to determine whether the level's margin appears at
#: all, and no shipped claim partitioned by it.
ORACLE_PATHS: tuple[str, ...] = CASE_ORACLE_PATHS


def oracle_path_strata(test_every: int = 5, min_sequence_len: int = 2) -> list[str]:
    """Which oracle path assigned each derived test token's label.

    **This is analysis, not observation.** The vendored artifact withholds
    ``deprel`` and ``feats[Case]`` on purpose: they are the fields the labels are
    *derived from*, and letting the model see them is the circularity
    :data:`FORBIDDEN_OBSERVATIONS` exists to prevent. This function recomputes
    them from the treebank to *partition the evaluation*, and the partition never
    reaches the model. Reading a field to stratify a result and reading it to
    make a prediction are different acts; only the second is the trap.

    Why it matters. ``stage4_derived_margin_pts`` is a mixture over three
    populations whose labels have different provenance. On the ``deprel`` path
    the label is a deterministic function of a syntactic relation that is itself
    strongly predictable from the UPOS tag the model observes, and that stratum
    carries the entire margin. On the ``morphology`` path, where the label is
    real English case marking rather than a research heuristic, the effect is not
    shown. On the ``adposition_token`` path it reverses. That last name is
    literal: the adposition's own lemma labels the ADP token, not its head. The
    aggregate is arithmetically correct and interpretively a blend of those three.

    The reconstruction mirrors ``scripts/generate_case_counts.py`` exactly --
    same document stride, same minimum sequence length, same precedence -- so the returned
    list aligns element-for-element with ``load_case_artifact("derived").test``
    flattened in order. Callers must assert that alignment rather than assume it;
    a silent misalignment would attribute every token to the wrong stratum while
    still summing to the right total.

    Args:
        test_every: Document-index stride for the held-out split.
        min_sequence_len: Sequences shorter than this are dropped, as upstream.

    Returns:
        One path name per derived test token, in artifact order.
    """
    from ailm.corpus.ud import (
        case_oracle_assignment,
        document_stride_partition,
        load_ud_ewt_dev,
    )

    sentences = load_ud_ewt_dev()
    _document_ids, split_by_document = document_stride_partition(sentences, test_every=test_every)
    strata: list[str] = []
    for sentence in sentences:
        if split_by_document[sentence.document_id] != "test":
            continue
        sequence: list[str] = []
        for token in sentence.tokens:
            case, path = case_oracle_assignment(token)
            if case is not None:
                if path is None:  # pragma: no cover - oracle assigned without a path
                    raise ValueError(
                        "a token carries a derived label but no oracle path claims it; "
                        "case_oracle_assignment returned an incomplete result"
                    )
                sequence.append(path)
        if len(sequence) >= min_sequence_len:
            strata.extend(sequence)
    return strata
