#!/usr/bin/env python3
"""Record the L4/L5/L6 live-pymdp inference-equivalence evidence.

Thin orchestrator; computation lives in ``ailm.levels.case``,
``ailm.levels.clause`` and ``ailm.levels.narrative``. Single writer for every
``stage4_pymdp_*``, ``stage5_pymdp_*`` and ``stage6_pymdp_*`` label.

What is being recorded
----------------------
Decision 68 left three residuals: L4's live comparison lived in a test over a
bounded prefix and was never recorded; L5 had matrix-array evidence only (the
gate test walks ``agent.A/B/D`` by hand); L6's live-filter score had no
attribution test. Decision 92 closed the L5 and L6 halves on bounded slices.
Decision 94 makes the L4 and L5 comparisons exhaustive over the held-out test
split and adds the joint-posterior tolerance the argmax statistic alone cannot
supply. This recorder measures, on the canonical held-out splits:

* ``stage4_pymdp_sentences`` / ``stage4_pymdp_positions``: every sentence and
  every labeled token of the derived-oracle test split, not a prefix;
* ``stage4_pymdp_argmax_disagreements``: live ``infer_states`` /
  ``update_empirical_prior`` vs the NumPy forward filter behind the recorded
  Stage 4 agreement, at the case-role posterior argmax, position for position;
* ``stage4_pymdp_posterior_max_abs_dev``: the largest absolute deviation
  between the live and NumPy case-role posteriors over every position;
* ``stage5_pymdp_prefix_sentences`` / ``stage5_pymdp_positions``: every
  sentence and every position of the clause test split. The first label keeps
  the name it carried in the bounded-slice era so the manuscript tokens stay
  stable; its value now counts the whole held-out set;
* ``stage5_pymdp_type_argmax_disagreements``: live inference vs the numpy
  joint forward filter at the sentence-type posterior argmax. On this
  population the filter's type argmax is one type at every position, so a
  zero here is necessary but not sufficient: a transition with no dynamics
  also scores zero (``tests/test_pymdp_l5_l6_equivalence.py`` pins that);
* ``stage5_pymdp_posterior_max_abs_dev``: the largest absolute deviation
  between the live and oracle joint (type x case) posteriors over every
  position. This is the statistic that carries the L5 equivalence;
* ``stage6_pymdp_live_vs_observed_delta_nats``: measured divergence between
  the live sequential estimand and the registered observed-position estimand
  on the same bounded document slice, so the manuscript can say which
  estimand is which with a number attached;
* ``stage6_pymdp_live_vs_observed_sign_agreement``: the share of sentences
  on which the two estimands agree on the sign of the improvement over the
  unigram bar (a [0, 1] fraction; the domain validator reads any
  ``*_agreement`` label as a share, so the raw count is not recorded). They
  index different conditionals by construction, so this is characterization,
  not an equivalence claim.

Guards, the recorder refuses to write when
------------------------------------------
1. The live pymdp walk disagrees with the numpy filter at the posterior argmax
   at L4 or L5 (a non-zero count would mean the equivalence the manuscript
   asserts does not hold; it must be fixed or rewritten, not recorded).
2. The live posterior deviates from the numpy filter by more than
   ``POSTERIOR_ATOL`` at any L4 or L5 position. The argmax guard alone is
   vacuous wherever the filter's argmax never changes; this one is not.
3. The normalization control fails (an unnormalized belief would make the
   argmax comparison vacuous).
4. The inline L4 filter's per-position hits differ from the shipped scorer's
   (the comparison would then be against something other than the recorded
   Stage 4 score).
5. The L6 live and observed runs score different sentence counts.

Usage
-----
    .venv/bin/python scripts/record_pymdp_equivalence_claims.py
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402

RUN_ID = "pymdp_equivalence_gate"
SOURCE = "scripts/record_pymdp_equivalence_claims.py"

#: Absolute tolerance on every per-position posterior comparison. The Agent
#: runs float32 and the numpy filters run float64, so the two may differ by
#: float32 rounding accumulated along a sentence and by nothing more; the
#: measured deviations sit orders of magnitude below this bound, and a
#: transition with no dynamics sits orders of magnitude above it.
POSTERIOR_ATOL = 1e-5

#: Documents in the L6 characterization slice. Decision 92 recorded the L6
#: rows on this slice and the manuscript names it; the L6 rows characterize
#: two different estimands rather than gating an equivalence, so widening
#: them is a separate decision from the exhaustive L4/L5 walks above.
L6_PREFIX_DOCUMENTS = 25


@dataclass(frozen=True)
class WalkComparison:
    """What one live-vs-numpy walk over a held-out split established."""

    sentences: int
    positions: int
    argmax_disagreements: int
    posterior_max_abs_dev: float
    norm_failures: int


def _compare_live_walk(
    agent,
    sequences: Sequence[Sequence[int]],
    numpy_posteriors: Callable[[Sequence[int]], np.ndarray],
    argmax_view: Callable[[np.ndarray], np.ndarray],
    *,
    per_position_hit: Callable[[int, np.ndarray], None] | None = None,
) -> WalkComparison:
    """Walk every sequence through the live Agent beside its numpy filter.

    Args:
        agent: The built pymdp Agent.
        sequences: Observation-index sequences, one per held-out sentence.
        numpy_posteriors: The independent filter; returns one row per position.
        argmax_view: Marginalizes a belief to the space whose argmax is compared
            (identity at L4, the type marginal at L5).
        per_position_hit: Optional hook receiving the running position index
            and the numpy posterior, so a caller can bind the inline filter to
            the shipped scorer.
    """
    import jax.numpy as jnp

    action = jnp.zeros((1, 1), dtype=jnp.int32)
    disagreements = 0
    norm_failures = 0
    positions = 0
    max_abs_dev = 0.0
    for observations in sequences:
        oracle = numpy_posteriors(observations)
        prior = [jnp.asarray(np.asarray(agent.D[0]))]
        for position, observation in enumerate(observations):
            qs = agent.infer_states([jnp.array([observation])], prior)
            belief = np.asarray(qs[0]).ravel()
            if belief.min() < -1e-9 or abs(float(belief.sum()) - 1.0) > 1e-5:
                norm_failures += 1
            live_argmax = int(np.argmax(argmax_view(belief)))
            oracle_argmax = int(np.argmax(argmax_view(oracle[position])))
            if live_argmax != oracle_argmax:
                disagreements += 1
            max_abs_dev = max(max_abs_dev, float(np.max(np.abs(belief - oracle[position]))))
            if per_position_hit is not None:
                per_position_hit(positions, oracle[position])
            prior = agent.update_empirical_prior(action, qs)
            positions += 1
    return WalkComparison(
        sentences=len(sequences),
        positions=positions,
        argmax_disagreements=disagreements,
        posterior_max_abs_dev=max_abs_dev,
        norm_failures=norm_failures,
    )


def _refuse_unless_equivalent(level: str, walk: WalkComparison) -> None:
    if walk.norm_failures != 0:
        raise SystemExit(
            f"{level} normalization guard FAILED: {walk.norm_failures} beliefs were not "
            f"distributions; the argmax comparison would be vacuous."
        )
    if walk.argmax_disagreements != 0:
        raise SystemExit(
            f"{level} equivalence guard FAILED: live pymdp inference disagrees with "
            f"the numpy filter at {walk.argmax_disagreements} of {walk.positions} "
            f"positions. Fix the model mapping or rewrite the equivalence claim; do "
            f"not record a broken equivalence as evidence."
        )
    if walk.posterior_max_abs_dev > POSTERIOR_ATOL:
        raise SystemExit(
            f"{level} posterior guard FAILED: the live posterior deviates from the "
            f"numpy filter by {walk.posterior_max_abs_dev:.3e} at some position, above "
            f"the {POSTERIOR_ATOL:.0e} tolerance. The argmax count cannot see this; "
            f"the equivalence does not hold and must not be recorded."
        )


def _significant(value: float) -> float:
    """Round a deviation to a few significant digits without hiding its scale."""
    return float(f"{value:.3g}")


def main() -> int:
    from ailm.cases import CaseRole
    from ailm.claims import load_ledger
    from ailm.levels.case import (
        CASE_TO_IDX,
        build_case_agent,
        build_case_emission,
        build_case_start,
        build_case_transition,
        decode_case_positions,
        load_case_artifact,
    )
    from ailm.levels.clause import (
        build_clause_agent,
        build_joint_emission,
        build_joint_start,
        build_joint_transition,
        build_type_models,
        load_clause_artifact,
    )
    from ailm.levels.narrative import (
        build_narrative_agent,
        build_phase_models,
        heldout_type_logloss,
        live_agent_type_logloss,
        load_narrative_artifact,
    )
    from ailm.verification.oracle import forward_filter

    # ── L4: live pymdp inference vs the scorer's numpy forward filter ────────
    case_artifact = load_case_artifact("derived")
    case_emission = build_case_emission(case_artifact)
    case_transition = build_case_transition(case_artifact)
    case_start = build_case_start(case_artifact)
    case_agent = build_case_agent(case_artifact)

    def case_filter_posteriors(observations: Sequence[int]) -> np.ndarray:
        # The same recursion as ``decode_case_positions``: filtering, not
        # smoothing, position ``t`` seeing observations ``1..t`` only.
        belief = case_start * case_emission[observations[0], :]
        belief = belief / belief.sum()
        rows = [belief.copy()]
        for observation in observations[1:]:
            belief = (case_transition @ belief) * case_emission[observation, :]
            belief = belief / belief.sum()
            rows.append(belief.copy())
        return np.asarray(rows)

    case_sequences = [
        [case_artifact.upos_to_idx[upos] for upos, _case in sequence]
        for sequence in case_artifact.test
    ]
    case_truth = [
        CASE_TO_IDX[CaseRole[case]] for sequence in case_artifact.test for _upos, case in sequence
    ]
    scorer_hits = decode_case_positions(case_artifact, case_emission, case_transition, case_start)
    filter_drift = 0

    def bind_to_scorer(index: int, numpy_posterior: np.ndarray) -> None:
        nonlocal filter_drift
        if bool(int(np.argmax(numpy_posterior)) == case_truth[index]) != scorer_hits[index]:
            filter_drift += 1

    l4 = _compare_live_walk(
        case_agent,
        case_sequences,
        case_filter_posteriors,
        lambda belief: belief,
        per_position_hit=bind_to_scorer,
    )
    if l4.positions != len(scorer_hits):
        raise SystemExit(
            f"L4 population guard FAILED: the live walk covered {l4.positions} positions "
            f"but the shipped scorer covers {len(scorer_hits)}; the comparison is not "
            f"exhaustive over the recorded population."
        )
    if filter_drift != 0:
        raise SystemExit(
            f"L4 scorer-binding guard FAILED: the inline filter's hits differ from "
            f"``decode_case_positions`` at {filter_drift} positions, so the comparison "
            f"would not be against the recorded Stage 4 score."
        )
    _refuse_unless_equivalent("L4", l4)

    # ── L5: live pymdp inference vs numpy joint oracle, every test sentence ──
    clause_artifact = load_clause_artifact()
    models = build_type_models(clause_artifact.train)
    clause_agent = build_clause_agent(clause_artifact)
    emission = build_joint_emission()
    joint_b = build_joint_transition(models)
    joint_d = build_joint_start(models)

    def joint_oracle_posteriors(observations: Sequence[int]) -> np.ndarray:
        posteriors, _log_evidence = forward_filter(emission, joint_b, joint_d, observations)
        return np.asarray(posteriors)

    def type_marginal(joint_belief: np.ndarray) -> np.ndarray:
        return joint_belief.reshape(2, -1).sum(axis=1)

    l5 = _compare_live_walk(
        clause_agent,
        [sequence[1] for sequence in clause_artifact.test],
        joint_oracle_posteriors,
        type_marginal,
    )
    _refuse_unless_equivalent("L5", l5)

    # ── L6: live sequential vs registered observed-position estimands ────────
    narrative_artifact = load_narrative_artifact()
    narrative_models = build_phase_models(narrative_artifact.train)
    narrative_agent = build_narrative_agent(narrative_artifact)
    slice_documents = narrative_artifact.test[:L6_PREFIX_DOCUMENTS]

    live = live_agent_type_logloss(slice_documents, narrative_agent, narrative_models["unigram"])
    observed = heldout_type_logloss(slice_documents, narrative_models)
    if live["n_sentences"] != observed["n_sentences"]:
        raise SystemExit(
            "L6 slice alignment FAILED: the live and observed runs scored "
            f"{live['n_sentences']} vs {observed['n_sentences']} sentences."
        )
    sign_agreement = sum(
        (live_delta > 0) == (observed_delta > 0)
        for live_delta, observed_delta in zip(live["deltas"], observed["deltas"], strict=True)
    )

    values: dict[str, float | int] = {
        "stage4_pymdp_sentences": l4.sentences,
        "stage4_pymdp_positions": l4.positions,
        "stage4_pymdp_argmax_disagreements": l4.argmax_disagreements,
        "stage4_pymdp_posterior_max_abs_dev": _significant(l4.posterior_max_abs_dev),
        "stage5_pymdp_prefix_sentences": l5.sentences,
        "stage5_pymdp_positions": l5.positions,
        "stage5_pymdp_type_argmax_disagreements": l5.argmax_disagreements,
        "stage5_pymdp_posterior_max_abs_dev": _significant(l5.posterior_max_abs_dev),
        "stage6_pymdp_prefix_sentences": live["n_sentences"],
        "stage6_pymdp_live_logloss": round(live["live_mean"], 4),
        "stage6_pymdp_observed_logloss": round(observed["phase_mean"], 4),
        "stage6_pymdp_live_vs_observed_delta_nats": round(
            live["live_mean"] - observed["phase_mean"], 4
        ),
        "stage6_pymdp_live_vs_observed_sign_agreement": round(
            sign_agreement / live["n_sentences"], 4
        ),
    }

    ledger = load_ledger(_PROJECT_ROOT)
    ledger.record_many(values, run_id=RUN_ID, source=SOURCE, status="descriptive")

    print(f"Recorded {len(values)} pymdp-equivalence claims.")
    for label, value in sorted(values.items()):
        print(f"  {label} = {value}")
    return 0


if __name__ == "__main__":
    # The ledger is single-writer: ``ClaimsLedger`` reads the whole file at
    # construction and writes the whole file at the end, so two overlapping
    # recorders lose one of the two results, with no error on either side.
    with single_writer_lock(ledger_lock_path(_PROJECT_ROOT)):
        raise SystemExit(main())
