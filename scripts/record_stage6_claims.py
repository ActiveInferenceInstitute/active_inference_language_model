#!/usr/bin/env python3
"""Record every Stage 6 narrative-level number, against decision 25's predictions.

Thin orchestrator; computation lives in ``ailm.levels.narrative``. Single
writer for every ``stage6_*`` label. Decision 25 pre-registered three
predictions BEFORE this script existed; this recorder measures them and
decision 26 records the outcome, whichever way it lands.

Guards — the recorder refuses to write when
-------------------------------------------
1. **The identity anchor breaks (P2).** The α = 0 channel must return the
   clause level's joint prior EXACTLY, for every phase. A shifted anchor
   means the plumbing moves priors on its own, and no channel measurement
   would mean anything.
2. **The anchor's non-vacuity fails.** At α = 1 the opening-phase prior must
   actually move — a channel that never changes anything satisfies the anchor
   trivially.
3. **The scramble is not a control.** The phase-permuted message must produce
   a different shift from the real message somewhere; if the permutation is
   invisible, the "real" effect never depended on which phase said what.

P1 (recognition) is recorded whichever sign it takes — a null here is a
result, recorded with its paired statistics.

Usage
-----
    .venv/bin/python scripts/record_stage6_claims.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ailm.gates import STAGE6_NARRATIVE_GATE_SESOI_NATS  # noqa: E402
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402
from ailm.statistics import exact_two_sided_binomial_p  # noqa: E402

RUN_ID = "stage6_narrative_gate"
SOURCE = "scripts/record_stage6_claims.py"


def main() -> int:
    from ailm.claims import load_ledger
    from ailm.levels.clause import build_type_models, load_clause_artifact
    from ailm.levels.narrative import (
        NARRATIVE_PHASES,
        build_narrative_agent,
        build_phase_models,
        build_phase_transition,
        channel_shift_summary,
        clause_prior_with_narrative,
        heldout_type_logloss,
        live_agent_type_logloss,
        load_narrative_artifact,
    )

    artifact = load_narrative_artifact()
    models = build_phase_models(artifact.train)
    clause_models = build_type_models(load_clause_artifact().train)

    # ── Guard 1: the α = 0 anchor, exact for every phase ─────────────────────
    from ailm.levels.clause import build_joint_start

    static = build_joint_start(clause_models)
    for phase in range(len(NARRATIVE_PHASES)):
        anchored = clause_prior_with_narrative(clause_models, phase, models, alpha=0.0)
        if not np.array_equal(anchored, static):
            raise SystemExit(
                f"P2 anchor FAILED at phase {phase}: the α=0 channel does not "
                f"reproduce the clause prior exactly "
                f"(max |Δ| = {np.max(np.abs(anchored - static)):.3e})."
            )

    # ── Guard 2 + 3: the channel moves, and the scramble differs ─────────────
    shifts = channel_shift_summary(clause_models, models, alpha=1.0)
    if not abs(shifts["shifts"]["opening"]) > 1e-4:
        raise SystemExit(
            f"P2 non-vacuity FAILED: the α=1 opening-phase shift is "
            f"{shifts['shifts']['opening']:.2e} — the channel changes nothing."
        )
    if all(
        abs(shifts["shifts"][name] - shifts["scrambled_shifts"][name]) < 1e-9
        for name in NARRATIVE_PHASES
    ):
        raise SystemExit(
            "P2 scramble FAILED: permuting the phases is invisible, so the "
            "effect never depended on which phase said what."
        )

    # ── P1: the held-out recognition measurement ─────────────────────────────
    recognition = heldout_type_logloss(artifact.test, models)
    positive = sum(1 for d in recognition["deltas"] if d > 0)
    negative = sum(1 for d in recognition["deltas"] if d < 0)
    sign_p = exact_two_sided_binomial_p(positive, negative)

    # The observed-position association above indexes the true tercile. Record
    # the distinct sequential pymdp estimand as well: predictive loss before
    # each observation, with D reset per document and B carrying the posterior
    # forward. These used to be conflated in prose even though no inference call
    # contributed to the observed-position score.
    live = live_agent_type_logloss(
        artifact.test,
        build_narrative_agent(artifact),
        models["unigram"],
    )
    live_positive = sum(1 for delta in live["deltas"] if delta > 0)
    live_negative = sum(1 for delta in live["deltas"] if delta < 0)
    live_sign_p = exact_two_sided_binomial_p(live_positive, live_negative)

    # Left-to-right structure: tercile never decreases, so in B[next, current]
    # orientation the strictly UPPER triangle must be exactly zero. (The first
    # draft recorded the lower triangle — the orientation trap of decision 1.)
    transition = build_phase_transition(artifact.train)
    above_diagonal = float(np.triu(transition, k=1).sum())
    if above_diagonal != 0.0:
        raise SystemExit(
            f"phase transition carries {above_diagonal:.3e} mass above the "
            f"diagonal — a tercile decreased, which is arithmetically "
            f"impossible; the counting is mis-oriented."
        )

    values: dict[str, float | int] = {
        "stage6_train_documents": len(artifact.train),
        "stage6_test_documents": len(artifact.test),
        "stage6_test_sentences": recognition["n_sentences"],
        "stage6_documents_excluded_short": artifact.provenance["documents_excluded_short"],
        "stage6_phase_logloss": round(recognition["phase_mean"], 4),
        "stage6_unigram_logloss": round(recognition["unigram_mean"], 4),
        "stage6_logloss_improvement_nats": round(recognition["mean_delta"], 4),
        "stage6_sign_test_positive": positive,
        "stage6_sign_test_negative": negative,
        "stage6_sign_test_p": float(sign_p),
        "stage6_live_filter_logloss": round(live["live_mean"], 4),
        "stage6_live_filter_improvement_nats": round(live["mean_delta"], 4),
        "stage6_live_filter_sign_positive": live_positive,
        "stage6_live_filter_sign_negative": live_negative,
        "stage6_live_filter_sign_p": float(live_sign_p),
        "stage6_opening_interrogative_shift": round(shifts["shifts"]["opening"], 4),
        "stage6_middle_interrogative_shift": round(shifts["shifts"]["middle"], 4),
        "stage6_closing_interrogative_shift": round(shifts["shifts"]["closing"], 4),
        "stage6_scrambled_opening_shift": round(shifts["scrambled_shifts"]["opening"], 4),
        "stage6_static_interrogative_mass": round(shifts["static_interrogative_mass"], 4),
        "stage6_transition_above_diagonal": above_diagonal,
        "stage6_narrative_gate_sesoi_nats": STAGE6_NARRATIVE_GATE_SESOI_NATS,
    }

    ledger = load_ledger(_PROJECT_ROOT)
    ledger.record_many(values, run_id=RUN_ID, source=SOURCE, status="descriptive")

    print(f"Recorded {len(values)} Stage 6 claims.")
    for label, value in sorted(values.items()):
        print(f"  {label} = {value}")
    return 0


if __name__ == "__main__":
    # The ledger is single-writer: ``ClaimsLedger`` reads the whole file at
    # construction and writes the whole file at the end, so two overlapping
    # recorders lose one of the two results, with no error on either side.
    with single_writer_lock(ledger_lock_path(_PROJECT_ROOT)):
        raise SystemExit(main())
