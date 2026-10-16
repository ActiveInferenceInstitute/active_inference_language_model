#!/usr/bin/env python3
"""Record the L1 -> L2 coupling measurement into data/claims.json.

**The result recorded here is a NULL, and it is recorded anyway.** Coupling the
letter level into the G2P likelihood as a soft observation costs accuracy at every
positive mixture weight on the fixed grid. A measured null is a result: it is what
licenses correcting the figure label in
``src/ailm/figures/architecture.py`` instead of shipping a coupling that makes the
model worse. Nothing here is tuned toward a pass — see the refusal guards below,
none of which can be satisfied by choosing a friendlier alpha or a friendlier
subsample.

## Why the sweep runs in numpy and not through pymdp

A full-set pymdp pass is ~42.8k ``infer_states`` calls, and the sweep needs six of
them plus six more for the scrambled control. The substitution is licensed exactly
the way ``scripts/record_stage2_claims.py`` licenses it: prove the coupled decoder
at ``alpha = 0`` is *identical* to the shipped pymdp G2P agent on a bounded subset,
and identical to the independent numpy oracle on the full set. Both proofs must
show ZERO disagreements or nothing is recorded.

## Refusal guards

Every one of these voids the whole recording, because each would mean the numbers
describe a different model than the baseline they are compared against:

  1. ``alpha = 0`` does not reproduce the uncoupled oracle decode exactly.
  2. ``alpha = 0`` does not reproduce the pymdp agent's decode on the subset.
  3. ``alpha = 0`` accuracy disagrees with the recorded ``stage2_g2p_accuracy``.
  4. The evaluated position count disagrees with the recorded
     ``stage2_g2p_eval_positions``.
  5. The alpha grid fails its own guard (missing 0.0, or fewer than three distinct
     positive values).
  6. The McNemar self-comparison null reports a discordant pair.
  7. The scrambled-message control is not clearly worse than the real message at
     ``alpha = 1``, which would mean the coupling channel carries no letter-model
     information and the null describes nothing.

Run:  .venv/bin/python scripts/record_coupling_claims.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402

warnings.filterwarnings("ignore", message=".*JAX array is being set as static.*")

from ailm.claims import load_ledger  # noqa: E402
from ailm.levels.coupling import (  # noqa: E402
    DEFAULT_ALPHA_GRID,
    build_g2p_matrices,
    build_letter_matrices,
    couple_letter_to_phoneme,
    letter_predictive_distributions,
    mcnemar,
    sweep_alpha,
    uncoupled_positions,
    validate_alpha_grid,
)
from ailm.levels.letter import ALPHA_TO_IDX, load_brown_counts  # noqa: E402
from ailm.levels.phoneme import (  # noqa: E402
    _iter_pairs,
    _word_letters,
    build_g2p_agent,
    build_g2p_emission,
    build_g2p_start,
    build_g2p_transition,
    decode_phonemes,
    g2p_split,
    load_phoneme_counts,
)

RUN_ID = "coupling_l1_l2_gate"
SOURCE = "scripts/record_coupling_claims.py"

#: Words decoded through pymdp for the production-path identity proof. Bounded for
#: speed; the proof is an identity, so it does not need a representative sample.
PYMDP_WORDS = 200

#: Significance level, matching the Stage 2 G2P gate.
ALPHA_LEVEL = 0.01

#: Tolerance when cross-checking against the recorded Stage 2 accuracy, in points.
STAGE2_TOLERANCE_PTS = 0.01


def _refuse(message: str) -> int:
    """Print a refusal and return the process exit code."""
    print(f"REFUSING TO RECORD: {message}", file=sys.stderr)
    return 1


def _accuracy_pct(positions: list[bool]) -> float:
    """Percentage of correct positions."""
    return 100.0 * sum(positions) / len(positions)


def message_quality(letter_matrices, pairs) -> tuple[float, float, int]:
    """How good is the downward message as a predictor of the observed letter?

    This is the quantitative explanation of the null: the message is better than
    uniform (so the channel is real) and far from certain (so substituting it for
    the observed glyph is expected to cost accuracy).

    Args:
        letter_matrices: The L1 model.
        pairs: ``(word, [phones])`` evaluation items.

    Returns:
        ``(mean_prob_of_observed_letter, argmax_accuracy_pct, n_positions)``.
    """
    probs: list[float] = []
    hits = 0
    for word, _phones in pairs:
        indices = [ALPHA_TO_IDX[c] for c in _word_letters(word)]
        if not indices:
            continue
        predictive = letter_predictive_distributions(letter_matrices, indices)
        for position, index in enumerate(indices):
            probs.append(float(predictive[position, index]))
            hits += int(np.argmax(predictive[position]) == index)
    if not probs:
        return 0.0, 0.0, 0
    return float(np.mean(probs)), 100.0 * hits / len(probs), len(probs)


def verify_pymdp_identity(letter_matrices, g2p_matrices, train_pairs, test_pairs) -> int:
    """Return the number of words where the alpha=0 coupled decode differs from pymdp.

    MUST be zero. The full-set sweep is computed in numpy, which is only
    legitimate while the coupled decoder at its anchor is provably the shipped
    pymdp G2P agent.
    """
    agent = build_g2p_agent(
        build_g2p_emission(train_pairs),
        build_g2p_transition(train_pairs),
        build_g2p_start(train_pairs),
    )
    subset = _iter_pairs(test_pairs, PYMDP_WORDS)
    mismatches = 0
    for word, _phones in subset:
        coupled, _diagnostics = couple_letter_to_phoneme(
            letter_matrices, g2p_matrices, word, alpha=0.0
        )
        if coupled != decode_phonemes(agent, word):
            mismatches += 1
    return mismatches


def main() -> int:
    # The coupling result is defined relative to the recorded Stage 2 anchor.
    # Refuse before the expensive sweep when that producer has not run: making
    # these guards conditional lets a fresh ledger skip them, while a populated
    # ledger silently validates yesterday's values.
    ledger = load_ledger(_PROJECT_ROOT)
    stage2_accuracy = ledger.get("stage2_g2p_accuracy")
    stage2_positions = ledger.get("stage2_g2p_eval_positions")
    missing_stage2 = [
        label
        for label, entry in (
            ("stage2_g2p_accuracy", stage2_accuracy),
            ("stage2_g2p_eval_positions", stage2_positions),
        )
        if entry is None
    ]
    if missing_stage2:
        return _refuse(
            "the recorded Stage 2 anchor is absent; run "
            "scripts/record_stage2_claims.py first. Missing: "
            f"{missing_stage2}."
        )
    if stage2_accuracy is None or stage2_positions is None:
        return _refuse("the Stage 2 anchor preflight did not establish both required entries")

    # ── Guard 5, before anything is measured ─────────────────────────────────
    try:
        grid = validate_alpha_grid(DEFAULT_ALPHA_GRID)
    except Exception as exc:  # GateVacuousError or ValueError
        return _refuse(
            f"the alpha grid {DEFAULT_ALPHA_GRID} failed its own guard ({exc}). A "
            f"sweep that has collapsed cannot support a 'no alpha helped' finding."
        )

    counts = load_phoneme_counts(_PROJECT_ROOT / "vendor")
    train, test = g2p_split(counts)
    letter_matrices = build_letter_matrices(load_brown_counts(_PROJECT_ROOT / "vendor"))
    g2p_matrices = build_g2p_matrices(train)

    # ── The uncoupled reference, via the oracle's hard-observation path ──────
    uncoupled = uncoupled_positions(g2p_matrices, test)
    sweep = sweep_alpha(letter_matrices, g2p_matrices, test, grid)
    scrambled = sweep_alpha(letter_matrices, g2p_matrices, test, grid, scramble=True)

    anchor = sweep[0.0]

    # ── Guard 1: the alpha=0 identity anchor ─────────────────────────────────
    if len(anchor) != len(uncoupled):
        return _refuse(
            f"position counts differ: coupled alpha=0 has {len(anchor)}, the "
            f"uncoupled oracle decode has {len(uncoupled)}. The comparison is not "
            f"paired, so no margin below would be a paired margin."
        )
    identity_diffs = sum(1 for a, b in zip(anchor, uncoupled, strict=True) if a != b)
    if identity_diffs != 0:
        return _refuse(
            f"alpha=0 disagrees with the uncoupled decode at {identity_diffs} of "
            f"{len(anchor)} positions. Every other number here is a comparison "
            f"against alpha=0, so they would all describe a different model than "
            f"the Stage 2 baseline."
        )

    # ── Guard 2: the pymdp production path ───────────────────────────────────
    pymdp_mismatches = verify_pymdp_identity(letter_matrices, g2p_matrices, train, test)
    if pymdp_mismatches != 0:
        return _refuse(
            f"the alpha=0 coupled decode differs from the pymdp G2P agent on "
            f"{pymdp_mismatches} of {PYMDP_WORDS} words. The numpy sweep does not "
            f"describe the shipped agent."
        )

    anchor_accuracy = _accuracy_pct(anchor)

    # ── Guard 3: the anchor equals the recorded Stage 2 accuracy ─────────────
    recorded = float(stage2_accuracy["value"])
    if abs(anchor_accuracy - recorded) > STAGE2_TOLERANCE_PTS:
        return _refuse(
            f"the alpha=0 anchor measured {anchor_accuracy:.4f}% but the "
            f"ledger records stage2_g2p_accuracy as {recorded}%. The coupling "
            f"baseline has drifted from the shipped uncoupled model."
        )

    # ── Guard 4: the same position set Stage 2 evaluated ─────────────────────
    if int(stage2_positions["value"]) != len(anchor):
        return _refuse(
            f"evaluated {len(anchor)} positions but Stage 2 recorded "
            f"{stage2_positions['value']}. The coupling sweep and the Stage 2 "
            f"baseline must cover the same held-out positions."
        )

    # ── Guard 6: the significance machinery must be able to say 'no effect' ──
    null_b, null_c, null_p = mcnemar(anchor, anchor)
    if (null_b, null_c) != (0, 0) or null_p != 1.0:
        return _refuse(
            f"the McNemar self-comparison null reported b={null_b}, c={null_c}, "
            f"p={null_p} instead of 0, 0, 1.0. The pairing is misaligned, so every "
            f"p-value here is an artifact of the procedure."
        )

    # ── The sweep ────────────────────────────────────────────────────────────
    positives = sorted(a for a in sweep if a > 0.0)
    best_alpha = max(positives, key=lambda a: sum(sweep[a]))
    best_accuracy = _accuracy_pct(sweep[best_alpha])
    margin_pts = best_accuracy - anchor_accuracy
    b, c, p_value = mcnemar(sweep[best_alpha], anchor)

    # ── Guard 7: the coupling channel must carry letter information ──────────
    real_at_one = _accuracy_pct(sweep[1.0])
    scrambled_at_one = _accuracy_pct(scrambled[1.0])
    if real_at_one <= scrambled_at_one:
        return _refuse(
            f"at alpha=1.0 the real message scored {real_at_one:.4f}% and the "
            f"reversed-axis control {scrambled_at_one:.4f}%. The coupling channel "
            f"is carrying no letter-model information, so the sweep says nothing "
            f"about coupling — it says the message never arrived. Fix the wiring "
            f"before recording any conclusion about coupling."
        )

    mean_prob, argmax_pct, message_positions = message_quality(letter_matrices, test)
    if message_positions != len(anchor):
        return _refuse(
            f"the message-quality measurement covered {message_positions} "
            f"positions but the sweep covered {len(anchor)}. The explanation and "
            f"the result would describe different position sets."
        )

    supported = int(margin_pts > 0.0 and p_value < ALPHA_LEVEL and b > c)

    # ── Print the sweep table before recording, so the run is auditable ──────
    print(f"Alpha sweep over {len(anchor)} held-out positions (grid fixed in advance at {grid}):\n")
    print(f"  {'alpha':>6}  {'accuracy':>10}  {'vs alpha=0':>11}  {'scrambled':>10}")
    for alpha in sorted(sweep):
        accuracy = _accuracy_pct(sweep[alpha])
        print(
            f"  {alpha:>6}  {accuracy:>9.4f}%  {accuracy - anchor_accuracy:>+10.4f}  "
            f"{_accuracy_pct(scrambled[alpha]):>9.4f}%"
        )

    recorded_claims = {
        "coupling_alpha0_accuracy": round(anchor_accuracy, 4),
        "coupling_best_alpha": best_alpha,
        "coupling_best_accuracy": round(best_accuracy, 4),
        "coupling_margin_pts": round(margin_pts, 4),
        "coupling_mcnemar_b": b,
        "coupling_mcnemar_c": c,
        "coupling_mcnemar_p": float(f"{p_value:.4g}"),
        "coupling_scrambled_accuracy": round(_accuracy_pct(scrambled[best_alpha]), 4),
        "coupling_alpha1_accuracy": round(real_at_one, 4),
        "coupling_alpha1_scrambled_accuracy": round(scrambled_at_one, 4),
        "coupling_message_mean_prob": round(mean_prob, 4),
        "coupling_message_argmax_pct": round(argmax_pct, 4),
        "coupling_eval_positions": len(anchor),
        "coupling_grid_size": len(grid),
        "coupling_alpha0_identity_diffs": identity_diffs,
        "coupling_pymdp_disagreements": pymdp_mismatches,
        "coupling_null_mcnemar_discordant": null_b + null_c,
        "coupling_supported": supported,
    }

    ledger.record_many(recorded_claims, run_id=RUN_ID, source=SOURCE)

    print(f"\nRecorded {len(recorded_claims)} coupling claims into data/claims.json:\n")
    width = max(len(k) for k in recorded_claims)
    for label, value in recorded_claims.items():
        print(f"  {label:<{width}} = {value}")

    verdict = "SUPPORTED" if supported else "NOT SUPPORTED"
    print(
        f"\nVerdict: coupling {verdict}. Best positive alpha {best_alpha} reached "
        f"{best_accuracy:.4f}% against the uncoupled {anchor_accuracy:.4f}% "
        f"({margin_pts:+.4f} pts); paired McNemar b={b} c={c} p={p_value:.3e}."
    )
    print(
        f"Channel non-vacuity: at alpha=1.0 the real message scores "
        f"{real_at_one:.4f}% vs {scrambled_at_one:.4f}% for the reversed control, "
        f"so the message is informative — it is the substitution that costs "
        f"accuracy. The message assigns mean {mean_prob:.4f} to the observed "
        f"letter and its argmax is right {argmax_pct:.2f}% of the time."
    )
    print(
        "Identity anchor: 0 differences vs the oracle over "
        f"{len(anchor)} positions and 0 vs pymdp over {PYMDP_WORDS} words."
    )
    if not supported:
        print(
            "\nRecommendation implied by this measurement: correct the "
            "'update_empirical_prior couples adjacent levels' label in "
            "src/ailm/figures/architecture.py to describe what is actually "
            "implemented. Do not ship this coupling in the pipeline."
        )
    return 0


if __name__ == "__main__":
    # The ledger is single-writer: ``ClaimsLedger`` reads the whole file at
    # construction and writes the whole file at the end, so two overlapping
    # recorders lose one of the two results, with no error on either side.
    with single_writer_lock(ledger_lock_path(_PROJECT_ROOT)):
        raise SystemExit(main())
