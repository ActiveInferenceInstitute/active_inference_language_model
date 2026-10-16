#!/usr/bin/env python3
"""Record the L3 → L2B lexical-channel experiment — the test of entry 18's criterion.

Thin orchestrator. Computation lives in ``ailm.levels.lexical_channel``; this
script sequences the measurements and writes the ledger. Single writer for every
``lexical_*`` label.

The experiment tests a prediction that was written down **before** it ran
(``docs/decisions.md`` entry 18): the novel-information criterion says this
channel should help, and should do so non-monotonically, because the lexicon is
information the G2P level lacks entirely. Entry 19 records what happened.

The script refuses to record if any of the following fails, because a positive
result needs the same controls a null does:

* the ``alpha = 0`` anchor is not bit-identical to the uncoupled decode;
* the anchor is vacuous, i.e. a positive alpha changes nothing;
* the channel had nothing lexical to say, i.e. it backed off to the
  context-free distribution at most positions.

Usage
-----
    python scripts/record_lexical_channel_claims.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from ailm.benchmarks import exact_mcnemar_p
from ailm.ledger_lock import ledger_lock_path, single_writer_lock
from ailm.levels.lexical_channel import LEXICAL_SWEEP_ALPHAS, lexical_sweep_claim_suffix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_ID = "lexical_channel"
SOURCE = "scripts/record_lexical_channel_claims.py"

#: Backwards-compatible local name; the protocol itself is owned by the model
#: module and shared with its verifier, figure, and manuscript generator.
GRID = LEXICAL_SWEEP_ALPHAS


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)

    from ailm.claims import load_ledger
    from ailm.levels.letter import ALPHA_TO_IDX
    from ailm.levels.lexical_channel import (
        build_lexical_message,
        lexical_message_for,
        sweep_lexical_alpha,
    )
    from ailm.levels.phoneme import (
        PHONES,
        build_g2p_emission,
        build_g2p_start,
        build_g2p_transition,
        g2p_split,
        load_phoneme_counts,
    )

    counts = load_phoneme_counts(PROJECT_ROOT / "vendor")
    train, test = g2p_split(counts)
    emission = np.asarray(build_g2p_emission(train))[0]
    transition = np.asarray(build_g2p_transition(train))[0, :, :, 0]
    start = np.asarray(build_g2p_start(train)).ravel()
    message = build_lexical_message(train, list(PHONES))

    # ── Did the channel have anything lexical to say? ─────────────────────────
    depths: Counter = Counter()
    for word, _ in test:
        _, per_position = lexical_message_for(message, word)
        depths.update(per_position)
    total_positions = sum(depths.values())
    lexical_pct = 100.0 * sum(v for k, v in depths.items() if k > 0) / total_positions
    if lexical_pct < 50.0:
        raise ValueError(
            f"The channel fell back to the context-free distribution at "
            f"{100 - lexical_pct:.1f}% of positions, so it is barely lexical. Any "
            f"null would say nothing about the criterion under test. Refusing to "
            f"record."
        )

    real = sweep_lexical_alpha(
        test, emission, transition, start, message, list(PHONES), ALPHA_TO_IDX, GRID
    )
    scrambled = sweep_lexical_alpha(
        test,
        emission,
        transition,
        start,
        message,
        list(PHONES),
        ALPHA_TO_IDX,
        GRID,
        scrambled=True,
    )

    anchor = real[0.0]
    n_positions = len(anchor)

    # ── The identity anchor, exact rather than approximate ────────────────────
    if anchor != scrambled[0.0]:
        raise ValueError(
            "At alpha = 0 the real and scrambled sweeps disagree, but the mixture "
            "term vanishes identically there so they must be bit-identical. "
            "Refusing to record."
        )
    if real[0.3] == anchor:
        raise ValueError(
            "A positive alpha produced the anchor's exact output, so the anchor is "
            "vacuous and the sweep measures nothing. Refusing to record."
        )

    def accuracy(hits: list[bool]) -> float:
        return 100.0 * sum(hits) / len(hits)

    base = accuracy(anchor)
    rows = []
    for alpha in GRID:
        hits = real[alpha]
        b = sum(1 for x, y in zip(hits, anchor, strict=True) if x and not y)
        c = sum(1 for x, y in zip(hits, anchor, strict=True) if y and not x)
        p = exact_mcnemar_p(b, c)
        rows.append(
            {
                "alpha": alpha,
                "accuracy": accuracy(hits),
                "delta": accuracy(hits) - base,
                "b": b,
                "c": c,
                "p": float(p),
                "scrambled": accuracy(scrambled[alpha]),
            }
        )

    best = max(rows, key=lambda r: r["delta"])
    high_row = max(rows, key=lambda r: r["alpha"])
    interior = 0.0 < best["alpha"] < max(GRID)

    # ── Is the interior optimum a DEMONSTRATED cost of over-weighting? ────────
    # The argmax being interior is an ordering of two sample accuracies. Whether
    # the highest weight is actually worse than the peak is a separate paired
    # question on the same positions, and it has to be recorded next to the
    # ordering or "interior optimum" gets read as "over-weighting hurts" without
    # anything having tested that.
    peak_hits, high_hits = real[best["alpha"]], real[max(GRID)]
    peak_b = sum(1 for x, y in zip(peak_hits, high_hits, strict=True) if x and not y)
    peak_c = sum(1 for x, y in zip(peak_hits, high_hits, strict=True) if y and not x)
    peak_p = exact_mcnemar_p(peak_b, peak_c)

    # A difference between two published accuracies is derived from the PUBLISHED
    # values, not from the full-precision ones. The manuscript prints
    # lexical_best_accuracy and lexical_best_scrambled_accuracy to three places,
    # and a reader who subtracts them must land on the recorded difference;
    # ailm.manuscript_variables.require_close enforces exactly that identity.
    # Rounding the full-precision difference instead double-rounds and can miss
    # by one unit in the last place.
    def published_difference(left: float, right: float) -> float:
        return round(round(left, 3) - round(right, 3), 3)

    recorded: dict[str, float | int] = {
        "lexical_peak_vs_high_alpha_pts": published_difference(
            float(best["accuracy"]), float(high_row["accuracy"])
        ),
        "lexical_peak_vs_high_mcnemar_b": peak_b,
        "lexical_peak_vs_high_mcnemar_c": peak_c,
        "lexical_peak_vs_high_mcnemar_p": float(f"{peak_p:.4g}"),
        "lexical_eval_positions": n_positions,
        "lexical_train_pairs": len(train),
        "lexical_test_pairs": len(test),
        "lexical_prefix_tables": len(message.by_prefix),
        "lexical_positions_with_prefix_pct": round(lexical_pct, 1),
        "lexical_anchor_accuracy": round(base, 3),
        "lexical_best_alpha": best["alpha"],
        "lexical_best_accuracy": round(best["accuracy"], 3),
        "lexical_best_margin_pts": round(best["delta"], 3),
        "lexical_best_mcnemar_b": best["b"],
        "lexical_best_mcnemar_c": best["c"],
        "lexical_best_mcnemar_p": float(f"{best['p']:.4g}"),
        "lexical_best_scrambled_accuracy": round(best["scrambled"], 3),
        "lexical_best_vs_scrambled_pts": published_difference(
            float(best["accuracy"]), float(best["scrambled"])
        ),
        "lexical_high_alpha": max(GRID),
        "lexical_high_alpha_margin_pts": round(high_row["delta"], 3),
        "lexical_high_alpha_accuracy": round(high_row["accuracy"], 3),
        "lexical_optimum_is_interior": int(interior),
    }

    # Record every sweep row so the full table is ledger-backed, not just the
    # best/worst summary. Each label encodes the alpha value (decimal stripped).
    for row in rows:
        a = lexical_sweep_claim_suffix(float(row["alpha"]))
        recorded[f"lexical_sweep_alpha_{a}_accuracy"] = round(row["accuracy"], 3)
        recorded[f"lexical_sweep_alpha_{a}_delta"] = round(row["delta"], 3)
        recorded[f"lexical_sweep_alpha_{a}_scrambled"] = round(row["scrambled"], 3)
        recorded[f"lexical_sweep_alpha_{a}_mcnemar_b"] = row["b"]
        recorded[f"lexical_sweep_alpha_{a}_mcnemar_c"] = row["c"]
        recorded[f"lexical_sweep_alpha_{a}_mcnemar_p"] = float(f"{row['p']:.4g}")

    ledger = load_ledger(PROJECT_ROOT)
    ledger.record_many(recorded, run_id=RUN_ID, source=SOURCE)

    print(f"Recorded {len(recorded)} lexical-channel claims:\n")
    width = max(len(k) for k in recorded)
    for label, value in recorded.items():
        print(f"  {label:<{width}} = {value}")

    print(
        f"\n{'alpha':>6} {'accuracy':>10} {'delta':>9} {'scrambled':>11} "
        f"{'b':>6} {'c':>6} {'p':>10}"
    )
    for row in rows:
        mark = "  <- best" if row["alpha"] == best["alpha"] else ""
        print(
            f"{row['alpha']:6.2f} {row['accuracy']:9.3f}% {row['delta']:+9.3f} "
            f"{row['scrambled']:10.3f}% {row['b']:6d} {row['c']:6d} "
            f"{row['p']:10.3g}{mark}"
        )

    verdict = "CONFIRMED" if best["delta"] > 0 and interior else "NOT CONFIRMED"
    print(
        f"\nEntry 18's prediction: {verdict}. The channel helps by "
        f"{best['delta']:+.3f} pts at alpha={best['alpha']} "
        f"(p = {best['p']:.3g}), and the optimum is "
        f"{'INTERIOR' if interior else 'AT AN ENDPOINT'} — at alpha={max(GRID)} "
        f"the effect is {high_row['delta']:+.3f} pts."
    )
    print(
        f"Peak vs highest weight: {best['accuracy'] - high_row['accuracy']:+.3f} pts, "
        f"b={peak_b} c={peak_c}, p = {peak_p:.3g}. The interior argmax is an "
        f"ordering; this is whether over-weighting is a demonstrated cost."
    )
    print(
        "Contrast with entry 9's channel into the SAME level, which fell "
        "monotonically and reached significance in the direction of harm. Same "
        "receiver, different sender, opposite sign."
    )
    return 0


if __name__ == "__main__":
    # The ledger is single-writer: ``ClaimsLedger`` reads the whole file at
    # construction and writes the whole file at the end, so two overlapping
    # recorders lose one of the two results, with no error on either side.
    with single_writer_lock(ledger_lock_path(PROJECT_ROOT)):
        raise SystemExit(main())
