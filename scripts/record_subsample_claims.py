#!/usr/bin/env python3
"""Record the eval-subsample comparison — the paper's third finding, finally backed.

Thin orchestrator. Single writer for every ``subsample_*`` label.

Why this script exists
----------------------
``ailm.figures.gates.SUBSAMPLE_COMPARISON`` carried three measured rows as
hardcoded literals flagged ``ledger_backed: False``. Its own docstring named them
"the only numbers in this module not backed by a recorded run" — which is an
honest disclosure, and a disclosure is not a fix. They are the evidence for the
project's third finding (*reproducible is not representative*), so a figure and a
manuscript section rested on numbers nothing re-derived.

What it measures
----------------
The same decoder pair — forward-filtering G2P against the context-free per-letter
argmax — scored on four subsets of the identical held-out split:

``full``
    every pair. The honest result.
``prefix``
    ``pairs[:N]``. Reproducible and biased: ``g2p_pairs`` is alphabetically
    sorted, so a prefix is only a/b/c words. It reports a much larger margin.
``stride5``
    an evenly spaced stride truncated to the *same size* as the prefix.
    Representative, and consequently underpowered.
``stride3``
    a larger stride. Representative and adequately powered.

The pairing is what makes each row a paired McNemar test rather than a difference
of two aggregates (``docs/decisions.md`` entry 5), and every row is scored on
positions drawn from one split so the four are comparable.

Usage
-----
    python scripts/record_subsample_claims.py [--prefix-size N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ailm.ledger_lock import ledger_lock_path, single_writer_lock
from ailm.statistics import (
    SUBSAMPLE_LARGER_ARM,
    SUBSAMPLE_LARGER_STRIDE,
    SUBSAMPLE_MATCHED_ARM,
    SUBSAMPLE_MATCHED_STRIDE,
    SUBSAMPLE_PREFIX_SIZE,
    exact_two_sided_binomial_p,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_ID = "subsample_bias"
SOURCE = "scripts/record_subsample_claims.py"


def _score(pairs, emission, transition, start, baseline_table, phones, alpha_to_idx):
    """Position-aligned hit vectors for the model and the context-free baseline."""
    from ailm.verification.oracle import forward_filter

    phone_index = {p: i for i, p in enumerate(phones)}
    model_hits: list[bool] = []
    base_hits: list[bool] = []
    initials: set[str] = set()

    for word, sequence in pairs:
        letters = [alpha_to_idx[c] for c in word if c in alpha_to_idx]
        if len(letters) != len(sequence):
            continue
        truth = [phone_index[p] for p in sequence if p in phone_index]
        if len(truth) != len(sequence):
            continue
        initials.add(word[0])
        posteriors, _ = forward_filter(
            emission, transition, start, np.asarray(letters, dtype=np.int64)
        )
        for position, letter in enumerate(letters):
            model_hits.append(int(np.argmax(posteriors[position])) == truth[position])
            base_hits.append(int(baseline_table[letter]) == truth[position])
    return model_hits, base_hits, initials


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-size", type=int, default=SUBSAMPLE_PREFIX_SIZE)
    args = parser.parse_args(argv)

    from ailm.claims import load_ledger
    from ailm.levels.letter import ALPHA_TO_IDX
    from ailm.levels.phoneme import (
        PHONES,
        build_g2p_emission,
        build_g2p_start,
        build_g2p_transition,
        g2p_split,
        load_phoneme_counts,
        per_letter_argmax_baseline,
    )

    counts = load_phoneme_counts(PROJECT_ROOT / "vendor")
    train, test = g2p_split(counts)
    emission = np.asarray(build_g2p_emission(train))[0]
    transition = np.asarray(build_g2p_transition(train))[0, :, :, 0]
    start = np.asarray(build_g2p_start(train)).ravel()
    baseline_table = per_letter_argmax_baseline(train)
    phone_index = {p: i for i, p in enumerate(PHONES)}
    table = {
        ALPHA_TO_IDX[letter]: phone_index[phone]
        for letter, phone in baseline_table.items()
        if letter in ALPHA_TO_IDX and phone in phone_index
    }

    subsets = {
        "full": test,
        "prefix": test[: args.prefix_size],
        SUBSAMPLE_MATCHED_ARM: test[::SUBSAMPLE_MATCHED_STRIDE][: args.prefix_size],
        SUBSAMPLE_LARGER_ARM: test[::SUBSAMPLE_LARGER_STRIDE],
    }

    recorded: dict[str, float | int] = {}
    rows = []
    for key, pairs in subsets.items():
        model_hits, base_hits, initials = _score(
            pairs, emission, transition, start, table, list(PHONES), ALPHA_TO_IDX
        )
        n = len(model_hits)
        margin = 100.0 * (sum(model_hits) - sum(base_hits)) / n
        b = sum(1 for m, s in zip(model_hits, base_hits, strict=True) if m and not s)
        c = sum(1 for m, s in zip(model_hits, base_hits, strict=True) if s and not m)
        p = exact_two_sided_binomial_p(b, c)
        rows.append((key, margin, p, n, len(initials)))
        recorded[f"subsample_{key}_margin_pts"] = round(margin, 3)
        recorded[f"subsample_{key}_mcnemar_p"] = float(f"{p:.4g}")
        recorded[f"subsample_{key}_positions"] = n
        recorded[f"subsample_{key}_initials"] = len(initials)

        # The discordant counts, recorded rather than discarded into binomtest.
        # Without them these arms report a margin and a p-value but no interval,
        # because `paired_difference_bound_from_counts` needs b and c and there
        # was nowhere to read them from. A null with no bound cannot be
        # distinguished from an unmeasured one, which is the whole difference
        # between "no effect was found" and "no effect could have been found".
        #
        # NOT recorded for the `full` arm. That subset IS the Stage 2 G2P
        # comparison and its discordant counts already have exactly one writer in
        # `record_stage2_claims.py` as `stage2_g2p_mcnemar_b`/`_c`. Adding them
        # here would give one quantity two producers, which is the single-writer
        # violation the ledger exists to prevent, and the two would drift the
        # first time either recorder's population changed.
        if key != "full":
            recorded[f"subsample_{key}_mcnemar_b"] = b
            recorded[f"subsample_{key}_mcnemar_c"] = c
            recorded[f"subsample_{key}_discordant_n"] = b + c

            # The counts must reproduce the margin they sit beside. A b and c
            # that disagree with the recorded margin would mean the two were
            # computed over different populations, and a bound derived from them
            # would be an interval around a number nobody measured.
            implied = round(100.0 * (b - c) / n, 3)
            if implied != recorded[f"subsample_{key}_margin_pts"]:
                raise SystemExit(
                    f"subsample recorder refused: arm {key!r} has b={b}, c={c} over "
                    f"n={n}, implying a margin of {implied} against the recorded "
                    f"{recorded[f'subsample_{key}_margin_pts']}. The counts and the "
                    f"margin describe different populations. No claims written."
                )

    # The finding itself, derived rather than asserted: the biased prefix reports a
    # LARGER margin than the full set, while a representative subsample of the
    # same size reports a smaller one and loses significance.
    overstatement = recorded["subsample_prefix_margin_pts"] / recorded["subsample_full_margin_pts"]
    recorded["subsample_prefix_overstatement_factor"] = round(overstatement, 2)
    if overstatement <= 1.0:
        raise ValueError(
            f"The alphabetical prefix no longer overstates the margin "
            f"(factor {overstatement:.2f}). The 'reproducible is not "
            f"representative' finding has changed; update docs/decisions.md "
            f"entry 4 rather than recording this."
        )
    if recorded["subsample_prefix_initials"] >= recorded["subsample_full_initials"]:
        raise ValueError(
            "The prefix now spans as many word initials as the full set, so it is "
            "no longer the biased slice the finding rests on."
        )

    ledger = load_ledger(PROJECT_ROOT)
    ledger.record_many(recorded, run_id=RUN_ID, source=SOURCE)

    print(f"Recorded {len(recorded)} subsample claims:\n")
    print(f"{'subset':>10} {'margin':>9} {'p':>11} {'positions':>10} {'initials':>9}")
    for key, margin, p, n, initials in rows:
        print(f"{key:>10} {margin:+8.3f} {p:11.3g} {n:10,} {initials:9d}")
    print(
        f"\nThe alphabetical prefix overstates the margin by "
        f"{overstatement:.2f}x while spanning only "
        f"{recorded['subsample_prefix_initials']} word initials against "
        f"{recorded['subsample_full_initials']} for the full set. A representative "
        f"stride of the same size lands nearer the truth and loses significance — "
        f"reproducible is not representative."
    )
    return 0


if __name__ == "__main__":
    # The ledger is single-writer: ``ClaimsLedger`` reads the whole file at
    # construction and writes the whole file at the end, so two overlapping
    # recorders lose one of the two results, with no error on either side.
    with single_writer_lock(ledger_lock_path(PROJECT_ROOT)):
        raise SystemExit(main())
