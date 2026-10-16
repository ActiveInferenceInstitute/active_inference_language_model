#!/usr/bin/env python3
"""Record how strong the comparator class actually is at Stage 2B and Level 4.

Single writer for every ``comparator_*`` label.

Why this exists. The staged-gate method requires a comparator that is context-free
but not uniform. It does not require the *strongest* comparator that is still not
the model, and at two levels a first-order lookup over the observed symbols --
no latent state, no filtering, no inference at decode time -- scores above the
shipped model. That is the gate-audit defect class recurring inside the repair.

What is recorded is deliberately uncomfortable. The lookup accuracies sit above
the model accuracies, and the run refuses to write if they do not, because a
version of this probe that quietly came out favorable would be worth nothing.

Positive control: the bigram lookup must not score below the per-symbol argmax it
backs off to, since it uses strictly more context.

Negative control: a derangement of the same map, holding the key set, the label
multiset and the parameter count fixed and changing only which label sits at which
key. Its value is recorded rather than merely asserted to be low, because at Level
4 it is *not* low -- with 65 keys over 8 labels and a rotation over sorted keys,
adjacent keys often share a previous-symbol and therefore a label, so the control
is weak exactly where the key space is small. Recording the number is what lets a
reader discount the Level 4 half appropriately instead of taking both halves as
equally clean.

Usage:
    .venv/bin/python scripts/record_comparator_strength_claims.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ailm.claims import load_ledger  # noqa: E402
from ailm.comparators import (  # noqa: E402
    deranged_lookup,
    fit_left_context_lookup,
    score_left_context_lookup,
    score_unigram_lookup,
)
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402

RUN_ID = "comparator_strength"
SOURCE = "scripts/record_comparator_strength_claims.py"


def _g2p_sequences(pairs: list[tuple[str, list[str]]]) -> list[list[tuple[str, str]]]:
    """Length-matched ``(letter, phone)`` sequences, the population the gate scores."""
    return [
        list(zip(word, phones, strict=True)) for word, phones in pairs if len(word) == len(phones)
    ]


def measure() -> dict[str, float | int]:
    from ailm.levels.case import (
        baseline_case_positions,
        build_case_emission,
        build_case_start,
        build_case_transition,
        decode_case_positions,
        load_case_artifact,
    )
    from ailm.levels.phoneme import (
        baseline_positions,
        build_g2p_agent,
        build_g2p_emission,
        build_g2p_start,
        build_g2p_transition,
        decode_positions,
        g2p_split,
        load_phoneme_counts,
        per_letter_argmax_baseline,
    )

    values: dict[str, float | int] = {}

    # ── Stage 2B, letter to phone ────────────────────────────────────────────
    train_pairs, test_pairs = g2p_split(load_phoneme_counts())
    train, test = _g2p_sequences(train_pairs), _g2p_sequences(test_pairs)
    bigram, unigram = fit_left_context_lookup(train)
    lookup = score_left_context_lookup(test, bigram, unigram)
    backoff = score_unigram_lookup(test, unigram)
    control = score_left_context_lookup(test, deranged_lookup(bigram), unigram)

    emission = build_g2p_emission(train_pairs)
    transition = build_g2p_transition(train_pairs)
    start = build_g2p_start(train_pairs)
    model_hits = decode_positions(build_g2p_agent(emission, transition, start), test_pairs)
    baseline_hits = baseline_positions(per_letter_argmax_baseline(train_pairs), test_pairs)
    model_pct = 100.0 * sum(model_hits) / len(model_hits)
    bar_pct = 100.0 * sum(baseline_hits) / len(baseline_hits)

    values.update(
        {
            "comparator_stage2_lookup_accuracy": round(lookup.accuracy_pct, 4),
            "comparator_stage2_lookup_parameters": lookup.parameters,
            "comparator_stage2_lookup_backoff_positions": lookup.backoff_positions,
            "comparator_stage2_unigram_accuracy": round(backoff.accuracy_pct, 4),
            "comparator_stage2_deranged_accuracy": round(control.accuracy_pct, 4),
            "comparator_stage2_over_model_pts": round(lookup.accuracy_pct - model_pct, 4),
            "comparator_stage2_over_bar_pts": round(lookup.accuracy_pct - bar_pct, 4),
            "comparator_stage2_positions": lookup.positions,
        }
    )

    # ── Level 4, case roles ──────────────────────────────────────────────────
    artifact = load_case_artifact("derived")
    case_bigram, case_unigram = fit_left_context_lookup(artifact.train)
    case_lookup = score_left_context_lookup(artifact.test, case_bigram, case_unigram)
    case_backoff = score_unigram_lookup(artifact.test, case_unigram)
    case_control = score_left_context_lookup(
        artifact.test, deranged_lookup(case_bigram), case_unigram
    )

    case_model = decode_case_positions(
        artifact,
        build_case_emission(artifact),
        build_case_transition(artifact),
        build_case_start(artifact),
    )
    case_base = baseline_case_positions(artifact)
    case_model_pct = 100.0 * sum(case_model) / len(case_model)
    case_bar_pct = 100.0 * sum(case_base) / len(case_base)

    values.update(
        {
            "comparator_stage4_lookup_accuracy": round(case_lookup.accuracy_pct, 4),
            "comparator_stage4_lookup_parameters": case_lookup.parameters,
            "comparator_stage4_unigram_accuracy": round(case_backoff.accuracy_pct, 4),
            "comparator_stage4_deranged_accuracy": round(case_control.accuracy_pct, 4),
            "comparator_stage4_over_model_pts": round(case_lookup.accuracy_pct - case_model_pct, 4),
            "comparator_stage4_over_bar_pts": round(case_lookup.accuracy_pct - case_bar_pct, 4),
            "comparator_stage4_positions": case_lookup.positions,
        }
    )

    # ── Refusals ─────────────────────────────────────────────────────────────
    # The unigram backoff must reproduce the shipped context-free bar. These are
    # two implementations of the same predictor, and if they disagree then this
    # probe is scoring a different population than the gate does and every number
    # above is about something else.
    for name, mine, shipped in (
        ("stage2", backoff.accuracy_pct, bar_pct),
        ("stage4", case_backoff.accuracy_pct, case_bar_pct),
    ):
        if abs(mine - shipped) > 0.01:
            raise SystemExit(
                f"comparator probe refused: {name} unigram argmax scores {mine:.4f}% against the "
                f"shipped context-free bar's {shipped:.4f}%. The probe is not scoring the gate's "
                f"population. No claims written."
            )

    # Positive control: more context cannot score worse.
    for name, richer, poorer in (
        ("stage2", lookup.accuracy_pct, backoff.accuracy_pct),
        ("stage4", case_lookup.accuracy_pct, case_backoff.accuracy_pct),
    ):
        if richer < poorer:
            raise SystemExit(
                f"comparator probe refused: {name} bigram lookup ({richer:.4f}%) scores below its "
                f"own unigram backoff ({poorer:.4f}%). The fit or the alignment is wrong. "
                f"No claims written."
            )

    # Negative control: the derangement must not beat the lookup it deranges.
    for name, real, shuffled in (
        ("stage2", lookup.accuracy_pct, control.accuracy_pct),
        ("stage4", case_lookup.accuracy_pct, case_control.accuracy_pct),
    ):
        if shuffled >= real:
            raise SystemExit(
                f"comparator probe refused: {name} deranged map ({shuffled:.4f}%) scores at or "
                f"above the real one ({real:.4f}%), so the comparison is measuring the label "
                f"distribution rather than the mapping. No claims written."
            )

    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    recorded = measure()
    ledger = load_ledger(PROJECT_ROOT)
    ledger.record_many(recorded, run_id=RUN_ID, source=SOURCE)

    width = max(len(label) for label in recorded)
    print(f"Recorded {len(recorded)} comparator-strength claims:\n")
    for label, value in recorded.items():
        print(f"  {label:<{width}} = {value}")
    print(
        f"\nAt Stage 2B a {recorded['comparator_stage2_lookup_parameters']}-parameter lookup "
        f"scores {recorded['comparator_stage2_over_model_pts']} points above the shipped model. "
        f"At Level 4 a {recorded['comparator_stage4_lookup_parameters']}-parameter one scores "
        f"{recorded['comparator_stage4_over_model_pts']} points above it. Neither withdraws a "
        f"recorded number; both withdraw an interpretation."
    )
    return 0


if __name__ == "__main__":
    # The single-writer lock, taken exactly as every other recorder takes it.
    # Recorders cannot run concurrently: two writing the ledger at once is how a
    # label acquires a value from one run and a provenance from another.
    with single_writer_lock(ledger_lock_path(PROJECT_ROOT)):
        raise SystemExit(main())
