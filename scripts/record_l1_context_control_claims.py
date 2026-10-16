#!/usr/bin/env python3
"""Record the Level 1 context-destruction control (decision 73).

Single writer for every ``l1_context_*`` label.

Why this exists. The Stage 1 prose claimed a model "cannot beat unigram letter
frequencies without using the previous letter". That is a comparison claim
dressed as a theorem: the unigram bar is estimated on training counts and scored
on a finite held-out passage, so a different context-free distribution can in
principle land below it through distribution shift alone. Attribution to the
previous letter is an experimental claim, so it gets a control.

The control destroys the context while keeping everything else: the trained
(30, 30) column-stochastic transition has its columns rotated by one over the
index order, preserving the exact multiset of conditional distributions and
mis-assigning every one of them to the wrong context. Beside it, the
context-marginalized distribution (the same columns averaged under the training
unigram weights) is measured as the derived context-free comparator the audit's
objection invokes. Both are scored on the identical authenticated population,
``ailm.evaluation.STAGE1_EVALUATION_TEXT``, that the Stage 1 gate scores.

Prospective development-session prediction (decision 73): the
deranged-context model scores no better than the unigram bar. If it beats the
bar, that is the specified falsifier — recorded with
``l1_context_control_confirmed = 0``, not refused — and the manuscript's
attribution must be weakened to a pure comparison claim. The decision and
recorder first appear in the same commit, so the result is exploratory rather
than independently preregistered.

Refusals: the run writes nothing if its fresh trained perplexity does not
reproduce the ledger's ``stage1_letter_perplexity``, if its fresh unigram bar
does not reproduce ``stage1_unigram_baseline``, or if the deranged model beats
the trained one (mis-assigned context outperforming correctly-assigned context
is a broken measurement, not a finding).

Usage:
    .venv/bin/python scripts/record_l1_context_control_claims.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import jax.numpy as jnp  # noqa: E402

from ailm.claims import load_ledger  # noqa: E402
from ailm.comparators import (  # noqa: E402
    context_marginal_distribution,
    derange_transition_columns,
)
from ailm.evaluation import STAGE1_EVALUATION_TEXT  # noqa: E402
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402
from ailm.levels.letter import (  # noqa: E402
    ALPHA_TO_IDX,
    N_SYMBOLS,
    build_letter_agent,
    build_letter_transition,
    compute_perplexity,
    load_brown_counts,
    normalize_text,
)

RUN_ID = "l1_context_control"
SOURCE = "scripts/record_l1_context_control_claims.py"

#: Tolerance for reproducing the ledger's stage1_letter_perplexity — the same
#: bound tests/test_stage1_gate.py::test_stage1_claim_verified enforces.
TRAINED_TOLERANCE = 5e-4

#: Tolerance for reproducing the shipped unigram bar, matching the comparator
#: probe's population-identity guard in record_comparator_strength_claims.py.
UNIGRAM_TOLERANCE = 0.01


def _gate_positions(text: str) -> list[int]:
    """The exact position set the Stage 1 gate scores: every normalized char."""
    return [ALPHA_TO_IDX[c] for c in normalize_text(text) if c in ALPHA_TO_IDX]


def _closed_form_chain_perplexity(matrix: np.ndarray, positions: list[int]) -> float:
    """Closed-form perplexity of a bigram chain over the gate's position set.

    Mirrors tests/test_stage1_gate.py::_closed_form_bigram_perplexity: position
    0 is scored under the uniform prior and every later position under column
    ``c_{t-1}`` of the matrix — the predictive distribution an identity-emission
    agent produces.
    """
    predictive = np.ones(N_SYMBOLS, dtype=np.float64) / N_SYMBOLS
    log_probs: list[float] = []
    for idx in positions:
        log_probs.append(math.log(max(float(predictive[idx]), 1e-10)))
        predictive = matrix[:, idx]
    return math.exp(-sum(log_probs) / len(log_probs))


def _context_free_perplexity(distribution: np.ndarray, positions: list[int]) -> float:
    """Perplexity of one fixed distribution over the same position set."""
    log_probs = [math.log(max(float(distribution[idx]), 1e-10)) for idx in positions]
    return math.exp(-sum(log_probs) / len(log_probs))


def _training_unigram(counts: dict) -> np.ndarray:
    """The shipped context-free bar's distribution, from the same counts.

    Same construction as tests/test_stage1_gate.py::_unigram_baseline_perplexity
    and scripts/record_stage1_claims.py::unigram_baseline_perplexity: add-1 over
    the current-character marginal of the bigram counts.
    """
    frequency = np.ones(N_SYMBOLS, dtype=np.float64)
    for key, count in counts.items():
        parts = key.split(",", 1) if isinstance(key, str) else list(key)
        if len(parts) != 2:
            continue
        _, curr = parts[0], parts[1]
        if curr in ALPHA_TO_IDX:
            frequency[ALPHA_TO_IDX[curr]] += float(count)
    return frequency / frequency.sum()


def measure() -> dict[str, float | int]:
    counts = load_brown_counts(PROJECT_ROOT / "vendor")
    matrix = np.asarray(build_letter_transition(counts))[0]  # (30, 30) column-stochastic
    positions = _gate_positions(STAGE1_EVALUATION_TEXT)

    # The trained model, through the production metric path, then cross-checked
    # against the closed form so the recorded control numbers describe the same
    # computation the gate performs.
    trained_ppl = compute_perplexity(
        build_letter_agent(build_letter_transition(counts)), STAGE1_EVALUATION_TEXT
    )
    trained_closed = _closed_form_chain_perplexity(matrix, positions)
    if abs(trained_ppl - trained_closed) / trained_closed > 1e-3:
        raise SystemExit(
            f"l1 context control refused: production perplexity {trained_ppl:.6f} and the "
            f"closed-form chain {trained_closed:.6f} disagree beyond 1e-3 relative. The "
            f"control would be scored on a different computation than the gate. "
            f"No claims written."
        )

    # The context-destruction control: same conditionals, wrong contexts. Scored
    # through the same production path (it is a legal transition), cross-checked
    # against the closed form like the trained model.
    deranged = derange_transition_columns(matrix)
    deranged_ppl = compute_perplexity(
        build_letter_agent(jnp.array(deranged[np.newaxis, :, :])), STAGE1_EVALUATION_TEXT
    )
    deranged_closed = _closed_form_chain_perplexity(deranged, positions)
    if abs(deranged_ppl - deranged_closed) / deranged_closed > 1e-3:
        raise SystemExit(
            f"l1 context control refused: deranged production perplexity {deranged_ppl:.6f} "
            f"and its closed form {deranged_closed:.6f} disagree beyond 1e-3 relative. "
            f"No claims written."
        )

    # The two context-free comparators: the shipped count-estimated unigram bar,
    # and the trained transition's own context-marginalized distribution.
    unigram = _training_unigram(counts)
    unigram_ppl = _context_free_perplexity(unigram, positions)
    marginal = context_marginal_distribution(matrix, unigram)
    # Context-freeness by construction: one vector scored at every position.
    marginal_ppl = _context_free_perplexity(marginal, positions)

    # ── Refusals ─────────────────────────────────────────────────────────────
    ledger = load_ledger(PROJECT_ROOT)
    for label, mine, tolerance in (
        ("stage1_letter_perplexity", trained_ppl, TRAINED_TOLERANCE),
        ("stage1_unigram_baseline", unigram_ppl, UNIGRAM_TOLERANCE),
    ):
        entry = ledger.get(label)
        if entry is None:
            raise SystemExit(
                f"l1 context control refused: claim:{label} is absent from the ledger; run "
                f"scripts/record_stage1_claims.py first. No claims written."
            )
        if abs(float(entry["value"]) - mine) > tolerance:
            raise SystemExit(
                f"l1 context control refused: fresh {label} measures {mine:.4f} against the "
                f"ledger's {entry['value']}. The control is not scoring the gate's "
                f"population. No claims written."
            )

    if deranged_ppl < trained_ppl:
        raise SystemExit(
            f"l1 context control refused: the deranged transition ({deranged_ppl:.4f}) scores "
            f"better than the trained one ({trained_ppl:.4f}). Mis-assigned context beating "
            f"correctly-assigned context means the measurement is broken, not that the "
            f"finding is interesting. No claims written."
        )

    # The registered falsifier (decision 73): recorded, flagged, never refused.
    confirmed = int(deranged_ppl >= unigram_ppl)

    return {
        "l1_context_positions": len(positions),
        "l1_context_trained_perplexity": round(trained_ppl, 4),
        "l1_context_unigram_perplexity": round(unigram_ppl, 4),
        "l1_context_derangement_perplexity": round(deranged_ppl, 4),
        "l1_context_marginal_perplexity": round(marginal_ppl, 4),
        "l1_context_derangement_minus_unigram": round(deranged_ppl - unigram_ppl, 4),
        "l1_context_marginal_minus_unigram": round(marginal_ppl - unigram_ppl, 4),
        "l1_context_control_confirmed": confirmed,
    }


def main() -> int:
    recorded = measure()
    ledger = load_ledger(PROJECT_ROOT)
    ledger.record_many(recorded, run_id=RUN_ID, source=SOURCE)

    width = max(len(label) for label in recorded)
    print(f"Recorded {len(recorded)} L1 context-control claims:\n")
    for label, value in recorded.items():
        print(f"  {label:<{width}} = {value}")

    if recorded["l1_context_control_confirmed"]:
        print(
            f"\nP1 held: destroying the context-to-conditional assignment costs "
            f"{recorded['l1_context_derangement_minus_unigram']} perplexity units against "
            f"the unigram bar. The margin over unigram is attributable to "
            f"correctly-assigned previous-letter conditioning."
        )
    else:
        print(
            "\nREGISTERED FALSIFIER FIRED: the deranged-context model beats the unigram "
            "bar. Decision 73's P2 applies — the L1 attribution must be weakened to a "
            "comparison claim in the manuscript. This is a recorded finding, not an error."
        )
    return 0


if __name__ == "__main__":
    # The single-writer lock, taken exactly as every other recorder takes it:
    # two recorders writing the ledger at once is how a label acquires a value
    # from one run and a provenance from another.
    with single_writer_lock(ledger_lock_path(PROJECT_ROOT)):
        raise SystemExit(main())
