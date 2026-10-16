#!/usr/bin/env python3
"""Record the budgeted-attention comparison. Single writer for ``budget_*``.

Thin orchestrator; the computation lives in
``scripts/probe_budgeted_attention.py``.

Why this exists
---------------
Five of these numbers were already in the paper, typed into
`docs/manuscript/05_discussion.md` as bare numerals: the fixed-schedule and adaptive
accuracies, the adaptivity margin, the peeking headroom, and the largest
uniform-stride phase spread. [@sec:reproducibility] states that every number in
the manuscript is a token resolved from the ledger, and these five were not —
they were transcribed by hand from a probe's stdout and no gate noticed, because
`tests/test_claims_discipline.py` scans for `claim:` tags with no backing entry
and a bare numeral has no tag to scan.

Recording them makes the prose statement true rather than aspirational, and
means the numbers move when the measurement does.

Guards — the recorder refuses to write when
-------------------------------------------
1. **The arms are not ordered as the design requires.** Peeking is not a policy
   — it observes the position it is choosing whether to observe — so it must
   score at or above the adaptive arm. If it does not, the peeking arm is not
   the upper bound the section calls it.
2. **The evaluation set is degenerate.** A phase spread needs at least two
   phases per length, and the accuracy comparison needs a substantial position
   count; both arms must be scored over the identical set.
3. **An accuracy is outside (0, 100).** A percentage outside its own range means
   the scoring denominator is wrong.

The adaptivity margin is recorded whichever sign it takes. It is negative: the
train-fitted fixed schedule beats the adaptive policy, and that is the recorded
result the limitations section is built on.

Usage
-----
    .venv/bin/python scripts/record_budgeted_attention_claims.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402

RUN_ID = "budgeted_attention"
SOURCE = "scripts/record_budgeted_attention_claims.py"

#: The budget the limitations section reports. Half the positions observed.
BUDGET_FRACTION = 0.5

#: Smallest evaluation the comparison is allowed to run on. Not a target — a
#: tripwire against a silently truncated split.
MIN_SCORED_POSITIONS = 1_000


def main() -> int:
    from ailm.claims import load_ledger
    from scripts.probe_budgeted_attention import measure_budgeted_attention

    result = measure_budgeted_attention(BUDGET_FRACTION)

    fixed = float(result["fixed_pct"])
    adaptive = float(result["adaptive_pct"])
    peeking = float(result["peeking_pct"])
    spreads: dict = result["phase_spreads"]  # type: ignore[assignment]

    # ── Guard 1: the peeking arm must bound the honest one ───────────────────
    if peeking < adaptive:
        raise SystemExit(
            f"the peeking arm ({peeking:.4f}%) scores below the adaptive arm "
            f"({adaptive:.4f}%). Peeking observes the position it is deciding "
            f"whether to observe, so it cannot be beaten by a policy that does "
            f"not; this ordering failing means the arms are mis-wired."
        )

    # ── Guard 2: the evaluation is not degenerate ────────────────────────────
    scored = int(result["scored_positions"])
    if scored < MIN_SCORED_POSITIONS:
        raise SystemExit(f"only {scored} positions scored; need {MIN_SCORED_POSITIONS}")
    if not spreads:
        raise SystemExit("no phase spreads measured; the stride comparator has no arms")
    for length, spread in spreads.items():
        if int(spread["phases"]) < 2:
            raise SystemExit(
                f"length {length} has {spread['phases']} stride phase(s); a spread "
                f"across one phase is not a spread, and the free parameter the "
                f"section names would be invisible."
            )

    # ── Guard 3: percentages are percentages ─────────────────────────────────
    for name, value in (("fixed", fixed), ("adaptive", adaptive), ("peeking", peeking)):
        if not 0.0 < value < 100.0:
            raise SystemExit(f"{name} accuracy {value} is outside (0, 100)")

    widest_length = max(spreads, key=lambda k: spreads[k]["spread_pts"])
    values: dict[str, float | int] = {
        "budget_fraction_pct": round(100.0 * float(result["budget_fraction"]), 2),
        "budget_scored_positions": scored,
        "budget_train_pairs": int(result["train_pairs"]),
        "budget_test_pairs": int(result["test_pairs"]),
        "budget_fixed_accuracy": round(fixed, 4),
        "budget_adaptive_accuracy": round(adaptive, 4),
        "budget_peeking_accuracy": round(peeking, 4),
        "budget_adaptivity_margin_pts": round(adaptive - fixed, 4),
        "budget_adaptivity_loss_pts": round(fixed - adaptive, 4),
        "budget_peeking_headroom_pts": round(peeking - adaptive, 4),
        "budget_max_phase_spread_pts": round(float(spreads[widest_length]["spread_pts"]), 4),
        "budget_max_phase_spread_length": int(widest_length),
        "budget_lengths_evaluated": len(spreads),
    }
    for length, spread in sorted(spreads.items()):
        values[f"budget_phase_spread_l{length}_pts"] = round(float(spread["spread_pts"]), 4)

    ledger = load_ledger(_PROJECT_ROOT)
    ledger.record_many(values, run_id=RUN_ID, source=SOURCE)

    print(f"Recorded {len(values)} budgeted-attention claims.")
    for label, value in sorted(values.items()):
        print(f"  {label} = {value}")
    print(
        f"\nThe fixed schedule beats the adaptive policy by "
        f"{fixed - adaptive:.4f} points over {scored:,} positions. Recorded as a "
        f"negative result, which is what the limitations section reports."
    )
    return 0


if __name__ == "__main__":
    with single_writer_lock(ledger_lock_path(_PROJECT_ROOT)):
        raise SystemExit(main())
