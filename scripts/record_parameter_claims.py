#!/usr/bin/env python3
"""Record the parameter count of every level and of the whole model.

Thin orchestrator; the computation lives in ``ailm.parameters``. Single writer
for every ``params_*`` label.

Three numbers per level, because one would mislead in three directions: dense
``stored`` entries, structurally ``nonzero`` entries, and ``free`` parameters
after the simplex constraint. See the module docstring of ``ailm.parameters``.

Guards — the recorder refuses to write when
-------------------------------------------
1. **An array contains non-finite or negative probabilities, or does not
   normalize on its declared axis.** Raised inside
   ``count_array`` before any total is formed. A count summed over the wrong
   axis is wrong without looking wrong, and this codebase's decision log
   records the orientation trap more often than any other defect.
2. **The three counts are not ordered.** ``free <= nonzero <= stored`` must
   hold at every level and overall; an inversion means the three were taken
   against different arrays.
3. **A total disagrees with its parts.** The headline number must be the sum of
   the levels, never separately maintained.
4. **The model looks trivial or implausible.** A free count of zero would mean
   no simplex capacity remains, and a count above the loose ceiling
   below would mean a dense array crept into the free column.
5. **A built array disagrees with the GNN file that declares it.** Every level
   is declared as a GNN spec under ``gnn/models/``, and the spec does not
   generate the builder — both are written by hand. A size mismatch means the
   file the manuscript points at as that level's definition describes a
   different model than the one counted, and nothing at runtime would notice.

Usage
-----
    .venv/bin/python scripts/record_parameter_claims.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402

RUN_ID = "model_parameters"
SOURCE = "scripts/record_parameter_claims.py"

#: Loose sanity bounds on the total free-parameter count. Not a target — a
#: tripwire. The model is a few thousand free parameters; a jump past this
#: ceiling means a dense array is being counted as free, and zero means no
#: simplex capacity remains. Capacity is not an estimate of fitted information.
MIN_FREE_PARAMETERS = 1_000
MAX_FREE_PARAMETERS = 1_000_000

#: Ledger label stems, in level order, matching ``ailm.parameters.count_model``.
LEVEL_LABELS: dict[str, str] = {
    "L1_letter": "l1",
    "L2A_phoneme": "l2a",
    "L2B_g2p": "l2b",
    "L3_word": "l3",
    "L4_case": "l4",
    "L5_clause": "l5",
    "L6_narrative": "l6",
}


def main() -> int:
    from ailm.claims import load_ledger
    from ailm.parameters import (
        GNN_AUXILIARY_SPECS,
        GNN_SPEC_FILES,
        check_level_against_gnn,
        check_model_against_gnn,
        count_case_only_clause,
        count_model,
    )

    model = count_model()
    case_only = count_case_only_clause()

    # ── Guard 2: the three counts are ordered ────────────────────────────────
    for level in (*model.levels, case_only):
        if not level.free <= level.nonzero <= level.stored:
            raise SystemExit(
                f"{level.level}: counts are not ordered "
                f"(free={level.free}, nonzero={level.nonzero}, stored={level.stored})"
            )
    if not model.free <= model.nonzero <= model.stored:
        raise SystemExit("model totals are not ordered")

    # ── Guard 3: totals are their parts ──────────────────────────────────────
    for attribute in ("stored", "nonzero", "free"):
        total = getattr(model, attribute)
        parts = sum(getattr(level, attribute) for level in model.levels)
        if total != parts:
            raise SystemExit(f"{attribute}: total {total} != sum of levels {parts}")

    # ── Guard 4: the count is neither trivial nor runaway ────────────────────
    if not MIN_FREE_PARAMETERS <= model.free <= MAX_FREE_PARAMETERS:
        raise SystemExit(
            f"total free parameters {model.free:,} is outside "
            f"[{MIN_FREE_PARAMETERS:,}, {MAX_FREE_PARAMETERS:,}]; inspect the "
            f"live-support simplex count and its array inventory."
        )

    # ── Guard 5: the built model matches what its GNN files declare ─────────
    # The specs and the builders are written by hand, independently. A size that
    # disagrees means the file the manuscript points at as a level's definition
    # describes a different model than the one being counted, and the parameter
    # table would be reporting a model the paper does not describe.
    conformance = check_model_against_gnn(model)
    conformance.extend(
        check_level_against_gnn(case_only, _PROJECT_ROOT / GNN_AUXILIARY_SPECS[case_only.level])
    )
    if conformance:
        raise SystemExit(
            "the built model does not conform to its GNN declarations:\n  "
            + "\n  ".join(conformance)
        )

    by_level = model.by_level()
    values: dict[str, float | int] = {
        "params_total_stored": model.stored,
        "params_total_nonzero": model.nonzero,
        "params_total_free": model.free,
        "params_total_preference_entries": model.preference_entries,
        "params_total_density_pct": round(100.0 * model.nonzero / model.stored, 4),
        "params_levels": len(model.levels),
        "params_gnn_specs": len(GNN_SPEC_FILES),
        "params_gnn_conformance_failures": len(conformance),
    }
    for level_name, stem in LEVEL_LABELS.items():
        level = by_level[level_name]
        values[f"params_{stem}_stored"] = level.stored
        values[f"params_{stem}_nonzero"] = level.nonzero
        values[f"params_{stem}_free"] = level.free
    for attribute in ("stored", "nonzero", "free"):
        values[f"params_l5_case_only_{attribute}"] = getattr(case_only, attribute)

    ledger = load_ledger(_PROJECT_ROOT)
    ledger.record_many(values, run_id=RUN_ID, source=SOURCE)

    print(f"Recorded {len(values)} parameter claims.\n")
    header = f"{'level':<14}{'stored':>12}{'nonzero':>12}{'free':>10}{'density':>10}"
    print(header)
    print("-" * len(header))
    for level_name in LEVEL_LABELS:
        level = by_level[level_name]
        density = 100.0 * level.nonzero / level.stored if level.stored else 0.0
        print(
            f"{level_name:<14}{level.stored:>12,}{level.nonzero:>12,}"
            f"{level.free:>10,}{density:>9.2f}%"
        )
    print("-" * len(header))
    print(
        f"{'TOTAL':<14}{model.stored:>12,}{model.nonzero:>12,}"
        f"{model.free:>10,}{100.0 * model.nonzero / model.stored:>9.2f}%"
    )
    print(
        f"\nPreference (C) entries, excluded from every column above: {model.preference_entries:,}"
    )
    print(
        "Retained L5 case-only ablation, excluded from canonical totals: "
        f"stored={case_only.stored:,}, nonzero={case_only.nonzero:,}, free={case_only.free:,}"
    )
    print(
        f"Every level is declared as a GNN file; {len(GNN_SPEC_FILES)} canonical "
        f"specs and the retained L5 ablation spec checked against their agents, "
        f"{len(conformance)} discrepancies."
    )
    return 0


if __name__ == "__main__":
    with single_writer_lock(ledger_lock_path(_PROJECT_ROOT)):
        raise SystemExit(main())
