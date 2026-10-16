#!/usr/bin/env python3
"""Record the numbers behind the manuscript's worked interpretability example.

The paper's central claim about this architecture is that asking why the model
predicted a given symbol is answered by naming a column of ``B`` and reading the
numbers in it. The manuscript asserted that and never showed it. This recorder
supplies the one concrete instance the prose walks through, so the demonstration
is a measurement from the vendored counts rather than an illustration typed by
hand.

The example is the letter level's transition column for the context letter
``q``. It is chosen because the answer is unambiguous and checkable against
ordinary knowledge of English orthography: a reader who knows that ``q`` is
almost always followed by ``u`` can confirm the column is the model's actual
belief, which is the whole point of a model whose parameters are readable. A
second, flatter column (``t``) is recorded alongside it so the example does not
imply that every column is near-deterministic.

Run:  .venv/bin/python scripts/record_worked_example_claims.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ailm.claims import load_ledger  # noqa: E402
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402
from ailm.levels.letter import (  # noqa: E402
    ALPHA_TO_IDX,
    build_letter_transition,
    load_brown_counts,
)

RUN_ID = "worked_example"
SOURCE = "scripts/record_worked_example_claims.py"

#: The two contexts the manuscript walks through: one near-deterministic, one
#: not. Recording both keeps the example honest — a single ``q`` column would
#: suggest the level is more confident than it typically is.
CONTEXTS = ("q", "t")

#: How many ranked successors to record per context. Three is what the prose
#: prints; recording more would put unused numbers in the ledger.
TOP_K = 3

#: Space is a symbol in this alphabet and is a frequent successor. The ledger
#: stores a readable name for it rather than a bare " ", which would be
#: invisible in the rendered manuscript and impossible to diff.
_SYMBOL_NAMES = {" ": "space", ".": "period", "'": "apostrophe", "-": "hyphen"}


def _readable(symbol: str) -> str:
    """A symbol as the manuscript should print it."""
    return _SYMBOL_NAMES.get(symbol, symbol)


def column_predictions(matrix: np.ndarray, context: str) -> list[tuple[str, float]]:
    """The ranked successors the transition column for ``context`` predicts.

    ``matrix[i, j]`` is P(next = i | current = j), so the column indexed by the
    context letter *is* the model's next-symbol distribution. No inference step
    intervenes: this function reads the parameter, which is the property the
    manuscript is demonstrating.
    """
    index_to_symbol = {index: symbol for symbol, index in ALPHA_TO_IDX.items()}
    column = matrix[:, ALPHA_TO_IDX[context]]
    ranked = np.argsort(column)[::-1][:TOP_K]
    return [(index_to_symbol[int(i)], float(column[int(i)])) for i in ranked]


def main() -> int:
    counts = load_brown_counts(_PROJECT_ROOT / "vendor")
    matrix = np.asarray(build_letter_transition(counts))[0]

    recorded: dict[str, object] = {}
    for context in CONTEXTS:
        prefix = f"worked_{context}"
        recorded[f"{prefix}_column_index"] = ALPHA_TO_IDX[context]
        column_mass = float(matrix[:, ALPHA_TO_IDX[context]].sum())
        # The column is a distribution; recording its sum makes the claim that
        # nothing was renormalized for display auditable rather than asserted.
        recorded[f"{prefix}_column_mass"] = round(column_mass, 6)
        for rank, (symbol, probability) in enumerate(column_predictions(matrix, context), start=1):
            recorded[f"{prefix}_top{rank}_symbol"] = _readable(symbol)
            recorded[f"{prefix}_top{rank}_prob"] = round(probability, 4)

    # One atomic batch: a partial worked example is worse than none, because
    # the prose walks a single column and would then quote half of it.
    load_ledger(_PROJECT_ROOT).record_many(recorded, run_id=RUN_ID, source=SOURCE)

    print(f"recorded {len(recorded)} worked-example claims from run {RUN_ID}")
    for label, value in recorded.items():
        print(f"  {label} = {value}")
    return 0


if __name__ == "__main__":
    with single_writer_lock(ledger_lock_path(_PROJECT_ROOT)):
        raise SystemExit(main())
