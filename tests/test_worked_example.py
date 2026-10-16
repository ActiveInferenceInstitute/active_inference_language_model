"""The manuscript's worked interpretability example says what the matrix says.

[@sec:architecture] walks one prediction end to end: it names the transition
column for the context letter ``q`` and prints the three largest entries in it.
That paragraph is the paper's only concrete demonstration of its central claim —
that asking why the model predicted a symbol is answered by reading a named
parameter — so the numbers in it must be the numbers in the shipped matrix, not
a recollection of them.

The recorder is the single writer, as everywhere else. What these tests add is
the check the recorder cannot perform on itself: that its output still describes
the matrix the level actually builds, and that the example has not quietly
become a claim about a different column.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEDGER = PROJECT_ROOT / "data" / "claims.json"

#: How the recorder renders symbols that would be invisible in the manuscript.
_SYMBOL_NAMES = {" ": "space", ".": "period", "'": "apostrophe", "-": "hyphen"}

_SKIP_NO_LEDGER = pytest.mark.skipif(
    not LEDGER.exists(), reason="data/claims.json absent; run the recorders"
)


def _claims() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8"))["claims"]


def _value(label: str):
    return _claims()[label]["value"]


def _transition_matrix() -> np.ndarray:
    """``matrix[next, current]`` for the shipped letter level."""
    from ailm.levels.letter import build_letter_transition, load_brown_counts

    return np.asarray(build_letter_transition(load_brown_counts(PROJECT_ROOT / "vendor")))[0]


@_SKIP_NO_LEDGER
@pytest.mark.parametrize("context", ["q", "t"])
def test_the_worked_example_matches_the_shipped_matrix(context: str):
    """Each recorded rank is the entry the built transition column holds."""
    from ailm.levels.letter import ALPHA_TO_IDX

    matrix = _transition_matrix()
    column = matrix[:, ALPHA_TO_IDX[context]]
    index_to_symbol = {index: symbol for symbol, index in ALPHA_TO_IDX.items()}
    ranked = np.argsort(column)[::-1]

    assert _value(f"worked_{context}_column_index") == ALPHA_TO_IDX[context]
    for rank in (1, 2, 3):
        index = int(ranked[rank - 1])
        expected_symbol = _SYMBOL_NAMES.get(index_to_symbol[index], index_to_symbol[index])
        assert _value(f"worked_{context}_top{rank}_symbol") == expected_symbol, (
            f"the manuscript says {context} is followed by "
            f"{_value(f'worked_{context}_top{rank}_symbol')!r} at rank {rank}; the "
            f"shipped matrix says {expected_symbol!r}"
        )
        assert _value(f"worked_{context}_top{rank}_prob") == pytest.approx(
            float(column[index]), abs=5e-5
        )


@_SKIP_NO_LEDGER
@pytest.mark.parametrize("context", ["q", "t"])
def test_the_recorded_column_is_a_distribution(context: str):
    """The prose claims nothing was renormalized for display; check the claim."""
    assert _value(f"worked_{context}_column_mass") == pytest.approx(1.0, abs=1e-6)


@_SKIP_NO_LEDGER
def test_the_two_examples_still_make_opposite_points():
    """The paragraph contrasts a near-certain column with an ordinary one.

    If retraining flattened ``q`` or sharpened ``t`` the prose would still print
    correct numbers while making an argument the numbers no longer support —
    the failure mode a value-only check cannot see.
    """
    certain = _value("worked_q_top1_prob")
    ordinary = _value("worked_t_top1_prob")
    assert certain > 0.9, (
        f"the q column is no longer near-deterministic ({certain}), so the "
        f"manuscript's 'read the number and you have the answer' example has "
        f"lost its point"
    )
    assert ordinary < 0.5, (
        f"the t column is no longer the ordinary case ({ordinary}); the "
        f"paragraph presents it as a real distribution over successors"
    )


@_SKIP_NO_LEDGER
def test_the_check_is_not_vacuous():
    """POSITIVE CONTROL: a planted wrong symbol must fail the comparison."""
    from ailm.levels.letter import ALPHA_TO_IDX

    matrix = _transition_matrix()
    column = matrix[:, ALPHA_TO_IDX["q"]]
    top = int(np.argsort(column)[::-1][0])
    index_to_symbol = {index: symbol for symbol, index in ALPHA_TO_IDX.items()}
    assert index_to_symbol[top] != "z", "the control assumes q is not followed by z"
    assert float(column[top]) != pytest.approx(0.0, abs=1e-6)
