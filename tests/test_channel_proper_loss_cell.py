"""The proper-loss cell scorer reads the same posterior rows as the hit scorer.

``run_cell_proper_loss`` promises that its accuracy axis is read off the
identical rows the hit vector comes from. That promise is checked here by
running both scorers on the same cell and the same bounded subsample and
requiring the two accuracies to agree exactly, plus the refusal when a cell
scores no positions at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ailm.channels import run_cell, run_cell_proper_loss

VENDOR = Path(__file__).resolve().parents[1] / "vendor"


@pytest.fixture(scope="module")
def factorial() -> dict:
    """Everything a cell needs, built once from the shipped artifacts."""
    from ailm.levels.coupling import build_g2p_matrices, build_letter_matrices
    from ailm.levels.letter import load_brown_counts
    from ailm.levels.lexical_channel import build_lexical_message
    from ailm.levels.phoneme import PHONES, g2p_split, load_phoneme_counts

    if not (VENDOR / "brown_counts.json").exists() or not (VENDOR / "cmudict_counts.json").exists():
        pytest.skip("vendored counts absent; run the generators")
    train, test = g2p_split(load_phoneme_counts(VENDOR))
    return {
        "letter_matrices": build_letter_matrices(load_brown_counts(VENDOR)),
        "g2p": build_g2p_matrices(train),
        "message": build_lexical_message(train, list(PHONES)),
        "pairs": test,
        "phone_to_index": {p: i for i, p in enumerate(PHONES)},
    }


def test_proper_loss_accuracy_equals_the_hit_vector_on_the_same_rows(factorial) -> None:
    hits = run_cell("letter", "evidence", alpha=0.0, limit=40, **factorial)
    mean_nats, positions, accuracy_pct = run_cell_proper_loss(
        "letter", "evidence", alpha=0.0, limit=40, **factorial
    )
    assert positions == len(hits) > 0
    assert accuracy_pct == pytest.approx(100.0 * float(np.mean(hits)))
    assert mean_nats > 0.0
    assert np.isfinite(mean_nats)


def test_proper_loss_moves_with_the_coupling_weight(factorial) -> None:
    """A cell that reads its message must score differently once alpha is on."""
    anchor = run_cell_proper_loss("lexicon", "predictive", alpha=0.0, limit=40, **factorial)
    coupled = run_cell_proper_loss("lexicon", "predictive", alpha=0.3, limit=40, **factorial)
    assert anchor[1] == coupled[1]
    assert anchor[0] != coupled[0]


def test_proper_loss_refuses_a_cell_with_no_positions(factorial) -> None:
    empty = {**factorial, "pairs": []}
    with pytest.raises(ValueError, match="scored no positions"):
        run_cell_proper_loss("letter", "evidence", alpha=0.0, **empty)
