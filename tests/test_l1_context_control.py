"""The L1 context-destruction control, and the properties that make it a control.

Decision 73 repairs an attribution: beating the training-estimated unigram bar
on a finite held-out passage does not by itself show the model uses the previous
letter. The repair is a derangement of the trained transition's columns — same
conditionals, wrong contexts — recorded by
``scripts/record_l1_context_control_claims.py``. These tests check the control
differs from its subject in exactly one respect, in the style of
``tests/test_comparators.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ailm.comparators import context_marginal_distribution, derange_transition_columns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEDGER_FILE = PROJECT_ROOT / "data" / "claims.json"

_SKIP_NO_LEDGER = pytest.mark.skipif(
    not LEDGER_FILE.exists(), reason="data/claims.json absent; run the recorders"
)


def _random_column_stochastic(n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.random((n, n)) + 1e-3
    return matrix / matrix.sum(axis=0, keepdims=True)


def _claims() -> dict:
    return json.loads(LEDGER_FILE.read_text(encoding="utf-8"))["claims"]


# ── The derangement must differ from its subject in one respect only ─────────


def test_the_derangement_preserves_the_multiset_of_conditionals():
    """Same columns, same normalization, same parameter count.

    If the derangement changed any conditional, a perplexity increase would be
    explained by the change rather than by the mis-assignment, and the control
    would be measuring itself.
    """
    matrix = _random_column_stochastic(30)
    deranged = derange_transition_columns(matrix)
    assert deranged.shape == matrix.shape
    original_columns = sorted(map(tuple, matrix.T.tolist()))
    deranged_columns = sorted(map(tuple, deranged.T.tolist()))
    assert original_columns == deranged_columns
    np.testing.assert_allclose(deranged.sum(axis=0), np.ones(30), atol=1e-12)


def test_the_derangement_leaves_no_context_its_own_conditional():
    """A fixed point would score some contexts under the correct conditional.

    Generic random columns are distinct, so equality of a column with its
    replacement would mean the rotation has a fixed point — the failure the
    lookup-derangement test guards in its own domain.
    """
    matrix = _random_column_stochastic(30)
    deranged = derange_transition_columns(matrix)
    for j in range(30):
        assert not np.allclose(deranged[:, j], matrix[:, j]), f"context {j} kept its conditional"


def test_the_derangement_refuses_a_matrix_it_cannot_derange():
    """A one-column matrix returned unchanged would be a control equal to its subject."""
    with pytest.raises(ValueError, match="square matrix"):
        derange_transition_columns(np.ones((1, 1)))
    with pytest.raises(ValueError, match="square matrix"):
        derange_transition_columns(np.ones((3, 2)))


def test_destroying_the_context_costs_perplexity_on_contextual_data():
    """POSITIVE CONTROL, on data built so context is everything.

    A two-state chain that always alternates: the true conditionals are peaked
    on the other symbol. The trained chain predicts perfectly; the deranged one
    (which for n=2 swaps the columns) predicts the wrong symbol with the same
    confidence. If the derangement did not hurt here, it hurts nowhere.
    """
    peak = np.array([[0.01, 0.99], [0.99, 0.01]])  # column j: successor of j
    deranged = derange_transition_columns(peak)
    sequence = [0, 1] * 50  # alternating, matching the peaked conditionals

    def chain_ppl(matrix: np.ndarray) -> float:
        predictive = np.full(2, 0.5)
        logs = []
        for idx in sequence:
            logs.append(math.log(max(float(predictive[idx]), 1e-10)))
            predictive = matrix[:, idx]
        return math.exp(-sum(logs) / len(logs))

    assert chain_ppl(deranged) > chain_ppl(peak)


# ── The marginal comparator must be context-free by construction ─────────────


def test_the_context_marginal_is_one_distribution_not_thirty():
    matrix = _random_column_stochastic(30)
    weights = np.full(30, 1.0 / 30)
    marginal = context_marginal_distribution(matrix, weights)
    assert marginal.shape == (30,)
    assert marginal.min() > 0.0
    assert marginal.sum() == pytest.approx(1.0, abs=1e-12)


def test_the_context_marginal_rejects_a_non_distribution():
    matrix = _random_column_stochastic(4)
    with pytest.raises(ValueError, match="probability distribution"):
        context_marginal_distribution(matrix, np.array([0.5, 0.5, 0.5, 0.5]))
    with pytest.raises(ValueError, match="shape mismatch"):
        context_marginal_distribution(matrix, np.full(3, 1.0 / 3))


# ── The recorded claims, verified rather than re-recorded ────────────────────


@_SKIP_NO_LEDGER
def test_the_recorded_control_holds_the_registered_direction():
    """Decision 73's P1, pinned so it cannot quietly evaporate.

    The confirmation flag and the gap it summarizes must agree, and the deranged
    model may never sit below the trained one — the recorder refuses that run,
    so a ledger where it holds means someone wrote the labels by another path.
    """
    claims = _claims()
    needed = (
        "l1_context_trained_perplexity",
        "l1_context_unigram_perplexity",
        "l1_context_derangement_perplexity",
        "l1_context_derangement_minus_unigram",
        "l1_context_control_confirmed",
    )
    if any(label not in claims for label in needed):
        pytest.skip("l1_context_* not recorded in this checkout")
    trained = float(claims["l1_context_trained_perplexity"]["value"])
    unigram = float(claims["l1_context_unigram_perplexity"]["value"])
    deranged = float(claims["l1_context_derangement_perplexity"]["value"])
    gap = float(claims["l1_context_derangement_minus_unigram"]["value"])
    confirmed = int(claims["l1_context_control_confirmed"]["value"])

    assert deranged >= trained, "the recorder's refusal guard was bypassed"
    assert gap == pytest.approx(deranged - unigram, abs=2e-4)
    assert confirmed == int(deranged >= unigram)
    assert confirmed == 1, (
        "decision 73's registered falsifier fired (the deranged model beats the "
        "unigram bar): the manuscript's attribution sentences must be weakened to "
        "comparison claims before this pin is updated"
    )


@_SKIP_NO_LEDGER
def test_the_control_scores_the_gate_population():
    """The fresh trained and unigram values must reproduce the shipped ones.

    Same population-identity claim the comparator-strength probe makes at Stage
    2B and Level 4: if these disagree, every l1_context_* number is about a
    different passage than the gate scores.
    """
    claims = _claims()
    for mine, shipped, tolerance in (
        ("l1_context_trained_perplexity", "stage1_letter_perplexity", 5e-4),
        ("l1_context_unigram_perplexity", "stage1_unigram_baseline", 0.01),
    ):
        if mine not in claims or shipped not in claims:
            pytest.skip(f"{mine} or {shipped} not recorded in this checkout")
        assert abs(float(claims[mine]["value"]) - float(claims[shipped]["value"])) <= tolerance, (
            f"{mine} does not reproduce {shipped}; wrong population"
        )


@_SKIP_NO_LEDGER
def test_the_control_labels_are_recorder_written():
    claims = _claims()
    labels = [label for label in claims if label.startswith("l1_context_")]
    if not labels:
        pytest.skip("l1_context_* not recorded in this checkout")
    for label in labels:
        assert claims[label]["source"].startswith("scripts/"), (
            f"{label} was last written by {claims[label]['source']!r}; recorders "
            f"write, tests verify"
        )
