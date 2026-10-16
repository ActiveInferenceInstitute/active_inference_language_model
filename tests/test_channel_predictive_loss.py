"""Tests for scripts/probe_channel_predictive_loss.py.

This file previously asserted two hardcoded ``True`` literals that the probe
returned without computing anything: the probe imported numpy and five phoneme
symbols it never used, ran a loop whose only body was a counter, discarded both
coupling diagnostics it built, and returned
``novel_and_consequential_inequality_holds: True``. The test's docstring said
"Verify that proper predictive loss is computed on true targets". No loss was
computed. Nothing in the assertion could fail for any reason connected to the
claim, and ``main()`` printed "SUCCESS".

Every assertion below is now a comparison between measured quantities, and each
positive check is paired with a control that must NOT fire.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.probe_channel_predictive_loss import (  # noqa: E402
    MIN_POSITIONS,
    compute_l1_l2b_predictive_loss,
)

#: The recorded letter@evidence cell from decision 46, against the uncoupled
#: anchor. The probe must reproduce these from the full test split, which is what
#: proves it is wired to that channel and not a neighboring one.
RECORDED_ANCHOR_ACCURACY = 70.801
RECORDED_LETTER_EVIDENCE_ACCURACY = 70.5371
RECORDED_LETTER_EVIDENCE_ALPHA = 0.1
RECORDED_POSITIONS = 42820


@pytest.fixture(scope="module")
def result() -> dict:
    """One measurement shared across the assertions — each arm is a full sweep."""
    return compute_l1_l2b_predictive_loss(sample_limit=150)


@pytest.fixture(scope="module")
def full_split() -> dict:
    """The full test split, so the recorded-cell cross-check is comparable.

    Roughly six seconds. A subsample would land on different positions and could
    not be compared against a recorded number measured on all of them.
    """
    return compute_l1_l2b_predictive_loss(sample_limit=100_000)


def test_probe_reproduces_the_recorded_letter_evidence_cell(full_split: dict):
    """Cross-artifact check: the probe must land on the recorded accuracy cell.

    The loss axis is new, but accuracy is already in the ledger for this exact
    channel. If the probe reproduces the recorded accuracy on the recorded
    position count, it is measuring letter@evidence; if it does not, it is
    measuring something else and its loss numbers describe a different channel
    than the docstring claims. This is the check that catches that.
    """
    assert full_split["total_positions_evaluated"] == RECORDED_POSITIONS
    assert full_split["anchor_accuracy"] == pytest.approx(RECORDED_ANCHOR_ACCURACY, abs=5e-4)
    assert full_split["accuracy_by_alpha"][RECORDED_LETTER_EVIDENCE_ALPHA] == pytest.approx(
        RECORDED_LETTER_EVIDENCE_ACCURACY, abs=5e-4
    )


def test_alpha_zero_is_the_uncoupled_decode(full_split: dict):
    """The research queue's void condition, checked where it has content.

    Bit-identity is asserted on the per-position posteriors. The aggregate mean
    is NOT compared exactly: two identical sums accumulated in different orders
    differ in the last ulp, so an exact check there reports a ~1e-15 mismatch
    that says nothing about the model.
    """
    assert full_split["anchor_matches_uncoupled"] is True
    assert full_split["anchor_loss_abs_gap"] < 1e-12


def test_both_axes_agree_that_this_cell_harms(full_split: dict):
    """The research-queue falsifier does not fire on letter@evidence.

    Accuracy and proper log-loss both say the channel harms the receiver. That
    agreement is what makes sec:hierarchy's monotone-decline diagnostic robust
    *on this cell* — and it is measured here rather than argued, because the two
    axes could have disagreed and the queue treats a disagreement as fatal to
    the diagnostic.

    Scope: letter@predictive, the cell that resolves neither way, uses a
    different construction and is not measured by this probe.
    """
    assert full_split["accuracy_says_helps"] is False
    assert full_split["loss_says_helps"] is False
    assert full_split["axes_agree_on_sign"] is True
    assert full_split["accuracy_gain_pts"] < 0.0


def test_proper_loss_is_actually_computed(result: dict):
    """The loss must be a real, finite, position-normalized quantity.

    NEGATIVE CONTROL: the loss must *vary* across alpha. A probe that returned a
    constant would satisfy every "is it finite and positive" check ever written,
    which is how the previous version passed.
    """
    assert result["total_positions_evaluated"] >= MIN_POSITIONS
    assert result["words_evaluated"] > 0

    losses = [result["loss_by_alpha"][a] for a in result["alpha_grid"]]
    assert all(0.0 < value < 50.0 for value in losses), f"implausible losses: {losses}"
    assert len(set(losses)) == len(losses), "loss is constant across alpha — nothing is measured"


def test_oracle_sender_is_exactly_the_anchor(result: dict):
    """The redundancy identity: a perfect letter message adds nothing.

    The sender's latent IS the letter the receiver already observes, so the
    mixture ``(1-a)*onehot + a*onehot`` collapses to ``onehot`` at every weight
    and the oracle arm must reproduce the alpha=0 loss exactly. This is the
    structural check that the mixture does what the docstring says: a non-zero
    spread would mean evidence is being formed some other way.
    """
    assert result["oracle_spread"] == pytest.approx(0.0, abs=1e-12)
    assert result["sensitivity"] == pytest.approx(0.0, abs=1e-12)

    # NEGATIVE CONTROL: the REAL message is not the oracle, so it must move the
    # loss. Without this, an implementation that ignored `arm` entirely would
    # pass the assertion above.
    assert result["readable"] < -1e-6, "the real message left the loss unchanged"


def test_channel_value_inequality_holds_and_the_channel_is_harmful(result: dict):
    """realized <= sensitivity at every engaged alpha, and the recorded null.

    Sensitivity is exactly zero for this channel, so the inequality reduces to
    "no mixture weight improves proper predictive loss". It holds, which means
    the recorded result here is NEGATIVE: the channel degrades the receiver's
    proper loss at every weight that actually engages it.

    The recorded ACCURACY for this same cell also shows harm
    (``claim:channel_letter_evidence_best_margin_pts`` is -0.264), so the two
    axes agree here — see ``test_both_axes_agree_that_this_cell_harms``. The
    cell with a positive accuracy margin is letter@**predictive**, a different
    construction this probe does not implement; nothing here speaks to it.
    """
    assert result["inequality_upper_bound_holds"] is True
    assert result["channel_improves_proper_loss"] is False
    assert result["readable"] <= result["sensitivity"] + 1e-12

    # Engaged alphas only: alpha = 0 has zero gain by definition of the anchor,
    # so including it would make "does not harm" true for a channel that is off.
    engaged = [a for a in result["alpha_grid"] if a > 0.0]
    assert engaged, "no engaged alpha in the grid"
    assert all(result["realized_by_alpha"][a] < 0.0 for a in engaged)


def test_more_mixture_weight_costs_more(result: dict):
    """Smearing the evidence monotonically worsens the loss.

    Predicted by the redundancy argument: every unit of alpha moves mass off the
    letter that was actually observed and onto letters that were not.
    """
    grid = sorted(a for a in result["alpha_grid"])
    losses = [result["loss_by_alpha"][a] for a in grid]
    assert losses == sorted(losses), (
        f"loss not monotone in alpha: {dict(zip(grid, losses, strict=True))}"
    )


def test_scrambled_control_coincides_at_the_anchor_and_diverges_after(result: dict):
    """The control must be identical where it cannot act, and differ where it can.

    At alpha = 0 no message is mixed in, so scrambling it is unobservable — the
    arms must agree exactly. At alpha > 0 they must part, or the "scramble" is
    not reaching the computation.
    """
    assert result["scrambled_loss_by_alpha"][0.0] == pytest.approx(
        result["loss_by_alpha"][0.0], abs=1e-12
    )
    engaged = [a for a in result["alpha_grid"] if a > 0.0]
    assert all(
        abs(result["scrambled_loss_by_alpha"][a] - result["loss_by_alpha"][a]) > 1e-9
        for a in engaged
    ), "the scrambled message is indistinguishable from the real one"


def test_a_grid_without_the_anchor_is_rejected():
    """Every gain is defined against alpha = 0; without it there is no gain.

    NEGATIVE CONTROL for the guard itself: a grid that HAS the anchor but no
    engaged point is also rejected, so the check cannot be satisfied by a
    degenerate single-point sweep.
    """
    with pytest.raises(ValueError, match="anchor"):
        compute_l1_l2b_predictive_loss(sample_limit=20, alpha_grid=(0.1, 0.5))

    with pytest.raises(ValueError, match="engaged"):
        compute_l1_l2b_predictive_loss(sample_limit=20, alpha_grid=(0.0,))
