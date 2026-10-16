"""Threshold-free scoring for the Level 5 null, with its controls.

A reviewer observed that reporting AUC alone leaves the Level 5 result looking
self-contradictory: a rank statistic around 0.70 beside zero minority recall.
It is not contradictory, but the manuscript asserted that rather than measuring
it. `proper_scores` measures it — average precision against prevalence, and a
strictly proper Brier score against the base-rate predictor.

Every test here fixes a property that would let a wrong implementation look
right: average precision that ignores prevalence, a Brier score that a
confidently-wrong model could improve, or a skill score computed against a
reference that peeked at the evaluation labels.
"""

from __future__ import annotations

import math

import pytest

from ailm.levels.clause import proper_scores


def test_perfect_ranking_reaches_one_and_random_ranking_reaches_prevalence():
    """POSITIVE CONTROL at both ends of the range.

    Average precision has no fixed chance level; its floor is the prevalence.
    A test that only checked "AP is high" would pass for an implementation that
    returned AUC by mistake, since both are high on separable data.
    """
    labels = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # 20% prevalence
    perfect = proper_scores([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], labels, 0.2)
    assert perfect["average_precision"] == pytest.approx(1.0)
    assert perfect["prevalence"] == pytest.approx(0.2)

    # Positives ranked last, on distinct scores so no tie group masks the
    # ordering: strictly below prevalence, which a rank-only metric would miss.
    worst = proper_scores(
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        0.2,
    )
    assert worst["average_precision"] < worst["prevalence"]


def test_brier_is_strictly_proper_so_overconfidence_cannot_help():
    """The defining property: only the true probability minimizes the score.

    NEGATIVE CONTROL for a common wrong choice. Log loss is also proper, but
    accuracy is not — a metric that rewarded confident-and-right without
    punishing confident-and-wrong would pass a naive check. Here the
    truth-telling predictor must beat both a hedged and an overconfident one.
    """
    labels = [1] * 30 + [0] * 70
    truth = 0.3
    honest = proper_scores([truth] * 100, labels, 0.5)["brier"]
    for miscalibrated in (0.05, 0.15, 0.5, 0.8, 0.99):
        worse = proper_scores([miscalibrated] * 100, labels, 0.5)["brier"]
        assert worse > honest, f"predicting {miscalibrated} scored no worse than the truth"


def test_skill_is_negative_when_the_posteriors_lose_to_the_base_rate():
    """A negative skill must be reportable, not clamped.

    A wrong implementation that took an absolute value or a max with zero would
    silently convert "worse than knowing nothing" into "no improvement", which
    is precisely the finding this level might have to report.
    """
    labels = [1] * 10 + [0] * 90
    inverted = proper_scores([0.05] * 10 + [0.95] * 90, labels, 0.1)
    assert inverted["brier_skill"] < 0.0
    assert inverted["brier"] > inverted["brier_reference"]


def test_the_reference_predictor_uses_the_prior_it_is_given():
    """The reference must not be recomputed from the evaluation labels.

    The recorder passes the *training* prevalence so the baseline never sees a
    test label. If the function silently substituted the test prevalence, the
    two calls below would return the same reference.
    """
    labels = [1] * 20 + [0] * 80  # test prevalence 0.20
    from_train = proper_scores([0.2] * 100, labels, 0.10)
    from_test = proper_scores([0.2] * 100, labels, 0.20)
    assert from_train["brier_reference"] != from_test["brier_reference"]
    assert from_test["brier_reference"] == pytest.approx(0.2 * 0.8)


@pytest.mark.parametrize(
    ("scores", "labels", "prior"),
    [
        ([0.5], [1, 0], 0.5),  # length mismatch
        ([], [], 0.5),  # empty
        ([0.5, 0.5], [1, 1], 0.5),  # single class
        ([0.5, 0.5], [0, 0], 0.5),  # single class, other side
        ([1.5, 0.5], [1, 0], 0.5),  # out of range
        ([math.nan, 0.5], [1, 0], 0.5),  # non-finite
        ([0.5, 0.5], [1, 0], 0.0),  # degenerate reference
        ([0.5, 0.5], [1, 0], 1.0),  # degenerate reference
    ],
)
def test_malformed_input_fails_closed(scores, labels, prior):
    with pytest.raises(ValueError):
        proper_scores(scores, labels, prior)


def test_ties_do_not_change_the_result_between_runs():
    """Equal posteriors are common here; the ordering must be deterministic."""
    scores = [0.5] * 6
    labels = [1, 0, 1, 0, 1, 0]
    first = proper_scores(scores, labels, 0.5)
    second = proper_scores(list(reversed(scores)), list(reversed(labels)), 0.5)
    assert first["average_precision"] == pytest.approx(second["average_precision"])


def test_the_recorded_level5_values_are_mutually_consistent():
    """The ledger's own numbers must satisfy the relations the prose asserts.

    The manuscript says the model ranks better than chance while being barely
    better calibrated than the base rate. Both halves are checked here against
    the recorded claims, so a re-measurement that broke either one fails rather
    than quietly contradicting the paragraph.
    """
    from pathlib import Path

    from ailm.claims import load_ledger

    ledger = load_ledger(Path(__file__).resolve().parents[1])

    def value(label: str) -> float:
        entry = ledger.get(label)
        assert entry is not None, f"{label} is not in the ledger"
        return float(entry["value"])

    assert value("stage5_average_precision") > value("stage5_positive_prevalence_pct") / 100.0, (
        "average precision no longer exceeds prevalence; the clause section says "
        "the ranking carries information"
    )
    # Decision 83/85 re-based the Stage 5 population on held-out typed EWT
    # sentences; the re-recorded skill is negative and docs/manuscript/07a now says
    # "The negative skill means the fitted posterior is worse than that
    # constant under this proper score." Pin that sign and magnitude band.
    assert -0.1 < value("stage5_brier_skill") < 0.0, (
        "Brier skill left the range the clause section describes as 'worse than "
        "that constant under this proper score'"
    )
    assert value("stage5_brier") > value("stage5_brier_reference")
