"""Boundary and inventory tests for the shared gate contract."""

from __future__ import annotations

import math

import pytest

from ailm.gates import (
    AUDITED_GATE_IDS,
    BOUND_COMPARATOR_TYPES,
    BRACKETED_VERDICTS,
    ComparatorType,
    GateId,
    GateVerdict,
    classify_gate,
    evaluate_gate,
    gate_is_baseline_relative,
    gate_margin,
    gate_passes,
    verdict_is_bracketed,
)

BOUND = ComparatorType.MODEL_CLASS_BOUND
REFERENCE = ComparatorType.REFERENCE_MODEL


@pytest.mark.parametrize(
    ("value", "threshold", "direction", "expected"),
    [
        (2.0, 1.0, "higher_is_better", True),
        (1.0, 1.0, "higher_is_better", False),
        (0.0, 1.0, "higher_is_better", False),
        (0.0, 1.0, "lower_is_better", True),
        (1.0, 1.0, "lower_is_better", False),
        (2.0, 1.0, "lower_is_better", False),
    ],
)
def test_gate_pass_is_strict(value, threshold, direction, expected):
    assert gate_passes(value, threshold, direction) is expected


@pytest.mark.parametrize(
    ("baseline", "ceiling", "ceiling_type", "threshold", "direction", "expected"),
    [
        (2.0, 4.0, BOUND, 1.0, "higher_is_better", GateVerdict.VACUOUS),
        (0.0, 1.0, BOUND, 2.0, "higher_is_better", GateVerdict.UNREACHABLE),
        (0.0, 1.0, REFERENCE, 2.0, "higher_is_better", GateVerdict.ABOVE_REFERENCE),
        (0.0, None, None, 2.0, "higher_is_better", GateVerdict.UNAUDITED),
        (0.0, 3.0, BOUND, 2.0, "higher_is_better", GateVerdict.BRACKETED_BOUND),
        (0.0, 3.0, REFERENCE, 2.0, "higher_is_better", GateVerdict.BRACKETED_REFERENCE),
        (0.0, -2.0, BOUND, 1.0, "lower_is_better", GateVerdict.VACUOUS),
        (2.0, 1.0, BOUND, 0.0, "lower_is_better", GateVerdict.UNREACHABLE),
        (2.0, 1.0, REFERENCE, 0.0, "lower_is_better", GateVerdict.ABOVE_REFERENCE),
        (2.0, None, None, 0.0, "lower_is_better", GateVerdict.UNAUDITED),
        (2.0, -1.0, BOUND, 0.0, "lower_is_better", GateVerdict.BRACKETED_BOUND),
        (2.0, -1.0, REFERENCE, 0.0, "lower_is_better", GateVerdict.BRACKETED_REFERENCE),
    ],
)
def test_every_gate_verdict_is_reachable(
    baseline, ceiling, ceiling_type, threshold, direction, expected
):
    assert (
        classify_gate(baseline, ceiling, threshold, direction, ceiling_type=ceiling_type)
        is expected
    )


@pytest.mark.parametrize(
    "ctype",
    [ComparatorType.PRIVILEGED_ORACLE, ComparatorType.REFERENCE_MODEL],
)
def test_only_a_bound_comparator_licenses_an_impossibility_verdict(ctype):
    """A reference or oracle failing the bar must never read as UNREACHABLE.

    ``c_reference < t`` does not imply ``sup over the model class < t``: a
    reference can be exceeded, and the Stage 4 derived row's shipped agent
    exceeding its smoothing reference is the recorded counterexample. Only a
    model-class or analytic bound turns "the ceiling fails" into "nothing can
    pass".
    """
    assert ctype not in BOUND_COMPARATOR_TYPES
    verdict = classify_gate(0.0, 1.0, 2.0, "higher_is_better", ceiling_type=ctype)
    assert verdict is GateVerdict.ABOVE_REFERENCE
    assert verdict is not GateVerdict.UNREACHABLE


@pytest.mark.parametrize("ctype", [ComparatorType.MODEL_CLASS_BOUND, ComparatorType.ANALYTIC_BOUND])
def test_bound_comparators_license_unreachable_and_bracketed_bound(ctype):
    assert ctype in BOUND_COMPARATOR_TYPES
    assert (
        classify_gate(0.0, 1.0, 2.0, "higher_is_better", ceiling_type=ctype)
        is GateVerdict.UNREACHABLE
    )
    assert (
        classify_gate(0.0, 3.0, 2.0, "higher_is_better", ceiling_type=ctype)
        is GateVerdict.BRACKETED_BOUND
    )


def test_a_measured_ceiling_with_an_undeclared_type_fails_closed():
    """An untyped comparator must not silently acquire bound semantics."""
    with pytest.raises(ValueError, match="ceiling_type"):
        classify_gate(0.0, 1.0, 2.0, "higher_is_better")


def test_a_type_declared_for_a_missing_ceiling_fails_closed():
    with pytest.raises(ValueError, match="missing"):
        classify_gate(0.0, None, 2.0, "higher_is_better", ceiling_type=BOUND)


def test_vacuity_needs_no_ceiling_and_no_type():
    """VACUOUS follows from the baseline alone, before any ceiling exists."""
    assert classify_gate(3.0, None, 2.0, "higher_is_better") is GateVerdict.VACUOUS


def test_verdict_is_bracketed_accepts_exactly_the_two_bracketed_verdicts():
    assert {
        GateVerdict.BRACKETED_BOUND,
        GateVerdict.BRACKETED_REFERENCE,
    } == BRACKETED_VERDICTS
    for verdict in GateVerdict:
        assert verdict_is_bracketed(verdict) is (verdict in BRACKETED_VERDICTS)


def test_exact_boundary_has_the_documented_asymmetry():
    """A baseline at the bar fails, and a ceiling at the bar cannot clear it."""
    assert (
        classify_gate(1.0, 2.0, 1.0, "higher_is_better", ceiling_type=BOUND)
        is GateVerdict.BRACKETED_BOUND
    )
    assert (
        classify_gate(0.0, 1.0, 1.0, "higher_is_better", ceiling_type=BOUND)
        is GateVerdict.UNREACHABLE
    )


def test_evaluation_keeps_placement_and_outcome_separate():
    failed = evaluate_gate(
        baseline=10.0,
        model=9.0,
        ceiling=2.0,
        threshold=8.0,
        direction="lower_is_better",
        ceiling_type=REFERENCE,
    )
    assert failed.verdict is GateVerdict.BRACKETED_REFERENCE
    assert failed.passes is False
    assert failed.margin == -1.0


def test_margin_sign_is_bound_to_pass_semantics():
    for direction in ("higher_is_better", "lower_is_better"):
        for value in (0.0, 1.0, 2.0):
            margin = gate_margin(value, 1.0, direction)
            assert (margin > 0.0) is gate_passes(value, 1.0, direction)


@pytest.mark.parametrize("bad", [True, "1", None, math.nan, math.inf, -math.inf])
def test_non_finite_or_non_numeric_metrics_fail_closed(bad):
    with pytest.raises(ValueError):
        gate_passes(bad, 1.0, "higher_is_better")


def test_invalid_direction_fails_closed():
    with pytest.raises(ValueError, match="direction"):
        gate_passes(1.0, 0.0, "sideways")


def test_invalid_comparator_type_fails_closed():
    with pytest.raises(ValueError, match="ceiling_type"):
        classify_gate(0.0, 1.0, 2.0, "higher_is_better", ceiling_type="upper_bound")


def test_audited_gate_inventory_is_exact_and_unique():
    assert tuple(GateId) == AUDITED_GATE_IDS
    assert len(AUDITED_GATE_IDS) == len(set(AUDITED_GATE_IDS)) == 8


def test_a_threshold_equal_to_its_baseline_predetermines_the_verdict():
    """The audit cannot audit a gate restated as "beat the baseline".

    A peer reviewer found this and it is real. `gate_passes` compares strictly,
    and equality-as-failure is load-bearing elsewhere: a baseline exactly on its
    bar must fail, and a ceiling exactly on its bar cannot establish
    reachability. But the restated column of the gate-audit table sets the
    threshold *to* the baseline — "beat the unigram bar", "beat the lookup
    table" — and under strict comparison such a baseline can never clear its own
    bar. VACUOUS becomes logically unreachable, the ceiling alone decides, and
    every restated gate comes out bracketed before anything is measured.

    The verdict is therefore not evidence about those gates. What is evidence
    sits beside it: whether the model clears the baseline, and whether the
    margin is supported.

    NEGATIVE CONTROL: a threshold strictly between baseline and ceiling must
    still be able to come out VACUOUS or UNREACHABLE, or the taxonomy would be
    uninformative everywhere rather than only here.
    """
    baseline, ceiling = 19.9558, 11.5  # lower is better: unigram bar, order-1 floor

    assert gate_is_baseline_relative(baseline, baseline)
    assert not gate_is_baseline_relative(baseline, 18.0)

    # With t == b, no ceiling whatsoever yields VACUOUS.
    verdicts = {
        classify_gate(
            baseline,
            candidate,
            baseline,
            "lower_is_better",
            ceiling_type=None if candidate is None else BOUND,
        )
        for candidate in (11.5, 19.0, 19.9558, 25.0, None)
    }
    assert GateVerdict.VACUOUS not in verdicts, (
        "a baseline-relative gate reported VACUOUS; the strict comparison should "
        "make that unreachable, so this test no longer describes the classifier"
    )
    assert (
        classify_gate(baseline, ceiling, baseline, "lower_is_better", ceiling_type=BOUND)
        is GateVerdict.BRACKETED_BOUND
    )

    # NEGATIVE CONTROL: an ordinary threshold retains all three outcomes.
    assert (
        classify_gate(baseline, ceiling, 21.0, "lower_is_better", ceiling_type=BOUND)
        is GateVerdict.VACUOUS
    )
    assert (
        classify_gate(baseline, ceiling, 10.0, "lower_is_better", ceiling_type=BOUND)
        is GateVerdict.UNREACHABLE
    )
    assert (
        classify_gate(baseline, ceiling, 15.0, "lower_is_better", ceiling_type=BOUND)
        is GateVerdict.BRACKETED_BOUND
    )


def test_the_manuscript_says_the_restated_gates_are_comparisons_not_thresholds():
    """The prose must not present a predetermined verdict as an audited one.

    Written as a separate check from the arithmetic above because the defect
    that reaches a reader is the sentence, not the classifier.
    """
    from pathlib import Path

    gate_design = (
        Path(__file__).resolve().parents[1] / "docs" / "manuscript" / "03_methods.md"
    ).read_text(encoding="utf-8")
    joined = " ".join(gate_design.split())

    assert "gate_is_baseline_relative" in joined, (
        "the gate-design section no longer names the predicate that reports a "
        "predetermined verdict; prose and code have drifted apart"
    )
    assert "is deliberately *not* an extra verdict" in joined, (
        "the section no longer explains why the property is a separate predicate "
        "rather than an additional GateVerdict case"
    )


def test_the_manuscript_types_every_comparator_in_the_verdict_definition():
    """The typed taxonomy must live in the formal definition, not only in a
    table footnote.

    The external audit's sharpest finding was that `UNREACHABLE` was licensed
    by any measured ceiling while the prose admitted some ceilings are
    exceedable references. The repair is only complete when the manuscript's
    verdict definition itself carries the comparator type.
    """
    from pathlib import Path

    gate_design = (
        Path(__file__).resolve().parents[1] / "docs" / "manuscript" / "03_methods.md"
    ).read_text(encoding="utf-8")
    joined = " ".join(gate_design.split())

    for needle in ("ABOVE_REFERENCE", "BRACKETED_BOUND", "BRACKETED_REFERENCE"):
        assert needle in joined, (
            f"the gate-design section no longer defines {needle}; the typed "
            "comparator taxonomy has drifted out of the formal definition"
        )


def test_min_effect_default_reproduces_the_strict_comparison():
    """min_effect=0.0 must change nothing: margin > 0 is exactly gate_passes."""
    for direction in ("higher_is_better", "lower_is_better"):
        for model in (0.5, 1.0, 1.5):
            plain = evaluate_gate(
                baseline=0.0 if direction == "higher_is_better" else 2.0,
                model=model,
                ceiling=None,
                threshold=1.0,
                direction=direction,
            )
            assert plain.passes is gate_passes(model, 1.0, direction)


def test_min_effect_raises_the_outcome_bar_without_touching_placement():
    """A declared smallest effect of interest gates the OUTCOME, not the verdict.

    The external audit's requirement: "beats baseline by enough to matter", so
    a margin inside (0, min_effect] is a fail even though it is positive.
    """
    inside = evaluate_gate(
        baseline=10.0,
        model=9.6,
        ceiling=2.0,
        threshold=10.0,
        direction="lower_is_better",
        ceiling_type=ComparatorType.MODEL_CLASS_BOUND,
        min_effect=0.5,
    )
    assert inside.margin == pytest.approx(0.4)
    assert inside.passes is False
    assert inside.verdict is GateVerdict.BRACKETED_BOUND

    beyond = evaluate_gate(
        baseline=10.0,
        model=9.0,
        ceiling=2.0,
        threshold=10.0,
        direction="lower_is_better",
        ceiling_type=ComparatorType.MODEL_CLASS_BOUND,
        min_effect=0.5,
    )
    assert beyond.passes is True


def test_min_effect_boundary_is_strict_and_negative_fails_closed():
    exact = evaluate_gate(
        baseline=0.0,
        model=1.5,
        ceiling=None,
        threshold=1.0,
        direction="higher_is_better",
        min_effect=0.5,
    )
    assert exact.margin == pytest.approx(0.5)
    assert exact.passes is False  # equality with the declared effect fails

    with pytest.raises(ValueError, match="min_effect"):
        evaluate_gate(
            baseline=0.0,
            model=2.0,
            ceiling=None,
            threshold=1.0,
            direction="higher_is_better",
            min_effect=-0.1,
        )
