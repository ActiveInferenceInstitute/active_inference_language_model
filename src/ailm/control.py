"""Control harness for AILM's validation doctrine.

Every gate in AILM requires:
1. A positive control — a test that PASSES with a good model.
2. A negative control — the SAME test run on a BROKEN model, which must FAIL.

If the negative control passes (broken model is not detected), the metric is
measuring nothing. This harness makes that explicit and raises loudly.

Usage:
    harness = ControlHarness("letter_perplexity_gate")
    # Assert the gate can distinguish good from bad
    harness.assert_gate_is_live(
        good_value=12.5,   # good model perplexity (lower = better)
        bad_value=28.9,    # broken model perplexity
        direction="lower_is_better",
    )
    # Actually run the broken model and check it's caught
    harness.verify_negative_control_fails(
        metric_fn=compute_perplexity,
        broken_input=shuffled_corpus,
        threshold=20.0,
        direction="lower_is_better",
    )
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ailm.gates import Direction, gate_passes

logger = logging.getLogger(__name__)


class GateVacuousError(Exception):
    """Raised when a gate cannot distinguish good from bad.

    This means the negative control PASSES — the broken model is not detected.
    The gate is measuring nothing useful. Fix the metric, not the gate.
    """


class ControlHarness:
    """Harness for validating that a gate is live (can actually fail).

    Args:
        gate_name: Human-readable name for logging and error messages.
    """

    def __init__(self, gate_name: str) -> None:
        self.gate_name = gate_name

    def check_metric_is_sensitive(
        self,
        good_value: float,
        bad_value: float,
        direction: Direction = "higher_is_better",
    ) -> bool:
        """Check that the metric discriminates between good and bad.

        Args:
            good_value: Metric value for the good model.
            bad_value: Metric value for the broken/random model.
            direction: "higher_is_better" or "lower_is_better".

        Returns:
            True if good_value is better than bad_value.

        Raises:
            GateVacuousError: If good_value == bad_value (gate cannot distinguish).
            ValueError: If direction is invalid.
        """
        if good_value == bad_value:
            raise GateVacuousError(
                f"Gate '{self.gate_name}' is VACUOUS: good_value == bad_value == {good_value}. "
                f"The metric cannot distinguish a good model from a broken one. "
                f"Fix the metric — the gate is not measuring anything useful."
            )

        is_better = gate_passes(good_value, bad_value, direction)

        if not is_better:
            logger.warning(
                "Gate '%s': good_value=%.4f is NOT better than bad_value=%.4f "
                "(direction=%s). Gate failed to discriminate.",
                self.gate_name,
                good_value,
                bad_value,
                direction,
            )
        else:
            logger.debug(
                "Gate '%s': good=%.4f better than bad=%.4f (direction=%s). Sensitive.",
                self.gate_name,
                good_value,
                bad_value,
                direction,
            )
        return is_better

    def verify_negative_control_fails(
        self,
        metric_fn: Callable[[Any], float],
        broken_input: Any,
        threshold: float,
        direction: Direction = "higher_is_better",
    ) -> bool:
        """Run metric_fn on a broken input and verify the metric is WORSE than threshold.

        The negative control must FAIL (produce a bad metric). If it passes
        (broken input gives a good metric), the gate is vacuous.

        Args:
            metric_fn: The metric function to evaluate.
            broken_input: Input that should produce a bad metric (e.g., shuffled corpus).
            threshold: The pass/fail threshold. Broken input must be on the WRONG side.
            direction: "higher_is_better" or "lower_is_better".

        Returns:
            True if the broken input produces a metric WORSE than threshold.

        Raises:
            GateVacuousError: If the broken input passes the threshold (gate is vacuous).
        """
        broken_metric = metric_fn(broken_input)
        broken_passes = gate_passes(broken_metric, threshold, direction)

        if broken_passes:
            raise GateVacuousError(
                f"Gate '{self.gate_name}' NEGATIVE CONTROL PASSED — "
                f"broken_metric={broken_metric:.4f} strictly passes "
                f"threshold={threshold:.4f} "
                f"(direction={direction}). The gate is vacuous: it would accept a broken model. "
                f"Fix the metric or the threshold."
            )

        logger.info(
            "Gate '%s' negative control confirmed: broken_metric=%.4f worse than "
            "threshold=%.4f (direction=%s). Gate is live.",
            self.gate_name,
            broken_metric,
            threshold,
            direction,
        )
        return True

    def assert_gate_is_live(
        self,
        good_value: float,
        bad_value: float,
        direction: Direction = "higher_is_better",
    ) -> None:
        """Assert that the gate discriminates — raises GateVacuousError if not.

        Convenience wrapper around check_metric_is_sensitive for use in tests
        where you want a hard assertion.

        Args:
            good_value: Metric value for the good model.
            bad_value: Metric value for the broken model.
            direction: "higher_is_better" or "lower_is_better".

        Raises:
            GateVacuousError: If the values are identical (gate is vacuous).
            AssertionError: If good_value is not better than bad_value.
        """
        result = self.check_metric_is_sensitive(good_value, bad_value, direction)
        # An explicit raise, not `assert`: `python -O` strips assert statements,
        # and this is the check that detects a miscalibrated gate. A gate-vacuity
        # mechanism that silently stops running under an optimization flag is the
        # worst possible thing for it to do. AssertionError is kept because the
        # docstring above declares it.
        if not result:
            raise AssertionError(
                f"Gate '{self.gate_name}': good_value={good_value} is not better than "
                f"bad_value={bad_value} (direction={direction}). "
                f"This indicates the gate is miscalibrated."
            )
