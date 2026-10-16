"""Tests for ailm.control — the control harness.

The control harness itself must be tested: if it cannot detect a broken model,
all subsequent gates are vacuous.
"""

from __future__ import annotations

import numpy as np
import pytest

from ailm.control import ControlHarness, GateVacuousError


class TestControlHarness:
    def test_higher_is_better_detects_good_model(self):
        harness = ControlHarness("accuracy_gate")
        result = harness.check_metric_is_sensitive(0.85, 0.45, "higher_is_better")
        assert result is True

    def test_lower_is_better_detects_good_model(self):
        harness = ControlHarness("perplexity_gate")
        result = harness.check_metric_is_sensitive(12.5, 28.9, "lower_is_better")
        assert result is True

    def test_raises_gate_vacuous_error_when_identical(self):
        """This IS the positive control for the negative-control mechanism."""
        harness = ControlHarness("vacuous_gate")
        with pytest.raises(GateVacuousError, match="vacuous"):
            harness.check_metric_is_sensitive(0.5, 0.5, "higher_is_better")

    def test_raises_gate_vacuous_when_identical_lower_is_better(self):
        harness = ControlHarness("vacuous_gate_2")
        with pytest.raises(GateVacuousError):
            harness.check_metric_is_sensitive(1.0, 1.0, "lower_is_better")

    def test_returns_false_when_bad_beats_good(self):
        """Gate returns False (not raises) when good_value is not better."""
        harness = ControlHarness("miscalibrated")
        result = harness.check_metric_is_sensitive(0.3, 0.9, "higher_is_better")
        assert result is False

    def test_invalid_direction_raises(self):
        harness = ControlHarness("bad_direction")
        with pytest.raises(ValueError, match="direction"):
            harness.check_metric_is_sensitive(0.8, 0.2, "sideways")

    def test_assert_gate_is_live_passes_discriminating(self):
        harness = ControlHarness("live_gate")
        harness.assert_gate_is_live(0.9, 0.1, "higher_is_better")  # must not raise

    def test_assert_gate_is_live_raises_vacuous(self):
        harness = ControlHarness("dead_gate")
        with pytest.raises(GateVacuousError):
            harness.assert_gate_is_live(0.5, 0.5)

    def test_assert_gate_is_live_raises_miscalibrated_higher_is_better(self):
        """S17: the miscalibrated branch fires in ordinary Python, directly.

        The optimized-bytecode control in
        ``tests/test_gates_survive_assertion_stripping.py`` proves this raise
        survives ``python -O``; this control proves the branch fires in plain
        mode too, so the explicit raise cannot silently disappear or degrade
        into a bare ``assert``-and-return without this file failing.
        """
        harness = ControlHarness("miscalibrated_gate")
        with pytest.raises(AssertionError, match="miscalibrated"):
            harness.assert_gate_is_live(0.1, 0.9, "higher_is_better")

    def test_assert_gate_is_live_raises_miscalibrated_lower_is_better(self):
        """The same branch from the other direction: good is worse than bad."""
        harness = ControlHarness("miscalibrated_gate_lower")
        with pytest.raises(AssertionError, match="miscalibrated"):
            harness.assert_gate_is_live(0.9, 0.1, "lower_is_better")

    def test_verify_negative_control_fails_when_broken(self):
        """A uniform (broken) model has higher entropy than a good one."""
        harness = ControlHarness("entropy_gate")

        def entropy(dist: np.ndarray) -> float:
            d = np.asarray(dist, dtype=np.float64)
            return float(-np.sum(d * np.log(d + 1e-12)))

        good_dist = np.array([0.9, 0.05, 0.03, 0.02])
        bad_dist = np.ones(4) / 4  # uniform = maximum entropy = broken

        good_entropy = entropy(good_dist)  # low: ~0.47 nats
        bad_entropy = entropy(bad_dist)  # high: ~1.39 nats = log(4)

        # Threshold: halfway between good and bad
        threshold = (good_entropy + bad_entropy) / 2

        result = harness.verify_negative_control_fails(
            entropy, bad_dist, threshold=threshold, direction="lower_is_better"
        )
        assert result is True

    def test_verify_negative_control_raises_when_broken_passes(self):
        """If broken model passes, GateVacuousError must be raised."""
        harness = ControlHarness("bad_threshold_gate")

        def always_low(_) -> float:
            return 0.01  # broken model somehow scores great

        with pytest.raises(GateVacuousError, match="NEGATIVE CONTROL"):
            harness.verify_negative_control_fails(
                always_low,
                "fake_broken_input",
                threshold=0.5,
                direction="lower_is_better",
            )

    @pytest.mark.parametrize("direction", ["higher_is_better", "lower_is_better"])
    def test_negative_control_at_the_strict_boundary_counts_as_failing(self, direction):
        harness = ControlHarness("strict_boundary")
        assert harness.verify_negative_control_fails(
            lambda _: 0.5,
            object(),
            threshold=0.5,
            direction=direction,
        )

    def test_negative_control_rejects_invalid_direction(self):
        harness = ControlHarness("invalid_direction")
        with pytest.raises(ValueError, match="direction"):
            harness.verify_negative_control_fails(
                lambda _: 1.0, object(), threshold=0.0, direction="sideways"
            )

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_negative_control_rejects_non_finite_metrics(self, value):
        harness = ControlHarness("non_finite")
        with pytest.raises(ValueError, match="finite"):
            harness.verify_negative_control_fails(
                lambda _: value,
                object(),
                threshold=0.0,
                direction="higher_is_better",
            )

    def test_gate_vacuous_error_is_exception(self):
        assert issubclass(GateVacuousError, Exception)

    def test_entropy_gate_catches_uniform_model(self):
        """End-to-end: uniform-distribution model is caught as broken.

        A highly peaked distribution has lower perplexity than a uniform one.
        This is a mathematical fact: exp(H[uniform]) = n > exp(H[peaked]).
        """
        harness = ControlHarness("stage0_entropy")

        n = 30  # 30 letters, Stage 1 context

        def perplexity(probs: np.ndarray) -> float:
            """Perplexity = exp(entropy) = exp(-sum p log p)."""
            p = np.asarray(probs, dtype=np.float64)
            # Normalize to be safe
            p = p / p.sum()
            entropy = -np.sum(p * np.log(p + 1e-30))
            return float(np.exp(entropy))

        # Good model: highly peaked on first symbol (English 'e' is very frequent)
        good_probs = np.ones(n) * 0.01
        good_probs[0] = 1.0 - (n - 1) * 0.01  # ~0.71 on the most frequent
        good_ppl = perplexity(good_probs)  # should be much less than n=30

        # Broken model: uniform (maximum perplexity = n)
        uniform_probs = np.ones(n) / n
        broken_ppl = perplexity(uniform_probs)  # = n = 30

        assert broken_ppl > good_ppl, (
            f"Uniform perplexity {broken_ppl:.2f} should exceed peaked perplexity {good_ppl:.2f}"
        )
        harness.assert_gate_is_live(good_ppl, broken_ppl, "lower_is_better")


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
