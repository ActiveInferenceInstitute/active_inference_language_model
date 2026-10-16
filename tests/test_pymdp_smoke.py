"""Tests that verify pymdp API works as needed for AILM.

These are the Stage 0 gate for pymdp. If any fail, the architecture cannot
be built on this pymdp version. Real pymdp — no mocks, no monkeypatching.
"""

from __future__ import annotations

import inspect

import pytest

from ailm.generative.smoke import (
    make_two_factor_agent,
    run_all_smoke_tests,
    verify_a_dependencies_shape,
    verify_empirical_prior_coupling,
    verify_infer_policies_output,
    verify_no_update_a_method,
)

# These are real pinned-runtime integration checks. A fresh pymdp/matplotlib
# import plus the first JAX compilation exceeded the ordinary 60-second budget
# under concurrent fleet load; interrupting that import also poisoned subsequent
# checks in the same process. Keep startup bounded without relaxing assertions.
pytestmark = pytest.mark.timeout(300)


class TestADependenciesShape:
    def test_pinned_validator_checks_structure_beyond_normalization(self):
        """A normalized tensor can still fail pymdp's dependency contract.

        This is the negative control for the manuscript's deliberately narrower
        statement: pymdp checks real structure, but cannot infer whether a
        compatible square matrix *means* an emission or a transition.
        """
        from pymdp.agent import Agent

        source = inspect.getsource(Agent._validate)
        for contract in (
            "A_dependencies",
            "B_dependencies",
            "control_fac_idx",
            "EXACT_METHOD",
        ):
            assert contract in source, f"pinned pymdp validator no longer checks {contract}"

        agent = make_two_factor_agent()
        # A[0] remains normalized. Only its declared dependency is wrong: it now
        # names factor 1 while its trailing dimension still has factor 0's size.
        agent.A_dependencies[0] = [1]
        with pytest.raises(AssertionError, match="A_dependencies"):
            agent._validate()

    def test_factorized_a_matrix_shape(self, two_factor_agent):
        result = verify_a_dependencies_shape(two_factor_agent)
        assert result["factorization_works"], (
            f"A_dependencies factorization failed:\n"
            f"  A[0].shape = {result.get('A0_shape')} (expected (1, 5, 4))\n"
            f"  A[1].shape = {result.get('A1_shape')} (expected (1, 4, 3))\n"
            f"If A[m] is 4D or 5D, A_dependencies is not being applied correctly.\n"
            f"Without factorization, A at the word level is impossibly large (70+ GiB)."
        )

    def test_a0_is_three_dimensional(self, two_factor_agent):
        result = verify_a_dependencies_shape(two_factor_agent)
        assert result.get("success"), f"Smoke test failed: {result}"
        assert len(result["A0_shape"]) == 3, (
            f"A[0] should be 3D (batch, obs0, states_factor0), "
            f"got shape {result['A0_shape']}. "
            f"A 4D or 5D A[0] means factor 1 is still included — factorization broken."
        )

    def test_a1_is_three_dimensional(self, two_factor_agent):
        result = verify_a_dependencies_shape(two_factor_agent)
        assert result.get("success"), f"Smoke test failed: {result}"
        assert len(result["A1_shape"]) == 3, (
            f"A[1] should be 3D (batch, obs1, states_factor1), got shape {result['A1_shape']}."
        )

    def test_a0_has_correct_dimensions(self, two_factor_agent):
        result = verify_a_dependencies_shape(two_factor_agent)
        if result.get("success") and len(result.get("A0_shape", ())) == 3:
            batch, obs, states = result["A0_shape"]
            assert batch == 1, f"batch dim should be 1, got {batch}"
            assert obs == 5, f"obs0 should be 5, got {obs}"
            assert states == 4, f"states_factor0 should be 4, got {states}"


class TestEmpiricalPriorCoupling:
    def test_update_empirical_prior_method_exists(self, two_factor_agent):
        result = verify_empirical_prior_coupling(two_factor_agent)
        assert result["method_exists"], (
            "update_empirical_prior must exist on Agent. "
            "It is the second half of the filtering recursion: an agent's own "
            "posterior pushed through its own B to form its next prior."
        )

    def test_empirical_prior_call_succeeds(self, two_factor_agent):
        result = verify_empirical_prior_coupling(two_factor_agent)
        assert result["call_succeeded"], (
            f"update_empirical_prior() call failed: {result.get('error')}. "
            f"Without this, no level can carry a belief from one symbol to the next."
        )


class TestNoUpdateAMethod:
    def test_update_a_does_not_exist(self, two_factor_agent):
        result = verify_no_update_a_method(two_factor_agent)
        assert not result["has_update_A"], (
            "update_A must NOT exist on Agent. "
            "Learning goes through infer_parameters. "
            "Calling a non-existent update_A silently skips all learning."
        )

    def test_infer_parameters_exists(self, two_factor_agent):
        result = verify_no_update_a_method(two_factor_agent)
        assert result["has_infer_parameters"], (
            "infer_parameters must exist — it is the correct learning method."
        )

    def test_update_empirical_prior_exists(self, two_factor_agent):
        result = verify_no_update_a_method(two_factor_agent)
        assert result["has_update_empirical_prior"]


class TestInferPoliciesOutput:
    def test_returns_two_arrays(self, two_factor_agent):
        result = verify_infer_policies_output(two_factor_agent)
        assert result["success"], (
            f"infer_policies failed: {result.get('error')}. "
            f"This is required for policy selection at every level."
        )

    def test_q_pi_and_neg_efe_shapes_match(self, two_factor_agent):
        result = verify_infer_policies_output(two_factor_agent)
        if result.get("success"):
            assert result["q_pi_shape"] == result["neg_efe_shape"], (
                f"q_pi shape {result['q_pi_shape']} != neg_efe shape {result['neg_efe_shape']}"
            )

    def test_output_has_batch_dimension(self, two_factor_agent):
        result = verify_infer_policies_output(two_factor_agent)
        if result.get("success"):
            # Both arrays should be 2D: (batch, n_policies)
            assert len(result["q_pi_shape"]) == 2, (
                f"q_pi should be 2D (batch, n_policies), got {result['q_pi_shape']}"
            )


class TestVerifyFunctionFailurePaths:
    """Cover the failure/error branches in each verify_ function."""

    def test_a_dependencies_shape_error_on_bad_agent(self):
        """Passing a non-agent produces a safe error dict."""
        result = verify_a_dependencies_shape(object())
        assert result.get("success") is False
        assert "error" in result
        assert result.get("factorization_works") is False

    def test_empirical_prior_coupling_no_method(self):
        """Object without update_empirical_prior → method_exists=False."""
        result = verify_empirical_prior_coupling(object())
        assert result["method_exists"] is False
        assert result["call_succeeded"] is False

    def test_no_update_a_method_on_plain_object(self):
        """Plain object has no update_A and no infer_parameters."""
        result = verify_no_update_a_method(object())
        assert result["has_update_A"] is False
        assert result["has_infer_parameters"] is False

    def test_infer_policies_output_error_on_bad_agent(self):
        """Passing None → error dict with success=False."""
        result = verify_infer_policies_output(None)
        assert result.get("success") is False
        assert "error" in result


class TestAllSmokeTests:
    def test_all_smoke_tests_pass(self):
        """STAGE 0 GATE: all pymdp smoke tests must pass.

        If any fail, Stage 0 is blocked and Stage 1 cannot start.
        """
        result = run_all_smoke_tests()
        failed = {
            k: v
            for k, v in result.items()
            if k != "all_passed" and isinstance(v, dict) and not v.get("success", True)
        }
        assert result["all_passed"], (
            f"Stage 0 blocked: pymdp smoke tests failed.\n"
            f"Failed tests: {list(failed.keys())}\n"
            f"Details: {failed}\n"
            f"Check pymdp version (expected 1.0.4) and API surface."
        )

    def test_smoke_result_is_dict(self):
        result = run_all_smoke_tests()
        assert isinstance(result, dict)
        assert "all_passed" in result

    def test_make_agent_smoke(self):
        """Agent construction itself must not raise."""
        agent = make_two_factor_agent()
        assert agent is not None
        assert hasattr(agent, "A")
        assert hasattr(agent, "B")
        assert len(agent.A) == 2
        assert len(agent.B) == 2


class TestNumericEnvironment:
    """The recorded numeric environment must still describe this interpreter.

    The manuscript's build-determinism caveat rests on pymdp computing in
    float32 under JAX defaults, and the Stage 0 recorder pins the live
    ``jax_enable_x64`` value (with the JAX, jaxlib, and pymdp versions) at
    record time. These tests read the flag AFTER the ailm pymdp path above has
    been imported, so an import anywhere in that path that flipped the config
    would be caught here rather than surfacing as a diffuse tolerance failure.
    """

    @staticmethod
    def _recorded(label: str):
        import json
        from pathlib import Path

        ledger_path = Path(__file__).resolve().parent.parent / "data" / "claims.json"
        claims = json.loads(ledger_path.read_text(encoding="utf-8"))["claims"]
        assert label in claims, (
            f"{label} is not in data/claims.json. "
            "Run: .venv/bin/python scripts/record_stage0_claims.py"
        )
        return claims[label]["value"]

    def test_jax_x64_is_disabled_and_matches_the_recorded_claim(self):
        import jax

        live = bool(jax.config.jax_enable_x64)
        assert live is False, (
            "jax_enable_x64 is True in this interpreter. The manuscript's "
            "float32 tolerance story does not survive a float64 run: rewrite "
            "the numeric-environment claims, do not just re-render."
        )
        assert self._recorded("env_jax_enable_x64") == live

    def test_recorded_versions_match_the_live_environment(self):
        import importlib.metadata

        import jax

        assert self._recorded("env_jax_version") == jax.__version__
        assert self._recorded("env_jaxlib_version") == importlib.metadata.version("jaxlib")
        assert self._recorded("env_pymdp_version") == importlib.metadata.version(
            "inferactively-pymdp"
        )

    def test_recorded_platform_is_provenance_not_prophecy(self):
        """The platform strings are record-time provenance; only the machine
        architecture is asserted against the live host, because an arm64 to
        x86_64 move genuinely changes the float story while an OS patch does not.
        """
        import platform

        assert self._recorded("env_machine") == platform.machine()
        recorded_platform = self._recorded("env_platform")
        assert isinstance(recorded_platform, str) and recorded_platform
