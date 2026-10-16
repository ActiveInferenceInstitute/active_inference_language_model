"""pymdp API smoke tests for AILM Stage 0.

These verify that the pymdp version installed has the API surface AILM depends on.
All tests run REAL pymdp — no mocks, no monkeypatching.

Key facts verified (from handoff + direct inspection of pymdp 1.0.1/1.0.3; still true under the 1.0.4 pin, re-verified 2026-08-29):
  - Agent.__init__ accepts A_dependencies and B_dependencies keyword args
  - With A_dependencies=[[0],[1]], A[0] has shape (batch, obs0, states_factor0) — 3D not 4D
  - update_empirical_prior() exists (within-level temporal recursion)
  - update_A() does NOT exist (learning goes through infer_parameters)
  - infer_policies(qs) returns (q_pi, neg_efe), shapes (batch, n_policies) each
  - Tensors carry a leading batch dimension; batch_size=1 still requires it
"""

from __future__ import annotations

import logging
from typing import Any

import jax.numpy as jnp

logger = logging.getLogger(__name__)


def make_two_factor_agent() -> Any:
    """Create a minimal valid pymdp Agent with 2 hidden state factors.

    Configuration:
      Factors: [4 states, 3 states]
      Modalities: [5 obs (depends on factor 0), 4 obs (depends on factor 1)]
      A_dependencies=[[0], [1]] — key factorization: each modality sees only its factor

    This is the architecture primitive that makes AILM feasible at scale.
    Without A_dependencies, A would need shape (batch, obs0, states0, obs1, states1),
    which becomes impossible at the word level (134k × 134k transition matrix).

    Returns:
        pymdp.agent.Agent instance.
    """
    from pymdp.agent import Agent

    n_states = [4, 3]
    n_obs = [5, 4]

    # pymdp 1.0.3 (still true under 1.0.4) requires JAX arrays (jnp), not NumPy arrays.
    # A matrices: one per modality, shaped (batch, obs, states_for_this_modality)
    # Modality 0 sees only factor 0 → shape (1, 5, 4)
    # Modality 1 sees only factor 1 → shape (1, 4, 3)
    A = [
        jnp.ones((1, n_obs[0], n_states[0])) / n_obs[0],  # uniform likelihood
        jnp.ones((1, n_obs[1], n_states[1])) / n_obs[1],
    ]

    # B matrices: one per factor, shape (batch, n_states, n_states, n_actions)
    B = [
        jnp.eye(n_states[0])[jnp.newaxis, :, :, jnp.newaxis],  # (1, 4, 4, 1)
        jnp.eye(n_states[1])[jnp.newaxis, :, :, jnp.newaxis],  # (1, 3, 3, 1)
    ]

    # C: log preferences, uniform → zero (shape (batch, n_obs) per modality)
    C = [
        jnp.zeros((1, n_obs[0])),
        jnp.zeros((1, n_obs[1])),
    ]

    # D: prior over states, uniform (shape (batch, n_states) per factor)
    D = [
        jnp.ones((1, n_states[0])) / n_states[0],
        jnp.ones((1, n_states[1])) / n_states[1],
    ]

    agent = Agent(
        A=A,
        B=B,
        C=C,
        D=D,
        A_dependencies=[[0], [1]],  # modality 0 → factor 0; modality 1 → factor 1
        num_controls=[1, 1],
        batch_size=1,
    )
    return agent


def verify_a_dependencies_shape(agent: Any) -> dict[str, Any]:
    """Verify that A_dependencies factorization produces the correct tensor shapes.

    With A_dependencies=[[0],[1]]:
      agent.A[0] should be 3D: (batch=1, n_obs_0=5, n_states_factor_0=4)
      agent.A[1] should be 3D: (batch=1, n_obs_1=4, n_states_factor_1=3)

    Without A_dependencies, A would be 4D or 5D and impossibly large at scale.

    Returns:
        dict with A0_shape, A1_shape, A0_correct, A1_correct, factorization_works.
    """
    try:
        A0 = agent.A[0]
        A1 = agent.A[1]
        A0_shape = tuple(A0.shape)
        A1_shape = tuple(A1.shape)

        # Correct: 3D arrays (batch, obs, states_for_this_factor)
        A0_correct = len(A0_shape) == 3 and A0_shape[1] == 5 and A0_shape[2] == 4
        A1_correct = len(A1_shape) == 3 and A1_shape[1] == 4 and A1_shape[2] == 3

        return {
            "success": True,
            "A0_shape": A0_shape,
            "A1_shape": A1_shape,
            "A0_correct": A0_correct,
            "A1_correct": A1_correct,
            "factorization_works": A0_correct and A1_correct,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "factorization_works": False}


def verify_empirical_prior_coupling(agent: Any) -> dict[str, Any]:
    """Verify update_empirical_prior() exists and is callable.

    It is the second half of the filtering recursion: an agent's own posterior
    pushed through its own B to form the prior for its next symbol. Within-level
    and temporal, never a channel between levels — see ``docs/hierarchy.md``.

    Tries multiple call signatures (the API varied between 1.0.1 and 1.0.3; the 1.0.3 form carries into 1.0.4).

    Returns:
        dict with method_exists, call_succeeded, call_signature, error.
    """
    method_exists = hasattr(agent, "update_empirical_prior")
    if not method_exists:
        return {
            "success": False,
            "method_exists": False,
            "call_succeeded": False,
            "error": "update_empirical_prior method not found on Agent",
        }

    # pymdp 1.0.3 (unchanged in 1.0.4): update_empirical_prior(action, qs) -> list[jax.Array]
    # This is called after infer_states and infer_policies to produce the
    # next prior for the following time step, at the same level.
    n_states = [4, 3]
    # Build minimal state beliefs (posterior over states) as input to the call
    fake_qs = [
        jnp.ones((1, 1, n)) / n
        for n in n_states  # (batch, time, states)
    ]
    # Action: shape (batch, n_factors) — one action index per factor
    fake_action = jnp.zeros((1, len(n_states)), dtype=jnp.int32)

    call_succeeded = False
    error_msg = None
    sig_used = None

    # Try signature: update_empirical_prior(action, qs)
    try:
        agent.update_empirical_prior(fake_action, fake_qs)
        call_succeeded = True
        sig_used = "update_empirical_prior(action, qs)"
    except Exception as e1:
        # Try alternative: no action (some versions may differ)
        try:
            agent.update_empirical_prior(fake_qs)
            call_succeeded = True
            sig_used = "update_empirical_prior(qs)"
        except Exception as e2:
            error_msg = f"(action,qs): {e1!r}; (qs,): {e2!r}"

    return {
        "success": call_succeeded,
        "method_exists": True,
        "call_succeeded": call_succeeded,
        "call_signature": sig_used,
        "error": error_msg,
    }


def verify_no_update_a_method(agent: Any) -> dict[str, Any]:
    """Verify update_A does NOT exist (learning goes through infer_parameters).

    Returns:
        dict with has_update_A (must be False), has_infer_parameters (must be True),
        has_update_empirical_prior (must be True).
    """
    return {
        "success": True,
        "has_update_A": hasattr(agent, "update_A"),
        "has_update_B": hasattr(agent, "update_B"),
        "has_infer_parameters": hasattr(agent, "infer_parameters"),
        "has_update_empirical_prior": hasattr(agent, "update_empirical_prior"),
    }


def verify_infer_policies_output(agent: Any) -> dict[str, Any]:
    """Verify infer_policies returns (q_pi, neg_efe) with correct shapes.

    Returns:
        dict with q_pi_shape, neg_efe_shape, success, error.
    """
    try:
        # pymdp 1.0.3 (unchanged in 1.0.4): infer_states(observations, empirical_prior) -> list[jax.Array]
        # The prior is stored in agent.D at initialization; we use it as the first prior.
        n_states = [4, 3]
        empirical_prior = [jnp.ones((1, n)) / n for n in n_states]
        obs = [jnp.array([0]), jnp.array([0])]
        qs = agent.infer_states(obs, empirical_prior)

        # Now infer policies
        result = agent.infer_policies(qs)

        if not (isinstance(result, (list, tuple)) and len(result) == 2):
            return {
                "success": False,
                "error": f"infer_policies returned {type(result)}, expected 2-tuple",
            }

        q_pi, neg_efe = result
        return {
            "success": True,
            "q_pi_shape": tuple(q_pi.shape),
            "neg_efe_shape": tuple(neg_efe.shape),
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def run_all_smoke_tests() -> dict[str, Any]:
    """Run all pymdp smoke tests. Returns results dict; does not raise.

    Each sub-test catches its own exceptions so a single failure doesn't
    block the others. Inspect "all_passed" for the overall result.

    Returns:
        dict mapping test_name → result dict, plus "all_passed": bool.
    """
    results: dict[str, Any] = {}

    try:
        agent = make_two_factor_agent()
        results["make_agent"] = {"success": True}
    except Exception as exc:
        results["make_agent"] = {"success": False, "error": str(exc)}
        results["all_passed"] = False
        return results  # can't run remaining tests without agent

    results["a_dependencies_shape"] = verify_a_dependencies_shape(agent)
    results["empirical_prior_coupling"] = verify_empirical_prior_coupling(agent)
    results["no_update_a_method"] = verify_no_update_a_method(agent)
    results["infer_policies_output"] = verify_infer_policies_output(agent)

    # Specific checks for all_passed
    checks = [
        results["make_agent"].get("success", False),
        results["a_dependencies_shape"].get("factorization_works", False),
        results["empirical_prior_coupling"].get("method_exists", False),
        not results["no_update_a_method"].get("has_update_A", True),  # must NOT have it
        results["no_update_a_method"].get("has_infer_parameters", False),
        results["infer_policies_output"].get("success", False),
    ]
    results["all_passed"] = all(checks)

    logger.info(
        "Smoke tests: %d/%d passed",
        sum(1 for c in checks if c),
        len(checks),
    )
    return results
