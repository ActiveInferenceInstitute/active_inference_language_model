"""The pymdp smoke harness must report a failing call, not hide it.

The empirical-prior probe tries two call signatures and records both errors
when neither works. That reporting path is exercised with a real pymdp agent
whose factor structure disagrees with the probe's fixed beliefs, so the
recorded error comes from pymdp itself rather than from a stand-in.
"""

from __future__ import annotations

import jax.numpy as jnp
from pymdp.agent import Agent

from ailm.generative.smoke import make_two_factor_agent, verify_empirical_prior_coupling


def _single_factor_agent() -> Agent:
    return Agent(
        A=[jnp.full((1, 2, 2), 0.5)],
        B=[jnp.eye(2)[jnp.newaxis, :, :, jnp.newaxis]],
        C=[jnp.zeros((1, 2))],
        D=[jnp.full((1, 2), 0.5)],
        num_controls=[1],
        batch_size=1,
    )


def test_the_probe_reports_both_signature_failures_for_a_mismatched_agent() -> None:
    result = verify_empirical_prior_coupling(_single_factor_agent())
    assert result["method_exists"] is True
    assert result["call_succeeded"] is False
    assert result["call_signature"] is None
    assert "(action,qs):" in result["error"] and "(qs,):" in result["error"]


def test_the_probe_succeeds_on_the_matching_two_factor_agent() -> None:
    result = verify_empirical_prior_coupling(make_two_factor_agent())
    assert result["call_succeeded"] is True
    assert result["error"] is None
