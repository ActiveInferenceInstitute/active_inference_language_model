"""P1 core-inference lane (2026-08-31): L5 live pymdp inference equivalence.

The existing L4 test proved ``infer_states`` reproduces the numpy decoder. This
module extends the same discipline to L5 (clause) and closes the L6 attribution
gap decision 68 named: the recorded ``stage6_live_filter_*`` numbers must be
shown to come from pymdp's inference path, not from a NumPy shortcut wearing
its name.

What each test buys:

- ``test_level_5_clause_pymdp_inference_reproduces_the_joint_oracle``: the live
  agent's ``infer_states`` + ``update_empirical_prior`` walk over a bounded
  held-out clause slice must agree with the numpy oracle joint forward filter
  (``ailm.verification.oracle.forward_filter``) at every position, twice over:
  at the sentence-type posterior argmax, and on the full joint (type x case)
  posterior within ``POSTERIOR_ATOL``. The second check is the one that
  carries weight (decision 94): on this population the oracle's type argmax
  is the same type at every position, so the argmax count alone cannot
  distinguish the shipped transition from one with no dynamics. The recorder
  (``scripts/record_pymdp_equivalence_claims.py``) walks the whole held-out
  set with the same two checks; this test verifies on a slice.
- ``test_live_type_belief_is_a_distribution_at_every_step``: normalization
  negative control, mirroring L4's.
- ``test_swapped_type_transition_breaks_the_agreement``: negative control on
  the argmax comparison. A ``B`` with the two type blocks swapped must
  produce visible type-argmax disagreements; if it did not, the comparison
  would pass regardless of what the model believes.
- ``test_identity_transition_breaks_the_posterior_agreement``: negative
  control on the tolerance, and the demonstration that the argmax statistic
  is vacuous here. An identity ``B`` (no case dynamics) must score zero
  type-argmax disagreements on the slice AND a joint-posterior deviation far
  above ``POSTERIOR_ATOL``. If the first half failed, the argmax count would
  have been sufficient after all; if the second half failed, the tolerance
  could not see the class of wrong model the block swap cannot see either.
- ``test_level_6_pymdp_filter_matches_a_manual_float32_matrix_walk``: the
  recorded ``live_agent_type_logloss`` score really came from pymdp's
  inference path — every per-step belief it implies must equal the manual
  Bayes step over the agent's own stored float32 matrices.
- The shared ``test_every_numpy_scored_level_still_constructs_a_real_agent``
  (in ``test_pymdp_inference_equivalence.py``) remains the isinstance control
  for both levels.

Timeouts: the repo default (60 s) cannot even import jax on cold caches, and
pytest-timeout runs per test — so the heavy imports happen at module
collection time, outside any per-test budget, and the module carries the
documented module-level mark for the bounded pymdp walks themselves.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from ailm.levels.clause import (
    N_CASES,
    build_clause_agent,
    build_joint_emission,
    build_joint_start,
    build_joint_transition,
    build_type_models,
    load_clause_artifact,
)
from ailm.levels.narrative import build_narrative_agent, load_narrative_artifact
from ailm.verification.oracle import forward_filter

pytestmark = pytest.mark.timeout(600)

#: Sentences or documents compared per level. Large enough that a systematic
#: disagreement shows, small enough that one call per observation stays quick.
PREFIX = 25

#: Per-position tolerance on the joint posterior. The Agent runs float32 and
#: the oracle float64, so rounding accumulated along a sentence is the only
#: admissible difference; the recorder enforces the same bound on the full set.
POSTERIOR_ATOL = 1e-5

#: The joint-posterior deviation an identity ``B`` must at least reach on the
#: slice for the tolerance to count as detecting it. Far above the rounding
#: scale, far below the largest possible deviation between two distributions.
IDENTITY_B_MIN_DEVIATION = 0.1


def _as_vector(qs) -> np.ndarray:
    """pymdp returns a list of batched factor beliefs; take factor 0, flattened."""
    return np.asarray(qs[0]).ravel()


def _type_marginal(joint_belief: np.ndarray) -> np.ndarray:
    """Marginalize a joint (type x case) belief down to the type blocks."""
    return joint_belief.reshape(2, -1).sum(axis=1)


def test_level_5_clause_pymdp_inference_reproduces_the_joint_oracle():
    """The recorded Stage 5 evidence is closed-form. pymdp's recursion must
    reconstruct the same per-position type posterior the numpy oracle computes
    from the identical joint matrices."""
    artifact = load_clause_artifact()
    models = build_type_models(artifact.train)
    agent = build_clause_agent(artifact)
    action = jnp.zeros((1, 1), dtype=jnp.int32)

    emission = build_joint_emission()
    joint_b = build_joint_transition(models)
    joint_d = build_joint_start(models)

    pymdp_types: list[int] = []
    oracle_types: list[int] = []
    max_abs_dev = 0.0
    for sequence in artifact.test[:PREFIX]:
        observations = sequence[1]
        oracle_posteriors, _log_evidence = forward_filter(emission, joint_b, joint_d, observations)
        prior = [jnp.asarray(np.asarray(agent.D[0]))]
        for position, observation in enumerate(observations):
            qs = agent.infer_states([jnp.array([observation])], prior)
            belief = _as_vector(qs)
            pymdp_types.append(int(np.argmax(_type_marginal(belief))))
            oracle_types.append(int(np.argmax(_type_marginal(oracle_posteriors[position]))))
            deviation = float(np.max(np.abs(belief - oracle_posteriors[position])))
            max_abs_dev = max(max_abs_dev, deviation)
            assert np.allclose(belief, oracle_posteriors[position], atol=POSTERIOR_ATOL), (
                f"pymdp's joint posterior deviates from the numpy oracle by {deviation:.3e} "
                f"at position {position} of a held-out clause; the argmax comparison "
                f"below cannot see this, and the recorded stage5_pymdp_* equivalence "
                f"would not hold"
            )
            prior = agent.update_empirical_prior(action, qs)

    compared = len(pymdp_types)
    assert compared > 100, "prefix too small for a disagreement to be visible"
    assert max_abs_dev <= POSTERIOR_ATOL
    disagreements = [
        i for i, (a, b) in enumerate(zip(pymdp_types, oracle_types, strict=True)) if a != b
    ]
    assert disagreements == [], (
        f"pymdp inference and the numpy joint oracle disagree on the sentence-type "
        f"posterior argmax at {len(disagreements)} of {compared} clause positions "
        f"(first at index {disagreements[:3]}). The recorded stage5_* numbers come "
        f"from the closed-form mixture over these same matrices, so a disagreement "
        f"means the live agent's recursion is not attributable to the model the "
        f"paper describes."
    )


def test_live_type_belief_is_a_distribution_at_every_step():
    """A negative control on the comparison above.

    If ``infer_states`` returned something unnormalized, an argmax comparison
    would still pass while the beliefs were meaningless. The equivalence test
    cannot see that; this one can.
    """
    artifact = load_clause_artifact()
    agent = build_clause_agent(artifact)
    action = jnp.zeros((1, 1), dtype=jnp.int32)

    prior = [jnp.asarray(np.asarray(agent.D[0]))]
    steps = 0
    for sequence in artifact.test[:5]:
        for case_index in sequence[1]:
            qs = agent.infer_states([jnp.array([case_index])], prior)
            belief = _as_vector(qs)
            assert belief.min() >= -1e-9, "a negative belief is not a distribution"
            assert abs(float(belief.sum()) - 1.0) < 1e-5, (
                f"pymdp belief sums to {belief.sum()}, not 1; the argmax comparison "
                f"elsewhere would pass regardless"
            )
            prior = agent.update_empirical_prior(action, qs)
            steps += 1
    assert steps > 50, "control walked too few steps to be evidence"


def test_swapped_type_transition_breaks_the_agreement():
    """Negative control: the equivalence must be able to FAIL.

    Build the agent with the two type blocks of ``B`` swapped — the model then
    believes declarative case statistics govern interrogative sentences. The
    type posterior must visibly diverge from the oracle at the final position
    of at least one sentence in the prefix; an agreement that survives a
    deliberately wrong transition is not checking anything.
    """
    artifact = load_clause_artifact()
    models = build_type_models(artifact.train)
    joint_b = build_joint_transition(models)
    # Permuting rows and columns by the same permutation preserves
    # column-stochasticity, so pymdp's own validation accepts the swap; only
    # the semantic wiring changes (which case chain governs which type).
    # Blocks are N_CASES wide (the joint space is type-major): swap the two
    # 8-state type blocks, keeping each block's internal case order intact.
    full_swap = np.concatenate([np.arange(N_CASES, 2 * N_CASES), np.arange(N_CASES)])
    swapped = joint_b[np.ix_(full_swap, full_swap)]

    agent = build_clause_agent(artifact, transition=swapped)
    emission = build_joint_emission()
    joint_d = build_joint_start(models)
    action = jnp.zeros((1, 1), dtype=jnp.int32)

    diverged = 0
    sentences = 0
    for sequence in artifact.test[:PREFIX]:
        observations = sequence[1]
        _posteriors, _ = forward_filter(emission, joint_b, joint_d, observations)
        prior = [jnp.asarray(np.asarray(agent.D[0]))]
        for observation in observations:
            qs = agent.infer_states([jnp.array([observation])], prior)
            prior = agent.update_empirical_prior(action, qs)
        live_type = int(np.argmax(_type_marginal(_as_vector(qs))))
        oracle_type = int(np.argmax(_posteriors[-1].reshape(2, -1).sum(axis=1)))
        sentences += 1
        if live_type != oracle_type:
            diverged += 1
    assert sentences > 0
    assert diverged > 0, (
        "a type-block-swapped transition produced the same sentence-type "
        "posterior argmax as the correct model on every prefix sentence — "
        "the equivalence comparison above cannot detect a wrong model"
    )


def test_identity_transition_breaks_the_posterior_agreement():
    """Negative control on the tolerance; positive proof the argmax is vacuous.

    An identity ``B`` carries no case dynamics at all, yet on this slice the
    oracle's sentence-type argmax never leaves the majority type, so the
    identity model scores zero type-argmax disagreements. That is what makes
    the argmax count a necessary-but-not-sufficient statistic here (decision
    94). The joint posterior tells the models apart at once: the identity
    model's beliefs must depart from the oracle by far more than the
    float32 rounding scale. Both halves are asserted, because each one alone
    would leave the other untested.
    """
    artifact = load_clause_artifact()
    models = build_type_models(artifact.train)
    joint_b = build_joint_transition(models)
    joint_d = build_joint_start(models)
    emission = build_joint_emission()
    identity = np.eye(2 * N_CASES)
    agent = build_clause_agent(artifact, transition=identity)
    action = jnp.zeros((1, 1), dtype=jnp.int32)

    disagreements = 0
    positions = 0
    max_abs_dev = 0.0
    for sequence in artifact.test[:PREFIX]:
        observations = sequence[1]
        oracle_posteriors, _ = forward_filter(emission, joint_b, joint_d, observations)
        prior = [jnp.asarray(np.asarray(agent.D[0]))]
        for position, observation in enumerate(observations):
            qs = agent.infer_states([jnp.array([observation])], prior)
            belief = _as_vector(qs)
            live_type = int(np.argmax(_type_marginal(belief)))
            oracle_type = int(np.argmax(_type_marginal(oracle_posteriors[position])))
            disagreements += int(live_type != oracle_type)
            max_abs_dev = max(
                max_abs_dev, float(np.max(np.abs(belief - oracle_posteriors[position])))
            )
            prior = agent.update_empirical_prior(action, qs)
            positions += 1

    assert positions > 100, "prefix too small for a disagreement to be visible"
    assert disagreements == 0, (
        f"an identity transition produced {disagreements} type-argmax disagreements "
        f"on the slice. If the argmax count can see a model with no dynamics, the "
        f"posterior tolerance is not the load-bearing statistic this module says it "
        f"is; re-derive which check carries the equivalence before trusting either"
    )
    assert max_abs_dev > IDENTITY_B_MIN_DEVIATION, (
        f"an identity transition's joint posterior stayed within {max_abs_dev:.3e} of "
        f"the correct model's at every position; the {POSTERIOR_ATOL:.0e} tolerance "
        f"could not distinguish a model with no case dynamics from the shipped one"
    )


def test_level_6_pymdp_filter_matches_a_manual_float32_matrix_walk():
    """The recorded live-filter score came from pymdp's inference, not from a
    NumPy shortcut wearing its name.

    ``live_agent_type_logloss`` computes each step's predictive as
    ``A @ prior`` and advances the belief with ``infer_states`` /
    ``update_empirical_prior``. Drive the same recursion manually and require
    the belief pymdp returns at every step to equal the manual Bayes step over
    the agent's own stored float32 matrices. A mismatch would mean the
    recorded ``stage6_live_filter_*`` numbers describe something other than
    the pymdp recursion the manuscript names.
    """
    artifact = load_narrative_artifact()
    agent = build_narrative_agent(artifact)
    emission = np.asarray(agent.A[0], dtype=np.float64)[0]
    action = jnp.zeros((1, 1), dtype=jnp.int32)

    checked = 0
    for types in artifact.test[:PREFIX]:
        prior = [jnp.asarray(np.asarray(agent.D[0]))]
        for sentence_type in types:
            prior_vector = np.asarray(prior[0], dtype=np.float64).ravel()
            likelihood = emission[sentence_type, :]
            manual = likelihood * prior_vector / float(likelihood @ prior_vector)

            qs = agent.infer_states([jnp.array([sentence_type])], prior)
            posterior = _as_vector(qs)
            assert np.allclose(posterior, manual, atol=1e-5), (
                f"pymdp's infer_states posterior disagrees with the manual Bayes "
                f"step over the agent's own stored matrices at step {checked}; "
                f"the recorded live-filter score would not be attributable to "
                f"pymdp inference"
            )
            prior = agent.update_empirical_prior(action, qs)
            checked += 1
    assert checked > 100, "prefix too small for a disagreement to be visible"
