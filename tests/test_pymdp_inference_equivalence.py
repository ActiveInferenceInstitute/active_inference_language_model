"""Identify exactly which recorded estimands pymdp inference reproduces.

The project's claim is that AILM is six *pymdp* generative models, so it matters
which levels actually route a recorded number through pymdp and which only hold an
Agent beside a numpy scorer.

Audited by source inspection rather than assumed:

===========  ==========================  ==================================
level        scoring function            pymdp on the measurement path
===========  ==========================  ==================================
L1 letter    ``compute_perplexity``      ``infer_states`` + ``update_empirical_prior``
L2A phoneme  ``compute_phoneme_perplexity``  same
L2B g2p      ``decode_positions``        same
L2B g2p      ``decode_phonemes``         same
L3 word      ``word_perplexity``         no; closed form, exact at any vocabulary size
L4 case      ``decode_case_positions``   no; numpy forward filter
L5 clause    sequence log-evidence       no; closed-form mixture
L6 observed-position association         no; true tercile indexes numpy emission
L6 live-filter score                     yes; separately recorded sequential predictor
===========  ==========================  ==================================

For L1/L2 that is settled elsewhere. For L3–L5 the Agent exists and its
*shapes* are conformance-checked; at L5 its *matrices* are also walked by hand,
but that does not turn a closed-form measurement into a pymdp inference result.
`test_stage5_gate.py`'s
``test_closed_form_oracle_and_live_agent_agree`` is precise about this in its
comment ("the live agent's float32 matrices walk to the same evidence") and its
name is not: it exercises pymdp's storage, not its inference.

This file closes the L4 comparison and proves every lower-level builder returns
a real Agent. It deliberately does not claim equivalence for a different
estimand. Stage 6 now records both quantities under different labels:
``heldout_type_logloss`` uses known position, while ``live_agent_type_logloss``
runs ``infer_states`` and ``update_empirical_prior`` through ``B`` and ``D``.
The fact that they differ is information about the estimands, not a tolerance to
hide.

The L4 comparison is bounded on purpose. ``infer_states`` is one call per observation, which is the
scaling limitation the manuscript already records, so these compare a prefix
rather than the full population. A prefix that agrees everywhere is not a proof of
full-set equality, and the manuscript says so; it is evidence over a stated
subset, which is more than shape conformance gives.
"""

from __future__ import annotations

import numpy as np
import pytest

#: Sentences or words compared per level. Large enough that a systematic
#: disagreement shows, small enough that one call per observation stays quick.
PREFIX = 25


def _as_vector(qs) -> np.ndarray:
    """pymdp returns a list of batched factor beliefs; take factor 0, flattened."""
    return np.asarray(qs[0]).ravel()


def test_level_4_case_pymdp_inference_reproduces_the_numpy_decoder():
    """The recorded Stage 4 hits come from numpy. pymdp must agree with them."""
    import jax.numpy as jnp

    from ailm.cases import CaseRole
    from ailm.levels.case import (
        CASE_TO_IDX,
        build_case_agent,
        build_case_emission,
        build_case_start,
        build_case_transition,
        decode_case_positions,
        load_case_artifact,
    )

    artifact = load_case_artifact("derived")
    numpy_hits = decode_case_positions(
        artifact,
        build_case_emission(artifact),
        build_case_transition(artifact),
        build_case_start(artifact),
    )
    agent = build_case_agent(artifact)
    action = jnp.zeros((1, 1), dtype=jnp.int32)

    pymdp_hits: list[bool] = []
    for sequence in artifact.test[:PREFIX]:
        observations = [artifact.upos_to_idx[upos] for upos, _case in sequence]
        truth = [CASE_TO_IDX[CaseRole[case]] for _upos, case in sequence]
        prior = [jnp.asarray(np.asarray(agent.D[0]))]
        for position, observation in enumerate(observations):
            qs = agent.infer_states([jnp.array([observation])], prior)
            pymdp_hits.append(bool(int(np.argmax(_as_vector(qs))) == truth[position]))
            prior = agent.update_empirical_prior(action, qs)

    compared = len(pymdp_hits)
    assert compared > 100, "prefix too small for a disagreement to be visible"
    disagreements = [
        i for i, (a, b) in enumerate(zip(pymdp_hits, numpy_hits[:compared], strict=True)) if a != b
    ]
    assert disagreements == [], (
        f"pymdp inference and the numpy decoder disagree at {len(disagreements)} of "
        f"{compared} Stage 4 positions (first at index {disagreements[:3]}). The recorded "
        f"stage4_* numbers come from the numpy path, so a disagreement means they are not "
        f"attributable to the pymdp agent the paper describes."
    )


def test_the_case_agent_belief_is_a_distribution_at_every_step():
    """A negative control on the comparison above.

    If ``infer_states`` returned something unnormalized, an argmax comparison
    would still pass while the beliefs were meaningless. The equivalence test
    cannot see that; this one can.
    """
    import jax.numpy as jnp

    from ailm.levels.case import build_case_agent, load_case_artifact

    artifact = load_case_artifact("derived")
    agent = build_case_agent(artifact)
    action = jnp.zeros((1, 1), dtype=jnp.int32)

    prior = [jnp.asarray(np.asarray(agent.D[0]))]
    for sequence in artifact.test[:5]:
        for upos, _case in sequence:
            qs = agent.infer_states([jnp.array([artifact.upos_to_idx[upos]])], prior)
            belief = _as_vector(qs)
            assert belief.min() >= -1e-9, "a negative belief is not a distribution"
            assert abs(float(belief.sum()) - 1.0) < 1e-5, (
                f"pymdp belief sums to {belief.sum()}, not 1; the argmax comparison "
                f"elsewhere would pass regardless"
            )
            prior = agent.update_empirical_prior(action, qs)


@pytest.mark.parametrize(
    ("level", "builder", "loader"),
    [
        ("L4 case", "build_case_agent", "load_case_artifact"),
        ("L5 clause", "build_clause_agent", "load_clause_artifact"),
        ("L6 narrative", "build_narrative_agent", "load_narrative_artifact"),
    ],
)
def test_every_numpy_scored_level_still_constructs_a_real_agent(level, builder, loader):
    """The Agent must be a live pymdp object, not a stand-in.

    Cheap, and it is the assumption every equivalence check above rests on. A
    builder that quietly returned a namespace with the right attribute names would
    satisfy the shape conformance tests.
    """
    import importlib

    module_name = {
        "L4 case": "ailm.levels.case",
        "L5 clause": "ailm.levels.clause",
        "L6 narrative": "ailm.levels.narrative",
    }[level]
    module = importlib.import_module(module_name)
    if not hasattr(module, loader):
        pytest.skip(f"{module_name} has no {loader}")
    agent = getattr(module, builder)(getattr(module, loader)())

    from pymdp.agent import Agent

    assert isinstance(agent, Agent), (
        f"{level} builder returned {type(agent).__name__}, not pymdp Agent"
    )
    assert hasattr(agent, "infer_states"), f"{level} agent cannot infer"
