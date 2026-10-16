"""The W1 migration gate: the spec-built agent IS the builder-built agent.

``ailm.spec_agent.build_agent_from_spec`` constructs a live pymdp agent from a
GNN spec alone. Before any hand-written builder may be retired, the spec-built
agent must be **bit-identical** — same shapes, same dtypes, same bytes — to the
builder-built agent for every shipped level. "Conformant" is not enough:
conformance checks structure and roles, and two structurally conformant agents
can still be different models. Every recorded claim in ``data/claims.json``
came from the builder-built agents; a spec path that reproduced them only
approximately would silently rebase the entire ledger onto a different model.

Alongside the identity gate, this file pins the construction layer's negative
space:

* the comparator itself is non-vacuous (a planted perturbation must fail);
* ``Learned`` provenance resolution is default-deny, including the case where a
  *registered* provenance is annotated on a role it is not registered for —
  the historical A/B swap arriving with a valid citation;
* every mutation spec either **fails construction** or constructs an agent that
  is **not** bit-identical to its base — and the partition between those two
  outcomes is asserted exactly, so a mutation drifting from "refused" to
  "constructible" is a visible event rather than a silent one.
"""

from __future__ import annotations

import numpy as np
import pytest

from ailm.gnn import discover_models, parse_gnn_string
from ailm.spec_agent import (
    LEARNED_REGISTRY,
    SpecConstructionError,
    agent_slot_arrays,
    assert_agents_bit_identical,
    build_agent_from_spec,
    parse_learned_kwargs,
)

# ── SHARED CONSTRUCTION STATE ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def context() -> dict:
    """One artifact memo for the whole module, so corpora load once."""
    return {}


@pytest.fixture(scope="module")
def spec_built(context) -> dict:
    """Model id string -> the spec-built agent, memoized."""
    cache: dict = {}

    def build(name: str):
        if name not in cache:
            cache[name] = build_agent_from_spec(name, context=context)
        return cache[name]

    return {"build": build}


# ── THE EQUIVALENCE GATE ─────────────────────────────────────────────────────
#
# One test per shipped level, each comparing the spec-built agent against the
# agent the SHIPPED builder constructs — the exact construction every recorded
# claim came from.


def test_l1_letter_spec_build_is_bit_identical(spec_built):
    from ailm.levels.letter import (
        build_letter_agent,
        build_letter_transition,
        load_brown_counts,
    )

    builder_agent = build_letter_agent(build_letter_transition(load_brown_counts()))
    assert_agents_bit_identical(spec_built["build"]("L1_letter"), builder_agent)


def test_l1g_generation_spec_build_is_bit_identical(spec_built):
    from ailm.generation import build_generative_letter_agent
    from ailm.levels.letter import build_letter_transition, load_brown_counts

    habit = np.asarray(build_letter_transition(load_brown_counts()))[0]
    builder_agent = build_generative_letter_agent(habit)
    spec_agent = spec_built["build"]("L1g_letter_generation")
    assert_agents_bit_identical(spec_agent, builder_agent)
    # gamma is a construction parameter of record, declared in ModelParameters.
    assert float(np.asarray(spec_agent.gamma).ravel()[0]) == float(
        np.asarray(builder_agent.gamma).ravel()[0]
    )


def test_l2a_phoneme_spec_build_is_bit_identical(spec_built):
    from ailm.levels.phoneme import (
        build_phone_transition,
        build_phone_unigram,
        build_phoneme_agent,
        load_phoneme_counts,
    )

    counts = load_phoneme_counts()
    builder_agent = build_phoneme_agent(
        build_phone_transition(counts["phone_bigrams"]),
        build_phone_unigram(counts["phone_unigrams"]),
    )
    assert_agents_bit_identical(spec_built["build"]("L2a_phoneme_sequence"), builder_agent)


def test_l2b_g2p_spec_build_is_bit_identical(spec_built):
    from ailm.levels.phoneme import (
        build_g2p_agent,
        build_g2p_emission,
        build_g2p_start,
        build_g2p_transition,
        g2p_split,
        load_phoneme_counts,
    )

    train, _test = g2p_split(load_phoneme_counts())
    builder_agent = build_g2p_agent(
        build_g2p_emission(train),
        build_g2p_transition(train),
        build_g2p_start(train),
    )
    assert_agents_bit_identical(spec_built["build"]("L2b_grapheme_to_phoneme"), builder_agent)


def test_l3_word_spec_build_is_bit_identical(spec_built):
    from ailm.levels.word import build_word_agent, load_word_artifact

    builder_agent = build_word_agent(load_word_artifact())
    assert_agents_bit_identical(spec_built["build"]("L3_word"), builder_agent)


def test_l4_case_spec_build_is_bit_identical(spec_built):
    from ailm.levels.case import build_case_agent, load_case_artifact

    builder_agent = build_case_agent(load_case_artifact())
    assert_agents_bit_identical(spec_built["build"]("L4_case"), builder_agent)


def test_l5_clause_spec_build_is_bit_identical(spec_built):
    from ailm.levels.clause import build_clause_agent, load_clause_artifact

    builder_agent = build_clause_agent(load_clause_artifact())
    assert_agents_bit_identical(spec_built["build"]("L5_clause"), builder_agent)


def test_l5w_widened_spec_build_is_bit_identical(spec_built):
    from ailm.levels.clause import build_type_models
    from ailm.levels.widened import (
        N_WIDENED_SYMBOLS,
        build_widened_agent,
        load_widened_clause_artifact,
    )

    artifact = load_widened_clause_artifact()
    builder_agent = build_widened_agent(
        build_type_models(artifact.train, n_symbols=N_WIDENED_SYMBOLS)
    )
    assert_agents_bit_identical(spec_built["build"]("L5w_clause_widened"), builder_agent)


def test_widened_emission_alphabet_argument_cannot_be_ignored(context):
    from ailm.gnn import spec_path

    source = spec_path("L5w_clause_widened").read_text(encoding="utf-8")
    assert "n_symbols=10" in source
    wrong = parse_gnn_string(source.replace("n_symbols=10", "n_symbols=8"))
    with pytest.raises(SpecConstructionError, match="not in LEARNED_REGISTRY"):
        build_agent_from_spec(wrong, context=context)


def test_l6_narrative_spec_build_is_bit_identical(spec_built):
    from ailm.levels.narrative import build_narrative_agent, load_narrative_artifact

    builder_agent = build_narrative_agent(load_narrative_artifact())
    assert_agents_bit_identical(spec_built["build"]("L6_narrative"), builder_agent)


def test_every_canonical_spec_is_covered_by_an_identity_test():
    """Discovery guard: a new canonical spec must join the gate above.

    Without this, a seventh level could ship with a spec and a builder that
    quietly disagree, and every test in this file would stay green.
    """
    covered = {
        "L1_letter",
        "L1g_letter_generation",
        "L2a_phoneme_sequence",
        "L2b_grapheme_to_phoneme",
        "L3_word",
        "L4_case",
        "L5_clause",
        "L5w_clause_widened",
        "L6_narrative",
    }
    canonical = {
        str(model_id)
        for model_id in discover_models()
        if model_id.is_canonical and model_id.kind == "L"
    }
    assert canonical == covered, (
        f"canonical specs {sorted(canonical - covered)} have no bit-identity "
        f"test; add one before relying on the spec path for them"
    )


# ── COMPARATOR NON-VACUITY ───────────────────────────────────────────────────


class _AgentShim:
    """A minimal stand-in carrying doctored slot arrays for the comparator."""

    def __init__(self, arrays: dict):
        for slot in ("A", "B", "C", "D"):
            setattr(self, slot, list(arrays.get(slot, [])))
        self.E = arrays.get("E", [None])[0] if "E" in arrays else None


def _shim_from(agent) -> _AgentShim:
    return _AgentShim(agent_slot_arrays(agent))


def test_the_comparator_fails_on_a_planted_perturbation(spec_built):
    """POSITIVE CONTROL: identity that cannot fail is not a gate."""
    agent = spec_built["build"]("L1_letter")
    doctored = _shim_from(agent)
    perturbed = np.array(doctored.B[0], copy=True)
    perturbed.flat[0] += 1e-6
    doctored.B[0] = perturbed
    with pytest.raises(AssertionError, match="values differ"):
        assert_agents_bit_identical(agent, doctored)


def test_the_comparator_fails_on_a_dtype_change(spec_built):
    """float64 values equal to the float32 ones are still a different agent."""
    agent = spec_built["build"]("L1_letter")
    doctored = _shim_from(agent)
    doctored.A[0] = np.asarray(doctored.A[0], dtype=np.float64)
    with pytest.raises(AssertionError, match="dtype"):
        assert_agents_bit_identical(agent, doctored)


def test_the_comparator_fails_on_a_missing_slot(spec_built):
    agent = spec_built["build"]("L1g_letter_generation")
    doctored = _shim_from(agent)
    doctored.E = None
    with pytest.raises(AssertionError, match="slot sets"):
        assert_agents_bit_identical(agent, doctored)


# ── THE REGISTRY IS DEFAULT-DENY ─────────────────────────────────────────────


_MINIMAL_SPEC = """\
## ModelName

Doctored letter level

## StateSpaceBlock

A[30,30,type=float]
B[30,30,1,type=float]
C[30,1,type=float]
D[30,1,type=float]

## InitialParameterization

A=Identity(30)
B={b_form}
C=Zeros(30)
D=Uniform(30)

## ActInfOntologyAnnotation

A={a_role}
B={b_role}
C=LogPreferenceVector
D=Prior

## ModelParameters

num_hidden_states: 30
num_obs: 30
num_factors: 1
num_modalities: 1
"""

_BROWN_BIGRAM_LEARNED = (
    "Learned(generator=scripts/generate_counts.py, "
    "artifact=vendor/brown_counts.json, estimator=add-1-Laplace, "
    "normalization=column-stochastic-over-next-letter)"
)


def _doctored_spec(
    *,
    a_role: str = "RecognitionMatrix",
    b_role: str = "TransitionMatrix",
    b_form: str = _BROWN_BIGRAM_LEARNED,
):
    return parse_gnn_string(
        _MINIMAL_SPEC.format(a_role=a_role, b_role=b_role, b_form=b_form),
        origin="<doctored>",
    )


def test_the_canonical_doctored_spec_constructs(context):
    """The doctored-spec harness itself must be able to succeed.

    Without this anchor, the two refusal tests below could pass because the
    harness is broken rather than because the registry refused.
    """
    agent = build_agent_from_spec(_doctored_spec(), context=context)
    assert np.asarray(agent.B[0]).shape == (1, 30, 30, 1)


def test_an_unregistered_provenance_is_refused(context):
    """Changing one keyword — the estimator — must refuse construction."""
    doctored = _doctored_spec(
        b_form=_BROWN_BIGRAM_LEARNED.replace("add-1-Laplace", "add-2-Laplace")
    )
    with pytest.raises(SpecConstructionError, match="not in LEARNED_REGISTRY"):
        build_agent_from_spec(doctored, context=context)


def test_a_registered_provenance_on_the_wrong_role_is_refused(context):
    """The historical A/B swap, arriving with a valid citation.

    The brown-bigram provenance is registered for TransitionMatrix. A spec
    annotating the same citation on the RecognitionMatrix slot describes the
    anti-informative wiring, and must be refused BY ROLE — before any shape or
    value check could pass it (the letter level's A and B are shape-identical).
    """
    doctored = parse_gnn_string(
        _MINIMAL_SPEC.format(
            a_role="TransitionMatrix",
            b_role="RecognitionMatrix",
            b_form=_BROWN_BIGRAM_LEARNED,
        ),
        origin="<doctored-swap>",
    )
    with pytest.raises(SpecConstructionError, match="registered for role.*TransitionMatrix"):
        build_agent_from_spec(doctored, context=context)


def test_learned_kwargs_parse_and_refuse_malformed_entries():
    parsed = parse_learned_kwargs("generator=a.py, artifact=b.json")
    assert parsed == {"generator": "a.py", "artifact": "b.json"}
    with pytest.raises(SpecConstructionError, match="key=value"):
        parse_learned_kwargs("generator")
    with pytest.raises(SpecConstructionError, match="duplicate"):
        parse_learned_kwargs("generator=a, generator=b")
    with pytest.raises(SpecConstructionError, match="no provenance"):
        parse_learned_kwargs("  ")


def test_the_registry_carries_no_role_free_entry():
    """Every entry binds a role from MATRIX_ROLES or the policy prior."""
    from ailm.gnn import MATRIX_ROLES

    permitted = set(MATRIX_ROLES) | {"PolicyPrior"}
    for role, _kwargs in LEARNED_REGISTRY:
        assert role in permitted, f"registry entry with unknown role {role!r}"


# ── EVERY REFUSAL PATH FIRES ─────────────────────────────────────────────────
#
# Construction is a security boundary as much as a convenience, so its refusal
# branches are behavior, not dead code. Each case below doctors the minimal
# constructible spec in exactly one way and must be refused with the named
# message. The anchor proving the undoctored template constructs is
# test_the_canonical_doctored_spec_constructs above.

_STRUCTURAL_BASE = _MINIMAL_SPEC.format(
    a_role="RecognitionMatrix",
    b_role="TransitionMatrix",
    b_form="Identity(30)",
)

_G2P_EMISSION_LEARNED = (
    "Learned(generator=scripts/generate_phoneme_counts.py, "
    "artifact=vendor/cmudict_counts.json, key=g2p_pairs, split=train, "
    "alignment=length-matched-1to1, estimator=add-1-Laplace, "
    "normalization=column-stochastic-over-letters)"
)

_REFUSALS = {
    "two_transitions_on_one_factor": (
        [
            ("B[30,30,1,type=float]", "B[30,30,1,type=float]\nB2[30,30,1,type=float]"),
            ("B=Identity(30)", "B=Identity(30)\nB2=Identity(30)"),
            ("B=TransitionMatrix", "B=TransitionMatrix\nB2=TransitionMatrix"),
        ],
        "both bind role TransitionMatrix",
    ),
    "no_prior_binds_the_factor": (
        [("D=Prior", "D=NextHiddenState")],
        "every factor needs exactly one Prior",
    ),
    "unsized_single_factor": (
        [("num_hidden_states: 30\n", "")],
        "does not size the single hidden factor",
    ),
    "missing_num_obs": (
        [("num_obs: 30\n", "")],
        "num_obs=None",
    ),
    "multi_modality_is_not_implemented": (
        [("num_modalities: 1", "num_modalities: 2")],
        "multi-modality construction is not implemented",
    ),
    "slot_without_a_declared_form": (
        [("D=Uniform(30)\n", "")],
        "no InitialParameterization entry",
    ),
    "uniform_recognition_matrix": (
        [("A=Identity(30)", "A=Uniform(30)")],
        r"not constructible in the\s+RecognitionMatrix slot",
    ),
    "identity_emission_over_mismatched_alphabets": (
        [
            ("num_obs: 30", "num_obs: 29"),
            ("C=Zeros(30)", "C=Zeros(29)"),
            ("C[30,1,type=float]", "C[29,1,type=float]"),
        ],
        r"Identity\(30\) cannot fill",
    ),
    "uniform_transition": (
        [("B=Identity(30)", "B=Uniform(30)")],
        r"not constructible in the\s+TransitionMatrix slot",
    ),
    "identity_transition_of_the_wrong_size": (
        [("B=Identity(30)", "B=Identity(29)")],
        r"does not match factor\s+0's size 30",
    ),
    "non_flat_preference": (
        [("C=Zeros(30)", "C=Uniform(30)")],
        r"only Zeros\(30\) is constructible",
    ),
    "no_preference_vector": (
        [("C=LogPreferenceVector", "C=NextHiddenState"), ("C=Zeros(30)\n", "")],
        "exactly one LogPreferenceVector",
    ),
    "zeros_prior": (
        [("D=Uniform(30)", "D=Zeros(30)")],
        "not constructible in the Prior slot",
    ),
    "uniform_prior_of_the_wrong_size": (
        [("D=Uniform(30)", "D=Uniform(29)")],
        r"Uniform\(29\) does not match factor\s+0's size 30",
    ),
    "policy_prior_that_is_not_uniform": (
        [
            ("D[30,1,type=float]", "D[30,1,type=float]\nE[30,1,type=float]"),
            ("D=Uniform(30)", "D=Uniform(30)\nE=Zeros(30)"),
            ("D=Prior", "D=Prior\nE=PolicyPrior"),
        ],
        "constructible in the PolicyPrior slot",
    ),
    "two_policy_priors": (
        [
            (
                "D[30,1,type=float]",
                "D[30,1,type=float]\nE[30,1,type=float]\nE2[30,1,type=float]",
            ),
            ("D=Uniform(30)", "D=Uniform(30)\nE=Uniform(30)\nE2=Uniform(30)"),
            ("D=Prior", "D=Prior\nE=PolicyPrior\nE2=PolicyPrior"),
        ],
        "at most one PolicyPrior",
    ),
    "resolved_recognition_matrix_contradicting_the_counts": (
        # The g2p emission resolves to (30, 39) and matches its declaration,
        # but the doctored ModelParameters size the hidden factor at 40 — the
        # resolved-vs-required check must catch what the declaration cannot.
        [
            ("A[30,30,type=float]", "A[30,39,type=float]"),
            ("B[30,30,1,type=float]", "B[39,39,1,type=float]"),
            ("D[30,1,type=float]", "D[39,1,type=float]"),
            ("A=Identity(30)", f"A={_G2P_EMISSION_LEARNED}"),
            ("B=Identity(30)", "B=Identity(39)"),
            ("D=Uniform(30)", "D=Uniform(39)"),
            ("num_hidden_states: 30", "num_hidden_states: 40"),
        ],
        r"ModelParameters require \(30, 40\)",
    ),
}


@pytest.mark.parametrize("case", sorted(_REFUSALS))
def test_construction_refusal(case, context):
    replacements, match = _REFUSALS[case]
    text = _STRUCTURAL_BASE
    for old, new in replacements:
        assert old in text, f"{case}: template drifted; {old!r} not found"
        text = text.replace(old, new)
    doctored = parse_gnn_string(text, origin=f"<doctored:{case}>")
    with pytest.raises(SpecConstructionError, match=match):
        build_agent_from_spec(doctored, context=context)


def test_a_learned_prior_contradicting_the_factor_size_is_refused(context):
    """Declared dims can agree with the artifact while ModelParameters do not."""
    text = _STRUCTURAL_BASE
    for old, new in [
        ("A[30,30,type=float]", "A[40,40,type=float]"),
        ("B[30,30,1,type=float]", "B[40,40,1,type=float]"),
        ("C[30,1,type=float]", "C[40,1,type=float]"),
        ("D[30,1,type=float]", "D[39,1,type=float]"),
        ("A=Identity(30)", "A=Identity(40)"),
        ("B=Identity(30)", "B=Identity(40)"),
        ("C=Zeros(30)", "C=Zeros(40)"),
        (
            "D=Uniform(30)",
            "D=Learned(generator=scripts/generate_phoneme_counts.py, "
            "artifact=vendor/cmudict_counts.json, key=g2p_pairs, split=train, "
            "statistic=first-phone-of-word, estimator=add-1-Laplace)",
        ),
        ("num_hidden_states: 30", "num_hidden_states: 40"),
        ("num_obs: 30", "num_obs: 40"),
    ]:
        assert old in text, old
        text = text.replace(old, new)
    doctored = parse_gnn_string(text, origin="<doctored:learned-prior-size>")
    with pytest.raises(SpecConstructionError, match=r"requires \(40,\)"):
        build_agent_from_spec(doctored, context=context)


def test_a_policy_prior_on_a_multifactor_level_is_refused(context):
    """No shipped multi-factor spec declares gamma or E; refusing beats guessing."""
    from ailm.gnn import spec_path

    spec_text = spec_path("L3_word").read_text(encoding="utf-8")
    doctored = parse_gnn_string(
        spec_text.replace("num_modalities: 1", "num_modalities: 1\ngamma: 1.0"),
        origin="<doctored:multifactor-gamma>",
    )
    with pytest.raises(SpecConstructionError, match="multi-factor"):
        build_agent_from_spec(doctored, context=context)


def test_the_comparator_fails_on_an_array_count_mismatch(spec_built):
    agent = spec_built["build"]("L1_letter")
    doctored = _shim_from(agent)
    doctored.B = doctored.B + [doctored.B[0]]
    with pytest.raises(AssertionError, match="arrays"):
        assert_agents_bit_identical(agent, doctored)


def test_the_comparator_fails_on_a_shape_change(spec_built):
    agent = spec_built["build"]("L1_letter")
    doctored = _shim_from(agent)
    doctored.B[0] = np.asarray(doctored.B[0]).reshape(30, 30, 1, 1)
    with pytest.raises(AssertionError, match="shape"):
        assert_agents_bit_identical(agent, doctored)


def test_the_structural_base_itself_constructs(context):
    """Anchor for the refusal battery: the undoctored template must build."""
    agent = build_agent_from_spec(
        parse_gnn_string(_STRUCTURAL_BASE, origin="<structural-base>"),
        context=context,
    )
    assert np.asarray(agent.B[0]).shape == (1, 30, 30, 1)


# ── MUTATIONS MUST FAIL CONSTRUCTION, OR CONSTRUCT A DIFFERENT MODEL ─────────
#
# The mutation suite predates construction: each mutation FAILS conformance
# against the correct live agent (the per-level gate tests). Construction is
# the stronger, cheaper check for most of them: the spec alone refuses. The
# three that do construct describe internally consistent but WRONG models, and
# for those the guard is that the constructed agent differs from the base —
# no mutation may silently reproduce the shipped model.

#: Mutations whose spec is internally consistent enough to construct. Each
#: builds a real agent that must then differ from its base. Exact partition:
#: a mutation moving between these sets is a change in what the construction
#: layer can catch, and must be a deliberate, reviewed event.
CONSTRUCTIBLE_MUTATIONS = {
    "L3_word@frozen_position",
    "L4_case@frozen_transition",
    "L4_case@uninformed_prior",
    "L5_clause@identity_joint_transition",
    "L6_narrative@uniform_phase_prior",
}


def _mutation_ids():
    return [str(model_id) for model_id in discover_models() if not model_id.is_canonical]


def test_the_mutation_partition_is_exact():
    """Both classes must be non-empty, and CONSTRUCTIBLE_MUTATIONS ⊆ discovered."""
    discovered = set(_mutation_ids())
    assert discovered, "no mutations discovered; the parametrised tests are vacuous"
    assert discovered >= CONSTRUCTIBLE_MUTATIONS
    assert discovered - CONSTRUCTIBLE_MUTATIONS, (
        "every mutation constructs; the refusal path is never exercised"
    )


@pytest.mark.parametrize("name", _mutation_ids())
def test_no_mutation_reproduces_the_shipped_model(name, context, spec_built):
    from ailm.gnn import parse_model_id

    model_id = parse_model_id(name)
    if name in CONSTRUCTIBLE_MUTATIONS:
        mutated = build_agent_from_spec(model_id, context=context)
        base = spec_built["build"](str(model_id.base))
        with pytest.raises(AssertionError):
            assert_agents_bit_identical(mutated, base)
    else:
        with pytest.raises(SpecConstructionError):
            build_agent_from_spec(model_id, context=context)
