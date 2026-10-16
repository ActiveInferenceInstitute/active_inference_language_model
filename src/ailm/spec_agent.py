"""Build a pymdp agent FROM its GNN spec — closing the spec → model loop.

Why this module exists
----------------------
Until this module, every AILM level had **two sources of truth**: the GNN spec
under ``gnn/models/`` and a hand-written Python builder under ``src/ailm/``.
Conformance (``ailm.gnn``) bridges them by checking the builder's live agent
against the spec's declared structure — strictly better than no spec, strictly
worse than a spec that *is* the model. This module makes the spec build the
agent: ``StateSpaceBlock`` gives dimensions, ``InitialParameterization`` gives
structural forms, ``ActInfOntologyAnnotation`` says which slot each variable
occupies, and ``ModelParameters`` gives the counts.

The migration discipline (``TODO.md`` W1) is sequencing: for every shipped
level the spec-built agent must be **bit-identical** to the builder-built one —
same shapes, same dtypes, same bytes in ``A``/``B``/``C``/``D``/``E`` — *before*
any builder is retired. ``tests/test_spec_agent.py`` is that gate. Anything
less than identity would mean the spec describes a different model from the one
every recorded claim came from, and the ledger would silently change meaning.

Why it is not part of ``ailm.gnn``
----------------------------------
``ailm.gnn`` is deliberately stdlib-plus-numpy: a conformance checker that
imported the machinery it audits would inherit the flaws it audits for. This
module is the other side of that boundary — it imports pymdp and jax because
its job is construction, and it *uses* the checker as its final validation
step. Keeping them in separate modules keeps the guard independent.

The generator registry
----------------------
``Learned(generator=..., artifact=..., ...)`` names its provenance as strings.
Executing a path a spec names would be a far wider capability than parsing one
— a spec could name any script on disk. Resolution therefore goes through
:data:`LEARNED_REGISTRY`, an explicit **default-deny** table: a ``Learned``
form is resolved only when its *complete* keyword set, together with the
ontology role of the variable it parameterizes, exactly matches a registered
entry. Anything else — an unknown generator, a changed estimator, the right
generator annotated on the wrong slot — refuses to construct. The historical
A/B swap becomes a construction failure here, not merely a conformance one: the
brown-bigram resolver is registered for the ``TransitionMatrix`` role only.

Every resolver calls the same shipped building-block functions the hand
builders call (``build_letter_transition``, ``build_emission``, ...), so there is
exactly one implementation of every learned matrix and bit-identity is a
property of the assembly, not a numerical coincidence.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ailm.gnn import (
    GnnModelId,
    GnnSpec,
    assert_conforms,
    describe_shapes,
    load_spec,
    parse_model_id,
)

if TYPE_CHECKING:
    from pymdp.agent import Agent as PymdpAgent

__all__ = [
    "LEARNED_REGISTRY",
    "SpecConstructionError",
    "agent_slot_arrays",
    "assert_agents_bit_identical",
    "build_agent_from_spec",
    "parse_learned_kwargs",
]


class SpecConstructionError(ValueError):
    """A spec cannot be built into an agent. The message names the reason.

    Raised for an unregistered ``Learned`` provenance, a structural form that
    cannot fill its declared slot, a resolved matrix whose shape contradicts
    the ``StateSpaceBlock``, or a spec whose ``ModelParameters`` leave a slot
    unsized. Refusing to construct is the point: a guessed matrix would bind
    every downstream number to a model nothing specified.
    """


# ── LEARNED-FORM RESOLUTION ──────────────────────────────────────────────────


def parse_learned_kwargs(args: str) -> dict[str, str]:
    """Parse ``Learned(...)`` arguments into an ordered ``key -> value`` dict.

    Args:
        args: The raw argument string, e.g.
            ``"generator=scripts/generate_counts.py, artifact=vendor/brown_counts.json"``.

    Returns:
        Mapping in declaration order.

    Raises:
        SpecConstructionError: On an entry without ``=`` or a duplicate key.
    """
    kwargs: dict[str, str] = {}
    for entry in args.split(","):
        entry = entry.strip()
        if not entry:
            continue
        key, sep, value = entry.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key or not value:
            raise SpecConstructionError(
                f"malformed Learned(...) entry {entry!r}; expected key=value"
            )
        if key in kwargs:
            raise SpecConstructionError(f"duplicate Learned(...) key {key!r}")
        kwargs[key] = value
    if not kwargs:
        raise SpecConstructionError("Learned(...) carries no provenance entries")
    return kwargs


def _registry_key(role: str, kwargs: dict[str, str]) -> tuple[str, frozenset]:
    return (role, frozenset(kwargs.items()))


# Each resolver receives a mutable context dict (a per-call memo of loaded
# artifacts, so building several levels does not reload the corpora) and
# returns the canonical numpy array WITHOUT batch or trailing action axes.


def _brown_bigram(context: dict) -> np.ndarray:
    """The Brown letter bigram, ``(30, 30)`` column-stochastic."""
    if "brown_bigram" not in context:
        from ailm.levels.letter import build_letter_transition, load_brown_counts

        context["brown_bigram"] = np.asarray(build_letter_transition(load_brown_counts()))[0]
    return context["brown_bigram"]


def _action_selects_symbol(context: dict) -> np.ndarray:
    """The controllable letter transition: action ``k`` emits letter ``k``."""
    from ailm.levels.letter import N_SYMBOLS

    transition = np.zeros((N_SYMBOLS, N_SYMBOLS, N_SYMBOLS), dtype=np.float32)
    for action in range(N_SYMBOLS):
        transition[action, :, action] = 1.0
    return transition


def _phoneme_counts(context: dict) -> dict:
    if "phoneme_counts" not in context:
        from ailm.levels.phoneme import load_phoneme_counts

        context["phoneme_counts"] = load_phoneme_counts()
    return context["phoneme_counts"]


def _phone_bigram(context: dict) -> np.ndarray:
    from ailm.levels.phoneme import build_phone_transition

    counts = _phoneme_counts(context)
    return np.asarray(build_phone_transition(counts["phone_bigrams"]))[0, :, :, 0]


def _phone_unigram(context: dict) -> np.ndarray:
    from ailm.levels.phoneme import build_phone_unigram

    counts = _phoneme_counts(context)
    return np.asarray(build_phone_unigram(counts["phone_unigrams"]))[0]


def _g2p_train(context: dict) -> list:
    if "g2p_train" not in context:
        from ailm.levels.phoneme import g2p_split

        train, _test = g2p_split(_phoneme_counts(context))
        context["g2p_train"] = train
    return context["g2p_train"]


def _g2p_emission(context: dict) -> np.ndarray:
    from ailm.levels.phoneme import build_g2p_emission

    return np.asarray(build_g2p_emission(_g2p_train(context)))[0]


def _g2p_transition(context: dict) -> np.ndarray:
    from ailm.levels.phoneme import build_g2p_transition

    return np.asarray(build_g2p_transition(_g2p_train(context)))[0, :, :, 0]


def _g2p_start(context: dict) -> np.ndarray:
    from ailm.levels.phoneme import build_g2p_start

    return np.asarray(build_g2p_start(_g2p_train(context)))[0]


def _word_artifact(context: dict):
    if "word_artifact" not in context:
        from ailm.levels.word import load_word_artifact

        context["word_artifact"] = load_word_artifact()
    return context["word_artifact"]


def _word_emission(context: dict) -> np.ndarray:
    from ailm.levels.word import build_emission

    return build_emission(_word_artifact(context))


def _position_shift(context: dict) -> np.ndarray:
    from ailm.levels.word import build_position_transition

    return build_position_transition(_word_artifact(context))


def _word_prior(context: dict) -> np.ndarray:
    from ailm.levels.word import build_word_prior

    return build_word_prior(_word_artifact(context))


def _position_start(context: dict) -> np.ndarray:
    artifact = _word_artifact(context)
    start = np.zeros(artifact.n_positions, dtype=np.float64)
    start[0] = 1.0
    return start


def _case_artifact(context: dict):
    if "case_artifact" not in context:
        from ailm.levels.case import load_case_artifact

        context["case_artifact"] = load_case_artifact()
    return context["case_artifact"]


def _case_emission(context: dict) -> np.ndarray:
    from ailm.levels.case import build_case_emission

    return build_case_emission(_case_artifact(context))


def _case_transition(context: dict) -> np.ndarray:
    from ailm.levels.case import build_case_transition

    return build_case_transition(_case_artifact(context))


def _case_start(context: dict) -> np.ndarray:
    from ailm.levels.case import build_case_start

    return build_case_start(_case_artifact(context))


def _clause_models(context: dict) -> dict:
    if "clause_models" not in context:
        from ailm.levels.clause import build_type_models, load_clause_artifact

        context["clause_models"] = build_type_models(load_clause_artifact().train)
    return context["clause_models"]


def _clause_emission(context: dict) -> np.ndarray:
    from ailm.levels.clause import build_joint_emission

    return build_joint_emission()


def _clause_transition(context: dict) -> np.ndarray:
    from ailm.levels.clause import build_joint_transition

    return build_joint_transition(_clause_models(context))


def _clause_start(context: dict) -> np.ndarray:
    from ailm.levels.clause import build_joint_start

    return build_joint_start(_clause_models(context))


def _widened_clause_models(context: dict) -> dict:
    if "widened_clause_models" not in context:
        from ailm.levels.clause import build_type_models
        from ailm.levels.widened import N_WIDENED_SYMBOLS, load_widened_clause_artifact

        artifact = load_widened_clause_artifact()
        context["widened_clause_models"] = build_type_models(
            artifact.train, n_symbols=N_WIDENED_SYMBOLS
        )
    return context["widened_clause_models"]


def _widened_clause_emission(context: dict) -> np.ndarray:
    from ailm.levels.clause import build_joint_emission
    from ailm.levels.widened import N_WIDENED_SYMBOLS

    return build_joint_emission(N_WIDENED_SYMBOLS)


def _widened_clause_transition(context: dict) -> np.ndarray:
    from ailm.levels.clause import build_joint_transition

    return build_joint_transition(_widened_clause_models(context))


def _widened_clause_start(context: dict) -> np.ndarray:
    from ailm.levels.clause import build_joint_start

    return build_joint_start(_widened_clause_models(context))


def _narrative_artifact(context: dict):
    if "narrative_artifact" not in context:
        from ailm.levels.narrative import load_narrative_artifact

        context["narrative_artifact"] = load_narrative_artifact()
    return context["narrative_artifact"]


def _narrative_emission(context: dict) -> np.ndarray:
    from ailm.levels.narrative import build_phase_models

    return build_phase_models(_narrative_artifact(context).train)["emission"]


def _narrative_transition(context: dict) -> np.ndarray:
    from ailm.levels.narrative import build_phase_transition

    return build_phase_transition(_narrative_artifact(context).train)


def _narrative_start(context: dict) -> np.ndarray:
    from ailm.levels.narrative import N_PHASES

    start = np.zeros(N_PHASES, dtype=np.float64)
    start[0] = 1.0
    return start


def _entry(
    role: str, resolver: Callable[[dict], np.ndarray], **kwargs: str
) -> tuple[tuple[str, frozenset], Callable[[dict], np.ndarray]]:
    return _registry_key(role, kwargs), resolver


#: The explicit default-deny table of permitted ``Learned`` provenances. The
#: key is (ontology role, the COMPLETE keyword set of the Learned form); the
#: value produces the canonical numpy array by calling the same shipped
#: building blocks the hand builders call. A Learned form absent from this
#: table refuses to construct — including a registered provenance annotated on
#: a role it is not registered for, which is how the historical A/B swap fails
#: at construction rather than at conformance.
LEARNED_REGISTRY: dict[tuple[str, frozenset], Callable[[dict], np.ndarray]] = dict(
    (
        # L1_letter: the bigram parameterizes the TRANSITION. This role binding
        # is the load-bearing line of the whole registry.
        _entry(
            "TransitionMatrix",
            _brown_bigram,
            generator="scripts/generate_counts.py",
            artifact="vendor/brown_counts.json",
            estimator="add-1-Laplace",
            normalization="column-stochastic-over-next-letter",
        ),
        # L1g_letter_generation: structural controllable transition.
        _entry(
            "TransitionMatrix",
            _action_selects_symbol,
            generator="src/ailm/generation.py",
            artifact="build_generative_letter_agent",
            estimator="structural-action-selects-symbol",
            normalization="column-stochastic-over-next-letter-per-action",
        ),
        # L2a_phoneme_sequence.
        _entry(
            "TransitionMatrix",
            _phone_bigram,
            generator="scripts/generate_phoneme_counts.py",
            artifact="vendor/cmudict_counts.json",
            key="phone_bigrams",
            estimator="add-1-Laplace",
            normalization="column-stochastic-over-next-phone",
        ),
        _entry(
            "Prior",
            _phone_unigram,
            generator="scripts/generate_phoneme_counts.py",
            artifact="vendor/cmudict_counts.json",
            key="phone_unigrams",
            estimator="add-1-Laplace",
        ),
        # L2b_grapheme_to_phoneme.
        _entry(
            "RecognitionMatrix",
            _g2p_emission,
            generator="scripts/generate_phoneme_counts.py",
            artifact="vendor/cmudict_counts.json",
            key="g2p_pairs",
            split="train",
            alignment="length-matched-1to1",
            estimator="add-1-Laplace",
            normalization="column-stochastic-over-letters",
        ),
        _entry(
            "TransitionMatrix",
            _g2p_transition,
            generator="scripts/generate_phoneme_counts.py",
            artifact="vendor/cmudict_counts.json",
            key="g2p_pairs",
            split="train",
            estimator="add-1-Laplace",
            normalization="column-stochastic-over-next-phone",
        ),
        _entry(
            "Prior",
            _g2p_start,
            generator="scripts/generate_phoneme_counts.py",
            artifact="vendor/cmudict_counts.json",
            key="g2p_pairs",
            split="train",
            statistic="first-phone-of-word",
            estimator="add-1-Laplace",
        ),
        # L3_word.
        _entry(
            "RecognitionMatrix",
            _word_emission,
            generator="scripts/generate_word_counts.py",
            artifact="vendor/brown_word_counts.json",
            estimator="deterministic-spelling",
            normalization="column-stochastic-over-observations",
        ),
        _entry(
            "TransitionMatrix",
            _position_shift,
            generator="src/ailm/levels/word.py",
            artifact="build_position_transition",
            estimator="structural-shift",
            normalization="column-stochastic-over-next-position",
        ),
        _entry(
            "Prior",
            _word_prior,
            generator="scripts/generate_word_counts.py",
            artifact="vendor/brown_word_counts.json",
            estimator="add-k-Laplace",
            normalization="sums-to-one-over-vocabulary",
        ),
        _entry(
            "Prior",
            _position_start,
            generator="src/ailm/levels/word.py",
            artifact="build_word_agent",
            estimator="structural-point-mass",
            normalization="sums-to-one-over-positions",
        ),
        # L6_narrative: document phases over sentence types.
        _entry(
            "RecognitionMatrix",
            _narrative_emission,
            generator="scripts/generate_narrative_counts.py",
            artifact="vendor/ud_ewt_narrative_counts.json",
            estimator="add-k-Laplace-per-phase",
            normalization="column-stochastic-over-types",
        ),
        _entry(
            "TransitionMatrix",
            _narrative_transition,
            generator="scripts/generate_narrative_counts.py",
            artifact="vendor/ud_ewt_narrative_counts.json",
            estimator="tercile-pair-counts",
            normalization="column-stochastic-left-to-right",
        ),
        _entry(
            "Prior",
            _narrative_start,
            generator="src/ailm/levels/narrative.py",
            artifact="build_narrative_agent",
            estimator="structural-point-mass-opening",
            normalization="sums-to-one-over-phases",
        ),
        # L5_clause: the joint (type × case) mixture.
        _entry(
            "RecognitionMatrix",
            _clause_emission,
            generator="src/ailm/levels/clause.py",
            artifact="build_joint_emission",
            estimator="structural-case-readoff",
            normalization="column-stochastic-over-roles",
        ),
        _entry(
            "TransitionMatrix",
            _clause_transition,
            generator="scripts/generate_clause_counts.py",
            artifact="vendor/ud_ewt_clause_counts.json",
            estimator="add-k-Laplace-per-type-block",
            normalization="column-stochastic-block-diagonal",
        ),
        _entry(
            "Prior",
            _clause_start,
            generator="scripts/generate_clause_counts.py",
            artifact="vendor/ud_ewt_clause_counts.json",
            estimator="type-prior-times-start",
            normalization="sums-to-one-over-joint",
        ),
        # L5w_clause_widened: the same joint construction at the declared wider
        # alphabet. The complete provenance key keeps it distinct from L5_clause.
        _entry(
            "RecognitionMatrix",
            _widened_clause_emission,
            generator="src/ailm/levels/clause.py",
            artifact="build_joint_emission",
            estimator="structural-case-readoff",
            normalization="column-stochastic-over-symbols",
            n_symbols="10",
        ),
        _entry(
            "TransitionMatrix",
            _widened_clause_transition,
            generator="scripts/generate_clause_widened_counts.py",
            artifact="data/widened/ud_ewt_clause_widened_counts.json",
            estimator="add-k-Laplace-per-type-block",
            normalization="column-stochastic-block-diagonal",
        ),
        _entry(
            "Prior",
            _widened_clause_start,
            generator="scripts/generate_clause_widened_counts.py",
            artifact="data/widened/ud_ewt_clause_widened_counts.json",
            estimator="type-prior-times-start",
            normalization="sums-to-one-over-joint",
        ),
        # L4_case.
        _entry(
            "RecognitionMatrix",
            _case_emission,
            generator="scripts/generate_case_counts.py",
            artifact="vendor/ud_ewt_case_counts.json",
            estimator="add-k-Laplace",
            normalization="column-stochastic-over-upos",
        ),
        _entry(
            "TransitionMatrix",
            _case_transition,
            generator="scripts/generate_case_counts.py",
            artifact="vendor/ud_ewt_case_counts.json",
            estimator="add-k-Laplace",
            normalization="column-stochastic-over-next-case",
        ),
        _entry(
            "Prior",
            _case_start,
            generator="scripts/generate_case_counts.py",
            artifact="vendor/ud_ewt_case_counts.json",
            estimator="add-k-Laplace",
            normalization="sums-to-one-over-cases",
        ),
    )
)


def _resolve_learned(name: str, role: str, args: str, context: dict) -> np.ndarray:
    """Resolve one ``Learned`` form through the registry, default-deny.

    Raises:
        SpecConstructionError: If the (role, complete keyword set) pair is not
            registered. The message distinguishes "unknown provenance" from
            "known provenance on a role it is not permitted to fill", because
            the second is the historical wiring bug wearing a valid citation.
    """
    kwargs = parse_learned_kwargs(args)
    resolver = LEARNED_REGISTRY.get(_registry_key(role, kwargs))
    if resolver is not None:
        return np.asarray(resolver(context))

    permitted_roles = sorted(
        {
            registered_role
            for (registered_role, registered_kwargs) in LEARNED_REGISTRY
            if registered_kwargs == frozenset(kwargs.items())
        }
    )
    if permitted_roles:
        raise SpecConstructionError(
            f"{name}: Learned provenance {dict(kwargs)!r} is registered for "
            f"role(s) {permitted_roles}, not for {role!r}. A known generator "
            f"annotated on the wrong slot is the historical anti-informative "
            f"wiring; it is refused at construction."
        )
    raise SpecConstructionError(
        f"{name}: Learned provenance {dict(kwargs)!r} (role {role!r}) is not in "
        f"LEARNED_REGISTRY. Specs cannot execute arbitrary named paths; add an "
        f"explicit registry entry that calls the shipped generator, or fix the "
        f"spec."
    )


# ── ASSEMBLY ─────────────────────────────────────────────────────────────────


def _declared_canonical_dims(spec: GnnSpec, name: str) -> tuple[int, ...]:
    """Declared dims with trailing size-1 axes stripped, matching the checker."""
    dims = tuple(spec.variable_dims(name))
    while len(dims) > 1 and dims[-1] == 1:
        dims = dims[:-1]
    return dims


def _factor_index(name: str) -> int:
    """Hidden-factor index bound by an ``_f<N>`` suffix; 0 when unsuffixed."""
    import re

    match = re.search(r"_f(?P<index>\d+)$", name)
    return int(match["index"]) if match else 0


def _role_variables_by_factor(spec: GnnSpec, role: str, expected: int) -> list[str]:
    """The variables carrying ``role``, ordered by factor index, one per factor.

    Raises:
        SpecConstructionError: If any factor has zero or several bindings.
    """
    by_index: dict[int, str] = {}
    for variable in spec.variables_with_role(role):
        index = _factor_index(variable)
        if index in by_index:
            raise SpecConstructionError(
                f"{variable} and {by_index[index]} both bind role {role} to "
                f"factor {index}; construction cannot choose between them."
            )
        by_index[index] = variable
    missing = [index for index in range(expected) if index not in by_index]
    if missing or len(by_index) != expected:
        raise SpecConstructionError(
            f"role {role} binds factors {sorted(by_index)} but the spec "
            f"declares {expected} factor(s); every factor needs exactly one "
            f"{role}."
        )
    return [by_index[index] for index in range(expected)]


def _factor_sizes(spec: GnnSpec, n_factors: int) -> list[int]:
    """Per-factor sizes from ``ModelParameters``, refusing to guess.

    Raises:
        SpecConstructionError: If any size is missing or not a positive int.
    """
    if n_factors == 1:
        value = spec.parameters.get("num_hidden_states")
        if not isinstance(value, int) or value < 1:
            raise SpecConstructionError(
                f"ModelParameters num_hidden_states={value!r} does not size the "
                f"single hidden factor; construction refuses to guess."
            )
        return [value]
    sizes: list[int] = []
    for index in range(n_factors):
        key = f"num_hidden_states_f{index}"
        value = spec.parameters.get(key)
        if not isinstance(value, int) or value < 1:
            raise SpecConstructionError(
                f"ModelParameters {key}={value!r} does not size factor {index}; "
                f"an unsized factor cannot be constructed."
            )
        sizes.append(value)
    return sizes


def _positive_int(spec: GnnSpec, key: str, default: int | None = None) -> int:
    value = spec.parameters.get(key, default)
    if not isinstance(value, int) or value < 1:
        raise SpecConstructionError(f"ModelParameters {key}={value!r} is not a positive integer.")
    return value


def _check_resolved_shape(spec: GnnSpec, name: str, array: np.ndarray) -> None:
    """The resolved array must have exactly the declared canonical dims.

    Raises:
        SpecConstructionError: On any mismatch. This is what makes a
            transposed-factor or swapped-alphabet spec fail *construction*: the
            registered generator produces the true matrix, whose axes then
            contradict the mutated declaration.
    """
    declared = _declared_canonical_dims(spec, name)
    actual = tuple(int(d) for d in array.shape)
    if actual != declared:
        raise SpecConstructionError(
            f"{name}: StateSpaceBlock declares dims {declared} but the resolved "
            f"array has {actual}. The spec contradicts the artifact it cites; "
            f"construction refuses rather than reshapes."
        )


def build_agent_from_spec(
    model: str | GnnModelId | GnnSpec,
    *,
    context: dict | None = None,
) -> PymdpAgent:
    """Construct a live pymdp Agent from a GNN spec, and prove it conforms.

    The spec is the only input: dimensions from ``StateSpaceBlock``, counts
    from ``ModelParameters``, slot roles from ``ActInfOntologyAnnotation``,
    and every matrix from ``InitialParameterization`` — structural forms
    materialised directly, ``Learned`` forms resolved through
    :data:`LEARNED_REGISTRY`. After assembly the live agent is checked against
    its own spec with :func:`ailm.gnn.assert_conforms`, so nothing this
    function builds can disagree with what the spec declares.

    Args:
        model: A model id (``"L3_word"``), a parsed :class:`GnnModelId`, or an
            already-parsed :class:`GnnSpec` (used by tests to probe doctored
            specs without touching the registry on disk).
        context: Optional artifact memo, shared across calls to avoid
            reloading corpora. Mutated in place.

    Returns:
        A configured ``pymdp.agent.Agent``.

    Raises:
        SpecConstructionError: If the spec cannot be built — unregistered
            provenance, a form that cannot fill its slot, a shape
            contradiction, or an unsized factor.
        GnnConformanceError: If the constructed agent somehow fails its own
            spec's conformance check. Reaching this indicates a bug in this
            module, not in the spec.
    """
    import jax.numpy as jnp
    from pymdp.agent import Agent

    if isinstance(model, GnnSpec):
        spec = model
    else:
        model_id = model if isinstance(model, GnnModelId) else parse_model_id(model)
        spec = load_spec(model_id)
    memo = context if context is not None else {}

    n_factors = _positive_int(spec, "num_factors", 1)
    n_modalities = _positive_int(spec, "num_modalities", 1)
    if n_modalities != 1:
        raise SpecConstructionError(
            f"num_modalities={n_modalities}: multi-modality construction is not "
            f"implemented; every shipped level has one modality."
        )
    num_obs = _positive_int(spec, "num_obs")
    num_controls = _positive_int(spec, "num_controls", 1)
    sizes = _factor_sizes(spec, n_factors)

    def resolve(name: str, role: str) -> tuple[str, Any]:
        """Return ``(form, canonical array or size)`` for one variable."""
        if name not in spec.parameterization:
            raise SpecConstructionError(
                f"{name} ({role}) has no InitialParameterization entry; a slot "
                f"without a declared form cannot be constructed."
            )
        form, args = spec.parameterization[name]
        if form == "Learned":
            array = _resolve_learned(name, role, args, memo)
            _check_resolved_shape(spec, name, array)
            return form, array
        return form, int(args)

    # ── A: one modality, indexed [observation, *state factors in order] ──────
    (a_name,) = _role_variables_by_factor(spec, "RecognitionMatrix", 1)
    a_form, a_value = resolve(a_name, "RecognitionMatrix")
    if a_form == "Identity":
        if sizes != [a_value] or num_obs != a_value:
            raise SpecConstructionError(
                f"{a_name}: Identity({a_value}) cannot fill a "
                f"({num_obs}, {tuple(sizes)}) recognition slot; an identity "
                f"emission requires the observation and state alphabets to be "
                f"the same."
            )
        a_slot = jnp.eye(a_value)[jnp.newaxis]
    elif a_form == "Learned":
        expected = (num_obs, *sizes)
        if tuple(a_value.shape) != expected:
            raise SpecConstructionError(
                f"{a_name}: resolved recognition matrix has shape "
                f"{tuple(a_value.shape)} but ModelParameters require {expected} "
                f"(observation axis first, then the factors in order)."
            )
        a_slot = jnp.array(a_value, dtype=jnp.float32)[jnp.newaxis]
    else:
        raise SpecConstructionError(
            f"{a_name}: form {a_form} is not constructible in the RecognitionMatrix slot."
        )

    # ── B: one transition per factor, square, with the declared control axis ──
    b_names = _role_variables_by_factor(spec, "TransitionMatrix", n_factors)
    b_slots = []
    for index, b_name in enumerate(b_names):
        size = sizes[index]
        factor_controls = num_controls if index == 0 else 1
        b_form, b_value = resolve(b_name, "TransitionMatrix")
        if b_form == "Identity":
            if factor_controls != 1:
                raise SpecConstructionError(
                    f"{b_name}: Identity({b_value}) cannot fill a controlled "
                    f"transition with num_controls={factor_controls}; a "
                    f"declared action set that is not in B is an action set the "
                    f"agent cannot take."
                )
            if b_value != size:
                raise SpecConstructionError(
                    f"{b_name}: Identity({b_value}) does not match factor {index}'s size {size}."
                )
            b_slots.append(jnp.eye(size)[jnp.newaxis, :, :, jnp.newaxis])
        elif b_form == "Learned":
            expected = (size, size) if factor_controls == 1 else (size, size, factor_controls)
            if tuple(b_value.shape) != expected:
                raise SpecConstructionError(
                    f"{b_name}: resolved transition has shape "
                    f"{tuple(b_value.shape)} but factor {index} with "
                    f"num_controls={factor_controls} requires {expected}."
                )
            base = jnp.array(b_value, dtype=jnp.float32)
            if factor_controls == 1:
                base = base[:, :, jnp.newaxis]
            b_slots.append(base[jnp.newaxis])
        else:
            raise SpecConstructionError(
                f"{b_name}: form {b_form} is not constructible in the TransitionMatrix slot."
            )

    # ── C: flat at construction for every shipped level ──────────────────────
    c_variables = spec.variables_with_role("LogPreferenceVector")
    if len(c_variables) != 1:
        raise SpecConstructionError(
            f"expected exactly one LogPreferenceVector, found {c_variables}."
        )
    c_form, c_value = resolve(c_variables[0], "LogPreferenceVector")
    if c_form != "Zeros" or c_value != num_obs:
        raise SpecConstructionError(
            f"{c_variables[0]}: only Zeros({num_obs}) is constructible in the "
            f"LogPreferenceVector slot — every shipped level's preference is "
            f"flat at construction and installed per-step by the level above."
        )
    c_slot = jnp.zeros((1, num_obs))

    # ── D: one prior per factor ──────────────────────────────────────────────
    d_names = _role_variables_by_factor(spec, "Prior", n_factors)
    d_slots = []
    for index, d_name in enumerate(d_names):
        size = sizes[index]
        d_form, d_value = resolve(d_name, "Prior")
        if d_form == "Uniform":
            if d_value != size:
                raise SpecConstructionError(
                    f"{d_name}: Uniform({d_value}) does not match factor {index}'s size {size}."
                )
            d_slots.append(jnp.ones((1, size)) / size)
        elif d_form == "Learned":
            if tuple(d_value.shape) != (size,):
                raise SpecConstructionError(
                    f"{d_name}: resolved prior has shape {tuple(d_value.shape)} "
                    f"but factor {index} requires ({size},)."
                )
            d_slots.append(jnp.array(d_value, dtype=jnp.float32)[jnp.newaxis])
        else:
            raise SpecConstructionError(
                f"{d_name}: form {d_form} is not constructible in the Prior slot."
            )

    # ── E: the policy prior, uniform at construction when declared ───────────
    e_variables = spec.variables_with_role("PolicyPrior")
    e_slot = None
    if e_variables:
        if len(e_variables) > 1:
            raise SpecConstructionError(f"expected at most one PolicyPrior, found {e_variables}.")
        e_form, e_value = resolve(e_variables[0], "PolicyPrior")
        if e_form != "Uniform" or e_value != num_controls:
            raise SpecConstructionError(
                f"{e_variables[0]}: only Uniform({num_controls}) is "
                f"constructible in the PolicyPrior slot — the habit is "
                f"installed per-step by the generation loop, not baked into "
                f"the constructed agent (docs/decisions.md entry 20)."
            )
        e_slot = jnp.full((1, num_controls), 1.0 / num_controls, dtype=jnp.float32)

    gamma = spec.parameters.get("gamma")

    if n_factors == 1:
        kwargs: dict[str, Any] = {
            "num_controls": [num_controls],
            "batch_size": 1,
        }
        if e_slot is not None:
            kwargs["E"] = e_slot
        if gamma is not None:
            kwargs["gamma"] = float(gamma)
        agent = Agent(A=[a_slot], B=b_slots, C=[c_slot], D=d_slots, **kwargs)
    else:
        if e_slot is not None or gamma is not None:
            raise SpecConstructionError(
                "policy priors on multi-factor levels are not implemented; no "
                "shipped spec declares one."
            )
        agent = Agent(
            A=[a_slot],
            B=b_slots,
            C=[c_slot],
            D=d_slots,
            A_dependencies=[list(range(n_factors))],
            B_dependencies=[[index] for index in range(n_factors)],
        )

    # The constructed agent must satisfy the very checker that guards the hand
    # builders. This cannot loosen anything — it can only catch assembly bugs.
    assert_conforms(spec, describe_shapes(agent))
    return agent


# ── BIT-IDENTITY ─────────────────────────────────────────────────────────────


def agent_slot_arrays(agent: Any) -> dict[str, list[np.ndarray]]:
    """Every parameter array of an agent, as numpy, keyed by slot letter.

    ``E`` is included when present so the generative level's policy prior is
    compared rather than silently skipped.
    """
    arrays: dict[str, list[np.ndarray]] = {}
    for attribute in ("A", "B", "C", "D"):
        value = getattr(agent, attribute)
        arrays[attribute] = [np.asarray(item) for item in value]
    e_value = getattr(agent, "E", None)
    if e_value is not None:
        arrays["E"] = [np.asarray(e_value)]
    return arrays


def assert_agents_bit_identical(left: Any, right: Any) -> None:
    """Assert two agents carry byte-for-byte identical parameter arrays.

    Not ``allclose``: identical shapes, identical dtypes, identical bytes. A
    close-but-different agent is a *different model* whose recorded numbers
    merely resemble the shipped ones — exactly the drift the W1 migration gate
    exists to refuse.

    Raises:
        AssertionError: Naming the first differing slot, with the maximum
            absolute difference when the values differ.
    """
    left_arrays = agent_slot_arrays(left)
    right_arrays = agent_slot_arrays(right)
    if set(left_arrays) != set(right_arrays):
        raise AssertionError(
            f"agents carry different slot sets: {sorted(left_arrays)} vs {sorted(right_arrays)}"
        )
    for slot in sorted(left_arrays):
        left_list, right_list = left_arrays[slot], right_arrays[slot]
        if len(left_list) != len(right_list):
            raise AssertionError(f"slot {slot}: {len(left_list)} arrays vs {len(right_list)}")
        for index, (left_item, right_item) in enumerate(zip(left_list, right_list, strict=True)):
            if left_item.shape != right_item.shape:
                raise AssertionError(
                    f"slot {slot}[{index}]: shape {left_item.shape} vs {right_item.shape}"
                )
            if left_item.dtype != right_item.dtype:
                raise AssertionError(
                    f"slot {slot}[{index}]: dtype {left_item.dtype} vs {right_item.dtype}"
                )
            if not np.array_equal(left_item, right_item):
                difference = float(
                    np.max(np.abs(left_item.astype(np.float64) - right_item.astype(np.float64)))
                )
                raise AssertionError(
                    f"slot {slot}[{index}]: values differ "
                    f"(max |delta| = {difference:.3e}). Bit-identity, not "
                    f"closeness, is the migration gate."
                )
