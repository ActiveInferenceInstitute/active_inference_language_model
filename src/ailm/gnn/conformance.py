"""Running every conformance check and reporting what disagreed.

This module is the entry point and nothing else: :func:`check_conformance`
collects the declaration checks from :mod:`ailm.gnn.declarations` and the
live-object checks from :mod:`ailm.gnn.live`, and :func:`assert_conforms` turns
a non-empty report into a raise.

It was one file of 1,260 lines holding all three concerns. The split is along
the seam that already existed in the code and was invisible in it: checks that
need only the spec, checks that need the built object, and the orchestration
over both. The private helpers are re-exported below so importers that reached
past the entry point keep working.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from ailm.gnn.declarations import (
    _check_declaration_coherence,
    _check_inference_configuration,
    _declared_factor_count,
)
from ailm.gnn.live import (
    _check_emission_factor_count,
    _check_multifactor_conformance,
    _check_precision,
    _check_structural_form,
)
from ailm.gnn.shapes import _canonical_dims
from ailm.gnn.types import (
    _MANDATORY_ROLES,
    MATRIX_ROLES,
    GnnConformanceError,
    GnnSpec,
    LiveConformanceInputs,
)

__all__ = ["assert_conforms", "check_conformance"]

_LIVE_FIELD_NAMES = frozenset(f.name for f in fields(LiveConformanceInputs))


def check_conformance(
    spec: GnnSpec,
    live: LiveConformanceInputs | None = None,
    **legacy: Any,
) -> list[str]:
    """Check a live model against its spec and return EVERY violation.

    Args:
        spec: Parsed spec.
        live: The live model, as the typed record :func:`ailm.gnn.shapes.describe_shapes`
            returns. Every check reads its slots from here.

    Returns:
        Violation strings, ordered from spec-level problems to slot-level ones.
    """
    if live is None:
        live = _coerce_legacy_live(legacy)
    else:
        _reject_legacy_live(legacy)
    return _check_conformance_recorded(spec, live)


def _coerce_legacy_live(legacy: dict[str, Any]) -> LiveConformanceInputs:
    """Build the record from the pre-record positional/keyword call form.

    The adapter exists only for callers not yet updated to the typed record; it
    applies the same rules the old signature did: absent means None, and an
    unrecognized keyword (or more than one positional live model) is a
    ``TypeError``, not a silently dropped value.
    """
    unknown = set(legacy) - _LIVE_FIELD_NAMES
    if unknown:
        raise TypeError(
            f"check_conformance() got unexpected keyword argument(s) "
            f"{sorted(unknown)}; expected the fields of LiveConformanceInputs"
        )
    kwargs = {name: legacy[name] for name in _LIVE_FIELD_NAMES if name in legacy}
    return LiveConformanceInputs(**kwargs)


def _reject_legacy_live(legacy: dict[str, Any]) -> None:
    """Refuse to mix the record form with the legacy keyword form."""
    if legacy:
        raise TypeError(
            f"check_conformance() got unexpected keyword argument(s) "
            f"{sorted(legacy)} alongside a LiveConformanceInputs; pass one "
            f"form, not both"
        )


def _check_conformance_recorded(spec: GnnSpec, live: LiveConformanceInputs) -> list[str]:
    """Run every check against the typed record. See :func:`check_conformance`.

    An empty list means conformant. Nothing here inspects a learned probability:
    the checks are the declared dimensions, the declared semantic role of each
    slot, the internal agreement of the spec with its own ``ModelParameters``,
    and the declared structural form of each matrix.

    Checks performed:

    1. ``ModelParameters`` carries ``num_hidden_states`` and ``num_obs``.
    2. Exactly one variable is annotated with each of ``RecognitionMatrix``,
       ``TransitionMatrix`` and ``Prior``.
    3. A shape was supplied for every role-carrying variable the spec declares.
    4. Declared dims equal actual dims, after removing the pymdp batch axis and
       trailing singleton axes (see :func:`_canonical_dims`).
    5. Spec-internal agreement: the ``RecognitionMatrix`` is declared
       ``num_obs`` by ``num_hidden_states``, the ``TransitionMatrix``
       ``num_hidden_states`` square, the ``Prior`` a ``num_hidden_states``
       vector, and the ``LogPreferenceVector`` (if present) a ``num_obs`` vector.
    6. The live ``TransitionMatrix`` is square in its state dims.
    7. The live ``RecognitionMatrix`` is ``num_obs`` by ``num_hidden_states``.
    8. ``num_hidden_states`` / ``num_obs`` / ``num_factors`` / ``num_modalities``
       agree with the live model.
    9. Every ``InitialParameterization`` structural form holds for the live
       array, when that array was supplied, including the ``normalization`` axis
       and the positivity implied by an unqualified Laplace ``estimator``.
    10. Every declared belief, observation and policy vector is sized by
        ``ModelParameters``, every ``type=`` matches what its role holds, the
        ``Time`` block names a declared time index, and the ``Connections``
        graph wires each ``Prior`` into a ``HiddenState``, leaves no variable
        unwired, and agrees with ``num_controls`` about whether a policy reaches
        the dynamics.
    11. A declared ``gamma`` equals the live agent's policy precision, and a
        declared ``PolicyPrior`` matches the live ``E``.

    """
    violations: list[str] = []
    supplied_shapes = {
        "a": live.a_shape,
        "b": live.b_shape,
        "c": live.c_shape,
        "d": live.d_shape,
        "e": live.e_shape,
    }
    supplied_arrays = {
        "a": live.a_matrix,
        "b": live.b_matrix,
        "c": live.c_vector,
        "d": live.d_vector,
        "e": live.e_vector,
    }

    # Checks that need the spec alone, or one slot that has no per-factor form.
    # Run before the dispatch so a model does not escape them by gaining a
    # factor — a check reachable from one path only is a check with a loophole.
    violations.extend(_check_declaration_coherence(spec))
    violations.extend(_check_precision(spec, live.gamma))
    violations.extend(
        _check_inference_configuration(
            spec,
            use_utility=live.use_utility,
            use_states_info_gain=live.use_states_info_gain,
            policy_len=live.policy_len,
            n_policies=live.n_policies,
            num_controls=live.num_controls,
            e_vector=live.e_vector,
        )
    )

    # A factorized model has one transition and one prior per factor, so the
    # "exactly one variable per role" rule and the single scalar
    # num_hidden_states both stop applying. Dispatch rather than special-case:
    # the single-factor path below is left exactly as it was, so generalizing
    # cannot perturb the levels already gated by it.
    if _declared_factor_count(spec) > 1:
        violations.extend(
            _check_multifactor_conformance(
                spec,
                a_matrices=live.a_matrices,
                b_matrices=live.b_matrices,
                c_vectors=live.c_vectors,
                d_vectors=live.d_vectors,
                e_vector=live.e_vector,
                n_modalities=live.n_modalities,
                n_factors=live.n_factors,
            )
        )
        return violations

    violations.extend(_check_emission_factor_count(spec, live.a_matrix))

    num_states = spec.parameters.get("num_hidden_states")
    num_obs = spec.parameters.get("num_obs")
    for key, value in (("num_hidden_states", num_states), ("num_obs", num_obs)):
        if value is None:
            violations.append(
                f"ModelParameters is missing {key!r}. Without it the spec cannot "
                f"pin the size of any slot."
            )
        elif not isinstance(value, int) or value < 1:
            violations.append(f"ModelParameters {key}={value!r} is not a positive integer.")
            if key == "num_hidden_states":
                num_states = None
            else:
                num_obs = None

    # Role -> the single variable annotated with it.
    role_variable: dict[str, str] = {}
    for role in MATRIX_ROLES:
        names = spec.variables_with_role(role)
        if len(names) > 1:
            violations.append(
                f"role {role} is annotated on {len(names)} variables {names} — "
                f"ambiguous, so no slot is pinned."
            )
        elif len(names) == 1:
            role_variable[role] = names[0]
        elif role in _MANDATORY_ROLES:
            violations.append(
                f"no variable is annotated {role} in ActInfOntologyAnnotation, so "
                f"nothing pins the {MATRIX_ROLES[role].upper()} slot."
            )

    # Declared vs live dims, per role.
    actual_dims: dict[str, tuple[int, ...] | None] = {}
    for role, name in role_variable.items():
        slot = MATRIX_ROLES[role]
        declared = _canonical_dims(spec.variable_dims(name), strip_batch=False)
        shape = supplied_shapes[slot]
        if shape is None:
            actual_dims[role] = None
            violations.append(
                f"{name}: spec declares {name}={role} but no {slot}_shape was "
                f"supplied to the conformance check, so the slot is unverified."
            )
            continue
        live_dims = _canonical_dims(shape, strip_batch=True)
        actual_dims[role] = live_dims
        if live_dims != declared:
            violations.append(
                f"{name} ({role}): declared dims {declared} but the live model "
                f"has {tuple(shape)} -> canonical {live_dims}."
            )

    # A level with real actions carries a third axis on B that _canonical_dims
    # does NOT reduce away, so the expected transition shape depends on how many
    # controls the spec declares. Every AILM level was single-control until
    # generation gave the letter level one action per symbol.
    num_controls = spec.parameters.get("num_controls")
    if num_controls is not None and (not isinstance(num_controls, int) or num_controls < 1):
        violations.append(
            f"ModelParameters num_controls={num_controls!r} is not a positive integer."
        )
        num_controls = None
    controlled = isinstance(num_controls, int) and num_controls > 1

    # Spec-internal agreement between StateSpaceBlock and ModelParameters.
    if num_states is not None and num_obs is not None:
        expectations = {
            "RecognitionMatrix": (num_obs, num_states),
            "TransitionMatrix": (
                (num_states, num_states, num_controls) if controlled else (num_states, num_states)
            ),
            "Prior": (num_states,),
            "LogPreferenceVector": (num_obs,),
        }
        if isinstance(num_controls, int) and num_controls > 0:
            # A policy prior is a distribution over the action set, so it is
            # sized by num_controls and not by either state or observation
            # alphabet. Without num_controls nothing sizes it, which
            # _check_vector_declarations reports separately.
            expectations["PolicyPrior"] = (num_controls,)
        for role, expected in expectations.items():
            variable_name = role_variable.get(role)
            if variable_name is None:
                continue
            declared = _canonical_dims(spec.variable_dims(variable_name), strip_batch=False)
            if declared != expected:
                violations.append(
                    f"{variable_name} ({role}): StateSpaceBlock declares {declared} but "
                    f"ModelParameters num_obs={num_obs}, "
                    f"num_hidden_states={num_states} require {expected}. The spec "
                    f"contradicts itself."
                )

    # Live transition must be square in its state dims.
    transition_name = role_variable.get("TransitionMatrix")
    transition_dims = actual_dims.get("TransitionMatrix")
    if transition_name is not None and transition_dims is not None:
        if len(transition_dims) < 2 or transition_dims[0] != transition_dims[1]:
            violations.append(
                f"{transition_name} (TransitionMatrix): live canonical dims "
                f"{transition_dims} are not square in the state axes. A "
                f"transition maps states to states."
            )
        elif controlled and (len(transition_dims) != 3 or transition_dims[2] != num_controls):
            # The control axis is the one _canonical_dims deliberately does not
            # reduce, so a level that declares real actions must have them.
            violations.append(
                f"{transition_name} (TransitionMatrix): ModelParameters declares "
                f"num_controls={num_controls} but the live canonical dims are "
                f"{transition_dims}. A declared action set that is not in B is an "
                f"action set the agent cannot take."
            )
        elif num_states is not None and transition_dims[0] != num_states:
            violations.append(
                f"{transition_name} (TransitionMatrix): live state dimension "
                f"{transition_dims[0]} != ModelParameters "
                f"num_hidden_states={num_states}."
            )

    # Live recognition matrix must be num_obs by num_hidden_states.
    recognition_name = role_variable.get("RecognitionMatrix")
    recognition_dims = actual_dims.get("RecognitionMatrix")
    if recognition_name is not None and recognition_dims is not None:
        if num_obs is not None and num_states is not None:
            expected = (num_obs, num_states)
            if recognition_dims != expected:
                violations.append(
                    f"{recognition_name} (RecognitionMatrix): live canonical dims "
                    f"{recognition_dims} != (num_obs, num_hidden_states) "
                    f"= {expected}. The emission must be indexed "
                    f"[observation, hidden state]; the transpose is a different "
                    f"model."
                )

    prior_name = role_variable.get("Prior")
    prior_dims = actual_dims.get("Prior")
    if prior_name is not None and prior_dims is not None and num_states is not None:
        if prior_dims != (num_states,):
            violations.append(
                f"{prior_name} (Prior): live canonical dims {prior_dims} != "
                f"(num_hidden_states,) = ({num_states},)."
            )

    preference_name = role_variable.get("LogPreferenceVector")
    preference_dims = actual_dims.get("LogPreferenceVector")
    if (
        preference_name is not None
        and preference_dims is not None
        and num_obs is not None
        and preference_dims != (num_obs,)
    ):
        violations.append(
            f"{preference_name} (LogPreferenceVector): live canonical dims "
            f"{preference_dims} != (num_obs,) = ({num_obs},)."
        )

    # Declared factor / modality counts.
    for key, live_count in (
        ("num_factors", live.n_factors),
        ("num_modalities", live.n_modalities),
    ):
        declared_count = spec.parameters.get(key)
        if declared_count is None or live_count is None:
            continue
        if declared_count != live_count:
            violations.append(
                f"ModelParameters {key}={declared_count} but the live model has {live_count}."
            )

    # Structural forms, checked against the live arrays.
    for name, (form, args) in spec.parameterization.items():
        ontology_role = spec.ontology.get(name)
        matrix_slot = MATRIX_ROLES.get(ontology_role or "")
        if matrix_slot is None:
            continue
        array = supplied_arrays[matrix_slot]
        if array is None:
            continue
        violations.extend(_check_structural_form(name, form, args, array))

    return violations


def assert_conforms(spec: GnnSpec, live: LiveConformanceInputs, **legacy: Any) -> None:
    """Assert a live model conforms to ``spec``, reporting every violation.

    Failing on the first problem only would force a one-at-a-time repair loop,
    so this raises once with the complete list.

    Args:
        spec: Parsed spec.
        live: The live model, as the typed record :func:`ailm.gnn.shapes.describe_shapes`
            returns — typically ``describe_shapes(agent)`` directly, no splat.

    Raises:
        GnnConformanceError: If any violation is found. The message lists all of
            them, one per line.
    """
    violations = check_conformance(spec, live, **legacy)
    if violations:
        listing = "\n".join(f"  {index}. {v}" for index, v in enumerate(violations, 1))
        raise GnnConformanceError(
            f"{len(violations)} GNN conformance violation(s) against "
            f"{spec.model_name!r} ({spec.origin}):\n{listing}\n"
            f"Conformance pins STRUCTURE and declared roles. If code and spec "
            f"disagree, decide which is wrong on the evidence — never edit the "
            f"spec just to make this pass."
        )
