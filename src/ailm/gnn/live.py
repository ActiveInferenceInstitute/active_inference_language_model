"""A built agent checked against the spec that declares it.

Where :mod:`ailm.gnn.declarations` reads the spec alone, every check here needs
the live object: the arrays pymdp actually holds, their shapes, their stochastic
orientation, the precision the agent was constructed with. These are the ways a
declaration and a built model disagree while each looks individually reasonable
— a transposed transition is still a valid stochastic matrix, and a spec that
declares the other orientation is still a valid spec.

The spec does not generate the builder. Both are written by hand, which is the
only reason running this is worth anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ailm.gnn.declarations import _declared_factor_count, _factor_index, _factor_sizes
from ailm.gnn.parsing import parse_learned_arguments
from ailm.gnn.shapes import _canonical_array, _canonical_dims
from ailm.gnn.types import (
    _COLUMN_STOCHASTIC_PREFIX,
    _LEARNED_FORM,
    _STRICTLY_POSITIVE_ESTIMATORS,
    _STRUCTURAL_TOL,
    _SUMS_TO_ONE_PREFIX,
    MATRIX_ROLES,
    GnnSpec,
)


def _check_precision(spec: GnnSpec, gamma: Any) -> list[str]:
    """Check a declared ``gamma`` against the live agent's policy precision.

    The defect: L1g declares ``gamma: 16.0`` in ``ModelParameters`` and nothing
    read it, so the spec and the shipped agent could name different operating
    points and both look conformant. Precision is the one parameter this project
    measured a two-sided failure in on the shipped artifact — too low reverts to
    the habit, too high collapses its greedy prefix path onto one word — so a
    spec that disagrees with the model about it describes a different experiment.
    """
    declared = spec.parameters.get("gamma")
    if declared is None:
        return []
    if isinstance(declared, (int, float)) and not np.isfinite(declared):
        return [f"ModelParameters gamma={declared!r} is not finite."]
    if not isinstance(declared, (int, float)) or declared <= 0:
        return [f"ModelParameters gamma={declared!r} is not a positive number."]
    if gamma is None:
        return [
            f"ModelParameters declares gamma={declared} but no live gamma was "
            f"supplied, so the policy precision is unverified."
        ]
    live = np.asarray(gamma, dtype=np.float64).ravel()
    if live.size == 0:
        return [f"ModelParameters declares gamma={declared} but the live gamma is empty."]
    if not np.all(np.isfinite(live)):
        return [f"ModelParameters declares gamma={declared} but the live gamma is non-finite."]
    deviation = float(np.max(np.abs(live - float(declared))))
    if deviation > _STRUCTURAL_TOL:
        return [
            f"ModelParameters gamma={declared} but the live agent has "
            f"gamma={live.tolist()}. Precision weighs the goal against the habit; "
            f"a spec and a model that disagree on it are two different models."
        ]
    return []


def _check_emission_factor_count(spec: GnnSpec, live_a: Any) -> list[str]:
    """Compare the live emission's state axes to the graph that declares them.

    The defect: ``Connections`` is where a spec says how many hidden factors the
    emission depends on — ``s_word-A`` and ``s_pos-A`` at L3 declare an emission
    over both — and nothing compared that to the live ``A``. A spec wiring one
    factor to an ``A`` that indexes two hides a dependency it never declared,
    and the dimension checks pass either way because they read ``A``'s dims from
    ``StateSpaceBlock``, not from the graph.
    """
    if live_a is None or not spec.connections:
        return []
    recognitions = set(spec.variables_with_role("RecognitionMatrix"))
    if not recognitions:
        return []
    hidden = set(spec.variables_with_role("HiddenState"))
    touching: set[str] = set()
    for connection in spec.connections:
        endpoints = set(connection.source) | set(connection.target)
        if endpoints & recognitions:
            touching |= endpoints & hidden
    if not touching:
        return [
            "no HiddenState is connected to the RecognitionMatrix, so the graph "
            "declares an emission depending on no hidden state at all."
        ]
    live_axes = _canonical_array(live_a).ndim - 1
    if live_axes != len(touching):
        return [
            f"Connections wire {len(touching)} hidden state(s) {sorted(touching)} "
            f"to the RecognitionMatrix, but the live emission carries "
            f"{live_axes} state axis/axes. A modality depending on a factor the "
            f"graph does not name is a dependency nothing declared."
        ]
    return []


def _check_multifactor_conformance(
    spec: GnnSpec,
    *,
    a_matrices: Sequence[Any] | None,
    b_matrices: Sequence[Any] | None,
    c_vectors: Sequence[Any] | None,
    d_vectors: Sequence[Any] | None,
    e_vector: Any,
    n_modalities: int | None,
    n_factors: int | None,
) -> list[str]:
    """Conformance for a model with more than one hidden factor.

    The single-factor rules generalize rather than relax:

    * "exactly one ``TransitionMatrix``" becomes "exactly one per factor", each
      bound to its factor by an ``_f<N>`` suffix; likewise ``Prior``.
    * "exactly one ``RecognitionMatrix``" becomes "exactly one per modality".
    * The scalar ``num_hidden_states`` becomes ``num_hidden_states_f<N>``.
    * A transition must be square **in its own factor's size**, and a prior a
      vector of that size — so a factorization cannot pass by having the right
      total number of states distributed wrongly.
    * A recognition matrix whose canonical dims are ``(num_obs, *sizes)``
      declares a modality depending on several factors. The factor order in
      that shape is checked, so transposing two factors of different sizes is a
      violation rather than a reshape.

    Returns:
        Every violation found, in spec-then-slot order.
    """
    violations: list[str] = []
    declared_factors = _declared_factor_count(spec)
    sizes, size_problems = _factor_sizes(spec, declared_factors)
    violations.extend(size_problems)

    num_obs = spec.parameters.get("num_obs")
    if num_obs is None:
        violations.append(
            "ModelParameters is missing 'num_obs'. Without it the emission cannot be pinned."
        )
    elif not isinstance(num_obs, int) or num_obs < 1:
        violations.append(f"ModelParameters num_obs={num_obs!r} is not a positive integer.")
        num_obs = None

    if n_factors is not None and n_factors != declared_factors:
        violations.append(
            f"ModelParameters num_factors={declared_factors} but the live model has {n_factors}."
        )
    declared_modalities = spec.parameters.get("num_modalities")
    if (
        isinstance(declared_modalities, int)
        and n_modalities is not None
        and declared_modalities != n_modalities
    ):
        violations.append(
            f"ModelParameters num_modalities={declared_modalities} but the live "
            f"model has {n_modalities}."
        )

    def _by_factor(role: str, expected: int, label: str) -> dict[int, str]:
        names = spec.variables_with_role(role)
        if len(names) != expected:
            violations.append(
                f"role {role} is annotated on {len(names)} variables {names}, but "
                f"the model has {expected} {label}. One {role} per {label[:-1]} is "
                f"what binds each slot; any other count leaves a slot unpinned or "
                f"ambiguous."
            )
        bound: dict[int, str] = {}
        for name in names:
            index = _factor_index(name)
            if index in bound:
                violations.append(
                    f"{name} and {bound[index]} both bind to factor {index}. Use "
                    f"an _f<N> suffix so each {role} names its own factor."
                )
                continue
            bound[index] = name
        return bound

    transitions = _by_factor("TransitionMatrix", declared_factors, "factors")
    priors = _by_factor("Prior", declared_factors, "factors")
    recognitions = _by_factor(
        "RecognitionMatrix", n_modalities if n_modalities is not None else 1, "modalities"
    )

    def _live(arrays: Sequence[Any] | None, index: int) -> np.ndarray | None:
        if arrays is None or index >= len(arrays):
            return None
        return _canonical_array(arrays[index])

    # Transitions: square in their own factor's size, and matching the spec.
    for index in range(declared_factors):
        name = transitions.get(index)
        if name is None:
            continue
        declared = _canonical_dims(spec.variable_dims(name), strip_batch=False)
        size = sizes[index] if index < len(sizes) else None
        if size is not None and declared[:2] != (size, size):
            violations.append(
                f"{name} (TransitionMatrix, factor {index}): StateSpaceBlock "
                f"declares {declared} but num_hidden_states_f{index}={size} "
                f"requires ({size}, {size}). The spec contradicts itself."
            )
        live = _live(b_matrices, index)
        if live is None:
            violations.append(
                f"{name}: spec declares a TransitionMatrix for factor {index} but "
                f"no live array was supplied, so the slot is unverified."
            )
            continue
        if live.ndim < 2 or live.shape[0] != live.shape[1]:
            violations.append(
                f"{name} (TransitionMatrix, factor {index}): live canonical dims "
                f"{live.shape} are not square in the state axes."
            )
        elif size is not None and live.shape[0] != size:
            violations.append(
                f"{name} (TransitionMatrix, factor {index}): live state dimension "
                f"{live.shape[0]} != num_hidden_states_f{index}={size}."
            )

    # Priors: one vector per factor, sized by that factor.
    for index in range(declared_factors):
        name = priors.get(index)
        if name is None:
            continue
        size = sizes[index] if index < len(sizes) else None
        live = _live(d_vectors, index)
        if live is None:
            violations.append(
                f"{name}: spec declares a Prior for factor {index} but no live "
                f"array was supplied, so the slot is unverified."
            )
            continue
        if size is not None and live.shape != (size,):
            violations.append(
                f"{name} (Prior, factor {index}): live canonical dims "
                f"{live.shape} != (num_hidden_states_f{index},) = ({size},)."
            )

    # Recognition matrices: (num_obs, *sizes of the factors they depend on).
    known_sizes = [s for s in sizes if s is not None]
    for index, name in sorted(recognitions.items()):
        live = _live(a_matrices, index)
        if live is None:
            violations.append(
                f"{name}: spec declares a RecognitionMatrix for modality {index} "
                f"but no live array was supplied, so the slot is unverified."
            )
            continue
        if num_obs is not None and live.shape[0] != num_obs:
            violations.append(
                f"{name} (RecognitionMatrix, modality {index}): live canonical "
                f"dims {live.shape} do not start with num_obs={num_obs}. The "
                f"emission must be indexed [observation, ...state factors]; the "
                f"transpose is a different model."
            )
        state_axes = tuple(int(d) for d in live.shape[1:])
        if len(known_sizes) == declared_factors and state_axes != tuple(known_sizes):
            violations.append(
                f"{name} (RecognitionMatrix, modality {index}): live state axes "
                f"{state_axes} != the declared factor sizes "
                f"{tuple(known_sizes)}, in order. A modality depending on several "
                f"factors must index them in factor order; swapping two is a "
                f"different model, not a reshape."
            )
        declared = _canonical_dims(spec.variable_dims(name), strip_batch=False)
        if declared != tuple(int(d) for d in live.shape):
            violations.append(
                f"{name} (RecognitionMatrix, modality {index}): declared dims "
                f"{declared} but the live canonical dims are "
                f"{tuple(int(d) for d in live.shape)}."
            )

    # Preferences are indexed by observation, exactly as in the single-factor case.
    for name in spec.variables_with_role("LogPreferenceVector"):
        live = _live(c_vectors, _factor_index(name))
        if live is None or num_obs is None:
            continue
        if live.shape != (num_obs,):
            violations.append(
                f"{name} (LogPreferenceVector): live canonical dims {live.shape} "
                f"!= (num_obs,) = ({num_obs},)."
            )

    # The control axis, which _canonical_dims deliberately does not reduce away.
    # Checked here as well as in the single-factor path: a factorized model that
    # gained real actions would otherwise declare them and have nobody look.
    num_controls = spec.parameters.get("num_controls")
    if num_controls is not None and (not isinstance(num_controls, int) or num_controls < 1):
        violations.append(
            f"ModelParameters num_controls={num_controls!r} is not a positive integer."
        )
    elif isinstance(num_controls, int) and num_controls > 1:
        for index in range(declared_factors):
            live = _live(b_matrices, index)
            if live is None:
                continue
            if live.ndim != 3 or live.shape[2] != num_controls:
                violations.append(
                    f"factor {index}: ModelParameters declares "
                    f"num_controls={num_controls} but the live transition's "
                    f"canonical dims are {live.shape}. A declared action set "
                    f"that is not in B is an action set the agent cannot take."
                )

    violations.extend(_check_emission_factor_count(spec, _live(a_matrices, 0)))

    # A policy prior is sized by the action set, not by any factor, so it is
    # checked the same way here as in the single-factor path rather than being
    # left to the structural-form loop alone — a declared E whose SIZE nothing
    # compares is the gap this audit started from, one level along.
    for name in spec.variables_with_role("PolicyPrior"):
        declared = _canonical_dims(spec.variable_dims(name), strip_batch=False)
        if isinstance(num_controls, int) and num_controls > 0 and declared != (num_controls,):
            violations.append(
                f"{name} (PolicyPrior): declared dims {declared} but "
                f"ModelParameters num_controls={num_controls} requires "
                f"({num_controls},)."
            )
        if e_vector is None:
            violations.append(
                f"{name}: spec declares a PolicyPrior but no live E was "
                f"supplied, so the slot is unverified."
            )
        elif _canonical_array(e_vector).shape != declared:
            violations.append(
                f"{name} (PolicyPrior): declared dims {declared} but the live E "
                f"has canonical dims {_canonical_array(e_vector).shape}."
            )

    # Structural forms, against whichever live array the variable binds to. E is
    # a bare array rather than a per-factor list — one policy space regardless
    # of how the state space factorizes — so it is wrapped to index uniformly.
    slot_arrays = {
        "a": a_matrices,
        "b": b_matrices,
        "c": c_vectors,
        "d": d_vectors,
        "e": None if e_vector is None else [e_vector],
    }
    for name, (form, args) in spec.parameterization.items():
        role = spec.ontology.get(name)
        slot = MATRIX_ROLES.get(role or "")
        if slot is None:
            continue
        live_raw = slot_arrays[slot]
        index = _factor_index(name)
        if live_raw is None or index >= len(live_raw):
            continue
        violations.extend(_check_structural_form(name, form, args, np.asarray(live_raw[index])))

    return violations


def _check_structural_form(name: str, form: str, args: str, array: np.ndarray) -> list[str]:
    """Check one structural-form declaration against a live array.

    Structure, not learned values: an ``Identity`` emission asserts that the
    hidden state is observed directly, a ``Uniform`` prior asserts that the
    model starts uninformed, and a ``Learned`` matrix that turns out to be the
    identity asserts nothing at all.
    """
    violations: list[str] = []
    canonical = _canonical_array(array)
    if not np.all(np.isfinite(canonical)):
        return [f"{name}: declared {form}({args}) but the live array contains non-finite values."]

    if form == "Identity":
        size = int(args)
        if canonical.shape != (size, size):
            violations.append(
                f"{name}: declared Identity({size}) but the live array has "
                f"canonical shape {canonical.shape}, which cannot be a "
                f"{size}x{size} identity."
            )
            return violations
        deviation = float(np.max(np.abs(canonical - np.eye(size))))
        if deviation > _STRUCTURAL_TOL:
            violations.append(
                f"{name}: declared Identity({size}) in InitialParameterization "
                f"but the live matrix is NOT the identity "
                f"(max |{name} - I| = {deviation:.6f}). A learned table in the "
                f"RecognitionMatrix slot is the historical anti-informative "
                f"wiring: it yields a posterior over the PRECEDING symbol."
            )
    elif form == "Uniform":
        size = int(args)
        if canonical.size != size:
            violations.append(
                f"{name}: declared Uniform({size}) but the live array holds "
                f"{canonical.size} entries."
            )
            return violations
        deviation = float(np.max(np.abs(canonical.ravel() - 1.0 / size)))
        if deviation > _STRUCTURAL_TOL:
            violations.append(
                f"{name}: declared Uniform({size}) but the live array is not "
                f"uniform (max deviation from {1.0 / size:.6f} is "
                f"{deviation:.6f})."
            )
    elif form == "Zeros":
        size = int(args)
        if canonical.size != size:
            violations.append(
                f"{name}: declared Zeros({size}) but the live array holds {canonical.size} entries."
            )
            return violations
        deviation = float(np.max(np.abs(canonical)))
        if deviation > _STRUCTURAL_TOL:
            violations.append(
                f"{name}: declared Zeros({size}) but the live array has a "
                f"maximum absolute entry of {deviation:.6f}."
            )
    elif form == _LEARNED_FORM:
        # Non-vacuity: a matrix declared as learned from a corpus artifact must
        # carry content. The identity is the specific degenerate case that the
        # historical A/B swap produces in the transition slot.
        if canonical.ndim == 2 and canonical.shape[0] == canonical.shape[1]:
            size = canonical.shape[0]
            deviation = float(np.max(np.abs(canonical - np.eye(size))))
            if deviation <= _STRUCTURAL_TOL:
                violations.append(
                    f"{name}: declared {_LEARNED_FORM}({args}) but the live "
                    f"matrix IS the identity, so it carries no learned content. "
                    f"An identity in the TransitionMatrix slot forwards the "
                    f"belief unchanged — sequence statistics belong in B."
                )
        violations.extend(_check_learned_arguments(name, args, canonical))
    return violations


def _check_learned_arguments(name: str, args: str, canonical: np.ndarray) -> list[str]:
    """Check the declarations INSIDE ``Learned(...)`` against the live array.

    ``normalization=`` and ``estimator=`` appear on every ``Learned`` form in
    every shipped spec, were parsed into :attr:`GnnSpec.parameterization`, and
    were read by nothing. Both are structural claims about the array rather than
    claims about a learned value, so both are checkable here:

    * ``normalization=column-stochastic-*`` says axis 0 sums to one for every
      index of the remaining axes. That is what makes a transposed emission a
      violation even when it is square — the case pymdp's own ``_validate``
      cannot see, because a renormalized transpose is column-stochastic too, but
      about the wrong axis.
    * ``normalization=sums-to-one-*`` says the whole array sums to one, which is
      what a prior over one factor claims and a per-column matrix does not.
    * an unqualified Laplace ``estimator`` says every entry is strictly
      positive. An unsmoothed matrix in a slot declared smoothed assigns
      probability zero to an unseen event, which is an infinite loss the first
      time one occurs — exactly the defect the smoothing exists to prevent.

    Args:
        name: Variable being checked, for the message.
        args: The ``Learned`` argument list.
        canonical: The live array, already reduced by :func:`_canonical_array`.

    Returns:
        Every violation found.
    """
    entries = parse_learned_arguments(args)
    violations: list[str] = []

    normalization = entries.get("normalization")
    if normalization is not None and canonical.size:
        if normalization.startswith(_COLUMN_STOCHASTIC_PREFIX):
            deviation = float(np.max(np.abs(canonical.sum(axis=0) - 1.0)))
            if deviation > _STRUCTURAL_TOL:
                violations.append(
                    f"{name}: declared normalization={normalization} but the "
                    f"live array's axis-0 sums deviate from 1 by up to "
                    f"{deviation:.6f}. A matrix normalized over the wrong axis "
                    f"is a different model that pymdp accepts."
                )
        elif normalization.startswith(_SUMS_TO_ONE_PREFIX):
            deviation = abs(float(canonical.sum()) - 1.0)
            if deviation > _STRUCTURAL_TOL:
                violations.append(
                    f"{name}: declared normalization={normalization} but the "
                    f"live array sums to {float(canonical.sum()):.6f}."
                )
        else:
            # Refused rather than skipped, for the reason this whole audit
            # exists: silently ignoring a declaration reports it verified.
            violations.append(
                f"{name}: declared normalization={normalization}, which this "
                f"checker does not know. Teach it the new axis claim rather "
                f"than leaving the slot unverified."
            )

    estimator = entries.get("estimator")
    if estimator in _STRICTLY_POSITIVE_ESTIMATORS and canonical.size:
        minimum = float(canonical.min())
        if minimum <= 0.0:
            violations.append(
                f"{name}: declared estimator={estimator} but the live array has "
                f"a minimum entry of {minimum:.6g}. Laplace smoothing exists so "
                f"that no event has probability zero; a zero here is an "
                f"unsmoothed table under a smoothed declaration."
            )
    return violations
