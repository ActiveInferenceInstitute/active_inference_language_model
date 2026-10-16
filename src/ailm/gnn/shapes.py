"""Canonical shape description of a live pymdp agent.

The bridge between a built agent and a declaration: it turns whatever pymdp is
holding into plain dimensions, without judging them. The judging is
:mod:`ailm.gnn.conformance`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ailm.gnn.types import LiveConformanceInputs

# ── SHAPE NORMALIZATION ──────────────────────────────────────────────────────


def _canonical_dims(dims: Sequence[int], *, strip_batch: bool) -> tuple[int, ...]:
    """Reduce a shape to its comparable core.

    Two normalizations, both symmetric and both deliberate:

    * ``strip_batch`` drops a leading axis of size 1. pymdp carries a batch axis
      that a GNN spec has no notion of: ``A`` is ``(1, n_obs, n_states)`` in the
      agent and ``A[n_obs, n_states]`` in the spec.
    * Trailing axes of size 1 are dropped from both sides, so the spec's
      ``B[39,39,1]`` (one control) and the agent's ``(1, 39, 39, 1)`` compare
      equal, and the spec's column-vector ``D[39,1]`` matches the agent's
      ``(1, 39)``.

    Non-singleton action axes survive this reduction. Conformance separately
    compares controlled transitions, policy dimensions and supplied live control
    counts with ``num_controls``; shape reduction alone does not establish them.

    Args:
        dims: A shape.
        strip_batch: True for a shape read off a live pymdp array.

    Returns:
        The reduced shape. Never empty for a non-empty input.
    """
    reduced = tuple(int(d) for d in dims)
    if strip_batch and len(reduced) > 1 and reduced[0] == 1:
        reduced = reduced[1:]
    while len(reduced) > 1 and reduced[-1] == 1:
        reduced = reduced[:-1]
    return reduced


def _canonical_array(array: Any) -> np.ndarray:
    """Return ``array`` as float64 numpy, reshaped to its canonical dims.

    Reshaping to :func:`_canonical_dims` only ever removes size-1 axes, so the
    element count is preserved exactly and no data is reordered.
    """
    arr = np.asarray(array, dtype=np.float64)
    return arr.reshape(_canonical_dims(arr.shape, strip_batch=True))


def describe_shapes(agent: Any) -> LiveConformanceInputs:
    """Extract the conformance-relevant structure from a LIVE pymdp Agent.

    This is what makes conformance a guard rather than a comment: the checks run
    against the object pymdp actually constructed, not against a tuple a test
    author typed out by hand.

    The returned record goes straight into :func:`check_conformance` /
    :func:`assert_conforms` as the ``live`` argument — no splat::

        assert_conforms(spec, describe_shapes(agent))

    Alongside ``a_shape`` / ``b_shape`` / ``c_shape`` / ``d_shape`` it carries the
    arrays themselves, because the structural-form checks (``Identity(n)``,
    ``Uniform(n)``, ``Zeros(n)``, ``Learned(...)``) cannot be decided from a
    shape — and for the letter level the correct and the historically swapped
    wiring have identical shapes.

    ``E`` and ``gamma`` are read too, and were not. That omission is the defect
    this function's audit was built around: L1g declares ``E=Uniform(30)`` and
    ``gamma: 16.0``, and until both travelled with the description they were
    declarations the checker could not see, however loudly the spec made them.
    Neither is a per-factor list — there is one policy space and one precision
    however many hidden factors a model has.

    Args:
        agent: A ``pymdp.agent.Agent``. Read through its public ``A``/``B``/
            ``C``/``D``/``E``/``gamma`` attributes only; this module never
            imports pymdp or jax.

    Returns:
        ``a_shape``, ``b_shape``, ``c_shape``, ``d_shape``, ``e_shape`` (tuples,
        or None when the agent has no such slot), ``a_matrix``, ``b_matrix``,
        ``c_vector``, ``d_vector``, ``e_vector`` (numpy arrays or None),
        ``gamma``, plus ``n_modalities`` and ``n_factors``.

    Raises:
        ValueError: If the agent has no ``A``, ``B`` or ``D``, or if any of them
            is empty. A model missing one of those is not describable, and
            silently reporting None would let conformance pass vacuously.
    """
    slots: dict[str, Any] = {}
    for attribute in ("A", "B", "C", "D"):
        value = getattr(agent, attribute, None)
        slots[attribute] = value

    for attribute in ("A", "B", "D"):
        value = slots[attribute]
        if value is None or len(value) == 0:
            raise ValueError(
                f"agent has no {attribute} to describe (got {value!r}). A model "
                f"without {attribute} cannot be checked for conformance, and "
                f"reporting None here would make the check vacuous."
            )

    def _first(attribute: str) -> Any:
        value = slots[attribute]
        if value is None or len(value) == 0:
            return None
        return value[0]

    described: dict[str, Any] = {
        "n_modalities": len(slots["A"]),
        "n_factors": len(slots["B"]),
    }
    for attribute, shape_key, array_key in (
        ("A", "a_shape", "a_matrix"),
        ("B", "b_shape", "b_matrix"),
        ("C", "c_shape", "c_vector"),
        ("D", "d_shape", "d_vector"),
    ):
        raw = _first(attribute)
        if raw is None:
            described[shape_key] = None
            described[array_key] = None
            continue
        array = np.asarray(raw, dtype=np.float64)
        described[shape_key] = tuple(int(d) for d in array.shape)
        described[array_key] = array

    # Multi-factor models carry one B and one D per factor, and one A per
    # modality. Reporting only slot 0 would leave every further factor
    # unverified while the check still returned "conformant" — an unverified
    # slot must be visible, so the full lists travel alongside the singulars.
    for attribute, list_key in (
        ("A", "a_matrices"),
        ("B", "b_matrices"),
        ("C", "c_vectors"),
        ("D", "d_vectors"),
    ):
        value = slots[attribute]
        described[list_key] = (
            None if value is None else [np.asarray(item, dtype=np.float64) for item in value]
        )

    # E is a bare array rather than a per-factor list: it is a distribution over
    # POLICIES, and a model has one policy space however it factorizes its
    # states. pymdp builds a uniform E over one trivial policy when the caller
    # supplies none, so a spec declaring no PolicyPrior simply makes no claim
    # about the array that is nonetheless present.
    policy_prior = getattr(agent, "E", None)
    if policy_prior is None:
        described["e_shape"] = None
        described["e_vector"] = None
    else:
        prior_array = np.asarray(policy_prior, dtype=np.float64)
        described["e_shape"] = tuple(int(d) for d in prior_array.shape)
        described["e_vector"] = prior_array

    # The inference configuration, which the matrices cannot express. A model can
    # carry a correct C, a correct E and a correct controllable B and still not
    # minimize expected free energy over them, because whether the preference and
    # epistemic terms enter the score at all is a flag on the agent rather than a
    # value in any slot. Reading only the arrays makes that difference invisible:
    # every shape and every entry checks out, the conformance report is clean, and
    # the claim that generation selects a policy by minimizing EFE is false.
    #
    # ``policy_len`` and the policy count travel for a related reason. E is a
    # distribution over policies, so ``Uniform(30)`` is coherent only while a
    # policy is one action deep; at depth two the space is 900 and a 30-entry E is
    # a different claim wearing the same shape.
    for attribute, key in (
        ("use_utility", "use_utility"),
        ("use_states_info_gain", "use_states_info_gain"),
        ("use_param_info_gain", "use_param_info_gain"),
        ("action_selection", "action_selection"),
        ("policy_len", "policy_len"),
    ):
        value = getattr(agent, attribute, None)
        described[key] = None if value is None else value
    policies = getattr(agent, "policies", None)
    try:
        described["n_policies"] = None if policies is None else int(len(policies))
    except TypeError:
        described["n_policies"] = None
    controls = getattr(agent, "num_controls", None)
    described["num_controls"] = (
        None if controls is None else tuple(int(c) for c in np.atleast_1d(controls))
    )

    # gamma is broadcast to (batch_size,) by pymdp, so it arrives as an array
    # even though the spec declares a scalar.
    precision = getattr(agent, "gamma", None)
    described["gamma"] = None if precision is None else np.asarray(precision, dtype=np.float64)

    unknown = set(described) - set(LiveConformanceInputs.__dataclass_fields__)
    if unknown:  # pragma: no cover - a failed field inventory is a bug, not a skip
        raise AssertionError(f"describe_shapes produced fields no record declares: {unknown}")
    return LiveConformanceInputs(**described)


def live_inputs(values: dict[str, Any]) -> LiveConformanceInputs:
    """Convert a plain mapping of live values into the typed record.

    The inverse of :meth:`LiveConformanceInputs.as_dict`. Keyed access cannot
    drift from the record: a key no field declares raises ``TypeError`` here,
    at the conversion boundary, instead of surfacing as a silently dropped
    slot inside the checker.
    """
    unknown = set(values) - set(LiveConformanceInputs.__dataclass_fields__)
    if unknown:
        raise TypeError(
            f"live_inputs() got key(s) {sorted(unknown)} that no "
            f"LiveConformanceInputs field declares"
        )
    return LiveConformanceInputs(**values)
