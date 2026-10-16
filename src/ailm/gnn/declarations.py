"""What a GNN spec says about itself, checked without building anything.

Every check here reads the declaration alone. A spec can be internally
incoherent — a vector whose declared length contradicts its factor's state
count, a connection naming a variable the spec never declares, an inference
block selecting an algorithm the model's own structure rules out — and none of
that needs a live agent to detect. Catching it here means the conformance run
against a built model reports disagreements between spec and model, rather than
a spec that never made sense.

The seam is the point. :mod:`ailm.gnn.live` checks a built object against the
declaration; this module checks the declaration against itself, and the two
were one 1,260-line file in which the distinction was invisible.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ailm.gnn.shapes import _canonical_dims
from ailm.gnn.types import (
    _FLOAT_ROLES,
    _INT_ROLES,
    _TIME_ASSIGNMENT_RE,
    GnnSpec,
)

#: Trailing ``_f<N>`` on a variable name binds it to hidden factor ``N``. A
#: model with several factors needs that binding stated rather than inferred
#: from declaration order, which nothing would notice reordering.
_FACTOR_SUFFIX_RE = re.compile(r"_f(?P<index>\d+)$")


def _factor_index(name: str) -> int:
    """Hidden-factor index a variable name binds to; 0 when unsuffixed."""
    match = _FACTOR_SUFFIX_RE.search(name)
    return int(match["index"]) if match else 0


def _declared_factor_count(spec: GnnSpec) -> int:
    """``num_factors`` from ``ModelParameters``, defaulting to 1."""
    declared = spec.parameters.get("num_factors")
    return declared if isinstance(declared, int) and declared > 0 else 1


def _factor_sizes(spec: GnnSpec, n_factors: int) -> tuple[list[int | None], list[str]]:
    """Read ``num_hidden_states_f<N>`` for each factor.

    Returns:
        ``(sizes, violations)``. A size is None when the key is missing or not a
        positive integer, so downstream checks can skip it rather than compare
        against a guess.
    """
    sizes: list[int | None] = []
    problems: list[str] = []
    for index in range(n_factors):
        key = f"num_hidden_states_f{index}"
        value = spec.parameters.get(key)
        if value is None:
            problems.append(
                f"ModelParameters is missing {key!r}. A model declaring "
                f"num_factors={n_factors} must size every factor; without it "
                f"factor {index} is unpinned."
            )
            sizes.append(None)
        elif not isinstance(value, int) or value < 1:
            problems.append(f"ModelParameters {key}={value!r} is not a positive integer.")
            sizes.append(None)
        else:
            sizes.append(value)
    return sizes, problems


def _declared_dims(spec: GnnSpec, name: str) -> tuple[int, ...]:
    """Canonical declared dims of ``name``, with no batch axis to strip."""
    return _canonical_dims(spec.variable_dims(name), strip_batch=False)


def _check_vector_declarations(spec: GnnSpec) -> list[str]:
    """Size every declared belief, observation and policy vector.

    The defect: ``StateSpaceBlock`` declares ``s_t``, ``s_prime``, ``o_t`` and
    ``pi`` alongside the matrices, and only the matrices were ever compared to
    anything. Resizing a level in ``ModelParameters`` and in ``A``/``B``/``D``
    while leaving ``s_t`` behind produced a spec that passed conformance while
    contradicting itself about the size of its own state space.

    These have no live counterpart — pymdp returns beliefs from inference rather
    than storing them on the ``Agent`` — so the comparison is against the sizes
    the same spec declares in ``ModelParameters``.

    Returns:
        Every violation found.
    """
    violations: list[str] = []
    factors = _declared_factor_count(spec)
    if factors > 1:
        sizes, _problems = _factor_sizes(spec, factors)
        state_sizes = sorted(size for size in sizes if size is not None)
    else:
        scalar = spec.parameters.get("num_hidden_states")
        state_sizes = [scalar] if isinstance(scalar, int) and scalar > 0 else []

    for role in ("HiddenState", "NextHiddenState"):
        pairs: list[tuple[str, int]] = []
        for name in spec.variables_with_role(role):
            dims = _declared_dims(spec, name)
            if len(dims) != 1:
                violations.append(
                    f"{name} ({role}): declared dims {dims}. A belief over one "
                    f"hidden factor is a vector; anything else is a joint that "
                    f"pymdp never forms and no ModelParameters entry can size."
                )
                continue
            pairs.append((name, dims[0]))
        if not state_sizes or not pairs:
            continue
        if factors > 1:
            # A multi-factor model names its beliefs for what they mean (s_word,
            # s_pos) rather than with an _f<N> suffix, so nothing binds one to a
            # particular factor and the per-name check of the single-factor case
            # has no analogue. The multiset still catches the defect that
            # matters: a factor resized in ModelParameters and nowhere else.
            declared = sorted(size for _name, size in pairs)
            if declared != state_sizes:
                violations.append(
                    f"the {role} beliefs are declared with sizes {declared} but "
                    f"the declared factor sizes are {state_sizes}. Every hidden "
                    f"factor needs exactly one belief of its own size."
                )
        else:
            for name, size in pairs:
                if size != state_sizes[0]:
                    violations.append(
                        f"{name} ({role}): declared size {size} but "
                        f"ModelParameters num_hidden_states={state_sizes[0]}. "
                        f"The spec contradicts itself about its state space."
                    )

    num_obs = spec.parameters.get("num_obs")
    modalities = spec.parameters.get("num_modalities")
    for name in spec.variables_with_role("Observation"):
        if not isinstance(num_obs, int) or num_obs < 1:
            continue
        if modalities is not None and modalities != 1:
            # With several modalities an observation variable would have to name
            # which one it indexes, and nothing in the notation binds it. Every
            # AILM level is single-modality; a multi-modality level must add that
            # binding before this can size its observations.
            continue
        dims = _declared_dims(spec, name)
        if dims != (num_obs,):
            violations.append(
                f"{name} (Observation): declared dims {dims} but ModelParameters "
                f"num_obs={num_obs} requires ({num_obs},)."
            )

    num_controls = spec.parameters.get("num_controls")
    for name in spec.variables_with_role("PolicyVector"):
        if not isinstance(num_controls, int) or num_controls < 1:
            violations.append(
                f"{name} is annotated PolicyVector but ModelParameters declares "
                f"no positive num_controls, so the size of the policy space it "
                f"ranges over is unpinned."
            )
            continue
        dims = _declared_dims(spec, name)
        if dims != (num_controls,):
            violations.append(
                f"{name} (PolicyVector): declared dims {dims} but ModelParameters "
                f"num_controls={num_controls} requires ({num_controls},)."
            )
    return violations


def _check_dtype_declarations(spec: GnnSpec) -> list[str]:
    """Check each variable's ``type=`` against what its role holds.

    The defect: ``type=`` was parsed into :attr:`GnnSpec.variables` and read by
    nothing, so ``o_t[30,1,type=float]`` — an observation that is an index, not
    a probability — would have parsed and conformed. Spec-internal on purpose:
    every pymdp array is float32 whatever the slot means, so the live model
    cannot contradict a declared dtype and a check against it would be theatre.
    """
    violations: list[str] = []
    for name, (_dims, dtype) in spec.variables.items():
        role = spec.ontology.get(name)
        if role in _FLOAT_ROLES and dtype != "float":
            violations.append(
                f"{name} ({role}) is declared type={dtype}. A {role} holds a "
                f"probability or a log-probability, which is float."
            )
        elif role in _INT_ROLES and dtype != "int":
            violations.append(
                f"{name} ({role}) is declared type={dtype}. A {role} is an index, which is int."
            )
    return violations


def _check_time_index(spec: GnnSpec) -> list[str]:
    """Check that ``Time=<variable>`` names a variable annotated ``Time``.

    The defect: the ``Time`` block was retained verbatim and never read, so
    ``Time=t`` could name a variable the spec never declared — or one annotated
    something else — and nothing would notice. ``Dynamic`` and ``Discrete`` stay
    unchecked and cannot be otherwise: a pymdp agent is discrete-time by
    construction, so comparing against it would compare a constant to itself.
    """
    body = spec.sections.get("Time")
    if body is None:
        return []
    named = _TIME_ASSIGNMENT_RE.findall(body)
    if len(named) != 1:
        return [
            f"the Time section carries {len(named)} 'Time=<variable>' line(s) "
            f"{named}; exactly one names the model's time index."
        ]
    (index,) = named
    if index not in spec.variables:
        return [
            f"Time={index} names no variable declared in StateSpaceBlock, so the "
            f"time index is a name bound to nothing."
        ]
    role = spec.ontology.get(index)
    if role != "Time":
        return [f"Time={index} but {index} is annotated {role!r} rather than Time."]
    return []


def _check_connection_graph(spec: GnnSpec) -> list[str]:
    """Check the ``Connections`` graph against the roles it wires together.

    The defect: endpoints were required to be declared variables and the graph
    was then read by nothing, so a spec could annotate ``D=Prior`` and wire it
    nowhere, or declare thirty actions with no policy node reaching the
    transition, and conformance would report the model clean.

    Three claims the graph makes and this reads back:

    * every declared variable is wired to something. The ``Time`` index is
      exempt — it indexes the graph rather than sitting in it, and no shipped
      spec places it in a connection.
    * every ``Prior`` is directed into a ``HiddenState``. A prior that seeds
      nothing is a prior the graph does not use.
    * a ``PolicyVector`` directed into the ``TransitionMatrix`` and a declared
      ``num_controls`` greater than one imply each other. A policy choosing
      among one action chooses nothing, and an action set no policy reaches is
      an action set the agent cannot take.
    """
    if not spec.connections:
        return []
    violations: list[str] = []

    connected: set[str] = set()
    for connection in spec.connections:
        connected.update(connection.source)
        connected.update(connection.target)
    orphans = [
        name
        for name in spec.variables
        if name not in connected and spec.ontology.get(name) != "Time"
    ]
    if orphans:
        violations.append(
            f"variable(s) {orphans} are declared and annotated but appear in no "
            f"Connections line, so the graph says nothing about how they enter "
            f"the model."
        )

    hidden = set(spec.variables_with_role("HiddenState"))
    for name in spec.variables_with_role("Prior"):
        targets = {
            target
            for connection in spec.connections
            if connection.directed and name in connection.source
            for target in connection.target
        }
        if not targets or not targets <= hidden:
            violations.append(
                f"{name} (Prior) is directed into {sorted(targets)}, not into a "
                f"HiddenState. A prior seeds the hidden state or it seeds nothing."
            )

    transitions = set(spec.variables_with_role("TransitionMatrix"))
    drivers = {
        name
        for name in spec.variables_with_role("PolicyVector")
        for connection in spec.connections
        if connection.directed
        and name in connection.source
        and transitions & set(connection.target)
    }
    num_controls = spec.parameters.get("num_controls")
    controlled = isinstance(num_controls, int) and num_controls > 1
    if drivers and not controlled:
        violations.append(
            f"{sorted(drivers)} is annotated PolicyVector and directed into the "
            f"TransitionMatrix, but ModelParameters declares "
            f"num_controls={num_controls!r}. A policy over one action is not a "
            f"choice."
        )
    if controlled and not drivers:
        violations.append(
            f"ModelParameters declares num_controls={num_controls} but no "
            f"PolicyVector is directed into the TransitionMatrix, so the graph "
            f"carries no path by which a chosen action reaches the dynamics."
        )
    return violations


#: Combination rules a spec may declare for an inter-level message. Declared
#: rather than inferred because decision 47 established that the rule, not the
#: insertion site, determines whether a message can help: under ``linear`` the
#: two sites behave differently and a uniform message moves posteriors; under
#: ``geometric`` the sites are provably one computation and a uniform message is
#: inert. A model whose spec is silent about the rule leaves a reader unable to
#: tell which of those two regimes its numbers came from.
COMBINATION_RULES: tuple[str, ...] = ("linear", "geometric")


def _check_combination_rule(spec: GnnSpec) -> list[str]:
    """A declared combination rule must be one the project implements.

    Optional: a level with no inter-level message makes no statement, and most
    do not. But a spec that declares a rule this project cannot run is worse than
    one that declares nothing, because it reads as a checked claim.

    The rule is deliberately NOT inferred from the model's matrices. It is not
    visible in them — the same emission, transition and prior serve both rules,
    and only the code combining them differs — which is exactly why it belongs in
    the spec rather than in a conformance check against a live agent.
    """
    declared = spec.parameters.get("combination_rule")
    if declared is None:
        return []
    if not isinstance(declared, str) or declared not in COMBINATION_RULES:
        return [
            f"ModelParameters declares combination_rule={declared!r}, which is not one of "
            f"{list(COMBINATION_RULES)}. A rule the project cannot run reads as a checked "
            f"claim and is not one."
        ]
    return []


def _check_inference_configuration(
    spec: GnnSpec,
    *,
    use_utility: Any = None,
    use_states_info_gain: Any = None,
    policy_len: Any = None,
    n_policies: Any = None,
    num_controls: Any = None,
    e_vector: Any = None,
) -> list[str]:
    """Checks that need the agent's inference flags, not its matrices.

    Every other check here reads a slot. These read the configuration that
    decides whether the slots are used, which is a different kind of claim and
    was previously unverifiable: an agent with a correct non-flat ``C``, a
    correct ``E`` and a genuinely controllable ``B`` still does not minimize
    expected free energy over them if ``use_utility`` is off. Nothing about the
    arrays changes, so the report stays clean while the model does something
    else entirely.

    The checks apply only to a spec that declares itself controlled
    (``num_controls > 1``). An uncontrolled level runs with a flat ``C`` and one
    action by design, and demanding a preference term of it would fail every
    comprehension level in the system for conforming exactly as intended.

    Args:
        spec: The parsed spec.
        use_utility: The agent's preference-term flag.
        use_states_info_gain: The agent's epistemic-term flag.
        policy_len: How many actions deep a policy is.
        n_policies: How many policies the agent enumerated.
        num_controls: The agent's per-factor control counts.
        e_vector: The policy prior, used to check it is sized over the policy
            space the agent actually built.

    Returns:
        Every violation found. Absent values are skipped rather than assumed:
        a description that omits a flag makes no claim about it, which is the
        same convention the shape checks use.
    """
    declared = spec.parameters.get("num_controls")
    if not (isinstance(declared, int) and declared > 1):
        return []

    violations: list[str] = []
    if use_utility is False:
        violations.append(
            "ModelParameters declares a controlled model, but the agent has "
            "use_utility=False, so the preference vector never enters the "
            "policy score. Every slot conforms and the model still does not "
            "minimize expected free energy over C."
        )
    if use_states_info_gain is False:
        violations.append(
            "ModelParameters declares a controlled model, but the agent has "
            "use_states_info_gain=False, so the epistemic term is absent and "
            "policy selection is utility maximization rather than expected "
            "free energy minimization."
        )
    if num_controls is not None:
        controls = tuple(int(c) for c in np.atleast_1d(num_controls))
        if controls and int(controls[0]) != declared:
            violations.append(
                f"ModelParameters declares num_controls={declared} but the agent "
                f"was built with {controls[0]}. The spec's action set and the "
                f"agent's are different sizes, which the B shape does not catch "
                f"once a trailing axis is reduced."
            )
    if e_vector is not None and n_policies is not None:
        size = int(np.asarray(e_vector).size)
        if size != int(n_policies):
            violations.append(
                f"the policy prior has {size} entries but the agent enumerated "
                f"{n_policies} policies. E is a distribution over the policy "
                f"space, so a mismatch means the prior is over something else "
                f"— at policy_len={policy_len!r} the space is not the action set."
            )
    return violations


def _check_declaration_coherence(spec: GnnSpec) -> list[str]:
    """Every check that needs the spec alone, in one place.

    Grouped so both the single-factor and the multi-factor paths run all of
    them: a check reachable from only one path is a check a model escapes by
    gaining a factor.
    """
    violations: list[str] = []
    violations.extend(_check_vector_declarations(spec))
    violations.extend(_check_dtype_declarations(spec))
    violations.extend(_check_time_index(spec))
    violations.extend(_check_connection_graph(spec))
    violations.extend(_check_combination_rule(spec))
    return violations
