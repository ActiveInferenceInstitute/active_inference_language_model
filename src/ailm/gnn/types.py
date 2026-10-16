"""Errors, section names, and the parsed-spec dataclasses.

The vocabulary every other module in this package speaks. Kept separate so
parsing, conformance and the registry can each import the types without
importing one another."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import Any

#: The canonical GNN section names, in canonical order. A section outside this
#: set is an error rather than an unparsed extension.
GNN_SECTIONS: tuple[str, ...] = (
    "GNNSection",
    "GNNVersionAndFlags",
    "ModelName",
    "ModelAnnotation",
    "StateSpaceBlock",
    "Connections",
    "InitialParameterization",
    "Equations",
    "Time",
    "ActInfOntologyAnnotation",
    "ModelParameters",
    "Footer",
    "Signature",
)

#: Sections without which a spec pins nothing and must not parse.
_REQUIRED_SECTIONS: tuple[str, ...] = (
    "ModelName",
    "StateSpaceBlock",
    "ActInfOntologyAnnotation",
)

#: Ontology role -> the ``<slot>_shape`` / ``<slot>_matrix`` keyword prefix that
#: :func:`check_conformance` binds it to. ``PolicyPrior`` is here because it was
#: the gap that motivated the audit above: L1g declares ``E=Uniform(30)`` and
#: nothing compared it to the live agent, so the whole argument for annotating
#: the Brown bigram ``PolicyPrior`` rather than ``TransitionMatrix`` — that the
#: bigram is installed per generation step while the construction-time ``E`` is
#: uniform — rested on prose. Every other ontology term either carries a size
#: obligation via :data:`_VECTOR_ROLE_SIZES` or is descriptive.
MATRIX_ROLES: dict[str, str] = {
    "RecognitionMatrix": "a",
    "TransitionMatrix": "b",
    "LogPreferenceVector": "c",
    "Prior": "d",
    "PolicyPrior": "e",
}

#: Roles a conformant AILM spec must annotate exactly one variable with.
_MANDATORY_ROLES: tuple[str, ...] = ("RecognitionMatrix", "TransitionMatrix", "Prior")

#: Ontology role -> which ``ModelParameters`` size its variable must equal.
#: pymdp holds beliefs in the inference state it returns, not on the ``Agent``,
#: so no live array corresponds to these; what IS checkable is that the declared
#: size agrees with the rest of the spec. Before this, ``s_t[30,1]`` could have
#: read ``s_t[7,1]`` and nothing anywhere would have noticed.
_VECTOR_ROLE_SIZES: dict[str, str] = {
    "HiddenState": "states",
    "NextHiddenState": "states",
    "Observation": "obs",
    "PolicyVector": "controls",
}

#: Roles whose variable must be declared ``type=float`` — it holds a probability
#: or a log-probability — and ``type=int`` — it holds an index. Spec-internal:
#: every pymdp array is float32 whatever the slot means, so the live model
#: cannot contradict a dtype. A dtype nothing reads is a dtype free to drift.
_FLOAT_ROLES: frozenset[str] = frozenset(
    {*MATRIX_ROLES, "HiddenState", "NextHiddenState", "PolicyVector"}
)
_INT_ROLES: frozenset[str] = frozenset({"Observation", "Time"})

#: Structural forms allowed in ``InitialParameterization``.
_SIZED_FORMS: frozenset[str] = frozenset({"Identity", "Uniform", "Zeros"})
_LEARNED_FORM: str = "Learned"

#: ``Learned(normalization=...)`` prefixes and the axis claim each one makes.
#: ``column-stochastic-*`` asserts axis 0 sums to one for every index of the
#: remaining axes; ``sums-to-one-*`` asserts the whole array sums to one. An
#: unrecognized prefix is a violation rather than a skip, for the reason the
#: whole audit exists: silently ignoring a declaration reports it verified.
_COLUMN_STOCHASTIC_PREFIX: str = "column-stochastic"
_SUMS_TO_ONE_PREFIX: str = "sums-to-one"

#: Estimators asserting that EVERY entry is strictly positive. Exact membership,
#: not a prefix: ``add-k-Laplace-per-type-block`` smooths within a block and
#: leaves the off-block entries structurally zero, so a prefix rule would report
#: L5's block-diagonal transition as unsmoothed. A qualified estimator names a
#: partition the spec does not enumerate, and inventing that partition here
#: would be a guess dressed as a check.
_STRICTLY_POSITIVE_ESTIMATORS: frozenset[str] = frozenset({"add-1-Laplace", "add-k-Laplace"})

#: float32 is JAX's default, so exact structural anchors land within ~2e-7.
_STRUCTURAL_TOL: float = 1e-5

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_VARIABLE_RE = re.compile(rf"^(?P<name>{_IDENT})\[(?P<body>[^\[\]]*)\]$")
_PARAM_FORM_RE = re.compile(rf"^(?P<name>{_IDENT})\s*=\s*(?P<form>{_IDENT})\s*\((?P<args>.*)\)$")
_ONTOLOGY_RE = re.compile(rf"^(?P<name>{_IDENT})\s*=\s*(?P<term>{_IDENT})$")
_MODEL_PARAM_RE = re.compile(rf"^(?P<key>{_IDENT})\s*[:=]\s*(?P<value>.+)$")

#: ``Time=<variable>`` inside the ``Time`` block names the model's time index.
_TIME_ASSIGNMENT_RE = re.compile(rf"^\s*Time\s*=\s*(?P<name>{_IDENT})\s*$", re.MULTILINE)


class GnnSpecError(ValueError):
    """A GNN spec is malformed. The message names the offending line."""


class GnnConformanceError(AssertionError):
    """A live model disagrees with its spec. Lists every violation at once."""


@dataclass(frozen=True)
class GnnConnection:
    """One line of the ``Connections`` block.

    Attributes:
        source: Variable names on the left of the operator.
        target: Variable names on the right.
        directed: True for ``>`` (directed), False for ``-`` (undirected).
        raw: The original line, for error messages.
    """

    source: tuple[str, ...]
    target: tuple[str, ...]
    directed: bool
    raw: str


@dataclass(frozen=True)
class GnnSpec:
    """A parsed GNN spec, reduced to what conformance needs.

    Attributes:
        model_name: Contents of ``ModelName``.
        variables: ``name -> (dims, dtype)`` from ``StateSpaceBlock``. ``dims``
            is a tuple of positive ints exactly as declared, with no batch axis.
        connections: Parsed ``Connections`` lines, in file order.
        ontology: ``name -> ontology term`` from ``ActInfOntologyAnnotation``.
            In bijection with ``variables``.
        parameters: ``key -> value`` from ``ModelParameters``, ints/floats
            coerced.
        parameterization: ``name -> (form, args)`` from
            ``InitialParameterization``, e.g. ``("Identity", "30")``.
        sections: The raw section text, keyed by section name. Nothing this
            module chooses not to interpret is thrown away.
        origin: Path or label the spec was read from.
    """

    model_name: str
    variables: dict[str, tuple[tuple[int, ...], str]]
    connections: list[GnnConnection]
    ontology: dict[str, str]
    parameters: dict[str, Any]
    parameterization: dict[str, tuple[str, str]] = field(default_factory=dict)
    sections: dict[str, str] = field(default_factory=dict)
    origin: str = "<string>"

    def variable_dims(self, name: str) -> tuple[int, ...]:
        """Declared dimensions of ``name``.

        Raises:
            KeyError: If ``name`` is not declared.
        """
        return self.variables[name][0]

    def role_of(self, name: str) -> str | None:
        """Ontology term annotated on ``name``, or None if unannotated."""
        return self.ontology.get(name)

    def variables_with_role(self, role: str) -> list[str]:
        """Every variable annotated with ``role``, in declaration order."""
        return [name for name in self.variables if self.ontology.get(name) == role]


@dataclass(frozen=True)
class LiveConformanceInputs:
    """The live-model side of a conformance check, as one typed record.

    :func:`ailm.gnn.shapes.describe_shapes` returns exactly this object, and
    :func:`ailm.gnn.conformance.check_conformance` consumes it, so the
    twenty-five-parameter positional signature — which every caller splatted
    from ``describe_shapes`` — is retired. The record keeps the old call
    surface's two loading behaviors, because callers rely on both:

    * every field defaults to ``None``, so a partial mapping (an absent slot
      means "unverified", not an error) splats cleanly;
    * a field name that is not part of the record is a ``TypeError`` rather
      than a silently dropped value — a typo'd or renamed slot must fail
      loudly, exactly as an unexpected keyword did before.

    ``as_dict`` supports the inverse direction: an existing ``dict[str, Any]``
    of live values can be converted with ``LiveConformanceInputs(**mapping)``,
    and a record back to the mapping form with ``as_dict``.

    Attributes:
        a_shape: Live emission shape, e.g. ``(1, 30, 39)``; None = slot withheld.
        b_shape: Live transition shape, e.g. ``(1, 39, 39, 1)``.
        d_shape: Live prior shape, e.g. ``(1, 39)``.
        c_shape: Live preference shape, e.g. ``(1, 30)``.
        e_shape: Live policy-prior shape, e.g. ``(1, 30)``.
        a_matrix: Live emission array, needed for structural-form checks.
        b_matrix: Live transition array.
        c_vector: Live preference array.
        d_vector: Live prior array.
        e_vector: Live policy prior.
        gamma: Live policy precision.
        n_modalities: ``len(agent.A)``.
        n_factors: ``len(agent.B)``.
        a_matrices: One emission array per modality.
        b_matrices: One transition array per factor.
        c_vectors: One preference array per modality.
        d_vectors: One prior array per factor.
        use_utility: Whether the agent scores expected utility.
        use_states_info_gain: Whether the agent scores epistemic value.
        use_param_info_gain: Whether the agent scores parameter information gain.
        action_selection: The agent's action-selection regime.
        policy_len: Policy search depth in actions.
        n_policies: Number of policies the agent enumerated.
        num_controls: Number of actions per control factor.
    """

    a_shape: tuple[int, ...] | None = None
    b_shape: tuple[int, ...] | None = None
    d_shape: tuple[int, ...] | None = None
    c_shape: tuple[int, ...] | None = None
    e_shape: tuple[int, ...] | None = None
    a_matrix: Any = None
    b_matrix: Any = None
    c_vector: Any = None
    d_vector: Any = None
    e_vector: Any = None
    gamma: Any = None
    n_modalities: int | None = None
    n_factors: int | None = None
    a_matrices: Sequence[Any] | None = None
    b_matrices: Sequence[Any] | None = None
    c_vectors: Sequence[Any] | None = None
    d_vectors: Sequence[Any] | None = None
    use_utility: Any = None
    use_states_info_gain: Any = None
    use_param_info_gain: Any = None
    action_selection: Any = None
    policy_len: Any = None
    n_policies: Any = None
    num_controls: Any = None

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a plain mapping, in field order."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
