"""Case roles, source-faithful CEREBRUM metadata, and the derived UD oracle.

Two role systems live here and label different kinds of thing:

1. TWO ROLE SYSTEMS: ``CaseRole`` labels noun phrases (linguistic case, Level 4).
   ``CerebrumFunctionRole`` labels roles a MODEL plays. CEREBRUM itself reuses
   ``[GEN]`` for core Genitive and a separate novel Generative case; scoped member
   names and serialized values preserve that collision without ambiguity.

2. SOURCE TUPLE AND AILM CROSSWALK: CEREBRUM Supplement 8.1 defines
   ``(S, A, T, Ω, O, F, π)``. Figure 13 is a caption-only placeholder, not a
   source crosswalk. ``AILM_PYMDP_HEURISTIC_CROSSWALK`` is explicitly this
   project's interpretation and records that pymdp C/D/E have no tuple analogue.

3. CASE INVENTORY: the eight core linguistic cases are Level 4's state alphabet.
   The separate novel Generative model role is represented in the source ontology,
   but it is not a Level 4 linguistic label.

4. UD CASE ORACLE: The mapping from UD deprel labels to CaseRole is a RESEARCH CLAIM,
   not ground truth. English marks only Nom/Acc/Gen morphologically; DAT/INS/LOC/ABL/VOC
   are derived. Review required before any accuracy number rests on this mapping.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

import numpy as np

logger = logging.getLogger(__name__)


# ── Linguistic case roles ─────────────────────────────────────────────────────


class CaseRole(Enum):
    """The 8 core linguistic case roles.

    These are LINGUISTIC roles — they describe the grammatical/semantic relation
    of a noun phrase to its predicate. Used at Level 4 (Clause/Case) of AILM.

    Grounded in: Cognitive Case Diagrams v1, §2–4; CEREBRUM v1.2, Supplement 11.
    """

    NOM = "Nominative"  # Subject; active agent; initiates predication
    ACC = "Accusative"  # Direct object; receives action
    GEN = "Genitive"  # Source/possessor; generates or originates
    DAT = "Dative"  # Recipient/beneficiary; receives transfer
    INS = "Instrumental"  # Means/instrument; mediates the action
    LOC = "Locative"  # Location/context; spatial or situational frame
    ABL = "Ablative"  # Origin/source; starting point of movement or causation
    VOC = "Vocative"  # Address; entity being called or addressed


# ── CEREBRUM functional roles for computational models ────────────────────────


class CerebrumFunctionRole(Enum):
    """Scoped model-role serialization for CEREBRUM's case ontology.

    Table 1 and Supplement 11 give a core Genitive model function. Supplement
    2.12 separately introduces a Generative case and reuses the source label
    ``[GEN]``. The enum names and values retain both instead of silently replacing
    Genitive with Generative.
    """

    CORE_NOMINATIVE = "cerebrum:core:NOM:ActivePredictor"
    CORE_ACCUSATIVE = "cerebrum:core:ACC:UpdateTarget"
    CORE_GENITIVE = "cerebrum:core:GEN:RelationalSource"
    CORE_DATIVE = "cerebrum:core:DAT:DataRecipient"
    CORE_INSTRUMENTAL = "cerebrum:core:INS:PolicyImplementor"
    CORE_LOCATIVE = "cerebrum:core:LOC:ContextProvider"
    CORE_ABLATIVE = "cerebrum:core:ABL:CausalOrigin"
    CORE_VOCATIVE = "cerebrum:core:VOC:AddressableEntity"
    NOVEL_GENERATIVE = "cerebrum:novel:GEN:OutputGenerator"


CEREBRUM_FUNCTION_ABBREVIATIONS: Mapping[CerebrumFunctionRole, str] = MappingProxyType(
    {
        CerebrumFunctionRole.CORE_NOMINATIVE: "NOM",
        CerebrumFunctionRole.CORE_ACCUSATIVE: "ACC",
        CerebrumFunctionRole.CORE_GENITIVE: "GEN",
        CerebrumFunctionRole.CORE_DATIVE: "DAT",
        CerebrumFunctionRole.CORE_INSTRUMENTAL: "INS",
        CerebrumFunctionRole.CORE_LOCATIVE: "LOC",
        CerebrumFunctionRole.CORE_ABLATIVE: "ABL",
        CerebrumFunctionRole.CORE_VOCATIVE: "VOC",
        CerebrumFunctionRole.NOVEL_GENERATIVE: "GEN",
    }
)


def validate_cerebrum_function_roles(
    abbreviations: Mapping[CerebrumFunctionRole, str],
) -> None:
    """Reject an incomplete or ambiguity-collapsing source-role inventory."""
    if set(abbreviations) != set(CerebrumFunctionRole):
        raise ValueError("CEREBRUM function-role abbreviation inventory is incomplete")
    serialized = [role.value for role in CerebrumFunctionRole]
    if len(serialized) != len(set(serialized)):
        raise ValueError("CEREBRUM function-role serialization is ambiguous")
    gen_roles = {role for role, label in abbreviations.items() if label == "GEN"}
    required_gen_roles = {
        CerebrumFunctionRole.CORE_GENITIVE,
        CerebrumFunctionRole.NOVEL_GENERATIVE,
    }
    if gen_roles != required_gen_roles:
        raise ValueError(
            "source [GEN] must retain distinct core Genitive and novel Generative roles"
        )


validate_cerebrum_function_roles(CEREBRUM_FUNCTION_ABBREVIATIONS)


# ── CEREBRUM source tuple and AILM-owned pymdp heuristic ─────────────────────

CEREBRUM_POMDP_SYMBOLS: Mapping[str, str] = MappingProxyType(
    {
        "S": "hidden state space",
        "A": "action space",
        "T": "state transition dynamics",
        "Ω": "observation space",
        "O": "likelihood mapping from states to observations",
        "F": "free-energy functional",
        "π": "precision parameters",
    }
)


@dataclass(frozen=True, slots=True)
class AilmPymdpCorrespondence:
    """One structured, explicitly heuristic source-to-pymdp correspondence."""

    source_symbol: str
    pymdp_component: str | None
    relation: str
    caveat: str


# This crosswalk is an AILM engineering heuristic, not a mapping printed by
# CEREBRUM. Structured identities make negated prose and T/O swaps impossible
# to accept via substring matching.
AILM_PYMDP_HEURISTIC_CROSSWALK: Mapping[str, AilmPymdpCorrespondence] = MappingProxyType(
    {
        "S": AilmPymdpCorrespondence(
            "S", "hidden-state factor domains", "domain correspondence", "AILM heuristic"
        ),
        "A": AilmPymdpCorrespondence(
            "A",
            "action and control-factor domains",
            "domain correspondence",
            "source A is action space; not pymdp likelihood A",
        ),
        "T": AilmPymdpCorrespondence(
            "T", "B", "transition dynamics", "source T corresponds heuristically to pymdp B"
        ),
        "Ω": AilmPymdpCorrespondence(
            "Ω", "observation-modality domains", "domain correspondence", "AILM heuristic"
        ),
        "O": AilmPymdpCorrespondence(
            "O", "A", "likelihood mapping", "source O corresponds heuristically to pymdp A"
        ),
        "F": AilmPymdpCorrespondence("F", None, "variational-free-energy objective", "not pymdp G"),
        "π": AilmPymdpCorrespondence("π", None, "precision hyperparameters", "not pymdp E"),
    }
)
PYMDP_COMPONENTS_WITHOUT_CEREBRUM_TUPLE_ANALOGUE = frozenset({"C", "D", "E"})


def validate_ailm_pymdp_crosswalk(
    crosswalk: Mapping[str, AilmPymdpCorrespondence],
    without_analogue: frozenset[str],
) -> None:
    """Reject the retired Figure-13 crosswalk and its symbol conflations."""
    if set(crosswalk) != set(CEREBRUM_POMDP_SYMBOLS):
        raise ValueError("AILM heuristic must preserve every CEREBRUM Supplement 8.1 symbol")
    if without_analogue != frozenset({"C", "D", "E"}):
        raise ValueError("pymdp C, D, and E must be explicit no-analogue components")
    if dict(crosswalk) != dict(AILM_PYMDP_HEURISTIC_CROSSWALK):
        raise ValueError("AILM structured source-to-pymdp heuristic is miswired")
    if any(symbol != correspondence.source_symbol for symbol, correspondence in crosswalk.items()):
        raise ValueError("AILM crosswalk source-symbol identity is inconsistent")


validate_ailm_pymdp_crosswalk(
    AILM_PYMDP_HEURISTIC_CROSSWALK,
    PYMDP_COMPONENTS_WITHOUT_CEREBRUM_TUPLE_ANALOGUE,
)


# ── Source case formulations and AILM's intentionally narrow scalar scaffold ─


class CerebrumCaseFormulationKind(Enum):
    """The structurally distinct formulation families in CEREBRUM Supplement 11."""

    LIKELIHOOD_WEIGHT = "likelihood_weight"
    COMPLEXITY_WEIGHT = "complexity_weight"
    AUGMENTED_LATENT = "augmented_latent"


@dataclass(frozen=True)
class CerebrumCaseFormulation:
    kind: CerebrumCaseFormulationKind
    auxiliary_symbol: str | None
    precision_relation: str


CEREBRUM_CASE_FORMULATIONS: Mapping[CaseRole, CerebrumCaseFormulation] = MappingProxyType(
    {
        CaseRole.NOM: CerebrumCaseFormulation(
            CerebrumCaseFormulationKind.LIKELIHOOD_WEIGHT, None, "alpha_NOM > 1"
        ),
        CaseRole.ACC: CerebrumCaseFormulation(
            CerebrumCaseFormulationKind.COMPLEXITY_WEIGHT, None, "beta_ACC < 1"
        ),
        CaseRole.GEN: CerebrumCaseFormulation(
            CerebrumCaseFormulationKind.AUGMENTED_LATENT, "r", "pi_GEN(r) > pi_GEN(s)"
        ),
        CaseRole.DAT: CerebrumCaseFormulation(
            CerebrumCaseFormulationKind.AUGMENTED_LATENT,
            "g",
            "pi_DAT(g) approximately equals pi_DAT(s)",
        ),
        CaseRole.INS: CerebrumCaseFormulation(
            CerebrumCaseFormulationKind.AUGMENTED_LATENT, "a", "pi_INS(a) > pi_INS(s)"
        ),
        CaseRole.LOC: CerebrumCaseFormulation(
            CerebrumCaseFormulationKind.AUGMENTED_LATENT, "e", "pi_LOC(e) > pi_LOC(s)"
        ),
        CaseRole.ABL: CerebrumCaseFormulation(
            CerebrumCaseFormulationKind.AUGMENTED_LATENT,
            "o_origin",
            "pi_ABL(o_origin) > pi_ABL(s)",
        ),
        CaseRole.VOC: CerebrumCaseFormulation(
            CerebrumCaseFormulationKind.AUGMENTED_LATENT, "c", "pi_VOC(c) > pi_VOC(s)"
        ),
    }
)


@dataclass(frozen=True)
class AilmScalarCaseWeights:
    """Project-owned numerical realization of a source inequality."""

    alpha: float
    beta: float


# CEREBRUM specifies only alpha_NOM > 1 and beta_ACC < 1; it does not supply
# these numerical values. They are AILM defaults for an inert NOM/ACC scaffold.
AILM_SCALAR_CASE_WEIGHTS: Mapping[CaseRole, AilmScalarCaseWeights] = MappingProxyType(
    {
        CaseRole.NOM: AilmScalarCaseWeights(alpha=1.5, beta=1.0),
        CaseRole.ACC: AilmScalarCaseWeights(alpha=1.0, beta=0.5),
    }
)


# ── UD deprel → CaseRole oracle ───────────────────────────────────────────────
#
# RESEARCH CLAIM — NOT GROUND TRUTH.
# English marks Nom/Acc/Gen morphologically. DAT/INS/LOC/ABL/VOC are DERIVED via
# this mapping. The mapping is a design decision; it must be reviewed and versioned
# before any accuracy number derived from it is treated as meaningful.
# See docs/design/cerebrum_mapping.md §"Case Oracle".
#
UD_DEPREL_TO_CASE: dict[str, CaseRole] = {
    "nsubj": CaseRole.NOM,  # nominal subject → Nominative
    "nsubj:pass": CaseRole.ACC,  # passive subject → Accusative (patient)
    "obj": CaseRole.ACC,  # direct object → Accusative
    "iobj": CaseRole.DAT,  # indirect object → Dative
    "nmod:poss": CaseRole.GEN,  # possessive modifier → Genitive
    "nmod": CaseRole.GEN,  # nominal modifier (approximation) → Genitive
    "obl": CaseRole.LOC,  # oblique (context-dependent; default LOC)
    "obl:agent": CaseRole.ABL,  # passive by-phrase → Ablative
    "vocative": CaseRole.VOC,  # vocative → Vocative
}

# Adposition lemma → CaseRole for extended coverage.
# Applied to the ``deprel="case"`` ADP token from that token's own lemma;
# its nominal head is neither consulted nor relabeled by this route.
# ALSO A RESEARCH CLAIM. Approximate and English-specific.
ADPOSITION_TO_CASE: dict[str, CaseRole] = {
    "to": CaseRole.DAT,
    "for": CaseRole.DAT,
    "with": CaseRole.INS,
    "by": CaseRole.INS,
    "in": CaseRole.LOC,
    "at": CaseRole.LOC,
    "on": CaseRole.LOC,
    "from": CaseRole.ABL,
    "of": CaseRole.GEN,
    "'s": CaseRole.GEN,
}


# ── Free-energy computations ──────────────────────────────────────────────────


def transformation_cost(
    q_src: np.ndarray,
    q_tgt: np.ndarray,
    target_free_energy: float,
) -> float:
    """CEREBRUM Supplement 8.2.3 transformation-specific free energy.

    ``F_trans = F_target + KL(q_target || q_source)``. The KL term is unscaled
    here. CEREBRUM's later expected-free-energy transition uses a separate
    precision-scaled term; :func:`expected_free_energy_transition` implements it.

    Args:
        q_src: Belief distribution under source case. Must sum to 1.
        q_tgt: Belief distribution under target case. Must sum to 1.
        target_free_energy: The target case's VFE term.

    Returns:
        Target VFE plus the directed, unscaled KL divergence.

    Raises:
        ValueError: If distributions are invalid (not normalized, negative, mismatched).
    """
    if not np.isfinite(target_free_energy):
        raise ValueError(f"target_free_energy must be finite, got {target_free_energy!r}")
    kl = _directed_kl(q_src, q_tgt)
    cost = float(target_free_energy + kl)
    logger.debug(
        "transformation_cost: target_vfe=%.4f KL=%.4f total=%.4f",
        target_free_energy,
        kl,
        cost,
    )
    return cost


def expected_free_energy_transition(
    q_src: np.ndarray,
    q_tgt: np.ndarray,
    base_expected_free_energy: float,
    gamma: float,
) -> float:
    """CEREBRUM Supplement 8.8.3's distinct gamma-scaled EFE transition.

    ``G_transition = G_base + gamma * KL(q_target || q_source)``. Keeping this
    separate from :func:`transformation_cost` prevents a precision weight from
    being retrofitted into the unscaled VFE equation.
    """
    if not np.isfinite(base_expected_free_energy):
        raise ValueError(
            f"base_expected_free_energy must be finite, got {base_expected_free_energy!r}"
        )
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError(f"gamma must be finite and positive, got {gamma!r}")
    kl = _directed_kl(q_src, q_tgt)
    objective = float(base_expected_free_energy + gamma * kl)
    logger.debug(
        "expected_free_energy_transition: base=%.4f gamma=%.4f KL=%.4f total=%.4f",
        base_expected_free_energy,
        gamma,
        kl,
        objective,
    )
    return objective


def _directed_kl(q_src: np.ndarray, q_tgt: np.ndarray) -> float:
    """Return ``KL(q_tgt || q_src)`` after strict probability validation."""
    source = _probability_vector("q_src", q_src)
    target = _probability_vector("q_tgt", q_tgt)
    if source.shape != target.shape:
        raise ValueError(f"Shape mismatch: q_src {source.shape} vs q_tgt {target.shape}")
    nonzero = target > 0
    if np.any(nonzero & (source <= 0)):
        return float("inf")
    return float(np.sum(target[nonzero] * np.log(target[nonzero] / source[nonzero])))


def _probability_vector(name: str, values: np.ndarray) -> np.ndarray:
    """Validate a finite, non-empty, one-dimensional probability vector."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional probability vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(vector < 0):
        raise ValueError(f"{name} must be non-negative")
    if not np.isclose(vector.sum(), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(f"{name} must sum to 1, got {vector.sum():.6f}")
    return vector


def case_conditioned_free_energy(
    q: np.ndarray,
    log_likelihood: np.ndarray,
    case_role: CaseRole,
    log_prior: np.ndarray | None = None,
) -> float:
    """AILM's scalar NOM/ACC approximation to two CEREBRUM formulations.

    ``F_scalar[q] = beta * KL(q || p) - alpha * E_q[log p(o|s)]``.

    CEREBRUM gives a scalar likelihood inequality for NOM and a scalar
    complexity inequality for ACC. AILM supplies the numerical defaults in
    ``AILM_SCALAR_CASE_WEIGHTS``. The other core cases use augmented latent
    variables and precision relations in the source; this function refuses them
    rather than pretending they are neutral scalar pairs.

    Args:
        q: Variational posterior (must sum to 1).
        log_likelihood: Discrete log-probability mass ``log p(o | s_i)`` for
            each state i. ``-inf`` represents zero mass. Log densities are not
            accepted by this discrete model.
        case_role: NOM or ACC. Other roles require their source-specific state.
        log_prior: Normalized ``log p(s_i)``. ``-inf`` represents zero prior
            mass and is valid only where ``q`` has zero support. Defaults to
            uniform ``log(1/n)``.

    Returns:
        Case-conditioned VFE (lower = better fit to data under this case).

    Raises:
        ValueError: If numerical inputs are invalid.
        NotImplementedError: If ``case_role`` is not part of the scalar scaffold.
    """
    q = _probability_vector("q", q)
    log_likelihood = np.asarray(log_likelihood, dtype=np.float64)
    n = len(q)

    if log_likelihood.shape != q.shape:
        raise ValueError(
            "log_likelihood must have the same one-dimensional shape as q: "
            f"{log_likelihood.shape} != {q.shape}"
        )
    if np.any(np.isnan(log_likelihood)) or np.any(np.isposinf(log_likelihood)):
        raise ValueError(
            "log_likelihood may contain finite discrete log masses or -inf for "
            "zero mass, but not NaN or +inf"
        )
    if np.any(log_likelihood > 1e-12):
        raise ValueError(
            "log_likelihood entries must be discrete log-probability masses <= 0; "
            "positive log densities are outside this AILM scaffold"
        )

    if log_prior is None:
        log_prior = np.full(n, -np.log(n))
    log_prior = np.asarray(log_prior, dtype=np.float64)
    if log_prior.shape != q.shape:
        raise ValueError(
            "log_prior must have the same one-dimensional shape as q: "
            f"{log_prior.shape} != {q.shape}"
        )
    if np.any(np.isnan(log_prior)) or np.any(np.isposinf(log_prior)):
        raise ValueError(
            "log_prior may contain finite normalized log masses or -inf for "
            "zero mass, but not NaN or +inf"
        )
    finite_prior = np.isfinite(log_prior)
    if not np.any(finite_prior):
        raise ValueError("log_prior must assign positive mass to at least one state")
    max_log_prior = float(np.max(log_prior[finite_prior]))
    log_normalizer = max_log_prior + float(
        np.log(np.sum(np.exp(log_prior[finite_prior] - max_log_prior)))
    )
    if not np.isclose(log_normalizer, 0.0, atol=1e-6):
        raise ValueError(
            "log_prior must be a normalized log-probability vector; "
            f"logsumexp(log_prior)={log_normalizer:.6g}"
        )

    try:
        weights = AILM_SCALAR_CASE_WEIGHTS[case_role]
    except KeyError as exc:
        role_name = case_role.name if isinstance(case_role, CaseRole) else repr(case_role)
        raise NotImplementedError(
            f"{role_name} uses a non-scalar CEREBRUM formulation; "
            "provide its auxiliary latent state in a future explicit implementation"
        ) from exc
    alpha = weights.alpha
    beta = weights.beta

    # KL(q || p_prior) = E_q[log q - log p]
    nonzero = q > 0
    if np.any(nonzero & np.isneginf(log_prior)):
        kl = float("inf")
    else:
        log_q = np.zeros_like(q)
        log_q[nonzero] = np.log(q[nonzero])
        kl = float(np.sum(q[nonzero] * (log_q[nonzero] - log_prior[nonzero])))

    # E_q[log p(o|s)]
    if np.any(nonzero & np.isneginf(log_likelihood)):
        expected_ll = float("-inf")
    else:
        expected_ll = float(np.dot(q[nonzero], log_likelihood[nonzero]))

    vfe = beta * kl - alpha * expected_ll
    logger.debug(
        "case_vfe(%s): α=%.2f β=%.2f KL=%.4f E_ll=%.4f VFE=%.4f",
        case_role.name,
        alpha,
        beta,
        kl,
        expected_ll,
        vfe,
    )
    return vfe
