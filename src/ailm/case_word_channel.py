"""The L4→L3 case-word bridge: model-sender, message, and prior inlet.

Protocol P3 (`output/reports/omp-dispatch-20260908/evidence/
extensions-E1R-frozen/docs/extension_protocols.md` §4): the executable seam
that marginalizes the case level's own role posterior into a word message and
inserts it into the Level 3 word prior via the shipped convex mixing.

## Runtime posterior semantics (finalized; no unregistered sequential dependence)

``q_L4(role | upos)`` is the case level's **real model posterior for one
observation from the reset prior**: the empirical prior is reset to the
shipped start distribution ``D`` at every token, the UPOS token is observed
once, and :meth:`pymdp.agent.Agent.infer_states` performs the Bayesian update.
This is the shipped scorer's own position-0 behavior
(``src/ailm/levels/case.py``: ``belief = start * emission[obs]`` normalized) —
the level's learned case-role transition ``B`` is deliberately NOT chained
across the receiver's tokens, because the L3 inlet consumes per-token messages
and a chained filter would inject a sequential dependence the shipped
estimands never registered. Every row of the posterior matrix is asserted
numerically identical to its closed-form Bayes expression
``D[role] · A[upos | role] / Σ_role' D[role'] · A[upos | role']``, so the
"real model's posterior" and the textbook inversion are one quantity here,
not two.

## Message and policies

``M(upos) = Σ_role q̃_L4(role | upos) · K(word | role)`` where ``K`` is fitted
**only** from the declared training oracle-label routes (train rows of the
case-word artifact, per-route ``label_source`` recorded), pooled over routes
as a declared, inspectable choice, counted over the lowercase exact-string
intersection with the Stage 3 vocabulary. ``K`` is the empirical distribution
of each role's in-vocabulary training words — no smoothing — so the message's
support is the intersection and every out-of-intersection receiver word keeps
mass ``0`` in the message; under the convex mix at :func:`ailm.levels.
clause_channel.mix_prior` each such word's prior mass shrinks by exactly
``(1 − α)``. That per-token effective-α heterogeneity is part of the declared
intervention.

**Zero-support policy:** roles with no in-vocabulary training word are
excluded from the marginalization and ``q_L4`` renormalized over the supported
roles. **No-message policy:** if no role has support, the inlet returns the
receiver prior unchanged — the deterministic ``D_α = D_0`` anchor — never an
unnormalized distribution.

## What this module deliberately does not do

- No oracle-sender path: the token's oracle role selecting ``K(word | role)``
  directly is a separately labeled diagnostic arm
  (:data:`ailm.channel_registry.ChannelEvidenceRule.CASE_WORD_ORACLE_DIAGNOSTIC`)
  and is not implemented here as the bridge estimand.
- No performance reading: the inlet returns a prior; scoring that prior is
  the receiver evaluation's business, under its own registration.
- The case labels of any evaluation population are never an input to any
  function here — the signatures have no label parameter, so the withheld
  oracle fields cannot influence the model sender structurally.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from ailm.levels.clause_channel import mix_prior

__all__ = [
    "POSTERIOR_IDENTITY_RTOL",
    "apply_case_word_inlet",
    "build_message",
    "fit_sender",
    "role_posterior_from_agent",
]

#: Tolerance for the numerical identity between ``infer_states`` and the
#: closed-form Bayes expression. The agent evaluates in float32; the closed
#: form in float64. Rounding is the only admissible difference.
POSTERIOR_IDENTITY_RTOL = 1e-5


def role_posterior_from_agent(
    agent,
    n_upos: int,
    n_cases: int,
) -> NDArray[np.float64]:
    """The case level's real per-UPOS role posterior, with reset semantics.

    For every UPOS observation index ``u``: reset the empirical prior to the
    agent's own start distribution ``D``, observe ``u`` once, and read the
    posterior from ``infer_states``. Each row is then asserted identical to
    its closed-form Bayes expression ``D[role] · A[u, role] / Σ`` — the
    shipped model's posterior and the textbook inversion must be the same
    quantity or nothing downstream is interpretable.

    Args:
        agent: The real case-level pymdp agent (``A`` of shape
            ``(1, n_upos, n_cases)``, ``D`` of shape ``(1, n_cases)``).
        n_upos: Number of UPOS observation rows of ``A``.
        n_cases: Number of case-role states.

    Returns:
        ``(n_upos, n_cases)`` float64 posterior matrix; each row sums to 1.

    Raises:
        ValueError: If any row disagrees with its Bayes expression beyond
            :data:`POSTERIOR_IDENTITY_RTOL`.
    """
    start = np.asarray(agent.D[0]).astype(np.float64).reshape(n_cases)
    emission_raw = np.asarray(agent.A[0]).astype(np.float64)
    if emission_raw.size != n_upos * n_cases:
        raise ValueError(
            f"agent A carries {emission_raw.size} entries; expected "
            f"{n_upos} UPOS rows × {n_cases} case states"
        )
    emission = emission_raw.reshape(n_upos, n_cases)  # P(upos | case), row per UPOS
    posterior = np.zeros((n_upos, n_cases), dtype=np.float64)
    for upos in range(n_upos):
        prior = [jnp.asarray(agent.D[0]).reshape(1, -1)]
        qs = agent.infer_states([jnp.array([upos])], prior)
        realized = np.asarray(qs[0]).ravel().astype(np.float64)
        closed = start * emission[upos, :]
        total = closed.sum()
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(
                f"UPOS row {upos} has zero Bayes mass under the start prior; "
                f"the emission and start prior disagree"
            )
        closed = closed / total
        if not np.allclose(realized, closed, rtol=POSTERIOR_IDENTITY_RTOL, atol=1e-7):
            worst = float(np.max(np.abs(realized - closed)))
            raise ValueError(
                f"UPOS row {upos}: infer_states disagrees with its Bayes "
                f"expression by {worst:.3e} (rtol {POSTERIOR_IDENTITY_RTOL}); "
                f"the shipped posterior is not the model's own inversion"
            )
        posterior[upos, :] = closed
    return posterior


def fit_sender(
    sender_map: Mapping[str, Mapping[str, int]],
    vocab: list[str],
) -> dict[str, NDArray[np.float64]]:
    """Fit ``K(word | role)`` from the artifact's train-route counts.

    The sender map is the case-word artifact's train-only, in-vocabulary count
    map (pooled over the declared routes); this function turns each role's
    counts into its empirical word distribution over the receiver vocabulary
    index order. No smoothing: the message's support is the intersection, and
    a role with no in-vocabulary training word must stay out of the
    marginalization rather than acquire fake uniform mass.

    Args:
        sender_map: ``role -> {lowercase in-vocabulary word: count}`` from the
            case-word artifact (train rows only, by construction).
        vocab: The receiver vocabulary in receiver index order.

    Returns:
        ``role -> (n_vocab,)`` float64 distribution; unsupported roles are
        absent from the mapping.

    Raises:
        ValueError: If a count is not a positive integer, a word is not in the
            vocabulary, or a role's counts are empty.
    """
    index = {word: position for position, word in enumerate(vocab)}
    sender: dict[str, NDArray[np.float64]] = {}
    for role, counts in sender_map.items():
        if not counts:
            continue
        vector = np.zeros(len(vocab), dtype=np.float64)
        for word, count in counts.items():
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise ValueError(
                    f"sender_map[{role!r}][{word!r}] must be a positive integer, got {count!r}"
                )
            if word not in index:
                raise ValueError(
                    f"sender_map[{role!r}] counts out-of-vocabulary word {word!r}; "
                    f"the artifact must be pre-intersected with the receiver vocabulary"
                )
            vector[index[word]] = float(count)
        total = vector.sum()
        sender[role] = vector / total
    if not sender:
        raise ValueError("the sender map carries no supported role; nothing to send")
    return sender


def build_message(
    posterior_row: NDArray[np.float64],
    role_names: list[str],
    sender: Mapping[str, NDArray[np.float64]],
) -> NDArray[np.float64] | None:
    """Marginalize the role posterior into one word message.

    ``M = Σ_role q̃(role) · K(word | role)`` over the supported roles — those
    the sender fitted — with ``q̃`` the posterior renormalized over them. A
    posterior row that is nonfinite, negative, or unnormalized is refused
    before it can reach the receiver.

    Args:
        posterior_row: ``(n_cases,)`` role posterior for one observed UPOS.
        role_names: Role name per posterior column.
        sender: ``role -> K(word | role)`` from :func:`fit_sender`.

    Returns:
        ``(n_vocab,)`` message distribution, or ``None`` when no role has
        support (the no-message case; the inlet then keeps ``D_α = D_0``).

    Raises:
        ValueError: If the posterior row is malformed or no column maps to a
            supported role's name.
    """
    row = np.asarray(posterior_row, dtype=np.float64)
    if row.ndim != 1:
        raise ValueError(f"posterior_row must be one-dimensional, got shape {row.shape}")
    if not np.all(np.isfinite(row)):
        raise ValueError("posterior_row must be finite")
    if np.any(row < 0.0):
        raise ValueError("posterior_row must be nonnegative")
    total = float(row.sum())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"posterior_row must sum to 1 within 1e-6, got {total!r}")

    supported_roles = set(sender)
    supported = [i for i, name in enumerate(role_names) if name in supported_roles]
    if not supported:
        raise ValueError(
            "no posterior column names a supported role; the role-name order "
            "and the sender's role keys disagree"
        )
    weights = row[supported]
    weights = weights / weights.sum()
    message = np.zeros(len(next(iter(sender.values()))), dtype=np.float64)
    for weight, column in zip(weights, supported, strict=True):
        message += weight * sender[role_names[column]]
    if not np.all(np.isfinite(message)) or np.any(message < 0.0):
        raise ValueError("the marginalized message is not a finite nonnegative distribution")
    return message


def apply_case_word_inlet(
    prior: NDArray[np.float64],
    message: NDArray[np.float64] | None,
    alpha: float,
) -> NDArray[np.float64]:
    """The concrete L3 prior inlet: insert the message into the word prior.

    ``None`` (the no-message case) and ``alpha == 0`` both return the receiver
    prior unchanged — the deterministic ``D_α = D_0`` anchor, bit-identical
    because no arithmetic is performed at all on the no-message path. Any
    other call is delegated to :func:`ailm.levels.clause_channel.mix_prior`,
    whose seam validation refuses a malformed or unnormalized operand.

    Args:
        prior: The receiver's L3 word prior (from the shipped
            :func:`ailm.levels.word.build_word_prior`).
        message: The marginalized message, or ``None``.
        alpha: Mixing weight in ``[0, 1]``.

    Returns:
        The mixed receiver prior.
    """
    if message is None:
        return prior
    return mix_prior(prior, message, alpha)
