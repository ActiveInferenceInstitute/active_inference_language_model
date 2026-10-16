"""Online parameter learning on non-stationary text — decision entry 28.

What this module is
-------------------
``docs/decisions.md`` entry 28 pre-registers the W2 increment where learning
could actually *earn* its way in: an agent that keeps learning on a changing
distribution should track a drift that a frozen (non-learning) agent cannot.
The setup is scoped to where conjugacy is exact — the letter level, whose
identity emission makes beliefs path-supervised-equivalent, so conjugate
counting is exact (decisions 22-23).

A non-stationary stream is built from two Brown genres whose letter-bigram
transition statistics genuinely differ (a principled, injected drift —
government, then romance; measured drift between the two is a control, not an
assertion). The **online** agent continues pymdp's ``infer_parameters`` updates
as the stream crosses the drift point. The **frozen** control is the add-k
counted ``B`` fixed from the first distribution only — it must lose on
held-out text from the second distribution.

The three pre-registered predictions:

- **P1 (the drift is real).** The two segments differ in held-out letter
  perplexity under the frozen matrix by more than a tolerance, so a later gap
  cannot be blamed on a stream that never changed.
- **P2 (frozen must lose).** On held-out text from the second distribution at
  (after) the drift point, the online agent's perplexity is strictly lower
  than the frozen control's. This is the load-bearing clause.
- **P3 (identity before the drift, non-vacuity).** On held-out text from the
  *first* distribution, the two agents score identically (at learning rate 1
  the online path equals counting), and a re-fit control (the frozen matrix
  re-estimated on the second segment) scores at least as well as the online
  agent — showing the drift itself, not estimator noise, is what the online
  path tracks.

The module returns a result object whose ``refuse_reason`` is non-None exactly
when any pre-registered clause fails; the recorder refuses to write in that
case, so a null can never be recorded as if the prediction held.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ailm.levels._stochastic import column_stochastic
from ailm.levels.letter import ALPHA_TO_IDX, N_SYMBOLS, normalize_text

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = [
    "OnlineLearningResult",
    "closed_form_perplexity",
    "letter_indices",
    "online_letter_experiment_numpy",
    "run_online_letter_experiment",
]

#: Default drift design: two Brown genres with genuinely different letter-bigram
#: statistics, held as the injectable knob so a test can substitute a no-drift
#: pair (same genre twice) as the negative control.
DEFAULT_PRE_GENRE = "government"
DEFAULT_POST_GENRE = "romance"

#: Per-cell Dirichlet concentration; lr=1 makes online equal counting (dec 23).
DEFAULT_PSEUDOCOUNT = 1.0

#: Tolerance for "drift is real" / "identical before drift" thresholds.
DRIFT_TOLERANCE = 0.01
IDENTITY_TOLERANCE = 1e-6

#: Smallest probability fed to log.
_LOG_FLOOR = 1e-10


def letter_indices(text: str) -> list[int]:
    """Normalize ``text`` to the letter alphabet and index it.

    Raises:
        ValueError: If fewer than two positions survive — a stream with no
            transition cannot exercise transition learning.
    """
    indices = [ALPHA_TO_IDX[c] for c in normalize_text(text) if c in ALPHA_TO_IDX]
    if len(indices) < 2:
        raise ValueError(
            f"stream retains {len(indices)} position(s); transition learning "
            f"needs at least one transition."
        )
    return indices


def closed_form_perplexity(transition: NDArray[np.float64], indices: list[int]) -> float:
    """Per-letter perplexity of the identity-emission chain in closed form.

    Under the identity emission, ``P(letter_t | observed letter_{t-1})`` is the
    matching column of ``transition``, so perplexity can be computed in numpy
    exactly as ``pymdp``'s forward filter would produce it (the same license
    the L1 ``compute_perplexity`` uses — verified by ``test_learning.py``).
    Position 0 has no predecessor, so it uses the uniform prior on ``D``.

    Raises:
        ValueError: If ``indices`` has fewer than two positions.
    """
    if len(indices) < 2:
        raise ValueError("indices must contain at least two positions")
    logs: list[float] = []
    prior = np.full(N_SYMBOLS, 1.0 / N_SYMBOLS)
    # Position 0: D is uniform over the initial letter.
    logs.append(float(np.log(max(float(prior[indices[0]]), _LOG_FLOOR))))
    prev = indices[0]
    for cur in indices[1:]:
        logs.append(float(np.log(max(float(transition[cur, prev]), _LOG_FLOOR))))
        prev = cur
    return float(np.exp(-sum(logs) / len(logs)))


def _count_transition(indices: list[int], pseudocount: float) -> NDArray[np.float64]:
    """add-k column-stochastic transition from an index stream."""
    mat = np.full((N_SYMBOLS, N_SYMBOLS), float(pseudocount), dtype=np.float64)
    for i in range(len(indices) - 1):
        mat[indices[i + 1], indices[i]] += 1.0
    return column_stochastic(mat)


@dataclass(frozen=True)
class OnlineLearningResult:
    """One online-vs-frozen run on a drifted letter stream.

    Attributes:
        online: ``(S, S)`` transition learned by pymdp's ``infer_parameters``
            as the stream crosses the drift point (pre then post segment).
        frozen: ``(S, S)`` add-k counted transition fixed from the pre (first)
            distribution only.
        online_pre: ``(S, S)`` the online agent's transition as it stood *at*
            the drift point, before consuming the post segment — identical to
            ``frozen`` at lr=1 (P3 identity).
        refit_post: ``(S, S)`` the frozen matrix re-estimated on the post
            (second) distribution alone (P3 re-fit control).
        ppl_frozen_pre: frozen perplexity on pre-distribution held-out text.
        ppl_frozen_post: frozen perplexity on post-distribution held-out text.
        ppl_online_post: online perplexity on post-distribution held-out text.
        ppl_refit_post: refit-post perplexity on post held-out text.
        n_pre_learn / n_post_learn / n_pre_hold / n_post_hold: stream sizes.
        pseudocount: The Dirichlet / add-k concentration.
        max_drift: ``|ppl_frozen_post - ppl_frozen_pre|`` — P1's measured drift.
        p1_passes / p2_passes / p3_passes: bool per pre-registered clause.
        refuse_reason: Message when any clause fails; None when all pass.
    """

    online: np.ndarray
    frozen: np.ndarray
    online_pre: np.ndarray
    refit_post: np.ndarray
    ppl_frozen_pre: float
    ppl_frozen_post: float
    ppl_online_post: float
    ppl_refit_post: float
    n_pre_learn: int
    n_post_learn: int
    n_pre_hold: int
    n_post_hold: int
    pseudocount: float
    max_drift: float
    p1_passes: bool
    p2_passes: bool
    p3_passes: bool
    refuse_reason: str | None


def _pymdp_online_transition(
    pre_indices: list[int],
    post_indices: list[int],
    pseudocount: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run pymdp's online learning and return ``(pre_state, final)``.

    The agent is created with ``learn_B=True`` and a Dirichlet prior at
    ``pseudocount``. It filters the pre segment (``infer_states`` +
    ``update_empirical_prior``), stores the pre beliefs, then filters the post
    segment *continuing from the same agent and its already-updated* state,
    storing post beliefs. ``infer_parameters`` is called once per segment so
    the posterior Dirichlet accumulates across the drift — this is the online
    path: it keeps learning on the changed distribution rather than being
    refit from scratch.

    Returns:
        ``(online_pre, online)`` — the transition after the pre segment alone,
        and after consuming both segments. ``online_pre`` must equal the
        counting estimate (P3 identity), and ``online`` is what the online
        agent uses at evaluation time.
    """
    import jax.numpy as jnp
    from pymdp.agent import Agent

    prior_concentration = jnp.full(
        (1, N_SYMBOLS, N_SYMBOLS, 1), float(pseudocount), dtype=jnp.float32
    )
    initial_transition = jnp.full((1, N_SYMBOLS, N_SYMBOLS, 1), 1.0 / N_SYMBOLS, dtype=jnp.float32)
    agent = Agent(
        A=[jnp.eye(N_SYMBOLS)[jnp.newaxis]],
        B=[initial_transition],
        C=[jnp.zeros((1, N_SYMBOLS))],
        D=[jnp.ones((1, N_SYMBOLS)) / N_SYMBOLS],
        pB=[prior_concentration],
        learn_B=True,
        num_controls=[1],
        batch_size=1,
    )

    def filter_and_learn(agent, indices, beliefs_so_far):
        empirical_prior = [jnp.ones((1, N_SYMBOLS)) / N_SYMBOLS]
        action = jnp.zeros((1, 1), dtype=jnp.int32)
        beliefs = list(beliefs_so_far)
        for observation in indices:
            posterior = agent.infer_states([jnp.array([observation])], empirical_prior)
            beliefs.append(posterior[0].reshape(1, 1, N_SYMBOLS))
            empirical_prior = agent.update_empirical_prior(action, posterior)
        return agent, beliefs

    # Pre segment — the agent stands here at the drift point.
    agent, beliefs = filter_and_learn(agent, pre_indices, [])
    belief_seq = jnp.concatenate(beliefs, axis=1)  # (1, T, S)
    observations = [jnp.asarray(pre_indices)[None, :]]
    actions = jnp.zeros((1, max(len(pre_indices) - 1, 0), 1), dtype=jnp.int32)
    agent = agent.infer_parameters(
        beliefs_A=[belief_seq], observations=observations, actions=actions, lr_pB=1.0
    )
    online_pre = np.asarray(agent.B[0], dtype=np.float64)[0, :, :, 0]

    # Post segment — continue from the already-updated agent.
    agent, beliefs = filter_and_learn(agent, post_indices, [])
    belief_seq = jnp.concatenate(beliefs, axis=1)
    observations = [jnp.asarray(post_indices)[None, :]]
    actions = jnp.zeros((1, max(len(post_indices) - 1, 0), 1), dtype=jnp.int32)
    agent = agent.infer_parameters(
        beliefs_A=[belief_seq], observations=observations, actions=actions, lr_pB=1.0
    )
    online = np.asarray(agent.B[0], dtype=np.float64)[0, :, :, 0]
    return online_pre, online


def online_letter_experiment_numpy(
    pre_learn_text: str,
    post_learn_text: str,
    pre_hold_text: str,
    post_hold_text: str,
    *,
    pseudocount: float = DEFAULT_PSEUDOCOUNT,
) -> dict:
    """Independent numpy-only recomputation of the P2 sign.

    Uses the add-k counting path for the online matrix (which, at lr=1 and an
    identity emission, equals pymdp's conjugate accumulation — decisions
    22-23) and the closed-form perplexity. Returns just the two perplexities
    and the P2 sign, so the recorder can confirm the pymdp path and the numpy
    path agree on which agent wins on post-drift text.

    Returns:
        ``{"ppl_online_post", "ppl_frozen_post", "online_beats_frozen"}``.
    """
    pre_learn = letter_indices(pre_learn_text)
    post_learn = letter_indices(post_learn_text)
    post_hold = letter_indices(post_hold_text)
    frozen = _count_transition(pre_learn, pseudocount)
    # online == counting over pre+post at lr=1 (exact conjugacy, decisions 22-23)
    online = _count_transition(pre_learn + post_learn, pseudocount)
    ppl_online_post = closed_form_perplexity(online, post_hold)
    ppl_frozen_post = closed_form_perplexity(frozen, post_hold)
    return {
        "ppl_online_post": ppl_online_post,
        "ppl_frozen_post": ppl_frozen_post,
        "online_beats_frozen": ppl_online_post < ppl_frozen_post,
    }


def run_online_letter_experiment(
    pre_learn_text: str,
    post_learn_text: str,
    pre_hold_text: str,
    post_hold_text: str,
    *,
    pseudocount: float = DEFAULT_PSEUDOCOUNT,
) -> OnlineLearningResult:
    """Run the entry-28 online-vs-frozen experiment and evaluate P1-P3.

    All arguments are raw text (or already-length-capped streams are not
    expected here; callers bound sizes via the stream, e.g. the generator).
    """
    pre_learn = letter_indices(pre_learn_text)
    post_learn = letter_indices(post_learn_text)
    pre_hold = letter_indices(pre_hold_text)
    post_hold = letter_indices(post_hold_text)

    # The online agent keeps learning across the drift (real pymdp path).
    online_pre, online = _pymdp_online_transition(pre_learn, post_learn, pseudocount)
    frozen = _count_transition(pre_learn, pseudocount)
    refit_post = _count_transition(post_learn, pseudocount)

    ppl_frozen_pre = closed_form_perplexity(frozen, pre_hold)
    ppl_frozen_post = closed_form_perplexity(frozen, post_hold)
    ppl_online_post = closed_form_perplexity(online, post_hold)
    ppl_refit_post = closed_form_perplexity(refit_post, post_hold)

    max_drift = abs(ppl_frozen_post - ppl_frozen_pre)

    refuse_reason = None
    # P1: drift is real.
    if max_drift <= DRIFT_TOLERANCE:
        refuse_reason = (
            f"P1 FAILED: the frozen matrix's held-out perplexity changes by "
            f"only {max_drift:.4f} between the two genres (<= tolerance "
            f"{DRIFT_TOLERANCE}). No drift to track — refuse to record."
        )
    # P2: frozen must lose (online strictly lower perplexity on post text).
    if refuse_reason is None and not ppl_online_post < ppl_frozen_post:
        refuse_reason = (
            f"P2 FAILED: online {ppl_online_post:.4f} does not beat frozen "
            f"{ppl_frozen_post:.4f} on post-drift held-out text. The online "
            f"conjugate path did not recover the drift — refuse to record."
        )
    # P3a: identity before the drift (online_pre == frozen to rounding).
    pre_drift_identity = float(np.max(np.abs(online_pre - frozen))) <= IDENTITY_TOLERANCE
    # P3b: re-fit control scores at least as well as online (<= perplexity).
    refit_at_least_as_well = ppl_refit_post <= ppl_online_post + 1e-9
    p3_passes = pre_drift_identity and refit_at_least_as_well
    if refuse_reason is None and not p3_passes:
        refuse_reason = (
            f"P3 FAILED: pre-drift identity={pre_drift_identity}, "
            f"refit-at-least-as-well={refit_at_least_as_well} (refit "
            f"{ppl_refit_post:.4f} vs online {ppl_online_post:.4f})."
        )

    return OnlineLearningResult(
        online=online,
        frozen=frozen,
        online_pre=online_pre,
        refit_post=refit_post,
        ppl_frozen_pre=ppl_frozen_pre,
        ppl_frozen_post=ppl_frozen_post,
        ppl_online_post=ppl_online_post,
        ppl_refit_post=ppl_refit_post,
        n_pre_learn=len(pre_learn),
        n_post_learn=len(post_learn),
        n_pre_hold=len(pre_hold),
        n_post_hold=len(post_hold),
        pseudocount=float(pseudocount),
        max_drift=max_drift,
        p1_passes=max_drift > DRIFT_TOLERANCE,
        p2_passes=ppl_online_post < ppl_frozen_post,
        p3_passes=p3_passes,
        refuse_reason=refuse_reason,
    )
