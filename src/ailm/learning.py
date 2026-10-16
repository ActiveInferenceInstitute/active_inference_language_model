"""Parameter learning inside pymdp — closing W2's verified gap.

Until this module, every matrix in AILM was estimated by counting *outside*
pymdp and installed as a fixed array: the agents inferred, but had never
learned. Here the letter level's transition is learned by pymdp itself —
`infer_states` collects the filtered beliefs, `infer_parameters` updates a
Dirichlet over ``B`` — and the result is held against the shipped counting
path under the pre-registered prediction of ``docs/decisions.md`` entry 22:

    With an identity emission the filtered posterior at every step is a
    one-hot on the observed letter, so pymdp's Dirichlet accumulation over
    posterior outer products IS the accumulation of observed bigram
    indicators, and the posterior mean IS the add-k estimate. The two
    estimators are the same arithmetic in different code, and must agree to
    float32 tolerance — not approximately, exactly.

The comparison is meaningful only because it can fail three ways, each carried
as a control: a mismatched Dirichlet prior must break the agreement (P2), the
learned matrix must differ from its initialization (P3), and a comparator
counted on a different stream must not agree either. Agreement with counting
is a *correctness anchor*, not a capability: nothing here claims learning
helps. The online non-stationary regime — the version that would be a real
active-inference claim — remains scoped in ``TODO.md`` and needs its own
pre-registration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ailm.levels.letter import (
    ALPHA_TO_IDX,
    N_SYMBOLS,
    build_letter_transition,
    normalize_text,
    train_on_corpus,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

__all__ = [
    "CONJUGATE_LEARNING_RATE",
    "LearningResult",
    "learn_letter_transition",
    "letter_stream_indices",
]

#: Full conjugate accumulation.  This is the value at which pymdp's update is
#: algebraically identical to add-k counting and is injected into the
#: manuscript rather than repeated as a prose literal.
CONJUGATE_LEARNING_RATE = 1.0


def letter_stream_indices(text: str) -> list[int]:
    """Normalize ``text`` to the letter alphabet and index it.

    Args:
        text: Raw text.

    Returns:
        One observation index per retained character.

    Raises:
        ValueError: If fewer than two positions survive — a stream with no
            transition cannot exercise transition learning, and running on it
            would "confirm" the prediction vacuously.
    """
    indices = [ALPHA_TO_IDX[c] for c in normalize_text(text) if c in ALPHA_TO_IDX]
    if len(indices) < 2:
        raise ValueError(
            f"stream retains {len(indices)} position(s); transition learning "
            f"needs at least one transition."
        )
    return indices


@dataclass(frozen=True)
class LearningResult:
    """One learning run beside its counting comparator.

    Attributes:
        learned: ``(S, S)`` posterior-mean transition from pymdp's
            ``infer_parameters``, float64 view of the float32 computation.
        counted: ``(S, S)`` add-k transition from the shipped counting path
            over the identical stream.
        initial: ``(S, S)`` the transition before learning — the Dirichlet
            prior's expected value.
        max_abs_delta: ``max |learned - counted|``.
        n_positions: Stream length fed to the agent.
        pseudocount: The Dirichlet initialization (per-cell concentration).
    """

    learned: np.ndarray
    counted: np.ndarray
    initial: np.ndarray
    max_abs_delta: float
    n_positions: int
    pseudocount: float

    @property
    def moved_from_initial(self) -> float:
        """``max |learned - initial|`` — P3's non-vacuity quantity."""
        return float(np.max(np.abs(self.learned - self.initial)))


def learn_letter_transition(
    text: str,
    *,
    pseudocount: float = 1.0,
    comparator_pseudocount: float | None = None,
    lr: float = CONJUGATE_LEARNING_RATE,
) -> LearningResult:
    """Learn the letter transition with pymdp and count it with the shipped path.

    The agent starts from the Dirichlet prior's expected value (uniform for a
    constant ``pseudocount``), filters the stream with its own
    ``infer_states`` / ``update_empirical_prior`` loop — the identical loop the
    Stage 1 perplexity walks — and then updates through ``infer_parameters``.
    The comparator is ``build_letter_transition(train_on_corpus([text]))``, the exact
    code path that produced every shipped matrix.

    Args:
        text: The learning stream. Normalized inside; the caller states the
            population.
        pseudocount: Per-cell Dirichlet concentration for ``pB``. The matched
            comparison uses the same value on both sides.
        comparator_pseudocount: Override the counting side's add-k alone —
            P2's control. ``None`` matches ``pseudocount``.
        lr: ``lr_pB`` forwarded to ``infer_parameters``. Any value other than
            1 breaks the conjugate identity and exists only so a test can
            prove that it does.

    Returns:
        The learning run and its comparator, with the deviation measured.

    Raises:
        ValueError: If the stream is degenerate (see
            :func:`letter_stream_indices`) or ``pseudocount`` is not positive.
    """
    import jax.numpy as jnp
    from pymdp.agent import Agent

    if pseudocount <= 0.0:
        raise ValueError(f"pseudocount must be positive, got {pseudocount}")

    indices = letter_stream_indices(text)

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

    # The agent's own filtering loop — not hand-built one-hots. That the
    # posteriors COME OUT one-hot under the identity emission is the mechanism
    # the prediction rests on; constructing them by hand would assume it.
    beliefs: list = []
    empirical_prior = [jnp.ones((1, N_SYMBOLS)) / N_SYMBOLS]
    action = jnp.zeros((1, 1), dtype=jnp.int32)
    for observation in indices:
        posterior = agent.infer_states([jnp.array([observation])], empirical_prior)
        beliefs.append(posterior[0].reshape(1, 1, N_SYMBOLS))
        empirical_prior = agent.update_empirical_prior(action, posterior)

    belief_sequence = jnp.concatenate(beliefs, axis=1)  # (1, T, S)
    observations = [jnp.asarray(indices)[None, :]]
    actions = jnp.zeros((1, len(indices) - 1, 1), dtype=jnp.int32)

    updated = agent.infer_parameters(
        beliefs_A=[belief_sequence],
        observations=observations,
        actions=actions,
        lr_pB=float(lr),
    )
    learned = np.asarray(updated.B[0], dtype=np.float64)[0, :, :, 0]

    counting_k = pseudocount if comparator_pseudocount is None else comparator_pseudocount
    from ailm.config import LevelConfig

    counted = np.asarray(
        build_letter_transition(train_on_corpus([text]), config=LevelConfig(smoothing=counting_k)),
        dtype=np.float64,
    )[0]

    initial = np.asarray(initial_transition, dtype=np.float64)[0, :, :, 0]
    return LearningResult(
        learned=learned,
        counted=counted,
        initial=initial,
        max_abs_delta=float(np.max(np.abs(learned - counted))),
        n_positions=len(indices),
        pseudocount=float(pseudocount),
    )
