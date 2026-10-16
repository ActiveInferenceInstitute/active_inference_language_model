"""Latent-state G2P emission learning experiment (decision 29).

Controlled design (pre-registered in ``docs/decisions.md`` entry 29): the
phone states are unobserved, so this is deliberately NOT an entry-22-style
conjugate-agreement test. pymdp's own filtering loop produces the marginal
beliefs that ``infer_parameters(..., learn_A=True)`` consumes, and the held-out
comparison of the resulting decoder against the fixed count-based emission is
the result — whichever direction it lands.

The learning lane is bounded by an *even stride* over the train split (never a
prefix: the vendored pair list is alphabetically sorted, so a prefix is a/b/c
words only — decision 4). The evaluation lane is the FULL held-out test split,
scored position-for-position by the real pymdp decoder and cross-checked
against the independent numpy oracle, which must agree on every position or the
recorder refuses to write.

P4 control: feeding the true phone one-hot beliefs to the same
``infer_parameters`` machinery must recover the count-based emission (the
entry-22 identity, states observed). It verifies that any non-agreement in the
main lane is attributable to latent-state inference, not to a broken parameter
update.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ailm.levels.phoneme import (
    ALPHA_TO_IDX,
    N_PHONES,
    PHONE_TO_IDX,
    build_g2p_agent,
    build_g2p_emission,
    build_g2p_start,
    build_g2p_transition,
    decode_positions,
)
from ailm.verification.oracle import forward_filter

N_OBS = len(ALPHA_TO_IDX)

__all__ = ["G2PLearningResult", "run_g2p_emission_learning"]


@dataclass(frozen=True)
class G2PLearningResult:
    """One latent-state emission-learning run and its controls.

    Attributes:
        learned_emission: Posterior-mean emission from pymdp ``learn_A``.
        initial_emission: The symmetric initialization it moved from.
        counted_emission: The shipped count-based emission.
        gold_emission: P4's emission learned from true phone one-hot beliefs.
        learned_accuracy: Held-out filtering accuracy of the learned agent.
        fixed_accuracy: Held-out filtering accuracy of the fixed agent.
        oracle_learned_accuracy: Oracle re-derivation of ``learned_accuracy``.
        oracle_fixed_accuracy: Oracle re-derivation of ``fixed_accuracy``.
        oracle_gold_lane_accuracy: Oracle accuracy of add-1 counting on the
            learning lane alone (isolates the lane-size contribution).
        max_movement: ``max |learned - initial|`` (P1 non-vacuity).
        max_count_deviation: ``max |learned - counted_full|`` (P2, measured,
            not gated).
        gold_max_abs_delta: ``max |gold - counted_lane|`` (P4, must be tiny).
        eval_positions: Held-out positions scored.
        train_pairs: Learning-lane population size.
        mcnemar_learned_only: Positions right only under the learned emission.
        mcnemar_fixed_only: Positions right only under the fixed emission.
        decoder_disagreements: Positions where pymdp and the oracle disagree.
    """

    learned_emission: np.ndarray
    initial_emission: np.ndarray
    counted_emission: np.ndarray
    gold_emission: np.ndarray
    learned_accuracy: float
    fixed_accuracy: float
    oracle_learned_accuracy: float
    oracle_fixed_accuracy: float
    oracle_gold_lane_accuracy: float
    max_movement: float
    max_count_deviation: float
    gold_max_abs_delta: float
    eval_positions: int
    train_pairs: int
    mcnemar_learned_only: int
    mcnemar_fixed_only: int
    decoder_disagreements: int


def _word_observations(word: str) -> list[int]:
    return [ALPHA_TO_IDX[c] for c in word if c in ALPHA_TO_IDX]


def _stride(train_pairs: list, stride: int) -> list:
    """Even stride over the sorted train list — representative, never a prefix."""
    if stride <= 1:
        return train_pairs
    return [pair for i, pair in enumerate(train_pairs) if i % stride == 0]


def _learn_agent(
    train_pairs: list[tuple[str, list[str]]],
    *,
    pseudocount: float,
    gold_phones: bool,
) -> np.ndarray:
    """Run real pymdp ``learn_A`` over the learning lane.

    Args:
        train_pairs: The learning population.
        pseudocount: Per-cell Dirichlet concentration for ``pA``.
        gold_phones: If True, supply the true phone one-hot beliefs (the P4
            observed-state control); otherwise use the agent's own filtered
            posterior (the latent-state lane).

    Returns:
        The posterior-mean emission ``(N_OBS, N_PHONES)``, float64 view.
    """
    if pseudocount <= 0.0:
        raise ValueError(f"pseudocount must be positive, got {pseudocount}")
    initial = np.full((N_OBS, N_PHONES), 1.0 / N_OBS, dtype=np.float32)
    transition = build_g2p_transition(train_pairs)
    start = build_g2p_start(train_pairs)

    import pymdp.agent

    agent = pymdp.agent.Agent(
        A=[jnp.asarray(initial)[None, ...]],
        B=[transition],
        C=[jnp.zeros((1, N_PHONES))],
        D=[start],
        pA=[jnp.full((1, N_OBS, N_PHONES), pseudocount, dtype=jnp.float32)],
        learn_A=True,
        learn_B=False,
        num_controls=[1],
        batch_size=1,
    )
    for word, phones in train_pairs:
        observations = _word_observations(word)
        if not observations:
            continue
        if gold_phones:
            beliefs = [
                jnp.eye(N_PHONES, dtype=jnp.float32)[PHONE_TO_IDX[p]][None, None, :] for p in phones
            ]
            sequence = jnp.concatenate(beliefs, axis=1)
        else:
            prior = [start]
            filtered_beliefs: list[jnp.ndarray] = []
            for obs in observations:
                posterior = agent.infer_states([jnp.asarray([obs])], prior)
                filtered_beliefs.append(posterior[0])
                prior = agent.update_empirical_prior(jnp.zeros((1, 1), dtype=jnp.int32), posterior)
            sequence = jnp.stack(filtered_beliefs, axis=1)
        agent = agent.infer_parameters(
            beliefs_A=[sequence],
            observations=[jnp.asarray([observations])],
            actions=None,
            lr_pA=1.0,
        )
    return np.asarray(agent.A[0])[0].astype(np.float64)


def _oracle_decode_positions(
    emission: np.ndarray,
    transition: np.ndarray,
    start: np.ndarray,
    pairs: list[tuple[str, list[str]]],
) -> list[bool]:
    """Position-wise decoding by the independent numpy forward filter."""
    positions: list[bool] = []
    for word, phones in pairs:
        obs = _word_observations(word)
        if len(obs) != len(phones):
            continue
        posterior, _ = forward_filter(emission, transition, start, obs)
        predicted = np.argmax(posterior, axis=1)
        gold = np.asarray([PHONE_TO_IDX[p] for p in phones])
        positions.extend((predicted == gold).tolist())
    if not positions:
        raise ValueError("no aligned evaluation positions")
    return positions


def run_g2p_emission_learning(
    train_pairs: list[tuple[str, list[str]]],
    test_pairs: list[tuple[str, list[str]]],
    *,
    pseudocount: float = 1.0,
    learning_stride: int = 1,
) -> G2PLearningResult:
    """Measure latent-state emission learning and its held-out decoder effect.

    Args:
        train_pairs: The full G2P train split.
        test_pairs: The full held-out test split.
        pseudocount: ``pA`` concentration (matches the shipped add-1 counting
            so the P4 control is a true identity test).
        learning_stride: Bounded even-stride subsample of the train split for
            the learning lane (runtime). The evaluation lane is always the
            full test split.

    Returns:
        The measured result.

    Raises:
        ValueError: If either split is empty, or ``pseudocount`` is not
            positive.
    """
    if not train_pairs or not test_pairs:
        raise ValueError("train and test pairs must both be non-empty")
    lane = _stride(train_pairs, learning_stride)
    transition_jax = build_g2p_transition(train_pairs)
    start_jax = build_g2p_start(train_pairs)
    transition = np.asarray(transition_jax, dtype=np.float64)[0, :, :, 0]
    start = np.asarray(start_jax, dtype=np.float64)[0]

    counted_lane = np.asarray(build_g2p_emission(lane), dtype=np.float64)[0]
    initial = np.full((N_OBS, N_PHONES), 1.0 / N_OBS, dtype=np.float64)
    learned = _learn_agent(lane, pseudocount=pseudocount, gold_phones=False)
    gold = _learn_agent(lane, pseudocount=pseudocount, gold_phones=True)

    learned_agent = build_g2p_agent(jnp.asarray(learned[None, ...]), transition_jax, start_jax)
    fixed_agent = build_g2p_agent(jnp.asarray(counted_lane[None, ...]), transition_jax, start_jax)
    learned_positions = decode_positions(learned_agent, test_pairs)
    fixed_positions = decode_positions(fixed_agent, test_pairs)
    learned_accuracy = sum(learned_positions) / len(learned_positions)
    fixed_accuracy = sum(fixed_positions) / len(fixed_positions)
    oracle_learned = _oracle_decode_positions(learned, transition, start, test_pairs)
    oracle_fixed = _oracle_decode_positions(counted_lane, transition, start, test_pairs)
    oracle_gold_lane = _oracle_decode_positions(counted_lane, transition, start, test_pairs)
    disagreements = sum(
        a != b
        for a, b in zip(
            learned_positions + fixed_positions,
            oracle_learned + oracle_fixed,
            strict=True,
        )
    )
    return G2PLearningResult(
        learned_emission=learned,
        initial_emission=initial,
        counted_emission=counted_lane,
        gold_emission=gold,
        learned_accuracy=learned_accuracy,
        fixed_accuracy=fixed_accuracy,
        oracle_learned_accuracy=sum(oracle_learned) / len(oracle_learned),
        oracle_fixed_accuracy=sum(oracle_fixed) / len(oracle_fixed),
        oracle_gold_lane_accuracy=sum(oracle_gold_lane) / len(oracle_gold_lane),
        max_movement=float(np.max(np.abs(learned - initial))),
        max_count_deviation=float(np.max(np.abs(learned - counted_lane))),
        gold_max_abs_delta=float(np.max(np.abs(gold - counted_lane))),
        eval_positions=len(learned_positions),
        train_pairs=len(lane),
        mcnemar_learned_only=sum(
            a and not b for a, b in zip(learned_positions, fixed_positions, strict=True)
        ),
        mcnemar_fixed_only=sum(
            b and not a for a, b in zip(learned_positions, fixed_positions, strict=True)
        ),
        decoder_disagreements=disagreements,
    )
