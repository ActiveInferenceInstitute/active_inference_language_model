"""The third degeneracy: action-independent transitions cancel a nonzero epistemic term.

``tests/test_free_energy_decomposition.py`` pins the first two routes to a zero
information-gain term — a deterministic transition, and an uninformative
likelihood. This file pins the third, which is the one the research queue's
epistemic-generation item walked into (decision 78): a hidden state carrying
*real* uncertainty, resolved by a *maximally informative* observation, still
contributes nothing to policy selection when the transitions are
**action-independent**, because both EFE terms are then constants over the
policy space and the softmax cancels constants.

The claim is universal — for ANY emission, belief, preference, and precision,
if ``B[:, :, k]`` does not depend on ``k`` then the per-action spread of both
terms is zero and the policy posterior equals ``softmax(ln E)``, the pure
habit — so the tests reject the general class over randomized instances rather
than the shipped identity-``B`` construction specifically, and a negative
control shows the assertions can fail: an action-dependent transition must
produce a nonzero spread and move the posterior off the habit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ailm.generation import expected_free_energy_terms

#: (n_obs, n_states) shapes for the randomized instances. Non-square on
#: purpose: the word-state construction is 27 observations over thousands of
#: states, and a square-only test would never exercise that asymmetry.
SHAPES: tuple[tuple[int, int], ...] = ((7, 7), (5, 11), (27, 40), (3, 2))


def _random_instance(
    seed: int, n_obs: int, n_states: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """One arbitrary ``(A, belief, C, ln E, gamma)`` drawn from broad supports."""
    rng = np.random.default_rng(seed)
    emission = rng.dirichlet(np.full(n_obs, 0.4), size=n_states).T
    belief = rng.dirichlet(np.full(n_states, 0.7))
    preference = np.log(rng.dirichlet(np.ones(n_obs)))
    log_habit = np.log(rng.dirichlet(np.ones(n_obs)))
    gamma = float(rng.uniform(0.05, 20.0))
    return emission, belief, preference, log_habit, gamma


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def _action_independent(core: np.ndarray, n_actions: int) -> np.ndarray:
    """``B`` whose every action slice is the same column-stochastic ``core``.

    Stacked as distinct copies rather than a broadcast view, so the assertion
    of exact equality below covers materially identical slices in separate
    memory — the stronger reading of "does not depend on the action".
    """
    return np.stack([core.copy() for _ in range(n_actions)], axis=2)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda pair: f"{pair[0]}x{pair[1]}")
@pytest.mark.parametrize("seed", range(6))
def test_action_independent_transitions_cancel_out_of_policy_selection(shape, seed) -> None:
    """THE CLASS, not the instance: any action-independent B is policy-inert.

    The transition core is a *random stochastic matrix* — not deterministic,
    not the identity — so the information gain has no reason to vanish, and on
    these draws it does not. What must vanish is its per-action spread, exactly:
    every action induces the same predictive, so the per-action computations
    are the same arithmetic and the ptp is 0.0, not merely small.
    """
    n_obs, n_states = shape
    emission, belief, preference, log_habit, gamma = _random_instance(seed, n_obs, n_states)
    rng = np.random.default_rng(seed + 1000)
    core = rng.dirichlet(np.full(n_states, 0.9), size=n_states).T

    terms = expected_free_energy_terms(
        emission, _action_independent(core, n_obs), preference, state_prior=belief
    )

    assert np.ptp(terms["information_gain"]) == 0.0
    assert np.ptp(terms["pragmatic_value"]) == 0.0
    assert np.ptp(terms["expected_free_energy"]) == 0.0

    posterior = _softmax(-gamma * terms["expected_free_energy"] + log_habit)
    assert np.allclose(posterior, _softmax(log_habit), atol=1e-12), (
        "with both EFE terms constant across actions the softmax must cancel "
        "them, leaving the pure habit — a deviation means the cancellation "
        "argument in [@sec:generation] is wrong"
    )


@pytest.mark.parametrize("seed", range(3))
def test_an_action_dependent_transition_restores_selection(seed) -> None:
    """NEGATIVE CONTROL: the assertions above must be able to fail.

    Mixing each action's slice with a different cyclic shift makes the
    predictive genuinely depend on the action; both spreads must then be
    nonzero and the policy posterior must leave the habit. Without this, a
    ``ptp`` that returned 0.0 unconditionally would pass every case above.
    """
    n_obs, n_states = 7, 9
    emission, belief, preference, log_habit, gamma = _random_instance(seed, n_obs, n_states)
    identity = np.eye(n_states)
    transition = np.empty((n_states, n_states, n_obs))
    for action in range(n_obs):
        transition[:, :, action] = 0.5 * identity + 0.5 * np.roll(
            identity, action % n_states, axis=0
        )

    terms = expected_free_energy_terms(emission, transition, preference, state_prior=belief)
    assert np.ptp(terms["information_gain"]) > 0.0
    assert np.ptp(terms["expected_free_energy"]) > 0.0

    posterior = _softmax(-gamma * terms["expected_free_energy"] + log_habit)
    assert not np.allclose(posterior, _softmax(log_habit), atol=1e-12)


def test_the_recorded_measurement_matches_and_its_control_fired() -> None:
    """The ledger's numbers, and the property that makes them mean something.

    Decision 78's three predictions, checked against what the single-writer
    recorder actually wrote: nonzero information gain at every selected trie
    node, per-action spreads of exactly zero, and a policy posterior on the
    habit to machine precision — with the action-dependent control's spreads
    strictly positive, so the zeros were recorded by machinery that can report
    a nonzero.
    """
    from ailm.claims import load_ledger

    ledger = load_ledger(Path(__file__).resolve().parents[1])

    def value(label: str) -> float:
        entry = ledger.get(label)
        assert entry is not None, f"{label} is not in the ledger"
        return float(entry["value"])

    gains = [value("generation_word_state_ig_empty_nats")] + [
        value(f"generation_word_state_ig_depth{depth}_nats") for depth in (1, 2, 3, 4)
    ]
    assert all(gain > 0.0 for gain in gains), (
        "a selected trie node recorded no information gain; the selection rule "
        "admits only ambiguous nodes, so the third degeneracy would have no "
        "exhibit"
    )
    # The empty prefix resolves more than any deeper node: the whole lexicon's
    # uncertainty exceeds any single subtree's.
    assert value("generation_word_state_ig_empty_nats") == max(gains)

    assert value("generation_word_state_max_ig_spread_nats") == 0.0
    assert value("generation_word_state_max_pragmatic_spread_nats") == 0.0
    assert value("generation_word_state_max_g_spread_nats") == 0.0
    assert value("generation_word_state_policy_habit_max_abs_dev") < 1e-12

    assert value("generation_word_state_control_ig_spread_nats") > 0.0
    assert value("generation_word_state_control_g_spread_nats") > 0.0
