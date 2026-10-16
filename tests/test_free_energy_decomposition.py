"""Where active inference does and does not buy anything in this construction.

[@sec:generation] makes the sharpest active-inference claim in the project: under
the construction these levels use, the expected free energy of [@eq:efe] has an
information-gain term that is *identically zero*, so policy selection reduces
analytically to a preference-weighted softmax over the habit,
``softmax(gamma * C + ln E)``.

That claim used to live in a docstring, asserted. A claim of the form "this
quantity is zero" reads as true whether or not anyone computed it, and a
function that returned zeros unconditionally would satisfy any test that only
checked the shipped case. So the term is computed, and two controls establish
that the zero is a property of the construction rather than of the arithmetic:

* **Stochastic transition, identity likelihood** — the predictive over states is
  no longer a point mass and the observation resolves it completely, so the gain
  must reach its ceiling of ``ln(n_states)``. If this did not fire, the shipped
  measurement would establish nothing.
* **Stochastic transition, flat likelihood** — the predictive is still spread but
  the observation carries nothing, so the gain returns to zero. This
  discriminates *which* condition is doing the work: the shipped generator has a
  maximally informative identity likelihood, so the vanishing is a fact about the
  deterministic transition and not about the observation model. An
  implementation that credited the likelihood would report the right number for
  the wrong reason, and only this control separates the two.

The result is a limitation and a contribution at once. It says the generator's
behavior is fully explained by preference and habit, which is what makes the C
and E ablations interpretable, and it says the distinctively active-inference
machinery contributes nothing here beyond that softmax.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ailm.generation import (
    build_generative_letter_agent,
    efe_policy_posterior,
    expected_free_energy_terms,
)
from ailm.levels.letter import N_SYMBOLS, build_letter_transition, load_brown_counts


def _deterministic_transition(n: int) -> np.ndarray:
    """``B[next, current, action]``: taking action k lands in state k."""
    transition = np.zeros((n, n, n), dtype=np.float64)
    for action in range(n):
        transition[action, :, action] = 1.0
    return transition


@pytest.fixture(scope="module")
def goal() -> np.ndarray:
    rng = np.random.default_rng(20260818)
    return np.log(rng.dirichlet(np.ones(N_SYMBOLS)))


def test_the_shipped_construction_has_no_information_gain(goal) -> None:
    """The claim itself, measured on the shipped shapes."""
    terms = expected_free_energy_terms(
        np.eye(N_SYMBOLS), _deterministic_transition(N_SYMBOLS), goal
    )
    assert np.allclose(terms["information_gain"], 0.0, atol=1e-12), (
        f"the information-gain term is not zero: "
        f"max |gain| = {np.max(np.abs(terms['information_gain'])):.3e}. "
        "[@sec:generation] derives that it must be, so either the derivation or "
        "the construction has changed."
    )
    # And therefore the expected free energy IS the negated preference.
    assert np.allclose(terms["expected_free_energy"], -terms["pragmatic_value"])


def test_a_stochastic_transition_makes_the_term_maximal(goal) -> None:
    """NEGATIVE CONTROL: the term must be able to fire, and here it must saturate.

    With a uniform predictive over states and an identity likelihood, observing
    the outcome resolves the state completely, so the gain is exactly the prior
    entropy ``ln(n_states)``. Landing on the analytic ceiling rather than merely
    somewhere positive is what rules out an implementation that returns a small
    arbitrary number when the inputs change.
    """
    terms = expected_free_energy_terms(
        np.eye(N_SYMBOLS),
        np.full((N_SYMBOLS, N_SYMBOLS, N_SYMBOLS), 1.0 / N_SYMBOLS),
        goal,
    )
    assert np.allclose(terms["information_gain"], math.log(N_SYMBOLS), atol=1e-10), (
        "a fully stochastic transition with an identity likelihood did not yield "
        f"ln({N_SYMBOLS}) = {math.log(N_SYMBOLS):.6f} of information gain, so the "
        "measurement cannot be read as evidence about the shipped zero"
    )


def test_a_flat_likelihood_returns_the_term_to_zero(goal) -> None:
    """DISCRIMINATING CONTROL: which condition drives the vanishing.

    Same spread predictive as above, but an observation that says nothing. The
    gain returns to zero, so a zero on its own does not identify a deterministic
    transition. The shipped construction has *both* a deterministic transition
    and an informative likelihood, and only the first is load-bearing.
    """
    terms = expected_free_energy_terms(
        np.full((N_SYMBOLS, N_SYMBOLS), 1.0 / N_SYMBOLS),
        np.full((N_SYMBOLS, N_SYMBOLS, N_SYMBOLS), 1.0 / N_SYMBOLS),
        goal,
    )
    assert np.allclose(terms["information_gain"], 0.0, atol=1e-12)


def test_a_deterministic_transition_alone_suffices(goal) -> None:
    """The derivation's actual sufficient condition, isolated.

    [@sec:generation] argues the vanishing follows from the transition being
    deterministic, with no appeal to the likelihood. So a deterministic
    transition paired with an *arbitrary* likelihood must also give zero. If
    this failed, the prose would be claiming a weaker premise than the result
    needs.
    """
    rng = np.random.default_rng(7)
    arbitrary = rng.dirichlet(np.ones(N_SYMBOLS), size=N_SYMBOLS).T
    terms = expected_free_energy_terms(arbitrary, _deterministic_transition(N_SYMBOLS), goal)
    assert np.allclose(terms["information_gain"], 0.0, atol=1e-12)


def test_the_reduction_reproduces_the_shipped_policy_posterior(goal) -> None:
    """EQUIVALENCE: the closed form the generator uses IS this decomposition.

    ``efe_policy_posterior`` computes ``softmax(gamma * C + ln E)`` directly. If
    the reduction is right, softmaxing the negated expected free energy from the
    full decomposition, with the same habit prior, must give the same answer.
    A drift between them would mean the generator is using a shortcut the
    manuscript's derivation does not license.
    """
    habit = np.asarray(build_letter_transition(load_brown_counts()))[0]
    gamma = 0.55
    agent = build_generative_letter_agent(habit, gamma=gamma)
    terms = expected_free_energy_terms(
        np.eye(N_SYMBOLS), _deterministic_transition(N_SYMBOLS), goal
    )
    for current in (0, 5, 13, N_SYMBOLS - 1):
        shipped = efe_policy_posterior(agent, current, goal, habit)
        scores = -gamma * terms["expected_free_energy"] + np.log(
            np.maximum(habit[:, current], 1e-300)
        )
        scores -= scores.max()
        derived = np.exp(scores) / np.exp(scores).sum()
        assert np.allclose(shipped, derived, atol=1e-12), (
            f"at current letter {current} the shipped posterior and the one built "
            "from the full decomposition disagree; the analytic reduction in "
            "[@sec:generation] does not describe the generator"
        )


def test_the_recorded_measurement_matches_and_its_control_fired() -> None:
    """The ledger's numbers, and the property that makes them mean something."""
    from pathlib import Path

    from ailm.claims import load_ledger

    ledger = load_ledger(Path(__file__).resolve().parents[1])

    def value(label: str) -> float:
        entry = ledger.get(label)
        assert entry is not None, f"{label} is not in the ledger"
        return float(entry["value"])

    assert value("generation_shipped_information_gain_nats") == 0.0
    assert value("generation_flat_likelihood_control_information_gain_nats") == 0.0
    assert value("generation_stochastic_control_information_gain_nats") == pytest.approx(
        value("generation_max_information_gain_possible_nats"), abs=1e-6
    ), (
        "the stochastic control no longer saturates; a control that does not "
        "reach its analytic ceiling weakens the evidence that the shipped zero "
        "is a property of the construction"
    )
    assert value("generation_max_information_gain_possible_nats") == pytest.approx(
        math.log(N_SYMBOLS), abs=1e-6
    )


@pytest.mark.parametrize(
    "emission_shape,transition_shape",
    [
        ((5, 4), (3, 3, 3)),  # state count disagrees between A and B
        ((4, 4), (4, 5, 4)),  # B is not square in its state axes
        ((4, 4), (4, 4)),  # B has no action axis
    ],
)
def test_shape_mismatches_fail_closed(emission_shape, transition_shape, goal) -> None:
    """A non-square observation space is legal, so only real mismatches are here.

    The first draft of this test asserted that emission ``(5, 4)`` with
    transition ``(4, 4, 4)`` should raise. It should not: five outcomes over four
    states is a perfectly ordinary generative model, and the L2B level is exactly
    that shape. The validator was right and the test was wrong, which is worth
    recording because a fail-closed check that rejects valid models is a worse
    defect than one that is slightly permissive.
    """
    with pytest.raises(ValueError):
        expected_free_energy_terms(
            np.ones(emission_shape) / emission_shape[0],
            np.ones(transition_shape) / transition_shape[0],
            np.zeros(emission_shape[0]),
        )


def test_unnormalized_inputs_fail_closed(goal) -> None:
    bad_emission = np.eye(N_SYMBOLS) * 2.0
    with pytest.raises(ValueError, match="columns must sum to one"):
        expected_free_energy_terms(bad_emission, _deterministic_transition(N_SYMBOLS), goal)


@pytest.mark.parametrize(
    "mutation",
    [
        "signed-emission",
        "nan-emission",
        "signed-transition",
        "inf-transition",
        "signed-prior",
        "nan-prior",
        "nan-preference",
        "inf-preference",
    ],
)
def test_sum_preserving_invalid_and_nonfinite_inputs_fail_closed(mutation) -> None:
    """NEGATIVE CONTROLS: unit sums cannot launder signed or nonfinite mass."""
    n = 4
    emission = np.eye(n, dtype=np.float64)
    transition = _deterministic_transition(n)
    preference = np.zeros(n, dtype=np.float64)
    prior = np.full(n, 1.0 / n, dtype=np.float64)

    if mutation == "signed-emission":
        emission[0, 0], emission[1, 0] = -0.5, 1.5
    elif mutation == "nan-emission":
        emission[0, 0] = np.nan
    elif mutation == "signed-transition":
        transition[0, 0, 0], transition[1, 0, 0] = -0.5, 1.5
    elif mutation == "inf-transition":
        transition[0, 0, 0] = np.inf
    elif mutation == "signed-prior":
        prior[0], prior[1] = 1.2, -0.2
        prior[2:] = 0.0
    elif mutation == "nan-prior":
        prior[0] = np.nan
    elif mutation == "nan-preference":
        preference[0] = np.nan
    else:
        preference[0] = np.inf

    with pytest.raises(ValueError, match="finite|non-negative|sum to one"):
        expected_free_energy_terms(
            emission,
            transition,
            preference,
            state_prior=prior,
        )
