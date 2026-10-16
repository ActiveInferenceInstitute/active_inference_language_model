"""Closed-form tests for the pure-numpy HMM oracle.

Doctrine for this file: **every expected value is derived analytically or by
hand arithmetic written out in a comment.** No number here was copied from a
run of the code under test. That is the whole point — the oracle exists to
disagree with pymdp, so it cannot be calibrated against anything but a closed
form.

Analytic anchors used:
  A1  uniform transition + uniform start + identity emission over n symbols
      => perplexity is exactly n, for ANY observation sequence
  A2  a deterministic chain observed along its own trajectory => perplexity 1
  A3  a 2x2 example whose forward pass is worked out with pencil arithmetic
  A4  transition orientation: predictive mass lands on the column, not the row
  A5  smoothing == filtering at t = T-1, and must DIFFER earlier (non-vacuity)
  A6  Viterbi is the joint-MAP path, so its joint log-prob dominates the
      greedy/filtered path's — with a case where they genuinely differ
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from ailm.control import ControlHarness
from ailm.verification.oracle import (
    forward_backward,
    forward_filter,
    identity_emission,
    letter_bigram_perplexity,
    markov_chain_perplexity,
    sequence_perplexity,
    viterbi,
)

# ── ANALYTIC HELPERS (independent of the module under test) ───────────────────


def cyclic_shift(n: int) -> np.ndarray:
    """Return the deterministic shift transition ``i -> i+1 mod n``.

    ``P[i, j] = 1`` iff ``i == (j + 1) % n``, so column ``j`` is a point mass on
    the successor of ``j``. Doubly stochastic, and NOT symmetric for ``n >= 3``,
    which is what makes it usable as an orientation probe.
    """
    mat = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        mat[(j + 1) % n, j] = 1.0
    return mat


def smoothed_shift(n: int, weight: float) -> np.ndarray:
    """Cyclic shift mixed with the uniform distribution.

    ``weight * shift + (1 - weight) / n``. Still column-stochastic (the two
    parts contribute ``weight`` and ``1 - weight`` per column) and still
    doubly stochastic, so an accidentally TRANSPOSED implementation passes
    validation and must be caught numerically instead.
    """
    return weight * cyclic_shift(n) + (1.0 - weight) / n


def path_log_joint(
    emission: np.ndarray,
    transition: np.ndarray,
    start: np.ndarray,
    obs: list[int],
    path: list[int],
) -> float:
    """log P(path, obs) from the generative definition, computed independently.

    ``log start[s0] + log emission[o0, s0]
      + sum_t (log transition[s_t, s_{t-1}] + log emission[o_t, s_t])``
    """
    total = math.log(start[path[0]]) + math.log(emission[obs[0], path[0]])
    for t in range(1, len(obs)):
        total += math.log(transition[path[t], path[t - 1]])
        total += math.log(emission[obs[t], path[t]])
    return total


# ── THE HAND-WORKED 2-STATE EXAMPLE (anchor A3) ───────────────────────────────
#
# emission[i, j] = P(obs = i | state = j)
#     emission = [[0.8, 0.3],
#                 [0.2, 0.7]]        columns sum to 1.0 and 1.0
#
# transition[i, j] = P(next = i | cur = j)
#     transition = [[0.7, 0.4],
#                   [0.3, 0.6]]      columns sum to 1.0 and 1.0
#
# start = [0.6, 0.4]
# obs   = [0, 1]
#
# t = 0
#   predictive     = start                     = [0.6, 0.4]
#   likelihood     = emission[0, :]            = [0.8, 0.3]
#   unnormalized   = [0.6*0.8, 0.4*0.3]        = [0.48, 0.12]
#   Z_0            = 0.48 + 0.12               = 0.60
#   posterior_0    = [0.48/0.60, 0.12/0.60]    = [0.8, 0.2]
#
# t = 1
#   predictive     = transition @ [0.8, 0.2]
#                  = [0.7*0.8 + 0.4*0.2, 0.3*0.8 + 0.6*0.2]
#                  = [0.56 + 0.08,       0.24 + 0.12]        = [0.64, 0.36]
#   likelihood     = emission[1, :]            = [0.2, 0.7]
#   unnormalized   = [0.64*0.2, 0.36*0.7]      = [0.128, 0.252]
#   Z_1            = 0.128 + 0.252             = 0.380
#   posterior_1    = [0.128/0.38, 0.252/0.38]  = [32/95, 63/95]
#
# log_evidence = log(0.60) + log(0.38) = log(0.228)
# perplexity   = exp(-log(0.228)/2)   = 0.228 ** -0.5
#
# Cross-check by brute-force marginalization over both state paths:
#   j0=0: 0.6*0.8 = 0.48; future = T[0,0]*e[1,0] + T[1,0]*e[1,1]
#                                = 0.7*0.2 + 0.3*0.7 = 0.14 + 0.21 = 0.35
#         contribution = 0.48 * 0.35 = 0.168
#   j0=1: 0.4*0.3 = 0.12; future = T[0,1]*e[1,0] + T[1,1]*e[1,1]
#                                = 0.4*0.2 + 0.6*0.7 = 0.08 + 0.42 = 0.50
#         contribution = 0.12 * 0.50 = 0.060
#   total = 0.168 + 0.060 = 0.228   <-- matches Z_0 * Z_1 = 0.60 * 0.38
#
# Smoothed posterior at t = 0 (this is what the backward pass must produce):
#   P(state_0 = 0 | obs) = 0.168 / 0.228 = 14/19
#   P(state_0 = 1 | obs) = 0.060 / 0.228 =  5/19
# Note 14/19 = 0.736842... != the FILTERED 0.8, so smoothing is not a no-op.

HAND_EMISSION = np.array([[0.8, 0.3], [0.2, 0.7]], dtype=np.float64)
HAND_TRANSITION = np.array([[0.7, 0.4], [0.3, 0.6]], dtype=np.float64)
HAND_START = np.array([0.6, 0.4], dtype=np.float64)
HAND_OBS = [0, 1]
HAND_Z0 = 0.60
HAND_Z1 = 0.38
HAND_JOINT = 0.228  # = HAND_Z0 * HAND_Z1, confirmed by brute-force above


# ── 1. ANCHOR A1: uniform gives perplexity exactly n ─────────────────────────


@pytest.mark.parametrize("n", [2, 5, 30, 39])
def test_uniform_chain_perplexity_is_exactly_n(n: int) -> None:
    """Uniform transition + uniform start + identity emission => perplexity n.

    Analytic: with an identity emission the posterior collapses onto the
    observed symbol, and a uniform transition pushes it straight back to
    1/n on every state. So every one-step predictive probability is exactly
    1/n, giving log-evidence -T*log(n) and perplexity exp(log n) = n,
    independent of the observation sequence.
    """
    transition = np.full((n, n), 1.0 / n, dtype=np.float64)
    start = np.full(n, 1.0 / n, dtype=np.float64)
    rng = np.random.default_rng(20260729)
    obs = rng.integers(0, n, size=97).tolist()

    ppl = sequence_perplexity(identity_emission(n), transition, start, obs)
    assert ppl == pytest.approx(float(n), abs=1e-9)

    # And the log-evidence is exactly -T * log(n).
    _, log_evidence = forward_filter(identity_emission(n), transition, start, obs)
    assert log_evidence == pytest.approx(-len(obs) * math.log(n), abs=1e-9)


def test_uniform_perplexity_independent_of_sequence() -> None:
    """The n-anchor holds for every sequence, not just one seeded draw."""
    n = 7
    transition = np.full((n, n), 1.0 / n)
    start = np.full(n, 1.0 / n)
    emission = identity_emission(n)
    for obs in ([0], [0, 0, 0, 0], [6, 5, 4, 3, 2, 1, 0], list(range(n)) * 3):
        assert sequence_perplexity(emission, transition, start, obs) == pytest.approx(
            float(n), abs=1e-9
        )


# ── 2. ANCHOR A2: deterministic chain gives perplexity 1 ─────────────────────


@pytest.mark.parametrize("n", [2, 5, 39])
def test_deterministic_chain_perplexity_is_one(n: int) -> None:
    """i -> i+1 mod n with probability 1, observed along its own trajectory.

    Analytic: start is a point mass on state 0 and the emission is the
    identity, so Z_0 = 1. Each predictive is a point mass on the true next
    state, so Z_t = 1 for every t. log-evidence = 0, perplexity = exp(0) = 1.
    """
    transition = cyclic_shift(n)
    start = np.zeros(n)
    start[0] = 1.0
    obs = [(t % n) for t in range(3 * n)]

    posteriors, log_evidence = forward_filter(identity_emission(n), transition, start, obs)
    assert log_evidence == pytest.approx(0.0, abs=1e-12)
    assert sequence_perplexity(identity_emission(n), transition, start, obs) == pytest.approx(
        1.0, abs=1e-12
    )

    # Every posterior is a point mass on the observed state.
    for t, o in enumerate(obs):
        expected = np.zeros(n)
        expected[o] = 1.0
        np.testing.assert_allclose(posteriors[t], expected, atol=1e-12)


# ── 3. ANCHOR A3: the hand-computed 2-state example ──────────────────────────


def test_hand_computed_forward_pass() -> None:
    """Pencil-arithmetic anchor: catches an indexing transpose or a lost step.

    Expected values are derived in the comment block above HAND_EMISSION.
    """
    posteriors, log_evidence = forward_filter(HAND_EMISSION, HAND_TRANSITION, HAND_START, HAND_OBS)

    np.testing.assert_allclose(posteriors[0], [0.8, 0.2], atol=1e-12)
    np.testing.assert_allclose(posteriors[1], [32 / 95, 63 / 95], atol=1e-12)
    assert log_evidence == pytest.approx(math.log(HAND_Z0) + math.log(HAND_Z1), abs=1e-12)
    # Same quantity written a second way: the brute-force joint over paths.
    assert math.exp(log_evidence) == pytest.approx(HAND_JOINT, abs=1e-12)

    ppl = sequence_perplexity(HAND_EMISSION, HAND_TRANSITION, HAND_START, HAND_OBS)
    assert ppl == pytest.approx(HAND_JOINT**-0.5, abs=1e-12)


def test_hand_computed_evidence_equals_brute_force_marginal() -> None:
    """The forward evidence must equal an explicit sum over all state paths.

    This is a second independent derivation of log_evidence, written from the
    generative definition rather than from the recursion.
    """
    n_states = HAND_START.shape[0]
    total = 0.0
    for s0 in range(n_states):
        for s1 in range(n_states):
            total += (
                HAND_START[s0]
                * HAND_EMISSION[HAND_OBS[0], s0]
                * HAND_TRANSITION[s1, s0]
                * HAND_EMISSION[HAND_OBS[1], s1]
            )
    assert total == pytest.approx(HAND_JOINT, abs=1e-12)

    _, log_evidence = forward_filter(HAND_EMISSION, HAND_TRANSITION, HAND_START, HAND_OBS)
    assert math.exp(log_evidence) == pytest.approx(total, abs=1e-12)


def test_hand_computed_smoothed_posterior() -> None:
    """Smoothed t=0 belief is 14/19, hand-derived from the path decomposition."""
    smoothed = forward_backward(HAND_EMISSION, HAND_TRANSITION, HAND_START, HAND_OBS)
    np.testing.assert_allclose(smoothed[0], [14 / 19, 5 / 19], atol=1e-12)
    # t = T-1 has no future evidence, so it equals the filtered belief.
    np.testing.assert_allclose(smoothed[1], [32 / 95, 63 / 95], atol=1e-12)


# ── 4. ANCHOR A4: transition orientation is pinned ───────────────────────────


def test_transition_orientation_two_state() -> None:
    """P(next=1|cur=0)=0.9 must place 0.9 predictive mass on state 1.

    With an identity emission and start concentrated on state 0, the first
    observation contributes Z_0 = 1 and the second contributes exactly
    Z_1 = transition[obs_1, 0]. So exp(log_evidence) reads the predictive mass
    off directly:
        obs = [0, 1]  ->  transition[1, 0] = 0.9
        obs = [0, 0]  ->  transition[0, 0] = 0.1
    A transposed implementation reads the ROW instead of the column and fails.
    """
    # column j = distribution over the NEXT state given current state j
    transition = np.array([[0.1, 0.1], [0.9, 0.9]], dtype=np.float64)
    assert transition[1, 0] == 0.9  # P(next=1 | cur=0)
    assert transition[0, 1] == 0.1  # P(next=0 | cur=1)
    start = np.array([1.0, 0.0])
    emission = identity_emission(2)

    _, log_ev_to_one = forward_filter(emission, transition, start, [0, 1])
    assert math.exp(log_ev_to_one) == pytest.approx(0.9, abs=1e-12)

    _, log_ev_to_zero = forward_filter(emission, transition, start, [0, 0])
    assert math.exp(log_ev_to_zero) == pytest.approx(0.1, abs=1e-12)

    # And the filtered posterior after obs 1 is a point mass on state 1.
    posteriors, _ = forward_filter(emission, transition, start, [0, 1])
    np.testing.assert_allclose(posteriors[1], [0.0, 1.0], atol=1e-12)


def test_transition_orientation_survives_doubly_stochastic_transpose() -> None:
    """A transpose that still VALIDATES must still be caught numerically.

    Every 2x2 doubly stochastic matrix is symmetric, so the two-state probe
    above catches a transpose only because the transposed matrix stops being
    column-stochastic and raises. This 3-state probe closes that gap: a
    smoothed cyclic shift is doubly stochastic AND asymmetric, so its transpose
    passes validation and must be distinguished by the numbers alone.
    """
    n, weight = 3, 0.8
    transition = smoothed_shift(n, weight)
    np.testing.assert_allclose(transition.sum(axis=0), np.ones(n), atol=1e-12)
    np.testing.assert_allclose(transition.sum(axis=1), np.ones(n), atol=1e-12)
    assert not np.allclose(transition, transition.T)

    start = np.zeros(n)
    start[0] = 1.0
    emission = identity_emission(n)

    # Analytic: Z_1 = transition[1, 0] = weight + (1-weight)/n
    forward_mass = weight + (1.0 - weight) / n
    # A transposed implementation would instead report transition[0, 1] = (1-weight)/n
    transposed_mass = (1.0 - weight) / n
    assert forward_mass != pytest.approx(transposed_mass)

    _, log_evidence = forward_filter(emission, transition, start, [0, 1])
    assert math.exp(log_evidence) == pytest.approx(forward_mass, abs=1e-12)

    # Positive/negative control: feeding the TRANSPOSE yields the wrong number,
    # proving this probe is live rather than accidentally satisfied.
    _, log_ev_transposed = forward_filter(emission, transition.T, start, [0, 1])
    assert math.exp(log_ev_transposed) == pytest.approx(transposed_mass, abs=1e-12)


# ── 5 & 6. ANCHOR A5: smoothing agrees at the end, differs earlier ───────────


def test_smoothing_equals_filtering_at_final_timestep() -> None:
    """No future evidence exists at t = T-1, so beta is uniform and cancels."""
    n = 4
    transition = smoothed_shift(n, 0.7)
    emission = 0.6 * identity_emission(n) + 0.4 / n
    start = np.array([0.4, 0.3, 0.2, 0.1])
    obs = [0, 1, 2, 3, 0, 1, 2]

    filtered, _ = forward_filter(emission, transition, start, obs)
    smoothed = forward_backward(emission, transition, start, obs)
    np.testing.assert_allclose(smoothed[-1], filtered[-1], atol=1e-12)


def test_smoothing_differs_from_filtering_when_future_is_informative() -> None:
    """NON-VACUITY CONTROL for the backward pass.

    If the backward recursion were a no-op (all-ones betas), smoothing would
    equal filtering everywhere. Here the future evidence is decisive: the first
    observation is the uninformative symbol 3, and the later observations pin
    the state exactly, so a near-deterministic chain implies the initial state
    by running backwards. Smoothing must move t=0; filtering cannot.
    """
    # emission: (4 obs, 3 states). Symbol s identifies state s with prob 0.7;
    # symbol 3 is emitted with prob 0.3 by every state -> carries no information.
    emission = np.array(
        [
            [0.7, 0.0, 0.0],
            [0.0, 0.7, 0.0],
            [0.0, 0.0, 0.7],
            [0.3, 0.3, 0.3],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(emission.sum(axis=0), np.ones(3), atol=1e-12)
    transition = smoothed_shift(3, 0.98)
    start = np.array([0.5, 0.3, 0.2])
    obs = [3, 2, 0]

    filtered, _ = forward_filter(emission, transition, start, obs)
    smoothed = forward_backward(emission, transition, start, obs)

    # Analytic: obs 3 at t=0 is uninformative, so filtering just returns start.
    np.testing.assert_allclose(filtered[0], start, atol=1e-12)
    # Backwards, obs 2 then obs 0 means state 2 then state 0, and the shift
    # 1 -> 2 -> 0 forces state_0 = 1. Smoothing must concentrate there.
    assert int(np.argmax(smoothed[0])) == 1
    assert smoothed[0][1] > 0.9
    assert not np.allclose(smoothed[0], filtered[0], atol=1e-3)

    # Every smoothed row is still a distribution.
    np.testing.assert_allclose(smoothed.sum(axis=1), np.ones(len(obs)), atol=1e-12)


# ── 7 & 8. ANCHOR A6: Viterbi ────────────────────────────────────────────────


def test_viterbi_recovers_planted_path() -> None:
    """Near-deterministic chain, near-identity emission => the planted path.

    Analytic: the transition puts 0.97 + 0.03/5 on the successor and the
    emission puts 0.9 + 0.1/5 on the matching symbol, so the observed symbol
    sequence 0,1,2,3,4,0,... has a uniquely dominant explanation: the states
    equal the observations.
    """
    n = 5
    transition = smoothed_shift(n, 0.97)
    emission = 0.9 * identity_emission(n) + 0.1 / n
    start = np.zeros(n)
    start[0] = 1.0 - 1e-6
    start += 1e-6 / n
    start /= start.sum()
    planted = [(t % n) for t in range(3 * n)]

    assert viterbi(emission, transition, start, planted) == planted


def test_viterbi_dominates_greedy_filtered_path() -> None:
    """Viterbi is the joint-MAP path, so it dominates the greedy filtered path.

    The comparison is made live (not tautological) by a model where the two
    paths genuinely differ: the first symbol is uninformative, so greedy
    filtering commits to the a-priori-likeliest state 0, while the MAP path
    reads the near-deterministic shift backwards from the later evidence and
    starts at state 1 instead.

        greedy  -> [0, 2, 0]   requires the improbable jump 0 -> 2
        viterbi -> [1, 2, 0]   rides the shift 1 -> 2 -> 0
    """
    emission = np.array(
        [
            [0.7, 0.0, 0.0],
            [0.0, 0.7, 0.0],
            [0.0, 0.0, 0.7],
            [0.3, 0.3, 0.3],
        ],
        dtype=np.float64,
    )
    transition = smoothed_shift(3, 0.98)
    start = np.array([0.5, 0.3, 0.2])
    obs = [3, 2, 0]

    filtered, _ = forward_filter(emission, transition, start, obs)
    greedy = [int(np.argmax(row)) for row in filtered]
    map_path = viterbi(emission, transition, start, obs)

    assert greedy == [0, 2, 0]
    assert map_path == [1, 2, 0]
    assert map_path != greedy  # the >= assertion below is not comparing a path to itself

    lp_map = path_log_joint(emission, transition, start, obs, map_path)
    lp_greedy = path_log_joint(emission, transition, start, obs, greedy)
    assert lp_map >= lp_greedy
    assert lp_map > lp_greedy  # strict here: the paths differ


def test_viterbi_dominates_every_path_on_a_small_model() -> None:
    """Exhaustive control: on a 3-state, 4-step model, Viterbi beats all 81 paths."""
    import itertools

    n = 3
    transition = smoothed_shift(n, 0.75)
    emission = 0.55 * identity_emission(n) + 0.45 / n
    start = np.array([0.2, 0.5, 0.3])
    obs = [1, 2, 0, 0]

    map_path = viterbi(emission, transition, start, obs)
    lp_map = path_log_joint(emission, transition, start, obs, map_path)

    best_brute = max(
        path_log_joint(emission, transition, start, obs, list(path))
        for path in itertools.product(range(n), repeat=len(obs))
    )
    assert lp_map == pytest.approx(best_brute, abs=1e-12)


def test_viterbi_dominates_filtered_path_on_a_noisy_sweep() -> None:
    """The MAP-path property must hold on many seeded models, not just one."""
    n = 6
    for seed in range(12):
        rng = np.random.default_rng(seed)
        transition = rng.dirichlet(np.ones(n), size=n).T  # columns sum to 1
        emission = rng.dirichlet(np.ones(n), size=n).T
        start = rng.dirichlet(np.ones(n))
        obs = rng.integers(0, n, size=25).tolist()

        filtered, _ = forward_filter(emission, transition, start, obs)
        greedy = [int(np.argmax(row)) for row in filtered]
        map_path = viterbi(emission, transition, start, obs)

        lp_map = path_log_joint(emission, transition, start, obs, map_path)
        lp_greedy = path_log_joint(emission, transition, start, obs, greedy)
        assert lp_map >= lp_greedy - 1e-12, f"seed {seed}: Viterbi lost to greedy"


# ── 9. markov_chain_perplexity is the identity-emission special case ─────────


def test_markov_chain_perplexity_matches_identity_emission() -> None:
    """The convenience wrapper must be exactly the identity-emission call."""
    n = 8
    rng = np.random.default_rng(4)
    transition = rng.dirichlet(np.ones(n), size=n).T
    start = rng.dirichlet(np.ones(n))
    obs = rng.integers(0, n, size=200).tolist()

    wrapped = markov_chain_perplexity(transition, start, obs)
    explicit = sequence_perplexity(identity_emission(n), transition, start, obs)
    assert wrapped == pytest.approx(explicit, rel=0, abs=0)


def test_markov_chain_perplexity_uniform_anchor() -> None:
    """Uniform chain observed directly => perplexity n, via the wrapper."""
    for n in (2, 5, 39):
        transition = np.full((n, n), 1.0 / n)
        start = np.full(n, 1.0 / n)
        obs = [(3 * t) % n for t in range(50)]
        assert markov_chain_perplexity(transition, start, obs) == pytest.approx(float(n), abs=1e-9)


# ── 10. INPUT VALIDATION ─────────────────────────────────────────────────────


def _good(n: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """A valid (emission, transition, start, obs) bundle."""
    return (
        identity_emission(n),
        smoothed_shift(n, 0.8),
        np.full(n, 1.0 / n),
        [0, 1, 2],
    )


ENTRY_POINTS = [forward_filter, forward_backward, viterbi, sequence_perplexity]


@pytest.mark.parametrize("fn", ENTRY_POINTS)
def test_non_stochastic_transition_raises(fn) -> None:
    emission, _, start, obs = _good()
    bad = np.array([[0.5, 0.4, 0.0], [0.3, 0.4, 0.5], [0.1, 0.2, 0.5]])  # col 0 sums 0.9
    with pytest.raises(ValueError, match="transition is not column-stochastic"):
        fn(emission, bad, start, obs)


@pytest.mark.parametrize("fn", ENTRY_POINTS)
def test_non_stochastic_emission_raises(fn) -> None:
    _, transition, start, obs = _good()
    bad = np.array([[0.9, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 0.9]])  # cols sum 0.9
    with pytest.raises(ValueError, match="emission is not column-stochastic"):
        fn(bad, transition, start, obs)


@pytest.mark.parametrize("fn", ENTRY_POINTS)
def test_bad_start_raises(fn) -> None:
    emission, transition, _, obs = _good()
    with pytest.raises(ValueError, match="start must sum to 1"):
        fn(emission, transition, np.array([0.5, 0.2, 0.2]), obs)


@pytest.mark.parametrize("fn", ENTRY_POINTS)
def test_out_of_range_obs_raises(fn) -> None:
    emission, transition, start, _ = _good()
    with pytest.raises(ValueError, match="obs_indices out of range"):
        fn(emission, transition, start, [0, 1, 3])
    with pytest.raises(ValueError, match="obs_indices out of range"):
        fn(emission, transition, start, [-1, 0])


@pytest.mark.parametrize("fn", ENTRY_POINTS)
def test_empty_obs_raises(fn) -> None:
    emission, transition, start, _ = _good()
    with pytest.raises(ValueError, match="obs_indices must be non-empty"):
        fn(emission, transition, start, [])


def test_negative_probability_raises() -> None:
    emission, transition, start, obs = _good()
    bad = transition.copy()
    bad[0, 0] -= 1.0
    bad[1, 0] += 1.0  # column still sums to 1 but holds a negative entry
    with pytest.raises(ValueError, match="negative probabilities"):
        forward_filter(emission, bad, start, obs)

    bad_start = np.array([-0.1, 0.6, 0.5])
    with pytest.raises(ValueError, match="negative probabilities"):
        forward_filter(emission, transition, bad_start, obs)


def test_non_finite_inputs_raise() -> None:
    emission, transition, start, obs = _good()
    nan_trans = transition.copy()
    nan_trans[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        forward_filter(emission, nan_trans, start, obs)

    nan_start = start.copy()
    nan_start[0] = np.inf
    with pytest.raises(ValueError, match="NaN or infinite"):
        forward_filter(emission, transition, nan_start, obs)


def test_shape_errors_raise() -> None:
    emission, transition, start, obs = _good()
    with pytest.raises(ValueError, match="must be 2-D"):
        forward_filter(np.ones(3) / 3, transition, start, obs)
    with pytest.raises(ValueError, match="must be non-empty"):
        forward_filter(np.zeros((0, 3)), transition, start, obs)
    with pytest.raises(ValueError, match="must be square with side n_states"):
        forward_filter(np.eye(4), transition, np.full(4, 0.25), [0, 1])
    with pytest.raises(ValueError, match="start must be 1-D"):
        forward_filter(emission, transition, np.full((1, 3), 1 / 3), obs)
    with pytest.raises(ValueError, match="start has length"):
        forward_filter(emission, transition, np.full(5, 0.2), obs)
    with pytest.raises(ValueError, match="obs_indices must be 1-D"):
        forward_filter(emission, transition, start, np.array([[0, 1]]))


def test_non_integral_obs_raises_but_integral_floats_are_accepted() -> None:
    emission, transition, start, _ = _good()
    with pytest.raises(ValueError, match="must contain integers"):
        forward_filter(emission, transition, start, [0.0, 1.5])
    with pytest.raises(ValueError, match="must contain integers"):
        forward_filter(emission, transition, start, [0.0, np.nan])
    # Integral floats are a legitimate carrier for indices.
    posteriors, _ = forward_filter(emission, transition, start, [0.0, 1.0, 2.0])
    assert posteriors.shape == (3, 3)


def test_identity_emission_validation() -> None:
    np.testing.assert_allclose(identity_emission(3), np.eye(3), atol=0)
    assert identity_emission(1).shape == (1, 1)
    with pytest.raises(ValueError, match="n must be >= 1"):
        identity_emission(0)
    with pytest.raises(ValueError, match="n must be an integer"):
        identity_emission(3.0)
    with pytest.raises(ValueError, match="n must be an integer"):
        identity_emission(True)


def test_markov_chain_perplexity_requires_square_transition() -> None:
    # (3, 2) column-stochastic but not square: it cannot be a transition.
    non_square = np.array([[0.5, 0.2], [0.3, 0.3], [0.2, 0.5]])
    np.testing.assert_allclose(non_square.sum(axis=0), np.ones(2), atol=1e-12)
    with pytest.raises(ValueError, match="transition must be square"):
        markov_chain_perplexity(non_square, np.array([0.5, 0.5]), [0, 1])


# ── 11. NUMERICAL STABILITY ──────────────────────────────────────────────────


def test_long_sequence_perplexity_is_finite() -> None:
    """5000 symbols must not underflow. Renormalization is what buys this.

    An unnormalized alpha recursion would drive the joint to ~1e-5000 and
    report 0.0 or nan. Bounds asserted are analytic: perplexity of any model
    over n symbols lies in [1, inf), and for a strictly positive transition
    with a floor of alpha/n per entry it is bounded above by n / (alpha-ish),
    so n is a safe loose ceiling for a Dirichlet-drawn model only in
    expectation — here we assert the hard bounds plus finiteness.
    """
    n = 39
    rng = np.random.default_rng(1234)
    transition = rng.dirichlet(np.ones(n) * 0.5, size=n).T
    start = rng.dirichlet(np.ones(n))
    obs = rng.integers(0, n, size=5000).tolist()

    posteriors, log_evidence = forward_filter(identity_emission(n), transition, start, obs)
    assert math.isfinite(log_evidence)
    assert log_evidence < 0.0
    np.testing.assert_allclose(posteriors.sum(axis=1), np.ones(len(obs)), atol=1e-10)

    ppl = markov_chain_perplexity(transition, start, obs)
    assert math.isfinite(ppl)
    assert ppl >= 1.0

    smoothed = forward_backward(identity_emission(n), transition, start, obs)
    assert np.all(np.isfinite(smoothed))
    np.testing.assert_allclose(smoothed.sum(axis=1), np.ones(len(obs)), atol=1e-10)

    path = viterbi(identity_emission(n), transition, start, obs)
    assert len(path) == len(obs)
    # With an identity emission the observed symbol IS the state, so the MAP
    # path is forced to equal the observations.
    assert path == obs


def test_deterministic_chain_off_trajectory_stays_finite() -> None:
    """A zero-probability observation must floor the log, not produce -inf.

    Analytic: transition 0 -> 1 with probability 1 and an identity emission
    means P(obs_1 = 0 | obs_0 = 0) is exactly 0. The oracle must clip to the
    log floor and keep going rather than returning -inf or nan.
    """
    transition = cyclic_shift(2)
    start = np.array([1.0, 0.0])
    emission = identity_emission(2)
    obs = [0, 0, 1]

    posteriors, log_evidence = forward_filter(emission, transition, start, obs)
    assert math.isfinite(log_evidence)
    assert log_evidence < math.log(1e-299)
    np.testing.assert_allclose(posteriors.sum(axis=1), np.ones(3), atol=1e-12)

    ppl = sequence_perplexity(emission, transition, start, obs)
    assert math.isfinite(ppl)
    assert ppl > 1e90  # a genuinely impossible sequence scores catastrophically

    smoothed = forward_backward(emission, transition, start, obs)
    assert np.all(np.isfinite(smoothed))
    np.testing.assert_allclose(smoothed.sum(axis=1), np.ones(3), atol=1e-12)


def test_impossible_observation_symbol_stays_finite() -> None:
    """An observation symbol no state can emit exercises both zero-mass paths."""
    # Symbol 2 has an all-zero row: no state ever emits it.
    emission = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    np.testing.assert_allclose(emission.sum(axis=0), np.ones(2), atol=1e-12)
    transition = identity_emission(2)
    start = np.array([1.0, 0.0])
    obs = [0, 2, 0]

    posteriors, log_evidence = forward_filter(emission, transition, start, obs)
    assert math.isfinite(log_evidence)
    np.testing.assert_allclose(posteriors.sum(axis=1), np.ones(3), atol=1e-12)

    smoothed = forward_backward(emission, transition, start, obs)
    assert np.all(np.isfinite(smoothed))
    np.testing.assert_allclose(smoothed.sum(axis=1), np.ones(3), atol=1e-12)

    path = viterbi(emission, transition, start, obs)
    assert len(path) == 3


def test_single_observation_sequence() -> None:
    """T = 1 is a boundary: no transition is ever applied."""
    emission, transition, start, _ = _good()
    posteriors, log_evidence = forward_filter(emission, transition, start, [1])
    np.testing.assert_allclose(posteriors[0], [0.0, 1.0, 0.0], atol=1e-12)
    assert log_evidence == pytest.approx(math.log(1 / 3), abs=1e-12)
    np.testing.assert_allclose(
        forward_backward(emission, transition, start, [1]), posteriors, atol=1e-12
    )
    assert viterbi(emission, transition, start, [1]) == [1]
    assert sequence_perplexity(emission, transition, start, [1]) == pytest.approx(3.0, abs=1e-12)


# ── THE REASON THIS MODULE EXISTS: catching anti-informative wiring ──────────


def _bigram_transition(text: str, symbols: str) -> tuple[np.ndarray, np.ndarray]:
    """Laplace-smoothed bigram transition + unigram start, from a symbol string.

    Returns ``(transition, start)`` with ``transition[i, j] = P(next=i | cur=j)``.
    """
    n = len(symbols)
    index = {c: i for i, c in enumerate(symbols)}
    counts = np.ones((n, n), dtype=np.float64)  # add-1
    for prev, curr in pairwise(text):
        counts[index[curr], index[prev]] += 1.0
    transition = counts / counts.sum(axis=0, keepdims=True)
    uni = np.ones(n, dtype=np.float64)
    for c in text:
        uni[index[c]] += 1.0
    return transition, uni / uni.sum()


def test_oracle_detects_stage1_anti_informative_wiring() -> None:
    """POSITIVE + NEGATIVE CONTROL for the oracle's whole purpose.

    Correct wiring puts the learned bigram statistics in the TRANSITION and
    uses an identity emission. The Stage 1 defect put them in the EMISSION with
    an identity transition — which is column-stochastic either way, so pymdp's
    validator accepts both. The oracle must separate them numerically.

    Analytic expectation on a strongly cyclic corpus 'abcdeabcde...':
      * correct wiring   -> the next symbol is nearly determined, so perplexity
                            sits just above 1
      * mis-wired        -> observing symbol c yields a posterior over c's
                            PREDECESSOR, and the identity transition carries
                            that backward-looking belief forward, so the model
                            predicts the symbol two steps back. On a cycle of
                            length 5 that is systematically wrong, so
                            perplexity must exceed the uniform bar of 5.
    """
    symbols = "abcde"
    n = len(symbols)
    text = symbols * 400
    bigram, start = _bigram_transition(text, symbols)
    obs = [symbols.index(c) for c in text]

    correct_ppl = sequence_perplexity(identity_emission(n), bigram, start, obs)
    miswired_ppl = sequence_perplexity(bigram, identity_emission(n), start, obs)
    uniform_ppl = float(n)

    # Both wirings pass a purely structural check — that is the trap.
    np.testing.assert_allclose(bigram.sum(axis=0), np.ones(n), atol=1e-12)

    assert correct_ppl < 1.05, f"correct wiring should be near-deterministic, got {correct_ppl}"
    assert miswired_ppl > uniform_ppl, (
        f"mis-wired model scored {miswired_ppl}, which is not worse than the "
        f"uniform bar {uniform_ppl} — the oracle would not have caught Stage 1"
    )

    harness = ControlHarness("oracle_detects_anti_informative_wiring")
    harness.assert_gate_is_live(
        good_value=correct_ppl,
        bad_value=miswired_ppl,
        direction="lower_is_better",
    )
    # And the gate is not merely ordering two numbers: the mis-wired model is
    # worse than doing nothing at all.
    harness.assert_gate_is_live(
        good_value=uniform_ppl,
        bad_value=miswired_ppl,
        direction="lower_is_better",
    )


def test_context_free_transition_reproduces_unigram_cross_entropy() -> None:
    """A rank-1 transition must give EXACTLY the plain unigram cross-entropy.

    Analytic: if every column of the transition equals the marginal ``m``, then
    ``transition @ q == m`` for any belief ``q``, so every predictive (including
    the one at t=0, which is ``start == m``) equals ``m``. Therefore

        log_evidence = sum_t log m[obs_t]      and      ppl = exp(-mean log m[obs_t])

    with no dependence on the observation order whatsoever. This pins the
    oracle against a closed form and doubles as a rank-deficiency check.
    """
    symbols = "abcde"
    n = len(symbols)
    text = "aabcaabdaabe" * 150
    _, marginal = _bigram_transition(text, symbols)
    obs = [symbols.index(c) for c in text]

    context_free = np.tile(marginal[:, np.newaxis], (1, n))
    np.testing.assert_allclose(context_free.sum(axis=0), np.ones(n), atol=1e-12)

    _, log_evidence = forward_filter(identity_emission(n), context_free, marginal, obs)
    closed_form = float(sum(math.log(marginal[o]) for o in obs))
    assert log_evidence == pytest.approx(closed_form, rel=1e-12)

    ppl = sequence_perplexity(identity_emission(n), context_free, marginal, obs)
    assert ppl == pytest.approx(math.exp(-closed_form / len(obs)), rel=1e-12)

    # Order-independence is the defining property of a context-free model.
    shuffled = list(reversed(obs))
    assert sequence_perplexity(
        identity_emission(n), context_free, marginal, shuffled
    ) == pytest.approx(ppl, rel=1e-12)


def test_oracle_beats_context_free_baseline_on_a_structured_corpus() -> None:
    """Gate against a non-crippled comparator, not against uniform.

    The context-free bar is the unigram model: predict every symbol from its
    marginal frequency, ignoring the previous symbol. On a corpus whose
    marginals are skewed AND whose successor distribution is sharper than its
    marginal, the bigram transition must beat it. Every number is derived from
    the corpus; none is hand-typed.

    The unigram bar is strictly below the uniform bar here only because the
    marginals are non-uniform — that is asserted, not assumed. (A corpus like
    ``"abcde" * k`` has a perfectly flat marginal, so its unigram bar sits AT
    the uniform bar and would be a crippled comparator.)
    """
    symbols = "abcde"
    n = len(symbols)
    text = "aabcaabdaabe" * 150
    bigram, marginal = _bigram_transition(text, symbols)
    obs = [symbols.index(c) for c in text]

    # The marginal must actually be skewed, or the "context-free" comparator
    # degenerates into the uniform one.
    assert not np.allclose(marginal, 1.0 / n, atol=1e-3)

    bigram_ppl = sequence_perplexity(identity_emission(n), bigram, marginal, obs)

    context_free = np.tile(marginal[:, np.newaxis], (1, n))
    unigram_ppl = sequence_perplexity(identity_emission(n), context_free, marginal, obs)

    # Derived, not asserted: a non-uniform marginal has entropy below log n.
    marginal_entropy = float(-(marginal * np.log(marginal)).sum())
    assert marginal_entropy < math.log(n)
    assert unigram_ppl < float(n)

    ControlHarness("oracle_bigram_beats_context_free").assert_gate_is_live(
        good_value=bigram_ppl,
        bad_value=unigram_ppl,
        direction="lower_is_better",
    )


# ── THE STAGE 3 BAR MUST FAIL CLOSED ON THE TOKENS ITS TWIN REJECTS ──────────


def test_letter_bigram_bar_refuses_tokens_that_are_not_lowercase_ascii() -> None:
    """A7: an out-of-range letter must raise, not alias onto another row.

    ``ord(ch) - 97`` maps code points 70..96 to -27..-1, and -27 is a perfectly
    valid index into a 27-row array: it is row 0, the row for 'a'. So without a
    guard the bar scores ``'aF'`` exactly like ``'aa'`` and reports it as a
    measurement, while ``word_path_logprob`` — the model side of the same
    comparison — raises on that token. The aliasing is asserted here rather than
    described, so the reason for the guard lives in the test.
    """
    n = 27
    transition = np.full((n, n), 1.0 / n)
    start = np.full(n, 1.0 / n)

    # The aliasing that makes the guard necessary.
    assert ord("F") - 97 == -27
    assert np.arange(n)[ord("F") - 97] == np.arange(n)[ord("a") - 97] == 0

    # A1 on this bar: uniform everywhere gives perplexity exactly n.
    clean, emissions = letter_bigram_perplexity(["aa"], transition, start)
    assert clean == pytest.approx(float(n))
    assert emissions == 3

    for token in ("aF", "a`", "A", "ab1", "a b", ""):
        with pytest.raises(ValueError, match="lowercase ASCII"):
            letter_bigram_perplexity([token], transition, start)

    with pytest.raises(ValueError, match="tokens is empty"):
        letter_bigram_perplexity([], transition, start)
