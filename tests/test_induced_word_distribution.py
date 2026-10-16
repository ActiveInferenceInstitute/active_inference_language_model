"""The generator's word distribution, computed, and its controls.

The generation gate's metric was a total-variation distance estimated from three
runs of four hundred words: twelve hundred tokens against a two-thousand-type
target, pooled into one statistic with no spread reported anywhere. A reviewer
observed that the sample was far too small to resolve the differences between
adjacent grid points. It was worse than that. An empirical distribution over
twelve hundred draws is supported on at most twelve hundred types, so its
distance to a two-thousand-type target is inflated *by construction*, and the
recorded operating point was the argmin of that inflated estimate rather than of
the quantity it estimates.

Both quantities are exact. The policy generator's whole state is the emitted
prefix, so a word's probability is the product of the action posteriors along
its own path through the trie. The ancestral baseline is the same path product
through the Brown transition alone. In both cases the finite lexicon's
complement is an explicit out-of-lexicon remainder.

These tests pin the properties a wrong implementation could violate while still
producing plausible numbers: normalization, agreement with the sampler it
replaces, the direction of the sampler's bias, and the survival of the interior
optimum that is the section's actual claim.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from ailm import generation as generation_module
from ailm.generation import (
    GenerationResult,
    ancestral_word_distribution,
    build_generative_letter_agent,
    exact_ancestral_lexicon_total_variation,
    exact_lexicon_total_variation,
    exact_sweep_precision,
    generate_by_policy_selection,
    generate_by_sampling,
    induced_word_distribution,
    lexicon_preference,
    lexicon_total_variation,
    sweep_precision,
)
from ailm.levels.letter import ALPHA_TO_IDX, N_SYMBOLS, build_letter_transition, load_brown_counts
from ailm.levels.word import build_word_prior, load_word_artifact


@pytest.fixture(scope="module")
def setup():
    artifact = load_word_artifact()
    habit = np.asarray(build_letter_transition(load_brown_counts()))[0]
    return artifact, build_word_prior(artifact), habit


class _TinyLexicon:
    vocab = ["a", "b"]


def _one_letter_habit(start: dict[str, float]) -> np.ndarray:
    """A planted process: one start draw, then deterministic termination."""
    habit = np.zeros((N_SYMBOLS, N_SYMBOLS), dtype=np.float64)
    space = ALPHA_TO_IDX[" "]
    habit[space, :] = 1.0
    habit[:, space] = 0.0
    for symbol, probability in start.items():
        habit[ALPHA_TO_IDX[symbol], space] = probability
    assert sum(start.values()) == pytest.approx(1.0)
    return habit


def test_ancestral_distribution_matches_a_planted_closed_form() -> None:
    """Analytic control with both an empty attempt and explicit OOV mass."""
    habit = _one_letter_habit({" ": 0.2, "a": 0.4, "b": 0.2, "z": 0.2})
    probabilities, out_of_lexicon = ancestral_word_distribution(_TinyLexicon(), habit)

    # GenerationResult drops the immediate-space attempt, so the exact result is
    # conditional on the remaining 0.8 mass: a=0.5, b=0.25, z/OOV=0.25.
    assert probabilities == pytest.approx([0.5, 0.25])
    assert out_of_lexicon == pytest.approx(0.25)
    assert float(probabilities.sum()) + out_of_lexicon == pytest.approx(1.0)
    assert exact_ancestral_lexicon_total_variation(
        _TinyLexicon(),
        np.array([0.5, 0.5]),
        habit,
    ) == pytest.approx(0.25)


def test_policy_distribution_matches_a_planted_closed_form() -> None:
    """At gamma two and a balanced habit, q(word) is normalized prior squared."""
    habit = _one_letter_habit({"a": 0.5, "b": 0.5})
    prior = np.array([0.25, 0.75], dtype=np.float64)
    probabilities, out_of_lexicon = induced_word_distribution(
        _TinyLexicon(),
        prior,
        habit,
        gamma=2.0,
    )
    assert probabilities == pytest.approx([0.1, 0.9])
    assert out_of_lexicon == pytest.approx(0.0)


def test_policy_distribution_accepts_a_true_zero_target_type() -> None:
    """Zero target mass is valid and must not be replaced by hidden prior mass."""
    habit = _one_letter_habit({"a": 0.5, "b": 0.5})
    probabilities, out_of_lexicon = induced_word_distribution(
        _TinyLexicon(),
        np.array([0.0, 1.0], dtype=np.float64),
        habit,
        gamma=2.0,
    )
    assert probabilities[0] < 1e-20
    assert probabilities[1] == pytest.approx(1.0)
    assert out_of_lexicon == pytest.approx(0.0)


def test_exact_policy_distribution_is_stable_at_large_finite_precision() -> None:
    habit = _one_letter_habit({"a": 0.5, "b": 0.5})
    probabilities, out_of_lexicon = induced_word_distribution(
        _TinyLexicon(),
        np.array([0.25, 0.75], dtype=np.float64),
        habit,
        gamma=np.finfo(np.float64).max,
    )
    assert probabilities == pytest.approx([0.0, 1.0])
    assert out_of_lexicon == pytest.approx(0.0)


def test_exact_path_probability_obeys_the_sampler_truncation_rule() -> None:
    """A horizon-length word is emitted without a terminal-space draw."""
    habit = _one_letter_habit({"a": 1.0})
    habit[:, ALPHA_TO_IDX["a"]] = 0.0
    habit[ALPHA_TO_IDX["b"], ALPHA_TO_IDX["a"]] = 1.0

    at_horizon, at_horizon_oov = ancestral_word_distribution(
        _TinyLexicon(),
        habit,
        max_word_len=1,
    )
    with_terminal_required, longer_horizon_oov = ancestral_word_distribution(
        _TinyLexicon(),
        habit,
        max_word_len=2,
    )
    assert at_horizon == pytest.approx([1.0, 0.0])
    assert at_horizon_oov == pytest.approx(0.0)
    assert with_terminal_required == pytest.approx([0.0, 0.0])
    assert longer_horizon_oov == pytest.approx(1.0)


def test_seeded_ancestral_sampler_converges_to_the_exact_distribution() -> None:
    """Independent stochastic control for the analytic ancestral recursion."""
    habit = _one_letter_habit({"a": 0.35, "b": 0.65})
    prior = np.array([0.5, 0.5], dtype=np.float64)
    exact_probabilities, exact_oov = ancestral_word_distribution(_TinyLexicon(), habit)
    run = generate_by_sampling(habit, _TinyLexicon(), n_words=20_000, seed=20260823)
    counts = Counter(run.words)
    empirical = np.array([counts[word] / len(run.words) for word in _TinyLexicon.vocab])

    assert exact_probabilities == pytest.approx([0.35, 0.65])
    assert exact_oov == pytest.approx(0.0)
    assert empirical == pytest.approx(exact_probabilities, abs=0.01)
    assert lexicon_total_variation([run], _TinyLexicon(), prior) == pytest.approx(
        exact_ancestral_lexicon_total_variation(_TinyLexicon(), prior, habit),
        abs=0.01,
    )


def test_sampled_sweep_remains_an_explicit_seeded_diagnostic() -> None:
    """Exercise the retained Monte Carlo path without treating it as canonical."""
    habit = _one_letter_habit({"a": 0.35, "b": 0.65})
    rows = sweep_precision(
        _TinyLexicon(),
        np.array([0.5, 0.5], dtype=np.float64),
        habit,
        (0.5, 1.0, 2.0),
        seeds=(17, 29),
        n_words=20,
    )
    assert [row["gamma"] for row in rows] == [0.5, 1.0, 2.0]
    assert all(0.0 <= row["total_variation"] <= 1.0 for row in rows)
    assert all(0.0 <= row["membership"] <= 1.0 for row in rows)
    assert all(row["distinct"] >= 1.0 for row in rows)


def test_flat_policy_preserves_true_zero_habit_support() -> None:
    """A log-habit implementation may not turn impossible actions into epsilon mass."""
    habit = _one_letter_habit({"a": 0.35, "b": 0.65})
    prior = np.array([0.5, 0.5], dtype=np.float64)
    agent = build_generative_letter_agent(habit, gamma=3.0)
    flat = generate_by_policy_selection(
        agent,
        _TinyLexicon(),
        prior,
        habit,
        n_words=500,
        seed=1729,
        flat=True,
    )
    ancestral = generate_by_sampling(
        habit,
        _TinyLexicon(),
        n_words=500,
        seed=1729,
    )
    assert flat.text == ancestral.text


def test_the_induced_distribution_is_a_distribution(setup) -> None:
    """POSITIVE CONTROL: vocabulary mass plus out-of-lexicon mass is one."""
    artifact, prior, habit = setup
    probabilities, out_of_lexicon = induced_word_distribution(artifact, prior, habit, 0.55)
    assert probabilities.shape == (len(artifact.vocab),)
    assert np.all(probabilities >= 0.0)
    assert out_of_lexicon >= 0.0
    assert float(probabilities.sum()) + out_of_lexicon == pytest.approx(1.0, abs=1e-9)


def test_high_precision_is_prefix_greedy_not_necessarily_the_global_prior_mode() -> None:
    """NEGATIVE CONTROL for the high-precision asymptotic's claim scope.

    The globally most probable word begins with ``b``, but the combined mass of
    the two ``a`` words makes ``a`` the locally preferred first branch. Once on
    that branch, ``aa`` wins. High precision therefore concentrates on ``aa``;
    describing the limit as global-MAP selection would be false.
    """

    class TinyLexicon:
        vocab = ["aa", "ab", "ba"]

    prior = np.array([0.30, 0.25, 0.45], dtype=np.float64)
    habit = np.full((N_SYMBOLS, N_SYMBOLS), 1.0 / N_SYMBOLS, dtype=np.float64)
    probabilities, out_of_lexicon = induced_word_distribution(
        TinyLexicon(),
        prior,
        habit,
        gamma=256.0,  # type: ignore[arg-type]
    )

    assert TinyLexicon.vocab[int(np.argmax(prior))] == "ba"
    assert TinyLexicon.vocab[int(np.argmax(probabilities))] == "aa"
    assert probabilities[0] > 1.0 - 1e-12
    assert out_of_lexicon == pytest.approx(0.0)


def test_it_matches_what_the_sampler_actually_emits(setup) -> None:
    """EQUIVALENCE: the computed distribution must describe the shipped generator.

    The recursion is a separate implementation of the same process, so this is
    the check that makes it a computation of the generator's behavior rather
    than of something adjacent to it. The tolerance is the sampling error on the
    frequencies, not on the recursion, which is exact.
    """
    artifact, prior, habit = setup
    gamma = 1.0
    probabilities, _ = induced_word_distribution(artifact, prior, habit, gamma)
    agent = build_generative_letter_agent(habit, gamma=gamma)
    words = [
        word
        for seed in (0, 1, 2, 3, 4, 5)
        for word in generate_by_policy_selection(
            agent, artifact, prior, habit, n_words=400, seed=seed
        ).words
    ]
    counts = Counter(words)
    total = len(words)
    index = {word: i for i, word in enumerate(artifact.vocab)}

    # The three most frequent emitted words, where the sampling error is small
    # enough for the comparison to mean something.
    for word, count in counts.most_common(3):
        assert word in index, f"the generator emitted {word!r}, not in the lexicon"
        empirical = count / total
        exact = float(probabilities[index[word]])
        # Binomial standard error at this n, times four.
        tolerance = 4.0 * (exact * (1.0 - exact) / total) ** 0.5
        assert abs(empirical - exact) < max(tolerance, 0.005), (
            f"{word!r}: sampled {empirical:.4f} vs computed {exact:.4f}; the "
            "recursion does not describe the shipped generator"
        )


def test_the_sampled_distance_is_biased_upward(setup) -> None:
    """NEGATIVE CONTROL for the reason the metric was replaced.

    If the sampled estimator were unbiased, the demotion would be about variance
    alone and a larger sample would have fixed it. It is not: an empirical
    distribution over n draws cannot cover a target with more than n types, so
    the sampled distance exceeds the exact one systematically. That is why the
    fix is to compute rather than to sample more.
    """
    artifact, prior, habit = setup
    gamma = 1.0
    exact = exact_lexicon_total_variation(artifact, prior, habit, gamma)
    agent = build_generative_letter_agent(habit, gamma=gamma)
    runs = [
        generate_by_policy_selection(agent, artifact, prior, habit, n_words=400, seed=seed)
        for seed in (0, 1, 2)
    ]
    sampled = lexicon_total_variation(runs, artifact, prior)
    assert sampled > exact, (
        f"the sampled distance {sampled:.4f} did not exceed the exact "
        f"{exact:.4f}; the coverage argument in [@sec:generation] no longer holds"
    )


def test_the_interior_optimum_survives_exact_computation(setup) -> None:
    """The section's actual claim, which the correction had to preserve.

    Had the U-shape been an artifact too, the generation gate would have lost
    its two-directional failure mode and the section would need rewriting rather
    than re-recording. It did not.
    """
    artifact, prior, habit = setup
    gammas = (0.1, 0.25, 0.5, 0.55, 0.75, 1.0, 2.0, 16.0)
    rows = exact_sweep_precision(artifact, prior, habit, gammas)
    distances = [row["total_variation"] for row in rows]
    best = min(range(len(distances)), key=distances.__getitem__)
    assert 0 < best < len(distances) - 1, "the optimum moved to an endpoint"
    assert distances[0] > distances[best] < distances[-1]


def test_the_recorded_operating_point_is_the_exact_argmin(setup) -> None:
    """The ledger's gamma must be the one the exact sweep selects."""
    from pathlib import Path

    from ailm.claims import load_ledger

    ledger = load_ledger(Path(__file__).resolve().parents[1])
    entry = ledger.get("generation_best_gamma")
    assert entry is not None
    recorded = float(entry["value"])

    artifact, prior, habit = setup
    around = (recorded - 0.1, recorded - 0.05, recorded, recorded + 0.05, recorded + 0.1)
    rows = exact_sweep_precision(artifact, prior, habit, tuple(g for g in around if g > 0.0))
    best = min(rows, key=lambda row: row["total_variation"])
    assert best["gamma"] == pytest.approx(recorded), (
        f"the ledger records gamma={recorded} but the exact sweep prefers "
        f"{best['gamma']} nearby; the recorder and the manuscript disagree"
    )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_a_degenerate_precision_fails_closed(setup, bad) -> None:
    artifact, prior, habit = setup
    with pytest.raises(ValueError):
        induced_word_distribution(artifact, prior, habit, bad)


def test_a_sweep_too_small_to_show_a_shape_is_refused(setup) -> None:
    """The exact sweep keeps the same live guard: two points cannot be U-shaped."""
    artifact, prior, habit = setup
    with pytest.raises(ValueError, match="three distinct positive"):
        exact_sweep_precision(artifact, prior, habit, (0.5, 1.0))


@pytest.mark.parametrize("mutation", ["scaled", "negative", "nan", "wrong-length"])
def test_every_word_prior_consumer_rejects_non_distributions(mutation) -> None:
    """A malformed target cannot become a TV outside its documented range."""

    class TinyLexicon:
        vocab = ["aa", "ab", "ba"]

    if mutation == "scaled":
        prior = np.array([3.0, 2.5, 4.5], dtype=np.float64)
    elif mutation == "negative":
        prior = np.array([1.1, -0.1, 0.0], dtype=np.float64)
    elif mutation == "nan":
        prior = np.array([0.5, np.nan, 0.5], dtype=np.float64)
    else:
        prior = np.array([0.5, 0.5], dtype=np.float64)
    artifact = TinyLexicon()
    habit = np.full((N_SYMBOLS, N_SYMBOLS), 1.0 / N_SYMBOLS, dtype=np.float64)
    run = GenerationResult("aa", frozenset(artifact.vocab))

    consumers = (
        lambda: lexicon_preference("", artifact, prior, flat=True),
        lambda: induced_word_distribution(artifact, prior, habit, 1.0),
        lambda: exact_ancestral_lexicon_total_variation(artifact, prior, habit),
        lambda: exact_lexicon_total_variation(artifact, prior, habit, 1.0),
        lambda: lexicon_total_variation([run], artifact, prior),
    )
    for consume in consumers:
        with pytest.raises(ValueError, match="word_prior"):
            consume()


@pytest.mark.parametrize(
    ("vocabulary", "message"),
    [
        (["a", "a"], "duplicate"),
        ([""], "non-empty"),
        (["A"], "outside"),
        ([], "must not be empty"),
    ],
)
def test_exact_path_enumerators_reject_invalid_finite_lexicons(vocabulary, message) -> None:
    class InvalidLexicon:
        vocab = vocabulary

    habit = _one_letter_habit({"a": 1.0})
    with pytest.raises(ValueError, match=message):
        ancestral_word_distribution(InvalidLexicon(), habit)


def test_exact_path_enumerators_require_a_vocabulary_attribute() -> None:
    class MissingVocabulary:
        pass

    with pytest.raises(ValueError, match="expose an iterable vocabulary"):
        ancestral_word_distribution(MissingVocabulary(), _one_letter_habit({"a": 1.0}))


def test_exact_path_enumerators_reject_a_wrong_shape_transition() -> None:
    with pytest.raises(ValueError, match="habit must be"):
        ancestral_word_distribution(_TinyLexicon(), np.eye(2, dtype=np.float64))


@pytest.mark.parametrize("max_word_len", [0, -1, 1.5, True])
def test_exact_path_enumerators_reject_invalid_horizons(max_word_len) -> None:
    habit = _one_letter_habit({"a": 1.0})
    with pytest.raises(ValueError, match="positive integer"):
        ancestral_word_distribution(_TinyLexicon(), habit, max_word_len=max_word_len)


def test_ancestral_distribution_is_undefined_when_no_word_can_be_emitted() -> None:
    habit = _one_letter_habit({" ": 1.0})
    with pytest.raises(ValueError, match="no non-empty word tokens"):
        ancestral_word_distribution(_TinyLexicon(), habit)


def test_empirical_distance_is_undefined_when_no_nonempty_token_was_emitted() -> None:
    empty = GenerationResult("", frozenset(_TinyLexicon.vocab))
    with pytest.raises(ValueError, match="no words generated"):
        lexicon_total_variation(
            [empty],
            _TinyLexicon(),
            np.array([0.5, 0.5], dtype=np.float64),
        )


def test_words_longer_than_the_horizon_have_zero_named_path_mass() -> None:
    class HorizonLexicon:
        vocab = ["aa", "b"]

    probabilities, out_of_lexicon = ancestral_word_distribution(
        HorizonLexicon(),
        _one_letter_habit({"a": 1.0}),
        max_word_len=1,
    )
    assert probabilities == pytest.approx([0.0, 0.0])
    assert out_of_lexicon == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("raw_probabilities", "empty_probability", "error", "message"),
    [
        (np.array([1.0]), 0.0, ValueError, "shape"),
        (np.array([np.nan, 0.0]), 0.0, ValueError, "finite"),
        (np.array([-0.1, 0.1]), 0.0, ValueError, "non-negative"),
        (np.array([0.5, 0.5]), np.nan, ValueError, "lie in"),
        (np.array([0.7, 0.7]), 0.0, RuntimeError, "disjoint"),
    ],
)
def test_conditioned_path_distribution_rejects_invalid_probability_mass(
    raw_probabilities,
    empty_probability,
    error,
    message,
) -> None:
    """NEGATIVE CONTROLS for the exact path-to-OOV probability boundary."""
    with pytest.raises(error, match=message):
        generation_module._conditioned_lexicon_distribution(
            raw_probabilities,
            empty_probability,
            vocabulary_size=2,
            name="planted",
        )


def test_conditioned_path_distribution_repairs_only_roundoff_overshoot() -> None:
    probabilities, out_of_lexicon = generation_module._conditioned_lexicon_distribution(
        np.array([0.5, 0.5 + 1e-10]),
        0.0,
        vocabulary_size=2,
        name="planted",
    )
    assert float(probabilities.sum()) == pytest.approx(1.0)
    assert out_of_lexicon == 0.0
