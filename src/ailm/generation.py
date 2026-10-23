"""Generation as policy selection — AILM's top-down channel.

Every level below this module is a *recognition* model: observations arrive and
beliefs are updated. Generation is the other direction, and until now the formal
apparatus for it was specified (``eq:efe``) and exercised by nothing — ``C`` was
zero and there was one trivial action at every built level, so there was nothing
to select between.

This module makes generation real, and it is the first place in AILM where a
level's ``C`` is set by another level rather than left flat.

The construction
----------------
The generating agent is the **letter level with a controllable transition**:

  - Hidden factor: the current letter, ``N_SYMBOLS`` states.
  - Actions: ``N_SYMBOLS`` of them. Action ``k`` means *emit letter k*, so
    ``B[next, current, k] = delta(next == k)`` — deterministic controllable
    dynamics. Choosing an action **is** choosing the next symbol, which is what
    makes policy selection generation rather than a metaphor for it.
  - ``E``, the prior over policies: **uniform** on the built agent. The learned
    Brown bigram column is NOT installed as the ``E`` attribute — it enters only
    as the explicit ``habit`` argument that the closed-form policy posterior
    consumes. That is where the corpus statistics enter generation — as a
    *habit*, the thing the agent would do absent any goal.
  - ``C``, log-preferences over letter observations: supplied **top-down** by the
    word level, from the lexicon posterior given the prefix emitted so far. This
    is the goal, the thing the agent is trying to say.

pymdp then computes ``q(pi) = softmax(gamma * neg_efe + ln E)``, so the emitted
letter is a compromise between what usually follows (``E``) and what the lexicon
wants next (``C``). Neither term alone is generation: ``E`` alone is ancestral
sampling from a bigram, and ``C`` alone ignores everything the corpus knows.

Why the gate is distributional fidelity, not lexicon membership
----------------------------------------------------------------
"It generated text" is unfalsifiable, but lexicon membership cannot repair that
problem here: trie-constrained preferences make membership green by construction,
including for a collapsed generator that emits one word forever. Membership is
therefore reported only as a diagnostic and is explicitly rejected as the gate.

The gate is **exact total-variation distance to the lexicon word distribution**,
against ancestral sampling from ``B`` — the strongest generator with the same
letter statistics and no top-down channel — paired with a diversity floor. The
comparator asks whether the top-down channel improves the *distribution* of words,
while the diversity control rejects a low-distance-looking collapsed endpoint.

Two controls, because a positive result needs them
--------------------------------------------------
``flat`` — replacing ``C`` with zeros must collapse the model exactly to the
ancestral baseline. If a flat preference vector still differs, the ``C`` channel
is not the only difference and the result is measuring something else.

``scrambled`` — reversing the letter axis of ``C`` is a fixed involutive
permutation, so the preference keeps its entropy and loses only its meaning. It
must do *worse* than the real channel. Replacing ``C`` with uniform instead would
confound content with entropy, which is why the reversal is the stronger control
(the same argument as ``docs/decisions.md`` entry 9).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from ailm.levels.letter import ALPHA_TO_IDX, IDX_TO_ALPHA, N_SYMBOLS

if TYPE_CHECKING:  # pragma: no cover
    from pymdp.agent import Agent as PymdpAgent

    from ailm.levels.word import WordArtifact

__all__ = [
    "GENERATION_DIVERSITY_FRACTION",
    "GenerationGateEvaluation",
    "GenerationResult",
    "ancestral_word_distribution",
    "build_generative_letter_agent",
    "efe_policy_posterior",
    "evaluate_generation_gate",
    "exact_ancestral_lexicon_total_variation",
    "exact_lexicon_total_variation",
    "exact_sweep_precision",
    "expected_free_energy_terms",
    "generate_by_policy_selection",
    "generate_by_sampling",
    "induced_word_distribution",
    "lexicon_preference",
    "lexicon_total_variation",
    "score_generation",
    "sweep_precision",
]

logger = logging.getLogger(__name__)

#: Index of the space symbol, which terminates a generated word.
SPACE_INDEX: int = ALPHA_TO_IDX[" "]

#: Floor for a log-preference, so an impossible letter is strongly disfavored
#: without producing ``-inf`` (which would make the softmax undefined when every
#: candidate is impossible).
_LOG_FLOOR: float = -30.0
_PROBABILITY_ATOL: float = 1e-7

#: The realized generator must retain more than this share of ancestral
#: distinct-word diversity.  Distance alone admits the shipped high-precision
#: collapse, so this conjunct is part of the gate rather than a diagnostic.
GENERATION_DIVERSITY_FRACTION: float = 0.2


def _validated_distribution(
    values: np.ndarray,
    *,
    size: int,
    name: str,
) -> NDArray[np.float64]:
    """Return a finite, non-negative, unit-mass one-dimensional array."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}, got {array.shape}")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must contain only finite probabilities")
    if bool(np.any(array < 0.0)):
        raise ValueError(f"{name} must contain only non-negative probabilities")
    mass = float(array.sum(dtype=np.float64))
    if not np.isclose(mass, 1.0, rtol=0.0, atol=_PROBABILITY_ATOL):
        raise ValueError(f"{name} must sum to one, got {mass}")
    return array


def _validated_letter_habit(habit: np.ndarray) -> NDArray[np.float64]:
    """Return a finite, non-negative, column-stochastic letter habit."""
    matrix = np.asarray(habit, dtype=np.float64)
    if matrix.shape != (N_SYMBOLS, N_SYMBOLS):
        raise ValueError(f"habit must be ({N_SYMBOLS}, {N_SYMBOLS}), got {matrix.shape}")
    if not bool(np.all(np.isfinite(matrix))):
        raise ValueError("habit must contain only finite probabilities")
    if bool(np.any(matrix < 0.0)):
        raise ValueError("habit must contain only non-negative probabilities")
    column_sums = matrix.sum(axis=0, dtype=np.float64)
    if not bool(np.allclose(column_sums, 1.0, rtol=0.0, atol=_PROBABILITY_ATOL)):
        raise ValueError(
            "habit must be column-stochastic over the next-letter axis; got "
            f"column sums in [{column_sums.min():.4f}, {column_sums.max():.4f}]"
        )
    # The shipped table originates in float32.  Once its measured mass is within
    # the explicit contract above, remove representation-only residuals so
    # NumPy's float64 sampler and the analytic path consume identical columns.
    return matrix / column_sums[np.newaxis, :]


def _validated_vocabulary(artifact: WordArtifact) -> tuple[str, ...]:
    """Return a unique, non-empty vocabulary over the generator's alphabet."""
    try:
        vocabulary = tuple(artifact.vocab)
    except (AttributeError, TypeError) as exc:
        raise ValueError("artifact must expose an iterable vocabulary") from exc
    if not vocabulary:
        raise ValueError("artifact vocabulary must not be empty")
    for index, word in enumerate(vocabulary):
        if not isinstance(word, str) or not word:
            raise ValueError(f"artifact vocabulary word {index} must be a non-empty string")
        invalid = sorted({symbol for symbol in word if symbol == " " or symbol not in ALPHA_TO_IDX})
        if invalid:
            raise ValueError(
                f"artifact vocabulary word {word!r} contains symbols outside the "
                f"lowercase letter alphabet: {invalid!r}"
            )
    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError("artifact vocabulary must not contain duplicate words")
    return vocabulary


def _validated_word_prior(artifact: WordArtifact, word_prior: np.ndarray) -> NDArray[np.float64]:
    """Bind a word-identity distribution to the artifact's vocabulary order."""
    vocabulary = _validated_vocabulary(artifact)
    return _validated_distribution(
        word_prior,
        size=len(vocabulary),
        name="word_prior",
    )


def _validated_max_word_len(max_word_len: int) -> int:
    """Validate the generator's finite word-attempt horizon."""
    if (
        isinstance(max_word_len, (bool, np.bool_))
        or not isinstance(max_word_len, (int, np.integer))
        or int(max_word_len) <= 0
    ):
        raise ValueError(f"max_word_len must be a positive integer, got {max_word_len!r}")
    return int(max_word_len)


def _conditioned_lexicon_distribution(
    raw_probabilities: np.ndarray,
    empty_probability: float,
    *,
    vocabulary_size: int,
    name: str,
) -> tuple[NDArray[np.float64], float]:
    """Condition one-attempt path masses on the non-empty token scoring unit.

    ``GenerationResult.words`` drops an attempt that terminates immediately. The
    empirical TV estimator therefore describes emitted, non-empty word tokens,
    not all requested attempts. Exact distributions use that identical sample
    unit: remove the empty-attempt probability, then put every retained path not
    named by the finite lexicon into one explicit out-of-lexicon bucket.
    """
    probabilities = np.asarray(raw_probabilities, dtype=np.float64)
    if probabilities.shape != (vocabulary_size,):
        raise ValueError(
            f"{name} lexicon probabilities must have shape {(vocabulary_size,)}, "
            f"got {probabilities.shape}"
        )
    if not bool(np.all(np.isfinite(probabilities))):
        raise ValueError(f"{name} lexicon probabilities must be finite")
    if bool(np.any(probabilities < 0.0)):
        raise ValueError(f"{name} lexicon probabilities must be non-negative")
    if not math.isfinite(empty_probability) or not 0.0 <= empty_probability <= 1.0:
        raise ValueError(
            f"{name} empty-attempt probability must lie in [0, 1], got {empty_probability!r}"
        )

    retained_mass = 1.0 - empty_probability
    if retained_mass <= 0.0:
        raise ValueError(f"{name} emits no non-empty word tokens; its distribution is undefined")
    probabilities = probabilities / retained_mass
    lexicon_mass = float(math.fsum(float(value) for value in probabilities))
    if lexicon_mass > 1.0 + _PROBABILITY_ATOL:
        raise RuntimeError(
            f"{name} assigns {lexicon_mass:.17g} conditional mass to disjoint "
            "lexicon paths; the path recursion is not a probability distribution"
        )
    if lexicon_mass > 1.0:
        # Only a representation-scale overshoot is admissible. Normalizing this
        # final ulp-scale residual is safer than silently clipping a negative OOV
        # bucket, which would leave the returned vector with mass greater than one.
        probabilities = probabilities / lexicon_mass
        lexicon_mass = 1.0
    out_of_lexicon = 1.0 - lexicon_mass
    full_distribution = _validated_distribution(
        np.concatenate((probabilities, np.array([out_of_lexicon], dtype=np.float64))),
        size=vocabulary_size + 1,
        name=f"{name} distribution including the out-of-lexicon bucket",
    )
    return full_distribution[:-1], float(full_distribution[-1])


def _exact_lexicon_tv_from_distribution(
    probabilities: np.ndarray,
    out_of_lexicon: float,
    truth: np.ndarray,
    *,
    name: str,
) -> float:
    """Validate and score a lexicon-plus-remainder distribution."""
    target = _validated_distribution(truth, size=len(probabilities), name="word_prior")
    generated = _validated_distribution(
        np.concatenate(
            (
                np.asarray(probabilities, dtype=np.float64),
                np.array([out_of_lexicon], dtype=np.float64),
            )
        ),
        size=len(probabilities) + 1,
        name=f"{name} distribution including the out-of-lexicon bucket",
    )
    target_with_remainder = np.concatenate((target, np.array([0.0], dtype=np.float64)))
    total_variation = 0.5 * float(np.abs(generated - target_with_remainder).sum())
    if not 0.0 <= total_variation <= 1.0:  # pragma: no cover - validated arithmetic
        raise RuntimeError(f"validated distributions produced invalid {name} TV={total_variation}")
    return total_variation


@dataclass(frozen=True)
class GenerationGateEvaluation:
    """Joint verdict for generation's exact-distance and sampled-diversity gate.

    The two quantities deliberately use different estimators: TV is a property
    of the complete finite induced distribution and is computed exactly, while
    diversity is a property of a declared realized-run protocol.  Keeping the
    conjuncts in one object prevents an audit from silently certifying only the
    scalar TV arm.
    """

    model_tv: float
    baseline_tv: float
    model_distinct: float
    baseline_distinct: float
    diversity_fraction: float = GENERATION_DIVERSITY_FRACTION

    def __post_init__(self) -> None:
        for name in ("model_tv", "baseline_tv"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
        for name in ("model_distinct", "baseline_distinct"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        fraction = float(self.diversity_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError(f"diversity_fraction must be finite and in (0, 1], got {fraction!r}")

    @property
    def diversity_floor(self) -> float:
        """Strict lower bound on realized distinct-word count."""
        return self.diversity_fraction * self.baseline_distinct

    @property
    def tv_passes(self) -> bool:
        """Whether exact policy TV strictly beats exact ancestral TV."""
        return self.model_tv < self.baseline_tv

    @property
    def diversity_passes(self) -> bool:
        """Whether realized diversity strictly clears its baseline-relative floor."""
        return self.model_distinct > self.diversity_floor

    @property
    def passes(self) -> bool:
        """The declared conjunctive gate outcome."""
        return self.tv_passes and self.diversity_passes


def evaluate_generation_gate(
    *,
    model_tv: float,
    baseline_tv: float,
    model_distinct: float,
    baseline_distinct: float,
    diversity_fraction: float = GENERATION_DIVERSITY_FRACTION,
) -> GenerationGateEvaluation:
    """Validate both generation metrics and return their joint gate verdict."""
    return GenerationGateEvaluation(
        model_tv=float(model_tv),
        baseline_tv=float(baseline_tv),
        model_distinct=float(model_distinct),
        baseline_distinct=float(baseline_distinct),
        diversity_fraction=float(diversity_fraction),
    )


class GenerationResult:
    """One generation run and everything needed to score it.

    Attributes:
        text: The generated character stream.
        words: Word tokens split on the space symbol, empties dropped.
        in_lexicon: Per-word membership, aligned with ``words`` so two runs can
            be compared position-for-position by a paired test.
    """

    def __init__(self, text: str, vocabulary: frozenset[str]) -> None:
        self.text = text
        self.words = [w for w in text.split(" ") if w]
        self.in_lexicon = [w in vocabulary for w in self.words]

    @property
    def membership_rate(self) -> float:
        """Fraction of generated word tokens that are real words.

        Raises:
            ValueError: If nothing was generated. A rate over zero words is
                undefined, not zero.
        """
        if not self.words:
            raise ValueError("no words generated; membership rate is undefined")
        return sum(self.in_lexicon) / len(self.words)

    @property
    def distinct_words(self) -> int:
        """Distinct word types produced — the anti-degeneracy quantity.

        A generator emitting one real word forever scores a perfect membership
        rate, so membership is only meaningful beside this.
        """
        return len({w for w in self.words})

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"GenerationResult(words={len(self.words)}, "
            f"distinct={self.distinct_words}, text={self.text[:40]!r})"
        )


def build_generative_letter_agent(bigram: np.ndarray, *, gamma: float = 16.0) -> PymdpAgent:
    """Build a letter agent whose transition is **controllable**.

    Unlike ``build_letter_agent``, which has a single trivial action and is a
    pure recognition model, this agent has one action per letter and therefore a
    policy space to select over.

    Args:
        bigram: ``(N_SYMBOLS, N_SYMBOLS)`` column-stochastic letter bigram,
            ``bigram[next, current]``. Returned as the ``habit`` value the caller
            must hand to :func:`efe_policy_posterior` — deliberately NOT
            installed as the agent's ``E`` attribute, which this builder sets
            uniform. Under a controllable ``B`` the corpus statistics are the
            agent's *habit* rather than its dynamics, and they reach policy
            selection through the closed form's explicit argument only.
        gamma: Policy precision. Higher values concentrate ``q(pi)`` on the
            lowest-EFE action, i.e. weight the goal over the habit.

    Returns:
        A configured ``pymdp.agent.Agent`` with ``N_SYMBOLS`` controls.

    Raises:
        ValueError: If ``bigram`` has the wrong shape or is not column-stochastic.
    """
    from pymdp.agent import Agent

    _validated_letter_habit(bigram)
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"gamma must be a positive finite number, got {gamma!r}")

    # B[next, current, action] = 1 iff next == action. Emitting is an action, so
    # the dynamics are deterministic given the choice; all the uncertainty lives
    # in which choice to make.
    transition = np.zeros((N_SYMBOLS, N_SYMBOLS, N_SYMBOLS), dtype=np.float32)
    for action in range(N_SYMBOLS):
        transition[action, :, action] = 1.0

    return Agent(
        A=[jnp.eye(N_SYMBOLS, dtype=jnp.float32)[jnp.newaxis]],
        B=[jnp.array(transition)[jnp.newaxis]],
        C=[jnp.zeros((1, N_SYMBOLS))],
        D=[jnp.full((1, N_SYMBOLS), 1.0 / N_SYMBOLS, dtype=jnp.float32)],
        # E is a bare array, not a per-factor list: it is a distribution over
        # policies, and there is one policy space regardless of factor count.
        E=jnp.full((1, N_SYMBOLS), 1.0 / N_SYMBOLS, dtype=jnp.float32),
        gamma=gamma,
    )


def _log_preference_from_mass(
    values: np.ndarray,
    *,
    scrambled: bool = False,
) -> NDArray[np.float64]:
    """Convert validated non-negative next-symbol mass into floored log preference."""
    mass = np.asarray(values, dtype=np.float64)
    if mass.shape != (N_SYMBOLS,):  # pragma: no cover - internal construction
        raise RuntimeError(f"next-symbol mass has invalid shape {mass.shape}")
    total = float(mass.sum(dtype=np.float64))
    if total <= 0.0:
        return np.zeros(N_SYMBOLS, dtype=np.float64)
    mass = mass / total
    if scrambled:
        mass = mass[::-1].copy()
    with np.errstate(divide="ignore"):
        preference = np.where(mass > 0.0, np.log(mass), _LOG_FLOOR)
    return np.maximum(preference, _LOG_FLOOR)


def _prefix_next_symbol_masses(
    vocabulary: tuple[str, ...],
    prior: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Precompute the trie marginal used at every exact policy path."""
    masses: dict[str, NDArray[np.float64]] = {}
    for index, word in enumerate(vocabulary):
        weight = float(prior[index])
        if weight == 0.0:
            continue
        for depth in range(len(word) + 1):
            prefix = word[:depth]
            mass = masses.get(prefix)
            if mass is None:
                mass = np.zeros(N_SYMBOLS, dtype=np.float64)
                masses[prefix] = mass
            symbol = word[depth] if depth < len(word) else " "
            mass[ALPHA_TO_IDX[symbol]] += weight
    return masses


def lexicon_preference(
    prefix: str,
    artifact: WordArtifact,
    word_prior: np.ndarray,
    *,
    scrambled: bool = False,
    flat: bool = False,
) -> np.ndarray:
    """The **top-down channel**: log-preferences over the next letter, from the lexicon.

    Given the letters emitted so far in the current word, the word level's
    posterior is supported on the vocabulary words carrying that prefix — the
    trie node (``docs/decisions.md`` entry 13). Marginalizing that posterior over
    each word's next letter gives a distribution over what should come next, and
    its log is the letter level's ``C``.

    This is the first real inter-level message in AILM. The one built previously
    ran the other way, from letters into phonemes, and measured harmful
    (``docs/decisions.md`` entry 9); the reason it failed does not apply here.
    That channel carried *less* information than the observation it competed
    with. This one carries information the letter level does not have at all: the
    lexicon.

    Args:
        prefix: Letters of the current word so far. Empty at a word boundary.
        artifact: The vendored word statistics.
        word_prior: ``(n_vocab,)`` distribution over word identity.
        scrambled: Reverse the letter axis — a fixed involutive permutation that
            preserves entropy and destroys meaning. The negative control.
        flat: Return zeros. The vacuity control: with no preference, policy
            selection must fall back to the habit.

    Returns:
        ``(N_SYMBOLS,)`` log-preferences, suitable as pymdp's ``C``.
    """
    vocabulary = _validated_vocabulary(artifact)
    prior = _validated_distribution(word_prior, size=len(vocabulary), name="word_prior")
    if flat:
        return np.zeros(N_SYMBOLS, dtype=np.float64)

    mass: NDArray[np.float64] = np.zeros(N_SYMBOLS, dtype=np.float64)
    depth = len(prefix)
    for index, word in enumerate(vocabulary):
        if not word.startswith(prefix):
            continue
        weight = float(prior[index])
        # A word that ends here votes for the space that terminates it.
        symbol = word[depth] if depth < len(word) else " "
        # Vocabulary validation above proves every symbol is in the alphabet.
        mass[ALPHA_TO_IDX[symbol]] += weight

    # A prefix outside the lexicon produces zero mass, which the helper maps to a
    # flat preference: inventing a distribution there would assert information
    # the lexicon does not support.
    return _log_preference_from_mass(mass, scrambled=scrambled)


def _policy_posterior_from_validated(
    gamma: float,
    current_letter: int,
    preference: NDArray[np.float64],
    habit: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Closed-form policy posterior after all public-boundary validation."""
    habit_column = habit[:, current_letter]
    log_habit = np.full(N_SYMBOLS, -np.inf, dtype=np.float64)
    positive_habit = habit_column > 0.0
    log_habit[positive_habit] = np.log(habit_column[positive_habit])
    # Softmax is invariant to a constant shift in C. Center before multiplying
    # by gamma so a very large but finite precision underflows losing actions to
    # ``-inf`` rather than overflowing every negative log-preference at once.
    centered_preference = preference - float(preference.max())
    with np.errstate(over="ignore"):
        scores = gamma * centered_preference + log_habit
    scores -= scores.max()
    weights = np.exp(scores)
    return _validated_distribution(
        weights / weights.sum(),
        size=N_SYMBOLS,
        name="policy posterior",
    )


def efe_policy_posterior(
    agent: PymdpAgent,
    current_letter: int,
    preference: np.ndarray,
    habit: np.ndarray,
) -> np.ndarray:
    """Posterior over actions, ``q(pi) = softmax(gamma * neg_efe + ln E)``.

    Computed in closed form rather than through ``agent.infer_policies`` so that
    the two can be checked against each other — the oracle-equivalence pattern
    this project applies to every other numeric claim
    (``docs/gate_design.md``). ``tests/test_generation.py`` asserts they agree.

    With a deterministic controllable ``B`` and an identity ``A``, taking action
    ``k`` from any state yields observation ``k`` with certainty. The expected
    free energy therefore has no epistemic term — there is nothing left to learn
    about the state once the action is chosen — and reduces to the pragmatic
    value ``C[k]``.

    That the epistemic term vanishes is a property of *this* construction, not a
    simplification: it is why the agent's behavior is fully explained by the
    preference and the habit, which is what makes the two controls below
    interpretable.

    Args:
        agent: The generative agent, for ``gamma``.
        current_letter: Index of the letter just emitted, selecting the habit column.
        preference: ``(N_SYMBOLS,)`` log-preferences, i.e. ``C``.
        habit: ``(N_SYMBOLS, N_SYMBOLS)`` column-stochastic bigram supplying the
            corpus habit; passed explicitly and never read from the agent, whose
            ``E`` attribute is uniform.

    Returns:
        ``(N_SYMBOLS,)`` distribution over actions.
    """
    gamma = float(np.asarray(agent.gamma).ravel()[0])
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"agent gamma must be a positive finite number, got {gamma!r}")
    if (
        isinstance(current_letter, (bool, np.bool_))
        or not isinstance(current_letter, (int, np.integer))
        or not 0 <= int(current_letter) < N_SYMBOLS
    ):
        raise ValueError(f"current_letter must index the alphabet, got {current_letter!r}")
    goal = np.asarray(preference, dtype=np.float64)
    if goal.shape != (N_SYMBOLS,) or not bool(np.all(np.isfinite(goal))):
        raise ValueError("preference must be a finite log-preference vector over letters")
    validated_habit = _validated_letter_habit(habit)
    return _policy_posterior_from_validated(gamma, int(current_letter), goal, validated_habit)


def expected_free_energy_terms(
    emission: np.ndarray,
    transition: np.ndarray,
    preference: np.ndarray,
    *,
    state_prior: np.ndarray | None = None,
) -> dict[str, NDArray[np.float64]]:
    r"""Both terms of [@eq:efe], per action, computed from first principles.

    The generation section makes an analytic claim about this construction: its
    information-gain term is identically zero, so policy selection reduces to a
    preference-weighted softmax over the habit. That claim was previously stated
    in a docstring and asserted rather than demonstrated. This function computes
    the two terms separately so the vanishing is *measured*, and so a
    construction in which it does *not* vanish can be exhibited beside it.

    For action :math:`k` the policy-conditioned predictive is
    :math:`q(s \mid k) = \mathbf{B}[:, :, k]\, d` and
    :math:`\tilde{q}(o \mid k) = \mathbf{A}\, q(s \mid k)`. Information gain is
    the expected KL between the state posterior after seeing an outcome and the
    predictive that preceded it, and pragmatic value is the expected
    log-preference.

    Two independent conditions each drive the information gain to zero, and
    telling them apart is the whole point of computing the term rather than
    reasoning about it:

    * **A degenerate predictive.** If ``transition`` is deterministic then
      :math:`q(s \mid k)` is a point mass, no outcome can sharpen it, and the KL
      is zero whatever ``emission`` looks like. This is the condition the shipped
      generator meets.
    * **An uninformative likelihood.** If ``emission`` carries no information
      about the state, the posterior equals the predictive and the KL is zero
      even when the predictive is spread.

    The shipped construction satisfies the first. It also has an identity
    ``emission``, which is maximally informative — so the vanishing here is a
    fact about the *transition*, not about the observation model, and an
    implementation that credited the identity likelihood would be right about
    the number and wrong about the reason.

    Args:
        emission: ``(n_obs, n_states)`` likelihood, column-stochastic.
        transition: ``(n_states, n_states, n_actions)`` transitions, with
            ``transition[next, current, action]`` column-stochastic over ``next``.
        preference: ``(n_obs,)`` log-preferences over outcomes, i.e. ``C``.
        state_prior: ``(n_states,)`` current belief. Uniform when omitted.

    Returns:
        Mapping with ``information_gain``, ``pragmatic_value`` and
        ``expected_free_energy``, each ``(n_actions,)``. ``expected_free_energy``
        is ``-(information_gain + pragmatic_value)``, matching [@eq:efe].

    Raises:
        ValueError: If the shapes disagree or a column fails to normalize.
    """
    likelihood = np.asarray(emission, dtype=np.float64)
    dynamics = np.asarray(transition, dtype=np.float64)
    goal = np.asarray(preference, dtype=np.float64)
    if likelihood.ndim != 2 or dynamics.ndim != 3:
        raise ValueError(
            f"emission must be 2-D and transition 3-D, got {likelihood.shape} and {dynamics.shape}"
        )
    n_obs, n_states = likelihood.shape
    if dynamics.shape[:2] != (n_states, n_states) or goal.shape != (n_obs,):
        raise ValueError(
            f"shape mismatch: emission {likelihood.shape}, transition "
            f"{dynamics.shape}, preference {goal.shape}"
        )
    if not bool(np.all(np.isfinite(likelihood))) or bool(np.any(likelihood < 0.0)):
        raise ValueError("emission must contain finite non-negative probabilities")
    if not bool(
        np.allclose(
            likelihood.sum(axis=0, dtype=np.float64),
            1.0,
            rtol=0.0,
            atol=_PROBABILITY_ATOL,
        )
    ):
        raise ValueError("emission columns must sum to one")
    if not bool(np.all(np.isfinite(dynamics))) or bool(np.any(dynamics < 0.0)):
        raise ValueError("transition must contain finite non-negative probabilities")
    if not bool(
        np.allclose(
            dynamics.sum(axis=0, dtype=np.float64),
            1.0,
            rtol=0.0,
            atol=_PROBABILITY_ATOL,
        )
    ):
        raise ValueError("transition columns must sum to one over the next state")
    if not bool(np.all(np.isfinite(goal))):
        raise ValueError("preference must contain only finite log-preferences")
    belief = (
        np.full(n_states, 1.0 / n_states)
        if state_prior is None
        else np.asarray(state_prior, dtype=np.float64)
    )
    belief = _validated_distribution(belief, size=n_states, name="state_prior")

    n_actions = dynamics.shape[2]
    information_gain = np.zeros(n_actions, dtype=np.float64)
    pragmatic_value = np.zeros(n_actions, dtype=np.float64)
    for action in range(n_actions):
        predictive_states = _validated_distribution(
            dynamics[:, :, action] @ belief,
            size=n_states,
            name=f"predictive state distribution for action {action}",
        )
        predictive_obs = _validated_distribution(
            likelihood @ predictive_states,
            size=n_obs,
            name=f"predictive observation distribution for action {action}",
        )
        pragmatic_value[action] = float(predictive_obs @ goal)
        for outcome in range(n_obs):
            weight = float(predictive_obs[outcome])
            if weight <= 0.0:
                continue
            joint = likelihood[outcome] * predictive_states
            total = joint.sum()
            # ``total`` is the already-positive predictive outcome probability:
            # both are the same finite sum under the validated likelihood and
            # predictive state distribution.
            posterior = _validated_distribution(
                joint / total,
                size=n_states,
                name=f"state posterior for action {action}, outcome {outcome}",
            )
            support = posterior > 0.0
            information_gain[action] += weight * float(
                np.sum(posterior[support] * np.log(posterior[support] / predictive_states[support]))
            )
    return {
        "information_gain": information_gain,
        "pragmatic_value": pragmatic_value,
        "expected_free_energy": -(information_gain + pragmatic_value),
    }


def generate_by_policy_selection(
    agent: PymdpAgent,
    artifact: WordArtifact,
    word_prior: np.ndarray,
    habit: np.ndarray,
    *,
    n_words: int,
    seed: int,
    scrambled: bool = False,
    flat: bool = False,
    max_word_len: int = 20,
) -> GenerationResult:
    """Generate text by selecting actions that minimize expected free energy.

    Args:
        agent: The generative letter agent.
        artifact: Word statistics supplying the lexicon.
        word_prior: Distribution over word identity.
        habit: The letter bigram, used as the policy prior.
        n_words: How many word tokens to emit.
        seed: RNG seed. Generation is stochastic, so every sampled diagnostic
            needs one; the canonical distributional gate is computed exactly.
        scrambled: Use the entropy-preserving scrambled preference control.
        flat: Use the zero-preference vacuity control.
        max_word_len: Force a word boundary after this many letters, so a
            degenerate run cannot spin forever.

    Returns:
        The generated text and its per-word membership vector.
    """
    _validated_word_prior(artifact, word_prior)
    _validated_letter_habit(habit)
    rng = np.random.default_rng(seed)
    vocabulary = frozenset(artifact.vocab)
    emitted: list[str] = []
    current = SPACE_INDEX

    for _ in range(n_words):
        prefix = ""
        for _ in range(max_word_len):
            preference = lexicon_preference(
                prefix, artifact, word_prior, scrambled=scrambled, flat=flat
            )
            posterior = efe_policy_posterior(agent, current, preference, habit)
            action = int(rng.choice(N_SYMBOLS, p=posterior))
            symbol = IDX_TO_ALPHA[action]
            current = action
            if symbol == " ":
                break
            prefix += symbol
        emitted.append(prefix)
        current = SPACE_INDEX

    return GenerationResult(" ".join(emitted), vocabulary)


def generate_by_sampling(
    habit: np.ndarray,
    artifact: WordArtifact,
    *,
    n_words: int,
    seed: int,
    max_word_len: int = 20,
) -> GenerationResult:
    """**The baseline**: ancestral sampling from the letter bigram, no top-down channel.

    The strongest generator with the same letter statistics and no lexicon. Any
    claim that generation works has to beat this, because beating uniform-random
    letters would demonstrate only that the bigram exists.

    Args:
        habit: ``(N_SYMBOLS, N_SYMBOLS)`` column-stochastic letter bigram.
        artifact: Supplies the vocabulary the output is scored against.
        n_words: How many word tokens to emit.
        seed: RNG seed.
        max_word_len: Forced word boundary, matching the policy generator so the
            two are comparable.

    Returns:
        The generated text and its per-word membership vector.
    """
    validated_habit = _validated_letter_habit(habit)
    rng = np.random.default_rng(seed)
    vocabulary = frozenset(artifact.vocab)
    emitted: list[str] = []
    current = SPACE_INDEX

    for _ in range(n_words):
        prefix = ""
        for _ in range(max_word_len):
            action = int(rng.choice(N_SYMBOLS, p=validated_habit[:, current]))
            symbol = IDX_TO_ALPHA[action]
            current = action
            if symbol == " ":
                break
            prefix += symbol
        emitted.append(prefix)
        current = SPACE_INDEX

    return GenerationResult(" ".join(emitted), vocabulary)


def score_generation(results: list[GenerationResult]) -> dict[str, float]:
    """Aggregate several seeds into sampled realized-run diagnostics.

    A single seed is a cherry-pick: generation is stochastic, so one run says
    nothing about its realized membership or diversity. These diagnostics
    average over seeds; the canonical TV gate does not call this function.

    Args:
        results: One :class:`GenerationResult` per seed.

    Returns:
        ``membership_mean``, ``membership_min``, ``membership_max``,
        ``distinct_mean``, ``n_words``, ``n_seeds``.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("no runs to score")
    rates = [r.membership_rate for r in results]
    distinct = [r.distinct_words for r in results]
    return {
        "membership_mean": float(np.mean(rates)),
        "membership_min": float(np.min(rates)),
        "membership_max": float(np.max(rates)),
        "distinct_mean": float(np.mean(distinct)),
        "n_words": float(np.mean([len(r.words) for r in results])),
        "n_seeds": float(len(results)),
    }


# ── SCORING THE DISTRIBUTION, NOT JUST THE MEMBERSHIP ────────────────────────


def lexicon_total_variation(
    results: list[GenerationResult],
    artifact: WordArtifact,
    word_prior: np.ndarray,
) -> float:
    """Monte Carlo TV estimate from generated tokens to the lexicon distribution.

    The underlying TV quantity is the gate metric; canonical claims compute it
    with :func:`exact_lexicon_total_variation` or
    :func:`exact_ancestral_lexicon_total_variation`. This empirical estimator is
    retained for seeded convergence controls and explicitly sampled diagnostics.

    Membership is not the gate. It is green by construction because
    :func:`lexicon_preference` floors any letter that would leave the trie at
    ``_LOG_FLOOR``. Measured, membership is
    claim:generation_best_membership_pct% at and beyond the first fully
    constrained positive-``gamma`` rung — a metric that cannot fail is not a
    gate, however good the number looks.

    What *does* vary is whether the generator emits words in the right
    proportions. Sampling a real word is easy once the channel exists; sampling
    ``the`` as often as English does is the actual claim.

    Every non-vocabulary token collapses into a single out-of-lexicon bucket, so
    the distance stays bounded in ``[0, 1]`` and a baseline that emits mostly
    non-words is scored honestly rather than by an infinite cross-entropy.

    Args:
        results: One run per seed.
        artifact: Supplies the vocabulary order.
        word_prior: ``(n_vocab,)`` true distribution over word identity.

    Returns:
        Total-variation distance in ``[0, 1]``. Lower is better.

    Raises:
        ValueError: If no words were generated.
    """
    from collections import Counter

    counts = Counter(word for run in results for word in run.words)
    total = sum(counts.values())
    if total == 0:
        raise ValueError("no words generated; the distance is undefined")

    prior = _validated_word_prior(artifact, word_prior)
    truth = {word: float(prior[i]) for i, word in enumerate(artifact.vocab)}
    distance = 0.0
    covered = 0.0
    for word, count in counts.items():
        expected = truth.get(word, 0.0)
        distance += abs(count / total - expected)
        covered += expected
    # Lexicon mass that was never generated at all still counts against us.
    distance += 1.0 - covered
    total_variation = distance / 2.0
    if not 0.0 <= total_variation <= 1.0 + _PROBABILITY_ATOL:  # pragma: no cover
        raise RuntimeError(f"validated distributions produced invalid TV={total_variation}")
    return min(1.0, total_variation)


def _word_path_probability(
    word: str,
    action_posterior: Callable[[str, int], NDArray[np.float64]],
    *,
    max_word_len: int,
) -> float:
    """Probability of one named word under the generator's truncation rule."""
    if len(word) > max_word_len:
        return 0.0
    probability = 1.0
    current = SPACE_INDEX
    for depth, letter in enumerate(word):
        symbol = ALPHA_TO_IDX[letter]
        probability *= float(action_posterior(word[:depth], current)[symbol])
        if probability == 0.0:
            return 0.0
        current = symbol
    if len(word) < max_word_len:
        probability *= float(action_posterior(word, current)[SPACE_INDEX])
    # At exactly max_word_len the implementation appends the accumulated prefix
    # without drawing a terminator. Omitting that final factor is therefore the
    # exact probability of the truncated output, not an approximation.
    return probability


def ancestral_word_distribution(
    artifact: WordArtifact,
    habit: np.ndarray,
    *,
    max_word_len: int = 20,
) -> tuple[NDArray[np.float64], float]:
    """Exact ancestral distribution over a finite lexicon plus an OOV bucket.

    The baseline begins each word attempt at the space state and repeatedly draws
    ``next ~ habit[:, current]``. For each vocabulary word, its mass is the
    product of the Brown-transition entries along that path and the terminating
    space; a word that exactly reaches ``max_word_len`` is emitted by truncation
    and therefore has no terminal-space factor. All other non-empty paths are
    represented by the returned out-of-lexicon remainder.

    The returned distribution is conditional on a non-empty emitted token. That
    is the sample unit used by :class:`GenerationResult`, which discards an
    immediate space, so this exact quantity and :func:`lexicon_total_variation`
    estimate the same object.

    Args:
        artifact: Supplies the finite vocabulary whose paths are enumerated.
        habit: Column-stochastic Brown transition, ``habit[next, current]``.
        max_word_len: The baseline's forced word boundary.

    Returns:
        ``(probabilities in artifact.vocab order, out-of-lexicon remainder)``;
        the two parts are finite, non-negative, and sum to one.

    Raises:
        ValueError: If the vocabulary, transition, or horizon is invalid, or the
            process emits no non-empty tokens.
        RuntimeError: If the enumerated disjoint paths exceed unit mass.
    """
    vocabulary = _validated_vocabulary(artifact)
    validated_habit = _validated_letter_habit(habit)
    horizon = _validated_max_word_len(max_word_len)

    def action_posterior(_prefix: str, current: int) -> NDArray[np.float64]:
        return validated_habit[:, current]

    raw_probabilities = np.array(
        [
            _word_path_probability(word, action_posterior, max_word_len=horizon)
            for word in vocabulary
        ],
        dtype=np.float64,
    )
    return _conditioned_lexicon_distribution(
        raw_probabilities,
        float(validated_habit[SPACE_INDEX, SPACE_INDEX]),
        vocabulary_size=len(vocabulary),
        name="ancestral generator",
    )


def exact_ancestral_lexicon_total_variation(
    artifact: WordArtifact,
    word_prior: np.ndarray,
    habit: np.ndarray,
    *,
    max_word_len: int = 20,
) -> float:
    """Exact TV from ancestral Brown sampling to the lexicon distribution."""
    truth = _validated_word_prior(artifact, word_prior)
    probabilities, out_of_lexicon = ancestral_word_distribution(
        artifact,
        habit,
        max_word_len=max_word_len,
    )
    return _exact_lexicon_tv_from_distribution(
        probabilities,
        out_of_lexicon,
        truth,
        name="ancestral",
    )


def induced_word_distribution(
    artifact: WordArtifact,
    word_prior: np.ndarray,
    habit: np.ndarray,
    gamma: float,
    *,
    max_word_len: int = 20,
) -> tuple[NDArray[np.float64], float]:
    """The generator's word distribution, computed rather than sampled.

    :func:`generate_by_policy_selection` is a Markov chain whose entire state is
    the prefix emitted so far: :func:`lexicon_preference` reads the prefix and
    nothing else, ``current`` is the prefix's last letter, and ``current`` is
    reset to a space at every word boundary. So the probability of emitting a
    particular word is the product of the action posteriors along that word's own
    path through the trie, times the probability of choosing the terminating
    space at its end (unless the truncation horizon emits it first). No
    enumeration of the off-lexicon paths is needed: their total mass is whatever
    the vocabulary does not account for.

    Why this exists rather than a larger Monte Carlo sample. The original sweep
    pooled fewer generated tokens than there are target types and reported no
    uncertainty, so its empirical distribution could not even support the full
    target. The resulting finite-sample bias and variance were large enough to
    change the selected operating point. The U-shape survives exact computation;
    the sampled argmin does not. Increasing the sample size would narrow the
    error; computing the finite-path quantity removes it.

    Args:
        artifact: Supplies the vocabulary, in the order ``word_prior`` indexes.
        word_prior: ``(n_vocab,)`` distribution the generator is trying to match.
        habit: ``(N_SYMBOLS, N_SYMBOLS)`` column-stochastic letter bigram.
        gamma: Policy precision.
        max_word_len: The generator's own truncation length. A word exactly this
            long is emitted without drawing a terminating space; a longer word
            has zero probability of being emitted intact.

    Returns:
        ``(probabilities over the vocabulary, out-of-lexicon mass)``, conditional
        on a non-empty emitted token so the result matches ``GenerationResult``.

    Raises:
        ValueError: If ``gamma``, the inputs, or the truncation horizon is invalid.
    """
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"gamma must be a positive finite number, got {gamma!r}")
    vocabulary = _validated_vocabulary(artifact)
    prior = _validated_distribution(word_prior, size=len(vocabulary), name="word_prior")
    validated_habit = _validated_letter_habit(habit)
    horizon = _validated_max_word_len(max_word_len)

    prefix_masses = _prefix_next_symbol_masses(vocabulary, prior)
    # One posterior per (prefix, current letter). Prefixes are shared across the
    # vocabulary, so the trie's branching does the memoization for us: the cache
    # holds one entry per trie node rather than one per word.
    cache: dict[tuple[str, int], NDArray[np.float64]] = {}

    def action_posterior(prefix: str, current: int) -> NDArray[np.float64]:
        key = (prefix, current)
        cached = cache.get(key)
        if cached is None:
            mass = prefix_masses.get(prefix)
            preference = (
                np.zeros(N_SYMBOLS, dtype=np.float64)
                if mass is None
                else _log_preference_from_mass(mass)
            )
            cached = _policy_posterior_from_validated(
                gamma,
                current,
                preference,
                validated_habit,
            )
            cache[key] = cached
        return cached

    raw_probabilities = np.array(
        [
            _word_path_probability(word, action_posterior, max_word_len=horizon)
            for word in vocabulary
        ],
        dtype=np.float64,
    )
    empty_probability = float(action_posterior("", SPACE_INDEX)[SPACE_INDEX])
    return _conditioned_lexicon_distribution(
        raw_probabilities,
        empty_probability,
        vocabulary_size=len(vocabulary),
        name="policy-selection generator",
    )


def exact_lexicon_total_variation(
    artifact: WordArtifact,
    word_prior: np.ndarray,
    habit: np.ndarray,
    gamma: float,
    *,
    max_word_len: int = 20,
) -> float:
    """:func:`lexicon_total_variation`'s quantity, without the sampling error.

    Scored the same way, so the two are directly comparable and the sampled
    generator can be checked for convergence against this: every non-vocabulary
    token collapses into one out-of-lexicon bucket, keeping the distance in
    ``[0, 1]``.

    Args:
        artifact: Supplies the vocabulary order.
        word_prior: The true distribution over word identity.
        habit: The letter bigram used as the policy prior.
        gamma: Policy precision.
        max_word_len: The generator's forced word boundary.

    Returns:
        Total-variation distance in ``[0, 1]``. Lower is better.
    """
    probabilities, out_of_lexicon = induced_word_distribution(
        artifact,
        word_prior,
        habit,
        gamma,
        max_word_len=max_word_len,
    )
    truth = _validated_word_prior(artifact, word_prior)
    return _exact_lexicon_tv_from_distribution(
        probabilities,
        out_of_lexicon,
        truth,
        name="policy-selection",
    )


def exact_sweep_precision(
    artifact: WordArtifact,
    word_prior: np.ndarray,
    habit: np.ndarray,
    gammas: tuple[float, ...],
) -> list[dict[str, float]]:
    """:func:`sweep_precision`'s frontier, computed exactly at each ``gamma``.

    Args:
        artifact: The vendored word statistics.
        word_prior: True distribution over word identity.
        habit: ``(N_SYMBOLS, N_SYMBOLS)`` letter bigram.
        gammas: Precisions to evaluate. Three distinct positive values is the
            minimum that can exhibit an interior optimum, matching the sampled
            sweep's own guard.

    Returns:
        One row per ``gamma``, each with ``gamma``, ``total_variation`` and
        ``out_of_lexicon``.

    Raises:
        ValueError: If ``gammas`` cannot support the claim.
    """
    if len({g for g in gammas if g > 0.0}) < 3:
        raise ValueError(
            f"an exact precision sweep needs at least three distinct positive "
            f"gammas to exhibit an interior optimum, got {gammas!r}"
        )
    rows: list[dict[str, float]] = []
    for gamma in gammas:
        probabilities, out_of_lexicon = induced_word_distribution(
            artifact, word_prior, habit, gamma
        )
        truth = _validated_word_prior(artifact, word_prior)
        rows.append(
            {
                "gamma": float(gamma),
                "total_variation": _exact_lexicon_tv_from_distribution(
                    probabilities,
                    out_of_lexicon,
                    truth,
                    name=f"policy-selection gamma={gamma}",
                ),
                "out_of_lexicon": out_of_lexicon,
            }
        )
    return rows


def sweep_precision(
    artifact: WordArtifact,
    word_prior: np.ndarray,
    habit: np.ndarray,
    gammas: tuple[float, ...],
    *,
    seeds: tuple[int, ...],
    n_words: int,
) -> list[dict[str, float]]:
    """Sampled diagnostic frontier across policy precision ``gamma``.

    The canonical distance sweep is :func:`exact_sweep_precision`. This retained
    Monte Carlo path supplies realized-run membership/diversity and checks that
    sampling converges toward the exact distribution. ``gamma`` weighs the goal
    (``C``, the lexicon) against the habit (``E``, the letter bigram), and both
    extremes generate badly for *opposite* reasons:

    * ``gamma -> 0`` reduces exactly to ancestral sampling from the bigram — the
      baseline. No goal, so no words.
    * ``gamma -> inf`` follows the locally most preferred trie branch at each
      prefix (with the habit breaking ties). That greedy prefix path need not be
      the globally most probable word. On the shipped Brown artifact it
      collapses onto one word and emits it forever: perfect membership, no
      language.

    So the distance to the true word distribution is U-shaped with an **interior**
    optimum, which is what makes this gate non-vacuous in both directions — a
    property none of the recognition gates has.

    Args:
        artifact: The vendored word statistics.
        word_prior: True distribution over word identity.
        habit: ``(N_SYMBOLS, N_SYMBOLS)`` letter bigram.
        gammas: Precisions to evaluate. Must contain at least three distinct
            positive values, or the U-shape cannot be demonstrated.
        seeds: RNG seeds per point. Generation is stochastic; a single seed is a
            cherry-pick.
        n_words: Word tokens per run.

    Returns:
        One row per ``gamma``, each with ``gamma``, ``membership``, ``distinct``
        and ``total_variation``.

    Raises:
        ValueError: If ``gammas`` or ``seeds`` is too small to support the claim.
    """
    if len({g for g in gammas if g > 0.0}) < 3:
        raise ValueError(
            f"a precision sweep needs at least three distinct positive gammas to "
            f"show an interior optimum; got {gammas!r}"
        )
    if len(seeds) < 2:
        raise ValueError(f"a single seed is a cherry-pick; got {seeds!r}")

    rows: list[dict[str, float]] = []
    for gamma in gammas:
        agent = build_generative_letter_agent(habit, gamma=gamma)
        runs = [
            generate_by_policy_selection(
                agent, artifact, word_prior, habit, n_words=n_words, seed=seed
            )
            for seed in seeds
        ]
        scored = score_generation(runs)
        rows.append(
            {
                "gamma": float(gamma),
                "membership": scored["membership_mean"],
                "distinct": scored["distinct_mean"],
                "total_variation": lexicon_total_variation(runs, artifact, word_prior),
            }
        )
    return rows
