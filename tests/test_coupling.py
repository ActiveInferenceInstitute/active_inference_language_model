"""Gate for L1 -> L2 coupling. THE FINDING IS A NULL, AND IT IS THE POINT.

What was claimed and what is true
---------------------------------
``src/ailm/figures/architecture.py`` labels an arrow between the letter level and
the phoneme level "update_empirical_prior couples adjacent levels". No code did
that: ``update_empirical_prior`` appeared only inside the single-level temporal
recursion of ``ailm.levels.letter`` and ``ailm.levels.phoneme``. This file gates
:mod:`ailm.levels.coupling`, which implements the message the figure describes —
the L1 predictive distribution over the next letter, fed into the L2B G2P agent as
a **soft** observation instead of a hard glyph index.

**Measured outcome: coupling does not help, and it is not marginal.** Full
held-out set, 42,820 positions, alpha grid fixed in advance at
``(0, 0.1, 0.25, 0.5, 0.75, 1.0)``:

    alpha   accuracy      vs alpha=0
    0.00    70.8010%       anchor (identical to the shipped uncoupled decode)
    0.10    70.5371%       -0.264 pts
    0.25    69.3344%       -1.467 pts
    0.50    63.1060%       -7.695 pts
    0.75    48.6595%      -22.141 pts
    1.00    16.9220%      -53.879 pts

Accuracy is strictly monotone *decreasing* in alpha. The best positive alpha is
the smallest one on the grid, and even that is significantly worse than the
anchor: paired McNemar b=217 (coupled right, uncoupled wrong), c=330 (the
reverse), p = 1.5e-06. The p-value is significant in the direction of **harm**.

No alpha was tuned to produce this. The grid was fixed before measuring, is
guarded by :func:`ailm.levels.coupling.validate_alpha_grid` against collapsing to
a single point, and the measurement is on the full held-out set, not on a
subsample chosen after the fact.

Why it fails, structurally
--------------------------
At position ``t`` the G2P agent already **observes** letter ``t``. The L1
predictive is a function of letters ``0 .. t-1`` only, so it is strictly weaker
evidence about letter ``t`` than the glyph is. Measured on the same held-out
positions, the L1 predictive assigns mean probability 0.082 to the letter actually
observed (uniform would be 0.033) and its argmax picks the right letter 11.0% of
the time. Mixing that into the likelihood spends ``alpha`` of the evidence budget
on an 11%-accurate guess in place of a certainty. The orthographic context the
message does carry is not new to L2B either — L2B already propagates context
through its own phone-bigram ``B``.

The failure is not specific to the mixture form. An exploratory product-of-experts
variant (keep the hard glyph likelihood, *multiply* the marginalized message in at
weight beta, so nothing is displaced) also degrades monotonically: 70.736% at
beta=0.05 falling to 69.206% at beta=1.0, every point significantly worse than the
anchor. That probe is not shipped and not recorded — it is mentioned because it is
the reason the recommendation is "correct the figure label", not "try a different
alpha".

The scrambled-message control, and how to read it
-------------------------------------------------
:func:`ailm.levels.coupling.scramble_letter_distribution` reverses the letter axis
of the message — a fixed permutation, so entropy and peakedness are preserved and
only the correspondence to the alphabet is destroyed. Two things come out of it,
and they point in opposite directions:

  - **At alpha = 1.0**, where the glyph is discarded and the message is the only
    evidence, the real message scores 16.922% against the scrambled 4.505%. The
    channel is therefore genuinely carrying letter-model information — this null
    is a measurement, not an inert pipe.
  - **At every intermediate alpha the scrambled message does LESS damage than the
    real one** (alpha=0.1: 70.752% scrambled vs 70.537% real). That is not a
    contradiction, it is the mechanism. A reversed distribution puts mass on
    letters that are near-irrelevant to the observed one, which mostly flattens
    the likelihood and rarely moves the argmax. The real predictive puts mass on
    *plausible alternative letters*, which shifts the phone posterior toward
    genuinely competing phones and flips decisions that were right. An informative
    message displacing exact evidence is worse than noise displacing it.

Read together: the coupling channel works, and using it this way is harmful.
``coupling_supported`` is recorded as 0.

Controls that keep this null from being vacuous
----------------------------------------------
A test suite that reports "no effect" needs the same scaffolding as one reporting
an effect, or "we measured nothing" and "we measured no effect" are
indistinguishable:

  1. :func:`test_alpha_zero_reproduces_uncoupled_decode_exactly` — the anchor.
     Zero position differences across all 42,820, against
     :func:`ailm.verification.oracle.forward_filter` driven by hard observations,
     which is a *different code path* from the soft recursion under test.
  2. :func:`test_anchor_is_not_vacuous_alpha_above_zero_does_differ` — the
     positive control on that anchor. At alpha=0.1 the two decodes MUST differ,
     otherwise the zero at alpha=0 would only show the comparison is blind.
  3. :func:`test_alpha_zero_matches_the_pymdp_agent` — binds the numpy anchor to
     the shipped pymdp production path on a bounded subset.
  4. :func:`test_mcnemar_self_comparison_finds_no_discordant_pairs` — the negative
     control on the significance machinery: alpha=0 against itself gives b=c=0,
     p=1.0.
  5. :func:`test_coupled_decoder_is_live_under_a_broken_emission` — the coupled
     accuracy metric must collapse on a mis-wired model, or it is not measuring
     the model.
  6. The grid guard, exercised in both directions.

Do not "fix" this file by widening the grid until something passes, by moving to a
subsample where the margin flips, or by deleting the monotonicity assertion. If a
future coupling form genuinely helps, the assertions here must be re-derived from
that measurement — not relaxed to accommodate it.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ailm.claims import load_ledger
from ailm.control import GateVacuousError
from ailm.levels.coupling import (
    DEFAULT_ALPHA_GRID,
    G2PMatrices,
    LetterMatrices,
    build_g2p_matrices,
    build_letter_matrices,
    couple_letter_to_phoneme,
    coupled_accuracy,
    coupled_positions,
    letter_predictive_distributions,
    mcnemar,
    mix_letter_distribution,
    scramble_letter_distribution,
    soft_observation_likelihood,
    sweep_alpha,
    uncoupled_positions,
    validate_alpha_grid,
)
from ailm.levels.letter import (
    ALPHA_LIST,
    ALPHA_TO_IDX,
    IDX_TO_ALPHA,
    N_SYMBOLS,
    load_brown_counts,
)

# ``_iter_pairs`` and ``_word_letters`` are imported for the reason
# ``tests/test_stage2_gate.py`` imports the former: an alignment claim between two
# decoders is only checkable if the test resolves the subsample and the letter
# normalization through the *same* functions the decoders do.
from ailm.levels.phoneme import (
    N_PHONES,
    PHONES,
    _iter_pairs,
    _word_letters,
    baseline_positions,
    build_g2p_agent,
    build_g2p_emission,
    build_g2p_start,
    build_g2p_transition,
    decode_phonemes,
    g2p_split,
    load_phoneme_counts,
    per_letter_argmax_baseline,
)

# ── PATHS ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor"
PHONEME_COUNTS_FILE = VENDOR_DIR / "cmudict_counts.json"
BROWN_COUNTS_FILE = VENDOR_DIR / "brown_counts.json"
LEDGER_FILE = PROJECT_ROOT / "data" / "claims.json"

_SKIP_NO_COUNTS = pytest.mark.skipif(
    not (PHONEME_COUNTS_FILE.exists() and BROWN_COUNTS_FILE.exists()),
    reason=(
        f"Needs both {PHONEME_COUNTS_FILE.name} and {BROWN_COUNTS_FILE.name} in "
        "vendor/. Run scripts/generate_phoneme_counts.py and "
        "scripts/generate_counts.py."
    ),
)

# ── GATE CONSTANTS ────────────────────────────────────────────────────────────
#: Significance level, matching the Stage 2 G2P gate.
ALPHA_LEVEL = 0.01

#: Words decoded through pymdp for the production-path identity check. Bounded
#: because pymdp costs ~4.3 ms/position; the check is an identity, so it does not
#: need a representative sample.
PYMDP_WORDS = 120

#: Words used for the fast alignment/stride checks.
SMALL_WORDS = 200

#: Minimum eval positions. Guards a silently truncated full-set evaluation.
MIN_EVAL_POSITIONS = 40000

#: The alpha whose damage is smallest on the grid, established by measurement in
#: :func:`test_alpha_sweep_is_monotone_decreasing` rather than assumed here.
_SMALLEST_POSITIVE_ALPHA = 0.1


# ── FIXTURES ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def counts() -> dict:
    """The vendored CMUdict statistics."""
    return load_phoneme_counts(VENDOR_DIR)


@pytest.fixture(scope="module")
def split(counts) -> tuple[list, list]:
    """``(train, test)`` G2P pairs — index-based split, never re-sorted."""
    return g2p_split(counts)


@pytest.fixture(scope="module")
def letter_matrices() -> LetterMatrices:
    """The L1 letter model as numpy, from the shipped Brown counts."""
    return build_letter_matrices(load_brown_counts(VENDOR_DIR))


@pytest.fixture(scope="module")
def g2p_matrices(split) -> G2PMatrices:
    """The L2B G2P model as numpy, trained on the TRAIN split only."""
    return build_g2p_matrices(split[0])


@pytest.fixture(scope="module")
def uncoupled_full(g2p_matrices, split) -> list[bool]:
    """Full-set uncoupled correctness, via the oracle's hard-observation path."""
    return uncoupled_positions(g2p_matrices, split[1])


@pytest.fixture(scope="module")
def full_sweep(letter_matrices, g2p_matrices, split) -> dict[float, list[bool]]:
    """Full-set correctness vectors for every alpha on the fixed grid."""
    return sweep_alpha(letter_matrices, g2p_matrices, split[1], DEFAULT_ALPHA_GRID)


@pytest.fixture(scope="module")
def scrambled_sweep(letter_matrices, g2p_matrices, split) -> dict[float, list[bool]]:
    """The same sweep with the message's letter axis reversed (the control)."""
    return sweep_alpha(letter_matrices, g2p_matrices, split[1], DEFAULT_ALPHA_GRID, scramble=True)


def _accuracy(positions: list[bool]) -> float:
    """Percentage of correct positions."""
    assert positions, "empty position vector — nothing was measured"
    return 100.0 * sum(positions) / len(positions)


def _best_positive_alpha(sweep: dict[float, list[bool]]) -> float:
    """The positive alpha with the highest accuracy — measured, not assumed."""
    positives = [a for a in sweep if a > 0.0]
    assert len(positives) >= 3, (
        f"sweep has only {len(positives)} positive alphas; the grid guard should "
        "have prevented this"
    )
    return max(positives, key=lambda a: sum(sweep[a]))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — the soft observation reduces exactly to the hard one
# ══════════════════════════════════════════════════════════════════════════════


@_SKIP_NO_COUNTS
def test_soft_observation_likelihood_reduces_to_the_hard_column(g2p_matrices):
    """A one-hot q must give back the emission row, EXACTLY, for all 30 letters.

    This is the proof that the generalization from hard to soft observation is
    faithful rather than merely plausible. Asserted with ``==`` on the raw bytes,
    not with a tolerance: ``sum_l A[l, p] * onehot(l)`` has exactly one non-zero
    term, so any deviation at all means the marginalization is wrong.
    """
    for letter_idx in range(N_SYMBOLS):
        one_hot = np.zeros(N_SYMBOLS, dtype=np.float64)
        one_hot[letter_idx] = 1.0
        soft = soft_observation_likelihood(g2p_matrices.emission, one_hot)
        hard = g2p_matrices.emission[letter_idx, :]
        assert soft.shape == (N_PHONES,)
        assert np.array_equal(soft, hard), (
            f"Soft likelihood for a one-hot on {IDX_TO_ALPHA[letter_idx]!r} does "
            f"not equal the hard emission column exactly. Max abs difference "
            f"{np.max(np.abs(soft - hard))!r}. The whole coupled decoder is "
            "calibrated against this reduction — if it is inexact, the alpha=0 "
            "anchor is meaningless."
        )


@_SKIP_NO_COUNTS
def test_soft_observation_likelihood_is_a_likelihood_not_a_distribution(g2p_matrices):
    """The returned vector must NOT be normalized over phones.

    Normalizing the evidence would turn a likelihood into a prior and silently
    change the generative model. At least one letter must therefore produce a
    vector whose sum over phones is clearly not 1.
    """
    sums = [
        float(
            soft_observation_likelihood(g2p_matrices.emission, np.eye(N_SYMBOLS)[letter_idx]).sum()
        )
        for letter_idx in range(N_SYMBOLS)
    ]
    assert max(abs(s - 1.0) for s in sums) > 0.01, (
        "Every letter's likelihood vector sums to ~1 over phones. That is the "
        "signature of a normalization having been applied to the evidence, which "
        f"would make it a prior rather than a likelihood. Sums: {sums}"
    )


@_SKIP_NO_COUNTS
def test_soft_observation_likelihood_rejects_malformed_inputs(g2p_matrices):
    """Bad orientation or a non-distribution must raise, not return a number."""
    with pytest.raises(ValueError, match="column-stochastic"):
        soft_observation_likelihood(g2p_matrices.emission * 2.0, np.eye(N_SYMBOLS)[0])
    with pytest.raises(ValueError, match="shape"):
        soft_observation_likelihood(g2p_matrices.emission, np.ones(N_PHONES) / N_PHONES)
    with pytest.raises(ValueError, match="sum to 1"):
        soft_observation_likelihood(g2p_matrices.emission, np.zeros(N_SYMBOLS))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — the mixture, the message, and the scramble
# ══════════════════════════════════════════════════════════════════════════════


@_SKIP_NO_COUNTS
def test_mixture_endpoints_are_exact(letter_matrices):
    """alpha=0 gives the one-hot exactly; alpha=1 gives the predictive exactly."""
    predictive = letter_predictive_distributions(letter_matrices, [ALPHA_TO_IDX[c] for c in "cat"])[
        1
    ]
    index = ALPHA_TO_IDX["a"]

    at_zero = mix_letter_distribution(index, predictive, 0.0)
    expected = np.zeros(N_SYMBOLS)
    expected[index] = 1.0
    assert np.array_equal(at_zero, expected), (
        "alpha=0 did not produce an exact one-hot. The bit-exact alpha=0 anchor depends on this."
    )

    at_one = mix_letter_distribution(index, predictive, 1.0)
    assert np.array_equal(at_one, predictive), (
        "alpha=1 did not return the predictive unchanged, so the sweep's upper "
        "endpoint is not 'listen only to the letter model'."
    )


@_SKIP_NO_COUNTS
def test_mixture_is_always_a_distribution(letter_matrices):
    """Every alpha on the grid must yield a normalized, non-negative q.

    The tolerance is 1e-6, not machine epsilon, and deliberately so: the L1
    transition is built by :func:`ailm.levels.letter.build_letter_transition`, which
    returns a **float32** JAX array (JAX's default). Its columns sum to 1 only to
    about 1e-8 before this module widens them to float64, and the predictive
    inherits that. 1e-6 is the same tolerance
    :mod:`ailm.levels.coupling` validates with, so the test asserts exactly the
    precision the shipped letter model actually has rather than one invented here.
    """
    predictive = letter_predictive_distributions(
        letter_matrices, [ALPHA_TO_IDX[c] for c in "phoneme"]
    )
    for alpha in DEFAULT_ALPHA_GRID:
        for position in range(predictive.shape[0]):
            mixed = mix_letter_distribution(position % N_SYMBOLS, predictive[position], alpha)
            assert mixed.shape == (N_SYMBOLS,)
            assert np.all(mixed >= 0.0)
            assert abs(float(mixed.sum()) - 1.0) < 1e-6


def test_mixture_rejects_bad_alpha_and_index():
    """alpha outside [0, 1], booleans, and out-of-range indices must raise."""
    predictive = np.ones(N_SYMBOLS) / N_SYMBOLS
    for bad in (-0.01, 1.01, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            mix_letter_distribution(0, predictive, bad)
    with pytest.raises(ValueError, match="observed_index"):
        mix_letter_distribution(N_SYMBOLS, predictive, 0.5)


def test_scramble_is_an_entropy_preserving_permutation():
    """The control must preserve entropy and normalization, and be involutive.

    That is what makes it a control on *content* rather than on shape: if
    scrambling also changed the message's peakedness, any difference between real
    and scrambled coupling would be confounded by the entropy change.
    """
    rng = np.random.default_rng(20260729)
    raw = rng.random(N_SYMBOLS) + 0.05
    dist = raw / raw.sum()

    flipped = scramble_letter_distribution(dist)
    assert abs(float(flipped.sum()) - 1.0) < 1e-12
    assert np.array_equal(np.sort(flipped), np.sort(dist)), (
        "scramble is not a permutation of the input mass"
    )
    entropy = -float(np.sum(dist * np.log(dist)))
    flipped_entropy = -float(np.sum(flipped * np.log(flipped)))
    assert abs(entropy - flipped_entropy) < 1e-12
    assert np.array_equal(scramble_letter_distribution(flipped), dist), (
        "reversal is not involutive, so the control is not a fixed permutation"
    )


def test_scramble_actually_moves_mass():
    """A control that is the identity on realiztic input would be vacuous."""
    peaked = np.full(N_SYMBOLS, 0.001)
    peaked[ALPHA_TO_IDX["e"]] = 1.0 - 0.001 * (N_SYMBOLS - 1)
    flipped = scramble_letter_distribution(peaked)
    assert not np.array_equal(flipped, peaked), (
        "reversal left a peaked distribution unchanged — the scramble control "
        "would then be indistinguishable from real coupling by construction"
    )
    assert int(np.argmax(flipped)) == N_SYMBOLS - 1 - ALPHA_TO_IDX["e"]


def test_scramble_rejects_wrong_alphabet_size():
    """A ``(39,)`` phone vector must not be silently accepted as a letter one."""
    with pytest.raises(ValueError, match="letter"):
        scramble_letter_distribution(np.ones(N_PHONES) / N_PHONES)


@_SKIP_NO_COUNTS
def test_letter_message_never_sees_the_current_letter(letter_matrices):
    """``predictive[t]`` must depend on letters ``<t`` only — re-derived here.

    Independently reconstructed from the bigram matrix: under the shipped identity
    ``A`` the L1 posterior is one-hot on the observed letter, so the predictive at
    ``t`` is the bigram column of letter ``t-1``, and at ``t = 0`` it is the
    uniform ``D``. If this drifted so that ``predictive[t]`` peaked on letter
    ``t``, the "coupling adds no information" finding would be an artifact of a
    leak rather than a property of the model.
    """
    word = "coupling"
    indices = [ALPHA_TO_IDX[c] for c in word]
    predictive = letter_predictive_distributions(letter_matrices, indices)

    assert predictive.shape == (len(indices), N_SYMBOLS)
    assert np.allclose(predictive.sum(axis=1), 1.0)
    assert np.allclose(predictive[0], letter_matrices.start), (
        "position 0's message is not the L1 start prior D"
    )
    for t in range(1, len(indices)):
        expected = letter_matrices.transition[:, indices[t - 1]]
        assert np.allclose(predictive[t], expected, atol=1e-12), (
            f"message at t={t} is not the bigram column of letter "
            f"{word[t - 1]!r} — the L1 forward pass is not doing what the "
            "coupling assumes"
        )


@_SKIP_NO_COUNTS
def test_letter_message_is_weak_evidence_about_the_observed_letter(letter_matrices, split):
    """Quantify WHY coupling fails: the message is a poor letter predictor.

    Measured on a bounded stride of held-out words. The message must be better
    than uniform (so the channel is not inert) and far worse than certainty (so
    displacing the observed glyph with it is expected to hurt). Both halves are
    asserted, because only the pair explains the null.
    """
    probs: list[float] = []
    argmax_hits = 0
    for word, _phones in _iter_pairs(split[1], SMALL_WORDS):
        indices = [ALPHA_TO_IDX[c] for c in _word_letters(word)]
        if not indices:
            continue
        predictive = letter_predictive_distributions(letter_matrices, indices)
        for t, index in enumerate(indices):
            probs.append(float(predictive[t, index]))
            argmax_hits += int(np.argmax(predictive[t]) == index)

    assert len(probs) > 500, f"only {len(probs)} positions measured"
    mean_prob = float(np.mean(probs))
    argmax_accuracy = argmax_hits / len(probs)
    uniform_prob = 1.0 / N_SYMBOLS

    assert mean_prob > uniform_prob, (
        f"The L1 message assigns mean {mean_prob:.4f} to the observed letter, no "
        f"better than uniform {uniform_prob:.4f}. The coupling channel would then "
        "be inert and the null result would say nothing about coupling."
    )
    assert argmax_accuracy < 0.5, (
        f"The L1 message's argmax picks the observed letter "
        f"{100 * argmax_accuracy:.1f}% of the time. If it were near-certain, "
        "mixing it in would not be expected to cost accuracy and the explanation "
        "recorded in the module docstring would be wrong."
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — the alpha grid guard
# ══════════════════════════════════════════════════════════════════════════════


def test_default_grid_passes_the_guard():
    """The shipped grid must satisfy its own guard."""
    validated = validate_alpha_grid(DEFAULT_ALPHA_GRID)
    assert validated == tuple(float(a) for a in DEFAULT_ALPHA_GRID)
    assert 0.0 in validated
    assert len({a for a in validated if a > 0.0}) >= 3


def test_grid_guard_rejects_a_missing_anchor():
    """Without 0.0 there is nothing to compare against."""
    with pytest.raises(GateVacuousError, match="0.0"):
        validate_alpha_grid((0.1, 0.25, 0.5, 0.75))


def test_grid_guard_rejects_a_collapsed_sweep():
    """Fewer than three distinct positives is not a sweep."""
    with pytest.raises(GateVacuousError, match="positive"):
        validate_alpha_grid((0.0, 0.5))
    with pytest.raises(GateVacuousError, match="positive"):
        validate_alpha_grid((0.0, 0.5, 0.5, 0.5))


def test_grid_guard_rejects_out_of_range_entries():
    """An alpha outside [0, 1] is not a mixture weight."""
    with pytest.raises(ValueError):
        validate_alpha_grid((0.0, 0.1, 0.5, 1.5))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — THE ANCHOR: alpha = 0 is the uncoupled decode, exactly
# ══════════════════════════════════════════════════════════════════════════════


@_SKIP_NO_COUNTS
@pytest.mark.timeout(300)
def test_alpha_zero_reproduces_uncoupled_decode_exactly(full_sweep, uncoupled_full):
    """ZERO position differences across the full held-out set.

    The two sides are different code paths: the coupled soft recursion in
    :mod:`ailm.levels.coupling` versus
    :func:`ailm.verification.oracle.forward_filter` driven by hard observation
    indices. So this is a differential check between two implementations, not a
    function compared with itself.

    Every other number in this file is a comparison against alpha=0. If the anchor
    is not the shipped uncoupled decoder, all of them describe a different model
    and none of them mean anything.
    """
    coupled = full_sweep[0.0]
    assert len(coupled) == len(uncoupled_full), (
        f"Position counts differ: coupled={len(coupled)} "
        f"uncoupled={len(uncoupled_full)}. The comparison is not paired."
    )
    assert len(coupled) >= MIN_EVAL_POSITIONS, (
        f"Only {len(coupled)} positions evaluated, below the "
        f"{MIN_EVAL_POSITIONS} floor. A silently truncated evaluation would make "
        "the zero-difference result trivial."
    )
    differences = [
        i for i, (a, b) in enumerate(zip(coupled, uncoupled_full, strict=True)) if a != b
    ]
    assert differences == [], (
        f"alpha=0 disagrees with the uncoupled oracle decode at "
        f"{len(differences)} of {len(coupled)} positions (first few: "
        f"{differences[:10]}). The soft-observation generalization is not faithful."
    )


@_SKIP_NO_COUNTS
@pytest.mark.timeout(300)
def test_anchor_is_not_vacuous_alpha_above_zero_does_differ(full_sweep, uncoupled_full):
    """POSITIVE CONTROL on the anchor: a positive alpha MUST change the decode.

    Without this, the zero at alpha=0 would also be produced by a coupling that
    ignored ``alpha`` entirely, or by a comparison that never looked.
    """
    coupled = full_sweep[_SMALLEST_POSITIVE_ALPHA]
    differences = sum(1 for a, b in zip(coupled, uncoupled_full, strict=True) if a != b)
    assert differences > 0, (
        f"alpha={_SMALLEST_POSITIVE_ALPHA} produced a decode identical to the "
        "uncoupled one at every position. The alpha parameter is not reaching the "
        "likelihood, so the zero-difference anchor above proves nothing."
    )


@_SKIP_NO_COUNTS
@pytest.mark.timeout(300)
def test_scrambled_control_coincides_with_real_coupling_at_the_anchor(full_sweep, scrambled_sweep):
    """At alpha=0 the message is unused, so the control must be identical there.

    A scramble that changed the alpha=0 decode would mean the message leaks into
    the likelihood even at zero weight.
    """
    assert full_sweep[0.0] == scrambled_sweep[0.0], (
        "The scrambled control differs from real coupling at alpha=0, where the "
        "message carries zero weight. The message is leaking."
    )


@_SKIP_NO_COUNTS
@pytest.mark.timeout(300)
def test_alpha_zero_matches_the_pymdp_agent(letter_matrices, g2p_matrices, split):
    """Bind the numpy anchor to the shipped pymdp production path.

    The sweep is computed in numpy because a full-set pymdp pass is ~42.8k
    ``infer_states`` calls. That substitution is only licensed while the two
    decoders are provably identical, which is what this bounded check establishes —
    the same argument ``scripts/record_stage2_claims.py`` makes for the Stage 2
    figures.
    """
    agent = build_g2p_agent(
        build_g2p_emission(split[0]),
        build_g2p_transition(split[0]),
        build_g2p_start(split[0]),
    )
    subset = _iter_pairs(split[1], PYMDP_WORDS)
    assert len(subset) == PYMDP_WORDS

    mismatched_words: list[str] = []
    positions = 0
    for word, _phones in subset:
        coupled, _diagnostics = couple_letter_to_phoneme(
            letter_matrices, g2p_matrices, word, alpha=0.0
        )
        pymdp_phones = decode_phonemes(agent, word)
        positions += len(pymdp_phones)
        if coupled != pymdp_phones:
            mismatched_words.append(word)

    assert positions > 0, "no positions decoded — the subset resolved to nothing"
    assert mismatched_words == [], (
        f"The alpha=0 coupled decode disagrees with the pymdp G2P agent on "
        f"{len(mismatched_words)} of {len(subset)} words "
        f"(e.g. {mismatched_words[:5]}). The numpy sweep does not describe the "
        "shipped agent, so no number derived from it may be recorded."
    )


@_SKIP_NO_COUNTS
@pytest.mark.timeout(300)
def test_alpha_zero_accuracy_matches_the_recorded_stage2_number(full_sweep):
    """The anchor must equal the Stage 2 G2P accuracy already in the ledger.

    Binds this file's baseline to the shipped, independently recorded Stage 2
    measurement instead of to a number produced inside this file.
    """
    if not LEDGER_FILE.exists():
        pytest.skip("data/claims.json absent")
    ledger = load_ledger(PROJECT_ROOT)
    entry = ledger.get("stage2_g2p_accuracy")
    if entry is None:
        pytest.skip("stage2_g2p_accuracy not recorded; run record_stage2_claims.py")

    measured = _accuracy(full_sweep[0.0])
    recorded = float(entry["value"])
    assert abs(measured - recorded) < 0.01, (
        f"alpha=0 measured {measured:.4f}% but the ledger records the shipped "
        f"Stage 2 G2P accuracy as {recorded}%. The coupling anchor and the "
        "uncoupled model have drifted apart."
    )


@_SKIP_NO_COUNTS
def test_coupled_positions_align_with_the_context_free_baseline(
    letter_matrices, g2p_matrices, split
):
    """Same length, same order as ``baseline_positions`` for the same limit.

    Both resolve their subsample through ``phoneme._iter_pairs``. If they did not,
    a paired test between coupling and the context-free bar would silently compare
    different position sets.
    """
    mapping = per_letter_argmax_baseline(split[0])
    coupled = coupled_positions(
        letter_matrices, g2p_matrices, split[1], alpha=0.0, limit=SMALL_WORDS
    )
    baseline = baseline_positions(mapping, split[1], limit=SMALL_WORDS)
    assert len(coupled) == len(baseline) > 0, (
        f"coupled produced {len(coupled)} positions, baseline {len(baseline)}. "
        "The vectors are not paired."
    )


@_SKIP_NO_COUNTS
def test_limit_resolves_as_a_stride_not_a_prefix(letter_matrices, g2p_matrices, split):
    """A prefix of the alphabetically sorted pairs once inflated a margin 3.1x.

    Checked structurally: the subsample the decoder scores must be the one
    ``_iter_pairs`` returns, which spans the alphabet rather than stopping at 'c'.
    """
    subset = _iter_pairs(split[1], SMALL_WORDS)
    initials = {word[0] for word, _ in subset if word}
    assert len(initials) >= 10, (
        f"The bounded subsample covers only {sorted(initials)} — that is a prefix, "
        "not a stride, and every bounded measurement in this file would be biased."
    )
    expected_positions = sum(len(phones) for _word, phones in subset)
    coupled = coupled_positions(
        letter_matrices, g2p_matrices, split[1], alpha=0.5, limit=SMALL_WORDS
    )
    assert len(coupled) == expected_positions, (
        f"Scored {len(coupled)} positions but the stride subsample contains "
        f"{expected_positions}. The decoder is not evaluating the subsample "
        "_iter_pairs defines."
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — THE SWEEP: the finding
# ══════════════════════════════════════════════════════════════════════════════


@_SKIP_NO_COUNTS
@pytest.mark.timeout(300)
def test_alpha_sweep_covers_the_full_grid_with_aligned_vectors(full_sweep):
    """Every grid point must be measured, on identical positions."""
    assert set(full_sweep) == {float(a) for a in DEFAULT_ALPHA_GRID}
    lengths = {len(v) for v in full_sweep.values()}
    assert len(lengths) == 1, (
        f"Sweep vectors have differing lengths {lengths}; no margin computed from "
        "them would be a paired margin."
    )
    assert lengths.pop() >= MIN_EVAL_POSITIONS


@_SKIP_NO_COUNTS
@pytest.mark.timeout(300)
def test_alpha_sweep_is_monotone_decreasing(full_sweep):
    """THE FINDING: accuracy falls monotonically as the message gains weight.

    Not "no significant improvement" — a strict, large, ordered degradation. More
    downward message means worse G2P, all the way to 16.9% at alpha=1 where the
    observed glyph is discarded entirely.

    This assertion is deliberately strong. If a future change to the letter model,
    the emission, or the mixture makes it fail, the correct response is to
    re-derive the finding from the new measurement and rewrite this file's
    docstring — not to weaken the assertion until it passes again.
    """
    ordered = sorted(full_sweep)
    accuracies = [_accuracy(full_sweep[a]) for a in ordered]
    report = "  ".join(f"{a}:{acc:.4f}%" for a, acc in zip(ordered, accuracies, strict=True))

    for earlier, later, acc_earlier, acc_later in zip(
        ordered[:-1], ordered[1:], accuracies[:-1], accuracies[1:], strict=True
    ):
        assert acc_later < acc_earlier, (
            f"Accuracy did not fall from alpha={earlier} ({acc_earlier:.4f}%) to "
            f"alpha={later} ({acc_later:.4f}%). The recorded finding is a strict "
            f"monotone degradation; re-derive it rather than relaxing this. "
            f"Sweep: {report}"
        )


@_SKIP_NO_COUNTS
@pytest.mark.timeout(300)
def test_no_positive_alpha_beats_the_uncoupled_anchor(full_sweep):
    """The gate, stated as the question that was asked: does any alpha>0 win?

    No. The best positive alpha is the smallest on the grid and it still loses.
    """
    anchor = _accuracy(full_sweep[0.0])
    best_alpha = _best_positive_alpha(full_sweep)
    best = _accuracy(full_sweep[best_alpha])
    assert best < anchor, (
        f"alpha={best_alpha} reached {best:.4f}%, at or above the uncoupled "
        f"{anchor:.4f}%. That would invert the recorded finding. Re-run "
        "scripts/record_coupling_claims.py, update data/claims.json, and rewrite "
        "this file's docstring from the new measurement — do not delete this "
        "assertion."
    )
    assert best_alpha == _SMALLEST_POSITIVE_ALPHA, (
        f"The least-damaging alpha is {best_alpha}, not "
        f"{_SMALLEST_POSITIVE_ALPHA}. Monotone degradation implies the smallest "
        "positive alpha is best; the two findings disagree."
    )


@_SKIP_NO_COUNTS
@pytest.mark.timeout(300)
def test_best_positive_alpha_is_significantly_WORSE_than_the_anchor(full_sweep):
    """Paired McNemar on the discordant pairs — significant, in the wrong direction.

    Same procedure as ``scripts/record_stage2_claims.py``: exact two-sided
    binomial test at p=0.5 on the discordant pairs only. Run over the full
    held-out set, because a fraction-of-a-point margin is underpowered on any
    subsample of it — the lesson already recorded in
    ``ailm.levels.phoneme._iter_pairs``.
    """
    anchor = full_sweep[0.0]
    best_alpha = _best_positive_alpha(full_sweep)
    coupled = full_sweep[best_alpha]

    b, c, p_value = mcnemar(coupled, anchor)
    assert b + c > 0, (
        "No discordant pairs at the best positive alpha — the coupled decode is "
        "identical to the anchor, so the sweep measured nothing."
    )
    assert c > b, (
        f"At alpha={best_alpha} the coupled decoder won {b} discordant positions "
        f"and lost {c}. The recorded finding is that coupling loses more than it "
        "wins; re-derive it if this has genuinely changed."
    )
    assert p_value < ALPHA_LEVEL, (
        f"The harm at alpha={best_alpha} (b={b}, c={c}, p={p_value:.3g}) is not "
        f"significant at {ALPHA_LEVEL}. The finding recorded in the ledger claims "
        "significance; a non-significant harm would be a different, weaker "
        "result and must be recorded as such."
    )


@_SKIP_NO_COUNTS
@pytest.mark.timeout(300)
def test_mcnemar_self_comparison_finds_no_discordant_pairs(full_sweep):
    """NEGATIVE CONTROL on the significance machinery: alpha=0 against itself.

    A test that can only ever say "significant" says nothing. Comparing the anchor
    with itself must produce exactly zero discordant pairs and p = 1.0 — if it
    reported an effect here, the p-value above would be an artifact of the
    procedure rather than a property of the data.
    """
    anchor = full_sweep[0.0]
    b, c, p_value = mcnemar(anchor, anchor)
    assert (b, c) == (0, 0), (
        f"Self-comparison found b={b}, c={c} discordant pairs. The pairing is "
        "misaligned, so every McNemar result in this file is void."
    )
    assert p_value == 1.0, f"Self-comparison p-value is {p_value}, not 1.0."


@_SKIP_NO_COUNTS
@pytest.mark.timeout(300)
def test_mcnemar_detects_a_planted_difference(full_sweep):
    """POSITIVE CONTROL on the machinery: a planted flip must be detected.

    Pairs the self-comparison null above. Without it, ``b = c = 0`` could also
    mean the discordant counter is broken.
    """
    anchor = list(full_sweep[0.0])
    planted = list(anchor)
    flipped = 0
    for i, value in enumerate(planted):
        if not value and flipped < 400:
            planted[i] = True
            flipped += 1
    assert flipped == 400, f"only {flipped} positions available to flip"

    b, c, p_value = mcnemar(planted, anchor)
    assert (b, c) == (400, 0), f"planted 400 wins but McNemar saw b={b}, c={c}"
    assert p_value < 1e-50, f"400-0 should be overwhelming, got p={p_value}"


def test_mcnemar_rejects_unpaired_vectors():
    """Different lengths must raise rather than silently truncate."""
    with pytest.raises(ValueError, match="paired"):
        mcnemar([True, False, True], [True, False])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — the scrambled-message control
# ══════════════════════════════════════════════════════════════════════════════


@_SKIP_NO_COUNTS
@pytest.mark.timeout(300)
def test_real_message_beats_scrambled_at_full_weight(full_sweep, scrambled_sweep):
    """NON-VACUITY OF THE CHANNEL, measured where the glyph cannot mask it.

    At alpha=1.0 the observed letter is discarded and the message is the only
    evidence. Real coupling scores far above the reversed-axis control, so the
    channel is demonstrably carrying letter-model information. Without this, the
    null could equally mean the message never reached the likelihood.

    alpha=1.0 is the right place to ask the question precisely because it is the
    only grid point where the two conditions are not both dominated by the exact
    glyph evidence.
    """
    real = _accuracy(full_sweep[1.0])
    scrambled = _accuracy(scrambled_sweep[1.0])
    assert real > scrambled, (
        f"At alpha=1.0 the real message ({real:.4f}%) did not beat the reversed "
        f"one ({scrambled:.4f}%). The coupling channel is then carrying no "
        "letter-model information, and the null result says nothing about "
        "coupling — it says the message never arrived."
    )
    assert real - scrambled > 5.0, (
        f"The real/scrambled separation at alpha=1.0 is only "
        f"{real - scrambled:.4f} points. That is too small to establish that the "
        "channel is informative rather than incidentally different."
    )


@_SKIP_NO_COUNTS
@pytest.mark.timeout(300)
def test_scrambled_message_does_LESS_damage_at_intermediate_alphas(full_sweep, scrambled_sweep):
    """The mechanism, asserted rather than asserted-away.

    At every positive alpha below 1.0 the reversed message costs *less* accuracy
    than the real one. This is not the coupling failing to arrive — the previous
    test rules that out — it is an informative message displacing exact evidence.
    A reversed distribution spreads mass over letters near-irrelevant to the
    observed one, which mostly flattens the likelihood without moving the argmax.
    The real predictive concentrates mass on plausible ALTERNATIVE letters, which
    pulls the phone posterior toward genuinely competing phones and flips
    decisions that were already right.

    This is the honest reason ``coupling_supported`` is 0: the channel works and
    the use of it is harmful.
    """
    intermediate = sorted(a for a in full_sweep if 0.0 < a < 1.0)
    assert len(intermediate) >= 3, f"only {intermediate} intermediate alphas"
    for alpha in intermediate:
        real = _accuracy(full_sweep[alpha])
        scrambled = _accuracy(scrambled_sweep[alpha])
        assert scrambled > real, (
            f"At alpha={alpha} the reversed message ({scrambled:.4f}%) did not do "
            f"less damage than the real one ({real:.4f}%). The mechanism recorded "
            "in this file's docstring is then wrong and must be re-derived."
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — the coupled metric must be able to fail
# ══════════════════════════════════════════════════════════════════════════════


@_SKIP_NO_COUNTS
def test_coupled_decoder_is_live_under_a_broken_emission(letter_matrices, g2p_matrices, split):
    """NEGATIVE CONTROL on the metric: a uniform emission must collapse accuracy.

    A uniform ``A`` carries no letter information at all, so the decode falls back
    to the phone prior. If :func:`coupled_accuracy` did not notice, it would not be
    measuring the model and neither the anchor nor the sweep would mean anything.
    """
    good = coupled_accuracy(letter_matrices, g2p_matrices, split[1], alpha=0.0, limit=SMALL_WORDS)
    broken = G2PMatrices(
        emission=np.full((N_SYMBOLS, N_PHONES), 1.0 / N_SYMBOLS),
        transition=g2p_matrices.transition,
        start=g2p_matrices.start,
    )
    broken_accuracy = coupled_accuracy(
        letter_matrices, broken, split[1], alpha=0.0, limit=SMALL_WORDS
    )
    assert broken_accuracy < good - 0.2, (
        f"An information-free emission scored {100 * broken_accuracy:.2f}% against "
        f"the real model's {100 * good:.2f}%. coupled_accuracy is not sensitive to "
        "the emission, so it is not measuring the model."
    )


@_SKIP_NO_COUNTS
def test_time_reversed_transition_is_caught(letter_matrices, g2p_matrices, split):
    """NEGATIVE CONTROL: a backward-looking phone ``B`` must lose accuracy.

    The Stage 1 mis-wiring in its Stage 2 flavour — "observing a letter tells you
    which phone came *before* it". Re-checked through the coupled decoder because
    the coupled path is a new implementation of the recursion and could have
    reintroduced the transpose.
    """
    good = coupled_accuracy(letter_matrices, g2p_matrices, split[1], alpha=0.0, limit=SMALL_WORDS)
    reversed_transition = g2p_matrices.transition.T
    column_sums = reversed_transition.sum(axis=0, keepdims=True)
    broken = G2PMatrices(
        emission=g2p_matrices.emission,
        transition=reversed_transition / column_sums,
        start=g2p_matrices.start,
    )
    broken_accuracy = coupled_accuracy(
        letter_matrices, broken, split[1], alpha=0.0, limit=SMALL_WORDS
    )
    assert broken_accuracy < good, (
        f"A time-reversed phone transition scored {100 * broken_accuracy:.2f}%, "
        f"not below the correct wiring's {100 * good:.2f}%."
    )


@_SKIP_NO_COUNTS
def test_builder_output_orientation_is_pinned(split):
    """The G2P bundle must be ``(30, 39)`` letters-by-phones, normalized on letters.

    A transposed emission is the mis-wiring Stage 2 documented as *not* caught by
    an accuracy bar, so the orientation is asserted structurally here rather than
    inferred from a score.
    """
    matrices = build_g2p_matrices(split[0])
    assert matrices.emission.shape == (N_SYMBOLS, N_PHONES)
    assert matrices.transition.shape == (N_PHONES, N_PHONES)
    assert matrices.start.shape == (N_PHONES,)
    assert np.allclose(matrices.emission.sum(axis=0), 1.0), (
        "emission columns (over letters) do not sum to 1 — the emission is "
        "normalized over the wrong axis"
    )
    assert np.allclose(matrices.transition.sum(axis=0), 1.0)
    assert abs(float(matrices.start.sum()) - 1.0) < 1e-6


@_SKIP_NO_COUNTS
def test_transposed_emission_is_rejected_by_the_soft_likelihood(g2p_matrices):
    """Feeding ``A.T`` in must raise, not silently marginalize the wrong axis.

    ``A.T`` is ``(39, 30)`` and its columns are the *rows* of ``A``, which do not
    sum to 1. A validator that let this through would happily return a plausible
    39-vector computed from the wrong marginalization.
    """
    with pytest.raises(ValueError, match="column-stochastic"):
        soft_observation_likelihood(g2p_matrices.emission.T, np.eye(N_PHONES)[0])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — diagnostics and edge cases
# ══════════════════════════════════════════════════════════════════════════════


@_SKIP_NO_COUNTS
def test_diagnostics_align_with_the_decode(letter_matrices, g2p_matrices):
    """Diagnostics must be one-per-position and internally consistent."""
    word = "phoneme"
    alpha = 0.25
    phones, diagnostics = couple_letter_to_phoneme(letter_matrices, g2p_matrices, word, alpha=alpha)
    letters = _word_letters(word)

    assert len(phones) == len(diagnostics) == len(letters)
    for position, (phone, diagnostic) in enumerate(zip(phones, diagnostics, strict=True)):
        assert diagnostic.index == position
        assert diagnostic.letter == letters[position]
        assert diagnostic.decoded_phone == phone
        assert phone in PHONES
        assert 0.0 < diagnostic.posterior_max <= 1.0
        assert 0.0 <= diagnostic.letter_predictive_prob <= 1.0
        # mixture mass on the observed letter = (1-alpha) + alpha * predictive
        expected = (1.0 - alpha) + alpha * diagnostic.letter_predictive_prob
        assert abs(diagnostic.mixture_prob_observed - expected) < 1e-12
        assert math.isfinite(diagnostic.log_evidence)
        assert diagnostic.log_evidence <= 0.0 or diagnostic.log_evidence < 1e-9


@_SKIP_NO_COUNTS
def test_internal_mixture_matches_the_public_mixer(letter_matrices, g2p_matrices):
    """Pin the duplication: ``_soft_forward`` inlines the mixture for speed.

    The inlined arithmetic must equal :func:`mix_letter_distribution`, or the
    validated public helper and the hot path could drift apart and the sweep would
    stop measuring what the unit tests check.
    """
    word = "coupling"
    letters = _word_letters(word)
    indices = [ALPHA_TO_IDX[c] for c in letters]
    predictive = letter_predictive_distributions(letter_matrices, indices)

    for alpha in (0.0, 0.1, 0.5, 1.0):
        _phones, diagnostics = couple_letter_to_phoneme(
            letter_matrices, g2p_matrices, word, alpha=alpha
        )
        for position, index in enumerate(indices):
            expected = mix_letter_distribution(index, predictive[position], alpha)
            assert abs(diagnostics[position].mixture_prob_observed - expected[index]) < 1e-15, (
                f"Inlined mixture at alpha={alpha}, position {position} differs "
                "from mix_letter_distribution."
            )


@_SKIP_NO_COUNTS
def test_alpha_zero_likelihood_short_circuit_equals_the_public_helper(g2p_matrices):
    """The alpha=0 fast path must equal the marginalization, not approximate it."""
    for letter in ("a", "e", "z", " "):
        index = ALPHA_TO_IDX[letter]
        one_hot = np.zeros(N_SYMBOLS)
        one_hot[index] = 1.0
        assert np.array_equal(
            soft_observation_likelihood(g2p_matrices.emission, one_hot),
            g2p_matrices.emission[index, :],
        )


@_SKIP_NO_COUNTS
def test_empty_word_returns_empty(letter_matrices, g2p_matrices):
    """A word with no retained letters must not raise or fabricate a phone."""
    phones, diagnostics = couple_letter_to_phoneme(letter_matrices, g2p_matrices, "", alpha=0.5)
    assert phones == []
    assert diagnostics == []


@_SKIP_NO_COUNTS
def test_coupled_positions_rejects_a_bad_alpha(letter_matrices, g2p_matrices, split):
    """An out-of-range alpha must raise before any decoding happens."""
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        coupled_positions(letter_matrices, g2p_matrices, split[1], alpha=1.5, limit=10)


@_SKIP_NO_COUNTS
def test_zero_limit_measures_nothing_and_says_so(letter_matrices, g2p_matrices, split):
    """``limit=0`` must yield an empty vector and 0.0 accuracy, not a silent pass."""
    assert coupled_positions(letter_matrices, g2p_matrices, split[1], alpha=0.5, limit=0) == []
    assert coupled_accuracy(letter_matrices, g2p_matrices, split[1], alpha=0.5, limit=0) == 0.0


@_SKIP_NO_COUNTS
def test_letter_matrices_mirror_the_shipped_letter_agent():
    """The L1 bundle must be the shipped model, not a look-alike.

    ``start`` must be the uniform ``D`` that
    :func:`ailm.levels.letter.build_letter_agent` installs, and ``transition`` must
    be exactly :func:`ailm.levels.letter.build_letter_transition`'s output. A different
    start prior would silently change position 0 of every coupled decode.
    """
    from ailm.levels.letter import build_letter_transition

    brown = load_brown_counts(VENDOR_DIR)
    matrices = build_letter_matrices(brown)
    assert matrices.transition.shape == (N_SYMBOLS, N_SYMBOLS)
    assert np.allclose(matrices.start, 1.0 / N_SYMBOLS)
    expected = np.asarray(build_letter_transition(brown), dtype=np.float64)[0]
    assert np.allclose(matrices.transition, expected, atol=1e-12)
    assert len(ALPHA_LIST) == N_SYMBOLS


def test_letter_predictive_rejects_an_empty_sequence(letter_matrices):
    """An empty index sequence has no position 0 to predict."""
    with pytest.raises(ValueError, match="non-empty"):
        letter_predictive_distributions(letter_matrices, [])


def test_letter_matrices_validate_their_inputs():
    """A transition that is not a finite stochastic matrix must be refused.

    ``build_letter_transition`` add-1 smooths and then normalizes, so the only way to make
    it produce a non-matrix is to poison a count. The column-stochastic branch of
    the same validator is exercised by
    :func:`test_transposed_emission_is_rejected_by_the_soft_likelihood`.
    """
    with pytest.raises(ValueError, match="NaN or infinite"):
        build_letter_matrices({"a,b": float("nan")})


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — the ledger records the null
# ══════════════════════════════════════════════════════════════════════════════


def _coupling_ledger_entry(label: str):
    """Return a recorded coupling claim, or skip if it has not been recorded."""
    if not LEDGER_FILE.exists():
        pytest.skip("data/claims.json absent")
    entry = load_ledger(PROJECT_ROOT).get(label)
    if entry is None:
        pytest.skip(
            f"{label} not recorded yet. Run .venv/bin/python scripts/record_coupling_claims.py"
        )
    return entry


@_SKIP_NO_COUNTS
@pytest.mark.timeout(300)
def test_recorded_claims_match_a_fresh_measurement(full_sweep, scrambled_sweep):
    """The ledger must describe this measurement, not an older one.

    A measured null is still a result and is still recorded. This test is what
    stops the recorded null from drifting away from the code that produced it.
    """
    anchor_entry = _coupling_ledger_entry("coupling_alpha0_accuracy")
    best_entry = _coupling_ledger_entry("coupling_best_accuracy")
    alpha_entry = _coupling_ledger_entry("coupling_best_alpha")
    margin_entry = _coupling_ledger_entry("coupling_margin_pts")
    positions_entry = _coupling_ledger_entry("coupling_eval_positions")
    scrambled_entry = _coupling_ledger_entry("coupling_scrambled_accuracy")

    best_alpha = _best_positive_alpha(full_sweep)
    anchor = _accuracy(full_sweep[0.0])
    best = _accuracy(full_sweep[best_alpha])

    assert float(alpha_entry["value"]) == best_alpha
    assert abs(float(anchor_entry["value"]) - anchor) < 0.01
    assert abs(float(best_entry["value"]) - best) < 0.01
    assert abs(float(margin_entry["value"]) - (best - anchor)) < 0.01
    assert int(positions_entry["value"]) == len(full_sweep[0.0])
    assert abs(float(scrambled_entry["value"]) - _accuracy(scrambled_sweep[best_alpha])) < 0.01


def test_recorded_margin_is_negative_and_support_flag_is_zero():
    """The recorded verdict must say "not supported" in both places.

    A negative margin recorded alongside ``coupling_supported = 1`` would be the
    kind of internally inconsistent evidence this project treats as a defect: the
    summary flag is what a reader trusts, so it must follow the number.
    """
    margin = float(_coupling_ledger_entry("coupling_margin_pts")["value"])
    supported = int(_coupling_ledger_entry("coupling_supported")["value"])
    assert margin < 0.0, (
        f"Recorded coupling margin is {margin} pts. The finding in this file's "
        "docstring is that coupling costs accuracy."
    )
    assert supported == 0, (
        f"coupling_supported is {supported} while the recorded margin is "
        f"{margin} pts. The flag and the measurement disagree."
    )


def test_recorded_mcnemar_p_is_significant_in_the_direction_of_harm():
    """The recorded p-value must be the harm p-value, not a null one."""
    p_value = float(_coupling_ledger_entry("coupling_mcnemar_p")["value"])
    assert 0.0 < p_value < ALPHA_LEVEL, (
        f"Recorded coupling McNemar p = {p_value}, which does not establish the "
        "significant degradation this file documents."
    )


def test_recorded_identity_and_channel_guards_are_clean():
    """The recorded run must show the guards actually passed, not merely ran.

    ``coupling_alpha0_identity_diffs`` and ``coupling_pymdp_disagreements`` are
    recorded so a reader can see the anchor held in the run that produced the
    numbers. Recording them as anything other than 0 alongside a set of accuracy
    claims would be internally inconsistent evidence.
    """
    assert int(_coupling_ledger_entry("coupling_alpha0_identity_diffs")["value"]) == 0
    assert int(_coupling_ledger_entry("coupling_pymdp_disagreements")["value"]) == 0
    assert int(_coupling_ledger_entry("coupling_null_mcnemar_discordant")["value"]) == 0
    assert int(_coupling_ledger_entry("coupling_grid_size")["value"]) == len(DEFAULT_ALPHA_GRID)
    real = float(_coupling_ledger_entry("coupling_alpha1_accuracy")["value"])
    scrambled = float(_coupling_ledger_entry("coupling_alpha1_scrambled_accuracy")["value"])
    assert real > scrambled, (
        f"The recorded alpha=1 channel check has real {real}% <= scrambled "
        f"{scrambled}%. The recorded null would then describe an inert channel."
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — the error paths must actually fire
# ══════════════════════════════════════════════════════════════════════════════


@_SKIP_NO_COUNTS
def test_emission_validator_rejects_structurally_bad_matrices(g2p_matrices):
    """A 1-D, empty, or negative "emission" must raise, not be broadcast."""
    one_hot = np.eye(N_SYMBOLS)[0]
    with pytest.raises(ValueError, match="2-D"):
        soft_observation_likelihood(g2p_matrices.emission[0], one_hot)
    with pytest.raises(ValueError, match="non-empty"):
        soft_observation_likelihood(np.zeros((0, N_PHONES)), one_hot)
    negative = g2p_matrices.emission.copy()
    negative[0, 0] -= 2.0
    negative[1, 0] += 2.0  # keep the column sum at 1 so only the sign is wrong
    with pytest.raises(ValueError, match="negative"):
        soft_observation_likelihood(negative, one_hot)


@_SKIP_NO_COUNTS
def test_distribution_validator_rejects_nan_and_negative_mass(g2p_matrices):
    """A message containing NaN or negative mass is not a distribution."""
    poisoned = np.full(N_SYMBOLS, 1.0 / N_SYMBOLS)
    poisoned[0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        soft_observation_likelihood(g2p_matrices.emission, poisoned)

    signed = np.full(N_SYMBOLS, 1.0 / N_SYMBOLS)
    signed[0] = -0.5
    signed[1] = 0.5 + 1.0 / N_SYMBOLS  # total still 1.0
    with pytest.raises(ValueError, match="negative"):
        soft_observation_likelihood(g2p_matrices.emission, signed)


def test_alpha_validator_rejects_non_numeric():
    """A string alpha must raise a ValueError, not a TypeError from arithmetic."""
    predictive = np.ones(N_SYMBOLS) / N_SYMBOLS
    with pytest.raises(ValueError, match="real number"):
        mix_letter_distribution(0, predictive, "half")


@_SKIP_NO_COUNTS
def test_single_letter_word_decodes_without_a_transition_step(letter_matrices, g2p_matrices):
    """A one-position word exercises the ``T == 1`` branch of the message pass.

    With a single letter there is no ``t >= 1`` predictive to build, so the
    message is the L1 start prior alone and the phone prior is never pushed
    through ``B``.
    """
    phones, diagnostics = couple_letter_to_phoneme(letter_matrices, g2p_matrices, "a", alpha=0.5)
    assert len(phones) == len(diagnostics) == 1
    assert phones[0] in PHONES
    predictive = letter_predictive_distributions(letter_matrices, [ALPHA_TO_IDX["a"]])
    assert predictive.shape == (1, N_SYMBOLS)
    assert np.allclose(predictive[0], letter_matrices.start)
    assert abs(diagnostics[0].letter_predictive_prob - 1.0 / N_SYMBOLS) < 1e-6


@_SKIP_NO_COUNTS
def test_zero_evidence_position_degrades_to_the_predictive(letter_matrices, g2p_matrices):
    """A letter no phone can emit must not produce NaN or crash.

    Constructed by zeroing one row of the emission while keeping every column
    summing to 1, so the matrix is still valid but letter ``q`` has zero
    likelihood under every phone. The decoder must fall back to the phone
    predictive — the same policy
    :func:`ailm.verification.oracle.forward_filter` uses — rather than dividing by
    zero. Cannot arise with the add-1 smoothed shipped matrices; this pins the
    behavior for any future unsmoothed model.
    """
    emission = g2p_matrices.emission.copy()
    dead_letter = ALPHA_TO_IDX["q"]
    mass = emission[dead_letter, :].copy()
    emission[dead_letter, :] = 0.0
    # Redistribute the removed mass onto another letter so columns still sum to 1.
    emission[ALPHA_TO_IDX["e"], :] += mass

    degenerate = G2PMatrices(
        emission=emission,
        transition=g2p_matrices.transition,
        start=g2p_matrices.start,
    )
    phones, diagnostics = couple_letter_to_phoneme(letter_matrices, degenerate, "q", alpha=0.0)
    assert len(phones) == 1
    assert phones[0] in PHONES
    assert math.isfinite(diagnostics[0].posterior_max)
    assert not math.isnan(diagnostics[0].log_evidence)


@_SKIP_NO_COUNTS
def test_length_mismatch_raises_in_both_position_functions(letter_matrices, g2p_matrices):
    """A pair whose phone count disagrees with its letters is a bug, not data.

    Tolerating it by skipping would silently shrink the evaluation set and
    de-pair the vectors the McNemar test depends on.
    """
    malformed = [("cat", ["K", "AE"])]
    with pytest.raises(ValueError, match="length"):
        coupled_positions(letter_matrices, g2p_matrices, malformed, alpha=0.0)
    with pytest.raises(ValueError, match="count"):
        uncoupled_positions(g2p_matrices, malformed)


@_SKIP_NO_COUNTS
def test_duplicate_alphas_collapse_to_one_measurement(letter_matrices, g2p_matrices, split):
    """A repeated grid entry must not be decoded twice or produce two keys."""
    grid = (0.0, 0.0, 0.1, 0.25, 0.5)
    result = sweep_alpha(letter_matrices, g2p_matrices, split[1], grid, limit=SMALL_WORDS)
    assert set(result) == {0.0, 0.1, 0.25, 0.5}
