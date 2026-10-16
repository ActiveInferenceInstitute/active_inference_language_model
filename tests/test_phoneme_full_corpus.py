"""Tests for the closed-form full-dictionary Part A perplexity.

``full_corpus_perplexity`` replaces a sequence walk with an algebraic identity, so
the tests have to establish that the identity actually holds rather than merely
that the function returns a plausible number. Two independent checks do that:

1. **Analytic anchors** — a uniform transition must give exactly N_PHONES, and a
   deterministic chain must give exactly 1. Both are derivable on paper.
2. **Agreement with a sequence walk** — on hand-built corpora where the exact
   positions are known, the closed form must equal a literal step-by-step
   computation.

Without (2) the closed form could be self-consistently wrong; without (1) an
indexing transpose could survive because both computations share the bug.
"""

from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from ailm.levels.phoneme import (
    N_PHONES,
    PHONE_TO_IDX,
    PHONES,
    build_phone_transition,
    build_phone_unigram,
    full_corpus_order1_floor_perplexity,
    full_corpus_perplexity,
    full_corpus_unigram_perplexity,
    load_phoneme_counts,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor"
COUNTS_FILE = VENDOR_DIR / "cmudict_counts.json"

_SKIP_NO_COUNTS = pytest.mark.skipif(
    not COUNTS_FILE.exists(),
    reason="vendor/cmudict_counts.json absent; run scripts/generate_phoneme_counts.py",
)

#: Comparison tolerance. ``build_phone_transition`` returns a JAX array, and JAX
#: defaults to float32, so agreement is bounded at roughly 1e-7 relative no matter
#: how exact the algebra is. Asserting tighter than the arithmetic can deliver
#: produces a test that fails on correct code.
FLOAT32_REL = 1e-6


def _counts_from_sequences(sequences: list[list[str]]) -> dict:
    """Build a minimal artifact from explicit phone sequences.

    Mirrors scripts/generate_phoneme_counts.py so the test exercises the same
    statistics the real artifact carries.
    """
    unigrams: dict[str, int] = {}
    bigrams: dict[str, int] = {}
    initials: dict[str, int] = {}
    for seq in sequences:
        if not seq:
            continue
        initials[seq[0]] = initials.get(seq[0], 0) + 1
        for phone in seq:
            unigrams[phone] = unigrams.get(phone, 0) + 1
        for a, b in pairwise(seq):
            key = f"{a},{b}"
            bigrams[key] = bigrams.get(key, 0) + 1
    return {
        "phone_unigrams": unigrams,
        "phone_bigrams": bigrams,
        "phone_initials": initials,
    }


def _sequence_walk_perplexity(counts: dict, sequences: list[list[str]]) -> float:
    """Literal step-by-step perplexity: the oracle for the closed form.

    Deliberately written as a naive loop over positions, resetting the prior at
    every sequence boundary, so it shares no code with the closed form.
    """
    transition = np.asarray(build_phone_transition(counts["phone_bigrams"]))[0, :, :, 0]
    start = np.asarray(build_phone_unigram(counts["phone_unigrams"])).ravel()

    total_logp = 0.0
    positions = 0
    for seq in sequences:
        prior = start.copy()
        for phone in seq:
            idx = PHONE_TO_IDX[phone]
            total_logp += math.log(max(float(prior[idx]), 1e-300))
            positions += 1
            posterior = np.zeros(N_PHONES)
            posterior[idx] = 1.0
            prior = transition @ posterior
    return math.exp(-total_logp / positions)


# ── Analytic anchors ─────────────────────────────────────────────────────────


class TestAnalyticAnchors:
    """Values derivable on paper, so a transpose or an off-by-one cannot hide."""

    def test_uniform_everything_gives_exactly_n_phones(self):
        """Uniform start and uniform transition: every prediction is 1/39."""
        # One occurrence of every bigram and every initial makes both tables uniform
        # after add-1 smoothing has equal mass everywhere.
        counts = {
            "phone_unigrams": {p: 1 for p in PHONES},
            "phone_bigrams": {f"{a},{b}": 1 for a in PHONES for b in PHONES},
            "phone_initials": {p: 1 for p in PHONES},
        }
        perplexity, positions = full_corpus_perplexity(counts)
        assert positions == len(PHONES) + len(PHONES) ** 2
        assert perplexity == pytest.approx(float(N_PHONES), rel=1e-6)

    def test_near_deterministic_chain_matches_its_exact_closed_form(self):
        """A chain with overwhelming mass on one successor, checked against algebra.

        The expected value is derived here rather than stapled. With ``k``
        word-initial occurrences of P0 and ``k`` P0->P1 transitions there are ``2k``
        positions, half scored under the start prior and half under the transition,
        so with add-1 smoothing over 39 phones:

            D[P0] = (k+1) / (2k + 39)        # unigrams are {P0: k, P1: k}
            B[P1|P0] = (k+1) / (k + 39)      # only P0->P1 was ever observed
            perplexity = 1 / sqrt(D[P0] * B[P1|P0])

        The transition term alone approaches 1, but the start-prior term does not —
        forgetting it is an easy way to write a wrong "should be near 1" assertion.
        """
        k = 100000
        counts = {
            "phone_unigrams": {PHONES[0]: k, PHONES[1]: k},
            "phone_bigrams": {f"{PHONES[0]},{PHONES[1]}": k},
            "phone_initials": {PHONES[0]: k},
        }
        d_p0 = (k + 1) / (2 * k + N_PHONES)
        b_p1_given_p0 = (k + 1) / (k + N_PHONES)
        expected = 1.0 / math.sqrt(d_p0 * b_p1_given_p0)

        perplexity, positions = full_corpus_perplexity(counts)
        assert positions == 2 * k
        assert perplexity == pytest.approx(expected, rel=FLOAT32_REL)
        # The transition is near-certain even though the whole sequence is not.
        assert b_p1_given_p0 > 0.999
        assert perplexity < 2.0

    def test_positions_counted_are_initials_plus_bigrams(self):
        counts = _counts_from_sequences([["AA", "B", "K"], ["D", "EH"]])
        _, positions = full_corpus_perplexity(counts)
        expected = sum(counts["phone_initials"].values()) + sum(counts["phone_bigrams"].values())
        assert positions == expected == 5

    def test_unsmoothed_floor_scores_starts_and_transitions_on_same_denominator(self):
        sequences = [["AA", "B"], ["AA", "B"], ["B", "AA"]]
        counts = _counts_from_sequences(sequences)
        floor, positions = full_corpus_order1_floor_perplexity(counts)
        start_loss = -(2 * math.log(2 / 3) + math.log(1 / 3))
        assert positions == 6
        assert floor == pytest.approx(math.exp(start_loss / positions))
        trained, trained_positions = full_corpus_perplexity(counts)
        assert trained_positions == positions
        assert trained >= floor


# ── Agreement with an independent sequence walk ──────────────────────────────


class TestClosedFormEqualsSequenceWalk:
    """The algebraic identity must hold against a literal position-by-position walk."""

    @pytest.mark.parametrize(
        "sequences",
        [
            [["AA", "B"]],
            [["AA", "B", "K"], ["D", "EH", "F"]],
            [["K", "AE", "T"], ["K", "AE", "T"], ["D", "AO", "G"]],
            [["AA"], ["B"], ["K"]],  # all word-initial, no transitions
            [["S", "T", "R", "IH", "NG"]],  # one long word
        ],
    )
    def test_closed_form_matches_walk(self, sequences):
        counts = _counts_from_sequences(sequences)
        closed, positions = full_corpus_perplexity(counts)
        walked = _sequence_walk_perplexity(counts, sequences)
        assert positions == sum(len(s) for s in sequences)
        assert closed == pytest.approx(walked, rel=FLOAT32_REL), (
            f"closed form {closed} != sequence walk {walked}"
        )

    def test_single_phone_words_are_scored_under_the_start_prior(self):
        """With no transitions at all, perplexity is purely the start-prior term."""
        sequences = [["AA"], ["AA"], ["B"]]
        counts = _counts_from_sequences(sequences)
        assert counts["phone_bigrams"] == {}
        closed, positions = full_corpus_perplexity(counts)
        assert positions == 3
        assert closed == pytest.approx(
            _sequence_walk_perplexity(counts, sequences), rel=FLOAT32_REL
        )


# ── Guard behavior ─────────────────────────────────────────────────────────


class TestMissingInitialsGuard:
    """Absent sufficient statistics must raise, not silently use the biased subset."""

    def test_missing_phone_initials_raises(self):
        counts = {"phone_unigrams": {"AA": 1}, "phone_bigrams": {"AA,B": 1}}
        with pytest.raises(ValueError, match="phone_initials"):
            full_corpus_perplexity(counts)

    def test_missing_phone_initials_raises_for_unigram_bar_too(self):
        counts = {"phone_unigrams": {"AA": 1}, "phone_bigrams": {"AA,B": 1}}
        with pytest.raises(ValueError, match="phone_initials"):
            full_corpus_unigram_perplexity(counts)

    def test_missing_phone_initials_raises_for_order1_floor(self):
        counts = {"phone_unigrams": {"AA": 1}, "phone_bigrams": {"AA,B": 1}}
        with pytest.raises(ValueError, match="phone_initials"):
            full_corpus_order1_floor_perplexity(counts)

    @pytest.mark.parametrize(
        "counts",
        (
            {"phone_initials": {"XX": 1}, "phone_bigrams": {}},
            {"phone_initials": {"AA": -1}, "phone_bigrams": {}},
            {"phone_initials": {"AA": 1}, "phone_bigrams": {"malformed": 1}},
            {"phone_initials": {"AA": 1}, "phone_bigrams": {"AA,B": True}},
        ),
    )
    def test_order1_floor_rejects_omitted_or_malformed_events(self, counts):
        with pytest.raises(ValueError):
            full_corpus_order1_floor_perplexity(counts)

    def test_guard_message_names_the_regeneration_script(self):
        counts = {"phone_unigrams": {}, "phone_bigrams": {}}
        with pytest.raises(ValueError, match="generate_phoneme_counts"):
            full_corpus_perplexity(counts)


# ── Malformed and empty inputs ───────────────────────────────────────────────


class TestSkipsUnscorablePositionsInsteadOfCrashing:
    """The two closed-form functions filter their inputs; pin what they filter.

    Both walk ``phone_initials`` and ``phone_bigrams`` and skip entries they
    cannot score — an unknown symbol, or a key that is not a well-formed pair.
    Untested, that filtering is indistinguishable from a silent miscount, because
    a skipped position lowers the denominator without any signal.
    """

    #: One real phone so the position count is non-zero and identifiable.
    BASE_UNIGRAMS = {"AA": 10, "B": 10}

    def test_unknown_initial_symbol_is_excluded_from_the_position_count(self):
        clean = {
            "phone_unigrams": self.BASE_UNIGRAMS,
            "phone_bigrams": {},
            "phone_initials": {"AA": 7},
        }
        polluted = {
            **clean,
            "phone_initials": {"AA": 7, "XX": 1000, "not_a_phone": 5},
        }
        _, clean_positions = full_corpus_perplexity(clean)
        polluted_ppl, polluted_positions = full_corpus_perplexity(polluted)

        assert clean_positions == 7
        assert polluted_positions == 7, (
            "1005 unscorable initial positions leaked into the denominator"
        )
        assert polluted_ppl == full_corpus_perplexity(clean)[0]

    def test_malformed_bigram_key_is_skipped(self):
        """A key with no comma is not a pair; it must not be scored."""
        counts = {
            "phone_unigrams": self.BASE_UNIGRAMS,
            "phone_bigrams": {"AA,B": 4, "no_comma_here": 99, "": 3},
            "phone_initials": {"AA": 2},
        }
        _, positions = full_corpus_perplexity(counts)
        # 2 initials + only the single well-formed bigram's 4 counts.
        assert positions == 6

    def test_bigram_with_an_unknown_symbol_is_skipped(self):
        counts = {
            "phone_unigrams": self.BASE_UNIGRAMS,
            "phone_bigrams": {"AA,B": 4, "AA,ZZZ": 50, "ZZZ,B": 50},
            "phone_initials": {"AA": 2},
        }
        _, positions = full_corpus_perplexity(counts)
        assert positions == 6

    def test_unigram_bar_applies_the_same_filtering(self):
        """The bar must score the identical position multiset, or it is a different corpus."""
        counts = {
            "phone_unigrams": self.BASE_UNIGRAMS,
            "phone_bigrams": {"AA,B": 4, "broken": 99, "AA,ZZZ": 50},
            "phone_initials": {"AA": 2, "XX": 77},
        }
        model, positions = full_corpus_perplexity(counts)
        bar = full_corpus_unigram_perplexity(counts)
        assert positions == 6
        assert math.isfinite(model) and math.isfinite(bar)
        assert model > 0.0 and bar > 0.0

    def test_no_scorable_position_returns_the_uniform_value(self):
        """Zero positions must yield the uniform perplexity, not a ZeroDivisionError."""
        counts = {
            "phone_unigrams": self.BASE_UNIGRAMS,
            "phone_bigrams": {},
            "phone_initials": {},
        }
        perplexity, positions = full_corpus_perplexity(counts)
        assert positions == 0
        assert perplexity == float(N_PHONES) == 39.0
        assert full_corpus_unigram_perplexity(counts) == float(N_PHONES)

    def test_no_scorable_position_when_every_entry_is_unusable(self):
        """The same, reached through filtering rather than through empty dicts."""
        counts = {
            "phone_unigrams": self.BASE_UNIGRAMS,
            "phone_bigrams": {"malformed": 9, "AA,ZZZ": 9},
            "phone_initials": {"XX": 9},
        }
        perplexity, positions = full_corpus_perplexity(counts)
        assert positions == 0
        assert perplexity == float(N_PHONES)
        assert full_corpus_unigram_perplexity(counts) == float(N_PHONES)


# ── Against the real artifact ───────────────────────────────────────────────


class TestRealArtifact:
    """The shipped artifact must support an unbiased full-dictionary measurement."""

    @_SKIP_NO_COUNTS
    def test_artifact_carries_phone_initials(self):
        counts = load_phoneme_counts(VENDOR_DIR)
        assert "phone_initials" in counts, (
            "The vendored artifact predates phone_initials; regenerate it or Part A "
            "can only be measured on the biased length-matched subset."
        )
        assert len(counts["phone_initials"]) > 20

    @_SKIP_NO_COUNTS
    def test_positions_cover_the_whole_dictionary(self):
        """Every phone of every base word must be scored exactly once."""
        counts = load_phoneme_counts(VENDOR_DIR)
        _, positions = full_corpus_perplexity(counts)
        n_words = sum(counts["phone_initials"].values())
        assert n_words == counts["source_base_words"], (
            f"{n_words} word-initial positions but the artifact reports "
            f"{counts['source_base_words']} base words — some words were dropped."
        )
        assert positions == n_words + sum(counts["phone_bigrams"].values())
        assert positions > 700000

    @_SKIP_NO_COUNTS
    def test_model_beats_the_context_free_bar_on_identical_positions(self):
        """The Part A gate, on the full dictionary rather than a flattering subset."""
        counts = load_phoneme_counts(VENDOR_DIR)
        model, _ = full_corpus_perplexity(counts)
        bar = full_corpus_unigram_perplexity(counts)
        assert model < bar, (
            f"full-corpus perplexity {model:.4f} does not beat the context-free "
            f"unigram bar {bar:.4f}"
        )
        assert bar < float(N_PHONES), "the unigram bar should itself beat uniform"

    @_SKIP_NO_COUNTS
    def test_full_corpus_is_harder_than_the_length_matched_subset(self):
        """Pins the bias this function exists to avoid.

        If the subset ever became the *harder* corpus this reasoning would need
        revisiting, so the direction is asserted rather than assumed.
        """
        counts = load_phoneme_counts(VENDOR_DIR)
        full, _ = full_corpus_perplexity(counts)
        subset_sequences = [s.split() for _, s in counts["g2p_pairs"]]
        subset = _sequence_walk_perplexity(counts, subset_sequences)
        assert subset < full, (
            f"The length-matched subset ({subset:.4f}) is expected to be EASIER than "
            f"the full dictionary ({full:.4f}); it measured harder, so the "
            f"subsample-bias argument in the docstring needs revisiting."
        )

    @_SKIP_NO_COUNTS
    def test_deterministic(self):
        counts = load_phoneme_counts(VENDOR_DIR)
        assert full_corpus_perplexity(counts) == full_corpus_perplexity(counts)
