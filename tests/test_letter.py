"""Unit tests for ailm.levels.letter — AILM Stage 1 letter-level agent.

Doctrine
--------
* Real pymdp only. No mocks, no monkeypatched dependencies, no stubbed arrays.
* Every numeric assertion is either an exact closed form (uniform dynamics ⇒
  perplexity exactly 30) or a value re-derived inside the test from the same
  inputs — never a hand-typed number copied from a previous run.
* Structural checks (shape, dtype, key parsing) are kept separate from
  behavioral checks (perplexity ordering) so a green structural block can never
  be read as evidence that the model learned anything.

Where the corpus statistics live
--------------------------------
``build_letter_transition`` returns a column-stochastic (1, 30, 30) bigram matrix, and
``build_letter_agent`` installs it as the pymdp **transition** B, with an
identity emission A. Several tests below pin that orientation directly, because
the inverse wiring (bigram in A, identity B) is silently anti-informative rather
than loudly broken.

Model-level quality gates — does the agent beat a unigram baseline, does its
inference match the closed-form bigram chain — live in test_stage1_gate.py.
This file covers the API surface of the module.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from ailm.corpus.brown import BROWN_BIGRAM_ARTIFACT_SCHEMA, brown_source_metadata
from ailm.levels.letter import (
    ALPHA_LIST,
    ALPHA_TO_IDX,
    ALPHABET,
    IDX_TO_ALPHA,
    N_SYMBOLS,
    build_letter_agent,
    build_letter_transition,
    compute_perplexity,
    load_brown_counts,
    normalize_text,
    train_on_corpus,
)

# ── PATHS ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor"
BROWN_COUNTS_FILE = VENDOR_DIR / "brown_counts.json"

_SKIP_NO_COUNTS = pytest.mark.skipif(
    not BROWN_COUNTS_FILE.exists(),
    reason=(
        f"Brown corpus counts not at {BROWN_COUNTS_FILE}. "
        "Run scripts/generate_counts.py to generate it."
    ),
)


def _canonical_brown_payload(counts: dict[str, int]) -> dict:
    """Minimal source-authenticated fixture accepted by the production loader."""
    total = sum(counts.values())
    return {
        "schema": BROWN_BIGRAM_ARTIFACT_SCHEMA,
        "provenance": {
            "corpus": "NLTK Brown",
            "canonical": True,
            "raw_source": brown_source_metadata(),
            "normalization": "test fixture",
            "alphabet": ALPHA_LIST,
            "normalized_characters": total + 1,
            "distinct_bigrams": len(counts),
            "bigram_tokens": total,
        },
        "counts": counts,
    }


# ── FIXED TEST TEXT ───────────────────────────────────────────────────────────
# Public-domain nursery-rhyme / pangram prose. A fixed string means a fixed
# perplexity for a given A matrix.
ENGLISH_TEXT = (
    "the cat sat on the mat and the rat ran from the flat "
    "she sells sea shells by the sea shore "
    "the quick brown fox jumps over the lazy dog "
) * 3


# ── SHARED AGENTS ─────────────────────────────────────────────────────────────
# Module-scoped so JAX compiles each agent's inference once for the whole file.


@pytest.fixture(scope="module")
def uniform_agent():
    """Agent with a uniform bigram matrix — the canonical BROKEN model.

    Analytically every predictive probability is exactly 1/N_SYMBOLS regardless
    of the belief state, so perplexity is exactly N_SYMBOLS for any input.
    """
    return build_letter_agent(jnp.ones((1, N_SYMBOLS, N_SYMBOLS)) / N_SYMBOLS)


@pytest.fixture(scope="module")
def in_sample_agent():
    """Agent whose bigram matrix was fit on ENGLISH_TEXT itself.

    In-sample only — it proves the *plumbing* responds to training data. It is
    NOT evidence of generalization; that is measured out-of-sample in
    test_stage1_gate.py.
    """
    return build_letter_agent(build_letter_transition(train_on_corpus([ENGLISH_TEXT] * 5)))


# ── ALPHABET ──────────────────────────────────────────────────────────────────


class TestAlphabet:
    """ALPHABET is exactly 30 symbols with deterministic sorted indexing."""

    def test_alphabet_size_is_30(self):
        assert len(ALPHABET) == 30

    def test_alpha_list_size_is_30(self):
        assert len(ALPHA_LIST) == 30

    def test_n_symbols_matches_alphabet_size(self):
        assert N_SYMBOLS == len(ALPHABET) == 30

    def test_all_lowercase_letters_present(self):
        missing = [c for c in "abcdefghijklmnopqrstuvwxyz" if c not in ALPHABET]
        assert missing == [], f"Letters missing from ALPHABET: {missing}"

    def test_space_period_apostrophe_hyphen_present(self):
        for symbol in (" ", ".", "'", "-"):
            assert symbol in ALPHABET, f"{symbol!r} missing from ALPHABET"

    def test_alphabet_is_exactly_letters_plus_four_symbols(self):
        expected = set("abcdefghijklmnopqrstuvwxyz") | {" ", ".", "'", "-"}
        assert set(ALPHABET) == expected

    def test_no_uppercase_in_alphabet(self):
        assert not any(c.isupper() for c in ALPHABET)

    def test_no_digits_in_alphabet(self):
        assert not any(c.isdigit() for c in ALPHABET)

    def test_alpha_list_is_sorted(self):
        assert sorted(ALPHABET) == ALPHA_LIST

    def test_alpha_list_head_is_punctuation_then_letters(self):
        # Sorted ASCII order: ' ', "'", '-', '.', 'a', ...
        assert ALPHA_LIST[:5] == [" ", "'", "-", ".", "a"]
        assert ALPHA_LIST[-1] == "z"

    def test_alpha_to_idx_covers_alphabet(self):
        assert set(ALPHA_TO_IDX) == set(ALPHABET)

    def test_idx_to_alpha_covers_range(self):
        assert set(IDX_TO_ALPHA) == set(range(N_SYMBOLS))

    def test_indices_are_contiguous_zero_based(self):
        assert sorted(ALPHA_TO_IDX.values()) == list(range(N_SYMBOLS))

    def test_char_to_idx_to_char_roundtrip(self):
        for char, idx in ALPHA_TO_IDX.items():
            assert IDX_TO_ALPHA[idx] == char, f"char roundtrip broke at {char!r}"

    def test_idx_to_char_to_idx_roundtrip(self):
        for idx, char in IDX_TO_ALPHA.items():
            assert ALPHA_TO_IDX[char] == idx, f"index roundtrip broke at {idx}"

    def test_alpha_to_idx_agrees_with_alpha_list_position(self):
        for position, char in enumerate(ALPHA_LIST):
            assert ALPHA_TO_IDX[char] == position


# ── normalize_text ────────────────────────────────────────────────────────────


class TestNormalizeText:
    """normalize_text maps arbitrary text onto ALPHABET, char for char."""

    def test_lowercase_alphabet_text_unchanged(self):
        assert normalize_text("hello world") == "hello world"

    def test_uppercase_is_lowercased(self):
        assert normalize_text("HELLO") == "hello"

    def test_mixed_case_is_lowercased(self):
        assert normalize_text("Hello World") == "hello world"

    def test_digits_become_spaces(self):
        # Every non-ALPHABET char maps to exactly one space (no collapsing).
        assert normalize_text("abc123") == "abc   "

    def test_newline_and_tab_become_spaces(self):
        assert normalize_text("a\nb\tc") == "a b c"

    def test_digits_inside_words_become_spaces_not_removed(self):
        # Guards the documented no-collapse behavior: length is preserved.
        assert normalize_text("line1\nline2") == "line  line "

    def test_alphabet_punctuation_preserved(self):
        assert normalize_text("it's") == "it's"
        assert normalize_text("well-known") == "well-known"
        assert normalize_text("end.") == "end."

    def test_empty_string_returns_empty(self):
        assert normalize_text("") == ""

    def test_spaces_preserved(self):
        assert normalize_text("   ") == "   "

    def test_length_is_preserved(self):
        text = "Hello, World! This is test #42.\n"
        assert len(normalize_text(text)) == len(text)

    def test_output_contains_only_alphabet_chars(self):
        text = "Hello, World! ¿Qué? 42 — done.\t\n"
        offenders = [c for c in normalize_text(text) if c not in ALPHABET]
        assert offenders == [], f"Non-ALPHABET chars survived: {offenders}"

    def test_is_idempotent(self):
        once = normalize_text("Hello, World! 42")
        assert normalize_text(once) == once


# ── train_on_corpus ───────────────────────────────────────────────────────────


class TestTrainOnCorpus:
    """train_on_corpus counts consecutive (prev, curr) pairs after normalizing."""

    def test_empty_corpus_returns_empty_dict(self):
        assert train_on_corpus([]) == {}

    def test_single_char_sentence_has_no_bigrams(self):
        assert train_on_corpus(["a"]) == {}

    def test_empty_sentence_has_no_bigrams(self):
        assert train_on_corpus([""]) == {}

    def test_single_bigram_counted_once(self):
        assert train_on_corpus(["ab"]) == {("a", "b"): 1}

    def test_repeated_bigram_accumulates_within_sentence(self):
        counts = train_on_corpus(["abab"])
        assert counts[("a", "b")] == 2
        assert counts[("b", "a")] == 1

    def test_counts_accumulate_across_sentences(self):
        assert train_on_corpus(["ab", "ab", "ab"])[("a", "b")] == 3

    def test_exact_counts_for_known_string(self):
        # "test test" normalizes to itself; bigrams enumerated by hand.
        counts = train_on_corpus(["test test"])
        expected = {
            ("t", "e"): 2,
            ("e", "s"): 2,
            ("s", "t"): 2,
            ("t", " "): 1,
            (" ", "t"): 1,
        }
        assert counts == expected

    def test_bigram_total_equals_length_minus_one(self):
        sentence = "hello world"
        counts = train_on_corpus([sentence])
        assert sum(counts.values()) == len(sentence) - 1

    def test_keys_are_two_char_tuples(self):
        for key in train_on_corpus(["hello world"]):
            assert isinstance(key, tuple) and len(key) == 2
            assert all(isinstance(part, str) and part in ALPHABET for part in key)

    def test_values_are_positive_ints(self):
        for value in train_on_corpus(["the quick brown fox"]).values():
            assert isinstance(value, int) and not isinstance(value, bool)
            assert value > 0

    def test_returns_plain_dict_not_defaultdict(self):
        counts = train_on_corpus(["ab"])
        assert type(counts) is dict
        with pytest.raises(KeyError):
            counts[("z", "z")]  # a defaultdict would silently insert 0

    def test_uppercase_is_normalized_before_counting(self):
        assert train_on_corpus(["AB"]) == {("a", "b"): 1}

    def test_non_alphabet_chars_break_the_bigram(self):
        # "a1b" → "a b": ('a',' ') and (' ','b'), never ('a','b').
        assert train_on_corpus(["a1b"]) == {("a", " "): 1, (" ", "b"): 1}

    def test_apostrophe_and_hyphen_are_counted_as_symbols(self):
        counts = train_on_corpus(["it's a co-op"])
        assert counts[("t", "'")] == 1
        assert counts[("'", "s")] == 1
        assert counts[("o", "-")] == 1
        assert counts[("-", "o")] == 1


# ── build_letter_transition ───────────────────────────────────────────────────────────


class TestBuildLetterTransition:
    """The builder produces a column-stochastic (1, 30, 30) transition."""

    def test_shape_is_1_30_30(self):
        assert build_letter_transition({}).shape == (1, N_SYMBOLS, N_SYMBOLS)

    def test_shape_holds_for_trained_counts(self):
        counts = train_on_corpus([ENGLISH_TEXT])
        assert build_letter_transition(counts).shape == (1, N_SYMBOLS, N_SYMBOLS)

    def test_output_is_a_jax_array(self):
        a_matrix = build_letter_transition({})
        assert isinstance(a_matrix, jnp.ndarray)
        assert bool(jnp.isfinite(a_matrix).all())

    def test_columns_sum_to_one(self):
        counts = train_on_corpus([ENGLISH_TEXT])
        col_sums = np.asarray(build_letter_transition(counts)[0]).sum(axis=0)
        np.testing.assert_allclose(
            col_sums,
            np.ones(N_SYMBOLS),
            atol=1e-5,
            err_msg="A must be column-stochastic: P(obs | state=j) sums to 1",
        )

    def test_all_values_in_unit_interval(self):
        arr = np.asarray(build_letter_transition(train_on_corpus([ENGLISH_TEXT]))[0])
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_laplace_smoothing_leaves_no_zeros(self):
        arr = np.asarray(build_letter_transition({("a", "b"): 10_000})[0])
        assert arr.min() > 0.0, "add-1 smoothing must keep every entry positive"

    def test_empty_counts_give_exactly_uniform(self):
        # With add-1 smoothing and no data, every cell is exactly 1/30.
        arr = np.asarray(build_letter_transition({})[0])
        np.testing.assert_allclose(arr, 1.0 / N_SYMBOLS, atol=1e-6)

    def test_tuple_keys_place_mass_at_obs_row_state_column(self):
        arr = np.asarray(build_letter_transition({("a", "b"): 1000})[0])
        i, j = ALPHA_TO_IDX["b"], ALPHA_TO_IDX["a"]
        assert arr[i, j] > 0.9, "A[0, curr, prev] must carry the bigram mass"
        # Orientation check: the transposed cell must NOT be inflated.
        assert arr[j, i] == pytest.approx(1.0 / N_SYMBOLS, abs=1e-6)

    def test_string_keys_from_json_roundtrip(self):
        arr = np.asarray(build_letter_transition({"a,b": 1000, "b,a": 1000})[0])
        assert arr[ALPHA_TO_IDX["b"], ALPHA_TO_IDX["a"]] > 0.9
        assert arr[ALPHA_TO_IDX["a"], ALPHA_TO_IDX["b"]] > 0.9

    def test_string_and_tuple_keys_agree(self):
        tuple_a = np.asarray(build_letter_transition({("t", "h"): 77})[0])
        string_a = np.asarray(build_letter_transition({"t,h": 77})[0])
        np.testing.assert_allclose(tuple_a, string_a, atol=1e-7)

    def test_space_keys_parse_from_json_format(self):
        # Brown counts encode the space symbol literally: " ,t".
        arr = np.asarray(build_letter_transition({" ,t": 1000})[0])
        assert arr[ALPHA_TO_IDX["t"], ALPHA_TO_IDX[" "]] > 0.9

    def test_malformed_string_key_is_skipped(self):
        # "ab" has no comma → cannot split into a pair → ignored entirely.
        arr = np.asarray(build_letter_transition({"ab": 10_000})[0])
        np.testing.assert_allclose(arr, 1.0 / N_SYMBOLS, atol=1e-6)

    def test_out_of_alphabet_key_is_skipped(self):
        arr = np.asarray(build_letter_transition({"1,2": 10_000, "z,": 10_000})[0])
        np.testing.assert_allclose(arr, 1.0 / N_SYMBOLS, atol=1e-6)

    def test_frequent_bigram_outranks_rare_bigram(self):
        arr = np.asarray(build_letter_transition({("t", "h"): 5000, ("t", "z"): 1})[0])
        j = ALPHA_TO_IDX["t"]
        assert arr[ALPHA_TO_IDX["h"], j] > arr[ALPHA_TO_IDX["z"], j]

    def test_trained_matrix_is_not_uniform(self):
        arr = np.asarray(build_letter_transition(train_on_corpus([ENGLISH_TEXT] * 5))[0])
        assert not np.allclose(arr, 1.0 / N_SYMBOLS, atol=0.01), (
            "A trained A matrix that is still uniform means the counts were dropped"
        )

    def test_counts_shift_probability_monotonically(self):
        low = np.asarray(build_letter_transition({("q", "u"): 10})[0])
        high = np.asarray(build_letter_transition({("q", "u"): 10_000})[0])
        i, j = ALPHA_TO_IDX["u"], ALPHA_TO_IDX["q"]
        assert high[i, j] > low[i, j] > 1.0 / N_SYMBOLS


# ── build_letter_agent ────────────────────────────────────────────────────────


class TestBuildLetterAgent:
    """build_letter_agent returns a real pymdp Agent wired for letter inference."""

    def test_returns_real_pymdp_agent(self, uniform_agent):
        from pymdp.agent import Agent

        assert isinstance(uniform_agent, Agent)

    def test_agent_exposes_a_b_c_d(self, uniform_agent):
        for attr in ("A", "B", "C", "D"):
            assert hasattr(uniform_agent, attr), f"Agent is missing {attr}"

    def test_a_shape(self, uniform_agent):
        assert len(uniform_agent.A) == 1
        assert uniform_agent.A[0].shape == (1, N_SYMBOLS, N_SYMBOLS)

    def test_a_is_identity_emission(self, uniform_agent):
        """Observing a letter must reveal the letter state exactly."""
        np.testing.assert_allclose(np.asarray(uniform_agent.A[0])[0], np.eye(N_SYMBOLS), atol=1e-6)

    def test_b_shape(self, uniform_agent):
        assert uniform_agent.B[0].shape == (1, N_SYMBOLS, N_SYMBOLS, 1)

    def test_supplied_bigram_matrix_lands_in_b_not_a(self):
        """Orientation guard: the corpus statistics parameterize the TRANSITION.

        Putting them in A instead is anti-informative (posterior over the
        *preceding* letter), which is a silent quality regression rather than a
        crash — so it is pinned here.
        """
        bigram = build_letter_transition(train_on_corpus([ENGLISH_TEXT]))
        agent = build_letter_agent(bigram)

        np.testing.assert_allclose(
            np.asarray(agent.B[0])[0, :, :, 0],
            np.asarray(bigram)[0],
            atol=1e-6,
            err_msg="B must carry the supplied bigram matrix verbatim",
        )
        assert not np.allclose(np.asarray(agent.A[0])[0], np.asarray(bigram)[0], atol=1e-3), (
            "The bigram matrix must NOT be installed as the emission A"
        )

    def test_c_is_zero_preferences(self, uniform_agent):
        assert uniform_agent.C[0].shape == (1, N_SYMBOLS)
        np.testing.assert_allclose(np.asarray(uniform_agent.C[0]), 0.0, atol=1e-9)

    def test_d_is_uniform_prior(self, uniform_agent):
        assert uniform_agent.D[0].shape == (1, N_SYMBOLS)
        np.testing.assert_allclose(np.asarray(uniform_agent.D[0]), 1.0 / N_SYMBOLS, atol=1e-6)

    def test_accepts_numpy_input(self):
        agent = build_letter_agent(np.ones((1, N_SYMBOLS, N_SYMBOLS)) / N_SYMBOLS)
        assert agent.A[0].shape == (1, N_SYMBOLS, N_SYMBOLS)

    def test_accepts_nested_list_input(self):
        # Exercises the shape-less branch: plain lists are converted to JAX.
        nested = [[[1.0 / N_SYMBOLS] * N_SYMBOLS for _ in range(N_SYMBOLS)]]
        agent = build_letter_agent(nested)
        assert agent.A[0].shape == (1, N_SYMBOLS, N_SYMBOLS)

    def test_missing_batch_dimension_raises_assertion(self):
        with pytest.raises(ValueError, match=r"\(1, 30, 30\)"):
            build_letter_agent(jnp.ones((N_SYMBOLS, N_SYMBOLS)) / N_SYMBOLS)

    def test_wrong_symbol_count_raises_assertion(self):
        with pytest.raises(ValueError, match=r"\(1, 30, 30\)"):
            build_letter_agent(jnp.ones((1, 5, 5)) / 5)

    def test_infer_states_runs_and_returns_a_distribution(self, uniform_agent):
        prior = [jnp.ones((1, N_SYMBOLS)) / N_SYMBOLS]
        qs = uniform_agent.infer_states([jnp.array([ALPHA_TO_IDX["a"]])], prior)
        assert qs is not None and len(qs) == 1
        assert qs[0].shape[-1] == N_SYMBOLS
        posterior = np.asarray(qs[0]).reshape(-1)
        assert posterior.min() >= 0.0
        assert posterior.sum() == pytest.approx(1.0, abs=1e-4)

    def test_infer_states_collapses_onto_the_observed_letter(self, in_sample_agent):
        """Identity emission ⇒ the posterior is a delta on the observed letter."""
        prior = [jnp.ones((1, N_SYMBOLS)) / N_SYMBOLS]
        idx = ALPHA_TO_IDX["q"]
        qs = in_sample_agent.infer_states([jnp.array([idx])], prior)
        posterior = np.asarray(qs[0]).reshape(-1)
        assert posterior[idx] == pytest.approx(1.0, abs=1e-4), (
            f"Posterior mass on the observed letter is {posterior[idx]:.6f}, not 1.0"
        )
        assert posterior.sum() - posterior[idx] == pytest.approx(0.0, abs=1e-4)

    def test_infer_policies_runs(self, uniform_agent):
        prior = [jnp.ones((1, N_SYMBOLS)) / N_SYMBOLS]
        qs = uniform_agent.infer_states([jnp.array([ALPHA_TO_IDX["e"]])], prior)
        q_pi, neg_efe = uniform_agent.infer_policies(qs)
        assert q_pi is not None and neg_efe is not None
        # A single null action ⇒ the policy posterior is degenerate at 1.0.
        np.testing.assert_allclose(np.asarray(q_pi).reshape(-1), 1.0, atol=1e-5)

    def test_update_empirical_prior_runs(self, uniform_agent):
        prior = [jnp.ones((1, N_SYMBOLS)) / N_SYMBOLS]
        qs = uniform_agent.infer_states([jnp.array([ALPHA_TO_IDX["e"]])], prior)
        action = jnp.zeros((1, 1), dtype=jnp.int32)
        new_prior = uniform_agent.update_empirical_prior(action, qs)
        assert new_prior is not None and len(new_prior) == 1
        assert new_prior[0].shape == (1, N_SYMBOLS)
        assert np.asarray(new_prior[0]).reshape(-1).sum() == pytest.approx(1.0, abs=1e-4)

    def test_empirical_prior_feeds_back_into_infer_states(self, uniform_agent):
        """The full per-letter recursion must chain without shape errors."""
        prior = [jnp.ones((1, N_SYMBOLS)) / N_SYMBOLS]
        action = jnp.zeros((1, 1), dtype=jnp.int32)
        for char in "abc":
            qs = uniform_agent.infer_states([jnp.array([ALPHA_TO_IDX[char]])], prior)
            prior = uniform_agent.update_empirical_prior(action, qs)
        assert np.asarray(prior[0]).reshape(-1).sum() == pytest.approx(1.0, abs=1e-4)

    def test_next_step_prior_is_the_bigram_column_of_the_observed_letter(self):
        """The whole model in one assertion: after seeing 'q', predict P(next | q).

        update_empirical_prior must push the delta posterior through B, so the
        next-step prior equals column 'q' of the supplied bigram matrix.
        """
        bigram = build_letter_transition(train_on_corpus([ENGLISH_TEXT] * 3))
        agent = build_letter_agent(bigram)
        idx = ALPHA_TO_IDX["q"]

        prior = [jnp.ones((1, N_SYMBOLS)) / N_SYMBOLS]
        qs = agent.infer_states([jnp.array([idx])], prior)
        action = jnp.zeros((1, 1), dtype=jnp.int32)
        next_prior = np.asarray(agent.update_empirical_prior(action, qs)[0]).reshape(-1)

        expected = np.asarray(bigram)[0][:, idx]
        np.testing.assert_allclose(
            next_prior,
            expected,
            atol=1e-5,
            err_msg="Next-step prior must equal B[:, observed, 0] (the bigram column)",
        )
        # And in this corpus 'u' follows 'q' every time, so the prediction is sharp.
        assert next_prior.argmax() == ALPHA_TO_IDX["u"]


# ── compute_perplexity ────────────────────────────────────────────────────────


class TestComputePerplexity:
    """compute_perplexity returns exp(per-letter cross-entropy in nats)."""

    def test_returns_a_python_float(self, uniform_agent):
        assert isinstance(compute_perplexity(uniform_agent, "hello world"), float)

    def test_short_text_returns_a_float(self, in_sample_agent):
        ppl = compute_perplexity(in_sample_agent, "ab")
        assert isinstance(ppl, float) and math.isfinite(ppl) and ppl > 0.0

    def test_perplexity_is_positive_and_finite(self, in_sample_agent):
        ppl = compute_perplexity(in_sample_agent, ENGLISH_TEXT)
        assert ppl > 0.0
        assert math.isfinite(ppl)

    def test_empty_text_returns_n_symbols(self, uniform_agent):
        assert compute_perplexity(uniform_agent, "") == float(N_SYMBOLS)

    def test_single_character_returns_n_symbols(self, uniform_agent):
        assert compute_perplexity(uniform_agent, "a") == float(N_SYMBOLS)

    def test_single_char_after_normalization_returns_n_symbols(self, uniform_agent):
        # "\t" normalizes to one space — still a single symbol, so short-circuit.
        assert compute_perplexity(uniform_agent, "\t") == float(N_SYMBOLS)

    def test_uniform_agent_perplexity_is_exactly_n_symbols(self, uniform_agent):
        """Closed form: uniform A ⇒ every prediction is 1/30 ⇒ perplexity = 30."""
        ppl = compute_perplexity(uniform_agent, ENGLISH_TEXT)
        assert ppl == pytest.approx(float(N_SYMBOLS), abs=1e-3), (
            f"Uniform A must give perplexity exactly {N_SYMBOLS}, got {ppl}"
        )

    def test_uniform_agent_perplexity_is_text_independent(self, uniform_agent):
        """A uniform model cannot prefer English over noise — that is the point."""
        english = compute_perplexity(uniform_agent, ENGLISH_TEXT)
        noise = compute_perplexity(uniform_agent, "zqxjk-vbn'wf.hpy" * 8)
        assert english == pytest.approx(noise, abs=1e-3)

    def test_matches_closed_form_bigram_chain(self):
        """Independent oracle: re-derive the perplexity in numpy, outside pymdp.

        With an identity emission the predictive distribution for letter t+1 is
        exactly column c_t of the bigram matrix, so the whole metric has a
        closed form. Binding to it catches inference wiring that merely *runs*.
        """
        bigram = build_letter_transition(train_on_corpus([ENGLISH_TEXT] * 3))
        agent = build_letter_agent(bigram)
        matrix = np.asarray(bigram)[0]
        text = "the quick brown fox jumps over the lazy dog"

        log_probs = []
        predictive = np.ones(N_SYMBOLS) / N_SYMBOLS  # uniform before the first letter
        for char in normalize_text(text):
            idx = ALPHA_TO_IDX[char]
            log_probs.append(math.log(max(float(predictive[idx]), 1e-10)))
            predictive = matrix[:, idx]
        expected = math.exp(-sum(log_probs) / len(log_probs))

        measured = compute_perplexity(agent, text)
        assert measured == pytest.approx(expected, rel=1e-4), (
            f"pymdp inference gives {measured:.6f} but the closed-form bigram "
            f"chain gives {expected:.6f} — the generative model is mis-wired."
        )

    def test_negative_control_uniform_agent_cannot_pass_stage1_gate(self, uniform_agent):
        """A broken (uniform) model must score WORSE than the Stage 1 threshold.

        If this ever fails, the gate is vacuous: it would admit a model that has
        learned nothing.
        """
        ppl = compute_perplexity(uniform_agent, ENGLISH_TEXT)
        assert ppl >= 28.0, (
            f"GATE VACUOUS: uniform agent scored {ppl:.4f} < 28.0, so the Stage 1 "
            "threshold would accept a model with zero learned structure."
        )

    def test_in_sample_trained_agent_beats_uniform(self, uniform_agent, in_sample_agent):
        """Plumbing control: counts fitted on the eval text must lower perplexity.

        This is an IN-SAMPLE comparison and proves only that training data reaches
        the metric. Generalization is gated out-of-sample in test_stage1_gate.py.
        """
        trained = compute_perplexity(in_sample_agent, ENGLISH_TEXT)
        uniform = compute_perplexity(uniform_agent, ENGLISH_TEXT)
        assert trained < uniform, (
            f"In-sample perplexity {trained:.4f} is not below uniform {uniform:.4f} — "
            "the bigram matrix is not reaching compute_perplexity at all."
        )

    def test_more_training_data_does_not_hurt_in_sample_perplexity(self):
        """Monotonicity control across two fit strengths on the same eval text."""
        weak = build_letter_agent(build_letter_transition(train_on_corpus([ENGLISH_TEXT])))
        strong = build_letter_agent(build_letter_transition(train_on_corpus([ENGLISH_TEXT] * 25)))
        weak_ppl = compute_perplexity(weak, ENGLISH_TEXT)
        strong_ppl = compute_perplexity(strong, ENGLISH_TEXT)
        assert strong_ppl <= weak_ppl + 1e-6, (
            f"Sharper counts raised in-sample perplexity: {strong_ppl:.4f} > {weak_ppl:.4f}"
        )

    def test_uppercase_input_is_normalized_not_rejected(self, in_sample_agent):
        upper = compute_perplexity(in_sample_agent, ENGLISH_TEXT.upper())
        lower = compute_perplexity(in_sample_agent, ENGLISH_TEXT)
        assert upper == pytest.approx(lower, abs=1e-3)

    def test_non_alphabet_input_is_normalized(self, in_sample_agent):
        ppl = compute_perplexity(in_sample_agent, "abc123def456")
        assert math.isfinite(ppl) and ppl > 0.0

    def test_all_space_text_is_evaluated_not_short_circuited(self, uniform_agent):
        """A ≥2-char all-space string goes through inference, not the early return.

        Under uniform A the answer is still exactly 30, but reached via the loop.
        """
        ppl = compute_perplexity(uniform_agent, "     ")
        assert ppl == pytest.approx(float(N_SYMBOLS), abs=1e-3)

    def test_is_deterministic_across_repeated_calls(self, in_sample_agent):
        text = "the quick brown fox jumps over the lazy dog"
        assert compute_perplexity(in_sample_agent, text) == compute_perplexity(
            in_sample_agent, text
        )

    def test_is_deterministic_across_rebuilt_agents(self):
        counts = train_on_corpus([ENGLISH_TEXT] * 3)
        text = "the quick brown fox"
        first = compute_perplexity(build_letter_agent(build_letter_transition(counts)), text)
        second = compute_perplexity(build_letter_agent(build_letter_transition(counts)), text)
        assert first == pytest.approx(second, abs=1e-6)

    def test_english_like_fixed_string_scores_finite(self, in_sample_agent):
        fixed = (
            "the quick brown fox jumps over the lazy dog and then the dog "
            "barked at the fox until it ran away."
        )
        assert len(fixed) == 97  # pinned so the fixture cannot drift silently
        ppl = compute_perplexity(in_sample_agent, fixed)
        assert math.isfinite(ppl) and ppl > 0.0


class TestEmpiricalPriorShapeIsAlwaysBatchLeading:
    """Pin the pymdp shape contract that ``compute_perplexity`` relies on.

    ``compute_perplexity`` flattens the empirical prior with an unconditional
    ``.ravel()``. It previously branched on ``prior_arr.ndim == 1``; that arm was
    unreachable, because the initial prior is constructed ``(1, N_SYMBOLS)`` and
    ``Agent.update_empirical_prior`` returns the same batch-leading shape on every
    step. The branch was removed.

    These tests are what makes that removal safe rather than lucky: if a future
    pymdp returned a flat ``(N_SYMBOLS,)`` prior, ``ravel()`` would still be
    correct, but this class documents and detects the change instead of leaving it
    to be discovered as a silent numerical shift.
    """

    def test_update_empirical_prior_stays_two_dimensional(self, uniform_agent):
        prior: list = [jnp.ones((1, N_SYMBOLS)) / N_SYMBOLS]
        assert prior[0].shape == (1, N_SYMBOLS)

        action = jnp.zeros((1, 1), dtype=jnp.int32)
        for step in range(4):
            qs = uniform_agent.infer_states([jnp.array([ALPHA_TO_IDX["e"]])], prior)
            prior = uniform_agent.update_empirical_prior(action, qs)
            assert len(prior) == 1, f"step {step}: expected one state factor"
            assert prior[0].ndim == 2, (
                f"step {step}: pymdp returned a {prior[0].ndim}-D empirical prior. "
                "compute_perplexity assumes batch-leading (1, N_SYMBOLS)."
            )
            assert prior[0].shape == (1, N_SYMBOLS), f"step {step}: {prior[0].shape}"

    def test_ravel_is_the_identity_on_an_already_flat_prior(self):
        """Why no ndim branch is needed: ravel() handles the 1-D case unchanged."""
        flat = jnp.ones((N_SYMBOLS,)) / N_SYMBOLS
        assert flat.ravel().shape == (N_SYMBOLS,)
        assert np.allclose(np.asarray(flat.ravel()), np.asarray(flat))

    def test_ravel_flattens_the_batch_leading_prior_to_the_same_vector(self):
        batched = jnp.ones((1, N_SYMBOLS)) / N_SYMBOLS
        assert batched.ravel().shape == (N_SYMBOLS,)
        assert np.allclose(
            np.asarray(batched.ravel()), np.asarray(jnp.ones((N_SYMBOLS,)) / N_SYMBOLS)
        )


# ── load_brown_counts ─────────────────────────────────────────────────────────


class TestLoadBrownCounts:
    """load_brown_counts reads vendor/brown_counts.json — no network, no NLTK."""

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="brown_counts.json"):
            load_brown_counts(tmp_path)

    def test_missing_directory_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_brown_counts(tmp_path / "does" / "not" / "exist")

    def test_explicit_vendor_dir_roundtrips_json(self, tmp_path):
        (tmp_path / "brown_counts.json").write_text(
            json.dumps(_canonical_brown_payload({"a,b": 5, "b,c": 3})),
            encoding="utf-8",
        )
        assert load_brown_counts(tmp_path) == {"a,b": 5, "b,c": 3}

    def test_accepts_string_path(self, tmp_path):
        (tmp_path / "brown_counts.json").write_text(
            json.dumps(_canonical_brown_payload({"t,h": 9})), encoding="utf-8"
        )
        assert load_brown_counts(str(tmp_path)) == {"t,h": 9}

    def test_loaded_counts_feed_build_letter_transition(self, tmp_path):
        (tmp_path / "brown_counts.json").write_text(
            json.dumps(_canonical_brown_payload({"t,h": 1000, " ,t": 500})),
            encoding="utf-8",
        )
        arr = np.asarray(build_letter_transition(load_brown_counts(tmp_path))[0])
        assert arr[ALPHA_TO_IDX["h"], ALPHA_TO_IDX["t"]] > 0.9
        np.testing.assert_allclose(arr.sum(axis=0), np.ones(N_SYMBOLS), atol=1e-5)

    def test_legacy_flat_mapping_is_refused(self, tmp_path):
        """NEGATIVE CONTROL: counts without source identity are not canonical."""
        (tmp_path / "brown_counts.json").write_text(json.dumps({"a,b": 5}), encoding="utf-8")
        with pytest.raises(ValueError, match="schema"):
            load_brown_counts(tmp_path)

    def test_wrong_raw_source_is_refused(self, tmp_path):
        """NEGATIVE CONTROL: valid counts cannot launder a moving source URL."""
        payload = _canonical_brown_payload({"a,b": 5})
        payload["provenance"]["raw_source"]["source_url"] = (
            "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/brown.zip"
        )
        (tmp_path / "brown_counts.json").write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="disagrees with the pin"):
            load_brown_counts(tmp_path)

    def test_interior_count_corruption_is_refused(self, tmp_path):
        """NEGATIVE CONTROL: source metadata does not excuse invalid counts."""
        payload = _canonical_brown_payload({"a,b": 5})
        payload["counts"]["a,b"] = 0
        (tmp_path / "brown_counts.json").write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="invalid count"):
            load_brown_counts(tmp_path)

    @_SKIP_NO_COUNTS
    def test_real_vendor_file_loads(self):
        counts = load_brown_counts(VENDOR_DIR)
        assert isinstance(counts, dict) and len(counts) > 0
        for key, value in counts.items():
            assert isinstance(key, str)
            assert isinstance(value, int) and value > 0

    @_SKIP_NO_COUNTS
    def test_real_vendor_keys_are_parseable_bigrams(self):
        """Every vendor key must parse into an ALPHABET pair.

        build_letter_transition silently skips keys it cannot parse, so an unparseable
        vendor file would train a uniform model while looking healthy.
        """
        counts = load_brown_counts(VENDOR_DIR)
        parseable = 0
        for key in counts:
            parts = key.split(",", 1)
            if len(parts) == 2 and parts[0] in ALPHABET and parts[1] in ALPHABET:
                parseable += 1
        assert parseable > 0, "No vendor key parsed into an ALPHABET bigram"
        assert parseable == len(counts), (
            f"Only {parseable}/{len(counts)} vendor keys parse as ALPHABET bigrams — "
            "build_letter_transition would silently drop the rest"
        )

    @_SKIP_NO_COUNTS
    def test_default_vendor_dir_resolves_to_project_vendor(self):
        """vendor_dir=None must resolve to <project_root>/vendor, not the cwd."""
        assert load_brown_counts() == load_brown_counts(VENDOR_DIR)
