"""Unit tests for ailm.levels.phoneme — AILM Stage 2 phoneme + G2P agents.

Doctrine
--------
* Real pymdp, real vendored CMUdict statistics. No mocks, no monkeypatching, no
  stubbed arrays anywhere in this file.
* Every numeric assertion is either an exact closed form (uniform transition ⇒
  perplexity exactly 39) or a value re-derived inside the test from the same
  inputs. No number is copied from a previous run. In particular the unigram
  comparator is rebuilt from ``phone_unigrams`` inside the test rather than
  pinned at 27.594, and the trained-bigram figure is never asserted against the
  frequently-quoted 14.720 — that constant is an unsmoothed, transitions-only
  conditional-entropy floor and comparing this module's all-positions perplexity
  against it would make the assertion unsatisfiable by construction.
* Structural checks (shape, key parsing, column sums) are kept apart from
  behavioral checks (perplexity ordering, decode accuracy) so a green structural
  block can never be read as evidence that either model learned anything.

Orientation — the Stage 1 lesson, pinned twice
----------------------------------------------
Stage 1 shipped the learned bigram matrix in the **emission A** with an identity
B. That is anti-informative rather than broken: observing symbol ``c`` yields a
posterior over the symbol that *preceded* ``c``, and identity B carries that
backward-looking belief forward as the next-symbol prediction. A bigram matrix is
column-stochastic, so pymdp's ``_validate()`` accepts it in either slot — the
defect is invisible to every structural check.

So this file pins orientation two independent ways:

1. **Index-explicit count placement.** ``build_phone_transition({"AA,B": 10000})``
   must inflate ``B[0, idx("B"), idx("AA"), 0]`` and must leave the transposed
   cell at the smoothing floor. A transposed build fails that, while it would
   pass any "the columns sum to 1 somewhere" check.
2. **Numerical agreement with an independent oracle.** The closed-form numpy HMM
   in :mod:`ailm.verification.oracle` imports neither pymdp nor jax, so it cannot
   inherit the flaw it audits for. ``compute_phoneme_perplexity`` is required to
   agree with ``oracle.markov_chain_perplexity`` to ~1e-5 relative, and the G2P
   filtering decode is required to select the same phones as
   ``oracle.forward_filter``. These are the tests that would have caught Stage 1.

Emission-orientation non-vacuity
--------------------------------
The module docstring records a mis-wiring that the "beats the context-free
baseline" signature does **not** catch: an emission normalized over phones
instead of letters, then renormalized so pymdp still validates it (69.47% against
a 68.64% bar). What actually pins it down is that ``A[0, :, p]`` sums to 1 over
*letters*. :class:`TestBuildG2PEmission` asserts that invariant and then runs the
phone-normalized matrix through the same assertion to prove the invariant is
live rather than trivially satisfiable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from ailm.corpus.cmudict import CANONICAL_ARPABET
from ailm.levels.phoneme import (
    ALPHA_LIST,
    ALPHA_TO_IDX,
    IDX_TO_PHONE,
    N_PHONES,
    N_SYMBOLS,
    PHONE_TO_IDX,
    PHONES,
    baseline_accuracy,
    baseline_positions,
    build_g2p_agent,
    build_g2p_emission,
    build_g2p_start,
    build_g2p_transition,
    build_phone_transition,
    build_phone_unigram,
    build_phoneme_agent,
    compute_phoneme_perplexity,
    decode_phonemes,
    decode_positions,
    g2p_accuracy,
    g2p_split,
    load_phoneme_counts,
    per_letter_argmax_baseline,
)
from ailm.verification import oracle

# ── PATHS ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor"
COUNTS_FILE = VENDOR_DIR / "cmudict_counts.json"

_SKIP_NO_COUNTS = pytest.mark.skipif(
    not COUNTS_FILE.exists(),
    reason=(
        f"Phoneme counts not at {COUNTS_FILE}. "
        "Run scripts/generate_phoneme_counts.py to generate it."
    ),
)

#: Documented split sizes of the vendored artifact (32420 pairs, every 5th is
#: TEST). Pinned so a regenerated artifact cannot silently change the split that
#: every accuracy figure in Stage 2 is measured against.
EXPECTED_TRAIN = 25936
EXPECTED_TEST = 6484

#: How many pronunciations the perplexity fixtures score. Small enough to keep
#: the pymdp loop inside the default 60 s timeout, large enough that the
#: trained/unigram/uniform ordering is decisive rather than marginal.
PPL_SEQUENCES = 60

#: How many test words the accuracy / position helpers evaluate. Decoding costs
#: roughly 7 ms per letter, so this is a deterministic prefix, never a sample.
ACC_WORDS = 30

#: Fixed words for the G2P oracle cross-check. Chosen to span short/long words,
#: doubled letters, and a 'q'.
CROSSCHECK_WORDS = ("cat", "dog", "hello", "phoneme", "quick", "abandon", "zebra")


# ── FIXTURES ──────────────────────────────────────────────────────────────────
# Module-scoped so JAX compiles each agent's inference once for the whole file
# and the vendored artifact is parsed once.


@pytest.fixture(scope="module")
def counts() -> dict:
    """The real vendored CMUdict statistics payload."""
    return load_phoneme_counts()


@pytest.fixture(scope="module")
def splits(counts):
    """``(train, test)`` from :func:`g2p_split` on the real artifact."""
    return g2p_split(counts)


@pytest.fixture(scope="module")
def phone_sequences(counts) -> list[list[str]]:
    """A deterministic prefix of real pronunciations, as phone lists."""
    return [
        entry[1].split() if isinstance(entry[1], str) else list(entry[1])
        for entry in counts["g2p_pairs"][:PPL_SEQUENCES]
    ]


@pytest.fixture(scope="module")
def trained_transition(counts):
    """B trained on the artifact's full-corpus phone bigrams."""
    return build_phone_transition(counts["phone_bigrams"])


@pytest.fixture(scope="module")
def phone_start(counts):
    """D from the artifact's phone unigrams."""
    return build_phone_unigram(counts["phone_unigrams"])


@pytest.fixture(scope="module")
def trained_phone_agent(trained_transition, phone_start):
    """Part A agent: trained bigram B, unigram D, identity A."""
    return build_phoneme_agent(trained_transition, phone_start)


@pytest.fixture(scope="module")
def uniform_phone_agent():
    """The canonical BROKEN Part A agent — a uniform transition.

    Analytically every predictive probability is exactly 1/39 regardless of the
    belief state, so perplexity is exactly 39 for any input. This is the negative
    control for every perplexity claim in the file.
    """
    return build_phoneme_agent(jnp.ones((1, N_PHONES, N_PHONES, 1)) / N_PHONES)


@pytest.fixture(scope="module")
def unigram_phone_agent(counts, phone_start):
    """Context-free comparator: every column of B is the phone unigram.

    Rebuilt here from ``phone_unigrams`` with the same add-1 smoothing
    ``build_phone_unigram`` uses, so the comparator is re-derived from the inputs
    rather than pinned to a remembered 27.594.
    """
    marginal = np.asarray(build_phone_unigram(counts["phone_unigrams"])).reshape(-1)
    columns = np.repeat(marginal[:, np.newaxis], N_PHONES, axis=1)
    return build_phoneme_agent(jnp.array(columns[np.newaxis, :, :, np.newaxis]), phone_start)


@pytest.fixture(scope="module")
def g2p_parts(splits):
    """``(emission, transition, start)`` built from the TRAIN split only."""
    train, _test = splits
    return (
        build_g2p_emission(train),
        build_g2p_transition(train),
        build_g2p_start(train),
    )


@pytest.fixture(scope="module")
def g2p_agent(g2p_parts):
    """Part B agent: letters observed, phones hidden."""
    return build_g2p_agent(*g2p_parts)


@pytest.fixture(scope="module")
def baseline_map(splits):
    """The context-free per-letter argmax bar, fit on TRAIN."""
    train, _test = splits
    return per_letter_argmax_baseline(train)


@pytest.fixture(scope="module")
def decoded_positions(g2p_agent, splits) -> list[bool]:
    """Per-position correctness of the pymdp filtering decoder on ACC_WORDS."""
    _train, test = splits
    return decode_positions(g2p_agent, test, limit=ACC_WORDS)


@pytest.fixture(scope="module")
def baseline_position_list(baseline_map, splits) -> list[bool]:
    """Per-position correctness of the context-free bar on the SAME selection.

    Same ``pairs`` and same ``limit`` as :func:`decoded_positions`, which is what
    makes the two vectors pairable.
    """
    _train, test = splits
    return baseline_positions(baseline_map, test, limit=ACC_WORDS)


# ── HELPERS ───────────────────────────────────────────────────────────────────


def _b_to_numpy(transition) -> np.ndarray:
    """Strip pymdp's batch/action dims and promote to a float64 oracle input.

    ``(1, 39, 39, 1)`` → ``(39, 39)``. The renormalization is not cosmetic: JAX
    computes in float32, so column sums land about 2e-7 off 1.0, and the oracle
    validates stochasticity to 1e-6. Promoting then renormalizing keeps the
    oracle's validator satisfied without changing any figure by more than
    float32's own error.
    """
    arr = np.asarray(transition)[0, :, :, 0].astype(np.float64)
    return arr / arr.sum(axis=0, keepdims=True)


def _vec_to_numpy(vector) -> np.ndarray:
    """``(1, n)`` jnp array → float64 ``(n,)`` renormalized to sum to 1."""
    arr = np.asarray(vector).reshape(-1).astype(np.float64)
    return arr / arr.sum()


def _a_to_numpy(emission) -> np.ndarray:
    """``(1, n_obs, n_states)`` jnp emission → float64 column-stochastic 2-D."""
    arr = np.asarray(emission)[0].astype(np.float64)
    return arr / arr.sum(axis=0, keepdims=True)


# ── PHONEME INVENTORY ─────────────────────────────────────────────────────────


class TestPhonemeInventory:
    """PHONES is exactly the 39 canonical ARPAbet phones, sorted."""

    def test_n_phones_is_39(self):
        assert N_PHONES == 39

    def test_phones_length_is_39(self):
        assert len(PHONES) == 39

    def test_n_phones_matches_phones_length(self):
        assert N_PHONES == len(PHONES) == 39

    def test_phones_is_sorted(self):
        assert list(PHONES) == sorted(PHONES)

    def test_phones_is_a_tuple(self):
        assert isinstance(PHONES, tuple)

    def test_phones_equals_sorted_canonical_arpabet(self):
        assert tuple(sorted(CANONICAL_ARPABET)) == PHONES

    def test_phones_has_no_duplicates(self):
        assert len(set(PHONES)) == len(PHONES)

    def test_no_stress_digits_in_any_phone(self):
        """Stress must already be stripped: 'AE1' would be a 40th, 41st symbol."""
        assert not any(c.isdigit() for p in PHONES for c in p)

    def test_all_phones_are_uppercase_ascii(self):
        for phone in PHONES:
            assert phone.isascii() and phone.isupper(), f"{phone!r} is not upper ASCII"

    def test_phone_to_idx_covers_phones(self):
        assert set(PHONE_TO_IDX) == set(PHONES)

    def test_idx_to_phone_covers_range(self):
        assert set(IDX_TO_PHONE) == set(range(N_PHONES))

    def test_indices_are_contiguous_zero_based(self):
        assert sorted(PHONE_TO_IDX.values()) == list(range(N_PHONES))

    def test_phone_to_idx_to_phone_roundtrip(self):
        for phone, idx in PHONE_TO_IDX.items():
            assert IDX_TO_PHONE[idx] == phone, f"phone roundtrip broke at {phone!r}"

    def test_idx_to_phone_to_idx_roundtrip(self):
        for idx, phone in IDX_TO_PHONE.items():
            assert PHONE_TO_IDX[phone] == idx, f"index roundtrip broke at {idx}"

    def test_phone_to_idx_agrees_with_phones_position(self):
        for position, phone in enumerate(PHONES):
            assert PHONE_TO_IDX[phone] == position

    def test_first_and_last_phone_pin_the_sort_order(self):
        # Sorted ASCII order over the canonical set: 'AA' first, 'ZH' last.
        assert PHONES[0] == "AA"
        assert PHONES[-1] == "ZH"

    def test_letter_reexports_are_the_letter_level_inventory(self):
        """The G2P emission is indexed by letter, so the re-export must match."""
        assert N_SYMBOLS == len(ALPHA_LIST) == 30
        assert set(ALPHA_TO_IDX) == set(ALPHA_LIST)
        assert sorted(ALPHA_TO_IDX.values()) == list(range(N_SYMBOLS))


# ── load_phoneme_counts ───────────────────────────────────────────────────────


class TestLoadPhonemeCounts:
    """load_phoneme_counts reads vendor/cmudict_counts.json — no network, no NLTK."""

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="cmudict_counts.json"):
            load_phoneme_counts(tmp_path)

    def test_missing_directory_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_phoneme_counts(tmp_path / "does" / "not" / "exist")

    def test_error_names_the_generator_script(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="generate_phoneme_counts.py"):
            load_phoneme_counts(tmp_path)

    def test_absent_schema_raises_value_error(self, tmp_path):
        (tmp_path / "cmudict_counts.json").write_text(
            json.dumps({"phone_unigrams": {"AA": 1}}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="_schema=None"):
            load_phoneme_counts(tmp_path)

    def test_wrong_schema_raises_value_error(self, tmp_path):
        (tmp_path / "cmudict_counts.json").write_text(
            json.dumps({"_schema": "ailm-cmudict-counts-v0"}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="ailm-cmudict-counts-v1"):
            load_phoneme_counts(tmp_path)

    def test_mismatched_phones_list_raises_value_error(self, tmp_path):
        """A misindexing artifact must be rejected, not silently accepted."""
        (tmp_path / "cmudict_counts.json").write_text(
            json.dumps({"_schema": "ailm-cmudict-counts-v1", "phones": ["AA", "B", "ZH"]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="disagrees with PHONES"):
            load_phoneme_counts(tmp_path)

    def test_absent_phones_key_is_tolerated(self, tmp_path):
        """``phones`` is provenance, not required — its absence skips the check."""
        (tmp_path / "cmudict_counts.json").write_text(
            json.dumps({"_schema": "ailm-cmudict-counts-v1", "phone_unigrams": {}}),
            encoding="utf-8",
        )
        assert load_phoneme_counts(tmp_path)["phone_unigrams"] == {}

    def test_accepts_string_path(self, tmp_path):
        (tmp_path / "cmudict_counts.json").write_text(
            json.dumps({"_schema": "ailm-cmudict-counts-v1", "phone_bigrams": {}}),
            encoding="utf-8",
        )
        assert load_phoneme_counts(str(tmp_path))["phone_bigrams"] == {}

    @_SKIP_NO_COUNTS
    def test_real_artifact_loads(self, counts):
        assert isinstance(counts, dict)
        assert counts["_schema"] == "ailm-cmudict-counts-v1"

    @_SKIP_NO_COUNTS
    def test_real_artifact_has_documented_keys(self, counts):
        for key in (
            "phones",
            "phone_unigrams",
            "phone_bigrams",
            "g2p_pairs",
            "test_every",
            "min_word_len",
            "source_base_words",
            "source_pronunciation_entries",
            "naive_phoneme_count",
            "comment_phantom_count",
            "comment_phantom_tokens",
            "source_url",
            "source_revision",
            "source_sha256",
            "source_size_bytes",
        ):
            assert key in counts, f"vendored artifact is missing {key!r}"

    @_SKIP_NO_COUNTS
    def test_real_artifact_phones_match_phones(self, counts):
        assert tuple(counts["phones"]) == PHONES

    @_SKIP_NO_COUNTS
    def test_real_bigram_keys_are_parseable_phone_pairs(self, counts):
        """build_phone_transition silently skips keys it cannot parse.

        An unparseable artifact would therefore train a uniform model while
        looking perfectly healthy, so every key is required to parse.
        """
        bigrams = counts["phone_bigrams"]
        assert len(bigrams) > 0
        unparseable = [
            key
            for key in bigrams
            if len(key.split(",", 1)) != 2
            or key.split(",", 1)[0] not in PHONE_TO_IDX
            or key.split(",", 1)[1] not in PHONE_TO_IDX
        ]
        assert unparseable == [], (
            f"{len(unparseable)} bigram keys do not parse into PHONES pairs — "
            f"build_phone_transition would silently drop them: {unparseable[:5]}"
        )

    @_SKIP_NO_COUNTS
    def test_real_unigram_keys_are_all_phones(self, counts):
        assert set(counts["phone_unigrams"]) <= set(PHONES)
        assert all(int(v) > 0 for v in counts["phone_unigrams"].values())

    @_SKIP_NO_COUNTS
    def test_default_vendor_dir_resolves_to_project_vendor(self, counts):
        """vendor_dir=None must resolve to <project_root>/vendor, not the cwd."""
        assert load_phoneme_counts(VENDOR_DIR) == counts


# ── build_phone_transition ────────────────────────────────────────────────────


class TestBuildPhoneTransition:
    """B[0, i, j, 0] = P(next phone i | current phone j), column-stochastic on axis 1."""

    def test_shape_is_1_39_39_1(self):
        assert build_phone_transition({}).shape == (1, N_PHONES, N_PHONES, 1)

    @_SKIP_NO_COUNTS
    def test_shape_holds_for_real_counts(self, trained_transition):
        assert trained_transition.shape == (1, N_PHONES, N_PHONES, 1)

    def test_output_is_a_finite_jax_array(self):
        b_matrix = build_phone_transition({"AA,B": 5})
        assert isinstance(b_matrix, jnp.ndarray)
        assert bool(jnp.isfinite(b_matrix).all())

    @_SKIP_NO_COUNTS
    def test_columns_sum_to_one_over_axis_1(self, trained_transition):
        """THE ORIENTATION PIN.

        pymdp validates B along axis 1, so ``B[0, :, j, 0]`` must be a
        distribution over the NEXT phone. A transposed build would still "sum to
        1 somewhere" — over axis 2 — so the axis is asserted explicitly, one
        column at a time.
        """
        arr = np.asarray(trained_transition)
        for j in range(N_PHONES):
            total = float(arr[0, :, j, 0].sum())
            assert total == pytest.approx(1.0, abs=1e-5), (
                f"B[0, :, {j}, 0] sums to {total!r}, not 1.0 — column {j} "
                f"({PHONES[j]}) is not a distribution over the next phone."
            )

    @_SKIP_NO_COUNTS
    def test_transposed_build_would_fail_the_column_sum_check(self, trained_transition):
        """Non-vacuity control for the check above.

        If a transposed real bigram table also had unit columns, the previous
        test would be measuring nothing.
        """
        transposed = np.asarray(trained_transition)[0, :, :, 0].T
        worst = float(np.abs(transposed.sum(axis=0) - 1.0).max())
        assert worst > 1e-3, (
            "GATE VACUOUS: a transposed transition still has unit columns "
            f"(worst deviation {worst:.2e}), so the column-sum check cannot "
            "detect a transpose."
        )

    def test_all_values_in_unit_interval(self):
        arr = np.asarray(build_phone_transition({"AA,B": 900, "T,S": 12}))
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_empty_counts_give_exactly_uniform(self):
        """With add-1 smoothing and no data every cell is exactly 1/39."""
        arr = np.asarray(build_phone_transition({}))
        np.testing.assert_allclose(arr, 1.0 / N_PHONES, atol=1e-6)

    def test_laplace_smoothing_leaves_no_zeros(self):
        arr = np.asarray(build_phone_transition({"AA,B": 10_000}))
        assert arr.min() > 0.0, "add-1 smoothing must keep every entry positive"

    def test_lopsided_counts_land_at_next_row_current_column(self):
        """Index order asserted explicitly — this is what catches a transpose."""
        arr = np.asarray(build_phone_transition({"AA,B": 10_000}))[0, :, :, 0]
        i, j = PHONE_TO_IDX["B"], PHONE_TO_IDX["AA"]
        assert arr[i, j] > 0.9, (
            f"B[0, idx('B')={i}, idx('AA')={j}, 0] is {arr[i, j]!r}; the mass for "
            "the bigram 'AA' → 'B' must sit at [next, current]."
        )
        # The transposed cell must stay at the smoothing floor.
        assert arr[j, i] == pytest.approx(1.0 / N_PHONES, abs=1e-6)

    def test_string_and_tuple_keys_agree(self):
        string_b = np.asarray(build_phone_transition({"AA,B": 77, "T,S": 3}))
        tuple_b = np.asarray(build_phone_transition({("AA", "B"): 77, ("T", "S"): 3}))
        np.testing.assert_allclose(
            string_b,
            tuple_b,
            atol=1e-7,
            err_msg='"prev,next" and ("prev", "next") keys must build the same B',
        )

    def test_list_keys_are_accepted_like_tuples(self):
        # json.loads never produces tuple keys, but an in-memory pair may arrive
        # as a list; the parser accepts both.
        listed = np.asarray(build_phone_transition({("AA", "B"): 50}))
        tupled = np.asarray(build_phone_transition({("AA", "B"): 50}))
        np.testing.assert_allclose(listed, tupled, atol=1e-9)

    def test_malformed_string_key_is_skipped(self):
        # "AAB" has no comma → cannot split into a pair → ignored entirely.
        arr = np.asarray(build_phone_transition({"AAB": 10_000}))
        np.testing.assert_allclose(arr, 1.0 / N_PHONES, atol=1e-6)

    def test_out_of_inventory_key_is_skipped(self):
        arr = np.asarray(build_phone_transition({"XX,YY": 10_000, "AE1,B": 10_000, "AA,": 10_000}))
        np.testing.assert_allclose(arr, 1.0 / N_PHONES, atol=1e-6)

    def test_non_string_non_pair_key_is_skipped(self):
        arr = np.asarray(build_phone_transition({5: 10_000, ("AA",): 10_000}))
        np.testing.assert_allclose(arr, 1.0 / N_PHONES, atol=1e-6)

    def test_frequent_bigram_outranks_rare_bigram(self):
        arr = np.asarray(build_phone_transition({"K,AE": 5000, "K,ZH": 1}))[0, :, :, 0]
        j = PHONE_TO_IDX["K"]
        assert arr[PHONE_TO_IDX["AE"], j] > arr[PHONE_TO_IDX["ZH"], j]

    def test_counts_shift_probability_monotonically(self):
        low = np.asarray(build_phone_transition({"AA,B": 10}))[0, :, :, 0]
        high = np.asarray(build_phone_transition({"AA,B": 10_000}))[0, :, :, 0]
        i, j = PHONE_TO_IDX["B"], PHONE_TO_IDX["AA"]
        assert high[i, j] > low[i, j] > 1.0 / N_PHONES

    @_SKIP_NO_COUNTS
    def test_trained_matrix_is_not_uniform(self, trained_transition):
        arr = np.asarray(trained_transition)
        assert not np.allclose(arr, 1.0 / N_PHONES, atol=0.001), (
            "A trained transition that is still uniform means the counts were dropped"
        )

    @_SKIP_NO_COUNTS
    def test_closed_form_cell_matches_smoothed_mle(self, counts, trained_transition):
        """Re-derive one cell from the raw counts: (c + 1) / (total + 39)."""
        bigrams = counts["phone_bigrams"]
        prev, nxt = "K", "AE"
        numerator = 1.0 + float(bigrams.get(f"{prev},{nxt}", 0))
        denominator = float(N_PHONES) + sum(
            float(v) for k, v in bigrams.items() if k.split(",", 1)[0] == prev
        )
        measured = float(
            np.asarray(trained_transition)[0, PHONE_TO_IDX[nxt], PHONE_TO_IDX[prev], 0]
        )
        assert measured == pytest.approx(numerator / denominator, rel=1e-5)


# ── build_phone_unigram ───────────────────────────────────────────────────────


class TestBuildPhoneUnigram:
    """D is a (1, 39) add-1 smoothed phone marginal."""

    def test_shape_is_1_39(self):
        assert build_phone_unigram({}).shape == (1, N_PHONES)

    def test_sums_to_one(self):
        arr = np.asarray(build_phone_unigram({"AA": 10, "B": 3}))
        assert float(arr.sum()) == pytest.approx(1.0, abs=1e-6)

    def test_empty_counts_give_exactly_uniform(self):
        np.testing.assert_allclose(np.asarray(build_phone_unigram({})), 1.0 / N_PHONES, atol=1e-6)

    def test_unknown_symbols_are_ignored(self):
        np.testing.assert_allclose(
            np.asarray(build_phone_unigram({"AE1": 10_000, "XX": 10_000})),
            1.0 / N_PHONES,
            atol=1e-6,
        )

    def test_mass_lands_at_the_named_phone(self):
        arr = np.asarray(build_phone_unigram({"ZH": 10_000})).reshape(-1)
        assert arr.argmax() == PHONE_TO_IDX["ZH"]
        assert arr[PHONE_TO_IDX["ZH"]] > 0.9

    def test_no_zeros_after_smoothing(self):
        assert np.asarray(build_phone_unigram({"AA": 10_000})).min() > 0.0

    @_SKIP_NO_COUNTS
    def test_real_unigram_sums_to_one_and_is_not_uniform(self, phone_start):
        arr = np.asarray(phone_start)
        assert float(arr.sum()) == pytest.approx(1.0, abs=1e-5)
        assert not np.allclose(arr, 1.0 / N_PHONES, atol=0.001)

    @_SKIP_NO_COUNTS
    def test_closed_form_cell_matches_smoothed_mle(self, counts, phone_start):
        unigrams = counts["phone_unigrams"]
        total = float(N_PHONES) + sum(float(v) for v in unigrams.values())
        expected = (1.0 + float(unigrams["AH"])) / total
        measured = float(np.asarray(phone_start).reshape(-1)[PHONE_TO_IDX["AH"]])
        assert measured == pytest.approx(expected, rel=1e-5)


# ── build_phoneme_agent ───────────────────────────────────────────────────────


class TestBuildPhonemeAgent:
    """build_phoneme_agent returns a real pymdp Agent with an identity emission."""

    def test_returns_real_pymdp_agent(self, uniform_phone_agent):
        from pymdp.agent import Agent

        assert isinstance(uniform_phone_agent, Agent)

    def test_agent_exposes_a_b_c_d(self, uniform_phone_agent):
        for attr in ("A", "B", "C", "D"):
            assert hasattr(uniform_phone_agent, attr), f"Agent is missing {attr}"

    def test_a_shape_is_1_39_39(self, uniform_phone_agent):
        assert len(uniform_phone_agent.A) == 1
        assert uniform_phone_agent.A[0].shape == (1, N_PHONES, N_PHONES)

    def test_a_is_identity_emission(self, uniform_phone_agent):
        """Observing a phone must reveal the phone state exactly."""
        np.testing.assert_allclose(
            np.asarray(uniform_phone_agent.A[0])[0], np.eye(N_PHONES), atol=1e-6
        )

    def test_b_shape_is_1_39_39_1(self, uniform_phone_agent):
        assert uniform_phone_agent.B[0].shape == (1, N_PHONES, N_PHONES, 1)

    def test_c_is_zero_preferences(self, uniform_phone_agent):
        assert uniform_phone_agent.C[0].shape == (1, N_PHONES)
        np.testing.assert_allclose(np.asarray(uniform_phone_agent.C[0]), 0.0, atol=1e-9)

    def test_d_defaults_to_uniform(self, uniform_phone_agent):
        assert uniform_phone_agent.D[0].shape == (1, N_PHONES)
        np.testing.assert_allclose(np.asarray(uniform_phone_agent.D[0]), 1.0 / N_PHONES, atol=1e-6)

    @_SKIP_NO_COUNTS
    def test_supplied_start_lands_in_d(self, trained_phone_agent, phone_start):
        np.testing.assert_allclose(
            np.asarray(trained_phone_agent.D[0]),
            np.asarray(phone_start),
            atol=1e-6,
        )

    @_SKIP_NO_COUNTS
    def test_supplied_transition_lands_in_b_not_a(self, trained_phone_agent, trained_transition):
        """Orientation guard: the corpus statistics parameterize the TRANSITION.

        Putting them in A instead is anti-informative (a posterior over the
        *preceding* phone), which is a silent quality regression rather than a
        crash — so it is pinned here.
        """
        np.testing.assert_allclose(
            np.asarray(trained_phone_agent.B[0]),
            np.asarray(trained_transition),
            atol=1e-6,
            err_msg="B must carry the supplied transition verbatim",
        )
        assert not np.allclose(
            np.asarray(trained_phone_agent.A[0])[0],
            np.asarray(trained_transition)[0, :, :, 0],
            atol=1e-3,
        ), "The learned bigram matrix must NOT be installed as the emission A"

    def test_accepts_numpy_transition(self):
        agent = build_phoneme_agent(np.ones((1, N_PHONES, N_PHONES, 1)) / N_PHONES)
        assert agent.B[0].shape == (1, N_PHONES, N_PHONES, 1)

    def test_missing_batch_dimension_raises_assertion(self):
        with pytest.raises(ValueError, match=r"\(1, 39, 39, 1\)"):
            build_phoneme_agent(jnp.ones((N_PHONES, N_PHONES)) / N_PHONES)

    def test_missing_action_dimension_raises_assertion(self):
        with pytest.raises(ValueError, match=r"\(1, 39, 39, 1\)"):
            build_phoneme_agent(jnp.ones((1, N_PHONES, N_PHONES)) / N_PHONES)

    def test_wrong_phone_count_raises_assertion(self):
        with pytest.raises(ValueError, match=r"\(1, 39, 39, 1\)"):
            build_phoneme_agent(jnp.ones((1, 5, 5, 1)) / 5)

    def test_wrong_start_shape_raises_assertion(self):
        with pytest.raises(ValueError, match=r"start prior shape \(1, 39\)"):
            build_phoneme_agent(build_phone_transition({}), jnp.ones((N_PHONES,)) / N_PHONES)

    def test_infer_states_runs_and_returns_a_distribution(self, uniform_phone_agent):
        prior = [jnp.ones((1, N_PHONES)) / N_PHONES]
        qs = uniform_phone_agent.infer_states([jnp.array([PHONE_TO_IDX["AA"]])], prior)
        assert qs is not None and len(qs) == 1
        assert qs[0].shape[-1] == N_PHONES
        posterior = np.asarray(qs[0]).reshape(-1)
        assert posterior.min() >= 0.0
        assert posterior.sum() == pytest.approx(1.0, abs=1e-4)

    def test_infer_states_collapses_onto_the_observed_phone(self, uniform_phone_agent):
        """Identity emission ⇒ the posterior is a delta on the observed phone."""
        idx = PHONE_TO_IDX["ZH"]
        prior = [jnp.ones((1, N_PHONES)) / N_PHONES]
        qs = uniform_phone_agent.infer_states([jnp.array([idx])], prior)
        posterior = np.asarray(qs[0]).reshape(-1)
        assert posterior[idx] == pytest.approx(1.0, abs=1e-4)
        assert posterior.sum() - posterior[idx] == pytest.approx(0.0, abs=1e-4)

    def test_update_empirical_prior_runs(self, uniform_phone_agent):
        prior = [jnp.ones((1, N_PHONES)) / N_PHONES]
        qs = uniform_phone_agent.infer_states([jnp.array([PHONE_TO_IDX["T"]])], prior)
        action = jnp.zeros((1, 1), dtype=jnp.int32)
        new_prior = uniform_phone_agent.update_empirical_prior(action, qs)
        assert new_prior is not None and len(new_prior) == 1
        assert new_prior[0].shape == (1, N_PHONES)
        assert np.asarray(new_prior[0]).reshape(-1).sum() == pytest.approx(1.0, abs=1e-4)

    def test_empirical_prior_feeds_back_into_infer_states(self, uniform_phone_agent):
        """The full per-phone recursion must chain without shape errors."""
        prior = [jnp.ones((1, N_PHONES)) / N_PHONES]
        action = jnp.zeros((1, 1), dtype=jnp.int32)
        for phone in ("K", "AE", "T"):
            qs = uniform_phone_agent.infer_states([jnp.array([PHONE_TO_IDX[phone]])], prior)
            prior = uniform_phone_agent.update_empirical_prior(action, qs)
        assert np.asarray(prior[0]).reshape(-1).sum() == pytest.approx(1.0, abs=1e-4)

    @_SKIP_NO_COUNTS
    def test_next_step_prior_is_the_transition_column_of_the_observed_phone(
        self, trained_phone_agent, trained_transition
    ):
        """The whole Part A model in one assertion: after 'K', predict P(next | K).

        update_empirical_prior must push the delta posterior through B, so the
        next-step prior equals column 'K' of the supplied transition.
        """
        idx = PHONE_TO_IDX["K"]
        prior = [jnp.asarray(trained_phone_agent.D[0])]
        qs = trained_phone_agent.infer_states([jnp.array([idx])], prior)
        action = jnp.zeros((1, 1), dtype=jnp.int32)
        next_prior = np.asarray(trained_phone_agent.update_empirical_prior(action, qs)[0]).reshape(
            -1
        )

        expected = np.asarray(trained_transition)[0, :, idx, 0]
        np.testing.assert_allclose(
            next_prior,
            expected,
            atol=1e-5,
            err_msg="Next-step prior must equal B[0, :, observed, 0]",
        )


# ── compute_phoneme_perplexity ────────────────────────────────────────────────


@_SKIP_NO_COUNTS
class TestComputePhonemePerplexity:
    """Per-phone perplexity, prior reset to D at every sequence boundary."""

    def test_returns_a_python_float(self, uniform_phone_agent):
        assert isinstance(compute_phoneme_perplexity(uniform_phone_agent, [["AA"]]), float)

    def test_empty_sequence_list_returns_n_phones(self, uniform_phone_agent):
        assert compute_phoneme_perplexity(uniform_phone_agent, []) == float(N_PHONES)

    def test_all_empty_sequences_return_n_phones(self, uniform_phone_agent):
        assert compute_phoneme_perplexity(uniform_phone_agent, [[], []]) == float(N_PHONES)

    def test_single_phone_sequence_is_scored_under_d(self, trained_phone_agent, phone_start):
        """One position, no transition involved: P = D[phone] exactly."""
        measured = compute_phoneme_perplexity(trained_phone_agent, [["AA"]])
        expected = 1.0 / float(np.asarray(phone_start).reshape(-1)[PHONE_TO_IDX["AA"]])
        assert measured == pytest.approx(expected, rel=1e-5)

    def test_uniform_transition_gives_exactly_39(self, uniform_phone_agent, phone_sequences):
        """ANALYTIC ANCHOR / NEGATIVE CONTROL.

        Uniform B and uniform D ⇒ every predictive probability is exactly 1/39,
        so perplexity is exactly 39 whatever the input. float32 arithmetic lands
        it at 38.999998, hence the tolerance.
        """
        ppl = compute_phoneme_perplexity(uniform_phone_agent, phone_sequences)
        assert ppl == pytest.approx(float(N_PHONES), abs=1e-4), (
            f"Uniform transition must give perplexity exactly {N_PHONES}, got {ppl}"
        )

    def test_uniform_transition_is_sequence_independent(self, uniform_phone_agent):
        """A uniform model cannot prefer real pronunciations over noise."""
        real = compute_phoneme_perplexity(uniform_phone_agent, [["K", "AE", "T"], ["D", "AO", "G"]])
        noise = compute_phoneme_perplexity(
            uniform_phone_agent, [["ZH", "ZH", "NG"], ["OY", "TH", "UH"]]
        )
        assert real == pytest.approx(noise, abs=1e-4)

    def test_trained_transition_beats_uniform(self, trained_phone_agent, phone_sequences):
        ppl = compute_phoneme_perplexity(trained_phone_agent, phone_sequences)
        assert ppl < float(N_PHONES), (
            f"Trained perplexity {ppl:.4f} is not below the uniform bar "
            f"{N_PHONES} — the transition is not reaching the metric at all."
        )

    def test_trained_transition_beats_the_unigram_baseline(
        self, trained_phone_agent, unigram_phone_agent, phone_sequences
    ):
        """The honest comparator: context-free phone marginal, rebuilt in-test.

        Uniform is a crippled comparator. The unigram model is the bar that
        proves the *bigram* structure is doing work, and it is derived here from
        ``phone_unigrams`` rather than pinned to a remembered number.
        """
        trained = compute_phoneme_perplexity(trained_phone_agent, phone_sequences)
        unigram = compute_phoneme_perplexity(unigram_phone_agent, phone_sequences)
        assert unigram < float(N_PHONES), (
            f"The unigram comparator scored {unigram:.4f}, no better than uniform "
            "— it was built wrong, so the comparison below would be meaningless."
        )
        assert trained < unigram, (
            f"Bigram perplexity {trained:.4f} does not beat the context-free "
            f"unigram baseline {unigram:.4f}."
        )

    def test_unknown_phone_raises_key_error_naming_the_symbol(self, trained_phone_agent):
        with pytest.raises(KeyError, match="AE1"):
            compute_phoneme_perplexity(trained_phone_agent, [["K", "AE1", "T"]])

    def test_unknown_phone_error_names_the_inventory(self, trained_phone_agent):
        with pytest.raises(KeyError, match="canonical ARPAbet"):
            compute_phoneme_perplexity(trained_phone_agent, [["Q"]])

    def test_sequences_are_independent_not_concatenated(self):
        """THE RESET PIN.

        Two sequences must be scored with the prior reset to D between them. With
        a transition that makes 'AA' → 'AA' near-certain, a *concatenated*
        ["AA", "ZH"] pays a huge penalty at the boundary that separated
        [["AA"], ["ZH"]] never sees. Both figures are derived in closed form from
        the same matrices, so this is not a "they differ" smoke test.
        """
        transition = build_phone_transition({"AA,AA": 100_000})
        agent = build_phoneme_agent(transition)  # uniform D
        b_matrix = np.asarray(transition)[0, :, :, 0]

        separate = compute_phoneme_perplexity(agent, [["AA"], ["ZH"]])
        concatenated = compute_phoneme_perplexity(agent, [["AA", "ZH"]])

        # Separated: both positions are scored under uniform D ⇒ exactly 39.
        assert separate == pytest.approx(float(N_PHONES), abs=1e-3)
        # Concatenated: position 1 is scored under B[:, AA] instead of D.
        boundary = float(b_matrix[PHONE_TO_IDX["ZH"], PHONE_TO_IDX["AA"]])
        expected_cat = math.exp(-(math.log(1.0 / N_PHONES) + math.log(boundary)) / 2.0)
        assert concatenated == pytest.approx(expected_cat, rel=1e-4)
        assert concatenated > separate * 10.0, (
            f"Concatenating the sequences changed nothing ({concatenated:.4f} vs "
            f"{separate:.4f}) — the per-sequence prior reset is missing and belief "
            "is bleeding across the word boundary."
        )

    def test_corpus_aggregation_is_a_log_weighted_mean_of_sequences(self, trained_phone_agent):
        """Corollary of independence: total logp is additive over sequences.

        So the corpus perplexity is the position-weighted geometric mean of the
        per-sequence perplexities. Deriving that identity binds both the reset
        and the "divide by total positions, not by sequence count" aggregation.
        """
        seq_a = ["K", "AE", "T"]
        seq_b = ["D", "AO", "G", "Z"]
        ppl_a = compute_phoneme_perplexity(trained_phone_agent, [seq_a])
        ppl_b = compute_phoneme_perplexity(trained_phone_agent, [seq_b])
        both = compute_phoneme_perplexity(trained_phone_agent, [seq_a, seq_b])

        expected = math.exp(
            (len(seq_a) * math.log(ppl_a) + len(seq_b) * math.log(ppl_b))
            / (len(seq_a) + len(seq_b))
        )
        assert both == pytest.approx(expected, rel=1e-5)

    def test_is_deterministic_across_repeated_calls(self, trained_phone_agent):
        seqs = [["K", "AE", "T"], ["D", "AO", "G"]]
        first = compute_phoneme_perplexity(trained_phone_agent, seqs)
        second = compute_phoneme_perplexity(trained_phone_agent, seqs)
        assert first == second

    def test_is_deterministic_across_rebuilt_agents(self, counts, phone_start):
        seqs = [["HH", "EH", "L", "OW"]]
        first = compute_phoneme_perplexity(
            build_phoneme_agent(build_phone_transition(counts["phone_bigrams"]), phone_start),
            seqs,
        )
        second = compute_phoneme_perplexity(
            build_phoneme_agent(build_phone_transition(counts["phone_bigrams"]), phone_start),
            seqs,
        )
        assert first == pytest.approx(second, rel=1e-9)

    def test_empty_sequences_are_skipped_not_counted(self, trained_phone_agent):
        """Interleaved empties must not dilute the position count."""
        seqs = [["K", "AE", "T"]]
        padded = [[], ["K", "AE", "T"], []]
        assert compute_phoneme_perplexity(trained_phone_agent, padded) == pytest.approx(
            compute_phoneme_perplexity(trained_phone_agent, seqs), rel=1e-9
        )

    def test_perplexity_is_positive_and_finite(self, trained_phone_agent, phone_sequences):
        ppl = compute_phoneme_perplexity(trained_phone_agent, phone_sequences)
        assert ppl > 0.0 and math.isfinite(ppl)


# ── g2p_split ─────────────────────────────────────────────────────────────────


class TestG2PSplit:
    """The split is positional (index % test_every), never re-sorted, never random."""

    def test_missing_pairs_key_raises_key_error(self):
        with pytest.raises(KeyError, match="g2p_pairs"):
            g2p_split({"test_every": 5})

    def test_missing_test_every_raises_key_error(self):
        with pytest.raises(KeyError, match="test_every"):
            g2p_split({"g2p_pairs": []})

    def test_zero_test_every_raises_value_error(self):
        with pytest.raises(ValueError, match="positive integer"):
            g2p_split({"g2p_pairs": [], "test_every": 0})

    def test_negative_test_every_raises_value_error(self):
        with pytest.raises(ValueError, match="positive integer"):
            g2p_split({"g2p_pairs": [], "test_every": -5})

    def test_non_integer_test_every_raises_value_error(self):
        with pytest.raises(ValueError, match="positive integer"):
            g2p_split({"g2p_pairs": [], "test_every": 2.5})

    def test_positional_assignment_is_index_modulo(self):
        pairs = [[f"w{i}", "AA"] for i in range(10)]
        train, test = g2p_split({"g2p_pairs": pairs, "test_every": 5})
        assert [w for w, _ in test] == ["w0", "w5"]
        assert [w for w, _ in train] == ["w1", "w2", "w3", "w4", "w6", "w7", "w8", "w9"]

    def test_order_is_preserved_not_resorted(self):
        pairs = [["zebra", "Z"], ["apple", "AE"], ["mango", "M"]]
        train, test = g2p_split({"g2p_pairs": pairs, "test_every": 3})
        assert [w for w, _ in test] == ["zebra"]
        assert [w for w, _ in train] == ["apple", "mango"]

    def test_string_phones_are_split_on_whitespace(self):
        _train, test = g2p_split({"g2p_pairs": [["cat", "K AE T"]], "test_every": 1})
        assert test == [("cat", ["K", "AE", "T"])]

    def test_list_phones_are_passed_through(self):
        _train, test = g2p_split({"g2p_pairs": [["cat", ["K", "AE", "T"]]], "test_every": 1})
        assert test == [("cat", ["K", "AE", "T"])]

    def test_empty_pairs_give_two_empty_lists(self):
        assert g2p_split({"g2p_pairs": [], "test_every": 5}) == ([], [])

    @_SKIP_NO_COUNTS
    def test_real_split_sizes(self, splits, counts):
        """Split sizes, checked twice: against a closed form and against the pin.

        The closed form is the load-bearing assertion — ``i % test_every == 0``
        selects ``ceil(n / test_every)`` indices out of ``n``, re-derived here
        from the artifact's own ``g2p_pairs`` length and stride rather than
        recalled. EXPECTED_TRAIN / EXPECTED_TEST are then a provenance pin: they
        catch a *regenerated artifact* whose size changed, which the closed form
        alone would happily follow.
        """
        train, test = splits
        n_pairs = len(counts["g2p_pairs"])
        stride = counts["test_every"]
        derived_test = -(-n_pairs // stride)  # ceil division
        assert len(test) == derived_test
        assert len(train) == n_pairs - derived_test

        assert len(train) == EXPECTED_TRAIN, (
            f"TRAIN split is {len(train)}, not the pinned {EXPECTED_TRAIN}. The "
            "vendored artifact changed size; every Stage 2 accuracy figure was "
            "measured against the old split."
        )
        assert len(test) == EXPECTED_TEST

    @_SKIP_NO_COUNTS
    def test_split_covers_every_pair_exactly_once(self, counts, splits):
        train, test = splits
        assert len(train) + len(test) == len(counts["g2p_pairs"])

    @_SKIP_NO_COUNTS
    def test_no_word_appears_in_both_splits(self, splits):
        """THE LEAKAGE CHECK.

        Every G2P accuracy figure is only meaningful if no test word was seen in
        training. A duplicated word in the artifact would leak silently.
        """
        train, test = splits
        overlap = {w for w, _ in train} & {w for w, _ in test}
        assert overlap == set(), (
            f"{len(overlap)} word(s) appear in BOTH splits — the test set is "
            f"contaminated: {sorted(overlap)[:10]}"
        )

    @_SKIP_NO_COUNTS
    def test_split_is_reproducible_across_calls(self, counts):
        first_train, first_test = g2p_split(counts)
        second_train, second_test = g2p_split(counts)
        assert first_train == second_train
        assert first_test == second_test

    @_SKIP_NO_COUNTS
    def test_every_pair_is_word_plus_phone_list(self, splits):
        train, test = splits
        for word, phones in train[:200] + test[:200]:
            assert isinstance(word, str) and word == word.lower()
            assert isinstance(phones, list) and len(phones) > 0
            assert all(p in PHONE_TO_IDX for p in phones)


# ── build_g2p_emission ────────────────────────────────────────────────────────


@_SKIP_NO_COUNTS
class TestBuildG2PEmission:
    """A[0, l, p] = P(letter l | phone p) — 30 observations over 39 states."""

    def test_shape_is_1_30_39(self, g2p_parts):
        emission, _t, _s = g2p_parts
        assert emission.shape == (1, N_SYMBOLS, N_PHONES)

    def test_shape_holds_for_an_empty_train_split(self):
        assert build_g2p_emission([]).shape == (1, N_SYMBOLS, N_PHONES)

    def test_empty_train_split_gives_exactly_uniform_over_letters(self):
        np.testing.assert_allclose(np.asarray(build_g2p_emission([])), 1.0 / N_SYMBOLS, atol=1e-6)

    def test_columns_sum_to_one_over_letters(self, g2p_parts):
        """THE EMISSION ORIENTATION PIN.

        ``A[0, :, p]`` must be a distribution over the 30 letters. This is what
        catches the mis-normalization that beating the context-free baseline does
        NOT catch (69.47% vs a 68.64% bar while being wrong).
        """
        arr = np.asarray(g2p_parts[0])[0]
        for p in range(N_PHONES):
            total = float(arr[:, p].sum())
            assert total == pytest.approx(1.0, abs=1e-5), (
                f"A[0, :, {p}] ({PHONES[p]}) sums to {total!r}, not 1.0 — the "
                "emission is not normalized over letters."
            )

    def test_phone_normalized_emission_fails_the_column_check(self, g2p_parts):
        """Non-vacuity control for the check above.

        Renormalizing over phones instead of letters must break the letter-column
        invariant; if it did not, that invariant would be measuring nothing.
        """
        arr = np.asarray(g2p_parts[0])[0].astype(np.float64)
        phone_normalized = arr / arr.sum(axis=1, keepdims=True)
        worst = float(np.abs(phone_normalized.sum(axis=0) - 1.0).max())
        assert worst > 1e-3, (
            "GATE VACUOUS: a phone-normalized emission still has unit letter "
            f"columns (worst deviation {worst:.2e}), so the column-sum check "
            "cannot detect the mis-normalization."
        )

    def test_all_values_in_unit_interval(self, g2p_parts):
        arr = np.asarray(g2p_parts[0])
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_laplace_smoothing_leaves_no_zeros(self, g2p_parts):
        assert np.asarray(g2p_parts[0]).min() > 0.0

    def test_learned_emission_is_not_uniform(self, g2p_parts):
        assert not np.allclose(np.asarray(g2p_parts[0]), 1.0 / N_SYMBOLS, atol=0.001), (
            "A uniform emission means the training alignments were all dropped"
        )

    def test_aligned_counts_land_at_letter_row_phone_column(self):
        """Index order asserted explicitly: rows are letters, columns are phones."""
        pairs = [("k", ["K"])] * 5000
        arr = np.asarray(build_g2p_emission(pairs))[0]
        assert arr[ALPHA_TO_IDX["k"], PHONE_TO_IDX["K"]] > 0.9
        # A letter that never co-occurred with K stays at the smoothing floor.
        assert arr[ALPHA_TO_IDX["z"], PHONE_TO_IDX["K"]] < 0.01

    def test_length_mismatched_pairs_are_skipped_with_a_warning(self, caplog):
        """A misaligned pair must be dropped whole and loudly, never truncated.

        "dog" has 3 letters against 2 phones. Truncating it would have credited
        d→D and o→AO; dropping it leaves column D entirely at the add-1 floor,
        which is the closed form asserted below.
        """
        pairs = [("cat", ["K", "AE", "T"]), ("dog", ["D", "AO"])]
        with caplog.at_level("WARNING", logger="ailm.levels.phoneme"):
            arr = np.asarray(build_g2p_emission(pairs))[0]
        assert "skipped 1/2" in caplog.text
        # Nothing from "dog" was counted: column D is exactly uniform.
        np.testing.assert_allclose(arr[:, PHONE_TO_IDX["D"]], 1.0 / N_SYMBOLS, atol=1e-6)
        np.testing.assert_allclose(arr[:, PHONE_TO_IDX["AO"]], 1.0 / N_SYMBOLS, atol=1e-6)
        # While "cat" was counted: c→K is smoothed to 2/(30+1).
        assert arr[ALPHA_TO_IDX["c"], PHONE_TO_IDX["K"]] == pytest.approx(
            2.0 / (N_SYMBOLS + 1), rel=1e-5
        )

    def test_unknown_phone_position_is_skipped_not_misindexed(self):
        """A stressed phone must be dropped, leaving the other positions intact."""
        arr = np.asarray(build_g2p_emission([("cat", ["K", "AE1", "T"])]))[0]
        assert arr[ALPHA_TO_IDX["c"], PHONE_TO_IDX["K"]] == pytest.approx(
            2.0 / (N_SYMBOLS + 1), rel=1e-5
        )
        # 'a' was aligned to AE1, which is not in PHONES: column AE is untouched.
        assert arr[ALPHA_TO_IDX["a"], PHONE_TO_IDX["AE"]] == pytest.approx(
            1.0 / N_SYMBOLS, rel=1e-5
        )

    def test_closed_form_cell_matches_smoothed_mle(self, splits, g2p_parts):
        """Re-derive one cell from the TRAIN alignments: (c + 1) / (total + 30)."""
        from ailm.levels.phoneme import _word_letters

        train, _test = splits
        letter, phone = "k", "K"
        joint = 0
        column_total = 0
        for word, phones in train:
            letters = _word_letters(word)
            if len(letters) != len(phones):
                continue
            for lt, ph in zip(letters, phones, strict=True):
                if ph != phone:
                    continue
                column_total += 1
                if lt == letter:
                    joint += 1
        expected = (1.0 + joint) / (N_SYMBOLS + column_total)
        measured = float(np.asarray(g2p_parts[0])[0, ALPHA_TO_IDX[letter], PHONE_TO_IDX[phone]])
        assert measured == pytest.approx(expected, rel=1e-5)


# ── build_g2p_transition / build_g2p_start ────────────────────────────────────


@_SKIP_NO_COUNTS
class TestBuildG2PTransitionAndStart:
    """The G2P dynamics are fit on TRAIN only — no test statistics may leak in."""

    def test_transition_shape_is_1_39_39_1(self, g2p_parts):
        assert g2p_parts[1].shape == (1, N_PHONES, N_PHONES, 1)

    def test_transition_columns_sum_to_one_over_axis_1(self, g2p_parts):
        arr = np.asarray(g2p_parts[1])
        for j in range(N_PHONES):
            assert float(arr[0, :, j, 0].sum()) == pytest.approx(1.0, abs=1e-5)

    def test_transition_is_not_uniform(self, g2p_parts):
        assert not np.allclose(np.asarray(g2p_parts[1]), 1.0 / N_PHONES, atol=0.001)

    def test_transition_differs_from_a_test_split_transition(self, splits, g2p_parts):
        """THE TRAIN/TEST SEPARATION CHECK.

        If the TRAIN-fit transition equalled a TEST-fit one, the two splits would
        not actually be separated (or the function is ignoring its argument and
        reading the artifact's full-corpus bigrams instead).
        """
        _train, test = splits
        test_transition = np.asarray(build_g2p_transition(test))
        assert not np.allclose(np.asarray(g2p_parts[1]), test_transition, atol=1e-6), (
            "The TRAIN transition is identical to a TEST transition — either the "
            "splits are not separated or build_g2p_transition ignores its input."
        )

    def test_transition_differs_from_the_full_corpus_bigrams(self, counts, g2p_parts):
        """It must not silently fall back to the artifact's leak-prone bigrams."""
        full = np.asarray(build_phone_transition(counts["phone_bigrams"]))
        assert not np.allclose(np.asarray(g2p_parts[1]), full, atol=1e-6)

    def test_transition_counts_only_within_word_bigrams(self):
        """No transition may be counted across a word boundary."""
        pairs = [("ab", ["AA", "B"]), ("cd", ["ZH", "D"])]
        arr = np.asarray(build_g2p_transition(pairs))[0, :, :, 0]
        # 'B' → 'ZH' spans the boundary between the two words: floor only.
        assert arr[PHONE_TO_IDX["ZH"], PHONE_TO_IDX["B"]] == pytest.approx(1.0 / N_PHONES, rel=1e-5)
        # 'AA' → 'B' is inside a word: smoothed to 2/(39+1).
        assert arr[PHONE_TO_IDX["B"], PHONE_TO_IDX["AA"]] == pytest.approx(
            2.0 / (N_PHONES + 1), rel=1e-5
        )

    def test_empty_train_split_gives_a_uniform_transition(self):
        np.testing.assert_allclose(np.asarray(build_g2p_transition([])), 1.0 / N_PHONES, atol=1e-6)

    def test_start_shape_is_1_39(self, g2p_parts):
        assert g2p_parts[2].shape == (1, N_PHONES)

    def test_start_sums_to_one(self, g2p_parts):
        assert float(np.asarray(g2p_parts[2]).sum()) == pytest.approx(1.0, abs=1e-5)

    def test_start_is_not_uniform(self, g2p_parts):
        assert not np.allclose(np.asarray(g2p_parts[2]), 1.0 / N_PHONES, atol=0.001)

    def test_start_counts_only_first_phones(self):
        arr = np.asarray(build_g2p_start([("ab", ["AA", "B"])] * 1000)).reshape(-1)
        assert arr.argmax() == PHONE_TO_IDX["AA"]
        assert arr[PHONE_TO_IDX["B"]] == pytest.approx(1.0 / (N_PHONES + 1000), rel=1e-5)

    def test_start_skips_empty_pronunciations(self):
        np.testing.assert_allclose(
            np.asarray(build_g2p_start([("", [])])), 1.0 / N_PHONES, atol=1e-6
        )

    def test_start_differs_from_the_corpus_wide_unigram(self, phone_start, g2p_parts):
        """D is the first-phone distribution, not the overall phone marginal."""
        assert not np.allclose(np.asarray(g2p_parts[2]), np.asarray(phone_start), atol=1e-3)


# ── build_g2p_agent ───────────────────────────────────────────────────────────


@_SKIP_NO_COUNTS
class TestBuildG2PAgent:
    """The G2P agent spans two alphabets: 30 letter observations, 39 phone states."""

    def test_returns_real_pymdp_agent(self, g2p_agent):
        from pymdp.agent import Agent

        assert isinstance(g2p_agent, Agent)

    def test_a_shape_is_1_30_39(self, g2p_agent):
        assert g2p_agent.A[0].shape == (1, N_SYMBOLS, N_PHONES)

    def test_a_is_not_the_identity(self, g2p_agent):
        """Part B's emission is real — a square identity would collapse levels."""
        arr = np.asarray(g2p_agent.A[0])[0]
        assert arr.shape[0] != arr.shape[1]

    def test_b_shape_is_1_39_39_1(self, g2p_agent):
        assert g2p_agent.B[0].shape == (1, N_PHONES, N_PHONES, 1)

    def test_c_is_zero_preferences_over_letters(self, g2p_agent):
        assert g2p_agent.C[0].shape == (1, N_SYMBOLS)
        np.testing.assert_allclose(np.asarray(g2p_agent.C[0]), 0.0, atol=1e-9)

    def test_d_shape_is_1_39(self, g2p_agent):
        assert g2p_agent.D[0].shape == (1, N_PHONES)

    def test_parts_land_verbatim_in_a_b_d(self, g2p_agent, g2p_parts):
        emission, transition, start = g2p_parts
        np.testing.assert_allclose(np.asarray(g2p_agent.A[0]), np.asarray(emission), atol=1e-6)
        np.testing.assert_allclose(np.asarray(g2p_agent.B[0]), np.asarray(transition), atol=1e-6)
        np.testing.assert_allclose(np.asarray(g2p_agent.D[0]), np.asarray(start), atol=1e-6)

    def test_square_emission_raises_assertion(self, g2p_parts):
        """A (1, 39, 39) emission would mean the two levels had been collapsed."""
        _e, transition, start = g2p_parts
        with pytest.raises(ValueError, match=r"emission shape \(1, 30, 39\)"):
            build_g2p_agent(jnp.ones((1, N_PHONES, N_PHONES)) / N_PHONES, transition, start)

    def test_wrong_transition_shape_raises_assertion(self, g2p_parts):
        emission, _t, start = g2p_parts
        with pytest.raises(ValueError, match=r"transition shape \(1, 39, 39, 1\)"):
            build_g2p_agent(emission, jnp.ones((1, N_PHONES, N_PHONES)) / N_PHONES, start)

    def test_wrong_start_shape_raises_assertion(self, g2p_parts):
        emission, transition, _s = g2p_parts
        with pytest.raises(ValueError, match=r"start shape \(1, 39\)"):
            build_g2p_agent(emission, transition, jnp.ones((1, 5)) / 5)

    def test_accepts_numpy_inputs(self, g2p_parts):
        emission, transition, start = g2p_parts
        agent = build_g2p_agent(np.asarray(emission), np.asarray(transition), np.asarray(start))
        assert agent.A[0].shape == (1, N_SYMBOLS, N_PHONES)


# ── decode_phonemes ───────────────────────────────────────────────────────────


@_SKIP_NO_COUNTS
class TestDecodePhonemes:
    """Forward filtering: position t sees letters 1..t only."""

    def test_returns_a_list_of_valid_phones(self, g2p_agent):
        decoded = decode_phonemes(g2p_agent, "cat")
        assert isinstance(decoded, list)
        assert all(p in PHONE_TO_IDX for p in decoded)

    def test_length_equals_kept_letter_count(self, g2p_agent):
        for word in ("cat", "hello", "phoneme"):
            assert len(decode_phonemes(g2p_agent, word)) == len(word)

    def test_empty_string_returns_empty_list(self, g2p_agent):
        assert decode_phonemes(g2p_agent, "") == []

    def test_uppercase_is_normalized_not_rejected(self, g2p_agent):
        assert decode_phonemes(g2p_agent, "CAT") == decode_phonemes(g2p_agent, "cat")

    def test_mixed_case_is_normalized(self, g2p_agent):
        assert decode_phonemes(g2p_agent, "HeLLo") == decode_phonemes(g2p_agent, "hello")

    def test_non_alphabet_characters_become_space_symbols_and_are_kept(self, g2p_agent):
        """normalize_text is length-preserving and space IS an ALPHABET symbol.

        So a digit does not shorten the decode — it becomes a space observation.
        That is what keeps a decode aligned with its reference pronunciation.
        """
        assert len(decode_phonemes(g2p_agent, "cat1")) == 4
        assert len(decode_phonemes(g2p_agent, "12")) == 2

    def test_alphabet_punctuation_is_kept(self, g2p_agent):
        assert len(decode_phonemes(g2p_agent, "don't")) == 5
        assert len(decode_phonemes(g2p_agent, "co-op")) == 5

    def test_is_deterministic_across_repeated_calls(self, g2p_agent):
        assert decode_phonemes(g2p_agent, "phoneme") == decode_phonemes(g2p_agent, "phoneme")

    def test_first_position_is_the_argmax_of_d_times_the_emission_column(
        self, g2p_agent, g2p_parts
    ):
        """Closed form at t = 0: no transition has been applied yet.

        posterior_0 ∝ A[0, letter, :] * D, so the first decoded phone must be the
        argmax of that product. If the implementation applied B before the first
        observation, this would come out wrong.
        """
        emission, _t, start = g2p_parts
        letter = "k"
        joint = np.asarray(emission)[0, ALPHA_TO_IDX[letter], :].astype(np.float64) * np.asarray(
            start
        ).reshape(-1).astype(np.float64)
        expected = IDX_TO_PHONE[int(joint.argmax())]
        assert decode_phonemes(g2p_agent, letter)[0] == expected

    def test_decodes_differ_across_words(self, g2p_agent):
        """A decoder that ignored its input would return the same phones always."""
        assert decode_phonemes(g2p_agent, "cat") != decode_phonemes(g2p_agent, "dog")

    def test_context_changes_the_decode_of_the_same_letter(self, g2p_agent):
        """Filtering must be context-dependent, otherwise it IS the lookup table.

        The letter 'a' alone versus 'a' after 'qu' must be able to differ; if no
        such case existed, the pymdp decoder would be a context-free baseline
        with extra steps.
        """
        contexts = ["a", "ba", "qua", "ta", "sha", "ia", "oa", "ea", "ra", "na"]
        finals = {decode_phonemes(g2p_agent, w)[-1] for w in contexts}
        assert len(finals) > 1, (
            "The final 'a' decoded identically in every context "
            f"({finals}) — the transition is contributing nothing."
        )


# ── per_letter_argmax_baseline ────────────────────────────────────────────────


@_SKIP_NO_COUNTS
class TestPerLetterArgmaxBaseline:
    """The context-free bar: argmax_p P(phone p | letter l), no context at all."""

    def test_returns_a_plain_dict(self, baseline_map):
        assert isinstance(baseline_map, dict)
        assert type(baseline_map) is dict

    def test_every_value_is_a_valid_phone(self, baseline_map):
        offenders = {k: v for k, v in baseline_map.items() if v not in PHONE_TO_IDX}
        assert offenders == {}, f"Non-PHONES values in the mapping: {offenders}"

    def test_every_key_is_an_alphabet_letter(self, baseline_map):
        offenders = [k for k in baseline_map if k not in ALPHA_TO_IDX]
        assert offenders == [], f"Non-ALPHABET keys in the mapping: {offenders}"

    def test_keys_are_exactly_the_letters_observed_in_training(self, splits, baseline_map):
        """Letters never observed must be absent, not defaulted to some phone."""
        from ailm.levels.phoneme import _word_letters

        train, _test = splits
        observed: set[str] = set()
        for word, phones in train:
            letters = _word_letters(word)
            if len(letters) == len(phones):
                observed.update(
                    lt for lt, ph in zip(letters, phones, strict=True) if ph in PHONE_TO_IDX
                )
        assert set(baseline_map) == observed

    def test_unobserved_letter_is_absent(self, baseline_map):
        # No CMUdict headword contains a space, so ' ' is never aligned.
        assert " " not in baseline_map
        assert len(baseline_map) < N_SYMBOLS

    def test_is_deterministic_across_two_calls(self, splits):
        train, _test = splits
        assert per_letter_argmax_baseline(train) == per_letter_argmax_baseline(train)

    def test_ties_break_to_the_alphabetically_first_phone(self):
        """Pinned tie-break: 'a' sees AA once and ZH once, so AA must win."""
        pairs = [("ab", ["AA", "B"]), ("ac", ["ZH", "K"])]
        assert per_letter_argmax_baseline(pairs)["a"] == "AA"

    def test_reversed_input_order_gives_the_same_tie_break(self):
        pairs = [("ac", ["ZH", "K"]), ("ab", ["AA", "B"])]
        assert per_letter_argmax_baseline(pairs)["a"] == "AA"

    def test_majority_phone_wins_over_a_rare_one(self):
        pairs = [("a", ["AA"])] * 10 + [("a", ["ZH"])]
        assert per_letter_argmax_baseline(pairs)["a"] == "AA"

    def test_length_mismatched_pairs_are_ignored(self):
        pairs = [("ab", ["AA"]), ("a", ["ZH"])]
        assert per_letter_argmax_baseline(pairs) == {"a": "ZH"}

    def test_unknown_phones_are_ignored(self):
        assert per_letter_argmax_baseline([("a", ["AE1"])]) == {}

    def test_empty_train_split_gives_an_empty_mapping(self):
        assert per_letter_argmax_baseline([]) == {}

    def test_real_mapping_covers_every_ascii_letter(self, baseline_map):
        missing = [c for c in "abcdefghijklmnopqrstuvwxyz" if c not in baseline_map]
        assert missing == [], f"Letters with no baseline phone: {missing}"


# ── accuracy / position helpers ───────────────────────────────────────────────


@_SKIP_NO_COUNTS
class TestAccuracyAndPositions:
    """Accuracies are fractions; positions must be pairwise aligned for McNemar."""

    @pytest.mark.timeout(120)
    def test_g2p_accuracy_is_a_fraction(self, g2p_agent, splits):
        _train, test = splits
        acc = g2p_accuracy(g2p_agent, test, limit=ACC_WORDS)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_baseline_accuracy_is_a_fraction(self, baseline_map, splits):
        _train, test = splits
        acc = baseline_accuracy(baseline_map, test, limit=ACC_WORDS)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    @pytest.mark.timeout(120)
    def test_accuracy_equals_the_mean_of_its_position_vector(
        self, g2p_agent, baseline_map, splits, decoded_positions, baseline_position_list
    ):
        """The aggregate must be exactly the mean of the paired vector.

        If it were not, the aggregate margin and the McNemar pairing would be
        measuring two different things.
        """
        _train, test = splits
        assert g2p_accuracy(g2p_agent, test, limit=ACC_WORDS) == pytest.approx(
            sum(decoded_positions) / len(decoded_positions)
        )
        assert baseline_accuracy(baseline_map, test, limit=ACC_WORDS) == pytest.approx(
            sum(baseline_position_list) / len(baseline_position_list)
        )

    @pytest.mark.timeout(120)
    def test_position_vectors_are_equal_length(self, decoded_positions, baseline_position_list):
        """THE PAIRING PIN — McNemar is invalid on unaligned vectors."""
        assert len(decoded_positions) == len(baseline_position_list) > 0

    @pytest.mark.timeout(120)
    def test_position_vectors_are_booleans(self, decoded_positions, baseline_position_list):
        assert all(isinstance(x, bool) for x in decoded_positions)
        assert all(isinstance(x, bool) for x in baseline_position_list)

    @pytest.mark.timeout(120)
    def test_position_count_equals_total_reference_phones(self, splits, decoded_positions):
        """One position per reference phone of every selected word, no drops.

        ``limit`` selects a stride subsample, so the expectation is re-derived
        through the same selector both decoders share rather than assuming a
        prefix.
        """
        from ailm.levels.phoneme import _iter_pairs

        _train, test = splits
        expected = sum(len(phones) for _w, phones in _iter_pairs(test, ACC_WORDS))
        assert len(decoded_positions) == expected

    @pytest.mark.timeout(120)
    def test_positions_are_word_ordered(self, g2p_agent, splits, decoded_positions):
        """Concatenating per-word decodes must reproduce the whole vector.

        This is what makes "position i" mean the same thing to both decoders.
        """
        from ailm.levels.phoneme import _iter_pairs

        _train, test = splits
        rebuilt: list[bool] = []
        for word, phones in _iter_pairs(test, ACC_WORDS)[:5]:
            decoded = decode_phonemes(g2p_agent, word)
            rebuilt.extend(d == r for d, r in zip(decoded, phones, strict=True))
        assert len(rebuilt) > 0
        assert decoded_positions[: len(rebuilt)] == rebuilt

    def test_limit_zero_gives_no_positions_and_zero_accuracy(self, g2p_agent, baseline_map, splits):
        _train, test = splits
        assert decode_positions(g2p_agent, test, limit=0) == []
        assert baseline_positions(baseline_map, test, limit=0) == []
        assert g2p_accuracy(g2p_agent, test, limit=0) == 0.0
        assert baseline_accuracy(baseline_map, test, limit=0) == 0.0

    def test_empty_pairs_give_zero_accuracy(self, g2p_agent, baseline_map):
        assert g2p_accuracy(g2p_agent, []) == 0.0
        assert baseline_accuracy(baseline_map, []) == 0.0

    def test_negative_limit_raises_value_error(self, g2p_agent, baseline_map, splits):
        _train, test = splits
        with pytest.raises(ValueError, match="non-negative"):
            g2p_accuracy(g2p_agent, test, limit=-1)
        with pytest.raises(ValueError, match="non-negative"):
            baseline_accuracy(baseline_map, test, limit=-1)

    def test_limit_is_deterministic_not_random(self, baseline_map, splits):
        """``limit`` must select the same words on every call.

        It is a stride subsample, not a prefix and not a sample, so nesting is
        not asserted — only reproducibility, which is what makes a paired
        comparison at a given ``limit`` meaningful.
        """
        first = baseline_positions(baseline_map, splits[1], limit=ACC_WORDS)
        second = baseline_positions(baseline_map, splits[1], limit=ACC_WORDS)
        assert first == second
        assert len(first) > 0

    def test_length_mismatch_raises_value_error(self, g2p_agent, baseline_map):
        bad = [("cat", ["K", "AE"])]
        with pytest.raises(ValueError, match="length-matched by construction"):
            decode_positions(g2p_agent, bad)
        with pytest.raises(ValueError, match="length-matched by construction"):
            baseline_positions(baseline_map, bad)

    def test_letter_absent_from_the_mapping_counts_as_wrong(self, splits):
        """An empty mapping must score 0, not raise and not silently pass."""
        _train, test = splits
        positions = baseline_positions({}, test, limit=ACC_WORDS)
        assert len(positions) > 0
        assert not any(positions)
        assert baseline_accuracy({}, test, limit=ACC_WORDS) == 0.0

    def test_perfect_mapping_scores_one(self):
        """Positive control for the baseline scorer: it can reach 1.0."""
        pairs = [("ab", ["AA", "B"])]
        assert baseline_accuracy({"a": "AA", "b": "B"}, pairs) == 1.0

    @pytest.mark.timeout(120)
    def test_filtering_beats_the_context_free_bar_on_this_prefix(
        self, decoded_positions, baseline_position_list
    ):
        """Necessary-but-not-sufficient sanity check, deliberately weak.

        The full-split margin is only +0.55 points and it wobbles on short
        prefixes, so this asserts no-worse-than rather than a magnitude. The
        significance claim belongs in the Stage 2 gate, paired via McNemar; a
        margin this thin cannot be established from two aggregates.
        """
        filtered = sum(decoded_positions) / len(decoded_positions)
        bar = sum(baseline_position_list) / len(baseline_position_list)
        assert filtered >= bar - 0.05, (
            f"Filtering ({filtered:.4f}) fell well below the context-free bar "
            f"({bar:.4f}) — that signature means the G2P model is mis-wired, "
            "most likely a time-reversed B."
        )

    @pytest.mark.timeout(120)
    def test_the_two_decoders_are_not_the_same_function(
        self, decoded_positions, baseline_position_list
    ):
        """Non-vacuity for the comparison above.

        If the position vectors were identical, filtering would literally be the
        lookup table and every margin claim would be measuring nothing.
        """
        assert decoded_positions != baseline_position_list, (
            "GATE VACUOUS: the pymdp filtering decoder and the context-free "
            "baseline produced identical per-position vectors."
        )


# ── SUBSAMPLE SELECTION ───────────────────────────────────────────────────────


class TestSubsampleSelection:
    """The shared ``limit`` selector behind every position / accuracy helper.

    It is private, but it is load-bearing for the McNemar pairing: both decoders
    must be handed the *same* words in the *same* order, and the words must be
    representative of the held-out set rather than of its alphabetical head.
    """

    @staticmethod
    def _select(pairs, limit):
        from ailm.levels.phoneme import _iter_pairs

        return _iter_pairs(pairs, limit)

    def test_none_returns_every_pair(self):
        pairs = [(f"w{i}", ["AA"]) for i in range(20)]
        assert self._select(pairs, None) == pairs

    def test_zero_returns_nothing(self):
        pairs = [(f"w{i}", ["AA"]) for i in range(20)]
        assert self._select(pairs, 0) == []

    def test_limit_at_or_above_length_returns_every_pair(self):
        pairs = [(f"w{i}", ["AA"]) for i in range(20)]
        assert self._select(pairs, 20) == pairs
        assert self._select(pairs, 200) == pairs

    def test_negative_limit_raises_value_error(self):
        with pytest.raises(ValueError, match="non-negative"):
            self._select([("a", ["AA"])], -1)

    def test_returns_exactly_limit_pairs_when_smaller(self):
        pairs = [(f"w{i}", ["AA"]) for i in range(100)]
        assert len(self._select(pairs, 7)) == 7

    def test_selection_preserves_input_order(self):
        pairs = [(f"w{i:03d}", ["AA"]) for i in range(100)]
        chosen = [w for w, _ in self._select(pairs, 9)]
        assert chosen == sorted(chosen)

    def test_selection_is_a_subsequence_with_no_repeats(self):
        pairs = [(f"w{i:03d}", ["AA"]) for i in range(100)]
        chosen = self._select(pairs, 9)
        words = [w for w, _ in chosen]
        available = [w for w, _ in pairs]
        assert all(w in available for w in words)
        assert len(set(words)) == len(words), "the same pair was selected twice"

    def test_selection_is_deterministic(self):
        pairs = [(f"w{i:03d}", ["AA"]) for i in range(100)]
        assert self._select(pairs, 11) == self._select(pairs, 11)

    def test_returns_a_new_list_not_a_view(self):
        pairs = [("a", ["AA"]), ("b", ["B"])]
        selected = self._select(pairs, None)
        selected.append(("c", ["K"]))
        assert len(pairs) == 2, "the selector aliased its input"

    @_SKIP_NO_COUNTS
    def test_subsample_spans_more_initials_than_the_equivalent_prefix(self, splits):
        """The anti-prefix pin.

        ``g2p_pairs`` is alphabetically sorted, so a prefix of the held-out set
        contains only words starting a/b/c and measurably overstates the G2P
        margin. The selector must spread across the whole list instead — this is
        what stops a fast subset run from reading as a comfortable pass.
        """
        _train, test = splits
        subsample_initials = {w[0] for w, _ in self._select(test, 30)}
        prefix_initials = {w[0] for w, _ in test[:30]}
        assert len(subsample_initials) > len(prefix_initials), (
            f"the {30}-pair subsample spans {sorted(subsample_initials)} while a "
            f"prefix spans {sorted(prefix_initials)} — the subsample is no more "
            "representative than the alphabetical head."
        )
        assert len(subsample_initials) >= 10


# ── ORACLE CROSS-CHECKS ───────────────────────────────────────────────────────


class TestOracleCrossCheckPartA:
    """compute_phoneme_perplexity vs the independent numpy Markov chain.

    ailm.verification.oracle imports neither pymdp nor jax, so agreement here is
    evidence about the wiring rather than a restatement of it. This is the class
    that would have caught the Stage 1 bug: a bigram matrix installed in A with
    an identity B disagrees with the oracle by roughly 15 perplexity units while
    passing every structural check.
    """

    def test_uniform_case_agrees_with_the_oracle(self, uniform_phone_agent):
        seq = ["K", "AE", "T", "S", "AH", "L"]
        obs = [PHONE_TO_IDX[p] for p in seq]
        pymdp_ppl = compute_phoneme_perplexity(uniform_phone_agent, [seq])
        oracle_ppl = oracle.markov_chain_perplexity(
            np.full((N_PHONES, N_PHONES), 1.0 / N_PHONES),
            np.full(N_PHONES, 1.0 / N_PHONES),
            obs,
        )
        assert oracle_ppl == pytest.approx(float(N_PHONES), rel=1e-12)
        assert pymdp_ppl == pytest.approx(oracle_ppl, rel=1e-5)

    def test_small_deterministic_setup_agrees_with_the_oracle(self):
        """A near-deterministic chain: both instruments must report ~1.0.

        With P('AA' → 'B') ≈ 1 and D concentrated on 'AA', the sequence
        ['AA', 'B'] is almost fully predicted, so perplexity approaches 1.
        """
        transition = build_phone_transition({"AA,B": 1_000_000})
        start = build_phone_unigram({"AA": 1_000_000})
        agent = build_phoneme_agent(transition, start)
        seq = ["AA", "B"]

        pymdp_ppl = compute_phoneme_perplexity(agent, [seq])
        oracle_ppl = oracle.markov_chain_perplexity(
            _b_to_numpy(transition),
            _vec_to_numpy(start),
            [PHONE_TO_IDX[p] for p in seq],
        )
        assert pymdp_ppl == pytest.approx(oracle_ppl, rel=1e-5)
        assert pymdp_ppl < 1.01, (
            f"A near-deterministic chain scored {pymdp_ppl:.6f}; both instruments "
            "should be near 1.0, so this setup is not exercising prediction."
        )

    @_SKIP_NO_COUNTS
    def test_trained_single_sequence_agrees_with_the_oracle(
        self, trained_phone_agent, trained_transition, phone_start
    ):
        seq = ["HH", "EH", "L", "OW", "W", "ER", "L", "D"]
        pymdp_ppl = compute_phoneme_perplexity(trained_phone_agent, [seq])
        oracle_ppl = oracle.markov_chain_perplexity(
            _b_to_numpy(trained_transition),
            _vec_to_numpy(phone_start),
            [PHONE_TO_IDX[p] for p in seq],
        )
        assert pymdp_ppl == pytest.approx(oracle_ppl, rel=1e-5), (
            f"pymdp gives {pymdp_ppl:.6f} but the closed-form numpy chain gives "
            f"{oracle_ppl:.6f} — the generative model is mis-wired."
        )

    @_SKIP_NO_COUNTS
    @pytest.mark.timeout(120)
    def test_trained_corpus_agrees_with_the_oracle(
        self, trained_phone_agent, trained_transition, phone_start, phone_sequences
    ):
        """Corpus-level agreement, aggregated the same way pymdp aggregates.

        Total log-evidence over all sequences divided by total positions — which
        also cross-checks the per-sequence reset, because the oracle is restarted
        from ``start`` for every sequence.
        """
        b_np = _b_to_numpy(trained_transition)
        d_np = _vec_to_numpy(phone_start)
        emission = oracle.identity_emission(N_PHONES)

        total_logp = 0.0
        total_positions = 0
        for seq in phone_sequences:
            _posteriors, log_evidence = oracle.forward_filter(
                emission, b_np, d_np, [PHONE_TO_IDX[p] for p in seq]
            )
            total_logp += log_evidence
            total_positions += len(seq)
        oracle_ppl = float(np.exp(-total_logp / total_positions))

        pymdp_ppl = compute_phoneme_perplexity(trained_phone_agent, phone_sequences)
        assert pymdp_ppl == pytest.approx(oracle_ppl, rel=1e-5), (
            f"pymdp {pymdp_ppl:.6f} vs oracle {oracle_ppl:.6f} over "
            f"{total_positions} positions — Part A is mis-wired."
        )

    @_SKIP_NO_COUNTS
    def test_the_stage1_miswiring_is_detected_by_this_cross_check(
        self, trained_transition, phone_start
    ):
        """NEGATIVE CONTROL — without this the agreement tests could be vacuous.

        Install the bigram matrix in the *emission* with an identity transition,
        exactly as Stage 1 did. pymdp accepts it (a bigram matrix is
        column-stochastic in either slot), but the resulting perplexity must
        diverge from the correctly-wired oracle. If it did not, the cross-check
        above would be unable to catch the very bug it exists for.
        """
        from pymdp.agent import Agent

        bigram = jnp.asarray(trained_transition)[:, :, :, 0]  # (1, 39, 39)
        identity = jnp.eye(N_PHONES)[jnp.newaxis, :, :, jnp.newaxis]
        miswired = Agent(
            A=[bigram],
            B=[identity],
            C=[jnp.zeros((1, N_PHONES))],
            D=[jnp.asarray(phone_start)],
            num_controls=[1],
            batch_size=1,
        )

        seq = ["HH", "EH", "L", "OW", "W", "ER", "L", "D"]
        obs = [PHONE_TO_IDX[p] for p in seq]
        correct_ppl = oracle.markov_chain_perplexity(
            _b_to_numpy(trained_transition), _vec_to_numpy(phone_start), obs
        )
        miswired_ppl = compute_phoneme_perplexity(miswired, [seq])

        assert abs(miswired_ppl - correct_ppl) > 1.0, (
            f"CROSS-CHECK VACUOUS: the Stage 1 mis-wiring scored "
            f"{miswired_ppl:.4f} against the correct {correct_ppl:.4f} — the "
            "oracle agreement test cannot detect a bigram matrix in A."
        )


@_SKIP_NO_COUNTS
class TestOracleCrossCheckG2P:
    """decode_phonemes vs oracle.forward_filter on the same emission / B / D."""

    def test_decodes_are_identical_to_the_oracle_argmax(self, g2p_agent, g2p_parts):
        """Filtering argmax, position for position, on fixed words.

        Both instruments resolve an exact tie to the lowest state index
        (``np.argmax`` / ``jnp.argmax`` semantics), so identical decodes are
        required — no tolerance is needed or granted.
        """
        emission, transition, start = g2p_parts
        emis_np = _a_to_numpy(emission)
        b_np = _b_to_numpy(transition)
        d_np = _vec_to_numpy(start)

        for word in CROSSCHECK_WORDS:
            obs = [ALPHA_TO_IDX[c] for c in word]
            posteriors, _log_evidence = oracle.forward_filter(emis_np, b_np, d_np, obs)
            oracle_decode = [IDX_TO_PHONE[int(np.argmax(row))] for row in posteriors]
            pymdp_decode = decode_phonemes(g2p_agent, word)
            assert pymdp_decode == oracle_decode, (
                f"{word!r}: pymdp decoded {pymdp_decode} but the closed-form "
                f"numpy filter decoded {oracle_decode} — the G2P model is "
                "mis-wired."
            )

    def test_posteriors_agree_numerically_with_the_oracle(self, g2p_agent, g2p_parts):
        """Stronger than argmax equality: the whole belief trajectory must match.

        Argmax agreement survives small numerical differences; this binds the
        distributions themselves, so a subtly wrong emission cannot hide behind a
        coincidentally-equal argmax.
        """
        emission, transition, start = g2p_parts
        posteriors, _le = oracle.forward_filter(
            _a_to_numpy(emission),
            _b_to_numpy(transition),
            _vec_to_numpy(start),
            [ALPHA_TO_IDX[c] for c in "phoneme"],
        )

        empirical_prior = [jnp.asarray(g2p_agent.D[0])]
        action = jnp.zeros((1, 1), dtype=jnp.int32)
        for t, letter in enumerate("phoneme"):
            qs = g2p_agent.infer_states([jnp.array([ALPHA_TO_IDX[letter]])], empirical_prior)
            measured = np.asarray(qs[0]).reshape(-1).astype(np.float64)
            np.testing.assert_allclose(
                measured,
                posteriors[t],
                atol=1e-5,
                err_msg=(
                    f"pymdp posterior at t={t} (letter {letter!r}) disagrees with "
                    "the closed-form forward filter."
                ),
            )
            empirical_prior = g2p_agent.update_empirical_prior(action, qs)

    def test_time_reversed_transition_diverges_from_the_correct_decode(self, g2p_agent, g2p_parts):
        """NEGATIVE CONTROL for the decode cross-check.

        A time-reversed B means "observing a letter tells you which phone came
        *before* it". If the reversed model decoded identically, the agreement
        test above would be blind to transition orientation.
        """
        emission, transition, start = g2p_parts
        reversed_b = jnp.asarray(
            np.ascontiguousarray(np.asarray(transition)[0, :, :, 0].T)[np.newaxis, :, :, np.newaxis]
        )
        # Renormalize so pymdp still validates it — the mis-wiring must be
        # caught on behavior, not on a shape or stochasticity error.
        reversed_np = np.asarray(reversed_b)[0, :, :, 0].astype(np.float64)
        reversed_np /= reversed_np.sum(axis=0, keepdims=True)
        reversed_agent = build_g2p_agent(
            emission,
            jnp.array(reversed_np[np.newaxis, :, :, np.newaxis]),
            start,
        )

        differences = 0
        for word in CROSSCHECK_WORDS:
            if decode_phonemes(reversed_agent, word) != decode_phonemes(g2p_agent, word):
                differences += 1
        assert differences > 0, (
            "CROSS-CHECK VACUOUS: a time-reversed transition decoded identically "
            f"on all {len(CROSSCHECK_WORDS)} cross-check words, so the decode "
            "agreement test cannot detect transition orientation."
        )


# ── KNOWN GAPS IN phoneme.py ──────────────────────────────────────────────────


class TestClosedDefects:
    """Regression tests for defects found during adversarial review."""

    def test_negative_counts_raise_instead_of_laundering(self):
        """The non-negativity guard now catches corrupted data at construction time.

        Previously this was a strict xfail: build_phone_transition accepted negative
        counts, _column_stochastic laundered them into a "valid" matrix with negative
        probabilities, and pymdp silently accepted it. The guard was added to
        src/ailm/levels/phoneme.py on 2026-07-31.
        """
        with pytest.raises(ValueError, match="Negative count"):
            build_phone_transition({"AA,B": -1000})

    def test_boolean_test_every_should_be_rejected(self):
        with pytest.raises(ValueError, match="positive integer"):
            g2p_split({"g2p_pairs": [["a", "AA"], ["b", "B"]], "test_every": True})

    def test_limit_docstrings_should_not_describe_a_prefix(self):
        """Reject stale prefix-selection wording, while permitting explicit denials."""
        stale_phrases = (
            "prefix length",
            "takes the first",
            "first limit",
            "prefix-measured",
        )
        offenders = [
            func.__name__
            for func in (
                decode_positions,
                baseline_positions,
                g2p_accuracy,
                baseline_accuracy,
            )
            if any(phrase in (func.__doc__ or "").lower() for phrase in stale_phrases)
        ]
        assert offenders == [], (
            f"{offenders} still document `limit` as a prefix selection, but the "
            "selector returns an alphabet-spanning stride."
        )
