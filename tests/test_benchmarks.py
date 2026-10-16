from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pytest

from ailm.benchmarks import (
    ALPHABETIC_SYMBOLS,
    CALIBRATION_MASS_FLOOR,
    DEFAULT_TASK_N,
    G2P_BUCKETS,
    HELD_OUT_TEXT,
    HELD_OUT_TEXT_LARGE,
    LETTER_INVENTORY_PHRASE,
    LETTER_INVENTORY_SYMBOLS,
    LETTER_WINDOW_LONG,
    LETTER_WINDOW_SHORT,
    _ailm_g2p_decode,
    _ailm_g2p_metrics,
    _answer_token_index,
    _candidate_symbol,
    _distribution_from_candidates,
    _letter_comparator,
    ask_ollama,
    ask_ollama_distribution,
    benchmark_model,
    brier_score,
    build_prompt,
    build_undelimited_letter_prompt,
    captured_mass,
    exact_mcnemar_p,
    expected_calibration_error,
    g2p_length_bucket,
    letter_passage_period,
    letter_positions,
    letter_posterior,
    levenshtein_align,
    negative_log_likelihood,
    ollama_available,
    paired_difference_interval,
    paired_power,
    parse_g2p_answer,
    parse_letter_answer,
    parse_phoneme_answer,
    phone_statistics,
    phoneme_positions,
    restrict_to_alphabetic,
    restrict_to_non_alphabetic,
    subsample_positions,
    unique_letter_positions,
    wilson_interval,
)
from ailm.benchmarks.constants import repeat_trial_protocols
from ailm.levels.letter import ALPHA_LIST, ALPHA_TO_IDX, build_letter_transition, load_brown_counts
from ailm.levels.phoneme import (
    build_g2p_emission,
    build_g2p_start,
    build_g2p_transition,
    g2p_split,
    load_phoneme_counts,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RECORDER = PROJECT_ROOT / "scripts" / "record_external_benchmark_claims.py"
LIVE_OLLAMA_TEST_ENV = "AILM_RUN_LIVE_OLLAMA_TESTS"
_RUN_LIVE_OLLAMA_TESTS = os.environ.get(LIVE_OLLAMA_TEST_ENV) == "1"
_NEEDS_LIVE_OLLAMA_OPT_IN = pytest.mark.skipif(
    not _RUN_LIVE_OLLAMA_TESTS,
    reason=f"live Ollama tests require explicit {LIVE_OLLAMA_TEST_ENV}=1 opt-in",
)


def _require_live_ollama(model: str) -> None:
    """Check availability only inside an explicitly opted-in test body."""
    if not ollama_available(model):
        pytest.skip(f"opted in, but local Ollama model {model!r} is unavailable")


def _load_recorder():
    """Import the external-benchmark recorder as a module object.

    The recorder is a script, not a package member, so it cannot be imported by
    name. Loading it by path is what lets a test assert against the registry and
    the family arithmetic the recorder actually ships. A second copy of either,
    written down in this file, would be free to keep agreeing with a value the
    recorder had already moved past.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_bench_recorder", RECORDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_large_letter_passage_is_nonrepeating_and_has_power_scale() -> None:
    """The external letter arm must have a genuinely larger distinct pool."""
    period = letter_passage_period(HELD_OUT_TEXT_LARGE)
    positions, _ = unique_letter_positions(
        HELD_OUT_TEXT_LARGE, window=LETTER_WINDOW_SHORT, long_window=LETTER_WINDOW_LONG
    )
    assert period == len(
        "".join(c for c in HELD_OUT_TEXT_LARGE.lower() if c in LETTER_INVENTORY_SYMBOLS)
    )
    assert len(positions) > 5400
    assert len(positions) > len(unique_letter_positions(HELD_OUT_TEXT)[0]) * 10


def test_prompt_sensitivity_arm_changes_only_delimitation() -> None:
    """The diagnostic prompt preserves inventory/schema wording and changes only boundary presentation."""
    value = "the "
    delimited = build_prompt("letter", value)
    undelimited = build_undelimited_letter_prompt(value)
    assert LETTER_INVENTORY_PHRASE in delimited
    assert LETTER_INVENTORY_PHRASE in undelimited
    assert 'exactly one JSON object, {"answer":"<one character>"}' in undelimited
    assert "Sequence: <the >" in delimited
    assert "Sequence: the " in undelimited
    assert "trailing space inside them is part of it" in delimited
    assert "trailing space inside them is part of it" not in undelimited


def test_subsample_is_stride_based_not_prefix() -> None:
    values = list(range(100))
    result = subsample_positions(values, 10, seed=7)
    assert result == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    assert result != values[:10]
    assert result == subsample_positions(values, 10, seed=7)


def test_subsample_rejects_invalid_requests() -> None:
    with pytest.raises(ValueError):
        subsample_positions([1], 2)
    with pytest.raises(ValueError):
        subsample_positions([1], -1)
    with pytest.raises(ValueError):
        subsample_positions([1], 1, seed=-1)
    assert subsample_positions([1], 0) == []


def test_prompts_are_exact_and_task_checked() -> None:
    assert build_prompt("letter", "abc").endswith("Sequence: <abc>")
    assert '"answer"' in build_prompt("letter", "abc")
    assert build_prompt("phoneme", "K AE").endswith("Sequence: K AE")
    assert build_prompt("g2p", "cat").endswith("Word: cat")
    with pytest.raises(ValueError):
        build_prompt("bad", "x")


def test_answer_parsers_accept_only_inventory() -> None:
    assert parse_letter_answer('{"answer":"A"}') == ("a", False)
    assert parse_letter_answer('{"answer":"ab"}') == (None, True)
    assert parse_letter_answer('{"answer":"!"}') == (None, True)
    assert parse_phoneme_answer('{"answer":"k"}') == ("K", False)
    assert parse_phoneme_answer('{"answer":"K AE"}') == (None, True)
    assert parse_phoneme_answer("the answer is K") == (None, True)
    assert parse_g2p_answer('{"answer":["K","AE","T"]}') == (["K", "AE", "T"], False)
    assert parse_g2p_answer('{"answer":["K","nope"]}') == ([], True)
    assert parse_g2p_answer("") == ([], True)
    assert parse_g2p_answer('```json {"answer":["K"]} ```') == ([], True)


def test_levenshtein_alignment_known_cases() -> None:
    result = levenshtein_align(["K", "AE", "T"], ["K", "AH", "T"])
    assert result["distance"] == 1
    assert result["correct_positions"] == 2
    assert result["position_accuracy"] == pytest.approx(2 / 3)
    assert levenshtein_align([], [])["exact"] is True
    assert levenshtein_align(["A"], [])["distance"] == 1
    assert levenshtein_align([], ["A"])["distance"] == 1


def test_garbage_is_rejected_as_negative_control() -> None:
    parsed, bad = parse_phoneme_answer('{"answer":"the model is uncertain"}')
    assert bad is True and parsed is None


def test_ollama_probe_is_boolean(local_http_server) -> None:
    """The default suite exercises availability only on its own loopback server."""
    local_http_server.enqueue_json("/api/tags", {"models": []})
    assert isinstance(
        ollama_available("definitely-not-installed", endpoint=local_http_server.base_url), bool
    )


@_NEEDS_LIVE_OLLAMA_OPT_IN
@pytest.mark.timeout(600)
def test_reachable_model_tiny_benchmark_is_deterministic() -> None:
    """Greedy decoding must reproduce itself call for call.

    The explicit timeout is the point of this decorator. Two tiny benchmarks are
    roughly two dozen live round trips to a local server, and the suite-wide
    default of 60s is sized for computation rather than for a model that may have
    to be paged in first. Standalone the test takes about 40s warm, so under a
    full-suite run competing for the same machine the default expires on a server
    that is working correctly — a red suite reporting nothing about the code. The
    ceiling is kept finite because a hung server must still fail rather than hang
    the run, and generous because the alternative fix, shrinking the sampled
    counts, would buy runtime by weakening the determinism claim itself.
    """
    _require_live_ollama("gemma3:4b")
    kwargs = {
        "letter_n": 5,
        "phoneme_n": 5,
        "g2p_n": 5,
        "calibration_n": 3,
        "ensemble_n": 2,
        "repeat_n": 2,
    }
    # Two runs, so two repeat-trial run scopes. The determinism the test asserts
    # is the model's, and it must survive the two runs writing their reliability
    # trials to different cache namespaces rather than competing for one.
    first = benchmark_model(
        "gemma3:4b",
        repeat_trial_protocols=repeat_trial_protocols("20260905T000000000010Z"),
        **kwargs,
    )
    second = benchmark_model(
        "gemma3:4b",
        repeat_trial_protocols=repeat_trial_protocols("20260905T000000000011Z"),
        **kwargs,
    )
    assert first == second
    for task in ("letter", "phoneme"):
        assert 0.0 <= first[task]["top1_accuracy"] <= 1.0
    assert 0.0 <= first["g2p"]["position_accuracy"] <= 1.0
    assert 0.0 <= first["g2p"]["exact_match_rate"] <= 1.0


def test_position_builders_are_nonempty_and_not_prefix_only() -> None:
    letters = letter_positions(HELD_OUT_TEXT)
    assert len(letters) > 5
    assert subsample_positions(letters, 5) != letters[:5]


def test_g2p_parser_rejects_non_inventory_digit() -> None:
    assert parse_g2p_answer('{"answer":["K","AE","999"]}') == ([], True)


# ── The letter inventory the prompt promises is the one that is scored ───────


def test_prompt_inventory_matches_the_scored_alphabet() -> None:
    """The sentence in the prompt and the symbol set must not drift apart.

    A prompt naming a narrower inventory than the scorer accepts penalizes the
    probe for obeying it, and space is the most frequent gold symbol. This binds
    the prose to ``ALPHA_LIST``: change one without the other and this fails.
    """
    assert LETTER_INVENTORY_SYMBOLS == set(ALPHA_LIST)
    for fragment in ("a-z", "space", "apostrophe", "hyphen", "period"):
        assert fragment in LETTER_INVENTORY_PHRASE
    assert LETTER_INVENTORY_PHRASE in build_prompt("letter", "abc")


def test_a_lone_space_answer_is_a_valid_symbol_not_an_invalid_one() -> None:
    """POSITIVE CONTROL: a space answer is scored as the symbol it is."""
    assert parse_letter_answer('{"answer":" "}') == (" ", False)
    assert " " in ALPHA_TO_IDX


def test_space_gold_positions_are_a_large_share_of_the_letter_task() -> None:
    """Non-vacuity for the fix: the defect it repairs has to be reachable."""
    positions, _ = unique_letter_positions(HELD_OUT_TEXT)
    spaces = sum(1 for _prefix, gold in positions if gold == " ")
    assert spaces > len(positions) // 10


def test_prompt_delimits_the_sequence_so_a_trailing_space_survives() -> None:
    """A prefix ending in a space must remain visible to the model."""
    prompt = build_prompt("letter", "the cat ")
    assert prompt.endswith("<the cat >")


# ── Position pool: duplicates removed, windows paired ────────────────────────


def test_held_out_text_repeats_and_the_pool_is_deduplicated() -> None:
    """The fixture is one passage three times; repeats are not new observations."""
    period = letter_passage_period(HELD_OUT_TEXT)
    raw = letter_positions(HELD_OUT_TEXT)
    unique, _ = unique_letter_positions(HELD_OUT_TEXT)
    assert period < len(raw)
    assert len(unique) < len(raw)
    assert len(unique) == len(set(raw))
    assert len(set(unique)) == len(unique)


def test_context_windows_are_paired_on_the_same_positions() -> None:
    short, long = unique_letter_positions(
        HELD_OUT_TEXT, window=LETTER_WINDOW_SHORT, long_window=LETTER_WINDOW_LONG
    )
    assert len(short) == len(long)
    assert [gold for _p, gold in short] == [gold for _p, gold in long]
    widened = [
        index
        for index, (s, item) in enumerate(zip(short, long, strict=True))
        if len(item[0]) > len(s[0])
    ]
    assert widened, "the long window must actually carry more context somewhere"
    for index in widened:
        assert long[index][0].endswith(short[index][0])


def test_a_window_at_the_passage_period_is_refused() -> None:
    """NEGATIVE CONTROL: a window that would leak the answer must not be allowed."""
    period = letter_passage_period(HELD_OUT_TEXT)
    with pytest.raises(ValueError, match="passage period"):
        unique_letter_positions(HELD_OUT_TEXT, long_window=period)
    assert LETTER_WINDOW_LONG < period


def test_the_order_one_comparator_is_context_invariant() -> None:
    """The premise of the context sweep, asserted rather than assumed."""
    brown = load_brown_counts(PROJECT_ROOT / "vendor")
    short, long = unique_letter_positions(
        HELD_OUT_TEXT, window=LETTER_WINDOW_SHORT, long_window=LETTER_WINDOW_LONG
    )
    sample = list(range(0, len(short), 7))
    assert _letter_comparator(brown, [short[i] for i in sample]) == _letter_comparator(
        brown, [long[i] for i in sample]
    )


# ── The phoneme comparator is held out from the words it is scored on ────────


def test_phoneme_positions_come_from_the_split_they_are_given() -> None:
    counts = load_phoneme_counts(PROJECT_ROOT / "vendor")
    train_pairs, test_pairs = g2p_split(counts)
    test_words = {word for word, _phones in test_pairs}
    train_words = {word for word, _phones in train_pairs}
    assert not (test_words & train_words)
    assert phoneme_positions(counts, test_pairs)
    assert len(phoneme_positions(counts, test_pairs)) < len(phoneme_positions(counts))


def test_train_only_phone_statistics_exclude_the_test_words() -> None:
    """The vendored counts pool every base word; a held-out split must not.

    Feeding the artifact's global ``phone_bigrams`` to the comparator fits it on
    the words it is then scored on. This checks the recomputed statistics are
    strictly the training split's, using a phone pair that only a test word has.
    """
    counts = load_phoneme_counts(PROJECT_ROOT / "vendor")
    train_pairs, test_pairs = g2p_split(counts)
    train_unigrams, train_bigrams = phone_statistics(train_pairs)
    _test_unigrams, test_bigrams = phone_statistics(test_pairs)
    assert sum(train_unigrams.values()) < sum(counts["phone_unigrams"].values())
    assert sum(train_bigrams.values()) < sum(counts["phone_bigrams"].values())
    for key, count in test_bigrams.items():
        assert train_bigrams.get(key, 0) + count <= counts["phone_bigrams"].get(key, 0) + count


# ── Paired difference interval and retrospective power ───────────────────────


def test_paired_difference_interval_brackets_the_point_estimate() -> None:
    b, c, n = 18, 30, 120
    low, high = paired_difference_interval(b, c, n)
    assert low < (b - c) / n < high
    assert -1.0 <= low <= high <= 1.0


def test_exact_mcnemar_count_api_matches_scipy_and_has_live_controls() -> None:
    from scipy.stats import binomtest

    assert exact_mcnemar_p(18, 30) == pytest.approx(binomtest(18, 48, 0.5).pvalue)
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(48, 0) < exact_mcnemar_p(24, 24)
    with pytest.raises(ValueError, match="non-negative integer"):
        exact_mcnemar_p(-1, 3)


def test_paired_difference_interval_is_symmetric_under_swap() -> None:
    low, high = paired_difference_interval(30, 18, 120)
    mirror_low, mirror_high = paired_difference_interval(18, 30, 120)
    assert low == pytest.approx(-mirror_high)
    assert high == pytest.approx(-mirror_low)


def test_paired_difference_interval_collapses_with_no_discordant_pairs() -> None:
    assert paired_difference_interval(0, 0, 50) == (0.0, 0.0)


@pytest.mark.parametrize(
    ("b", "c", "n"),
    [(-1, 0, 5), (0, -1, 5), (1, 1, 0), (5, 5, 4)],
)
def test_paired_difference_interval_rejects_impossible_counts(b: int, c: int, n: int) -> None:
    with pytest.raises(ValueError):
        paired_difference_interval(b, c, n)


def test_power_rises_with_discordant_pairs_and_is_zero_without_them() -> None:
    assert paired_power(0, 0) == 0.0
    assert paired_power(5, 5) < paired_power(100, 100) < paired_power(2000, 2000)


def test_power_is_evaluated_at_the_reference_not_the_observed_effect() -> None:
    """Two splits with the same pair count must report the same power.

    Power at the observed effect is a monotone transform of the p-value and adds
    no evidence. If this ever fails, the diagnostic has drifted into that trap.
    """
    assert paired_power(48, 0) == paired_power(24, 24)
    assert paired_power(24, 24, reference_share=0.8) > paired_power(24, 24, reference_share=0.55)
    with pytest.raises(ValueError, match="reference_share"):
        paired_power(0, 0, reference_share=1.5)


# ── Calibration arithmetic ───────────────────────────────────────────────────


def test_ece_is_zero_when_confidence_matches_accuracy() -> None:
    assert expected_calibration_error([1.0, 1.0], [True, True]) == pytest.approx(0.0)
    assert expected_calibration_error([0.5] * 4, [True, False, True, False]) == pytest.approx(0.0)


def test_ece_is_one_for_a_confidently_wrong_forecaster() -> None:
    """NEGATIVE CONTROL: the metric must be able to reach its worst value."""
    assert expected_calibration_error([1.0] * 8, [False] * 8) == pytest.approx(1.0)


def test_ece_rejects_mismatched_inputs_and_bad_binning() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [True, False])
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [True], bins=0)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), "0.5", True])
def test_ece_rejects_values_outside_the_probability_domain(confidence: object) -> None:
    """NEGATIVE CONTROL: invalid confidence must not vanish outside the bins."""
    with pytest.raises(ValueError):
        expected_calibration_error([confidence], [False])  # type: ignore[list-item]


@pytest.mark.parametrize("bins", [True, 1.5, "10"])
def test_ece_requires_an_integer_bin_count(bins: object) -> None:
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [True], bins=bins)  # type: ignore[arg-type]


def test_ece_requires_boolean_correctness() -> None:
    with pytest.raises(ValueError, match="boolean"):
        expected_calibration_error([0.5], [1])  # type: ignore[list-item]


def test_brier_spans_its_full_range() -> None:
    assert brier_score([{"a": 1.0}], ["a"]) == pytest.approx(0.0)
    assert brier_score([{"a": 1.0}], ["b"]) == pytest.approx(2.0)
    uniform = {symbol: 1.0 / len(ALPHA_LIST) for symbol in ALPHA_LIST}
    assert 0.0 < brier_score([uniform], ["a"]) < 2.0


def test_proper_scores_accept_sparse_normalized_distributions() -> None:
    assert negative_log_likelihood([{"a": 1.0}], ["a"]) == pytest.approx(0.0)
    assert brier_score([{"a": 1.0}], ["b"]) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "distribution",
    [
        {"a": 2.0},
        {"a": -0.1, "b": 1.1},
        {"a": float("nan")},
        {"a": 0.4, "b": 0.4},
        {"outside-inventory": 1.0},
        {},
    ],
)
def test_proper_scores_reject_invalid_probability_distributions(
    distribution: dict[str, float],
) -> None:
    """NEGATIVE CONTROL: impossible forecasts cannot earn a proper score."""
    with pytest.raises(ValueError):
        negative_log_likelihood([distribution], ["a"])
    with pytest.raises(ValueError):
        brier_score([distribution], ["a"])


def test_proper_scores_reject_unknown_gold_and_invalid_log_floor() -> None:
    with pytest.raises(ValueError, match="outside the declared inventory"):
        negative_log_likelihood([{"a": 1.0}], ["not-a-letter"])
    with pytest.raises(ValueError, match="outside the declared inventory"):
        brier_score([{"a": 1.0}], ["not-a-letter"])
    for floor in (0.0, -1.0, 1.1, float("nan"), "1e-12", True):
        with pytest.raises(ValueError):
            negative_log_likelihood([{"a": 1.0}], ["a"], floor=floor)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("successes", "total", "z"),
    [(-1, 5, 1.96), (1, -5, 1.96), (6, 5, 1.96), (1, 5, 0.0), (1, 5, float("nan"))],
)
def test_wilson_interval_rejects_impossible_inputs(successes: int, total: int, z: float) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, total, z)


def test_wilson_interval_defines_only_the_empty_zero_over_zero_case() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_letter_posterior_is_a_distribution_whose_argmax_is_the_comparator() -> None:
    """The calibration path and the scored path must be the same model.

    ECE for AILM is only meaningful if the distribution it is computed from is
    the one the accuracy number comes from.
    """
    brown = load_brown_counts(PROJECT_ROOT / "vendor")
    positions, _ = unique_letter_positions(HELD_OUT_TEXT)
    sample = [positions[index] for index in range(0, len(positions), 23)]
    predictions = _letter_comparator(brown, sample)
    for (prefix, _gold), predicted in zip(sample, predictions, strict=True):
        posterior = letter_posterior(brown, prefix)
        assert sum(posterior.values()) == pytest.approx(1.0)
        assert set(posterior) == set(ALPHA_LIST)
        assert max(posterior, key=lambda symbol: posterior[symbol]) == predicted


def test_letter_posterior_removes_float32_reduction_drift_before_proper_scoring() -> None:
    """The real ``h`` column reproduces the warm-cache calibration failure.

    Dividing Python-float entries by the backend's float32 reduction leaves the
    row just outside the scorer's unit-mass tolerance. The production helper
    must normalize after conversion; the scorer must continue rejecting the
    historical row so this repair cannot weaken validation instead.
    """
    brown = load_brown_counts(PROJECT_ROOT / "vendor")
    column = np.asarray(build_letter_transition(brown))[0][:, ALPHA_TO_IDX["h"]]
    backend_total = float(column.sum())
    historical = {
        symbol: float(column[index]) / backend_total for index, symbol in enumerate(ALPHA_LIST)
    }
    assert math.fsum(historical.values()) > 1.0 + 1e-7
    with pytest.raises(ValueError, match="must sum to 1"):
        brier_score([historical], ["e"])

    posterior = letter_posterior(brown, "th")
    assert math.fsum(posterior.values()) == pytest.approx(1.0, abs=1e-15)
    assert brier_score([posterior], ["e"]) >= 0.0


def test_answer_token_index_finds_the_slot_under_any_envelope_tokenisation() -> None:
    assert _answer_token_index(['{"', "answer", '":"', "b", '"}']) == 3
    assert _answer_token_index(["{", '"answer"', ":", '"', "b", '"}']) == 4
    assert _answer_token_index(["```", "json", "\n"]) is None
    assert _answer_token_index([]) is None


def test_candidate_distribution_bounds_the_unreported_symbols() -> None:
    """Truncation is completed by an upper bound, and forbidden mass is dropped.

    The candidate list is the raw next-token distribution truncated at
    ``top_logprobs``, so most of its mass is on tokens the grammar forbids and
    several inventory symbols are absent. Redistributing the forbidden mass over
    the absent symbols would let the completion dominate: here it would hand
    them 40% of the distribution on the strength of a token spelling "dog".
    """
    candidates = [
        {"token": "a", "logprob": np.log(0.4)},
        {"token": "dog", "logprob": np.log(0.5)},
        {"token": "b", "logprob": np.log(0.1)},
    ]
    distribution = _distribution_from_candidates(candidates)
    assert set(distribution) == set(ALPHA_LIST)
    assert sum(distribution.values()) == pytest.approx(1.0)
    # Absent symbols each carry at most the smallest reported probability.
    assert distribution["c"] == pytest.approx(distribution["b"])
    assert distribution["c"] <= distribution["a"]
    # "a" keeps four times "b"'s mass: the relative ordering the server reported
    # survives, which redistributing the forbidden mass would not preserve.
    assert distribution["a"] == pytest.approx(4.0 * distribution["b"])


def test_candidate_distribution_falls_back_to_uniform_when_nothing_is_usable() -> None:
    distribution = _distribution_from_candidates([{"token": "zzz", "logprob": -0.1}])
    assert sum(distribution.values()) == pytest.approx(1.0)
    assert len(set(distribution.values())) == 1


# ── G2P stratification and the comparator-decode refactor ────────────────────


def test_g2p_buckets_cover_every_length_without_overlap() -> None:
    assert [g2p_length_bucket(n) for n in (0, 5, 6, 9, 10, 99)] == [
        "short",
        "short",
        "mid",
        "mid",
        "long",
        "long",
    ]
    names = [name for name, _low, _high in G2P_BUCKETS]
    assert len(names) == len(set(names))
    with pytest.raises(ValueError):
        g2p_length_bucket(-1)


def test_comparator_metrics_return_the_decodes_they_were_computed_from() -> None:
    """The deduplicated decode must be the same decode, value for value.

    The comparator's paired flags reuse the decodes ``_ailm_g2p_metrics``
    already produced rather than decoding a second time. This recomputes that
    decode independently, from freshly built matrices, and requires identity —
    so reuse is verified against an independent path rather than assumed.
    """
    counts = load_phoneme_counts(PROJECT_ROOT / "vendor")
    train_pairs, test_pairs = g2p_split(counts)
    words = subsample_positions(test_pairs, 12)
    position, exact, predictions = _ailm_g2p_metrics(train_pairs, words)

    emission = np.asarray(build_g2p_emission(train_pairs))[0]
    transition = np.asarray(build_g2p_transition(train_pairs))[0, :, :, 0]
    start = np.asarray(build_g2p_start(train_pairs)).ravel()
    independent = [_ailm_g2p_decode(emission, transition, start, word) for word, _gold in words]

    assert predictions == independent
    total = sum(len(gold) for _word, gold in words)
    assert position == pytest.approx(
        sum(
            predicted == expected
            for prediction, (_word, gold) in zip(predictions, words, strict=True)
            for predicted, expected in zip(prediction, gold, strict=False)
        )
        / total
    )
    assert exact == pytest.approx(
        sum(
            prediction == gold for prediction, (_word, gold) in zip(predictions, words, strict=True)
        )
        / len(words)
    )


# ── Live-probe checks (skipped without a reachable server) ───────────────────


@_NEEDS_LIVE_OLLAMA_OPT_IN
def test_adding_logprobs_does_not_change_the_reply_that_is_scored() -> None:
    """The calibration request must be the scored request, plus a report."""
    _require_live_ollama("gemma3:4b")
    prompt = build_prompt("letter", "the ca")
    scored = ask_ollama("gemma3:4b", prompt, "letter", 32)
    text, distribution, captured = ask_ollama_distribution("gemma3:4b", prompt, "letter", 32)
    assert text == scored, "adding logprobs changed the reply the benchmark scores"
    assert sum(distribution.values()) == pytest.approx(1.0)
    assert 0.0 < captured <= 1.0 + 1e-9, "no reported mass landed on the scored inventory"


@_NEEDS_LIVE_OLLAMA_OPT_IN
@pytest.mark.timeout(600)
def test_reconstructed_argmax_mostly_but_not_always_matches_the_scored_answer() -> None:
    """Bound how far the reported distribution describes the decode it came from.

    ``top_logprobs`` is the raw pre-mask distribution while generation is
    grammar-constrained, so the constrained decoder can be pushed off its raw
    argmax and onto a token outside the reported top-k. Agreement is therefore
    high but not total, and asserting equality would be asserting something
    untrue of the endpoint — an earlier single-position version of this test
    passed only by luck. The calibration pass scores the parsed answer for this
    reason; this measures the gap rather than pretending it closes.
    """
    _require_live_ollama("gemma3:4b")
    positions, _ = unique_letter_positions(HELD_OUT_TEXT)
    sample = subsample_positions(positions, 25)
    agree = comparable = 0
    for prefix, _gold in sample:
        prompt = build_prompt("letter", prefix)
        text, distribution, _captured = ask_ollama_distribution("gemma3:4b", prompt, "letter", 32)
        answer, bad = parse_letter_answer(text)
        if bad:
            continue
        comparable += 1
        agree += int(max(distribution, key=lambda symbol: distribution[symbol]) == answer)
    assert comparable >= 20, "too few parsable replies to measure agreement"
    rate = agree / comparable
    assert rate >= 0.85, (
        f"reconstructed argmax matched the scored answer on only {agree}/{comparable} "
        f"= {rate:.3f} of positions. Below this the reported distribution no longer "
        f"describes the decode it came from and calibration is not interpretable."
    )


@_NEEDS_LIVE_OLLAMA_OPT_IN
def test_the_same_seed_twice_returns_the_same_string() -> None:
    _require_live_ollama("gemma3:4b")
    prompt = build_prompt("letter", "the quick brown fo")
    first = ask_ollama("gemma3:4b", prompt, "letter", 32, seed=1234)
    second = ask_ollama("gemma3:4b", prompt, "letter", 32, seed=1234)
    assert first == second


def test_the_probe_distribution_is_not_a_restatement_of_its_accuracy() -> None:
    """NEGATIVE CONTROL for the calibration panel being worth reporting.

    If the endpoint returned a one-hot distribution at every position, ECE would
    equal one minus accuracy and the panel would be accuracy under a new name.
    A spread distribution has to separate the two numbers.
    """
    confidences = [0.9, 0.8, 0.3, 0.2]
    correct = [True, False, True, False]
    accuracy = sum(correct) / len(correct)
    assert expected_calibration_error(confidences, correct) != pytest.approx(1.0 - accuracy)


def test_captured_mass_counts_only_in_inventory_candidates() -> None:
    """The candidate list is the raw next-token distribution, not the masked one.

    It carries multi-character tokens the response schema could never emit, so
    the mass reported over the scored inventory is strictly less than one and
    the shortfall is filled in by assumption. Recording it keeps that visible.
    """
    candidates = [
        {"token": "a", "logprob": np.log(0.5)},
        {"token": "dog", "logprob": np.log(0.4)},
        {"token": "b", "logprob": np.log(0.05)},
    ]
    assert captured_mass(candidates) == pytest.approx(0.55)
    assert captured_mass([]) == 0.0


def test_alphabetic_arm_restricts_positions_without_rescoring_them() -> None:
    """Both inventory readings come from one set of calls, not two runs."""
    positions = [("x", "a"), ("y", " "), ("z", "b"), ("w", "-")]
    scored = {
        "model_correct": [True, False, False, False],
        "comparator_correct": [False, True, True, False],
    }
    arm = restrict_to_alphabetic(positions, scored)
    assert arm["n"] == 2
    assert arm["top1_accuracy"] == pytest.approx(0.5)
    assert arm["comparator_top1_accuracy"] == pytest.approx(0.5)
    assert arm["paired"]["b"] == 1
    assert arm["paired"]["c"] == 1


def test_alphabetic_arm_is_a_strict_subset_of_the_full_inventory_arm() -> None:
    """Non-vacuity: the two inventory readings must differ in population."""
    positions, _ = unique_letter_positions(HELD_OUT_TEXT)
    alphabetic = [pair for pair in positions if pair[1] in ALPHABETIC_SYMBOLS]
    assert 0 < len(alphabetic) < len(positions)
    assert ALPHABETIC_SYMBOLS < LETTER_INVENTORY_SYMBOLS


def test_non_alphabetic_arm_is_the_exact_complement_of_the_alphabetic_arm() -> None:
    """The two arms must partition the scored positions, covering each once.

    The supplement explains the disagreement between the inventory arms by the
    non-alphabetic targets, so the split has to be a genuine partition — an arm
    that dropped or double-counted positions would make that explanation
    arithmetic rather than evidence.
    """
    positions = [("x", "a"), ("y", " "), ("z", "b"), ("w", "-")]
    scored = {
        "model_correct": [True, False, False, False],
        "comparator_correct": [False, True, True, False],
    }
    alphabetic = restrict_to_alphabetic(positions, scored)
    non_alphabetic = restrict_to_non_alphabetic(positions, scored)
    assert alphabetic["n"] + non_alphabetic["n"] == len(positions)
    assert non_alphabetic["n"] == 2
    # the probe answers neither non-alphabetic target; the comparator answers one
    assert non_alphabetic["top1_accuracy"] == pytest.approx(0.0)
    assert non_alphabetic["comparator_top1_accuracy"] == pytest.approx(0.5)
    assert non_alphabetic["paired"]["b"] == 0
    assert non_alphabetic["paired"]["c"] == 1


def test_non_alphabetic_arm_is_non_empty_on_the_real_fixture() -> None:
    """Non-vacuity: a partition where one side is empty would explain nothing."""
    positions, _ = unique_letter_positions(HELD_OUT_TEXT)
    non_alphabetic = [pair for pair in positions if pair[1] not in ALPHABETIC_SYMBOLS]
    assert 0 < len(non_alphabetic) < len(positions)
    # almost all of them are the space symbol, which is the mechanism the
    # supplement names
    spaces = [pair for pair in non_alphabetic if pair[1] == " "]
    assert len(spaces) > len(non_alphabetic) / 2


def test_multiclass_brier_cannot_be_gamed_by_being_uniformly_wrong() -> None:
    """NEGATIVE CONTROL against the top-1 Brier trap.

    ``mean((confidence - correct) ** 2)`` over the top symbol alone is not a
    proper scoring rule across systems at different accuracies: a system that is
    always wrong at low confidence beats a system that is often right at
    moderate confidence, so it can rank a model that got nothing right as the
    better-calibrated one. The multiclass form scores the whole distribution and
    cannot invert that way. This pins the definition; if someone reduces
    ``brier_score`` to the top-1 form, this fails.
    """
    inventory = sorted(ALPHA_LIST)
    confident_and_wrong = {symbol: 0.0 for symbol in inventory}
    confident_and_wrong["z"] = 1.0
    spread_and_right = {symbol: 0.02 for symbol in inventory}
    spread_and_right["a"] = 1.0 - 0.02 * (len(inventory) - 1)

    always_wrong = brier_score([confident_and_wrong] * 4, ["a"] * 4)
    often_right = brier_score([spread_and_right] * 4, ["a"] * 4)
    assert often_right < always_wrong

    # The top-1 reading of the same two systems inverts the ranking, which is
    # exactly the failure this test exists to keep out of the recorder.
    top1_always_wrong = sum((1.0 - 0.0) ** 2 for _ in range(4)) / 4
    top1_often_right = sum((spread_and_right["a"] - 1.0) ** 2 for _ in range(4)) / 4
    assert top1_often_right < top1_always_wrong  # here they agree...
    low_confidence_wrong = {symbol: 1.0 / len(inventory) for symbol in inventory}
    assert sum((1.0 / len(inventory) - 0.0) ** 2 for _ in range(4)) / 4 < top1_often_right, (
        "the top-1 form rewards being uniformly unsure and wrong"
    )
    assert brier_score([low_confidence_wrong] * 4, ["a"] * 4) > often_right


def test_a_symbol_merged_with_its_closing_quote_is_still_that_symbol() -> None:
    """The tokeniser may emit the answer and the string terminator as one token.

    Matching candidates against the inventory literally drops exactly the token
    the model emitted, which understates the observed mass and lets the
    reconstructed distribution's argmax disagree with the answer the benchmark
    scored. Calibration would then describe a decision that was never made.
    """
    assert _candidate_symbol('."') == "."
    assert _candidate_symbol("r") == "r"
    assert _candidate_symbol(' "') == " "
    assert _candidate_symbol("rose") is None
    assert _candidate_symbol('"') is None
    assert _candidate_symbol("") is None

    merged = [{"token": '."', "logprob": np.log(0.7)}, {"token": "r", "logprob": np.log(0.2)}]
    assert captured_mass(merged) == pytest.approx(0.9)
    distribution = _distribution_from_candidates(merged)
    assert max(distribution, key=lambda symbol: distribution[symbol]) == "."


def test_two_tokens_spelling_one_symbol_have_their_mass_added() -> None:
    """The bare symbol and the merged form are two paths to the same answer."""
    split = [{"token": "r", "logprob": np.log(0.3)}, {"token": 'r"', "logprob": np.log(0.4)}]
    assert captured_mass(split) == pytest.approx(0.7)
    distribution = _distribution_from_candidates(split)
    assert max(distribution, key=lambda symbol: distribution[symbol]) == "r"


def test_the_two_letter_arms_decompose_the_recorded_totals() -> None:
    """The alphabetic and non-alphabetic arms must add back to the full arm.

    The supplement's central letter caveat rests on the difference between the
    two recorded arms being the accuracy on non-alphabetic targets. That is
    arithmetic on two measurements rather than a third measurement, so it is
    derived in ``manuscript_variables``; this checks the derivation against the
    ledger it is derived from, and that the split is non-degenerate.
    """
    claims = json.loads((PROJECT_ROOT / "data" / "claims.json").read_text())["claims"]
    if "bench_letter_az_positions" not in claims:
        pytest.skip("external benchmark not recorded in this checkout")
    value = lambda label: claims[label]["value"]  # noqa: E731
    total = value("bench_letter_positions")
    alphabetic = value("bench_letter_az_positions")
    other = total - alphabetic
    assert 0 < alphabetic < total, "one arm is empty; the decomposition says nothing"
    for system in ("ailm", "gemma3", "qwen25", "smollm"):
        full_correct = round(value(f"bench_letter_top1_{system}") * total)
        arm_correct = round(value(f"bench_letter_az_top1_{system}") * alphabetic)
        assert 0 <= full_correct - arm_correct <= other, (
            f"{system}: the two arms do not decompose — "
            f"{full_correct} full, {arm_correct} alphabetic, {other} other positions"
        )


def test_the_registered_probes_are_the_three_the_benchmark_ships() -> None:
    """The probe registry, and the multiplicity family its size determines.

    DEFECT: the registry is the only place the benchmark's correction family is
    sized from — the recorder builds one paired p-value per contrast template per
    selected model — so retiring a probe loosens the Bonferroni threshold every
    interpretive claim in the supplement is read against. Nothing about that is
    visible at the point a model is deleted from a dict. This states the
    inventory and re-derives the family from it, so a probe added or removed
    fails here with the arithmetic spelled out rather than quietly moving a
    threshold under prose that still cites the old one.

    The registry is asserted by exact set equality rather than by size, because a
    count is satisfied by any three models: a probe swapped in under a different key
    would pass a size check and silently change what every interpretive claim in
    the supplement is read against.
    """
    recorder = _load_recorder()

    assert set(recorder.MODELS) == {"gemma3", "qwen25", "smollm"}, (
        f"the benchmark ships three registered probes; the registry now holds "
        f"{sorted(recorder.MODELS)}"
    )
    assert recorder.COMPARATOR not in recorder.MODELS, (
        "the comparator is not an Ollama model and must not be measured as one"
    )

    templates = recorder.FAMILY_TEMPLATES
    assert len(set(templates)) == len(templates), "a duplicated template double-counts an arm"
    family = [
        template.format(model=name) for name in sorted(recorder.MODELS) for template in templates
    ]
    assert len(set(family)) == len(family) == len(templates) * len(recorder.MODELS), (
        f"the declared family is one contrast per template per registered model; "
        f"{len(templates)} templates and {len(recorder.MODELS)} models give "
        f"{len(templates) * len(recorder.MODELS)} tests, not {len(set(family))}"
    )
    # The threshold the supplement reads every p-value against, rebuilt here from
    # the same two inputs the recorder writes it from. Asserting a digit would
    # only prove this file and the recorder were edited together once.
    assert recorder.POWER_ALPHA / len(family) < recorder.POWER_ALPHA, (
        "a Bonferroni correction over a non-empty family must tighten the threshold"
    )


def test_the_recorder_clears_only_the_labels_it_owns() -> None:
    """A shared namespace must not be cleared wholesale.

    ``bench_`` is written by more than one recorder: this one owns the external
    reference labels and another owns ``bench_synth_`` for the
    bounded stipulated-language exact-artifact-overlap probe. Removing every
    ``bench_`` label before
    writing would delete the other producer's measurements on every run, with no
    error and no test failure until someone noticed the numbers were gone.
    """
    recorder = _load_recorder()
    mine = RECORDER.name

    class _Ledger:
        entries = {
            "bench_letter_top1_gemma3": {"source": f"/repo/scripts/{mine}"},
            "bench_letter_retired_metric": {"source": f"/repo/scripts/{mine}"},
            "bench_synth_letter_top1_ailm": {"source": "/repo/scripts/record_synthetic.py"},
            "stage1_letter_perplexity": {"source": "/repo/scripts/record_stage1_claims.py"},
        }

        def all_labels(self):
            return list(self.entries)

        def get(self, label):
            return self.entries.get(label)

    produced = {"bench_letter_top1_gemma3": 0.1}
    stale = recorder.stale_labels(_Ledger(), produced)

    assert stale == ["bench_letter_retired_metric"], (
        "must clear this recorder's own superseded labels, and nothing else"
    )
    assert "bench_synth_letter_top1_ailm" not in stale, "another producer's labels are not ours"
    assert "bench_letter_top1_gemma3" not in stale, "a label about to be rewritten is not stale"


def test_a_mostly_assumed_distribution_reports_a_flattering_calibration() -> None:
    """POSITIVE CONTROL for the captured-mass gate: the failure it prevents.

    When the observed candidates carry little mass, the completion dominates and
    pushes the distribution toward its floor. That lowers the stated confidence,
    and on a system that is usually wrong a lowered confidence *reduces* the
    calibration error. Without the gate, a reconstruction failure is reported as
    good calibration — the worst direction for an error to run.
    """
    inventory = sorted(ALPHA_LIST)
    # A confident wrong answer, reported almost in full: honestly bad ECE.
    observed = [{"token": "z", "logprob": np.log(0.9)}, {"token": "a", "logprob": np.log(0.05)}]
    # The same confident wrong answer, with nearly all mass on forbidden tokens.
    starved = [{"token": "z", "logprob": np.log(0.03)}, {"token": "a", "logprob": np.log(0.02)}]

    rich = _distribution_from_candidates(observed)
    poor = _distribution_from_candidates(starved)
    assert captured_mass(observed) > CALIBRATION_MASS_FLOOR
    assert captured_mass(starved) < CALIBRATION_MASS_FLOOR

    rich_ece = expected_calibration_error([rich["z"]] * 10, [False] * 10)
    poor_ece = expected_calibration_error([poor["z"]] * 10, [False] * 10)
    assert poor_ece < rich_ece, (
        "the starved reconstruction must be the one that looks better calibrated; "
        "if it does not, this control no longer describes the hazard the gate blocks"
    )
    assert len(inventory) == len(rich) == len(poor)


def test_the_mass_floor_is_a_declared_constant_not_a_fitted_one() -> None:
    """The threshold is 'mostly measurement', not wherever a model landed."""
    assert CALIBRATION_MASS_FLOOR == 0.5


def test_the_captured_share_is_summarized_by_more_than_its_mean() -> None:
    """A bimodal share is not described by averaging its two populations.

    Most positions are near-fully observed and a minority are near-empty, so a
    single mean sits between two clusters and matches neither. The median and
    the starved share are what a reader needs to judge how much of a published
    calibration number rests on mostly-reconstructed positions.
    """
    captured = [0.99] * 8 + [0.01] * 2
    mean = sum(captured) / len(captured)
    ordered = sorted(captured)
    median = (ordered[4] + ordered[5]) / 2
    starved = sum(1 for value in captured if value < CALIBRATION_MASS_FLOOR) / len(captured)

    assert mean == pytest.approx(0.794)
    assert median == pytest.approx(0.99)
    assert starved == pytest.approx(0.2)
    assert abs(median - mean) > 0.15, "the two summaries must be able to disagree"
    # The mean is the stricter gate of the two, which is the safe direction.
    assert mean < median


def test_the_prompt_arm_varies_the_delimiter_and_nothing_else() -> None:
    """Isolating a protocol variable means changing exactly that variable.

    The undelimited prompt must keep the inventory statement, the JSON contract
    and the sequence itself identical; only the boundary marker and the sentence
    explaining it may differ. If anything else drifts, the paired contrast stops
    measuring the delimiter and starts measuring an unlabeled prompt rewrite.
    """
    delimited = build_prompt("letter", "the cat ")
    plain = build_undelimited_letter_prompt("the cat ")

    assert delimited != plain
    assert delimited.endswith("<the cat >")
    assert plain.endswith("Sequence: the cat ")
    for shared in (LETTER_INVENTORY_PHRASE, '{"answer":"<one character>"}'):
        assert shared in delimited
        assert shared in plain
    assert "angle brackets" in delimited
    assert "angle brackets" not in plain
    # The boundary marker and the sentence explaining it are the whole
    # difference; the sequence itself is byte-identical in both.
    assert len(delimited) > len(plain)
    assert plain.split("Sequence: ", 1)[1] == "the cat "


def _manuscript_variables_source() -> str:
    """Concatenated source of every file in the ``manuscript_variables`` package.

    The module used to be one file; it is now a package split by producer
    family. The AST scans below want to walk it as a single tree, so this
    glues every submodule's text back into one parseable module — same scan,
    same guarantee, now source-agnostic to which file a producer lives in.
    """
    package_dir = PROJECT_ROOT / "src" / "ailm" / "manuscript_variables"
    return "\n".join(path.read_text() for path in sorted(package_dir.glob("*.py")))


def test_no_derived_token_shadows_a_recorded_claim() -> None:
    """Exactly one producer per manuscript token.

    ``generate_manuscript_variables`` first writes ``variables[label.upper()]``
    for every ledger claim, then computes derived quantities. A derived
    assignment whose name matches a ledger label therefore *overwrites* the
    recorded value, and nothing reports it: the manuscript shows a recomputed
    number while the ledger holds the measured one, and the two can differ by
    rounding or by method. That happened once here, with the letter
    non-alphabetic accuracies, and was found by hand an hour later.

    The scan resolves literal subscript keys and enumerates dynamic ones, so a
    token built by an f-string over ledger labels is reported rather than
    silently skipped. This guard is about claim/token producer discipline
    generally and would sit equally well beside the other claims-discipline
    checks.
    """
    import ast

    source = _manuscript_variables_source()
    tree = ast.parse(source)
    ledger_labels = {
        label.lower()
        for label in json.loads((PROJECT_ROOT / "data" / "claims.json").read_text())["claims"]
    }
    assert ledger_labels, "no ledger labels loaded; this scan would be vacuous"

    literal_keys: set[str] = set()
    dynamic_keys: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        target = node.value
        if not (isinstance(target, ast.Name) and target.id == "variables"):
            continue
        key = node.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            literal_keys.add(key.value)
        else:
            dynamic_keys.append(ast.unparse(key))

    shadowed = sorted(key for key in literal_keys if key.lower() in ledger_labels)
    assert shadowed == [], (
        f"these tokens are assigned in manuscript_variables AND recorded in the ledger, "
        f"so the assignment silently overwrites the measured value: {shadowed}. "
        f"Delete the computation and let the recorder be the single producer, or give "
        f"the derived quantity a name no claim uses."
    )

    # The ledger loop and its provenance companions are the canonical dynamic
    # writers; anything else dynamic must be reviewed rather than assumed safe.
    allowed_dynamic = {
        "label.upper()",
        "f'{label.upper()}_RUN'",
        'f"{label.upper()}_RUN"',
        "f'{label.upper()}_SOURCE'",
        'f"{label.upper()}_SOURCE"',
        # These names come from closed, typed registries. Their exact current
        # expansions are checked below against the ledger rather than granting
        # a generic ``variables[name]`` exemption.
        "_literal_name",
        "_vendor_name",
        # The derived paired-bound loop. Reviewed 2026-08-17: it writes
        # ``<ARM>_DELTA_LOW_PTS``, ``<ARM>_DELTA_HIGH_PTS`` and
        # ``<ARM>_DISCORDANT`` for the seven arms named in its own tuple, and
        # none of the twenty-one resulting names is a ledger label. The
        # collision check below re-derives that rather than trusting this
        # comment, so the review cannot go stale silently.
        "f'{_arm.upper()}_DELTA_LOW_PTS'",
        "f'{_arm.upper()}_DELTA_HIGH_PTS'",
        "f'{_arm.upper()}_DISCORDANT'",
        # The subsample paired-bound loop, same shape over a different count
        # naming. Reviewed 2026-08-18: it writes the same three suffixes for the
        # arms named in SUBSAMPLE_BOUND_ARMS. Expanded below rather than trusted.
        "f'{_sub.upper()}_DELTA_LOW_PTS'",
        "f'{_sub.upper()}_DELTA_HIGH_PTS'",
        "f'{_sub.upper()}_DISCORDANT'",
        # The aligned-G2P comparator's Wilson display loop. Reviewed
        # 2026-09-14: ``_produce_aligned_g2p_comparator`` writes
        # ``<ARM>_<READING>_PCT`` / ``_WILSON_LOW_PCT`` / ``_WILSON_HIGH_PCT``
        # for the arms named in ALIGNED_G2P_COMPARATOR_ARMS over the position
        # and exact readings. Expanded below rather than trusted.
        "f'{stem}_PCT'",
        "f'{stem}_WILSON_LOW_PCT'",
        "f'{stem}_WILSON_HIGH_PCT'",
        # Hosted display names label retained cohorts, not recorded measurements.
        # Expand every registry member below to keep collision detection live.
        "f'BENCH_HOSTED_COHORT_{probe.key.upper()}_DISPLAY'",
    }

    # The reviewed dynamic writers are exempt from the *name* scan above, which
    # would otherwise have to evaluate an f-string. Their expansions still must
    # not shadow a claim, so they are expanded here against the arms the module
    # actually loops over. Without this, "reviewed" would mean "listed once".
    from ailm.manuscript_variables import (
        _LITERAL_VARIABLE_NAMES,
        ALIGNED_G2P_COMPARATOR_ARMS,
        DERIVED_BOUND_ARMS,
        SUBSAMPLE_BOUND_ARMS,
    )

    _suffixes = ("_delta_low_pts", "_delta_high_pts", "_discordant")
    expanded = {
        f"{arm}{suffix}".lower() for arm, _positions in DERIVED_BOUND_ARMS for suffix in _suffixes
    } | {f"{arm}{suffix}".lower() for arm in SUBSAMPLE_BOUND_ARMS for suffix in _suffixes}
    expanded |= {name.lower() for name in _LITERAL_VARIABLE_NAMES}
    expanded |= {
        f"bench_synthg2p_comparator_{arm}_{reading}{suffix}".lower()
        for arm in ALIGNED_G2P_COMPARATOR_ARMS
        for reading in ("position", "exact")
        for suffix in ("_pct", "_wilson_low_pct", "_wilson_high_pct")
    }
    from ailm.benchmarks.constants import SERVED_PROBES

    expanded |= {f"bench_hosted_cohort_{probe.key}_display" for probe in SERVED_PROBES}
    assert expanded, "no derived-bound arms found; this expansion scan would be vacuous"
    collisions = sorted(expanded & ledger_labels)
    assert collisions == [], (
        f"these derived paired-bound tokens shadow recorded claims: {collisions}. "
        f"Rename the derived quantity; the recorder must stay the single producer."
    )
    unreviewed = [
        key
        for key in dynamic_keys
        if key not in allowed_dynamic and "prefix" not in key and "system" not in key
    ]
    assert unreviewed == [], (
        f"unreviewed dynamic token assignments in manuscript_variables: {unreviewed}. "
        f"Each one can collide with a ledger label; name them explicitly here once checked."
    )


def test_every_manuscript_producer_is_in_the_shadow_scan_shape() -> None:
    """Non-vacuity, per producer, for the shadowing scan above.

    ``generate_manuscript_variables`` delegates to one ``_produce_<section>``
    writer per section, each taking a parameter named exactly ``variables``
    and writing its tokens by subscript on that name. The scan above is a
    module-wide walk, so it sees those subscripts wherever the writes happen —
    but only if the producers keep the shape. If a producer renamed its
    parameter, or built its tokens without a subscript on ``variables``, the
    scan would pass vacuously over a map the guard can no longer vouch for.
    This test holds each producer to the shape the scan assumes.
    """
    import ast

    source = _manuscript_variables_source()
    tree = ast.parse(source)
    producers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_produce_")
    ]
    assert producers, "no _produce_* producers found; this scan would be vacuous"
    for producer in producers:
        assert producer.args.args, f"{producer.name} has no parameters"
        assert producer.args.args[0].arg == "variables", (
            f"{producer.name} must take a parameter named exactly `variables` "
            "first; the shadow scan keys on subscripts of that name"
        )
        writes = [
            node
            for node in ast.walk(producer)
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "variables"
        ]
        assert writes, (
            f"{producer.name} contains no subscript on `variables`; the shadowing "
            "scan cannot see its token writes, so a collision there would go "
            "unreported. Write the tokens by subscript on the `variables` "
            "parameter, or fold the block into a producer that does."
        )


def _recorder_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_bench_recorder_lock", PROJECT_ROOT / "scripts" / "record_external_benchmark_claims.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_lock_is_never_observably_empty(tmp_path) -> None:
    """The acquire must be one step, or two processes can both hold the lock.

    Creating the file and then writing the pid leaves a window where the lock
    exists with no contents. A second process reading it in that window finds no
    pid, concludes the lock is stale, removes it and takes its own — and the
    mutual exclusion is gone with no error on either side. This pins the property
    that closes the window: the file appears with the pid already in it.
    """
    from ailm.ledger_lock import _claim, _read_holder

    lock = tmp_path / "claims.lock"
    assert _claim(lock)
    assert _read_holder(lock) == os.getpid(), "lock existed without a readable pid"
    assert not _claim(lock), "a second claim on a held lock must fail"


def test_release_does_not_delete_a_lock_owned_by_someone_else(tmp_path) -> None:
    """NEGATIVE CONTROL: unconditional release cascades into a third writer.

    A process that has lost its lock must not remove whatever lock is present on
    the way out, or one stolen lock admits another writer, and then another.
    """
    from ailm.ledger_lock import single_writer_lock

    lock = tmp_path / "claims.lock"
    lock.write_text("999999\n")  # a dead holder, so the lock is reclaimable
    with single_writer_lock(lock):
        lock.write_text("424242\n")  # a live third party now records itself
    assert lock.exists(), "release deleted a lock this process no longer owned"
    assert lock.read_text().strip() == "424242"


def test_a_second_recorder_run_is_refused_while_one_holds_the_lock(tmp_path) -> None:
    """The ledger is single-writer, and that is enforced rather than assumed.

    ``ClaimsLedger`` reads the whole file at construction and writes the whole
    file at the end, so two overlapping runs are a lost update: the second to
    finish discards everything the first wrote, with no error anywhere.
    """
    from ailm.ledger_lock import single_writer_lock

    lock = tmp_path / "claims.lock"
    with single_writer_lock(lock):
        assert lock.read_text().strip() == str(os.getpid())
        with pytest.raises(SystemExit, match="another recorder run"):
            with single_writer_lock(lock):
                pytest.fail("two runs acquired the same lock")
    assert not lock.exists(), "the lock must be released even though the inner call raised"


def test_a_lock_left_by_a_dead_process_is_reclaimed(tmp_path) -> None:
    """A lock that outlives its holder must not wedge the repository.

    A killed recorder would otherwise block every future run, which is worse than
    the race the lock prevents: the race wastes compute, a block stops recording.
    """
    from ailm.ledger_lock import holder_is_alive, single_writer_lock

    lock = tmp_path / "claims.lock"
    lock.write_text("999999\n")
    assert not holder_is_alive(999999)
    with single_writer_lock(lock):
        assert lock.read_text().strip() == str(os.getpid())
    assert not lock.exists()


def test_liveness_check_recognizes_a_running_process() -> None:
    """Non-vacuity: if this always returned False the lock would never hold."""
    from ailm.ledger_lock import holder_is_alive

    assert holder_is_alive(os.getpid())
    assert not holder_is_alive(-1)


def _recorder_sources() -> dict[str, str]:
    """Every claims recorder's source text, keyed by file name."""
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted((PROJECT_ROOT / "scripts").glob("record_*_claims.py"))
    }


def test_every_recorder_sharing_the_bench_namespace_takes_the_lock() -> None:
    """Coverage: a lock one of two contending writers takes is not mutual exclusion.

    Contenders are DISCOVERED rather than named. A hardcoded pair goes stale the
    moment a third recorder starts writing ``bench_*`` labels, and it goes stale
    silently: the two listed files still pass while the new one races them.
    """
    contenders = {
        name: source for name, source in _recorder_sources().items() if "bench_" in source
    }
    assert len(contenders) >= 2, (
        f"only {sorted(contenders)} write the bench_ namespace; a mutual-exclusion "
        f"test with fewer than two contenders proves nothing"
    )
    unlocked = sorted(
        name for name, source in contenders.items() if "single_writer_lock" not in source
    )
    assert unlocked == [], f"{unlocked} write bench_ labels without taking the lock"


def test_no_recorder_writes_the_ledger_without_the_lock() -> None:
    """Ratchet: the lock cannot be dropped from a recorder without failing here.

    This began as a frozen inventory of the four recorders that had been
    converted, because the other seventeen were a known gap and a shrinking gap
    should be visible in a diff. The gap is now closed, so the frozen list is
    retired in favor of the rule it was approximating: every recorder takes the
    lock, and any recorder that stops doing so fails here. An enumeration would
    now be strictly weaker — it would have to be edited to admit each new
    recorder, and the edit is exactly the moment the lock gets forgotten.

    ``tests/test_claims_single_writer.py`` states the same rule from the ledger
    side, scoped to recorders that call ``load_ledger``. Both are cheap; the
    duplication is deliberate, because this file is where the ``bench_``
    contention that motivated the lock is tested.
    """
    sources = _recorder_sources()
    # An empty scan satisfies "none are unlocked" without checking anything.
    assert len(sources) >= 14, f"only {len(sources)} recorders discovered; the scan is broken"
    unlocked = sorted(
        name for name, source in sources.items() if "single_writer_lock" not in source
    )
    assert unlocked == [], (
        f"{unlocked} write the claims ledger without taking the single-writer "
        f"lock, reopening the lost-update race the lock exists to close. Wrap "
        f"the entrypoint rather than removing this assertion."
    )


def test_every_manuscript_token_is_guarded_by_required_claims() -> None:
    """Closure the other direction: prose -> REQUIRED_CLAIMS.

    ``REQUIRED_CLAIMS`` is checked against the ledger, and nothing checked the
    manuscript against ``REQUIRED_CLAIMS``. A token could therefore be cited in
    prose, backed in the ledger, and guarded by nothing: delete the recorder
    that produced it and the render keeps succeeding until the ledger is rebuilt
    from scratch, at which point the failure surfaces far from its cause.

    Only ledger-shaped tokens are in scope. ``META_*`` are build provenance and
    derived tokens are computed in ``manuscript_variables``, so both are
    resolved without a claim by design.

    A token is matched to its label case-insensitively, not by ``token.lower()``.
    Fifteen ledger labels keep an interior capital — ``g2p_learn_A_train_pairs``
    and its siblings — so a lowercasing match skips every one of them and the
    nine that the phoneme section cites are scanned by nothing. They happen to be
    in ``REQUIRED_CLAIMS``, which is exactly why the gap was invisible.
    """
    from ailm.manuscript_variables import REQUIRED_CLAIMS, generate_manuscript_variables

    ledger = set(json.loads((PROJECT_ROOT / "data" / "claims.json").read_text())["claims"])
    ledger_by_token = {label.upper(): label for label in ledger}
    assert len(ledger_by_token) == len(ledger), (
        "two ledger labels differ only in case, so a token cannot name one of them"
    )
    cited: set[str] = set()
    for section in sorted((PROJECT_ROOT / "docs" / "manuscript").glob("*.md")):
        cited |= set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", section.read_text(encoding="utf-8")))
    assert cited, "no tokens found in the manuscript; this scan would be vacuous"
    assert generate_manuscript_variables(PROJECT_ROOT).keys() >= cited, (
        "a manuscript token does not resolve at all; test_manuscript_tokens.py owns that"
    )

    required = {label.upper() for label in REQUIRED_CLAIMS}
    unguarded = sorted(
        token
        for token in cited
        if token in ledger_by_token and token not in required and not token.startswith("META_")
    )
    assert unguarded == [], (
        f"these manuscript tokens resolve from the ledger but are not in "
        f"REQUIRED_CLAIMS, so nothing fails if their producer disappears: "
        f"{unguarded}. Add them to REQUIRED_CLAIMS."
    )


def test_the_sample_size_is_large_enough_for_the_resolvable_arms() -> None:
    """A default lowered for runtime must not silently cost the paper its results.

    The benchmark's decisive contrasts are phoneme and G2P. At 100 positions
    neither clears the corrected family threshold; at 400 both clear it by
    several orders of magnitude. Because every other test stays green when the
    sample shrinks, nothing else in the suite notices that the conclusions have
    gone. This pins the floor and states why, so the next person to shorten a
    run has to argue with the number rather than edit past it.
    """
    from ailm.statistics import REFERENCE_SHARE, exact_binomial_power

    assert DEFAULT_TASK_N >= 400, (
        f"DEFAULT_TASK_N is {DEFAULT_TASK_N}. Below 400 the phoneme and G2P "
        f"contrasts stop clearing the corrected threshold, which removes the only "
        f"decisive findings the external benchmark has."
    )

    # The constant governs only the no-flag path. `--letter-n 100` writes 100 into
    # the ledger and leaves every assertion in the suite green, which is exactly
    # how the sample silently dropped once already. The paper is built from the
    # ledger, so the floor has to be asserted against the ledger too.
    ledger = json.loads((PROJECT_ROOT / "data" / "claims.json").read_text())["claims"]
    for label in ("bench_letter_positions", "bench_phoneme_positions", "bench_g2p_words"):
        if label not in ledger:
            pytest.skip("external benchmark not recorded in this checkout")
        recorded = ledger[label]["value"]
        assert recorded >= 400, (
            f"{label} is {recorded} in data/claims.json. The recorded sample, not the "
            f"module default, is what the manuscript reports; below 400 the phoneme "
            f"and G2P contrasts stop clearing the corrected threshold and the paper "
            f"loses its only decisive results. Re-record with the default flags."
        )
    # Non-vacuity: the letter arm is genuinely underpowered even at this size, so
    # the floor is a minimum for the resolvable arms and not a claim of adequacy.
    #
    # The threshold is rebuilt from the recorder's own family arithmetic rather
    # than written as a digit. A literal here was a copy of 0.05 over eighteen
    # tests; retiring a probe shrinks the family, loosens the threshold, and
    # raises this power — and the stale literal would have gone on certifying the
    # arm as underpowered against a correction nobody applies any more.
    recorder = _load_recorder()
    family_alpha = recorder.POWER_ALPHA / (len(recorder.FAMILY_TEMPLATES) * len(recorder.MODELS))
    discordant = int(0.295 * DEFAULT_TASK_N)
    assert exact_binomial_power(discordant, REFERENCE_SHARE, family_alpha) < 0.5


# ── The Ollama transport, exercised through a real loopback server ───────────


def test_project_root_is_found_by_marker_not_by_hop_count(tmp_path) -> None:
    """POSITIVE: the walk finds the real root. NEGATIVE: no markers, no guess.

    The hop-count version broke when `benchmarks.py` became `benchmarks/`, and
    it failed by returning a plausible-looking wrong path. The refusal branch is
    the half that protects the next refactor: from a file with no marked
    ancestor the function must raise, not return whatever directory sits at the
    old depth.
    """
    from ailm.benchmarks import ollama as ollama_module

    root = ollama_module._project_root()
    assert (root / "vendor").is_dir() and (root / "src").is_dir()

    with pytest.raises(RuntimeError, match="vendor"):
        ollama_module._project_root(tmp_path / "unmarked" / "deep" / "module.py")


def test_ollama_available_matches_exact_tags_and_family_names(local_http_server) -> None:
    """POSITIVE: an exact tag and a same-family tag both count as available.

    NEGATIVE CONTROL: a model the server does not carry reports False even
    though the server answered — availability is about the model, not the
    socket.
    """
    local_http_server.enqueue_json(
        "/api/tags",
        {"models": [{"name": "gemma3:latest"}, {"name": "phi4:14b"}]},
        repeat=3,
    )
    endpoint = local_http_server.base_url
    assert ollama_available("gemma3:latest", endpoint=endpoint) is True
    assert ollama_available("gemma3:4b", endpoint=endpoint) is True, (
        "a same-family tag should count"
    )
    assert ollama_available("qwen3:8b", endpoint=endpoint) is False

    local_http_server.enqueue_disconnect("/api/tags")
    assert ollama_available("gemma3:latest", endpoint=endpoint) is False


def test_post_generate_sends_the_payload_it_was_given_and_returns_the_body(
    local_http_server,
) -> None:
    """The one function that touches the network, checked at the boundary.

    The request must land on `/api/generate` as JSON carrying the caller's
    payload unaltered — a transport that re-shapes the payload would decouple
    the cache key (computed from the payload) from the question actually asked.
    """
    from ailm.benchmarks.ollama import _generate_payload, post_generate_uncached

    local_http_server.enqueue_json("/api/generate", {"response": '{"answer":"b"}'})
    payload = _generate_payload("gemma3:4b", "the quick brown fo", "letter", 4, 1234)
    body = post_generate_uncached(payload, endpoint=local_http_server.base_url)

    [request] = local_http_server.requests_for("/api/generate")
    assert request.method == "POST"
    assert json.loads(request.body) == payload
    assert body == {"response": '{"answer":"b"}'}
    # The payload itself is the scored request: schema-constrained, greedy,
    # seeded. A drift in any of these changes what the benchmark measures.
    assert payload["stream"] is False
    assert payload["options"] == {
        "temperature": 0,
        "seed": 1234,
        "num_predict": 4,
        "use_mmap": False,
        "num_ctx": 8192,
    }
    assert payload["format"]["required"] == ["answer"]


def test_an_unknown_task_is_refused_before_it_reaches_the_server() -> None:
    """A task with no schema would send an unconstrained prompt and score it."""
    from ailm.benchmarks.ollama import _generate_payload

    with pytest.raises(ValueError, match="unknown task"):
        _generate_payload("gemma3:4b", "prompt", "essay", 4, 1234)


def test_ask_ollama_returns_the_reply_text_and_tolerates_a_missing_field(
    local_http_server,
) -> None:
    local_http_server.enqueue_json("/api/generate", {"response": '{"answer":"r"}'})
    local_http_server.enqueue_json("/api/generate", {"done": True})
    endpoint = local_http_server.base_url
    assert ask_ollama("gemma3:4b", "prompt", "letter", endpoint=endpoint) == '{"answer":"r"}'
    assert ask_ollama("gemma3:4b", "prompt", "letter", endpoint=endpoint) == ""


def test_the_answer_slot_scan_gives_up_past_the_envelope_length() -> None:
    """A reply that never opens the envelope must terminate the scan early.

    The accumulated prefix is bounded at 32 characters because the envelope is
    11; a model emitting a long preamble would otherwise be scanned token by
    token to the end of a 32-token reply for a slot that cannot be there.
    """
    tokens = ["this is a long preamble that never opens a JSON envelope"] * 5
    assert _answer_token_index(tokens) is None


def test_a_candidate_list_covering_the_whole_inventory_needs_no_completion() -> None:
    """POSITIVE: full coverage renormalizes what was observed and adds nothing.

    NEGATIVE CONTROL: dropping one symbol switches to the min-bound completion,
    so this cannot pass on an implementation that always completes.
    """
    full = [{"token": symbol, "logprob": -2.0} for symbol in ALPHA_LIST]
    distribution = _distribution_from_candidates(full)
    assert set(distribution) == set(ALPHA_LIST)
    assert sum(distribution.values()) == pytest.approx(1.0)
    # Equal logprobs over the whole inventory renormalize to exactly uniform.
    assert all(v == pytest.approx(1.0 / len(ALPHA_LIST)) for v in distribution.values())

    partial = _distribution_from_candidates(full[:-1])
    assert set(partial) == set(ALPHA_LIST), "the missing symbol was not completed"
    assert partial[ALPHA_LIST[-1]] == pytest.approx(partial[ALPHA_LIST[0]]), (
        "the completed symbol should carry the min-bound, here equal to every observed probability"
    )


def test_answer_candidates_are_read_from_the_answer_slot_of_the_reply(
    local_http_server,
) -> None:
    """POSITIVE: the slot's own top_logprobs come back beside the text.

    NEGATIVE CONTROL: a reply with no logprobs — or logprobs that never open
    the envelope — yields an empty candidate list, not a crash and not a
    fabricated distribution.
    """
    from ailm.benchmarks.ollama import ask_ollama_answer_candidates

    slot_candidates = [
        {"token": "b", "logprob": -0.1},
        {"token": 'r"', "logprob": -2.5},
    ]
    body = {
        "response": '{"answer":"b"}',
        "logprobs": [
            {"token": '{"answer":"', "top_logprobs": []},
            {"token": "b", "top_logprobs": slot_candidates},
            {"token": '"}', "top_logprobs": []},
        ],
    }
    local_http_server.enqueue_json("/api/generate", body)
    endpoint = local_http_server.base_url
    text, candidates = ask_ollama_answer_candidates(
        "gemma3:4b", "prompt", "letter", endpoint=endpoint
    )
    assert text == '{"answer":"b"}'
    assert candidates == slot_candidates

    local_http_server.enqueue_json("/api/generate", {"response": "no logprobs"})
    text, candidates = ask_ollama_answer_candidates(
        "gemma3:4b", "prompt", "letter", endpoint=endpoint
    )
    assert (text, candidates) == ("no logprobs", [])

    preamble = {"response": "```json", "logprobs": [{"token": "```json", "top_logprobs": []}]}
    local_http_server.enqueue_json("/api/generate", preamble)
    assert ask_ollama_answer_candidates("gemma3:4b", "prompt", "letter", endpoint=endpoint)[1] == []


def test_the_distribution_reading_is_the_candidates_reading_and_nothing_more(
    local_http_server,
) -> None:
    """`ask_ollama_distribution` must be a thin reading of the candidate list.

    Its distribution and mass must equal what the shared rules compute from the
    same candidates — a second opinion here would let calibration describe a
    decision the benchmark never scored. An unlocated answer slot must come
    back as the uniform column with zero captured mass, exactly as the module
    docstring claims.
    """
    slot_candidates = [
        {"token": "b", "logprob": -0.5},
        {"token": "zzz", "logprob": -0.2},
    ]
    body = {
        "response": '{"answer":"b"}',
        "logprobs": [
            {"token": '{"answer":"', "top_logprobs": []},
            {"token": "b", "top_logprobs": slot_candidates},
        ],
    }
    local_http_server.enqueue_json("/api/generate", body)
    endpoint = local_http_server.base_url
    text, distribution, mass = ask_ollama_distribution(
        "gemma3:4b", "prompt", "letter", endpoint=endpoint
    )
    assert text == '{"answer":"b"}'
    assert distribution == _distribution_from_candidates(slot_candidates)
    assert mass == pytest.approx(captured_mass(slot_candidates))

    local_http_server.enqueue_json("/api/generate", {"response": "nothing"})
    _, uniform, zero_mass = ask_ollama_distribution(
        "gemma3:4b", "prompt", "letter", endpoint=endpoint
    )
    assert zero_mass == 0.0
    assert all(v == pytest.approx(1.0 / len(ALPHA_LIST)) for v in uniform.values())


def test_every_task_constrains_the_answer_to_its_own_inventory() -> None:
    """The response schema is what makes the reply parseable at all.

    Each task's ``answer`` slot must enumerate exactly its own inventory: a
    letter task offering phones, or a G2P task accepting free strings, would
    let the model emit an answer the scorer cannot score, and the failure
    would read as model error rather than harness error.
    """
    from ailm.benchmarks.ollama import _generate_payload
    from ailm.levels.phoneme import PHONES

    letter = _generate_payload("m", "p", "letter", 4, 1)["format"]["properties"]["answer"]
    assert letter == {"type": "string", "enum": sorted(ALPHA_LIST)}

    phoneme = _generate_payload("m", "p", "phoneme", 4, 1)["format"]["properties"]["answer"]
    assert phoneme == {"type": "string", "enum": sorted(PHONES)}

    g2p = _generate_payload("m", "p", "g2p", 4, 1)["format"]["properties"]["answer"]
    assert g2p["type"] == "array" and g2p["minItems"] == 1
    assert g2p["items"]["enum"] == sorted(PHONES)
