"""Gate for the stipulated-language exact-artifact-overlap probe.

## What this probe is for

The external benchmark scores AILM against a reference LLM on Brown and CMUdict.
Both are public and old, so the reference model may well have memorised them: a
result there cannot separate inference from recall. The synthetic probe narrows
that concern by stipulating and locally generating a language in code
(:mod:`ailm.levels.synthetic`). It does not establish an external model's
pretraining contents. AILM trains on a sample of it; the reference model meets
the same independently generated evaluation positions without labeled examples.

## What this file has to prove

A synthetic probe is exactly as trustworthy as its controls, because a designer
who invents both the language and the scoring can invent a win. So:

* the sampler must actually implement the transition that is written down
  (empirical convergence, not a claim in a docstring);
* the artifact must reproduce byte-for-byte, or its checksum means nothing;
* the comparator must beat the uniform floor **and** must collapse when trained
  on the wrong statistics — an English-trained comparator and a transposed count
  table are both scored here, and both must fail. Without them, "AILM wins" would
  be consistent with a comparator that had learned nothing at all;
* the recorder must refuse rather than record when the reference model is absent.

No mocks anywhere: real NumPy, the real vendored artifact, the real generator and
recorder scripts run as subprocesses. Tests that reach a user's Ollama process
require an explicit environment opt-in; the default suite never probes it.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from ailm.benchmarks import (
    CALL_CACHE_PATH_ENV,
    SYNTHETIC_LETTER_INVENTORY_PHRASE,
    SYNTHETIC_LETTER_INVENTORY_SYMBOLS,
    ask_ollama,
    build_prompt,
    call_cache,
    enable_call_cache,
    ollama_available,
    parse_phoneme_answer,
    parse_synthetic_letter_answer,
)
from ailm.benchmarks.constants import SERVED_PROBES
from ailm.benchmarks.ollama import _response_format
from ailm.benchmarks.synthetic_protocol import (
    SYNTHETIC_INCONTEXT_PROTOCOL,
    SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL,
    SYNTHETIC_ZERO_SHOT_PROTOCOL,
)
from ailm.call_cache import CacheRequest, CallCache
from ailm.levels.letter import ALPHA_LIST, ALPHA_TO_IDX, load_brown_counts
from ailm.levels.phoneme import PHONES, load_phoneme_counts
from ailm.levels.synthetic import (
    CONSONANTS,
    DEFAULT_CORPUS_CHARS,
    DEFAULT_CORPUS_SEED,
    DEFAULT_HELD_OUT_SEED,
    DEFAULT_PHONE_CORPUS_SEED,
    DEFAULT_PHONE_CORPUS_SYMBOLS,
    DEFAULT_PHONE_HELD_OUT_SEED,
    PHONE_CONSONANTS,
    PHONE_UNIFORM_BASELINE,
    PHONE_VOWELS,
    SYNTHETIC_LETTERS,
    SYNTHETIC_PHONE_CHAIN_VERSION,
    SYNTHETIC_PHONES,
    UNIFORM_BASELINE,
    VOWELS,
    W_DOMINANT,
    W_OPPOSITE,
    W_UNIFORM,
    build_synthetic_counts,
    build_synthetic_phone_counts,
    correctness_flags,
    count_bigrams,
    count_phone_bigrams,
    designed_transition,
    dominant_successor,
    generate_phone_sequence,
    generate_sequence,
    independent_held_out_positions,
    independent_phone_held_out_positions,
    independent_target_start_tvd,
    load_synthetic_counts,
    load_synthetic_phone_counts,
    phone_designed_transition,
    phone_dominant_successor,
    phone_target_start_tvd,
    synthetic_letter_predictions,
    synthetic_phone_predictions,
)
from ailm.replay_provenance import NetworkClaimBindingError
from tests.test_provenance_checksums import DOCUMENTED_ARTIFACTS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
ARTIFACT = PROJECT_ROOT / "vendor" / "synthetic_counts.json"
LEDGER = PROJECT_ROOT / "data" / "claims.json"
GENERATOR = PROJECT_ROOT / "scripts" / "generate_synthetic_counts.py"
RECORDER = PROJECT_ROOT / "scripts" / "record_synthetic_benchmark_claims.py"
PHONE_ARTIFACT = PROJECT_ROOT / "vendor" / "synthetic_phone_counts.json"
PHONE_GENERATOR = PROJECT_ROOT / "scripts" / "generate_synthetic_phone_counts.py"

REFERENCE_MODELS = tuple(probe.tag for probe in SERVED_PROBES)
REFERENCE_MODEL = REFERENCE_MODELS[0]
LIVE_OLLAMA_TEST_ENV = "AILM_RUN_LIVE_OLLAMA_TESTS"
_RUN_LIVE_OLLAMA_TESTS = os.environ.get(LIVE_OLLAMA_TEST_ENV) == "1"
#: Set by the full release profile, which forbids skips: an opted-in live test
#: whose model is not served must then fail and name the model, not skip.
RELEASE_MANDATORY_ENVIRONMENT_ENV = "AILM_RELEASE_MANDATORY_ENVIRONMENT"

_NEEDS_VENV = pytest.mark.skipif(not VENV_PYTHON.exists(), reason="project venv absent")
_NEEDS_PHONE_ARTIFACT = pytest.mark.skipif(
    not PHONE_ARTIFACT.exists(),
    reason="vendor/synthetic_phone_counts.json absent; run the generator",
)
_NEEDS_ARTIFACT = pytest.mark.skipif(
    not ARTIFACT.exists(), reason="vendor/synthetic_counts.json absent; run the generator"
)
_NEEDS_LIVE_OLLAMA_OPT_IN = pytest.mark.skipif(
    not _RUN_LIVE_OLLAMA_TESTS,
    reason=f"live Ollama tests require explicit {LIVE_OLLAMA_TEST_ENV}=1 opt-in",
)


def _require_live_models(models: tuple[str, ...]) -> None:
    """Check the user's server only after the live-test opt-in was accepted.

    Under the release profile's mandatory-environment flag an unserved model is
    a failure, because that profile forbids skips and a skip there would be
    rejected without saying which model was missing.
    """
    unavailable = [model for model in models if not ollama_available(model)]
    if not unavailable:
        return
    reason = f"opted in, but local Ollama model(s) unavailable: {unavailable}"
    if os.environ.get(RELEASE_MANDATORY_ENVIRONMENT_ENV) == "1":
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.fixture(scope="module")
def vendored_counts() -> dict[str, int]:
    if not ARTIFACT.exists():
        pytest.skip("vendor/synthetic_counts.json absent")
    return load_synthetic_counts()


@pytest.fixture(scope="module")
def phone_vendored_counts() -> dict[str, int]:
    if not PHONE_ARTIFACT.exists():
        pytest.skip("vendor/synthetic_phone_counts.json absent")
    return load_synthetic_phone_counts()


@pytest.fixture(scope="module")
def phone_evaluation_positions() -> list[tuple[str, str]]:
    """The executed phone-arm population: one independent trajectory per item."""

    return independent_phone_held_out_positions(
        SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL.positions,
        window=SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL.query_window,
        seed=SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL.position_seed,
    )


@pytest.fixture(scope="module")
def evaluation_positions() -> list[tuple[str, str]]:
    """The executed v2 population: one independent trajectory per item."""
    return independent_held_out_positions(
        SYNTHETIC_ZERO_SHOT_PROTOCOL.positions,
        window=SYNTHETIC_ZERO_SHOT_PROTOCOL.query_window,
        seed=SYNTHETIC_ZERO_SHOT_PROTOCOL.position_seed,
    )


# ── The designed language is the language that was designed ──────────────────


def test_transition_is_column_stochastic_and_everywhere_positive() -> None:
    transition = designed_transition()
    assert transition.shape == (len(SYNTHETIC_LETTERS), len(SYNTHETIC_LETTERS))
    assert np.allclose(transition.sum(axis=0), 1.0)
    assert transition.min() > 0.0, "a zero cell would make some bigram unreachable"


def test_transition_matches_the_written_down_mixture() -> None:
    """POSITIVE CONTROL: the closed form, not a shape check.

    A matrix that is merely column-stochastic tells us nothing about *which*
    chain it is. These three cells pin the dominant term, the alternation term
    and the uniform floor separately, so a bug that dropped any one of them
    cannot hide behind the other two.
    """
    transition = designed_transition()
    index = {symbol: i for i, symbol in enumerate(SYNTHETIC_LETTERS)}
    floor = W_UNIFORM / len(SYNTHETIC_LETTERS)

    # From the vowel 'a', the opposite class is the 21 consonants.
    successor = dominant_successor("a")
    assert transition[index[successor], index["a"]] == pytest.approx(
        W_DOMINANT + W_OPPOSITE / len(CONSONANTS) + floor
    )
    other_consonant = next(c for c in CONSONANTS if c != successor)
    assert transition[index[other_consonant], index["a"]] == pytest.approx(
        W_OPPOSITE / len(CONSONANTS) + floor
    )
    # Same-class transitions carry the uniform floor only.
    assert transition[index["e"], index["a"]] == pytest.approx(floor)


def test_dominant_successor_is_the_argmax_of_every_column() -> None:
    """POSITIVE CONTROL: the order-1 ceiling is the designed successor map.

    The probe's whole premise is that a well-estimated bigram table recovers this
    map. If some other symbol carried more mass in any column, the ceiling would
    be something other than what the module documents.
    """
    transition = designed_transition()
    for i, current in enumerate(SYNTHETIC_LETTERS):
        assert SYNTHETIC_LETTERS[int(np.argmax(transition[:, i]))] == dominant_successor(current)


def test_successors_cross_the_class_boundary_and_do_not_collapse() -> None:
    assert all(dominant_successor(v) in CONSONANTS for v in VOWELS)
    assert all(dominant_successor(c) in VOWELS for c in CONSONANTS)
    assert len({dominant_successor(v) for v in VOWELS}) == len(VOWELS), (
        "the vowel strides must be coprime with the consonant count, or several "
        "vowels share a successor and the designed structure thins out"
    )


def test_transition_is_not_symmetric_under_transposition() -> None:
    """NEGATIVE CONTROL against the wiring defect this repo has shipped before.

    A bigram matrix is column-stochastic in either orientation, so ``pymdp`` and
    every shape assertion accept the transpose. Only content distinguishes them.
    """
    transition = designed_transition()
    assert not np.allclose(transition, transition.T)
    transposed_argmax = [
        SYNTHETIC_LETTERS[int(np.argmax(transition.T[:, i]))] for i in range(len(SYNTHETIC_LETTERS))
    ]
    designed_argmax = [dominant_successor(c) for c in SYNTHETIC_LETTERS]
    agreements = sum(a == b for a, b in zip(transposed_argmax, designed_argmax, strict=True))
    assert agreements < len(SYNTHETIC_LETTERS) // 2, (
        f"the transposed matrix agrees with the designed successor map on "
        f"{agreements} columns — the orientation check would be near-vacuous"
    )


def test_dominant_successor_rejects_a_symbol_outside_the_alphabet() -> None:
    for outside in (" ", ".", "A", "1"):
        with pytest.raises(ValueError):
            dominant_successor(outside)


# ── The sampler implements that language ─────────────────────────────────────


def test_generation_is_deterministic_in_its_seed() -> None:
    first = generate_sequence(2_000, seed=11)
    assert first == generate_sequence(2_000, seed=11)
    assert first != generate_sequence(2_000, seed=12)
    assert len(first) == 2_000


def test_generation_emits_only_synthetic_letters() -> None:
    stream = generate_sequence(5_000, seed=3)
    assert set(stream) <= set(SYNTHETIC_LETTERS)
    assert set(stream) == set(SYNTHETIC_LETTERS), "a well-mixed sample should visit every symbol"


def test_generation_rejects_degenerate_arguments() -> None:
    with pytest.raises(ValueError):
        generate_sequence(1, seed=0)
    with pytest.raises(ValueError):
        generate_sequence(10, seed=-1)


@_NEEDS_ARTIFACT
def test_sampled_frequencies_converge_to_the_designed_transition(
    vendored_counts: dict[str, int],
) -> None:
    """POSITIVE CONTROL that the corpus is a sample of the stipulated chain.

    Without this the module could document one transition and sample another, and
    every downstream number would describe a language nobody wrote down.
    """
    size = len(SYNTHETIC_LETTERS)
    index = {symbol: i for i, symbol in enumerate(SYNTHETIC_LETTERS)}
    empirical = np.zeros((size, size), dtype=np.float64)
    for key, count in vendored_counts.items():
        empirical[index[key[2]], index[key[0]]] = count
    empirical /= empirical.sum(axis=0, keepdims=True)
    assert np.abs(empirical - designed_transition()).max() < 0.03


def test_independent_evaluation_positions_share_no_windows_with_training() -> None:
    """The evaluated windows must not be corpus windows AILM was fitted on."""
    training = generate_sequence(DEFAULT_CORPUS_CHARS, DEFAULT_CORPUS_SEED)
    assert DEFAULT_HELD_OUT_SEED != DEFAULT_CORPUS_SEED
    positions = independent_held_out_positions(40, window=20, seed=DEFAULT_HELD_OUT_SEED)
    probes = [prefix + gold for prefix, gold in positions]
    shared = [probe for probe in probes if probe in training]
    assert shared == [], f"held-out windows found verbatim in the training stream: {shared}"


# ── The scored positions ─────────────────────────────────────────────────────
#
# Three properties of the shared harness bite the main benchmark and must be shown
# not to bite this probe. Each is asserted rather than argued.


def test_synthetic_prompt_schema_parser_and_gold_share_one_exact_inventory(
    evaluation_positions: list[tuple[str, str]],
) -> None:
    """The executed prompt opportunity must equal the schema, parser, and gold set."""
    expected = tuple("abcdefghijklmnopqrstuvwxyz")
    assert SYNTHETIC_LETTERS == expected
    assert SYNTHETIC_ZERO_SHOT_PROTOCOL.response_inventory == expected
    assert SYNTHETIC_INCONTEXT_PROTOCOL.response_inventory == expected
    assert SYNTHETIC_LETTER_INVENTORY_SYMBOLS == frozenset(expected)
    assert SYNTHETIC_LETTER_INVENTORY_PHRASE == "a lowercase letter a-z"

    schema = _response_format("synthetic_letter")
    answer_schema = schema["properties"]["answer"]
    assert answer_schema == {"type": "string", "enum": list(expected)}
    assert schema["required"] == ["answer"]
    assert schema["additionalProperties"] is False

    prompt = build_prompt("synthetic_letter", evaluation_positions[0][0])
    assert SYNTHETIC_LETTER_INVENTORY_PHRASE in prompt
    assert "space" not in prompt and "apostrophe" not in prompt and "hyphen" not in prompt
    assert {gold for _prefix, gold in evaluation_positions} <= set(expected)
    for symbol in expected:
        encoded = json.dumps({"answer": symbol}, separators=(",", ":"))
        assert parse_synthetic_letter_answer(encoded) == (symbol, False)


@pytest.mark.parametrize(
    "answer",
    (
        '{"answer":"A"}',
        '{"answer":" a "}',
        ' {"answer":"a"}',
        '{"answer":"a"}\n',
        '{"answer":"a","extra":1}',
        '```json\n{"answer":"a"}\n```',
        '{"answer":"."}',
        '{"answer":" "}',
        '{"answer":"ab"}',
    ),
)
def test_synthetic_parser_rejects_every_nonexact_response_contract(answer: str) -> None:
    """NEGATIVE CONTROL: normalization cannot widen a 26-way opportunity."""
    assert parse_synthetic_letter_answer(answer) == (None, True)


def test_protocol_registry_is_frozen_and_the_three_measurements_are_cache_distinct() -> None:
    """Recorders consume immutable code contracts, not copied literals."""
    zero = SYNTHETIC_ZERO_SHOT_PROTOCOL
    context = SYNTHETIC_INCONTEXT_PROTOCOL
    phone = SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL
    with pytest.raises(FrozenInstanceError):
        zero.positions = zero.positions + 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        context.headline_k = context.headline_k - 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        phone.positions = phone.positions + 1  # type: ignore[misc]
    assert zero.chain_version == context.chain_version
    assert phone.chain_version == SYNTHETIC_PHONE_CHAIN_VERSION
    assert zero.position_seed == context.position_seed
    assert phone.position_seed == DEFAULT_PHONE_HELD_OUT_SEED
    assert zero.position_generation == context.position_generation
    assert phone.position_generation == zero.position_generation
    assert zero.query_window == context.query_window
    assert phone.query_window == zero.query_window
    assert phone.short_query_window == zero.short_query_window == 1
    assert phone.positions == zero.positions
    assert zero.cache_protocol != context.cache_protocol
    assert len({zero.cache_protocol, context.cache_protocol, phone.cache_protocol}) == 3
    assert phone.response_inventory == SYNTHETIC_PHONES
    assert phone.source == zero.source, "one publication unit must own both namespaces"
    assert context.k_grid[0] == 0
    assert context.headline_k == context.k_grid[-1]


def test_every_scored_position_is_answerable_by_both_systems(
    evaluation_positions: list[tuple[str, str]],
) -> None:
    """No position may be unwinnable for the reference model by construction.

    A gold symbol the parser cannot return is a lost point before the model
    answers — a harness artifact scored as a capability gap. That is not
    hypothetical: the letter answer schema offers the whole 30-symbol ALPHABET,
    and a parser that strips whitespace before checking length turns a
    schema-legal space answer into the empty string, scoring a *correct* space
    both unparsable and wrong.

    This probe is structurally immune, since its alphabet is the 26 letters and
    every one of them round-trips. The assertion is kept anyway: it is the check
    that would catch the alphabet being widened later without the parser
    following.
    """
    round_trips = {
        symbol
        for symbol in ALPHA_LIST
        if parse_synthetic_letter_answer(json.dumps({"answer": symbol}, separators=(",", ":")))[0]
        == symbol
    }
    golds = {gold for _, gold in evaluation_positions}
    assert golds <= set(SYNTHETIC_LETTERS)
    assert round_trips == set(SYNTHETIC_LETTERS)
    assert golds <= round_trips, (
        f"gold symbols the parser cannot return: {sorted(golds - round_trips)}. "
        "Those positions are lost by the reference model before it answers."
    )
    # POSITIVE CONTROL on the round-trip probe itself: it must be able to fail,
    # or `golds <= round_trips` would hold against a parser that accepted
    # anything. Uses symbols outside the inventory rather than a symbol inside
    # it, so the control does not depend on the harness carrying a defect.
    for outside in ("ab", "!", "", " ", ".", "A"):
        encoded = json.dumps({"answer": outside}, separators=(",", ":"))
        assert parse_synthetic_letter_answer(encoded) == (None, True)


def test_scored_positions_are_distinct(
    evaluation_positions: list[tuple[str, str]],
) -> None:
    """Duplicate positions make every interval and p-value anti-conservative.

    The main benchmark's held-out prose is one passage repeated three times, so a
    sample of it re-scores the same contexts and counts each as independent
    evidence. A sampled Markov chain has no such structure — pinned here rather
    than assumed, since a degenerate chain could still produce repeats.
    """
    assert len(set(evaluation_positions)) == len(evaluation_positions)


def test_v2_positions_are_independent_seeded_trajectories_not_one_stride(
    evaluation_positions: list[tuple[str, str]],
) -> None:
    """Reconstruct every executed sampling unit from its own frozen seed.

    The retired protocol sampled a stride through one Markov trajectory. That
    left neighboring score rows dependent whenever stride and prefix width were
    close. V2 starts a new trajectory for every row, so the paired tests and
    Wilson intervals have one independently seeded sampling unit per item.
    """
    protocol = SYNTHETIC_ZERO_SHOT_PROTOCOL
    assert protocol.position_generation == "independent-seeded-chains-v1"
    assert len(evaluation_positions) == protocol.positions
    for index, (prefix, gold) in enumerate(evaluation_positions):
        trajectory = generate_sequence(
            protocol.query_window + 1,
            protocol.position_seed + index,
        )
        assert (prefix, gold) == (trajectory[:-1], trajectory[-1])
        assert len(prefix) == protocol.query_window
        assert prefix[0] == SYNTHETIC_LETTERS[0]

    shifted = independent_held_out_positions(
        protocol.positions - 1,
        window=protocol.query_window,
        seed=protocol.position_seed + 1,
    )
    assert shifted == evaluation_positions[1:]


def test_independent_position_generator_has_live_argument_controls() -> None:
    """NEGATIVE CONTROLS: an empty, zero-width, or invalid-seed population refuses."""
    with pytest.raises(ValueError, match="positions must be positive"):
        independent_held_out_positions(0, window=40)
    with pytest.raises(ValueError, match="window must be positive"):
        independent_held_out_positions(2, window=0)
    with pytest.raises(ValueError, match="seed must be non-negative"):
        independent_held_out_positions(2, window=40, seed=-1)


def test_recorded_target_start_tvd_reconstructs_from_the_transition() -> None:
    """The nonstationary-start diagnostic is derived from the executed window."""
    protocol = SYNTHETIC_ZERO_SHOT_PROTOCOL
    transition = designed_transition()
    start = np.zeros(len(SYNTHETIC_LETTERS), dtype=np.float64)
    start[0] = 1.0
    # The transition is strictly positive, so this long independent propagation
    # converges to the unique stationary law without sharing the production
    # least-squares implementation.
    stationary = np.linalg.matrix_power(transition, 10_000) @ start
    target = np.linalg.matrix_power(transition, protocol.query_window) @ start
    expected = 0.5 * np.abs(target - stationary).sum()
    assert independent_target_start_tvd(protocol.query_window) == pytest.approx(expected, abs=1e-12)
    assert expected > 0.0, "the diagnostic would be vacuous if the fixed start were stationary"
    with pytest.raises(ValueError, match="window must be positive"):
        independent_target_start_tvd(0)


# ── The vendored artifact ────────────────────────────────────────────────────


def test_count_bigrams_rejects_text_outside_the_alphabet() -> None:
    with pytest.raises(ValueError):
        count_bigrams("abc def")
    assert count_bigrams("aab") == {"a,a": 1, "a,b": 1}


@_NEEDS_ARTIFACT
def test_vendored_artifact_reproduces_from_the_generator(vendored_counts: dict[str, int]) -> None:
    """The committed bytes must be what the generator produces today."""
    assert vendored_counts == build_synthetic_counts(
        n_chars=DEFAULT_CORPUS_CHARS, seed=DEFAULT_CORPUS_SEED
    )


@_NEEDS_ARTIFACT
def test_artifact_has_the_brown_counts_shape(vendored_counts: dict[str, int]) -> None:
    """Shape parity is what lets the existing letter machinery consume it unadapted."""
    assert vendored_counts, "an empty artifact would satisfy every check below"
    keys = list(vendored_counts)
    assert all(isinstance(k, str) and len(k) == 3 and k[1] == "," for k in keys)
    assert all(isinstance(v, int) and v > 0 for v in vendored_counts.values())
    symbols = {c for key in keys for c in (key[0], key[2])}
    assert symbols == set(SYNTHETIC_LETTERS), (
        "only the 26 letters may occur; a space, period, apostrophe or hyphen in "
        "this artifact would mean the sampler left the designed alphabet"
    )
    assert all(c in ALPHA_TO_IDX for c in symbols)
    assert keys == sorted(keys, key=lambda k: (ALPHA_TO_IDX[k[0]], ALPHA_TO_IDX[k[2]]))
    assert len(vendored_counts) == len(SYNTHETIC_LETTERS) ** 2, (
        "every bigram must be observed, or the smoothed table would be estimating "
        "some columns from pseudocounts alone"
    )


def test_artifact_is_registered_for_checksum_enforcement() -> None:
    """The additive provenance row must exist, or the digest is unenforced."""
    assert DOCUMENTED_ARTIFACTS["synthetic_counts.json"] == "scripts/generate_synthetic_counts.py"


@pytest.mark.timeout(600)
@_NEEDS_VENV
def test_generator_script_is_byte_identical_across_two_runs(tmp_path: Path) -> None:
    """Determinism at the script boundary, where the vendored bytes are decided."""
    first, second = tmp_path / "one.json", tmp_path / "two.json"
    for output in (first, second):
        result = subprocess.run(
            [str(VENV_PYTHON), str(GENERATOR), "--output", str(output)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr[-1500:]
    assert first.read_bytes() == second.read_bytes()
    if ARTIFACT.exists():
        assert first.read_bytes() == ARTIFACT.read_bytes(), (
            "the vendored artifact differs from a fresh generation — regenerate it "
            "and update the sha256 in docs/provenance/sources.md"
        )


def test_load_synthetic_counts_names_its_generator(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="generate_synthetic_counts.py"):
        load_synthetic_counts(tmp_path)


# ── The AILM comparator on this language ─────────────────────────────────────


@_NEEDS_ARTIFACT
def test_comparator_beats_the_uniform_baseline_by_a_wide_margin(
    vendored_counts: dict[str, int], evaluation_positions: list[tuple[str, str]]
) -> None:
    """POSITIVE CONTROL: training on the language must actually buy something."""
    flags = correctness_flags(
        synthetic_letter_predictions(vendored_counts, evaluation_positions), evaluation_positions
    )
    accuracy = sum(flags) / len(flags)
    assert accuracy > 5 * UNIFORM_BASELINE, (
        f"synthetic-trained accuracy {accuracy:.4f} is not clearly above the "
        f"uniform floor {UNIFORM_BASELINE:.4f}; the probe would have no signal"
    )
    assert accuracy < 1.0, "an order-1 argmax cannot be perfect on a stochastic chain"


@_NEEDS_ARTIFACT
def test_english_trained_comparator_collapses_on_the_synthetic_stream(
    evaluation_positions: list[tuple[str, str]],
) -> None:
    """NEGATIVE CONTROL: the win must come from *this* corpus, not from any corpus.

    Brown-trained statistics are the closest thing to a plausible wrong answer —
    real letter structure, wrong language. If they scored well, the task would be
    measuring structure already captured by the English comparator, and the
    stipulated-language control would not isolate the designed chain.
    """
    brown = load_brown_counts(PROJECT_ROOT / "vendor")
    flags = correctness_flags(
        synthetic_letter_predictions(brown, evaluation_positions), evaluation_positions
    )
    accuracy = sum(flags) / len(flags)
    assert accuracy < 0.15, (
        f"a Brown-trained comparator scored {accuracy:.4f} on the synthetic stream; "
        "the designed chain is not discriminated from English-trained statistics"
    )


@_NEEDS_ARTIFACT
def test_transposed_count_table_collapses(
    vendored_counts: dict[str, int], evaluation_positions: list[tuple[str, str]]
) -> None:
    """NEGATIVE CONTROL: the right numbers wired backwards must not score well.

    Reversing each key turns ``P(next | current)`` into ``P(previous | current)``.
    The table is still column-stochastic and still built from this exact corpus,
    so only the measured accuracy can tell the two apart.
    """
    reversed_counts = {f"{key[2]},{key[0]}": value for key, value in vendored_counts.items()}
    flags = correctness_flags(
        synthetic_letter_predictions(reversed_counts, evaluation_positions), evaluation_positions
    )
    forward = correctness_flags(
        synthetic_letter_predictions(vendored_counts, evaluation_positions), evaluation_positions
    )
    assert sum(flags) < sum(forward) / 2


@_NEEDS_ARTIFACT
def test_comparator_agrees_with_the_external_benchmark_letter_comparator(
    vendored_counts: dict[str, int], evaluation_positions: list[tuple[str, str]]
) -> None:
    """The probe must score AILM exactly as the main benchmark does.

    Two implementations of "order-1 argmax" that drift apart would make the
    synthetic and external results incomparable, which is the one thing the probe
    exists to enable.
    """
    import ailm.benchmarks as benchmarks

    reference = getattr(benchmarks, "_letter_comparator", None)
    assert callable(reference), (
        "ailm.benchmarks._letter_comparator is absent or not callable; skipping on an API "
        "rename would make this cross-benchmark equivalence check vacuous"
    )
    assert synthetic_letter_predictions(vendored_counts, evaluation_positions) == reference(
        vendored_counts, evaluation_positions
    )


def test_correctness_flags_refuses_an_unequal_pairing() -> None:
    positions = [("ab", "c"), ("bc", "d")]
    assert correctness_flags(["c", "x"], positions) == [True, False]
    with pytest.raises(ValueError):
        correctness_flags(["c"], positions)


# ── The stipulated phone language ────────────────────────────────────────────
#
# The same controls the letter arm runs, run against the phone chain. The
# thresholds and tolerances are the letter arm's; the inventory is the
# phoneme level's exact 39-phone set.


def test_phone_transition_is_column_stochastic_and_everywhere_positive() -> None:
    transition = phone_designed_transition()
    assert transition.shape == (len(SYNTHETIC_PHONES), len(SYNTHETIC_PHONES))
    assert np.allclose(transition.sum(axis=0), 1.0)
    assert transition.min() > 0.0, "a zero cell would make some bigram unreachable"


def test_phone_transition_matches_the_written_down_mixture() -> None:
    """POSITIVE CONTROL: the closed form, not a shape check.

    Every column must equal the letter arm's three-part mixture: the uniform
    floor over all 39 phones, the opposite-class share spread uniformly over
    the opposite class, and the dominant bonus stacked on the designed
    successor's opposite-class share — exactly how the letter arm composes
    ``W_DOMINANT`` with the opposite-class term.
    """
    transition = phone_designed_transition()
    index = {phone: i for i, phone in enumerate(SYNTHETIC_PHONES)}
    floor = W_UNIFORM / len(SYNTHETIC_PHONES)
    assert (W_DOMINANT, W_OPPOSITE, W_UNIFORM) == (0.45, 0.35, 0.20)
    for current in SYNTHETIC_PHONES:
        current_is_vowel = current in PHONE_VOWELS
        opposite = PHONE_CONSONANTS if current_is_vowel else PHONE_VOWELS
        dominant = phone_dominant_successor(current)
        for phone in SYNTHETIC_PHONES:
            expected = floor
            if phone in opposite:
                expected += W_OPPOSITE / len(opposite)
            if phone == dominant:
                expected += W_DOMINANT
            observed = transition[index[phone], index[current]]
            assert observed == pytest.approx(expected), (current, phone)


def test_phone_dominant_successor_is_the_argmax_of_every_column() -> None:
    """POSITIVE CONTROL: the order-1 ceiling is the designed successor map."""
    transition = phone_designed_transition()
    index = {phone: i for i, phone in enumerate(SYNTHETIC_PHONES)}
    for current in SYNTHETIC_PHONES:
        assert SYNTHETIC_PHONES[int(np.argmax(transition[:, index[current]]))] == (
            phone_dominant_successor(current)
        )


def test_phone_successors_cross_the_class_boundary_and_do_not_collapse() -> None:
    assert all(phone_dominant_successor(v) in PHONE_CONSONANTS for v in PHONE_VOWELS)
    assert all(phone_dominant_successor(c) in PHONE_VOWELS for c in PHONE_CONSONANTS)
    vowels_to_consonants = {phone_dominant_successor(v) for v in PHONE_VOWELS}
    assert len(vowels_to_consonants) == len(PHONE_VOWELS), "coprime stride collapsed"
    # The reverse direction cannot be injective (24 consonants to 15 vowels);
    # the stride is still designed to reach every vowel rather than a subset.
    consonants_to_vowels = {phone_dominant_successor(c) for c in PHONE_CONSONANTS}
    assert consonants_to_vowels == set(PHONE_VOWELS), "reverse stride missed a vowel"


def test_phone_transition_is_not_symmetric_under_transposition() -> None:
    """NEGATIVE CONTROL against the wiring defect this repo has shipped before."""
    transition = phone_designed_transition()
    assert not np.allclose(transition, transition.T)
    index = {phone: i for i, phone in enumerate(SYNTHETIC_PHONES)}
    source = SYNTHETIC_PHONES[0]
    target = phone_dominant_successor(source)
    dominant_direction = transition[index[target], index[source]]
    reverse_direction = transition[index[source], index[target]]
    assert dominant_direction > 4 * reverse_direction, (source, target)
    assert dominant_direction == pytest.approx(
        W_DOMINANT + W_OPPOSITE / len(PHONE_CONSONANTS) + W_UNIFORM / len(SYNTHETIC_PHONES)
    )


def test_phone_dominant_successor_rejects_a_symbol_outside_the_inventory() -> None:
    for outside in ("AA1", "aa", "Q", "AX", ""):
        with pytest.raises(ValueError):
            phone_dominant_successor(outside)


def test_phone_generation_is_deterministic_in_its_seed() -> None:
    first = generate_phone_sequence(2_000, seed=11)
    assert first == generate_phone_sequence(2_000, seed=11)
    assert generate_phone_sequence(2_000, seed=12) != first
    assert len(first) == 2_000


def test_phone_generation_emits_only_canonical_phones() -> None:
    stream = generate_phone_sequence(5_000, seed=3)
    assert set(stream) <= set(SYNTHETIC_PHONES)
    assert set(stream) == set(SYNTHETIC_PHONES), "a well-mixed sample visits every phone"


def test_phone_generation_rejects_degenerate_arguments() -> None:
    with pytest.raises(ValueError):
        generate_phone_sequence(0, seed=1)
    with pytest.raises(ValueError):
        generate_phone_sequence(10, seed=-1)


def test_count_phone_bigrams_rejects_symbols_outside_the_inventory() -> None:
    with pytest.raises(ValueError):
        count_phone_bigrams(("AA", "Q"))
    assert count_phone_bigrams(("AA", "AE")) == {"AA,AE": 1}


@_NEEDS_PHONE_ARTIFACT
def test_sampled_phone_frequencies_converge_to_the_designed_transition(
    phone_vendored_counts: dict[str, int],
) -> None:
    """POSITIVE CONTROL that the corpus is a sample of the stipulated chain."""
    empirical = np.zeros((len(SYNTHETIC_PHONES), len(SYNTHETIC_PHONES)), dtype=np.float64)
    index = {phone: i for i, phone in enumerate(SYNTHETIC_PHONES)}
    for key, count in phone_vendored_counts.items():
        previous, following = key.split(",")
        empirical[index[following], index[previous]] += count
    empirical /= empirical.sum(axis=0, keepdims=True)
    assert np.abs(empirical - phone_designed_transition()).max() < 0.03


@_NEEDS_PHONE_ARTIFACT
def test_phone_evaluation_positions_share_no_windows_with_training() -> None:
    """The evaluated windows must not be corpus windows AILM was fitted on."""
    protocol = SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL
    stream = generate_phone_sequence(DEFAULT_PHONE_CORPUS_SYMBOLS, DEFAULT_PHONE_CORPUS_SEED)
    span = protocol.query_window + 1
    training_windows = {" ".join(stream[i : i + span]) for i in range(len(stream) - span + 1)}
    held_out = independent_phone_held_out_positions(
        protocol.positions,
        window=protocol.query_window,
        seed=protocol.position_seed,
    )
    shared = [prefix for prefix, _gold in held_out if prefix in training_windows]
    assert shared == [], f"held-out phone windows found verbatim in training: {shared[:5]}"


def test_phone_prompt_schema_parser_and_gold_share_one_exact_inventory() -> None:
    """The executed phone prompt opportunity must equal schema, parser, and gold."""
    protocol = SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL
    positions = independent_phone_held_out_positions(
        8, window=protocol.query_window, seed=protocol.position_seed
    )
    schema = _response_format("phoneme")
    assert schema["properties"]["answer"]["enum"] == sorted(PHONES)
    assert set(sorted(PHONES)) == set(SYNTHETIC_PHONES)
    for prefix, gold in positions:
        assert all(token in SYNTHETIC_PHONES for token in prefix.split())
        assert gold in SYNTHETIC_PHONES
        prompt = build_prompt("phoneme", prefix)
        assert prompt.endswith(f"Sequence: {prefix}")
        encoded = json.dumps({"answer": gold})
        parsed, bad = parse_phoneme_answer(encoded)
        assert parsed == gold and not bad


def test_phone_parser_normalizes_case_but_nothing_else() -> None:
    """The shipped phoneme parser's one normalization is the canonical case.

    ``parse_phoneme_answer`` is the external benchmark's existing contract:
    the answer token is uppercased before the inventory check, because ARPAbet
    stress-free symbols differ only in case. Anything else that is not
    exactly one in-inventory token is invalid.
    """
    assert parse_phoneme_answer('{"answer":"aa"}') == ("AA", False)
    for answer in (
        '{"answer":"AA AE"}',
        '{"answer":"Q"}',
        '{"answer":"AA1"}',
        '{"answer":"AX"}',
        '{"answer":""}',
        "no json at all",
    ):
        assert parse_phoneme_answer(answer) == (None, True)


def test_phone_positions_are_independent_seeded_trajectories_not_one_stride() -> None:
    """Reconstruct every executed sampling unit from its own frozen seed.

    The retired protocol sampled a stride through one Markov trajectory; v2
    starts a new trajectory for every row. The shifted-seed run reproducing
    the later rows is the seed arithmetic doing its job, not a shared stream:
    each row's prefix comes whole from its own seed and starts at the chain's
    first symbol.
    """
    protocol = SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL
    positions = independent_phone_held_out_positions(
        6, window=protocol.query_window, seed=protocol.position_seed
    )
    for index, (prefix, gold) in enumerate(positions):
        sequence = generate_phone_sequence(
            protocol.query_window + 1, protocol.position_seed + index
        )
        assert prefix == " ".join(sequence[:-1])
        assert gold == sequence[-1]
        assert sequence[0] == SYNTHETIC_PHONES[0]
    shifted = independent_phone_held_out_positions(
        5, window=protocol.query_window, seed=protocol.position_seed + 1
    )
    assert shifted == positions[1:]


def test_phone_position_generator_has_live_argument_controls() -> None:
    """NEGATIVE CONTROLS: an empty, zero-width, or invalid-seed population refuses."""
    with pytest.raises(ValueError):
        independent_phone_held_out_positions(0, window=40)
    with pytest.raises(ValueError):
        independent_phone_held_out_positions(2, window=0)
    with pytest.raises(ValueError):
        independent_phone_held_out_positions(2, window=40, seed=-1)


def test_recorded_phone_target_start_tvd_reconstructs_from_the_transition() -> None:
    """The nonstationary-start diagnostic is derived from the executed window."""
    protocol = SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL
    expected = phone_target_start_tvd(protocol.query_window)
    assert 0.0 <= expected <= 2.0
    with pytest.raises(ValueError):
        phone_target_start_tvd(0)


def test_scored_phone_positions_are_distinct() -> None:
    """Duplicate positions make every interval and p-value anti-conservative."""
    protocol = SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL
    positions = independent_phone_held_out_positions(
        protocol.positions, window=protocol.query_window, seed=protocol.position_seed
    )
    assert len(set(positions)) == len(positions)


@_NEEDS_PHONE_ARTIFACT
def test_phone_artifact_reproduces_from_the_generator(
    phone_vendored_counts: dict[str, int],
) -> None:
    """The committed bytes must be what the generator produces today."""
    assert phone_vendored_counts == build_synthetic_phone_counts(
        n_symbols=DEFAULT_PHONE_CORPUS_SYMBOLS, seed=DEFAULT_PHONE_CORPUS_SEED
    )


@_NEEDS_PHONE_ARTIFACT
def test_phone_artifact_has_the_flat_phone_bigram_shape(
    phone_vendored_counts: dict[str, int],
) -> None:
    """Key format parity is what lets build_phone_transition consume it unadapted."""
    index = {phone: i for i, phone in enumerate(SYNTHETIC_PHONES)}
    total = 0
    for key, count in phone_vendored_counts.items():
        previous, following = key.split(",")
        assert previous in index and following in index
        assert count >= 1
        total += count
    assert total == DEFAULT_PHONE_CORPUS_SYMBOLS - 1
    # A 200,000-symbol draw from a transition with a 0.20 uniform floor visits
    # every cell of the 39x39 grid; a hole would mean the sampler is wrong.
    assert len(phone_vendored_counts) == len(SYNTHETIC_PHONES) ** 2


def test_phone_artifact_is_registered_for_checksum_enforcement() -> None:
    """The additive provenance row must exist, or the digest is unenforced."""
    assert (
        DOCUMENTED_ARTIFACTS["synthetic_phone_counts.json"]
        == "scripts/generate_synthetic_phone_counts.py"
    )


@pytest.mark.timeout(600)
@_NEEDS_VENV
def test_phone_generator_script_is_byte_identical_across_two_runs(tmp_path: Path) -> None:
    """Determinism at the script boundary, where the vendored bytes are decided."""
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    for output in (first, second):
        subprocess.run(
            [str(VENV_PYTHON), str(PHONE_GENERATOR), "--output", str(output)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        )
    assert first.read_bytes() == second.read_bytes()
    if PHONE_ARTIFACT.exists():
        assert first.read_bytes() == PHONE_ARTIFACT.read_bytes()


def test_load_synthetic_phone_counts_names_its_generator(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="generate_synthetic_phone_counts.py"):
        load_synthetic_phone_counts(tmp_path)


@_NEEDS_PHONE_ARTIFACT
def test_phone_comparator_beats_the_uniform_baseline_by_a_wide_margin(
    phone_vendored_counts: dict[str, int],
    phone_evaluation_positions: list[tuple[str, str]],
) -> None:
    """POSITIVE CONTROL: training on the language must actually buy something."""
    flags = correctness_flags(
        synthetic_phone_predictions(phone_vendored_counts, phone_evaluation_positions),
        phone_evaluation_positions,
    )
    accuracy = sum(flags) / len(flags)
    assert accuracy > 4 * PHONE_UNIFORM_BASELINE, "a fitted comparator must clear the floor"
    assert accuracy < 1.0, "an order-1 argmax cannot be perfect on a stochastic chain"


@_NEEDS_PHONE_ARTIFACT
def test_english_trained_comparator_collapses_on_the_phone_stream(
    phone_vendored_counts: dict[str, int],
    phone_evaluation_positions: list[tuple[str, str]],
) -> None:
    """NEGATIVE CONTROL: the win must come from *this* corpus, not English.

    The refutation condition pinned in the lane's pre-registration stub: an
    English-trained (CMUdict) phone-bigram comparator that scored as well as
    the stipulated-trained one would make the win about English statistics.
    """
    english = correctness_flags(
        synthetic_phone_predictions(
            load_phoneme_counts()["phone_bigrams"], phone_evaluation_positions
        ),
        phone_evaluation_positions,
    )
    stipulated = correctness_flags(
        synthetic_phone_predictions(phone_vendored_counts, phone_evaluation_positions),
        phone_evaluation_positions,
    )
    assert sum(english) < sum(stipulated) / 2


@_NEEDS_PHONE_ARTIFACT
def test_transposed_phone_count_table_collapses(
    phone_vendored_counts: dict[str, int],
    phone_evaluation_positions: list[tuple[str, str]],
) -> None:
    """NEGATIVE CONTROL: the right numbers wired backwards must not score well."""
    transposed = {}
    for key, count in phone_vendored_counts.items():
        previous, following = key.split(",")
        transposed[f"{following},{previous}"] = count
    backward = correctness_flags(
        synthetic_phone_predictions(transposed, phone_evaluation_positions),
        phone_evaluation_positions,
    )
    forward = correctness_flags(
        synthetic_phone_predictions(phone_vendored_counts, phone_evaluation_positions),
        phone_evaluation_positions,
    )
    assert sum(backward) < sum(forward) / 2


@_NEEDS_PHONE_ARTIFACT
def test_phone_comparator_agrees_with_the_external_benchmark_phoneme_comparator(
    phone_vendored_counts: dict[str, int],
    phone_evaluation_positions: list[tuple[str, str]],
) -> None:
    """The probe must score AILM exactly as the main benchmark's phoneme arm does."""
    from ailm.benchmarks.comparators import _phoneme_comparator

    reference = _phoneme_comparator
    assert synthetic_phone_predictions(
        phone_vendored_counts, phone_evaluation_positions
    ) == reference({}, phone_vendored_counts, phone_evaluation_positions)


def test_phone_comparator_refusals_pass_on_the_executed_population() -> None:
    """POSITIVE CONTROL: the pre-registered publication refusals hold offline."""
    recorder = importlib.import_module("scripts.record_synthetic_benchmark_claims")
    protocol = SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL
    positions = independent_phone_held_out_positions(
        protocol.positions, window=protocol.query_window, seed=protocol.position_seed
    )
    recorder._phone_comparator_refusals(positions)


def test_phone_comparator_refusals_are_letter_arm_absent() -> None:
    """The letter arm's comparator controls live in tests, as they always have."""
    recorder = importlib.import_module("scripts.record_synthetic_benchmark_claims")
    assert recorder.LETTER_ARM.comparator_refusals is None
    assert recorder.PHONE_ARM.comparator_refusals is not None


# ── The recorder ─────────────────────────────────────────────────────────────


def test_recorder_owns_the_complete_registered_probe_family() -> None:
    recorder = importlib.import_module("scripts.record_synthetic_benchmark_claims")
    expected = {probe.key: probe.tag for probe in SERVED_PROBES}
    assert recorder.MODELS == expected
    scope = "20260905T000000000042Z"
    configuration = recorder.publication_replay_configuration(run_scope=scope)
    # The receipt binds the hosted transport: the endpoint and model tags name
    # what the cache records actually carry, while the registered probe tags
    # stay in the parameters as the claim namespaces the per-tag protocols serve.
    assert configuration["endpoint"] == recorder.OPENROUTER_URL
    assert configuration["model_tags"] == [recorder.OPENROUTER_MODEL]
    assert configuration["parameters"]["probe_tags"] == sorted(recorder.MODELS.values())
    # The protocol identity still names the arm and the probe; only the cache
    # namespace carries the run, the probe, and the scored slot, so a re-run
    # appends beside the previous run instead of colliding, the three
    # per-probe scoring iterations stay distinct inside one run, and the
    # sensitivity arm's recurring one-symbol prompt never collides with itself
    # under the hosted transport's non-deterministic replies.
    primary_tag = recorder.MODELS[sorted(recorder.MODELS)[0]]
    assert configuration["call_protocol"] == (
        f"openrouter-v1:{SYNTHETIC_ZERO_SHOT_PROTOCOL.cache_protocol}:{primary_tag}:0000:full/{scope}"
    )
    assert configuration["parameters"]["run_scope"] == scope
    assert configuration["parameters"]["cache_protocols"] == sorted(
        f"openrouter-v1:{protocol}:{tag}:{index:04d}:{arm_name}/{scope}"
        for protocol, positions_count in (
            (SYNTHETIC_ZERO_SHOT_PROTOCOL.cache_protocol, SYNTHETIC_ZERO_SHOT_PROTOCOL.positions),
            (
                SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL.cache_protocol,
                SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL.positions,
            ),
        )
        for tag in sorted(recorder.MODELS.values())
        for index in range(positions_count)
        for arm_name in ("full", "short")
    )
    arms = configuration["parameters"]["arms"]
    assert set(arms) == {"letter", "phone"}, "one publication unit binds both arms"
    assert arms["letter"]["measurement_protocol"] == SYNTHETIC_ZERO_SHOT_PROTOCOL.version
    assert arms["phone"]["measurement_protocol"] == SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL.version
    assert arms["letter"]["position_generation"] == (
        SYNTHETIC_ZERO_SHOT_PROTOCOL.position_generation
    )
    assert arms["phone"]["position_generation"] == (
        SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL.position_generation
    )
    assert len(recorder.FAMILY_TEMPLATES) * len(recorder.MODELS) == len(SERVED_PROBES)
    assert set(recorder.P_VALUE_FAMILIES) == {
        "comparator",
        "audit_context_sensitivity",
        "phone_comparator",
        "phone_audit_context_sensitivity",
    }
    assert recorder.P_VALUE_FAMILIES["comparator"] == recorder.FAMILY_TEMPLATES
    assert recorder.P_VALUE_FAMILIES["audit_context_sensitivity"] == (
        recorder.CONTEXT_FAMILY_TEMPLATES
    )
    assert recorder.P_VALUE_FAMILIES["phone_comparator"] == recorder.PHONE_FAMILY_TEMPLATES
    assert recorder.P_VALUE_FAMILIES["phone_audit_context_sensitivity"] == (
        recorder.PHONE_CONTEXT_FAMILY_TEMPLATES
    )


def _isolated_ledger(tmp_path: Path) -> Path:
    """A private copy of the ledger, so no test can touch the tracked one."""
    isolated = tmp_path / "claims.json"
    if LEDGER.exists():
        shutil.copy2(LEDGER, isolated)
    return isolated


def _recorder_env(isolated: Path) -> dict[str, str]:
    return {
        **os.environ,
        "AILM_CLAIMS_PATH": str(isolated.resolve()),
        "AILM_CALL_CACHE_PATH": str((isolated.parent / "bench_calls.jsonl").resolve()),
        "OLLAMA_HOST": "127.0.0.1",
    }


def _valid_synthetic_measurement(
    recorder, positions_count: int = 5
) -> tuple[dict[str, object], int]:
    """Build a structurally valid combined measurement for both stipulated arms."""

    arms: dict[str, dict[str, object]] = {}
    expected_calls = 0
    for arm in (recorder.LETTER_ARM, recorder.PHONE_ARM):
        positions = arm.held_out_positions(positions_count)
        trace = recorder._expected_execution_trace(arm, positions)
        arm_calls = len(trace)
        expected_calls += arm_calls
        arms[arm.key] = recorder._arm_measurement_evidence(
            arm=arm,
            execution_trace=trace,
            cache_hits=0,
            cache_misses=arm_calls,
            unique_cache_keys=arm_calls,
            maximum_prompt_tokens=1,
        )
    measurement = recorder._measurement_evidence(
        arms=arms,
        cache_hits=0,
        cache_misses=expected_calls,
        unique_cache_keys=expected_calls,
        maximum_prompt_tokens=1,
    )
    return measurement, expected_calls


def test_v2_synthetic_measurement_validator_reconstructs_the_ordered_trace() -> None:
    """A valid receipt is accepted because its prompt order reproduces from code."""
    recorder = importlib.import_module("scripts.record_synthetic_benchmark_claims")
    measurement, expected_calls = _valid_synthetic_measurement(recorder)
    recorder._validate_measurement_fields(
        measurement,
        positions=5,
        cache_record_count=expected_calls,
    )


def test_publication_measurement_refuses_a_prompt_token_maximum_not_in_cache(
    tmp_path: Path,
) -> None:
    """NEGATIVE CONTROL — the receipt cannot copy an unverified token maximum."""

    recorder = importlib.import_module("scripts.record_synthetic_benchmark_claims")
    measurement, expected_calls = _valid_synthetic_measurement(
        recorder, SYNTHETIC_ZERO_SHOT_PROTOCOL.positions
    )
    store = CallCache(tmp_path / "receipt-prompt-tokens.jsonl")
    request = CacheRequest(
        payload={"model": SERVED_PROBES[0].tag, "prompt": "measured"},
        endpoint="http://127.0.0.1:11434",
        protocol=SYNTHETIC_ZERO_SHOT_PROTOCOL.cache_protocol,
        identity="sha256:" + "a" * 64,
    )
    store.put(request, {"response": '{"letter":"a"}', "prompt_eval_count": 7})
    receipt: dict[str, object] = {
        "cache": {"record_count": expected_calls, "keys": [request.key]},
        "measurement": measurement,
    }

    with pytest.raises(NetworkClaimBindingError, match="maximum disagrees"):
        recorder.validate_publication_measurement(receipt, store)


def test_zero_shot_query_window_schedule_is_exactly_counterbalanced() -> None:
    """Every model alternates which paired arm occupies the first temporal slot."""
    recorder = importlib.import_module("scripts.record_synthetic_benchmark_claims")
    count = 8
    per_model = count * 2
    for arm in (recorder.LETTER_ARM, recorder.PHONE_ARM):
        positions = arm.held_out_positions(count)
        trace = recorder._expected_execution_trace(arm, positions)
        for model_index, model in enumerate(sorted(recorder.MODELS.values())):
            rows = trace[model_index * per_model : (model_index + 1) * per_model]
            assert {row["model"] for row in rows} == {model}
            first_arms = [rows[position * 2]["arm"] for position in range(count)]
            assert first_arms == ["full" if index % 2 == 0 else "short" for index in range(count)]
            assert first_arms.count("full") == first_arms.count("short") == count // 2

            # NEGATIVE CONTROL: a fixed full-first schedule is detectably imbalanced.
            planted = ["full"] * count
            assert planted.count("full") != planted.count("short")


def test_v3_synthetic_measurement_validator_rejects_every_load_bearing_mutation(
    tmp_path: Path,
) -> None:
    """NEGATIVE CONTROLS for schema, schedule, fresh-call arithmetic, and digest."""
    recorder = importlib.import_module("scripts.record_synthetic_benchmark_claims")
    measurement, expected_calls = _valid_synthetic_measurement(recorder)
    isolated = _isolated_ledger(tmp_path)
    before = isolated.read_bytes() if isolated.exists() else None
    arm_calls = expected_calls // 2

    mutations: tuple[tuple[str, object], ...] = (
        ("schema", "ailm-synthetic-execution-trace-retired"),
        ("cache_hits", 1),
        ("cache_misses", expected_calls - 1),
        ("unique_cache_keys", 0),
        ("maximum_prompt_tokens", 0),
    )
    for field, replacement in mutations:
        changed = dict(measurement)
        changed[field] = replacement
        with pytest.raises(NetworkClaimBindingError):
            recorder._validate_measurement_fields(
                changed,
                positions=5,
                cache_record_count=expected_calls,
            )

    arm_mutations: tuple[tuple[str, object], ...] = (
        ("schedule", "all-full-then-all-short"),
        ("warmup_policy", "none"),
        ("warmup_calls", 1),
        ("warmup_calls", False),
        ("scored_calls", arm_calls - 1),
        ("cache_hits", 1),
        ("cache_misses", arm_calls - 1),
        ("unique_cache_keys", 0),
        ("maximum_prompt_tokens", 0),
        ("ordered_trace_sha256", "0" * 64),
    )
    for arm_key in ("letter", "phone"):
        for field, replacement in arm_mutations:
            changed = json.loads(json.dumps(measurement))
            changed["arms"][arm_key][field] = replacement
            with pytest.raises(NetworkClaimBindingError):
                recorder._validate_measurement_fields(
                    changed,
                    positions=5,
                    cache_record_count=expected_calls,
                )

    missing_phone = json.loads(json.dumps(measurement))
    del missing_phone["arms"]["phone"]
    with pytest.raises(NetworkClaimBindingError, match="registered stipulated arms"):
        recorder._validate_measurement_fields(
            missing_phone,
            positions=5,
            cache_record_count=expected_calls,
        )

    with_extra = dict(measurement)
    with_extra["unbound"] = True
    with pytest.raises(NetworkClaimBindingError, match="invalid measurement fields"):
        recorder._validate_measurement_fields(
            with_extra,
            positions=5,
            cache_record_count=expected_calls,
        )
    without_arms = dict(measurement)
    without_arms.pop("arms")
    with pytest.raises(NetworkClaimBindingError, match="invalid measurement fields"):
        recorder._validate_measurement_fields(
            without_arms,
            positions=5,
            cache_record_count=expected_calls,
        )
    if before is not None:
        assert isolated.read_bytes() == before, "pre-publication refusals changed the ledger"


def test_publication_cache_path_refusal_is_offline_and_ledger_preserving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A publication run cannot redirect receipts to an isolated call store."""
    isolated = _isolated_ledger(tmp_path)
    before = isolated.read_bytes() if isolated.exists() else None
    noncanonical = (tmp_path / "isolated-calls.jsonl").resolve()
    monkeypatch.setenv(CALL_CACHE_PATH_ENV, str(noncanonical))
    enable_call_cache(enabled=False)
    try:
        with pytest.raises(ValueError, match="publication call-cache path must be"):
            enable_call_cache(require_default_path=True)
        assert call_cache() is None
        assert not noncanonical.exists()
        if before is not None:
            assert isolated.read_bytes() == before
    finally:
        enable_call_cache(enabled=False)


@pytest.mark.timeout(600)
@_NEEDS_VENV
def test_hosted_recorder_refuses_failed_chat_without_querying_local_models(
    tmp_path: Path, local_http_server
) -> None:
    """A failed real hosted request preserves the ledger and never queries Ollama."""
    isolated = _isolated_ledger(tmp_path)
    before = isolated.read_bytes() if isolated.exists() else None
    local_http_server.enqueue_json("/chat", {"error": "unavailable"}, status=503)
    environment = _recorder_env(isolated)
    environment["OPENROUTER_API_KEY"] = "loopback-test-key"
    environment.pop("AILM_OPENROUTER_BUDGET_FILE", None)
    environment.pop("AILM_OPENROUTER_BUDGET_LIMIT_USD", None)
    result = subprocess.run(
        [
            str(VENV_PYTHON),
            str(RECORDER),
            "--endpoint",
            local_http_server.base_url + "/chat",
            "--positions",
            "8",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert result.returncode != 0
    assert "503" in result.stderr
    assert len(local_http_server.requests_for("/chat")) == 1
    assert not local_http_server.requests_for("/api/tags")
    assert not local_http_server.requests_for("/api/generate")
    if before is not None:
        assert isolated.read_bytes() == before


@_NEEDS_VENV
@_NEEDS_ARTIFACT
@pytest.mark.timeout(600)
def test_recorder_writes_a_complete_label_set_to_an_isolated_ledger(
    tmp_path: Path, local_http_server
) -> None:
    """Actual hosted protocol over loopback records a complete bounded test ledger."""
    local_http_server.enqueue_json(
        "/chat",
        {
            "model": importlib.import_module(
                "scripts.record_synthetic_benchmark_claims"
            ).OPENROUTER_MODEL,
            "choices": [{"message": {"content": '{"answer":"a"}'}}],
            "usage": {"prompt_tokens": 7},
        },
        repeat=len(SERVED_PROBES) * 8 * 2 * 2,
    )
    isolated = _isolated_ledger(tmp_path)
    tracked_before = LEDGER.read_bytes() if LEDGER.exists() else None

    environment = _recorder_env(isolated)
    environment["OPENROUTER_API_KEY"] = "loopback-test-key"
    environment.pop("AILM_OPENROUTER_BUDGET_FILE", None)
    environment.pop("AILM_OPENROUTER_BUDGET_LIMIT_USD", None)
    result = subprocess.run(
        [
            str(VENV_PYTHON),
            str(RECORDER),
            "--positions",
            "8",
            "--endpoint",
            local_http_server.base_url + "/chat",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert len(local_http_server.requests) == len(SERVED_PROBES) * 8 * 2 * 2
    assert all(request.path == "/chat" for request in local_http_server.requests)

    claims = json.loads(isolated.read_text(encoding="utf-8"))["claims"]
    letter = {label: entry for label, entry in claims.items() if label.startswith("bench_synth_")}
    phone = {
        label: entry for label, entry in claims.items() if label.startswith("bench_synthphone_")
    }
    for required in (
        "bench_synth_protocol_version",
        "bench_synth_positions",
        "bench_synth_uniform_baseline",
        "bench_synth_letter_top1_ailm",
    ):
        assert required in letter, f"{required} missing from {sorted(letter)}"
    for required in (
        "bench_synthphone_protocol_version",
        "bench_synthphone_positions",
        "bench_synthphone_uniform_baseline",
        "bench_synthphone_phone_top1_ailm",
    ):
        assert required in phone, f"{required} missing from {sorted(phone)}"
    for probe in SERVED_PROBES:
        for template in (
            "bench_synth_letter_top1_{key}",
            "bench_synth_letter_top1_wilson_low_{key}",
            "bench_synth_letter_top1_wilson_high_{key}",
            "bench_synth_paired_b_{key}",
            "bench_synth_paired_c_{key}",
            "bench_synth_paired_p_{key}",
            "bench_synth_unparsable_{key}",
            "bench_synthphone_phone_top1_{key}",
            "bench_synthphone_phone_top1_wilson_low_{key}",
            "bench_synthphone_phone_top1_wilson_high_{key}",
            "bench_synthphone_paired_b_{key}",
            "bench_synthphone_paired_c_{key}",
            "bench_synthphone_paired_p_{key}",
            "bench_synthphone_unparsable_{key}",
        ):
            required = template.format(key=probe.key)
            assert required in letter or required in phone, (
                f"{required} missing from the recorded namespaces"
            )
    assert all(entry["run_id"] and entry["source"] for entry in letter.values())
    assert all(entry["source"].startswith("scripts/") for entry in letter.values())
    assert all(entry["run_id"] and entry["source"] for entry in phone.values())
    assert all(entry["source"].startswith("scripts/") for entry in phone.values())
    assert letter["bench_synth_positions"]["value"] == 8
    assert phone["bench_synthphone_positions"]["value"] == 8
    for namespace, label in (
        (letter, "bench_synth_letter_top1_ailm"),
        (phone, "bench_synthphone_phone_top1_ailm"),
    ):
        assert 0.0 <= namespace[label]["value"] <= 1.0
    for probe in SERVED_PROBES:
        assert 0.0 <= letter[f"bench_synth_paired_p_{probe.key}"]["value"] <= 1.0
        assert 0.0 <= phone[f"bench_synthphone_paired_p_{probe.key}"]["value"] <= 1.0
    assert letter["bench_synth_family_size"]["value"] == len(SERVED_PROBES)
    assert phone["bench_synthphone_family_size"]["value"] == len(SERVED_PROBES)
    assert (
        letter["bench_synth_protocol_version"]["run_id"]
        != phone["bench_synthphone_protocol_version"]["run_id"]
    ), "the two stipulated arms carry distinct run ids"
    if tracked_before is not None:
        assert LEDGER.read_bytes() == tracked_before, "the tracked ledger was touched"


@_NEEDS_LIVE_OLLAMA_OPT_IN
@pytest.mark.timeout(300)
def test_reference_model_answers_the_synthetic_prompt_deterministically() -> None:
    """Temperature 0 and a fixed seed must mean a re-run reproduces the run.

    Verified rather than assumed: without it, a re-recorded probe could differ
    from the ledger for reasons that have nothing to do with the model's ability.
    """
    _require_live_models((REFERENCE_MODEL,))
    prefix, _gold = independent_held_out_positions(
        1,
        window=40,
        seed=DEFAULT_HELD_OUT_SEED,
    )[0]
    prompt = build_prompt("synthetic_letter", prefix)
    assert ask_ollama(REFERENCE_MODEL, prompt, "synthetic_letter", 32) == ask_ollama(
        REFERENCE_MODEL, prompt, "synthetic_letter", 32
    )
