"""Stage 1 meta-gate: the letter-level agent must beat uniform on held-out English.

What this file gates
--------------------
1. Load ``vendor/brown_counts.json`` (skip if it has not been generated yet).
2. Build the letter agent from those Brown corpus bigram counts.
3. Measure perplexity on fixed held-out English prose never used for fitting.
4. POSITIVE CONTROL — the declared Stage 1 threshold: perplexity < 30 − 2 = 28.
5. NEGATIVE CONTROL — a uniform model must score worse than the threshold, or
   the gate is vacuous.
6. STRONG CONTROL — the agent must also beat a *context-free letter-frequency*
   model built from the same counts. This is the load-bearing gate: the declared
   28.0 threshold is only 2 units below uniform and, measured, it also admits the
   anti-informative wiring (bigram in A, identity B) at 27.25. Uniform alone is a
   crippled comparator; the unigram model is the honest one.
7. ORACLE — pymdp's inference must reproduce the closed-form bigram chain, so a
   passing perplexity cannot come from inference that merely runs.
8. LEDGER — the measured value is recorded in ``data/claims.json`` so prose can
   cite ``claim:stage1_letter_perplexity`` instead of a hand-typed number.

Units caveat (kept explicit because the wording is easy to misread)
------------------------------------------------------------------
The declared gate subtracts 2 from a *perplexity*, so "≥2 nats better than
uniform" is really a 2-perplexity-unit margin. Two nats below uniform would be
30/e² ≈ 4.06. The comparisons below use the same units as the gate they check,
and the log-scale margin is reported separately in the failure messages.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from ailm.claims import load_ledger
from ailm.control import ControlHarness
from ailm.evaluation import STAGE1_EVALUATION_TEXT
from ailm.levels.letter import (
    ALPHA_LIST,
    ALPHA_TO_IDX,
    ALPHABET,
    N_SYMBOLS,
    build_letter_agent,
    build_letter_transition,
    compute_perplexity,
    load_brown_counts,
    normalize_text,
    train_on_corpus,
)

# ── PATHS ─────────────────────────────────────────────────────────────────────
# Resolved from this file so the suite works through the repo path and through
# the template's projects/working/ symlink alike.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor"
BROWN_COUNTS_FILE = VENDOR_DIR / "brown_counts.json"
LEDGER_FILE = PROJECT_ROOT / "data" / "claims.json"

_SKIP_NO_COUNTS = pytest.mark.skipif(
    not BROWN_COUNTS_FILE.exists(),
    reason=(
        f"Brown corpus counts not at {BROWN_COUNTS_FILE}. "
        "Run: .venv/bin/python scripts/generate_counts.py"
    ),
)

# Backward-compatible local name. The authenticated protocol bytes live in
# ``ailm.evaluation`` and are shared with every recorder and benchmark consumer.
HELD_OUT_TEXT = STAGE1_EVALUATION_TEXT

# ── GATE CONSTANTS ────────────────────────────────────────────────────────────
UNIFORM_BASELINE = float(N_SYMBOLS)  # exp(log 30) = 30.0, analytic
GATE_MARGIN = 2.0
GATE_THRESHOLD = UNIFORM_BASELINE - GATE_MARGIN  # 28.0


# ── FIXTURES ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def brown_counts() -> dict:
    """Vendored Brown corpus bigram counts (offline; no network, no NLTK)."""
    if not BROWN_COUNTS_FILE.exists():
        pytest.skip(f"missing {BROWN_COUNTS_FILE}")
    return load_brown_counts(VENDOR_DIR)


@pytest.fixture(scope="module")
def bigram_matrix(brown_counts: dict):
    """Column-stochastic (1, 30, 30) bigram matrix from the Brown counts."""
    return build_letter_transition(brown_counts)


@pytest.fixture(scope="module")
def trained_perplexity(bigram_matrix) -> float:
    """Perplexity of the trained agent on HELD_OUT_TEXT (measured once)."""
    return compute_perplexity(build_letter_agent(bigram_matrix), HELD_OUT_TEXT)


@pytest.fixture(scope="module")
def uniform_perplexity() -> float:
    """Perplexity of the uniform (broken) agent on the same held-out text."""
    uniform = jnp.ones((1, N_SYMBOLS, N_SYMBOLS)) / N_SYMBOLS
    return compute_perplexity(build_letter_agent(uniform), HELD_OUT_TEXT)


# ── BASELINES AND ORACLES (numpy, independent of pymdp) ───────────────────────


def _unigram_baseline_perplexity(counts: dict, text: str) -> float:
    """Perplexity of a context-free letter-frequency model from the same counts.

    A bigram-conditioned model that cannot beat this has learned no context —
    uniform is too weak a comparator to detect that.
    """
    frequency = np.ones(N_SYMBOLS, dtype=np.float64)  # same add-1 smoothing
    for key, count in counts.items():
        parts = key.split(",", 1) if isinstance(key, str) else list(key)
        if len(parts) != 2:
            continue
        prev_char, curr_char = parts[0], parts[1]
        if prev_char in ALPHA_TO_IDX and curr_char in ALPHA_TO_IDX:
            frequency[ALPHA_TO_IDX[curr_char]] += float(count)
    frequency /= frequency.sum()

    chars = [c for c in normalize_text(text) if c in ALPHABET]
    log_probs = [math.log(float(frequency[ALPHA_TO_IDX[c]])) for c in chars]
    return math.exp(-sum(log_probs) / len(log_probs))


def _closed_form_bigram_perplexity(bigram_matrix, text: str) -> float:
    """Closed-form perplexity of the letter bigram chain, computed in numpy.

    With an identity emission, the predictive distribution for letter t+1 is
    exactly column c_t of the bigram matrix.
    """
    matrix = np.asarray(bigram_matrix)[0]
    predictive = np.ones(N_SYMBOLS, dtype=np.float64) / N_SYMBOLS
    log_probs: list[float] = []
    for char in (c for c in normalize_text(text) if c in ALPHABET):
        idx = ALPHA_TO_IDX[char]
        log_probs.append(math.log(max(float(predictive[idx]), 1e-10)))
        predictive = matrix[:, idx]
    return math.exp(-sum(log_probs) / len(log_probs))


def _anti_informative_agent(bigram_matrix):
    """The historical mis-wiring: bigram counts in A, identity transition in B.

    Kept as a real, runnable broken model so the negative controls target the
    production metric path (compute_perplexity) rather than a toy.
    """
    from pymdp.agent import Agent

    return Agent(
        A=[jnp.asarray(bigram_matrix)],
        B=[jnp.eye(N_SYMBOLS)[jnp.newaxis, :, :, jnp.newaxis]],
        C=[jnp.zeros((1, N_SYMBOLS))],
        D=[jnp.ones((1, N_SYMBOLS)) / N_SYMBOLS],
        num_controls=[1],
        batch_size=1,
    )


# ── THE GATE ──────────────────────────────────────────────────────────────────


@_SKIP_NO_COUNTS
@pytest.mark.timeout(120)
def test_stage1_letter_perplexity_gate(trained_perplexity: float):
    """POSITIVE CONTROL: trained perplexity must clear the declared threshold."""
    good_ppl = trained_perplexity

    assert math.isfinite(good_ppl) and good_ppl > 0.0, (
        f"Perplexity is not a usable measurement: {good_ppl!r}"
    )
    assert good_ppl < GATE_THRESHOLD, (
        f"STAGE 1 GATE FAILED: trained perplexity {good_ppl:.4f} is not below "
        f"{GATE_THRESHOLD:.1f} (uniform {UNIFORM_BASELINE:.1f} − {GATE_MARGIN:.1f}). "
        f"Log-scale margin achieved: {math.log(UNIFORM_BASELINE / good_ppl):.4f} nats."
    )

    # The harness raises GateVacuousError if the two values cannot be told apart.
    ControlHarness("letter_perplexity_gate").assert_gate_is_live(
        good_ppl, UNIFORM_BASELINE, direction="lower_is_better"
    )

    print(
        f"\nStage 1 gate: perplexity {good_ppl:.4f} < {GATE_THRESHOLD:.1f} "
        f"(uniform {UNIFORM_BASELINE:.1f}; "
        f"{math.log(UNIFORM_BASELINE / good_ppl):.4f} nats better)"
    )


@_SKIP_NO_COUNTS
@pytest.mark.timeout(120)
def test_stage1_negative_control_uniform_model(uniform_perplexity: float):
    """NEGATIVE CONTROL: a uniform model must NOT pass the gate.

    A uniform bigram matrix is the zero-knowledge model. If it clears the
    threshold, the gate admits a model that learned nothing.
    """
    assert uniform_perplexity > GATE_THRESHOLD, (
        f"GATE IS VACUOUS: the uniform model scored {uniform_perplexity:.4f}, which "
        f"passes the {GATE_THRESHOLD:.1f} threshold. A model with zero learned "
        "structure would be admitted — fix the metric or the threshold."
    )
    # Analytic check: uniform dynamics ⇒ every prediction is 1/30 ⇒ ppl = 30.
    assert uniform_perplexity == pytest.approx(UNIFORM_BASELINE, abs=1e-3), (
        f"Uniform model scored {uniform_perplexity:.6f}, not the analytic "
        f"{UNIFORM_BASELINE:.1f} — the baseline itself is miscomputed."
    )
    print(f"\nNegative control: uniform perplexity = {uniform_perplexity:.4f}")


@_SKIP_NO_COUNTS
@pytest.mark.timeout(120)
def test_stage1_gate_is_live_via_control_harness(
    trained_perplexity: float, uniform_perplexity: float
):
    """Both controls measured through ControlHarness, plus the margin check."""
    harness = ControlHarness("letter_perplexity_gate_live_check")
    harness.assert_gate_is_live(trained_perplexity, uniform_perplexity, direction="lower_is_better")

    gap = uniform_perplexity - trained_perplexity
    assert gap >= GATE_MARGIN, (
        f"Discriminative gap is only {gap:.4f} perplexity units (need "
        f"{GATE_MARGIN:.1f}). Trained {trained_perplexity:.4f}, "
        f"uniform {uniform_perplexity:.4f}."
    )

    # verify_negative_control_fails runs the broken model through the *production*
    # metric and raises GateVacuousError if it sneaks under the threshold.
    uniform_agent = build_letter_agent(jnp.ones((1, N_SYMBOLS, N_SYMBOLS)) / N_SYMBOLS)
    harness.verify_negative_control_fails(
        metric_fn=lambda text: compute_perplexity(uniform_agent, text),
        broken_input=HELD_OUT_TEXT,
        threshold=GATE_THRESHOLD,
        direction="lower_is_better",
    )


@_SKIP_NO_COUNTS
@pytest.mark.timeout(120)
def test_stage1_beats_unigram_letter_frequency_baseline(
    brown_counts: dict, trained_perplexity: float
):
    """STRONG CONTROL: beat a context-free letter-frequency model, not just uniform.

    Uniform (30.0) is a crippled comparator — a model can beat it while carrying
    no context at all. The unigram model built from the same counts is the honest
    bar for anything that claims to use the previous letter.
    """
    unigram_ppl = _unigram_baseline_perplexity(brown_counts, HELD_OUT_TEXT)

    assert unigram_ppl < UNIFORM_BASELINE, (
        f"Sanity: the unigram baseline ({unigram_ppl:.4f}) should itself beat "
        f"uniform ({UNIFORM_BASELINE:.1f}); the baseline is miscomputed."
    )
    assert trained_perplexity < unigram_ppl, (
        f"Bigram agent perplexity {trained_perplexity:.4f} does not beat the "
        f"context-free unigram baseline {unigram_ppl:.4f}. The agent is not using "
        "the previous letter, even though it clears the weaker 28.0 threshold."
    )
    print(
        f"\nUnigram baseline {unigram_ppl:.4f} vs agent {trained_perplexity:.4f} "
        f"(uniform {UNIFORM_BASELINE:.1f})"
    )


@_SKIP_NO_COUNTS
@pytest.mark.timeout(120)
def test_stage1_unigram_gate_rejects_the_anti_informative_wiring(brown_counts: dict, bigram_matrix):
    """NEGATIVE CONTROL on the strong gate, using a real mis-wired agent.

    The historical bug installed the bigram counts as the emission A with an
    identity transition, which makes the belief backward-looking. Measured, that
    variant slips under the declared 28.0 threshold, so this control is what
    actually keeps the gate live: the unigram bar must reject it.
    """
    broken_ppl = compute_perplexity(_anti_informative_agent(bigram_matrix), HELD_OUT_TEXT)
    unigram_ppl = _unigram_baseline_perplexity(brown_counts, HELD_OUT_TEXT)

    assert broken_ppl > unigram_ppl, (
        f"GATE IS VACUOUS: the anti-informative wiring scored {broken_ppl:.4f}, "
        f"beating the unigram baseline {unigram_ppl:.4f}. The strong gate no longer "
        "distinguishes correct wiring from backward wiring."
    )
    print(
        f"\nAnti-informative wiring: {broken_ppl:.4f} "
        f"(declared threshold {GATE_THRESHOLD:.1f}, unigram bar {unigram_ppl:.4f})"
    )


@_SKIP_NO_COUNTS
@pytest.mark.timeout(120)
def test_stage1_matches_closed_form_bigram_chain(bigram_matrix, trained_perplexity: float):
    """ORACLE: pymdp inference must reproduce the closed-form bigram chain.

    Without this, a passing perplexity only proves that inference ran — not that
    it computed P(letter_t | letter_{t-1}).
    """
    expected = _closed_form_bigram_perplexity(bigram_matrix, HELD_OUT_TEXT)
    assert trained_perplexity == pytest.approx(expected, rel=1e-3), (
        f"pymdp gives {trained_perplexity:.6f} but the closed-form bigram chain "
        f"gives {expected:.6f}. The generative model is not the bigram chain it "
        "claims to be."
    )


@_SKIP_NO_COUNTS
def test_stage1_bigram_statistics_live_in_the_transition(bigram_matrix):
    """Orientation guard against a regression to the anti-informative wiring."""
    agent = build_letter_agent(bigram_matrix)
    matrix = np.asarray(bigram_matrix)[0]

    np.testing.assert_allclose(
        np.asarray(agent.B[0])[0, :, :, 0],
        matrix,
        atol=1e-6,
        err_msg="Brown bigram statistics must parameterize B (the transition)",
    )
    np.testing.assert_allclose(
        np.asarray(agent.A[0])[0],
        np.eye(N_SYMBOLS),
        atol=1e-6,
        err_msg="The emission A must be the identity",
    )
    # English orientation check: 'u' is the most likely successor of 'q'.
    assert matrix[:, ALPHA_TO_IDX["q"]].argmax() == ALPHA_TO_IDX["u"], (
        "P(next | 'q') does not peak at 'u' — the matrix is transposed"
    )


@_SKIP_NO_COUNTS
def test_stage1_vendor_counts_are_usable(brown_counts: dict, bigram_matrix):
    """The vendored counts must be non-trivial, parseable, and normalized."""
    assert len(brown_counts) > 100, (
        f"Only {len(brown_counts)} bigram counts vendored — too sparse to be Brown"
    )

    arr = np.asarray(bigram_matrix)[0]
    assert arr.min() > 0.0, "add-1 smoothing must leave no zero entries"
    np.testing.assert_allclose(
        arr.sum(axis=0),
        np.ones(N_SYMBOLS),
        atol=1e-5,
        err_msg="The bigram matrix must be column-stochastic",
    )
    assert not np.allclose(arr, 1.0 / N_SYMBOLS, atol=0.01), (
        "The trained matrix is indistinguishable from uniform — the vendored "
        "counts were dropped during parsing"
    )


# ── CLAIMS LEDGER ─────────────────────────────────────────────────────────────


@_SKIP_NO_COUNTS
@pytest.mark.timeout(120)
def test_stage1_claim_verified(trained_perplexity: float):
    """claim:stage1_letter_perplexity must match a fresh measurement of THIS run.

    This test **verifies**; it does not record. The canonical value is written by
    ``scripts/record_stage1_claims.py``. When this test also wrote the label, the
    stored value depended on whether the test or the recorder ran last — the same
    order-dependence defect that decision 10 documents at Stage 2, reproduced one
    level down and caught by ``tests/test_claims_single_writer.py``.
    """
    # Fail loudly and legibly if the on-disk ledger is malformed: ClaimsLedger
    # indexes _data["claims"] directly, so a schema-less file raises a bare
    # KeyError that says nothing about the cause.
    if LEDGER_FILE.exists():
        raw = json.loads(LEDGER_FILE.read_text(encoding="utf-8"))
        assert isinstance(raw, dict) and "claims" in raw, (
            f"{LEDGER_FILE} is missing the 'claims' key (content starts "
            f"{json.dumps(raw)[:40]!r}). ClaimsLedger cannot read it — either seed "
            "the file with the ailm-claims-v2 schema or make ClaimsLedger._load() "
            "repair a malformed ledger."
        )

    entry = load_ledger(PROJECT_ROOT).get("stage1_letter_perplexity")
    assert entry is not None, (
        "claim:stage1_letter_perplexity is absent from the ledger. Run "
        "scripts/record_stage1_claims.py — this test may not write it."
    )
    assert entry["source"].startswith("scripts/"), (
        f"stage1_letter_perplexity was last written by {entry['source']!r}. Only "
        f"recorder scripts may write canonical claim values; tests verify them."
    )
    assert entry["value"] > 0.0
    assert entry["value"] < GATE_THRESHOLD, (
        f"Recorded perplexity {entry['value']} does not clear the Stage 1 "
        f"threshold {GATE_THRESHOLD:.1f}"
    )

    # The stored value must be the measurement this run reproduces, not a stale
    # one: a ledger that has drifted from the code is worse than an absent one.
    assert entry["value"] == pytest.approx(trained_perplexity, abs=5e-4), (
        f"Ledger stage1_letter_perplexity={entry['value']} does not match a fresh "
        f"measurement {trained_perplexity:.4f}. Re-run "
        f"scripts/record_stage1_claims.py — and check nothing else writes this "
        f"label."
    )


def test_stage1_bits_and_signal_capture_are_exact_ledger_derivations():
    """Derived headline quantities must track their recorded operands exactly."""
    ledger = load_ledger(PROJECT_ROOT)

    def value(label: str) -> float:
        entry = ledger.get(label)
        assert entry is not None, f"missing claim:{label}"
        return float(entry["value"])

    perplexity = value("stage1_letter_perplexity")
    uniform = value("stage1_uniform_baseline")
    floor = value("stage1_bigram_floor_perplexity")
    unigram = value("stage1_unigram_baseline")

    expected_bits = round(math.log(perplexity) / math.log(2.0), 4)
    assert value("stage1_letter_bits") == expected_bits
    # NEGATIVE CONTROL: natural-log loss is a plausible but wrong unit.
    assert value("stage1_letter_bits") != round(math.log(perplexity), 4)

    numerator = math.log(uniform) - math.log(perplexity)
    expected_capture = round(100.0 * numerator / (math.log(uniform) - math.log(floor)), 1)
    assert value("stage1_bigram_signal_captured_pct") == expected_capture
    # NEGATIVE CONTROL: the context-free unigram is not the order-specific
    # entropy floor; substituting it changes the denominator materially.
    wrong_denominator = round(100.0 * numerator / (math.log(uniform) - math.log(unigram)), 1)
    assert value("stage1_bigram_signal_captured_pct") != wrong_denominator


# ── STRUCTURAL CHECKS (no vendored corpus required) ───────────────────────────


def test_stage1_module_exports_are_importable():
    """levels.letter must expose the full Stage 1 surface."""
    from ailm.levels import letter

    for name in (
        "ALPHABET",
        "ALPHA_LIST",
        "ALPHA_TO_IDX",
        "IDX_TO_ALPHA",
        "N_SYMBOLS",
        "normalize_text",
        "train_on_corpus",
        "build_letter_transition",
        "build_unigram_a",
        "build_letter_agent",
        "compute_perplexity",
        "load_brown_counts",
    ):
        assert hasattr(letter, name), f"ailm.levels.letter is missing {name}"

    assert len(ALPHABET) == len(ALPHA_LIST) == N_SYMBOLS == 30
    assert letter.build_unigram_a is letter.build_letter_transition


def test_stage1_pipeline_runs_end_to_end_on_a_tiny_corpus():
    """counts → matrix → agent → perplexity must work without the vendored file."""
    counts = train_on_corpus(["the quick brown fox jumps over the lazy dog"] * 10)
    agent = build_letter_agent(build_letter_transition(counts))
    ppl = compute_perplexity(agent, "the fox")

    assert isinstance(ppl, float) and math.isfinite(ppl) and ppl > 0.0
    assert ppl < UNIFORM_BASELINE, (
        f"A model fitted on text containing 'the fox' scored {ppl:.4f}, no better "
        f"than uniform {UNIFORM_BASELINE:.1f}"
    )


def test_stage1_gate_constants_are_consistent():
    """The gate arithmetic must stay tied to the alphabet size, not a literal."""
    assert float(len(ALPHA_LIST)) == UNIFORM_BASELINE
    assert pytest.approx(math.exp(math.log(N_SYMBOLS))) == UNIFORM_BASELINE
    assert GATE_THRESHOLD == UNIFORM_BASELINE - GATE_MARGIN


# ── THE DEGENERATE-INPUT DEFAULT MUST SIT ON THE FAILING SIDE ────────────────


def test_a_degenerate_evaluation_stream_cannot_produce_a_passing_perplexity(bigram_matrix):
    """``compute_perplexity`` returns the uniform value, never a passing one.

    Fewer than two scorable characters is a documented refusal rather than a
    raise: three tests in ``tests/test_letter.py`` pin ``float(N_SYMBOLS)`` as
    the contract, and the oracle's ``forward_filter`` raises on the same input,
    so the two instruments do disagree. What makes the disagreement safe is that
    the returned value is exactly the bar every Stage 1 comparison asks the model
    to beat, so no emptied stream can be laundered into a pass through this path.

    This test is that safety property, stated on the gate rather than on the
    function. If someone ever moves the default off the uniform value, this
    fails even though the three contract tests still pass.
    """
    from ailm.gates import evaluate_gate

    agent = build_letter_agent(bigram_matrix)
    # Genuinely degenerate only: the space is a scorable symbol, so a run of
    # whitespace is a real (bad) measurement rather than a refusal.
    for degenerate in ("", "a", "\t", "z", "9"):
        measured = compute_perplexity(agent, degenerate)
        assert measured == float(N_SYMBOLS), (
            f"{degenerate!r} scored {measured}, not the uniform value; the "
            f"refusal is no longer pinned to the failing side of the bar"
        )
        outcome = evaluate_gate(
            baseline=float(N_SYMBOLS),
            model=measured,
            ceiling=None,
            threshold=float(N_SYMBOLS),
            direction="lower_is_better",
        )
        assert not outcome.passes, (
            f"a degenerate stream ({degenerate!r}) cleared a bar set at the "
            f"uniform value; the default is fail-open"
        )
