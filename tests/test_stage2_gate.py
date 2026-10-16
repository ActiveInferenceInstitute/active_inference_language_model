"""Stage 2 meta-gate: the phoneme level must beat the CONTEXT-FREE comparator.

The gate has two independent halves and both are stated in full here, including
the part that did not come out the way the specification anticipated.

Part A — phoneme bigram perplexity
----------------------------------
1. Load ``vendor/cmudict_counts.json`` (skip if it has not been generated).
2. Build the Part A agent: phone bigrams in **B**, identity emission in **A**.
3. POSITIVE CONTROL — trained perplexity must beat the *context-free phone
   unigram* model computed inside the test from ``phone_unigrams``. Uniform (39.0)
   is a crippled comparator: it is beaten by a model carrying no context at all.
4. NEGATIVE CONTROL — a uniform transition must measure 39.0 (analytic) through
   the production metric and must FAIL the unigram bar.
5. ORDER-1 INFORMATION FLOOR — ``exp(H(next | cur))`` computed from
   ``phone_bigrams``. A trained model scoring *below* the conditional entropy of
   its own training distribution is reporting test leakage or a metric error, not
   skill, so the gate asserts ``trained >= floor * 0.98``.
6. CEILING — the total information available over uniform is
   ``H_uniform - H(next|cur) = 3.6636 - 2.6892 = 0.9744`` nats, derived below from
   the artifact rather than asserted in prose. A gate phrased as "at least 2 nats
   below uniform" is therefore **unreachable at this level for any
   implementation**: 2 nats below uniform is 39/e² = 5.28 perplexity, far under the
   corpus floor of 14.72. Stage 1's "uniform − 2" wording must not be carried over.

Part B — G2P beats the context-free bar, *with* significance
------------------------------------------------------------
The honest margin is +0.55 points (70.801% filtering vs 70.255% per-letter
argmax) over 42,820 held-out positions. A bare ``>`` on half a point is not
evidence, so the gate pairs the two decoders position-for-position and runs an
exact McNemar (binomial) test on the discordant pairs.

**Finding, stated plainly because the task specification assumed otherwise.**
A bounded pymdp subsample is *underpowered* for this margin. Measured in this
file at ``limit=2000`` (13,224 positions): b=434, c=371, p = 2.9e-02 — **not**
significant at α = 0.01. The margin is real but a 2,000-word subsample cannot
establish it. This is not worked around by loosening α or by hunting for a
friendlier subsample. It is resolved the way ``ailm.levels.phoneme`` and
``scripts/record_stage2_claims.py`` already prescribe:

  1. prove the pymdp decoder and the independent numpy oracle agree on
     **zero** positions out of 13,224 (:func:`test_stage2_g2p_pymdp_matches_oracle`),
  2. then compute the full-set statistics with the oracle, which is licensed by
     that identity and costs 0.3 s instead of 3 minutes.

Full held-out set: b=1379, c=1145, p = 3.5e-06 — significant. Both numbers are
reported. The subsample result is kept as a *live* control: it demonstrates the
same McNemar procedure returning "no effect" on real data, which is what keeps
the full-set p-value honest (a test that can only say "significant" says nothing).

Two further nulls calibrate the machinery (item 8): a cyclic rotation of the
baseline vector against itself (exactly b = c = 8946, p = 1.0) and 20 seeded
accuracy-matched Bernoulli comparators (0/20 significant at α = 0.01, min
p = 0.239). A single null would be a cherry-pick; 20 measures the false-positive
rate against its nominal 1%.

Controls on the model itself
---------------------------
  - uniform emission (no letter information)      → 12.74%, caught
  - time-reversed B (backward-looking dynamics)   → 53.88%, caught
  - column-permuted B on perplexity               → 81.08 ppl, caught
  - A normalized over phones, then renormalized
    so pymdp still validates it                   → 70.56%, **NOT** caught by
    the accuracy bar (it clears 70.255%)

That last one is why "beats the context-free baseline" is necessary but not
sufficient, and why :func:`test_stage2_g2p_emission_orientation_is_pinned`
re-derives ``A[l, p] = P(letter l | phone p)`` independently in numpy and compares
values. The accuracy bar cannot see a mis-normalized emission; an independent
re-derivation of the matrix can.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import binomtest

from ailm.claims import load_ledger
from ailm.control import ControlHarness
from ailm.levels.letter import ALPHA_TO_IDX, N_SYMBOLS, normalize_text

# ``_iter_pairs`` is imported deliberately despite the underscore. The oracle
# equivalence proof is only valid if the oracle scores *exactly* the positions
# pymdp scored, and that subsample is defined by this function. Re-deriving the
# stride formula in the test would let the two drift apart silently — the very
# failure mode the proof exists to exclude.
from ailm.levels.phoneme import (
    N_PHONES,
    PARTA_HELDOUT_STRIDE,
    PHONE_TO_IDX,
    PHONES,
    _iter_pairs,
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
    decode_positions,
    full_corpus_order1_floor_perplexity,
    full_corpus_perplexity,
    full_corpus_unigram_perplexity,
    g2p_accuracy,
    g2p_split,
    load_phoneme_counts,
    part_a_counts,
    part_a_split_perplexity,
    per_letter_argmax_baseline,
    phoneme_heldout_split,
)
from ailm.statistics import (
    FAMILYWISE_ALPHA as ALPHA,
)
from ailm.statistics import (
    SIGNIFICANCE_FAMILY,
)
from ailm.verification.oracle import forward_filter, identity_emission

# ── PATHS ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor"
PHONEME_COUNTS_FILE = VENDOR_DIR / "cmudict_counts.json"
LEDGER_FILE = PROJECT_ROOT / "data" / "claims.json"

_SKIP_NO_COUNTS = pytest.mark.skipif(
    not PHONEME_COUNTS_FILE.exists(),
    reason=(
        f"CMUdict phoneme counts not at {PHONEME_COUNTS_FILE}. "
        "Run: .venv/bin/python scripts/generate_phoneme_counts.py"
    ),
)

# ── GATE CONSTANTS ────────────────────────────────────────────────────────────
#: Uniform over 39 phones — analytic, exp(log 39). The *weak* comparator.
UNIFORM_PERPLEXITY = float(N_PHONES)

#: Tolerance on the order-1 floor. The floor is an unsmoothed transitions-only
#: conditional entropy over the whole corpus while the metric is add-1 smoothed
#: and pays a unigram price at every word-initial position, so exact equality is
#: not expected — only that the model does not dip implausibly below it.
FLOOR_TOLERANCE = 0.98

#: The "2 nats below uniform" wording inherited from Stage 1. Asserted below to
#: be unreachable at this level, from measured entropies rather than from prose.
UNREACHABLE_NAT_TARGET = 2.0

#: Stride for the Part A pymdp measurement. A stride, never a prefix: the
#: vendored g2p_pairs list is alphabetically sorted.
PART_A_STRIDE = 60

#: Words decoded through pymdp for the Part B production-path gate and for the
#: oracle equivalence proof. ~54 s at roughly 4 ms/position.
PYMDP_G2P_WORDS = 2000

#: Words decoded through pymdp for the mis-wired negative controls. These fail by
#: 15-55 points, so they do not need the large sample the +0.55 pt gate needs.
PYMDP_NEG_WORDS = 300

#: The executable repository-wide family includes Stage 6 and counts the
#: full-set Stage 2 comparison once, not again under its figure-specific alias.
FAMILY_N = len(SIGNIFICANCE_FAMILY)

#: Number of seeded null comparators used to calibrate the McNemar machinery.
N_NULL_SEEDS = 20

#: Nominal false-positive budget over N_NULL_SEEDS nulls at ALPHA. Binomial(20,
#: 0.01) puts P(>=3 hits) at 1.0e-3, so 2 is a generous ceiling that still fails
#: if the test rubber-stamps.
MAX_NULL_FALSE_POSITIVES = 2

#: Minimum eval positions. Guards a silently truncated evaluation.
MIN_EVAL_POSITIONS = 1000


# ── INDEPENDENT NUMPY HELPERS (no pymdp, no jax) ──────────────────────────────


def _word_letter_indices(word: str) -> list[int]:
    """Letter indices of a word under the production normalization.

    Mirrors ``ailm.levels.phoneme._word_letters`` so an oracle decode scores the
    same positions as the pymdp decoder.

    Args:
        word: Raw word.

    Returns:
        Indices into ``ALPHA_LIST``, one per retained character.
    """
    return [ALPHA_TO_IDX[c] for c in normalize_text(word) if c in ALPHA_TO_IDX]


def _phone_information_bounds(counts: dict) -> dict[str, float]:
    """Uniform / unigram / order-1 entropies of the phone bigram table, in nats.

    Everything is derived from the artifact's own counts. Nothing here is a
    literal, so the reachable-headroom claim is measured rather than asserted.

    Args:
        counts: Payload from :func:`load_phoneme_counts`.

    Returns:
        Mapping with ``h_uniform_nats``, ``h_unigram_nats``, ``h_cond_nats``,
        ``floor_perplexity``, ``unigram_perplexity`` and ``max_gain_nats``.
    """
    joint = np.zeros((N_PHONES, N_PHONES), dtype=np.float64)
    for key, count in counts["phone_bigrams"].items():
        prev_phone, next_phone = str(key).split(",", 1)
        if prev_phone in PHONE_TO_IDX and next_phone in PHONE_TO_IDX:
            joint[PHONE_TO_IDX[next_phone], PHONE_TO_IDX[prev_phone]] += float(count)
    joint /= joint.sum()
    p_prev = joint.sum(axis=0)

    # H(next | cur) = -sum_{i,j} p(i,j) log p(i|j). Unsmoothed MLE, transitions
    # only: this is the information floor of the *training distribution itself*.
    h_cond = 0.0
    for j in range(N_PHONES):
        if p_prev[j] <= 0.0:
            continue
        column = joint[:, j]
        nonzero = column > 0.0
        h_cond -= float(np.sum(column[nonzero] * np.log(column[nonzero] / p_prev[j])))

    unigram = _unigram_distribution(counts)
    h_unigram = -float(np.sum(unigram * np.log(unigram)))
    h_uniform = math.log(N_PHONES)

    return {
        "h_uniform_nats": h_uniform,
        "h_unigram_nats": h_unigram,
        "h_cond_nats": h_cond,
        "floor_perplexity": math.exp(h_cond),
        "unigram_perplexity": math.exp(h_unigram),
        "max_gain_nats": h_uniform - h_cond,
    }


def _unigram_distribution(counts: dict) -> np.ndarray:
    """Add-1 smoothed phone unigram distribution from the artifact.

    Args:
        counts: Payload from :func:`load_phoneme_counts`.

    Returns:
        ``(39,)`` float64 array summing to 1.
    """
    tallies = np.array(
        [float(counts["phone_unigrams"].get(p, 0)) + 1.0 for p in PHONES],
        dtype=np.float64,
    )
    return tallies / tallies.sum()


def _unigram_perplexity_on(counts: dict, sequences: list[list[str]]) -> float:
    """Context-free unigram perplexity over exactly the given phone positions.

    This is the honest comparator for :func:`compute_phoneme_perplexity` on the
    same sequences: same positions, no context.

    Args:
        counts: Payload from :func:`load_phoneme_counts`.
        sequences: Phone sequences.

    Returns:
        Per-phone perplexity of the context-free model.

    Raises:
        ValueError: If ``sequences`` contains no scorable position.
    """
    unigram = _unigram_distribution(counts)
    indices = [PHONE_TO_IDX[p] for seq in sequences for p in seq]
    if not indices:
        raise ValueError("No scorable phone positions — the comparator is vacuous.")
    total = sum(math.log(float(unigram[i])) for i in indices)
    return math.exp(-total / len(indices))


def _oracle_corpus_perplexity(
    transition: np.ndarray,
    start: np.ndarray,
    sequences: list[list[str]],
) -> tuple[float, int]:
    """Corpus per-phone perplexity of a directly-observed phone chain, in numpy.

    Independent of pymdp and jax by construction (see
    ``ailm.verification.oracle``). The prior is reset to ``start`` at every
    sequence boundary and log-evidence is pooled across sequences, matching
    :func:`compute_phoneme_perplexity`'s aggregation exactly.

    Args:
        transition: ``(39, 39)`` with ``transition[i, j] = P(next i | cur j)``.
        start: ``(39,)`` distribution over the first phone.
        sequences: Phone sequences.

    Returns:
        ``(perplexity, n_positions)``.

    Raises:
        ValueError: If no sequence contains a scorable position.
    """
    emission = identity_emission(N_PHONES)
    total_logp = 0.0
    total_positions = 0
    for seq in sequences:
        indices = [PHONE_TO_IDX[p] for p in seq]
        if not indices:
            continue
        _, log_evidence = forward_filter(emission, transition, start, indices)
        total_logp += log_evidence
        total_positions += len(indices)
    if total_positions == 0:
        raise ValueError("No scorable phone positions.")
    return math.exp(-total_logp / total_positions), total_positions


def _oracle_g2p_positions(
    emission: np.ndarray,
    transition: np.ndarray,
    start: np.ndarray,
    pairs: list[tuple[str, list[str]]],
) -> list[bool]:
    """Per-position correctness of a filtered argmax decode, in numpy.

    Args:
        emission: ``(30, 39)`` with ``emission[l, p] = P(letter l | phone p)``.
        transition: ``(39, 39)`` with ``transition[i, j] = P(next i | cur j)``.
        start: ``(39,)`` distribution over the first phone.
        pairs: ``(word, [phones])`` items, already subsampled by the caller.

    Returns:
        One bool per position, words in order and letters in order.

    Raises:
        ValueError: If a decode length differs from its reference length.
    """
    correct: list[bool] = []
    for word, phones in pairs:
        indices = _word_letter_indices(word)
        if not indices:
            continue
        posteriors, _ = forward_filter(emission, transition, start, indices)
        decoded = [PHONES[int(np.argmax(row))] for row in posteriors]
        if len(decoded) != len(phones):
            raise ValueError(
                f"Oracle decoded {len(decoded)} phones for {word!r} but the "
                f"reference has {len(phones)}. The comparison is not paired."
            )
        correct.extend(d == r for d, r in zip(decoded, phones, strict=True))
    return correct


def _mcnemar(model_hits: list[bool], baseline_hits: list[bool]) -> dict[str, float]:
    """Exact paired McNemar (binomial) test on two aligned correctness vectors.

    ``b`` counts positions where only the model is right, ``c`` positions where
    only the baseline is right. Concordant pairs carry no information about the
    difference and are discarded — which is exactly why the aggregate accuracies
    cannot establish a half-point margin on their own.

    The exact binomial form is used rather than the chi-square approximation
    because ``b + c`` can be small on subsamples.

    Args:
        model_hits: Per-position correctness of the model.
        baseline_hits: Per-position correctness of the comparator, position-aligned.

    Returns:
        Mapping with ``b``, ``c``, ``discordant``, ``p_value``, ``n_positions``,
        ``model_accuracy``, ``baseline_accuracy`` and ``margin_pts``.

    Raises:
        ValueError: If the vectors differ in length (the pairing is invalid) or
            are empty, or if there are no discordant pairs — with ``b + c == 0``
            the test has nothing to condition on and is vacuous.
    """
    if len(model_hits) != len(baseline_hits):
        raise ValueError(
            f"Unpaired vectors: model has {len(model_hits)} positions, baseline "
            f"has {len(baseline_hits)}. McNemar requires position alignment."
        )
    if not model_hits:
        raise ValueError("Empty correctness vectors — the test would be vacuous.")

    model = np.asarray(model_hits, dtype=bool)
    baseline = np.asarray(baseline_hits, dtype=bool)
    b = int(np.sum(model & ~baseline))
    c = int(np.sum(baseline & ~model))
    if b + c == 0:
        raise ValueError(
            "b + c == 0: the two decoders never disagree, so McNemar is vacuous. "
            "Either they are the same object or the pairing is broken."
        )

    return {
        "b": b,
        "c": c,
        "discordant": b + c,
        "p_value": float(binomtest(b, b + c, 0.5).pvalue),
        "n_positions": int(model.size),
        "model_accuracy": float(model.mean()),
        "baseline_accuracy": float(baseline.mean()),
        "margin_pts": 100.0 * float(model.mean() - baseline.mean()),
    }


def _independent_emission(train_pairs: list[tuple[str, list[str]]]) -> np.ndarray:
    """Re-derive ``A[l, p] = P(letter l | phone p)`` from scratch in numpy.

    Deliberately does not call :func:`build_g2p_emission`. A mis-normalized
    emission (counts normalized over phones and then renormalized over letters so
    pymdp still validates it) is column-stochastic and *clears the context-free
    accuracy bar* at 70.56%, so accuracy cannot detect it. Comparing values
    against an independent derivation can.

    Args:
        train_pairs: ``(word, [phones])`` items from :func:`g2p_split`.

    Returns:
        ``(30, 39)`` add-1 smoothed matrix, column-stochastic over letters.
    """
    tallies = np.ones((N_SYMBOLS, N_PHONES), dtype=np.float64)  # add-1 Laplace
    for word, phones in train_pairs:
        letters = _word_letter_indices(word)
        if len(letters) != len(phones):
            continue
        for letter_idx, phone in zip(letters, phones, strict=True):
            phone_idx = PHONE_TO_IDX.get(phone)
            if phone_idx is not None:
                tallies[letter_idx, phone_idx] += 1.0
    return tallies / tallies.sum(axis=0, keepdims=True)


def _mis_normalized_emission(emission: np.ndarray) -> np.ndarray:
    """The emission mis-wiring that the accuracy bar does NOT catch.

    Normalizes over phones (the wrong axis) and then renormalizes over letters so
    that pymdp's ``_validate()`` still accepts it. Measured at 70.56% — above the
    70.255% context-free bar, below the 70.801% correct wiring.

    Args:
        emission: A correct ``(30, 39)`` emission.

    Returns:
        A ``(30, 39)`` column-stochastic but wrongly-derived emission.
    """
    wrong = emission / emission.sum(axis=1, keepdims=True)
    return wrong / wrong.sum(axis=0, keepdims=True)


# ── FIXTURES ──────────────────────────────────────────────────────────────────

#: Stage 2 decodes the full held-out G2P set through both pymdp and the oracle,
#: so its module fixtures cost minutes rather than seconds and that cost is
#: charged to whichever test requests them first. Under the suite-wide 60s
#: default those tests error whenever the machine is busy, and the error reads as
#: a broken gate rather than as a clock — this module produced eleven such errors
#: in one run while a benchmark census saturated the machine. This is the fifth
#: place in this project where a default sized for computation was inherited by a
#: test whose cost is a corpus, a live server, or a fleet of subprocesses. The
#: ceiling stays finite so a genuinely wedged decode still fails, and the
#: evaluation set is left alone: shrinking it to fit a clock would weaken the
#: very margin this module exists to establish.
pytestmark = pytest.mark.timeout(1800)


@pytest.fixture(scope="module")
def counts() -> dict:
    """Vendored CMUdict phoneme statistics (offline; no network, no NLTK)."""
    if not PHONEME_COUNTS_FILE.exists():
        pytest.skip(f"missing {PHONEME_COUNTS_FILE}")
    return load_phoneme_counts(VENDOR_DIR)


@pytest.fixture(scope="module")
def bounds(counts: dict) -> dict[str, float]:
    """Information-theoretic bounds derived from the artifact's bigram table."""
    return _phone_information_bounds(counts)


@pytest.fixture(scope="module")
def part_a(counts: dict) -> dict:
    """Part A measurements: pymdp on a stride sample plus full-corpus oracle.

    The pymdp numbers are the production path and carry the gate. The oracle
    numbers cover the whole 32,420-sequence corpus and are licensed by the
    agreement between the two on the sample (asserted in
    :func:`test_stage2_phoneme_pymdp_matches_oracle`).
    """
    sample = [str(s).split() for _, s in counts["g2p_pairs"][::PART_A_STRIDE]]
    full = [str(s).split() for _, s in counts["g2p_pairs"]]

    transition = build_phone_transition(counts["phone_bigrams"])
    start = build_phone_unigram(counts["phone_unigrams"])
    agent = build_phoneme_agent(transition, start)
    uniform_transition = jnp.ones((1, N_PHONES, N_PHONES, 1)) / N_PHONES
    uniform_agent = build_phoneme_agent(uniform_transition)

    trans_np = np.asarray(transition)[0, :, :, 0].astype(np.float64)
    trans_np /= trans_np.sum(axis=0, keepdims=True)  # undo float32 drift
    start_np = np.asarray(start).ravel().astype(np.float64)
    start_np /= start_np.sum()

    oracle_sample, n_sample = _oracle_corpus_perplexity(trans_np, start_np, sample)
    oracle_full, n_full = _oracle_corpus_perplexity(trans_np, start_np, full)

    return {
        "sample": sample,
        "full": full,
        "agent": agent,
        "uniform_agent": uniform_agent,
        "transition_np": trans_np,
        "start_np": start_np,
        "pymdp_sample": compute_phoneme_perplexity(agent, sample),
        "pymdp_uniform_sample": compute_phoneme_perplexity(uniform_agent, sample),
        "unigram_sample": _unigram_perplexity_on(counts, sample),
        "unigram_full": _unigram_perplexity_on(counts, full),
        "oracle_sample": oracle_sample,
        "oracle_full": oracle_full,
        "n_sample_positions": n_sample,
        "n_full_positions": n_full,
    }


@pytest.fixture(scope="module")
def part_b(counts: dict) -> dict:
    """Part B measurements: pymdp subsample, oracle equivalence, full-set McNemar.

    One pymdp pass over ``PYMDP_G2P_WORDS`` words serves three purposes — the
    production-path positive control, the oracle equivalence proof, and the
    honest demonstration that a subsample of this size is underpowered.
    """
    train, test = g2p_split(counts)
    emission_j = build_g2p_emission(train)
    transition_j = build_g2p_transition(train)
    start_j = build_g2p_start(train)
    agent = build_g2p_agent(emission_j, transition_j, start_j)
    mapping = per_letter_argmax_baseline(train)

    emission = np.asarray(emission_j)[0].astype(np.float64)
    emission /= emission.sum(axis=0, keepdims=True)
    transition = np.asarray(transition_j)[0, :, :, 0].astype(np.float64)
    transition /= transition.sum(axis=0, keepdims=True)
    start = np.asarray(start_j).ravel().astype(np.float64)
    start /= start.sum()

    # ── one pymdp pass, three uses ───────────────────────────────────────────
    subset = _iter_pairs(test, PYMDP_G2P_WORDS)
    pymdp_hits = decode_positions(agent, test, limit=PYMDP_G2P_WORDS)
    base_hits_sub = baseline_positions(mapping, test, limit=PYMDP_G2P_WORDS)
    oracle_hits_sub = _oracle_g2p_positions(emission, transition, start, subset)

    # ── full held-out set through the oracle ─────────────────────────────────
    full_model = _oracle_g2p_positions(emission, transition, start, test)
    full_base = baseline_positions(mapping, test)

    # ── mis-wired variants, full set through the oracle ──────────────────────
    reversed_transition = transition.T / transition.T.sum(axis=0, keepdims=True)
    uniform_emission = np.full((N_SYMBOLS, N_PHONES), 1.0 / N_SYMBOLS)
    mis_normalized = _mis_normalized_emission(emission)

    return {
        "train": train,
        "test": test,
        "agent": agent,
        "mapping": mapping,
        "emission": emission,
        "transition": transition,
        "start": start,
        "subset": subset,
        "pymdp_hits": pymdp_hits,
        "base_hits_sub": base_hits_sub,
        "oracle_hits_sub": oracle_hits_sub,
        "full_model": full_model,
        "full_base": full_base,
        "subsample": _mcnemar(pymdp_hits, base_hits_sub),
        "full": _mcnemar(full_model, full_base),
        "reversed_b_hits": _oracle_g2p_positions(emission, reversed_transition, start, test),
        "uniform_a_hits": _oracle_g2p_positions(uniform_emission, transition, start, test),
        "mis_normalized_hits": _oracle_g2p_positions(mis_normalized, transition, start, test),
        "reversed_transition": reversed_transition,
        "uniform_emission": uniform_emission,
        "mis_normalized": mis_normalized,
        "length_matched_pct": (
            100.0 * len(counts["g2p_pairs"]) / float(counts["source_base_words"])
        ),
    }


# ══ PART A ════════════════════════════════════════════════════════════════════


@_SKIP_NO_COUNTS
def test_stage2_phoneme_beats_context_free_unigram(part_a: dict):
    """POSITIVE CONTROL (1): trained bigram must beat the context-free unigram.

    The unigram comparator is computed inside this test from the artifact's
    ``phone_unigrams`` over exactly the positions the trained model scored.
    Uniform is not used as the bar: at 39.0 it is beaten by a model that carries
    no context whatsoever.
    """
    trained = part_a["pymdp_sample"]
    unigram = part_a["unigram_sample"]

    assert math.isfinite(trained) and trained > 0.0, (
        f"Perplexity is not a usable measurement: {trained!r}"
    )
    assert unigram < UNIFORM_PERPLEXITY, (
        f"Sanity: the context-free unigram model ({unigram:.4f}) must itself beat "
        f"uniform ({UNIFORM_PERPLEXITY:.1f}) or the comparator is miscomputed."
    )
    assert trained < unigram, (
        f"STAGE 2 PART A GATE FAILED: trained phoneme bigram perplexity "
        f"{trained:.4f} does not beat the CONTEXT-FREE unigram baseline "
        f"{unigram:.4f} over {part_a['n_sample_positions']} positions. The model "
        f"is not using the previous phone, even if it clears uniform "
        f"({UNIFORM_PERPLEXITY:.1f})."
    )

    print(
        f"\nPart A: trained {trained:.4f} < unigram {unigram:.4f} "
        f"< uniform {UNIFORM_PERPLEXITY:.1f} over "
        f"{part_a['n_sample_positions']} positions "
        f"({math.log(unigram / trained):.4f} nats over the context-free bar)"
    )


@_SKIP_NO_COUNTS
def test_stage2_phoneme_negative_control_uniform_transition(part_a: dict):
    """NEGATIVE CONTROL (2): a uniform transition must measure 39.0 and FAIL.

    Run through the production metric, not a toy, so the broken model traverses
    the same code path the gate trusts.
    """
    uniform_measured = part_a["pymdp_uniform_sample"]
    unigram_bar = part_a["unigram_sample"]

    assert uniform_measured == pytest.approx(UNIFORM_PERPLEXITY, rel=1e-5), (
        f"A uniform transition measured {uniform_measured:.6f}, not the analytic "
        f"{UNIFORM_PERPLEXITY:.1f}. The metric itself is wrong, so nothing else "
        f"in Part A means anything."
    )
    assert uniform_measured > unigram_bar, (
        f"GATE IS VACUOUS: the uniform (zero-knowledge) transition scored "
        f"{uniform_measured:.4f}, which passes the unigram bar {unigram_bar:.4f}."
    )

    ControlHarness("stage2_phoneme_unigram_gate").verify_negative_control_fails(
        metric_fn=lambda seqs: compute_phoneme_perplexity(part_a["uniform_agent"], seqs),
        broken_input=part_a["sample"],
        threshold=unigram_bar,
        direction="lower_is_better",
    )
    print(
        f"\nNegative control: uniform transition = {uniform_measured:.6f} "
        f"(bar {unigram_bar:.4f}) — caught through the production metric."
    )


@_SKIP_NO_COUNTS
def test_stage2_phoneme_gate_is_live_via_control_harness(part_a: dict):
    """CONTROL HARNESS (3): the gate must be able to tell trained from broken."""
    harness = ControlHarness("stage2_phoneme_perplexity_gate")
    harness.assert_gate_is_live(
        part_a["pymdp_sample"], part_a["unigram_sample"], direction="lower_is_better"
    )
    harness.assert_gate_is_live(
        part_a["pymdp_sample"],
        part_a["pymdp_uniform_sample"],
        direction="lower_is_better",
    )

    # A second, independent broken model: the phone bigram table with its columns
    # permuted. Structurally valid, informationally destroyed. Run through the
    # numpy oracle on the whole corpus so this control is not sample-bound.
    permutation = np.random.default_rng(20260729).permutation(N_PHONES)
    permuted = part_a["transition_np"][:, permutation]
    permuted /= permuted.sum(axis=0, keepdims=True)
    permuted_ppl, _ = _oracle_corpus_perplexity(permuted, part_a["start_np"], part_a["full"])

    assert permuted_ppl > part_a["unigram_full"], (
        f"GATE IS VACUOUS: a column-permuted bigram table scored "
        f"{permuted_ppl:.4f}, beating the unigram bar {part_a['unigram_full']:.4f}."
    )
    print(
        f"\nColumn-permuted B: {permuted_ppl:.4f} ppl "
        f"(unigram bar {part_a['unigram_full']:.4f}) — caught."
    )


@_SKIP_NO_COUNTS
def test_stage2_phoneme_does_not_beat_its_own_order1_floor(part_a: dict, bounds: dict[str, float]):
    """FLOOR (4): the model must not score below the entropy of its own training set.

    ``exp(H(next | cur))`` is the best any order-1 model can do on the
    distribution it was fitted to. Scoring materially below it means the metric
    is wrong or the evaluation leaked, not that the model is unusually good.

    The tolerance is one-sided and small: the floor is unsmoothed and
    transitions-only, while the metric is add-1 smoothed and pays a unigram price
    at every word-initial position, so the honest expectation is that the model
    sits *above* the floor.
    """
    floor = bounds["floor_perplexity"]
    trained_full = part_a["oracle_full"]
    trained_sample = part_a["pymdp_sample"]

    assert floor > 1.0, f"Degenerate floor {floor!r} — the bigram table is broken."
    assert floor < bounds["unigram_perplexity"], (
        f"An order-1 floor ({floor:.4f}) at or above the order-0 unigram "
        f"perplexity ({bounds['unigram_perplexity']:.4f}) is impossible; the "
        f"entropy computation is wrong."
    )

    measurements = (("full-corpus", trained_full), ("pymdp sample", trained_sample))
    for label, value in measurements:
        assert value >= floor * FLOOR_TOLERANCE, (
            f"IMPLAUSIBLE: {label} perplexity {value:.4f} is below "
            f"{FLOOR_TOLERANCE:.2f} x the order-1 information floor "
            f"{floor:.4f} = {floor * FLOOR_TOLERANCE:.4f}. A model cannot beat the "
            f"conditional entropy of its own training distribution — suspect test "
            f"leakage or a metric error before celebrating."
        )

    # Fraction of the available order-1 signal the model actually captured, in
    # nats: (H_unigram - H_model) / (H_unigram - H_cond).
    captured = (bounds["h_unigram_nats"] - math.log(trained_full)) / (
        bounds["h_unigram_nats"] - bounds["h_cond_nats"]
    )
    print(
        f"\nOrder-1 floor {floor:.4f}; full-corpus {trained_full:.4f} "
        f"(gap {trained_full - floor:+.4f}), pymdp sample {trained_sample:.4f} "
        f"(gap {trained_sample - floor:+.4f}). Signal captured over unigram: "
        f"{100 * captured:.1f}%"
    )


@_SKIP_NO_COUNTS
def test_stage2_two_nat_target_is_provably_unreachable(bounds: dict[str, float]):
    """CEILING (5): "2 nats below uniform" cannot be reached at this level.

    Derived from the artifact's own entropies, not asserted in prose: the total
    order-1 information available over uniform is
    ``H_uniform - H(next | cur) = 3.6636 - 2.6892 = 0.9744`` nats. Stage 1's
    "uniform − 2" wording must not be carried into Stage 2 — no implementation,
    however correct, can satisfy it.
    """
    max_gain = bounds["max_gain_nats"]

    assert bounds["h_uniform_nats"] == pytest.approx(math.log(N_PHONES), rel=1e-12)
    assert max_gain == pytest.approx(bounds["h_uniform_nats"] - bounds["h_cond_nats"], rel=1e-12)
    assert 0.0 < max_gain < UNREACHABLE_NAT_TARGET, (
        f"The order-1 headroom over uniform is {max_gain:.4f} nats. If this ever "
        f"exceeds {UNREACHABLE_NAT_TARGET:.1f} the unreachability argument in this "
        f"module's docstring is stale and must be rewritten."
    )

    unreachable_ppl = N_PHONES / math.exp(UNREACHABLE_NAT_TARGET)
    assert unreachable_ppl < bounds["floor_perplexity"], (
        f"A {UNREACHABLE_NAT_TARGET:.1f}-nat gain would require perplexity "
        f"{unreachable_ppl:.4f}, which must be below the order-1 floor "
        f"{bounds['floor_perplexity']:.4f} for the target to be unreachable."
    )
    print(
        f"\nCeiling: uniform {bounds['h_uniform_nats']:.4f} nats − order-1 "
        f"{bounds['h_cond_nats']:.4f} nats = {max_gain:.4f} nats available. "
        f"A {UNREACHABLE_NAT_TARGET:.1f}-nat target needs ppl "
        f"{unreachable_ppl:.4f} < floor {bounds['floor_perplexity']:.4f}: "
        f"UNREACHABLE for any implementation."
    )


@_SKIP_NO_COUNTS
def test_stage2_phoneme_pymdp_matches_oracle(part_a: dict):
    """ORACLE (Part A): pymdp must reproduce the closed-form phone chain.

    Without this a passing perplexity only proves that inference ran. It also
    licenses the full-corpus oracle figures used by the floor test and the ledger.
    """
    pymdp = part_a["pymdp_sample"]
    oracle = part_a["oracle_sample"]
    assert pymdp == pytest.approx(oracle, rel=1e-5), (
        f"pymdp gives {pymdp:.8f} but the independent numpy chain gives "
        f"{oracle:.8f} (rel {abs(pymdp - oracle) / oracle:.2e}). The generative "
        f"model is not the bigram chain it claims to be, so the full-corpus "
        f"oracle numbers are void."
    )
    print(
        f"\nPart A oracle: pymdp {pymdp:.8f} vs numpy {oracle:.8f} "
        f"(rel {abs(pymdp - oracle) / oracle:.2e}) over "
        f"{part_a['n_sample_positions']} positions; full corpus "
        f"{part_a['oracle_full']:.4f} over {part_a['n_full_positions']} positions."
    )


@_SKIP_NO_COUNTS
def test_stage2_phoneme_statistics_live_in_the_transition(part_a: dict):
    """Orientation guard: learned dynamics in B, identity emission in A.

    The Stage 1 bug put the learned matrix in A with an identity B, which is
    anti-informative and which pymdp validates happily.
    """
    agent = part_a["agent"]
    np.testing.assert_allclose(
        np.asarray(agent.A[0])[0],
        np.eye(N_PHONES),
        atol=1e-6,
        err_msg="Part A's emission must be the identity, not the learned matrix",
    )
    b_matrix = np.asarray(agent.B[0])[0, :, :, 0]
    assert not np.allclose(b_matrix, np.eye(N_PHONES), atol=1e-3), (
        "B is the identity — the learned statistics are in the wrong slot"
    )
    np.testing.assert_allclose(
        b_matrix.sum(axis=0),
        np.ones(N_PHONES),
        atol=1e-5,
        err_msg="B[0, :, j, 0] must sum to 1 over the NEXT phone",
    )
    # Phonotactic orientation: 'NG' never starts an English word, so the column
    # for 'NG' as *previous* phone must be well populated while the row for 'NG'
    # as *next* phone after a vowel carries most of its mass. A transposed matrix
    # inverts which of these is which; check a concrete asymmetry instead.
    assert not np.allclose(b_matrix, b_matrix.T, atol=1e-3), (
        "B is symmetric, so a transpose would be undetectable — the phone bigram "
        "table cannot be symmetric on real speech data"
    )


# ══ PART B ════════════════════════════════════════════════════════════════════


@_SKIP_NO_COUNTS
def test_stage2_g2p_beats_context_free_baseline(part_b: dict):
    """POSITIVE CONTROL (6): pymdp filtering must beat the per-letter argmax bar.

    Measured through the production pymdp path on the same held-out pairs with
    the same ``limit``, so both decoders see identical positions.
    """
    sub = part_b["subsample"]
    full = part_b["full"]

    assert sub["model_accuracy"] > sub["baseline_accuracy"], (
        f"STAGE 2 PART B GATE FAILED (pymdp, {sub['n_positions']} positions): "
        f"filtering {100 * sub['model_accuracy']:.3f}% does not beat the "
        f"context-free bar {100 * sub['baseline_accuracy']:.3f}%. A filtering "
        f"decode that cannot beat a lookup table is mis-wired, not merely weak."
    )
    assert full["model_accuracy"] > full["baseline_accuracy"], (
        f"STAGE 2 PART B GATE FAILED (full set, {full['n_positions']} positions): "
        f"{100 * full['model_accuracy']:.3f}% vs "
        f"{100 * full['baseline_accuracy']:.3f}%."
    )

    ControlHarness("stage2_g2p_context_free_gate").assert_gate_is_live(
        full["model_accuracy"],
        full["baseline_accuracy"],
        direction="higher_is_better",
    )
    print(
        f"\nPart B pymdp subsample: {100 * sub['model_accuracy']:.3f}% vs "
        f"{100 * sub['baseline_accuracy']:.3f}% = {sub['margin_pts']:+.3f} pts "
        f"over {sub['n_positions']} positions.\n"
        f"Part B full held-out set: {100 * full['model_accuracy']:.3f}% vs "
        f"{100 * full['baseline_accuracy']:.3f}% = {full['margin_pts']:+.3f} pts "
        f"over {full['n_positions']} positions."
    )


@_SKIP_NO_COUNTS
def test_stage2_g2p_pymdp_matches_oracle(part_b: dict):
    """ORACLE EQUIVALENCE: pymdp and the numpy decoder must agree on every position.

    This is what licenses computing the full-set McNemar with the oracle. A
    full-set pymdp pass is ~42,820 ``infer_states`` calls (about 3 minutes); the
    oracle does it in 0.3 s. That substitution is only legitimate while the two
    decoders are provably identical, so a single disagreement voids the
    significance test in :func:`test_stage2_g2p_margin_is_significant`.
    """
    pymdp_hits = part_b["pymdp_hits"]
    oracle_hits = part_b["oracle_hits_sub"]

    assert len(pymdp_hits) == len(oracle_hits), (
        f"Position counts differ: pymdp {len(pymdp_hits)}, oracle "
        f"{len(oracle_hits)}. The comparison is not paired."
    )
    disagreements = sum(1 for a, b in zip(pymdp_hits, oracle_hits, strict=True) if a != b)
    assert disagreements == 0, (
        f"pymdp and the oracle disagree on {disagreements}/{len(pymdp_hits)} "
        f"positions. The full-set figures no longer describe the pymdp agent, so "
        f"every Part B number computed through the oracle is void."
    )

    # Non-vacuity: the proof must span a real subsample, not two empty lists, and
    # the subsample must not have collapsed to an alphabetical prefix.
    assert len(pymdp_hits) > MIN_EVAL_POSITIONS, (
        f"Equivalence proved over only {len(pymdp_hits)} positions — too few to "
        f"license the full-set substitution."
    )
    initials = {word[0] for word, _ in part_b["subset"] if word}
    assert len(initials) >= 15, (
        f"The pymdp subsample spans only {len(initials)} distinct initial letters "
        f"({sorted(initials)}). g2p_pairs is alphabetically sorted, so this is a "
        f"prefix, not a stride — measured margins on a prefix overstate by ~3x."
    )
    print(
        f"\nDecoder equivalence: 0 disagreements over {len(pymdp_hits)} positions "
        f"spanning {len(initials)} initial letters."
    )


@_SKIP_NO_COUNTS
def test_stage2_g2p_margin_is_significant(part_b: dict):
    """PAIRED McNEMAR (7): the +0.55 pt margin must be significant, not asserted.

    Concordant positions are discarded; only the discordant pairs carry
    information about the difference. ``b + c > 0`` is checked explicitly (inside
    :func:`_mcnemar`) because with no discordant pairs the test is vacuous.

    Reported at two sample sizes, and the smaller one does NOT reach
    significance. That is a real result, not a defect to route around: see the
    module docstring.
    """
    sub = part_b["subsample"]
    full = part_b["full"]

    assert sub["discordant"] > 0 and full["discordant"] > 0, (
        "b + c == 0: no discordant pairs, so McNemar is vacuous."
    )
    assert full["n_positions"] > MIN_EVAL_POSITIONS

    print(
        f"\nMcNemar, pymdp subsample ({sub['n_positions']} positions): "
        f"b={sub['b']} c={sub['c']} net={sub['b'] - sub['c']:+d} "
        f"p={sub['p_value']:.4g} -> "
        f"{'significant' if sub['p_value'] < ALPHA else 'NOT significant'} "
        f"at alpha={ALPHA}\n"
        f"McNemar, full held-out set ({full['n_positions']} positions): "
        f"b={full['b']} c={full['c']} net={full['b'] - full['c']:+d} "
        f"p={full['p_value']:.4g} -> "
        f"{'significant' if full['p_value'] < ALPHA else 'NOT significant'} "
        f"at alpha={ALPHA}"
    )

    assert full["b"] > full["c"], (
        f"The discordant pairs favor the BASELINE: b={full['b']} <= "
        f"c={full['c']}. Filtering is not adding information."
    )
    assert full["p_value"] < ALPHA, (
        f"STAGE 2 PART B SIGNIFICANCE GATE FAILED: full-set McNemar gives "
        f"p={full['p_value']:.4g} >= alpha={ALPHA} with b={full['b']}, "
        f"c={full['c']} over {full['n_positions']} positions. The "
        f"{full['margin_pts']:+.3f} pt margin is not distinguishable from noise. "
        f"Do not lower alpha and do not search for a friendlier subsample — "
        f"report it."
    )

    # The subsample result is asserted as a *finding*, so that a future change
    # which silently made a 2000-word eval sufficient cannot pass unnoticed while
    # the docstring still claims it is underpowered.
    assert sub["p_value"] >= ALPHA, (
        f"The {PYMDP_G2P_WORDS}-word pymdp subsample now reaches p="
        f"{sub['p_value']:.4g} < alpha={ALPHA}. That is good news, but this "
        f"module's docstring documents it as underpowered (measured p=2.9e-02) "
        f"and must be rewritten before this assertion is relaxed."
    )


@_SKIP_NO_COUNTS
def test_stage2_mcnemar_can_return_no_effect(part_b: dict):
    """NEGATIVE CONTROL ON THE STATISTICS (8): the test must be able to say "null".

    Three independent nulls, because a test that only ever returns "significant"
    is rubber-stamping and would make the p-value in
    :func:`test_stage2_g2p_margin_is_significant` meaningless:

    1. **Real underpowered data.** The same procedure on the pymdp subsample
       returns p = 2.9e-02, i.e. not significant at alpha = 0.01.
    2. **Exact null.** A cyclic rotation of the baseline vector against itself
       preserves the number of hits, so b == c exactly and p == 1.0.
    3. **Twenty seeded nulls.** Accuracy-matched independent Bernoulli
       comparators. A single seed would be a cherry-pick; twenty measure the
       false-positive rate against its nominal 1%.
    """
    baseline = np.asarray(part_b["full_base"], dtype=bool)

    # 1. real underpowered data
    assert part_b["subsample"]["p_value"] >= ALPHA

    # 2. exact null via cyclic rotation
    rotated = np.roll(baseline, 977)
    rotation = _mcnemar(baseline.tolist(), rotated.tolist())
    assert rotation["discordant"] > 0, (
        "The rotation null produced no discordant pairs, so it proves nothing about the machinery."
    )
    assert rotation["b"] == rotation["c"], (
        f"A rotation preserves the hit count, so b must equal c; got "
        f"b={rotation['b']}, c={rotation['c']}."
    )
    assert rotation["p_value"] >= ALPHA, (
        f"McNemar called an exact null significant (p={rotation['p_value']:.4g}). "
        f"The machinery is broken and every other p-value here is meaningless."
    )

    # 3. twenty seeded accuracy-matched nulls
    accuracy = float(baseline.mean())
    p_values: list[float] = []
    for seed in range(N_NULL_SEEDS):
        rng = np.random.default_rng(1000 + seed)
        fake = rng.random(baseline.size) < accuracy
        p_values.append(_mcnemar(fake.tolist(), baseline.tolist())["p_value"])

    false_positives = sum(1 for p in p_values if p < ALPHA)
    assert false_positives <= MAX_NULL_FALSE_POSITIVES, (
        f"{false_positives}/{N_NULL_SEEDS} accuracy-matched nulls came out "
        f"significant at alpha={ALPHA} (budget {MAX_NULL_FALSE_POSITIVES}). The "
        f"McNemar procedure is rubber-stamping; the real p-value cannot be "
        f"trusted. p-values: {[f'{p:.3g}' for p in sorted(p_values)]}"
    )
    print(
        f"\nSignificance machinery controls: underpowered real subsample "
        f"p={part_b['subsample']['p_value']:.4g}; rotation null "
        f"b=c={rotation['b']} p={rotation['p_value']:.4g}; "
        f"{false_positives}/{N_NULL_SEEDS} seeded nulls significant at "
        f"alpha={ALPHA} (min p={min(p_values):.4g}, "
        f"median p={sorted(p_values)[N_NULL_SEEDS // 2]:.4g})."
    )


@_SKIP_NO_COUNTS
def test_stage2_g2p_negative_control_uniform_emission(part_b: dict):
    """NEGATIVE CONTROL ON THE MODEL (9): an information-free A must lose.

    A uniform emission carries no letter information at all, so the posterior
    never depends on what was observed. Run through the production pymdp path as
    well as the full-set oracle, so the control targets the real code.
    """
    mapping = part_b["mapping"]
    test = part_b["test"]

    baseline_bar = baseline_accuracy(mapping, test, limit=PYMDP_NEG_WORDS)
    uniform_agent = build_g2p_agent(
        jnp.asarray(part_b["uniform_emission"][np.newaxis, :, :]),
        jnp.asarray(part_b["transition"][np.newaxis, :, :, np.newaxis]),
        jnp.asarray(part_b["start"][np.newaxis, :]),
    )

    ControlHarness("stage2_g2p_uniform_emission").verify_negative_control_fails(
        metric_fn=lambda pairs: g2p_accuracy(uniform_agent, pairs, limit=PYMDP_NEG_WORDS),
        broken_input=test,
        threshold=baseline_bar,
        direction="higher_is_better",
    )

    full_uniform = float(np.asarray(part_b["uniform_a_hits"], dtype=bool).mean())
    full_bar = part_b["full"]["baseline_accuracy"]
    assert full_uniform < full_bar, (
        f"GATE IS VACUOUS: a uniform emission reached "
        f"{100 * full_uniform:.3f}% on the full set, beating the context-free bar "
        f"{100 * full_bar:.3f}%."
    )
    print(
        f"\nUniform emission: {100 * full_uniform:.3f}% (full set) vs bar "
        f"{100 * full_bar:.3f}% — caught."
    )


@_SKIP_NO_COUNTS
def test_stage2_g2p_negative_control_anti_informative_transition(part_b: dict):
    """ANTI-INFORMATIVE WIRING (10): backward dynamics in B must fail the bar.

    The Stage 1 analogue at this level: a time-reversed B means "observing a
    letter tells you which phone came *before* it", carried forward as the
    prediction for the phone that comes *after*. Column-stochastic, accepted by
    ``_validate()``, and quietly wrong.

    Also pins the slots structurally: a square ``(1, 39, 39)`` matrix must be
    *rejected* in the emission slot, so the two matrices cannot be swapped
    silently the way Stage 1's could.
    """
    mapping = part_b["mapping"]
    test = part_b["test"]

    baseline_bar = baseline_accuracy(mapping, test, limit=PYMDP_NEG_WORDS)
    reversed_agent = build_g2p_agent(
        jnp.asarray(part_b["emission"][np.newaxis, :, :]),
        jnp.asarray(part_b["reversed_transition"][np.newaxis, :, :, np.newaxis]),
        jnp.asarray(part_b["start"][np.newaxis, :]),
    )

    ControlHarness("stage2_g2p_reversed_transition").verify_negative_control_fails(
        metric_fn=lambda pairs: g2p_accuracy(reversed_agent, pairs, limit=PYMDP_NEG_WORDS),
        broken_input=test,
        threshold=baseline_bar,
        direction="higher_is_better",
    )

    full_reversed = float(np.asarray(part_b["reversed_b_hits"], dtype=bool).mean())
    full_bar = part_b["full"]["baseline_accuracy"]
    assert full_reversed < full_bar, (
        f"GATE IS VACUOUS: time-reversed dynamics reached "
        f"{100 * full_reversed:.3f}% on the full set, beating the context-free bar "
        f"{100 * full_bar:.3f}%. The gate no longer distinguishes forward from "
        f"backward wiring."
    )

    # The slots must not be interchangeable by shape, unlike Stage 1's square
    # letter matrices. A (1, 39, 39) transition-shaped matrix in the emission slot
    # is a hard error, not a silently valid alternative.
    with pytest.raises(ValueError):
        build_g2p_agent(
            jnp.asarray(part_b["transition"][np.newaxis, :, :]),
            jnp.asarray(part_b["transition"][np.newaxis, :, :, np.newaxis]),
            jnp.asarray(part_b["start"][np.newaxis, :]),
        )
    print(
        f"\nTime-reversed B: {100 * full_reversed:.3f}% (full set) vs bar "
        f"{100 * full_bar:.3f}% — caught; emission/transition slots are not "
        f"shape-interchangeable."
    )


@_SKIP_NO_COUNTS
def test_stage2_g2p_emission_orientation_is_pinned(part_b: dict):
    """The accuracy bar cannot catch a mis-normalized A — value equality can.

    An emission normalized over phones and then renormalized over letters passes
    pymdp's validation *and* clears the context-free accuracy bar (measured
    70.56% against a 70.255% bar). "Beats the baseline" is therefore necessary but
    not sufficient, so the shipped emission is compared value-for-value against an
    independent numpy re-derivation of ``P(letter | phone)``.

    Both controls are here: the guard accepts the correct matrix (positive) and
    rejects the mis-normalized one (negative).
    """
    shipped = part_b["emission"]
    independent = _independent_emission(part_b["train"])
    mis_normalized = part_b["mis_normalized"]

    np.testing.assert_allclose(
        shipped,
        independent,
        atol=1e-6,
        err_msg=(
            "build_g2p_emission does not reproduce an independent numpy "
            "derivation of A[l, p] = P(letter l | phone p). The emission is "
            "mis-normalized, transposed, or mis-indexed."
        ),
    )

    deviation = float(np.max(np.abs(mis_normalized - independent)))
    assert deviation > 1e-3, (
        f"GUARD IS VACUOUS: the mis-normalized emission differs from the correct "
        f"one by only {deviation:.2e}, so value equality cannot detect it."
    )

    # And the reason this guard is needed at all: accuracy does not catch it.
    mis_accuracy = float(np.asarray(part_b["mis_normalized_hits"], dtype=bool).mean())
    correct_accuracy = part_b["full"]["model_accuracy"]
    assert mis_accuracy < correct_accuracy, (
        f"The mis-normalized emission scored {100 * mis_accuracy:.3f}%, at or "
        f"above the correct wiring's {100 * correct_accuracy:.3f}%. Then the "
        f"normalization is not what the docstring says it is."
    )
    caught_by_accuracy = mis_accuracy < part_b["full"]["baseline_accuracy"]
    print(
        f"\nMis-normalized A: {100 * mis_accuracy:.3f}% vs bar "
        f"{100 * part_b['full']['baseline_accuracy']:.3f}% and correct "
        f"{100 * correct_accuracy:.3f}%. Caught by the accuracy bar: "
        f"{caught_by_accuracy}; caught by value re-derivation: True "
        f"(max deviation {deviation:.3e})."
    )


@_SKIP_NO_COUNTS
def test_stage2_g2p_decode_length_invariant(part_b: dict):
    """LENGTH INVARIANT (11): the paired vectors must be equal-length and large.

    An unnoticed truncation would silently shrink the eval and inflate the
    p-value; a length mismatch would make the pairing — and therefore McNemar —
    meaningless.
    """
    pymdp_hits = part_b["pymdp_hits"]
    base_hits = part_b["base_hits_sub"]
    full_model = part_b["full_model"]
    full_base = part_b["full_base"]

    assert len(pymdp_hits) == len(base_hits), (
        f"pymdp produced {len(pymdp_hits)} positions but the baseline produced "
        f"{len(base_hits)} on the same pairs and limit. McNemar is invalid."
    )
    assert len(full_model) == len(full_base), (
        f"Full-set vectors are unpaired: {len(full_model)} vs {len(full_base)}."
    )
    assert len(pymdp_hits) > MIN_EVAL_POSITIONS, (
        f"Only {len(pymdp_hits)} pymdp positions evaluated (need > "
        f"{MIN_EVAL_POSITIONS}) — the evaluation was silently truncated."
    )
    assert len(full_model) > MIN_EVAL_POSITIONS, (
        f"Only {len(full_model)} full-set positions (need > {MIN_EVAL_POSITIONS})."
    )
    # The full set must be strictly larger than the subsample, or the "full-set"
    # figures are just the subsample under another name.
    assert len(full_model) > len(pymdp_hits), (
        f"The 'full' set ({len(full_model)}) is not larger than the pymdp "
        f"subsample ({len(pymdp_hits)})."
    )
    print(
        f"\nEval sizes: pymdp subsample {len(pymdp_hits)} positions "
        f"({len(part_b['subset'])} words), full held-out set {len(full_model)} "
        f"positions ({len(part_b['test'])} words)."
    )


@_SKIP_NO_COUNTS
def test_stage2_g2p_public_accuracy_api_matches_paired_positions(part_b: dict):
    """``g2p_accuracy`` / ``baseline_accuracy`` must equal the paired-position means.

    The gate reads accuracies off the position vectors; this binds those numbers
    to the public API so the two cannot drift apart.
    """
    small = 120
    model_positions = decode_positions(part_b["agent"], part_b["test"], limit=small)
    base_positions_small = baseline_positions(part_b["mapping"], part_b["test"], limit=small)

    assert model_positions, "The small decode produced no positions."
    assert len(model_positions) == len(base_positions_small)
    assert g2p_accuracy(part_b["agent"], part_b["test"], limit=small) == pytest.approx(
        sum(model_positions) / len(model_positions), abs=1e-12
    )
    assert baseline_accuracy(part_b["mapping"], part_b["test"], limit=small) == pytest.approx(
        sum(base_positions_small) / len(base_positions_small), abs=1e-12
    )


@_SKIP_NO_COUNTS
def test_stage2_g2p_split_is_disjoint_and_positional(counts: dict, part_b: dict):
    """The train/test split must be positional, disjoint, and non-degenerate."""
    train, test = part_b["train"], part_b["test"]
    stride = counts["test_every"]

    assert len(train) + len(test) == len(counts["g2p_pairs"])
    assert len(test) == sum(1 for i in range(len(counts["g2p_pairs"])) if i % stride == 0)
    assert not (set(w for w, _ in train) & set(w for w, _ in test)), (
        "Train and test share words — the held-out accuracy is contaminated."
    )
    assert len(test) > 1000, f"Only {len(test)} held-out pairs — too few to gate on."


# ══ CLAIMS LEDGER ═════════════════════════════════════════════════════════════


@_SKIP_NO_COUNTS
def test_stage2_claims_recorded(part_a: dict, part_b: dict, bounds: dict[str, float], counts: dict):
    """(12) Every Stage 2 number agrees with THIS run's measurements.

    This is verify-only. ``scripts/record_stage2_claims.py`` owns every Stage 2
    label, including the distinct length-matched and underpowered-subsample
    populations. An ordinary test writing the project ledger performs an
    unlocked whole-file replacement during the suite and makes concurrent test
    outcomes order-dependent, even when its values happen to agree.
    """
    if LEDGER_FILE.exists():
        raw = json.loads(LEDGER_FILE.read_text(encoding="utf-8"))
        assert isinstance(raw, dict) and "claims" in raw, (
            f"{LEDGER_FILE} is missing the 'claims' key (content starts {json.dumps(raw)[:40]!r})."
        )

    full = part_b["full"]
    sub = part_b["subsample"]
    disagreements = sum(
        1 for a, b in zip(part_b["pymdp_hits"], part_b["oracle_hits_sub"], strict=True) if a != b
    )
    all_position_floor, all_position_floor_n = full_corpus_order1_floor_perplexity(counts)
    full_model, full_model_n = full_corpus_perplexity(counts)
    assert all_position_floor_n == full_model_n
    assert full_model >= all_position_floor

    measured: dict[str, float | int] = {
        # Part A
        "stage2_phoneme_subset_perplexity": round(part_a["oracle_full"], 4),
        "stage2_phoneme_subset_unigram_baseline": round(part_a["unigram_full"], 4),
        "stage2_phoneme_uniform_baseline": round(part_a["pymdp_uniform_sample"], 4),
        "stage2_phoneme_all_position_floor_perplexity": round(all_position_floor, 4),
        "stage2_phoneme_all_position_max_gain_nats": round(
            math.log(N_PHONES) - math.log(all_position_floor), 4
        ),
        "stage2_phoneme_floor_perplexity": round(bounds["floor_perplexity"], 4),
        "stage2_phoneme_max_gain_nats": round(bounds["max_gain_nats"], 4),
        # Part B
        "stage2_g2p_accuracy": round(100.0 * full["model_accuracy"], 3),
        "stage2_g2p_baseline_accuracy": round(100.0 * full["baseline_accuracy"], 3),
        "stage2_g2p_margin_pts": round(full["margin_pts"], 3),
        "stage2_g2p_mcnemar_p": float(f"{full['p_value']:.4g}"),
        "stage2_g2p_mcnemar_b": full["b"],
        "stage2_g2p_mcnemar_c": full["c"],
        "stage2_g2p_eval_positions": full["n_positions"],
        "stage2_g2p_train_pairs": len(part_b["train"]),
        "stage2_g2p_test_pairs": len(part_b["test"]),
        "stage2_g2p_split_stride": counts["test_every"],
        "stage2_g2p_length_matched_pct": round(part_b["length_matched_pct"], 1),
        "stage2_g2p_subsample_positions": sub["n_positions"],
        "stage2_g2p_subsample_mcnemar_p": float(f"{sub['p_value']:.4g}"),
        "stage2_g2p_subsample_mcnemar_b": sub["b"],
        "stage2_g2p_subsample_mcnemar_c": sub["c"],
        "stage2_decoder_disagreements": disagreements,
    }

    ledger = load_ledger(PROJECT_ROOT)
    for label, value in measured.items():
        entry = ledger.get(label)
        assert entry is not None, (
            f"{label} is absent from the ledger. Run "
            f"scripts/record_stage2_claims.py; tests may not write it."
        )
        assert entry["source"] == "scripts/record_stage2_claims.py", (
            f"{label} was last written by {entry['source']!r}; the Stage 2 "
            "recorder must be its sole writer."
        )
        assert entry["value"] == pytest.approx(value, rel=1e-4), (
            f"Ledger {label}={entry['value']!r} does not match this run's "
            f"measurement {value!r}. Re-run scripts/record_stage2_claims.py — and "
            f"check nothing else writes this label."
        )

    # Cross-checks so the ledger cannot hold an internally inconsistent story.
    assert (
        measured["stage2_phoneme_subset_perplexity"]
        < measured["stage2_phoneme_subset_unigram_baseline"]
        < measured["stage2_phoneme_uniform_baseline"]
    ), "Recorded Part A ordering is not model < unigram < uniform."

    # ── ONE WRITER PER LABEL ─────────────────────────────────────────────────
    # This test measures Part A over the length-matched g2p_pairs subset. The
    # canonical stage2_phoneme_perplexity is measured over the FULL dictionary in
    # closed form by scripts/record_stage2_claims.py, and the two differ by ~1.46
    # perplexity units because the subset is biased toward short, transparent
    # orthography. Both are legitimate; sharing one label is not. When this test
    # also wrote the canonical label, the ledger value — and therefore the
    # manuscript's prose — depended on whether the test or the recorder ran last.
    # So: verify the canonical value here, never overwrite it.
    canonical = ledger.get("stage2_phoneme_perplexity")
    if canonical is not None:
        fresh, _ = full_corpus_perplexity(counts)
        assert canonical["value"] == pytest.approx(fresh, rel=1e-4), (
            f"Ledger stage2_phoneme_perplexity={canonical['value']} does not match a "
            f"fresh full-dictionary measurement {fresh:.4f}. Re-run "
            f"scripts/record_stage2_claims.py — and check nothing else is writing "
            f"this label."
        )
        assert canonical["source"].startswith("scripts/"), (
            f"stage2_phoneme_perplexity was last written by "
            f"{canonical['source']!r}. Only the recorder scripts may write canonical "
            f"claim values; tests must verify them."
        )
    assert measured["stage2_g2p_accuracy"] > measured["stage2_g2p_baseline_accuracy"]
    assert measured["stage2_g2p_mcnemar_p"] < ALPHA
    assert measured["stage2_g2p_subsample_mcnemar_p"] >= ALPHA
    assert measured["stage2_g2p_subsample_mcnemar_p"] == pytest.approx(
        binomtest(
            int(measured["stage2_g2p_subsample_mcnemar_b"]),
            int(measured["stage2_g2p_subsample_mcnemar_b"])
            + int(measured["stage2_g2p_subsample_mcnemar_c"]),
            0.5,
        ).pvalue,
        rel=1e-4,
    )
    assert measured["stage2_decoder_disagreements"] == 0
    assert measured["stage2_g2p_train_pairs"] > measured["stage2_g2p_test_pairs"]
    assert 0.0 < measured["stage2_g2p_length_matched_pct"] < 100.0

    width = max(len(k) for k in measured)
    print("\nStage 2 claims (all verified against the recorder):")
    for label, value in measured.items():
        print(f"  v {label:<{width}} = {value}")


# ══ STRUCTURAL CHECKS (no vendored corpus required) ═══════════════════════════


def test_stage2_module_exports_are_importable():
    """``levels.phoneme`` must expose the full Stage 2 surface."""
    from ailm.levels import phoneme

    for name in (
        "PHONES",
        "PHONE_TO_IDX",
        "IDX_TO_PHONE",
        "N_PHONES",
        "ALPHA_LIST",
        "ALPHA_TO_IDX",
        "N_SYMBOLS",
        "load_phoneme_counts",
        "build_phone_transition",
        "build_phone_unigram",
        "build_phoneme_agent",
        "compute_phoneme_perplexity",
        "g2p_split",
        "build_g2p_emission",
        "build_g2p_transition",
        "build_g2p_start",
        "build_g2p_agent",
        "decode_phonemes",
        "decode_positions",
        "g2p_accuracy",
        "per_letter_argmax_baseline",
        "baseline_accuracy",
        "baseline_positions",
        "PARTA_HELDOUT_STRIDE",
        "part_a_counts",
        "part_a_split_perplexity",
        "phoneme_heldout_split",
    ):
        assert hasattr(phoneme, name), f"ailm.levels.phoneme is missing {name}"

    assert len(PHONES) == N_PHONES == 39
    assert len(PHONE_TO_IDX) == 39


def test_stage2_gate_constants_are_derived_not_typed():
    """The gate arithmetic must stay tied to the inventory size, not a literal."""
    assert float(len(PHONES)) == UNIFORM_PERPLEXITY
    assert pytest.approx(math.exp(math.log(N_PHONES))) == UNIFORM_PERPLEXITY
    assert 0.0 < FLOOR_TOLERANCE < 1.0
    assert 0.0 < ALPHA < 1.0


# ══ PART A HELD-OUT WORD SPLIT (pre-registered arm) ══════════════════════════


#: Relative tolerance for the positive control against the in-sample closed
#: form. Both sides share the same float32-rounded matrices, so they agree far
#: below this bound; the value is the pre-registered one.
PARTA_ANCHOR_REL = 1e-9

#: Relative tolerance for agreement with a float64 sequence walk. The closed
#: forms log float32 probabilities (`np.log` on a float32 scalar computes in
#: single precision), so float64-walk agreement is bounded near 1e-7 — the
#: same reality that sets `FLOAT32_REL` in test_phoneme_full_corpus.py. The
#: positive control against the in-sample closed form is exempt: both sides
#: take the same float32 logs, so they agree bit for bit at `PARTA_ANCHOR_REL`.
PARTA_WALK_REL = 1e-6

#: The held-out negative control may not beat the bar by more than this many
#: nats — the same pre-registered bound the recorder enforces on the real
#: split. An order of magnitude below the predicted honest gap, and well above
#: the estimation noise a within-word shuffle can leave behind.
PARTA_SHUFFLED_GAP_MAX_NATS = 0.05


def _chain_corpus(n_words: int, seed: int) -> list[list[str]]:
    """Synthetic words with real transition structure and a fixed onset.

    Every word starts with ``B`` and then follows a biased 4-phone chain, so an
    order-1 fit carries genuine left-context information the context-free bar
    cannot have, and the word-initial distribution differs sharply from the
    within-word phone composition — which is what the negative control's
    shuffle destroys.

    Args:
        n_words: Number of words to generate.
        seed: Seed for the word-length draw. Phone content is deterministic.

    Returns:
        Phone sequences over canonical ARPAbet symbols.
    """
    rng = np.random.default_rng(seed)
    successor = {"B": "AO", "AO": "AE", "AE": "AA", "AA": "AA"}
    words: list[list[str]] = []
    for _ in range(n_words):
        word = ["B"]
        for _ in range(int(rng.integers(2, 5))):
            word.append(successor[word[-1]])
        words.append(word)
    return words


def test_parta_heldout_split_is_a_stride_not_a_prefix():
    """The split rule: an evenly spaced stride over the caller's order."""
    words = [f"w{i:03d}" for i in range(50)]
    train, held = phoneme_heldout_split(words)
    assert held == words[::5]
    assert train == [word for i, word in enumerate(words) if i % 5 != 0]
    assert not set(train) & set(held)
    assert set(train) | set(held) == set(words)
    # A prefix would put the held-out side entirely at the front of a sorted
    # list; the stride spans the whole range (decision 4).
    assert held != words[: len(held)]
    assert held[-1] == words[-5]
    # Deterministic: same order in, same split out.
    assert phoneme_heldout_split(words) == (train, held)
    # A custom stride is honored and keeps both sides live.
    train7, held7 = phoneme_heldout_split(words, stride=7)
    assert held7 == words[::7] and len(train7) + len(held7) == len(words)


def test_parta_heldout_split_rejects_invalid_strides():
    """Stride validation mirrors ``g2p_split``: positive integer, both sides."""
    words = ["w0", "w1", "w2"]
    for bad_stride in (0, -2, 1.5, True):
        with pytest.raises(ValueError, match="stride"):
            phoneme_heldout_split(words, stride=bad_stride)
    with pytest.raises(ValueError, match="empty side"):
        phoneme_heldout_split(words, stride=1)
    with pytest.raises(ValueError, match="empty side"):
        phoneme_heldout_split(["only"], stride=5)


@_SKIP_NO_COUNTS
def test_parta_split_scoring_reproduces_the_in_sample_closed_form(counts: dict):
    """POSITIVE CONTROL — the split-aware closed form is the in-sample metric.

    Scoring the in-sample protocol's own population (fit and scored as one
    sample) must reproduce claim:stage2_phoneme_perplexity and its bar to the
    pre-registered `1e-9` relative, with the identical position count. That is
    what licenses reading the held-out difference as a split effect rather
    than a code-path difference.
    """
    model, bar, positions = part_a_split_perplexity(counts, counts)
    full_model, full_positions = full_corpus_perplexity(counts)
    assert positions == full_positions
    assert model == pytest.approx(full_model, rel=PARTA_ANCHOR_REL)
    assert bar == pytest.approx(full_corpus_unigram_perplexity(counts), rel=PARTA_ANCHOR_REL)


def test_parta_split_scoring_matches_a_sequence_walk():
    """The split scorer agrees with a naive position-by-position walk.

    Hand-built corpus, so the exact scored positions are known — the same
    pattern the full-corpus closed form is verified with in
    ``tests/test_phoneme_full_corpus.py``, written out here because that
    helper is that file's local oracle.
    """
    sequences = _chain_corpus(300, seed=7)
    fit_counts = part_a_counts(sequences[:200])
    score_counts = part_a_counts(sequences[200:])
    model, bar, positions = part_a_split_perplexity(fit_counts, score_counts)
    assert positions == sum(len(seq) for seq in sequences[200:])

    transition = np.asarray(build_phone_transition(fit_counts["phone_bigrams"]))[0, :, :, 0]
    start = np.asarray(build_phone_unigram(fit_counts["phone_unigrams"])).ravel()
    total_logp = 0.0
    for seq in sequences[200:]:
        prior = start.copy()
        for phone in seq:
            idx = PHONE_TO_IDX[phone]
            total_logp += math.log(max(float(prior[idx]), 1e-300))
            posterior = np.zeros(N_PHONES)
            posterior[idx] = 1.0
            prior = transition @ posterior
    assert model == pytest.approx(math.exp(-total_logp / positions), rel=PARTA_WALK_REL)

    bar_logp = sum(
        math.log(float(start[PHONE_TO_IDX[phone]])) for seq in sequences[200:] for phone in seq
    )
    assert bar == pytest.approx(math.exp(-bar_logp / positions), rel=PARTA_WALK_REL)


def test_parta_heldout_gap_gate_is_live_and_redeal_control_fails():
    """The held-out gap discriminates real structure from a structure-free fit.

    POSITIVE CONTROL: the honest fit beats the re-fit bar on held-out words.
    DISCRIMINATION: adjacency carries information beyond the bag — the honest
    gap exceeds the within-word-shuffled fit's gap, whose fit preserves each
    word's phone content but destroys its order.
    NEGATIVE CONTROL: a model fit on phones re-dealt across word boundaries —
    same corpus unigram, same word lengths, no surviving structure — must not
    beat the bar by more than the pre-registered bound. The harness pins the
    live direction against the structure-free fit.
    """
    sequences = _chain_corpus(2000, seed=11)
    words = [f"w{i}" for i in range(len(sequences))]
    train_words, held_words = phoneme_heldout_split(words)
    train = [sequences[int(word[1:])] for word in train_words]
    held = [sequences[int(word[1:])] for word in held_words]

    fit_counts = part_a_counts(train)
    held_counts = part_a_counts(held)
    model, bar, positions = part_a_split_perplexity(fit_counts, held_counts)
    gap = math.log(bar) - math.log(model)
    assert positions == sum(len(seq) for seq in held)
    assert gap > 0.0, "the honest fit must beat the context-free bar on held-out words"

    shuffled = [[seq[i] for i in np.random.default_rng(0).permutation(len(seq))] for seq in train]
    shuf_model, shuf_bar, shuf_positions = part_a_split_perplexity(
        part_a_counts(shuffled), held_counts
    )
    assert shuf_positions == positions
    shuf_gap = math.log(shuf_bar) - math.log(shuf_model)
    assert gap > shuf_gap, "adjacency must carry information beyond the bag"

    stream = [phone for seq in train for phone in seq]
    np.random.default_rng(0).shuffle(stream)
    redealt: list[list[str]] = []
    position = 0
    for seq in train:
        redealt.append(stream[position : position + len(seq)])
        position += len(seq)
    redeal_model, redeal_bar, redeal_positions = part_a_split_perplexity(
        part_a_counts(redealt), held_counts
    )
    assert redeal_positions == positions
    redeal_gap = math.log(redeal_bar) - math.log(redeal_model)

    harness = ControlHarness("parta_heldout_gap_gate")
    harness.assert_gate_is_live(good_value=gap, bad_value=redeal_gap, direction="higher_is_better")
    assert redeal_gap <= PARTA_SHUFFLED_GAP_MAX_NATS, (
        f"re-dealt fit beats the bar by {redeal_gap:.4f} nats; the re-deal did "
        "not destroy the corpus's structure"
    )


@_SKIP_NO_COUNTS
def test_stage2_heldout_claims_are_self_consistent():
    """The recorded held-out arm tells one internally consistent story.

    Verify-only: the recorder owns the labels. This checks the recorded values
    against each other and against the pre-registered prediction — not against
    a re-measurement, because the raw per-word pronunciations the arm needs are
    a recorder input, not a test fixture.
    """
    assert LEDGER_FILE.exists()
    ledger = load_ledger(PROJECT_ROOT)

    def owned_value(label: str) -> float | int:
        entry = ledger.get(label)
        assert entry is not None, (
            f"{label} is absent from the ledger. Run "
            f"scripts/record_stage2_claims.py; tests may not write it."
        )
        assert entry["source"] == "scripts/record_stage2_claims.py", (
            f"{label} was last written by {entry['source']!r}; the Stage 2 "
            "recorder must be its sole writer."
        )
        return entry["value"]

    stride = owned_value("stage2_phoneme_heldout_split_stride")
    train_words = owned_value("stage2_phoneme_heldout_train_words")
    test_words = owned_value("stage2_phoneme_heldout_test_words")
    positions = owned_value("stage2_phoneme_heldout_eval_positions")
    model = owned_value("stage2_phoneme_heldout_perplexity")
    bar = owned_value("stage2_phoneme_heldout_unigram_baseline")
    gap = owned_value("stage2_phoneme_heldout_gap_nats")
    penalty = owned_value("stage2_phoneme_heldout_penalty_ppl")
    shuf_gap = owned_value("stage2_phoneme_heldout_shuffled_gap_nats")
    redeal_gap = owned_value("stage2_phoneme_heldout_redeal_gap_nats")

    in_sample = ledger.get("stage2_phoneme_perplexity")
    assert in_sample is not None, "stage2_phoneme_perplexity must be recorded first"
    base_words = ledger.get("cmudict_base_word_count")
    assert base_words is not None, "cmudict_base_word_count must be recorded first"

    # Split bookkeeping: the recorded stride over the whole dictionary, with
    # both sides live.
    assert stride == PARTA_HELDOUT_STRIDE
    assert test_words == len(range(0, base_words["value"], stride))
    assert train_words + test_words == base_words["value"]
    assert train_words > test_words > 0
    assert 0 < positions < owned_value("stage2_phoneme_eval_positions")

    # The recorded gap is the log ratio of the two recorded perplexities, and
    # the penalty is the distance above the in-sample number.
    assert gap == pytest.approx(math.log(bar) - math.log(model), abs=1e-3)
    assert penalty == pytest.approx(model - in_sample["value"], abs=1e-3)

    # The pre-registered prediction: held-out above in-sample by a bounded
    # amount, the bar still beaten, adjacency worth more than the bag, and the
    # re-dealt structure-free control nowhere near the honest margin. The
    # within-word-shuffled gap is a recorded finding, not a gated bound — the
    # first pre-registered control assumed it must lose and it fired; see the
    # held-out decision entry.
    assert 0.0 < penalty < 1.0
    assert model < bar
    assert gap > 0.0
    assert gap > shuf_gap
    assert redeal_gap <= PARTA_SHUFFLED_GAP_MAX_NATS


# ── THE DEGENERATE-INPUT DEFAULT MUST SIT ON THE FAILING SIDE ────────────────


def test_an_empty_phone_population_cannot_produce_a_passing_perplexity(part_a):
    """``compute_phoneme_perplexity`` returns the uniform value, never a passing one.

    An empty population is a documented refusal rather than a raise, because
    ``tests/test_phoneme.py`` pins ``float(N_PHONES)`` as the contract while the
    oracle raises on an empty sequence. The disagreement is safe only for as long
    as the returned value stays on the failing side of the bar, which is what is
    asserted here — on the gate, not on the function, so moving the default off
    the uniform value fails this even though the contract tests still pass.
    """
    from ailm.gates import evaluate_gate

    agent = part_a["uniform_agent"]
    for degenerate in ([], [[]], [[], []]):
        measured = compute_phoneme_perplexity(agent, degenerate)
        assert measured == UNIFORM_PERPLEXITY, (
            f"{degenerate!r} scored {measured}, not the uniform value; the "
            f"refusal is no longer pinned to the failing side of the bar"
        )
        outcome = evaluate_gate(
            baseline=UNIFORM_PERPLEXITY,
            model=measured,
            ceiling=None,
            threshold=UNIFORM_PERPLEXITY,
            direction="lower_is_better",
        )
        assert not outcome.passes, (
            f"an empty population ({degenerate!r}) cleared a bar set at the "
            f"uniform value; the default is fail-open"
        )
