#!/usr/bin/env python3
"""Record Stage 2 quantitative claims into data/claims.json.

Every number appearing in Stage 2 prose is derived here. Nothing is hand-typed.

## Why the G2P figures are computed through the oracle

The honest G2P margin is about half a percentage point, and a paired McNemar test
only reaches significance on the **full** held-out set (42,820 positions). Smaller
subsamples are underpowered:

    full test set          +0.546 pts   p = 3.5e-06   significant
    stride  test[::3]      +0.489 pts   p = 1.7e-02   NOT significant
    prefix  test[:1200]    +1.691 pts   p = 6.1e-08   significant for the WRONG
                                                      reason — that slice is only
                                                      a/b/c words and inflates the
                                                      margin 3.1x

So the gate needs the full set, but a full-set pymdp pass is ~42.8k
``infer_states`` calls. The resolution is not to shrink the eval set: it is to
prove the two decoders are *identical* on a bounded subset, then use the fast
numpy oracle for the full-set statistics. ``verify_decoder_equivalence`` below is
that proof, and it must show ZERO position disagreements — if it ever does not,
the full-set numbers are void because they no longer describe the pymdp agent.

## The held-out Part A word split

The full-dictionary Part A numbers are an in-sample dictionary protocol
(decision 6): the same pronunciations fit the model and are scored. The
held-out arm below re-fits the identical add-one-smoothed model on a stride
sample of the sorted dictionary and scores the closed form on the disjoint
remainder, recorded beside the in-sample number. Its split rule, prediction,
falsifier, and both controls were pre-registered in its decision entry before
the measurement ran.

Run:  .venv/bin/python scripts/record_stage2_claims.py
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402

warnings.filterwarnings("ignore", message=".*JAX array is being set as static.*")

import jax.numpy as jnp  # noqa: E402

from ailm.claims import load_ledger  # noqa: E402
from ailm.corpus.cmudict import _strip_stress, load_pinned_cmudict_source  # noqa: E402
from ailm.gates import (  # noqa: E402
    STAGE2_G2P_GATE_SESOI_PTS,
    STAGE2_PHONEME_GATE_SESOI_NATS,
)
from ailm.levels.letter import ALPHA_TO_IDX  # noqa: E402
from ailm.levels.phoneme import (  # noqa: E402
    G2P_FAMILY_PREFIX_LENGTH,
    N_PHONES,
    PARTA_HELDOUT_STRIDE,
    PHONE_TO_IDX,
    PHONES,
    _iter_pairs,
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
    g2p_split,
    load_phoneme_counts,
    part_a_counts,
    part_a_split_perplexity,
    per_letter_argmax_baseline,
    phoneme_heldout_split,
)
from ailm.statistics import exact_two_sided_binomial_p  # noqa: E402
from ailm.verification.oracle import (  # noqa: E402
    forward_backward,
    forward_filter,
    viterbi,
)

RUN_ID = "stage2_phoneme_gate"
SOURCE = "scripts/record_stage2_claims.py"

#: Words used for the pymdp-vs-oracle equivalence proof. Bounded for speed; the
#: proof is an identity check, so it does not need to be a representative sample.
EQUIVALENCE_WORDS = 2000

#: Sequences used for the phoneme perplexity measurement (pymdp, so bounded).
#: A deterministic STRIDE, not a prefix — g2p_pairs is alphabetically sorted, so a
#: prefix would cover only early-alphabet words.
PERPLEXITY_STRIDE = 13

#: Seed for the Part A held-out controls, which shuffle the training phones.
#: Fixed so both controls are reproducible.
PARTA_SHUFFLE_SEED = 0

#: Relative tolerance for the Part A held-out positive control: scoring the
#: in-sample protocol's own population through the split-aware closed form
#: must reproduce the in-sample closed form and its bar.
PARTA_ANCHOR_REL_TOL = 1e-9

#: The Part A held-out negative control (cross-word re-deal) must not beat the
#: held-out unigram bar by more than this many nats; a larger gap would mean
#: the re-deal failed to destroy the corpus's structure.
PARTA_SHUFFLED_GAP_MAX_NATS = 0.05


def phone_information_bounds(counts: dict) -> dict[str, float]:
    """Uniform / unigram / order-1 conditional entropies of the phone bigrams."""
    joint = np.zeros((N_PHONES, N_PHONES))
    for key, cnt in counts["phone_bigrams"].items():
        prev, nxt = key.split(",", 1)
        if prev in PHONE_TO_IDX and nxt in PHONE_TO_IDX:
            joint[PHONE_TO_IDX[nxt], PHONE_TO_IDX[prev]] += cnt
    joint /= joint.sum()
    p_prev = joint.sum(axis=0)

    h_cond = 0.0
    for j in range(N_PHONES):
        if p_prev[j] <= 0:
            continue
        col = joint[:, j]
        nz = col > 0
        h_cond -= float(np.sum(col[nz] * np.log(col[nz] / p_prev[j])))

    uni = np.array([counts["phone_unigrams"].get(p, 0) + 1.0 for p in PHONES], dtype=float)
    uni /= uni.sum()
    h_unigram = -float(np.sum(uni * np.log(uni)))
    h_uniform = math.log(N_PHONES)

    return {
        "h_uniform_nats": h_uniform,
        "h_unigram_nats": h_unigram,
        "h_cond_nats": h_cond,
        "ppl_floor": math.exp(h_cond),
        "max_gain_nats": h_uniform - h_cond,
    }


def length_matched_phone_perplexities(counts: dict) -> tuple[float, float, int]:
    """Model and unigram perplexity on every length-matched G2P phone.

    This is the subset population reported beside the full-dictionary Part A
    result. It is evaluated in closed form from the same smoothed ``B`` and ``D``
    matrices the Agent receives, with ``D`` reset at every word boundary.
    """
    transition = np.asarray(build_phone_transition(counts["phone_bigrams"]))[0, :, :, 0].astype(
        np.float64
    )
    transition /= transition.sum(axis=0, keepdims=True)
    start = np.asarray(build_phone_unigram(counts["phone_unigrams"])).ravel().astype(np.float64)
    start /= start.sum()
    unigram = np.asarray(
        [counts["phone_unigrams"].get(phone, 0) + 1.0 for phone in PHONES],
        dtype=np.float64,
    )
    unigram /= unigram.sum()

    model_logp = unigram_logp = 0.0
    positions = 0
    for _word, phones_raw in counts["g2p_pairs"]:
        phones = phones_raw.split() if isinstance(phones_raw, str) else list(phones_raw)
        if not phones:
            continue
        indices = [PHONE_TO_IDX[phone] for phone in phones]
        model_logp += math.log(float(start[indices[0]]))
        unigram_logp += sum(math.log(float(unigram[index])) for index in indices)
        model_logp += sum(
            math.log(float(transition[current, previous]))
            for previous, current in zip(indices[:-1], indices[1:], strict=True)
        )
        positions += len(indices)
    if positions == 0:
        raise ValueError("length-matched G2P population has no scorable phone positions")
    return (
        math.exp(-model_logp / positions),
        math.exp(-unigram_logp / positions),
        positions,
    )


def _oracle_matrices(train_pairs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The G2P model as plain numpy, in the oracle's expected orientation."""
    emission = np.array(build_g2p_emission(train_pairs))[0]  # (30, 39)
    transition = np.array(build_g2p_transition(train_pairs))[0, :, :, 0]  # (39, 39)
    start = np.array(build_g2p_start(train_pairs)).ravel()  # (39,)
    return emission, transition, start


def oracle_decode(emission, transition, start, word: str) -> list[str]:
    """Filtered argmax decode via the independent numpy oracle."""
    idx = [ALPHA_TO_IDX[c] for c in word if c in ALPHA_TO_IDX]
    if not idx:
        return []
    posteriors, _ = forward_filter(emission, transition, start, idx)
    return [PHONES[int(np.argmax(row))] for row in posteriors]


def smoothing_and_viterbi_ceilings(
    emission: np.ndarray,
    transition: np.ndarray,
    start: np.ndarray,
    pairs,
    mapping: dict[str, str],
) -> dict[str, float]:
    """Whole-word decoding ceilings for G2P, over the full held-out set.

    Forward filtering conditions only on observations up to the current position,
    which is what a pymdp ``Agent`` does natively and what the streaming
    architecture will compose with. Forward-backward smoothing and Viterbi both see
    the entire word, so they bound how much the extra context is worth.

    These are recorded as *ceilings the shipped model does not claim*, not as
    results. Reporting a smoothed number as the level's accuracy would flatter the
    gate while describing a computation the online architecture never performs.
    Recording them pins the upper edge of the gate-taxonomy figure to a
    measurement instead of leaving it unlabeled.
    """
    smoothing_hits = viterbi_hits = positions = 0
    for word, phones in pairs:
        idx = [ALPHA_TO_IDX[c] for c in word if c in ALPHA_TO_IDX]
        if len(idx) != len(phones):
            continue

        smoothed = forward_backward(emission, transition, start, idx)
        smoothing_hits += sum(
            PHONES[int(np.argmax(row))] == phone
            for row, phone in zip(smoothed, phones, strict=True)
        )

        path = viterbi(emission, transition, start, idx)
        viterbi_hits += sum(
            PHONES[state] == phone for state, phone in zip(path, phones, strict=True)
        )
        positions += len(phones)

    del mapping  # baseline is computed by the caller; kept for signature clarity
    return {
        "smoothing": 100.0 * smoothing_hits / positions,
        "viterbi": 100.0 * viterbi_hits / positions,
        "positions": positions,
    }


def verify_decoder_equivalence(agent, emission, transition, start, pairs) -> tuple[int, list[bool]]:
    """Return oracle disagreements and the bounded pymdp correctness vector.

    MUST be zero. The full-set statistics are computed with the oracle for speed,
    which is only legitimate while the two decoders are provably identical.
    """
    # Both decoders must see the SAME words. decode_positions() resolves `limit`
    # through _iter_pairs, which takes an evenly spaced stride — so slicing the
    # oracle loop with pairs[:N] would compare a stride against a prefix and the
    # position counts would not even match. Resolve the subsample once, here.
    subset = _iter_pairs(pairs, EQUIVALENCE_WORDS)
    pymdp_correct = decode_positions(agent, subset, limit=None)
    oracle_correct: list[bool] = []
    for word, phones in subset:
        decoded = oracle_decode(emission, transition, start, word)
        if len(decoded) != len(phones):
            continue
        oracle_correct.extend(d == p for d, p in zip(decoded, phones, strict=True))

    if len(pymdp_correct) != len(oracle_correct):
        raise ValueError(
            f"Position counts differ: pymdp={len(pymdp_correct)} "
            f"oracle={len(oracle_correct)}. The comparison is not paired."
        )
    disagreements = sum(1 for a, b in zip(pymdp_correct, oracle_correct, strict=True) if a != b)
    return disagreements, pymdp_correct


def _paired_counts(model_hits: list[bool], baseline_hits: list[bool]) -> dict[str, int | float]:
    """Exact paired sufficient counts and p-value on one aligned population."""
    if len(model_hits) != len(baseline_hits) or not model_hits:
        raise ValueError(
            f"paired score vectors must be non-empty and equal length, got "
            f"{len(model_hits)} and {len(baseline_hits)}"
        )
    b = sum(
        model and not baseline for model, baseline in zip(model_hits, baseline_hits, strict=True)
    )
    c = sum(
        baseline and not model for model, baseline in zip(model_hits, baseline_hits, strict=True)
    )
    return {
        "b": b,
        "c": c,
        "p_value": exact_two_sided_binomial_p(b, c),
        "n_positions": len(model_hits),
    }


def _initial_letter_tvd(
    train: list[tuple[str, list[str]]], test: list[tuple[str, list[str]]]
) -> float:
    """Total variation between the two splits' initial-letter distributions.

    Near zero means the held-out set is not a different alphabetical neighborhood
    from the training set, which is the covariate shift a contiguous cut would
    produce and a stride does not.
    """
    from collections import Counter

    def distribution(pairs: list[tuple[str, list[str]]]) -> dict[str, float]:
        counts = Counter(word[0] for word, _phones in pairs if word)
        total = sum(counts.values())
        return {letter: n / total for letter, n in counts.items()}

    left, right = distribution(train), distribution(test)
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(k, 0.0) - right.get(k, 0.0)) for k in keys)


def _shared_prefix_pct(
    train: list[tuple[str, list[str]]], test: list[tuple[str, list[str]]], length: int
) -> float:
    """Share of held-out words whose first ``length`` characters occur in training.

    Reported rather than suppressed. On letter-to-phone the overlap is partly the
    regularity being modeled, so this is a bound on independence rather than a
    verdict of contamination, and the manuscript says so where it cites it.
    """
    prefixes = {word[:length] for word, _phones in train if len(word) >= length}
    scored = [word for word, _phones in test if len(word) >= length]
    if not scored:
        raise ValueError("no held-out word reaches the prefix length; the share is undefined")
    return 100.0 * sum(1 for word in scored if word[:length] in prefixes) / len(scored)


def _recomputed_primary_pronunciations() -> dict[str, list[str]]:
    """Recompute per-word primary pronunciations from the pinned CMUdict.

    The vendored artifact carries only aggregate sufficient statistics, so the
    held-out word split must re-derive the per-word sequences inside the
    analysis code. The derivation mirrors ``scripts/generate_phoneme_counts.py``
    exactly — sorted base words, first pronunciation, stress digits stripped —
    and ``main`` refuses to record unless the recomputed aggregates equal the
    vendored artifact's tables, so the split cannot drift from the artifact's
    population.
    """
    source = load_pinned_cmudict_source(_PROJECT_ROOT / "vendor" / "cmudict.dict", download=False)
    return {
        word: [_strip_stress(p) for p in pronunciations[0]]
        for word, pronunciations in sorted(source.dictionary.items())
        if pronunciations
    }


def _shuffle_word_phone_sequences(sequences: list[list[str]], seed: int) -> list[list[str]]:
    """Shuffle the phone order within each word, preserving every word's content.

    Within-word shuffling keeps the unigram content of every word exactly and
    destroys the adjacency structure — but NOT the within-word co-occurrence
    structure, which on English turns out to carry real predictive information.
    The result is therefore recorded as a measurement (how much of the bigram's
    held-out margin survives on bag-of-phones information alone), not used as
    the gate's negative control; see ``_redeal_word_phone_sequences``.
    """
    rng = np.random.default_rng(seed)
    return [[seq[i] for i in rng.permutation(len(seq))] for seq in sequences]


def _redeal_word_phone_sequences(sequences: list[list[str]], seed: int) -> list[list[str]]:
    """Re-deal the corpus phones into the original word lengths, globally shuffled.

    The Part A held-out negative control. Concatenating every training word's
    phones into one stream, shuffling it with a fixed-seed generator, and
    re-chunking at the original word boundaries preserves the corpus unigram
    and every word length while destroying BOTH the adjacency and the
    within-word co-occurrence structure. A model fit on the result reduces to
    the context-free bar up to estimation noise, so it must not beat the bar.
    """
    rng = np.random.default_rng(seed)
    stream = [phone for seq in sequences for phone in seq]
    rng.shuffle(stream)
    redealt: list[list[str]] = []
    position = 0
    for seq in sequences:
        redealt.append(stream[position : position + len(seq)])
        position += len(seq)
    return redealt


def main() -> int:
    counts = load_phoneme_counts(_PROJECT_ROOT / "vendor")
    bounds = phone_information_bounds(counts)

    # ── Part A: phoneme sequence perplexity ──────────────────────────────────
    # Recorded over the FULL dictionary (all 126,052 base words / ~800k positions)
    # via the closed-form sufficient statistics, not over the length-matched
    # g2p_pairs subset. That subset is biased toward short transparent-orthography
    # words and flatters the model by ~1.46 perplexity units (15.29 vs 16.74);
    # recording it would repeat the subsample-bias error this project already
    # documented once. The closed form is exact, verified against a
    # sequence-by-sequence walk to 1.05e-11 relative.
    ppl_model, n_positions = full_corpus_perplexity(counts)
    ppl_unigram = full_corpus_unigram_perplexity(counts)
    ppl_all_position_floor, floor_positions = full_corpus_order1_floor_perplexity(counts)
    if floor_positions != n_positions:
        print(
            "REFUSING TO RECORD: the full-corpus order-1 bound and model have "
            f"different denominators ({floor_positions} vs {n_positions}).",
            file=sys.stderr,
        )
        return 1
    subset_ppl, subset_unigram, _subset_positions = length_matched_phone_perplexities(counts)

    # The uniform comparator is analytic: with an identity emission and a uniform
    # transition every predictive probability is 1/39 regardless of observation.
    # Measured on a bounded pymdp sample purely to confirm the code path agrees
    # with the analytic value (float32 lands at 38.999998, not exactly 39).
    seqs = [s.split() for _, s in counts["g2p_pairs"][::PERPLEXITY_STRIDE]]
    ppl_uniform = compute_phoneme_perplexity(
        build_phoneme_agent(jnp.ones((1, N_PHONES, N_PHONES, 1)) / N_PHONES), seqs
    )
    if abs(ppl_uniform - N_PHONES) > 1e-3:
        print(
            f"REFUSING TO RECORD: the uniform-transition anchor measured "
            f"{ppl_uniform:.6f} but must equal {N_PHONES} analytically. The "
            f"inference loop is not doing what the perplexity metric assumes.",
            file=sys.stderr,
        )
        return 1

    # Cross-check the closed form against the pymdp path on the bounded sample.
    # They measure different position sets so they will not be equal, but the
    # pymdp sample must land in a plausible band around the closed-form value —
    # a wild divergence means the two disagree about the model, not the corpus.
    ppl_sample = compute_phoneme_perplexity(
        build_phoneme_agent(
            build_phone_transition(counts["phone_bigrams"]),
            build_phone_unigram(counts["phone_unigrams"]),
        ),
        seqs,
    )
    if not (0.5 * ppl_model < ppl_sample < 2.0 * ppl_model):
        print(
            f"REFUSING TO RECORD: closed-form full-corpus perplexity "
            f"{ppl_model:.4f} and the pymdp sample {ppl_sample:.4f} differ by more "
            f"than a factor of two. These should describe the same model.",
            file=sys.stderr,
        )
        return 1

    # ── Part A: held-out word split ──────────────────────────────────────────
    # The numbers above fit and score the same pronunciations (decision 6).
    # The held-out arm re-fits the identical model on a stride sample of the
    # sorted dictionary and scores the closed form on the disjoint remainder.
    # Split rule, prediction, falsifier, and both controls were pre-registered
    # before this measurement ran; see the held-out decision entry.
    prons = _recomputed_primary_pronunciations()
    recomputed = part_a_counts(prons.values())
    for table in ("phone_unigrams", "phone_bigrams", "phone_initials"):
        if recomputed[table] != counts[table]:
            print(
                f"REFUSING TO RECORD: the recomputed {table} disagree with the "
                "vendored artifact, so the held-out split would not describe "
                "the artifact's population.",
                file=sys.stderr,
            )
            return 1

    words = sorted(prons)
    train_words, test_words = phoneme_heldout_split(words, stride=PARTA_HELDOUT_STRIDE)
    train_counts = part_a_counts(prons[word] for word in train_words)
    test_counts = part_a_counts(prons[word] for word in test_words)
    held_model, held_bar, held_positions = part_a_split_perplexity(train_counts, test_counts)
    held_gap = math.log(held_bar) - math.log(held_model)
    held_penalty = held_model - ppl_model

    # POSITIVE CONTROL — scoring the in-sample protocol's own population
    # through the split-aware closed form must reproduce the in-sample closed
    # form and its bar, or the held-out number would not measure the same
    # metric the recorded in-sample number comes from.
    anchor_model, anchor_bar, anchor_positions = part_a_split_perplexity(counts, counts)
    anchor_rel_model = abs(anchor_model - ppl_model) / ppl_model
    anchor_rel_bar = abs(anchor_bar - ppl_unigram) / ppl_unigram
    if (
        anchor_positions != n_positions
        or anchor_rel_model > PARTA_ANCHOR_REL_TOL
        or anchor_rel_bar > PARTA_ANCHOR_REL_TOL
    ):
        print(
            f"REFUSING TO RECORD: the split-aware closed form does not "
            f"reproduce the in-sample closed form (model relative difference "
            f"{anchor_rel_model:.2e}, bar {anchor_rel_bar:.2e}, positions "
            f"{anchor_positions} vs {n_positions}).",
            file=sys.stderr,
        )
        return 1

    # CONTROL MEASUREMENT — a model fit on within-word-shuffled training words.
    # The first pre-registered control assumed this shuffle destroys all
    # transferable structure; measured, it does not (it preserves which phones
    # co-occur in a word, and that bag-of-phones information beats the
    # context-free bar on real English). Recorded as a finding, not gated.
    shuffled_counts = part_a_counts(
        _shuffle_word_phone_sequences([prons[word] for word in train_words], PARTA_SHUFFLE_SEED)
    )
    shuf_model, shuf_bar, shuf_positions = part_a_split_perplexity(shuffled_counts, test_counts)
    shuf_gap = math.log(shuf_bar) - math.log(shuf_model)

    # NEGATIVE CONTROL — a model fit on phones re-dealt across word boundaries
    # must not beat the held-out bar: the re-deal preserves the unigram and the
    # word lengths but no structure a bigram can transfer, so a positive gap
    # beyond the pre-registered bound means the control is broken.
    redealt_counts = part_a_counts(
        _redeal_word_phone_sequences([prons[word] for word in train_words], PARTA_SHUFFLE_SEED)
    )
    redeal_model, redeal_bar, redeal_positions = part_a_split_perplexity(
        redealt_counts, test_counts
    )
    redeal_gap = math.log(redeal_bar) - math.log(redeal_model)
    if shuf_positions != held_positions or redeal_positions != held_positions:
        print(
            f"REFUSING TO RECORD: a shuffled control scored "
            f"{shuf_positions}/{redeal_positions} positions but the held-out "
            f"model scored {held_positions}; the comparison is not on one "
            "denominator.",
            file=sys.stderr,
        )
        return 1
    if redeal_gap > PARTA_SHUFFLED_GAP_MAX_NATS:
        print(
            f"REFUSING TO RECORD: the re-dealt negative control beats the "
            f"held-out bar by {redeal_gap:.4f} nats (bound "
            f"{PARTA_SHUFFLED_GAP_MAX_NATS}). The re-deal did not destroy the "
            "corpus's structure.",
            file=sys.stderr,
        )
        return 1

    # ── Part B: G2P ──────────────────────────────────────────────────────────
    train, test = g2p_split(counts)
    emission, transition, start = _oracle_matrices(train)
    agent_b = build_g2p_agent(
        build_g2p_emission(train), build_g2p_transition(train), build_g2p_start(train)
    )

    disagreements, subsample_model_hits = verify_decoder_equivalence(
        agent_b, emission, transition, start, test
    )
    if disagreements != 0:
        print(
            f"REFUSING TO RECORD: pymdp and the oracle disagree on "
            f"{disagreements} positions. The full-set figures below would not "
            f"describe the pymdp agent.",
            file=sys.stderr,
        )
        return 1

    mapping = per_letter_argmax_baseline(train)
    subsample_baseline_hits = baseline_positions(mapping, test, limit=EQUIVALENCE_WORDS)
    subsample = _paired_counts(subsample_model_hits, subsample_baseline_hits)
    model_hits: list[bool] = []
    base_hits: list[bool] = []
    for word, phones in test:
        decoded = oracle_decode(emission, transition, start, word)
        if len(decoded) != len(phones):
            continue
        letters = [c for c in word if c in ALPHA_TO_IDX]
        model_hits.extend(d == p for d, p in zip(decoded, phones, strict=True))
        base_hits.extend(mapping.get(c) == p for c, p in zip(letters, phones, strict=True))

    acc_model = sum(model_hits) / len(model_hits)
    acc_base = sum(base_hits) / len(base_hits)
    b = sum(1 for m, s in zip(model_hits, base_hits, strict=True) if m and not s)
    c = sum(1 for m, s in zip(model_hits, base_hits, strict=True) if s and not m)
    p_value = exact_two_sided_binomial_p(b, c)

    ceilings = smoothing_and_viterbi_ceilings(emission, transition, start, test, mapping)
    if ceilings["positions"] != len(model_hits):
        print(
            f"REFUSING TO RECORD: ceiling measured on {ceilings['positions']} "
            f"positions but the model on {len(model_hits)}. The ceiling and the "
            f"model must be measured on the same position set.",
            file=sys.stderr,
        )
        return 1
    if ceilings["smoothing"] < acc_model * 100:
        print(
            f"REFUSING TO RECORD: smoothing ({ceilings['smoothing']:.3f}%) scored "
            f"below filtering ({100 * acc_model:.3f}%). Smoothing sees strictly more "
            f"evidence, so this indicates a bug in one of the two decoders.",
            file=sys.stderr,
        )
        return 1

    matched_pct = 100.0 * len(counts["g2p_pairs"]) / counts["source_base_words"]

    recorded = {
        # Part A — full dictionary, closed form
        "stage2_phoneme_perplexity": round(ppl_model, 4),
        "stage2_phoneme_unigram_baseline": round(ppl_unigram, 4),
        "stage2_phoneme_uniform_baseline": round(ppl_uniform, 4),
        "stage2_phoneme_all_position_floor_perplexity": round(ppl_all_position_floor, 4),
        "stage2_phoneme_all_position_max_gain_nats": round(
            math.log(N_PHONES) - math.log(ppl_all_position_floor), 4
        ),
        "stage2_phoneme_floor_perplexity": round(bounds["ppl_floor"], 4),
        "stage2_phoneme_max_gain_nats": round(bounds["max_gain_nats"], 4),
        "stage2_phoneme_gate_sesoi_nats": STAGE2_PHONEME_GATE_SESOI_NATS,
        "stage2_phoneme_eval_positions": n_positions,
        # Part A's distinct length-matched population. It has its own explicit
        # labels, but the recorder still owns it: ordinary tests never write the
        # shared ledger.
        "stage2_phoneme_subset_perplexity": round(subset_ppl, 4),
        "stage2_phoneme_subset_unigram_baseline": round(subset_unigram, 4),
        # Part A's held-out word split. Pre-registered arm: the same model and
        # bar re-fit on a stride sample of the dictionary and scored on the
        # disjoint remainder; see the held-out decision entry.
        "stage2_phoneme_heldout_split_stride": PARTA_HELDOUT_STRIDE,
        "stage2_phoneme_heldout_train_words": len(train_words),
        "stage2_phoneme_heldout_test_words": len(test_words),
        "stage2_phoneme_heldout_eval_positions": held_positions,
        "stage2_phoneme_heldout_perplexity": round(held_model, 4),
        "stage2_phoneme_heldout_unigram_baseline": round(held_bar, 4),
        "stage2_phoneme_heldout_gap_nats": round(held_gap, 4),
        "stage2_phoneme_heldout_penalty_ppl": round(held_penalty, 4),
        "stage2_phoneme_heldout_shuffled_gap_nats": round(shuf_gap, 4),
        "stage2_phoneme_heldout_redeal_gap_nats": round(redeal_gap, 4),
        # Part B
        "stage2_g2p_accuracy": round(100.0 * acc_model, 3),
        "stage2_g2p_baseline_accuracy": round(100.0 * acc_base, 3),
        "stage2_g2p_margin_pts": round(100.0 * (acc_model - acc_base), 3),
        "stage2_g2p_mcnemar_p": float(f"{p_value:.4g}"),
        "stage2_g2p_mcnemar_b": b,
        "stage2_g2p_gate_sesoi_pts": STAGE2_G2P_GATE_SESOI_PTS,
        "stage2_g2p_mcnemar_c": c,
        "stage2_g2p_eval_positions": len(model_hits),
        "stage2_g2p_train_pairs": len(train),
        "stage2_g2p_test_pairs": len(test),
        "stage2_g2p_subsample_positions": subsample["n_positions"],
        "stage2_g2p_subsample_mcnemar_p": float(f"{subsample['p_value']:.4g}"),
        "stage2_g2p_subsample_mcnemar_b": subsample["b"],
        "stage2_g2p_subsample_mcnemar_c": subsample["c"],
        # Two properties of the SPLIT rather than of the model, recorded because
        # an external review read the manuscript's earlier wording ("split by
        # sorted index") as a contiguous alphabetical cut and predicted severe
        # lexical-initial covariate shift. The stride is an interleave, so that
        # shift does not occur -- but the reviewer was right that a sorted source
        # invites the question, and the answer belongs in the ledger rather than
        # in a rebuttal.
        "stage2_g2p_split_stride": counts["test_every"],
        "stage2_g2p_split_initial_tvd": round(_initial_letter_tvd(train, test), 5),
        "stage2_g2p_split_shared_prefix_pct": round(
            _shared_prefix_pct(train, test, G2P_FAMILY_PREFIX_LENGTH), 1
        ),
        "stage2_g2p_length_matched_pct": round(matched_pct, 1),
        "stage2_decoder_disagreements": disagreements,
        # Ceilings the shipped model deliberately does not claim (see the function
        # docstring): filtering is what the streaming architecture performs.
        "stage2_g2p_smoothing_ceiling": round(ceilings["smoothing"], 3),
        "stage2_g2p_viterbi_ceiling": round(ceilings["viterbi"], 3),
    }

    ledger = load_ledger(_PROJECT_ROOT)
    ledger.record_many(recorded, run_id=RUN_ID, source=SOURCE)

    print(f"Recorded {len(recorded)} Stage 2 claims into data/claims.json:\n")
    width = max(len(k) for k in recorded)
    for label, value in recorded.items():
        print(f"  {label:<{width}} = {value}")

    print(
        f"\nPart A: model {ppl_model:.4f} ppl vs unigram {ppl_unigram:.4f} "
        f"(uniform {ppl_uniform:.4f}, comparable all-position order-1 floor "
        f"{ppl_all_position_floor:.4f}, transitions-only diagnostic "
        f"{bounds['ppl_floor']:.4f})."
    )
    print(
        f"Part A held out: model {held_model:.4f} ppl vs re-fit unigram bar "
        f"{held_bar:.4f} = {held_gap:+.4f} nats over {held_positions} positions "
        f"({len(test_words)} of {len(words)} words held out at stride "
        f"{PARTA_HELDOUT_STRIDE}; in-sample penalty {held_penalty:+.4f} ppl; "
        f"within-word-shuffle gap {shuf_gap:+.4f} nats, re-deal control gap "
        f"{redeal_gap:+.4f} nats)."
    )
    print(
        f"Part B: {100 * acc_model:.3f}% vs context-free {100 * acc_base:.3f}% "
        f"= {100 * (acc_model - acc_base):+.3f} pts over {len(model_hits)} positions; "
        f"McNemar b={b} c={c} p={p_value:.3e}."
    )
    print(
        f"Decoder equivalence: {disagreements} disagreements over "
        f"{EQUIVALENCE_WORDS} words — the oracle stands in for pymdp on the full set."
    )
    return 0


if __name__ == "__main__":
    # The ledger is single-writer: ``ClaimsLedger`` reads the whole file at
    # construction and writes the whole file at the end, so two overlapping
    # recorders lose one of the two results, with no error on either side.
    with single_writer_lock(ledger_lock_path(_PROJECT_ROOT)):
        raise SystemExit(main())
