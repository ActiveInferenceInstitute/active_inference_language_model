"""Stipulated aligned spelling-to-pronunciation artifact (protocol P2, design B).

This module implements the pair-structured stipulated artifact of
``docs/extension_protocols.md`` §3 (frozen copy:
``output/reports/omp-dispatch-20260908/evidence/extensions-E1R-frozen/``):
a seeded per-letter correspondence ``T`` from the 26 lowercase letters onto a
fixed subset of the canonical 39-phone ARPAbet inventory, spelling words cut
from independently seeded trajectories of the shipped stipulated letter chain,
and the train/test pair splits whose pronunciations are ``T`` applied per
position.  It is deliberately *not* an English letter-to-phone reading: the
table is stipulated, recorded, and digested, and "the inventories are met, the
stipulated correspondence is new" is the design-B framing.

Pretraining-exposure limitation (stated, never hidden): generator ownership
does not certify the absence of leakage or of unknown pretraining exposure —
the same standing limitation the whole stipulated family carries
(``ailm.levels.synthetic``'s docstring, and ERRATUM-S2).  The artifact is never
called contamination-free.

Smoothing and unseen policies (explicit, per the E1R residual):
- The native model is the shipped G2P engine refit on the stipulated train
  pairs: ``build_g2p_emission`` (add-k Laplace over the full 30×39 grid with
  ``resolved_emission_smoothing``, default k = 1), ``build_g2p_transition`` and
  ``build_g2p_start`` (add-k with ``smoothing``, default k = 1).  Unseen
  letter-phone pairs therefore carry exactly the pseudocount, never zero.
- The decoder is the shipped benchmark comparator path
  ``comparators._ailm_g2p_metrics`` → ``_ailm_g2p_decode``: position-wise
  filtering (posterior = emission row · prior, argmax, prior ← transition ·
  posterior), not Viterbi, prior reset per word by ``build_g2p_start``.
- The stipulated ``T`` lookup is a *separately named baseline*: the trivial
  closed-form map is exactly right on ``T``-generated data by construction, so
  the native arm can only ever tie it here; the comparison is reported, not
  turned into a beat-the-lookup gate.
- The A1 English-transfer control fits the same engine on the shipped CMUdict
  stride split; the engine's own skip of non-1:1 English pairs is that shipped
  policy applied to English data and is disclosed wherever the control is
  quoted.  This artifact's own pairs are 1:1 by construction and its validator
  refuses any misaligned pair before output instead of skipping it.
- The A3 floor is the analytic uniform-guess accuracy: 1/39 per position and
  (1/39)**L per word.

Dataflow separation (L1/L2): the probe prompt is the shipped
``prompts.build_prompt("g2p", word)`` — a pure function of the spelling and
the declared canonical-inventory phrase; no correspondence constant, no
English G2P hint, and no gold phone string can enter a prompt, and the probe
payloads built here never serialize scorer gold.

This module reads and computes.  It never writes, prints, parses argv, or
touches the network; ``scripts/generate_synthetic_aligned_g2p.py`` owns the
artifact and ``scripts/evaluate_synthetic_aligned_g2p.py`` owns the offline
evaluation.  No claim is recorded anywhere in this unit.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from ailm.benchmarks.comparators import _ailm_g2p_metrics
from ailm.benchmarks.prompts import build_prompt
from ailm.levels.phoneme import PHONES
from ailm.levels.synthetic import (
    SYNTHETIC_CHAIN_VERSION,
    SYNTHETIC_LETTERS,
    generate_sequence,
)
from ailm.resources import resolve_vendored_artifact

#: On-disk schema tag and artifact filename (protocol §3.2, proposed names adopted).
ALIGNED_G2P_SCHEMA = "ailm-synthetic-aligned-g2p-v1"
ALIGNED_G2P_ARTIFACT_NAME = "synthetic_aligned_g2p_counts.json"

#: Version of the stipulated correspondence table itself.
CORRESPONDENCE_VERSION = "aligned-g2p-correspondence-v1"

#: Fixed word length (protocol §3.1: proposed L = 6, adopted for this bounded
#: artifact).  Fixed-length construction bounds the collision space and keeps
#: prompts tiny.
ALIGNED_G2P_WORD_LENGTH = 6

#: Bounded deterministic resample budget per word — the
#: ``derangeable_draw_attempts`` precedent (64).  Exhaustion is a refusal, not
#: a relaxed rule.
ALIGNED_G2P_RESAMPLE_ATTEMPTS_CAP = 64

#: Seed domains.  Train trajectories draw consecutive integer seeds from one
#: base, evaluation trajectories from a second base placed exactly past the
#: train domain so the two domains cannot overlap for the declared population
#: sizes.  Both sit clear of the shipped letter/phone arms' held-out seed runs
#: (20260809 + ≤400 positions), so no trajectory stream is shared across arms.
ALIGNED_G2P_TRAIN_SEED_BASE = 20261300
ALIGNED_G2P_TRAIN_WORDS = 240
ALIGNED_G2P_HELD_OUT_SEED_BASE = ALIGNED_G2P_TRAIN_SEED_BASE + ALIGNED_G2P_TRAIN_WORDS
ALIGNED_G2P_TEST_WORDS = 60

#: Resample stride: attempt ``a ≥ 2`` of word ``i`` seeds at
#: ``base + i + a * ATTEMPT_STRIDE``, a lane far above both seed domains, so a
#: resample can neither re-enter nor cross a split's own seed domain.
ALIGNED_G2P_ATTEMPT_STRIDE = 10_000_000

#: Seed of the stipulated correspondence draw, and the size of its codomain —
#: a fixed subset of the canonical 39 phones, 2 letters per phone, so the map
#: is a complete balanced surjection over the letter inventory.
ALIGNED_G2P_CORRESPONDENCE_SEED = 20260912
ALIGNED_G2P_CODEDOMAIN_SIZE = 13

#: Seed selecting the broken-map permutation σ (a derangement of T's codomain).
ALIGNED_G2P_BROKEN_MAP_SEED = 20260913

#: Train-only demonstration plumbing: per-position derived seeds (the
#: ``bench_ctx_*`` draw pattern) and the A4 multiset-matched derangement cap.
ALIGNED_G2P_DEMONSTRATION_SEED_BASE = 20260914
ALIGNED_G2P_DEMONSTRATION_ATTEMPTS_CAP = 64

#: The evaluator's declared policies, quoted verbatim in its report.
SMOOTHING_POLICY = (
    "add-k Laplace: emission over the full 30x39 grid with "
    "resolved_emission_smoothing (k=1 default), transition and start with "
    "smoothing (k=1 default); unseen letter-phone pairs carry exactly the "
    "pseudocount, never zero"
)
UNSEEN_POLICY = (
    "no unseen-symbol case exists on the stipulated inventory: every letter is "
    "a row of the emission grid and every gold phone is a column, both "
    "pseudocount-backed; the uniform floor is the analytic 1/39 per position"
)
DECODER_POLICY = (
    "shipped benchmark comparator path comparators._ailm_g2p_metrics -> "
    "_ailm_g2p_decode: position-wise filtering (posterior = emission row * "
    "prior, argmax, prior <- transition @ posterior), prior reset per word by "
    "build_g2p_start; not Viterbi, no backtrace"
)


def _canonical_digest(value: Any) -> str:
    """sha256 over a canonical JSON serialization (sorted mapping keys)."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def stipulated_correspondence(
    seed: int = ALIGNED_G2P_CORRESPONDENCE_SEED,
    codomain_size: int = ALIGNED_G2P_CODEDOMAIN_SIZE,
) -> dict[str, str]:
    """Draw the stipulated per-letter correspondence ``T``.

    The codomain is a seeded sample of ``codomain_size`` distinct canonical
    phones (sorted after sampling for a deterministic index), and the letters
    are dealt onto it in a seeded shuffled order, two letters per phone — a
    complete, balanced surjection over the 26-letter inventory.  Deliberately
    not an English reading; the exact table is recorded in the artifact.
    """

    if not 1 <= codomain_size <= len(PHONES):
        raise ValueError(f"codomain_size must be within 1..{len(PHONES)}, got {codomain_size}")
    if codomain_size > len(SYNTHETIC_LETTERS):
        raise ValueError(
            "the codomain cannot exceed the letter inventory "
            f"({len(SYNTHETIC_LETTERS)}); a per-letter map is total over letters"
        )
    rng = np.random.default_rng(seed)
    chosen = sorted(str(phone) for phone in rng.choice(PHONES, size=codomain_size, replace=False))
    order = rng.permutation(len(SYNTHETIC_LETTERS))
    shuffled = [SYNTHETIC_LETTERS[int(index)] for index in order]
    per_phone = len(SYNTHETIC_LETTERS) // codomain_size
    if per_phone * codomain_size != len(SYNTHETIC_LETTERS):
        raise ValueError(
            "the balanced dealing needs the letter count divisible by the "
            f"codomain size ({len(SYNTHETIC_LETTERS)} letters, {codomain_size} phones)"
        )
    table: dict[str, str] = {}
    for index, letter in enumerate(shuffled):
        table[letter] = chosen[index // per_phone]
    if sorted(table) != sorted(SYNTHETIC_LETTERS):
        raise AssertionError("the correspondence must be complete over the letter inventory")
    return table


def broken_correspondence(
    correspondence: dict[str, str],
    seed: int = ALIGNED_G2P_BROKEN_MAP_SEED,
) -> dict[str, str]:
    """Derive the shape-valid broken map ``T' = sigma . T`` (control A2).

    ``sigma`` is a seeded rotation of ``T``'s phone codomain — a complete
    derangement of the codomain onto itself, so ``T'`` is a total
    letter→phone map with exactly ``T``'s codomain and no letter keeps its own
    phone.  This is well-defined for a rectangular letter→phone map where a
    "transpose" is not.
    """

    codomain = sorted(set(correspondence.values()))
    if len(codomain) < 2:
        raise ValueError("the broken map needs a codomain of at least two phones")
    offset = 1 + (seed % (len(codomain) - 1))
    rotated = {
        phone: codomain[(index + offset) % len(codomain)] for index, phone in enumerate(codomain)
    }
    broken = {letter: rotated[phone] for letter, phone in correspondence.items()}
    if any(broken[letter] == phone for letter, phone in correspondence.items()):
        raise AssertionError("sigma must derange the codomain: no letter keeps its own phone")
    return broken


def stipulated_word(seed: int, word_length: int = ALIGNED_G2P_WORD_LENGTH) -> str:
    """Cut one spelling word from an independently seeded chain trajectory.

    One word per trajectory under consecutive integer seeds (the shipped
    held-out pattern): the trajectory is ``word_length + 1`` symbols of the
    stipulated letter chain, and the word is its trailing ``word_length``
    window — the chain's fixed start symbol is dropped so no window is
    pinned to the start state.  Deterministic in ``(seed, word_length)``.
    """

    if word_length < 2:
        raise ValueError(f"word_length must be at least 2, got {word_length}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    trajectory = generate_sequence(word_length + 1, seed)
    return trajectory[1:]


def build_splits(
    correspondence: dict[str, str],
    *,
    n_train: int = ALIGNED_G2P_TRAIN_WORDS,
    n_test: int = ALIGNED_G2P_TEST_WORDS,
    word_length: int = ALIGNED_G2P_WORD_LENGTH,
    train_seed_base: int = ALIGNED_G2P_TRAIN_SEED_BASE,
    held_out_seed_base: int = ALIGNED_G2P_HELD_OUT_SEED_BASE,
    attempts_cap: int = ALIGNED_G2P_RESAMPLE_ATTEMPTS_CAP,
    attempt_stride: int = ALIGNED_G2P_ATTEMPT_STRIDE,
) -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]], dict[str, int]]:
    """Build the train and test ``(spelling, phones)`` splits.

    Disjointness is set-theoretic, never seed-domain inference: within-split
    duplicates and cross-split collisions are checked against the union of
    every word accepted so far, on every draw.  A draw that collides is
    resampled deterministically (attempt ``a`` shifts the word's seed by
    ``a * attempt_stride``); exhausting ``attempts_cap`` is a ``SystemExit``
    refusal — the generator then writes nothing at all.

    Returns:
        ``(train, test, stats)`` where the pairs are ``(spelling, phone list)``
        with ``T`` applied per position, and ``stats`` records the resample
        counts per split for the artifact's provenance.
    """

    if n_train <= 0 or n_test <= 0:
        raise ValueError(f"split populations must be positive, got {n_train}, {n_test}")
    if held_out_seed_base < train_seed_base + n_train:
        raise ValueError(
            "the held-out seed domain must start at or after the end of the "
            f"train domain ({train_seed_base} + {n_train}); got {held_out_seed_base}"
        )

    def draw(split: str, index: int, attempt: int) -> str:
        if attempt <= 1:
            seed = (train_seed_base if split == "train" else held_out_seed_base) + index
        else:
            seed = (
                (train_seed_base if split == "train" else held_out_seed_base)
                + index
                + attempt * attempt_stride
            )
        return stipulated_word(seed, word_length)

    def aligned(word: str) -> list[str]:
        return [correspondence[letter] for letter in word]

    splits: dict[str, list[tuple[str, list[str]]]] = {"train": [], "test": []}
    stats: dict[str, int] = {"train_resamples": 0, "test_resamples": 0}
    seen: set[str] = set()
    for split, count in (("train", n_train), ("test", n_test)):
        for index in range(count):
            for attempt in range(1, attempts_cap + 1):
                word = draw(split, index, attempt)
                if word not in seen:
                    break
            else:
                raise SystemExit(
                    f"aligned-g2p refusal: split {split!r} word index {index} could "
                    f"not draw a fresh word in {attempts_cap} attempts; the collision "
                    "space is exhausted, so no artifact is written and no claims are "
                    "made."
                )
            if attempt > 1:
                stats[f"{split}_resamples"] += 1
            seen.add(word)
            splits[split].append((word, aligned(word)))

    train_words = [word for word, _ in splits["train"]]
    test_words = [word for word, _ in splits["test"]]
    if len(set(train_words)) != len(train_words) or len(set(test_words)) != len(test_words):
        raise AssertionError("within-split duplicates survived the set-theoretic guard")
    if set(train_words) & set(test_words):
        raise AssertionError("cross-split collisions survived the set-theoretic guard")
    return splits["train"], splits["test"], stats


def build_artifact_payload(
    correspondence: dict[str, str],
    train: list[tuple[str, list[str]]],
    test: list[tuple[str, list[str]]],
    stats: dict[str, int],
    *,
    word_length: int | None = None,
    train_seed_base: int = ALIGNED_G2P_TRAIN_SEED_BASE,
    held_out_seed_base: int = ALIGNED_G2P_HELD_OUT_SEED_BASE,
    attempts_cap: int = ALIGNED_G2P_RESAMPLE_ATTEMPTS_CAP,
    attempt_stride: int = ALIGNED_G2P_ATTEMPT_STRIDE,
    generator_path: str = "scripts/generate_synthetic_aligned_g2p.py",
) -> dict[str, Any]:
    """Assemble the artifact payload (protocol §3.2 schema) with its digests.

    The split parameters are recorded as actually used, so a fixture-sized
    build validates against its own provenance and the canonical build
    validates against the protocol constants.
    """

    train_words = [word for word, _ in train]
    test_words = [word for word, _ in test]
    if word_length is None:
        if not train:
            raise ValueError("cannot derive word_length from an empty train split")
        word_length = len(train[0][0])
    return {
        "schema": ALIGNED_G2P_SCHEMA,
        "provenance": {
            "chain_version": SYNTHETIC_CHAIN_VERSION,
            "correspondence_version": CORRESPONDENCE_VERSION,
            "corpus_seed": train_seed_base,
            "held_out_seed": held_out_seed_base,
            "correspondence_seed": ALIGNED_G2P_CORRESPONDENCE_SEED,
            "broken_map_seed": ALIGNED_G2P_BROKEN_MAP_SEED,
            "word_length_L": word_length,
            "resample_attempts_cap": attempts_cap,
            "train_words": len(train),
            "test_words": len(test),
            "train_resamples": stats["train_resamples"],
            "test_resamples": stats["test_resamples"],
            "attempt_stride": attempt_stride,
            "chain_generation": "ailm.levels.synthetic.generate_sequence (one word per trajectory)",
            "generator_path": generator_path,
            "letter_inventory_source": "ailm.levels.synthetic.SYNTHETIC_LETTERS (26 lowercase letters)",
            "phone_inventory_source": "ailm.levels.phoneme.PHONES (claim:phoneme_count_39 canonical ARPAbet)",
            "train_words_digest": _canonical_digest(train_words),
            "test_words_digest": _canonical_digest(test_words),
            "correspondence_table_digest": _canonical_digest(correspondence),
            "smoothing_policy": SMOOTHING_POLICY,
            "unseen_policy": UNSEEN_POLICY,
            "decoder_policy": DECODER_POLICY,
            "exposure_limitation": (
                "generator ownership does not certify the absence of unknown "
                "pretraining exposure; the artifact is never called "
                "contamination-free (ERRATUM-S2)"
            ),
        },
        "correspondence": dict(correspondence),
        "splits": {
            "train": [[word, " ".join(phones)] for word, phones in train],
            "test": [[word, " ".join(phones)] for word, phones in test],
        },
    }


def validate_artifact_payload(payload: dict[str, Any]) -> None:
    """Refuse any malformed, colliding, or uncovered artifact before output.

    Every structural guarantee is re-checked from the bytes alone: schema,
    complete correspondence over the letter inventory with a codomain inside
    the canonical 39, fixed word length, 1:1 alignment with ``T`` applied per
    position, within-split duplicates, cross-split collisions, recorded
    population sizes, and all three digests.  A single bad pair refuses the
    whole artifact — the generator never silently skips unaligned pairs.
    """

    if payload.get("schema") != ALIGNED_G2P_SCHEMA:
        raise ValueError(f"schema must be {ALIGNED_G2P_SCHEMA!r}, got {payload.get('schema')!r}")
    provenance = payload.get("provenance")
    correspondence = payload.get("correspondence")
    splits = payload.get("splits")
    if (
        not isinstance(provenance, dict)
        or not isinstance(correspondence, dict)
        or not isinstance(splits, dict)
    ):
        raise ValueError("provenance, correspondence and splits must all be mappings")

    if sorted(correspondence) != sorted(SYNTHETIC_LETTERS):
        raise ValueError(
            "the correspondence must be complete over exactly the 26-letter "
            f"inventory; got {len(correspondence)} keys"
        )
    outside = sorted(set(correspondence.values()) - set(PHONES))
    if outside:
        raise ValueError(
            f"correspondence values outside the canonical 39-phone inventory: {outside}"
        )
    if _canonical_digest(correspondence) != provenance.get("correspondence_table_digest"):
        raise ValueError("correspondence_table_digest does not match the recorded table")

    word_length = provenance.get("word_length_L")
    if isinstance(word_length, bool) or not isinstance(word_length, int) or word_length < 2:
        raise ValueError(f"provenance.word_length_L must be an integer >= 2, got {word_length!r}")
    cap = provenance.get("resample_attempts_cap")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError(
            f"provenance.resample_attempts_cap must be a positive integer, got {cap!r}"
        )
    for field in ("train_words", "test_words"):
        size = provenance.get(field)
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(f"provenance.{field} must be a positive integer, got {size!r}")

    pair_lists: dict[str, list[tuple[str, list[str]]]] = {}
    for split in ("train", "test"):
        rows = splits.get(split)
        if not isinstance(rows, list) or len(rows) != provenance[f"{split}_words"]:
            raise ValueError(
                f"split {split!r} must hold exactly provenance.{split}_words "
                f"({provenance.get(f'{split}_words')!r}) pairs"
            )
        pairs: list[tuple[str, list[str]]] = []
        for row in rows:
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError(f"split {split!r} rows must be [spelling, phone_string] pairs")
            spelling, phone_string = row
            if not isinstance(spelling, str) or not isinstance(phone_string, str):
                raise ValueError(f"split {split!r} rows must carry string members")
            if len(spelling) != word_length:
                raise ValueError(
                    f"spelling {spelling!r} must be exactly word_length_L={word_length} letters"
                )
            outside_letters = sorted(set(spelling) - set(SYNTHETIC_LETTERS))
            if outside_letters:
                raise ValueError(
                    f"spelling {spelling!r} uses letters outside the stipulated "
                    f"inventory: {outside_letters}"
                )
            phones = phone_string.split()
            if len(phones) != len(spelling):
                raise ValueError(
                    f"pronunciation {phone_string!r} is not 1:1 with spelling "
                    f"{spelling!r} ({len(phones)} phones, {len(spelling)} letters)"
                )
            expected = [correspondence[letter] for letter in spelling]
            if phones != expected:
                raise ValueError(
                    f"gold {phone_string!r} for {spelling!r} is not the stipulated "
                    f"correspondence applied per position ({' '.join(expected)!r})"
                )
            pairs.append((spelling, phones))
        words = [word for word, _ in pairs]
        if len(set(words)) != len(words):
            raise ValueError(f"split {split!r} carries duplicate spellings")
        pair_lists[split] = pairs

    train_words = [word for word, _ in pair_lists["train"]]
    test_words = [word for word, _ in pair_lists["test"]]
    if set(train_words) & set(test_words):
        raise ValueError("cross-split collision: a spelling appears in both train and test")
    if _canonical_digest(train_words) != provenance.get("train_words_digest"):
        raise ValueError("train_words_digest does not match the recorded train spellings")
    if _canonical_digest(test_words) != provenance.get("test_words_digest"):
        raise ValueError("test_words_digest does not match the recorded test spellings")
    for field in ("corpus_seed", "held_out_seed", "attempt_stride", "resample_attempts_cap"):
        if field not in provenance:
            raise ValueError(
                f"provenance is missing the {field} field required to re-assert seed domains"
            )
    train_base = provenance["corpus_seed"]
    held_out_base = provenance["held_out_seed"]
    if (
        not isinstance(train_base, int)
        or not isinstance(held_out_base, int)
        or isinstance(train_base, bool)
        or isinstance(held_out_base, bool)
    ):
        raise ValueError("seed bases must be integers")
    if held_out_base < train_base + len(pair_lists["train"]):
        raise ValueError(
            "the held-out seed domain overlaps the train domain: held_out_seed "
            f"({held_out_base}) must start at or after corpus_seed + train_words "
            f"({train_base} + {len(pair_lists['train'])})"
        )


def load_aligned_g2p(vendor_dir: Path | None = None) -> dict[str, Any]:
    """Load and fully validate the vendored artifact.

    Mirrors ``ailm.levels.synthetic.load_synthetic_counts``: a missing artifact
    names its generator so the failure is actionable, and every structural
    guarantee is re-checked from the bytes before anything is returned.
    """

    artifact = resolve_vendored_artifact(ALIGNED_G2P_ARTIFACT_NAME, directory=vendor_dir)
    if not artifact.is_file():
        raise FileNotFoundError(
            f"Aligned G2P artifact not found at {artifact}. "
            "Run scripts/generate_synthetic_aligned_g2p.py to generate it."
        )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    validate_artifact_payload(payload)
    return payload


def artifact_pairs(
    payload: dict[str, Any],
) -> tuple[
    dict[str, str],
    list[tuple[str, list[str]]],
    list[tuple[str, list[str]]],
]:
    """Return ``(correspondence, train_pairs, test_pairs)`` from a validated payload."""

    correspondence = dict(payload["correspondence"])
    train = [(row[0], row[1].split()) for row in payload["splits"]["train"]]
    test = [(row[0], row[1].split()) for row in payload["splits"]["test"]]
    return correspondence, train, test


def t_lookup_baseline(
    correspondence: dict[str, str], words: list[tuple[str, list[str]]]
) -> tuple[float, float, list[list[str]]]:
    """The stipulated ``T`` applied per position — a separately named baseline.

    The trivial closed-form map is exactly right on ``T``-generated data by
    construction; it is never presented as the shipped G2P engine, and the
    native comparison against it is reported rather than gated.
    """

    predictions = [[correspondence[letter] for letter in word] for word, _ in words]
    total = sum(len(gold) for _, gold in words)
    position = (
        sum(
            predicted == gold_phone
            for prediction, (_, gold) in zip(predictions, words, strict=True)
            for predicted, gold_phone in zip(prediction, gold, strict=True)
        )
        / total
        if total
        else 0.0
    )
    exact = (
        sum(prediction == gold for prediction, (_, gold) in zip(predictions, words, strict=True))
        / len(words)
        if words
        else 0.0
    )
    return position, exact, predictions


def evaluate_aligned_g2p(
    correspondence: dict[str, str],
    train: list[tuple[str, list[str]]],
    test: list[tuple[str, list[str]]],
    *,
    english_train_pairs: list[tuple[str, list[str]]] | None = None,
) -> dict[str, Any]:
    """Run the offline arm set and report it without any gate.

    Arms: the native shipped engine refit on the stipulated train pairs; the
    separately named ``T``-lookup baseline; the A1 English-transfer control
    (same engine, CMUdict stride-split train side); the A2 broken-map control
    (engine refit on ``sigma . T`` pairs, scored against ``T`` gold); the A3
    analytic uniform floor.  Same-task comparisons may tie — the report says
    so rather than constructing a beat-the-lookup gate.  The numeric gate
    arithmetic a future recorder would refuse on is derived here and reported
    as ``pre_registration``: pre-registered values on the frozen artifact,
    computed before any probe call, never a recorded claim.
    """

    words = list(test)
    native_position, native_exact, native_predictions = _ailm_g2p_metrics(train, words)
    lookup_position, lookup_exact, lookup_predictions = t_lookup_baseline(correspondence, words)

    english: dict[str, Any] | None = None
    if english_train_pairs:
        english_position, english_exact, english_predictions = _ailm_g2p_metrics(
            english_train_pairs, words
        )
        english = {
            "position_accuracy": english_position,
            "exact_accuracy": english_exact,
            "predictions": english_predictions,
        }

    broken = broken_correspondence(correspondence)
    broken_train = [(word, [broken[letter] for letter in word]) for word, _ in train]
    broken_position, broken_exact, broken_predictions = _ailm_g2p_metrics(broken_train, words)

    position_floor = 1.0 / len(PHONES)
    exact_floor = position_floor ** len(words[0][0]) if words and words[0][0] else 0.0

    report: dict[str, Any] = {
        "arms": {
            "native_stipulated_engine": {
                "position_accuracy": native_position,
                "exact_accuracy": native_exact,
                "predictions": native_predictions,
            },
            "t_lookup_baseline": {
                "position_accuracy": lookup_position,
                "exact_accuracy": lookup_exact,
                "predictions": lookup_predictions,
            },
            "a1_english_transfer": english,
            "a2_broken_correspondence": {
                "position_accuracy": broken_position,
                "exact_accuracy": broken_exact,
                "predictions": broken_predictions,
            },
            "a3_uniform_floor": {
                "position_floor": position_floor,
                "exact_floor": exact_floor,
            },
        },
        "pre_registration": {
            "bar_candidate": (
                "the native stipulated engine's own held-out position accuracy — "
                "the E1R P1-6 ceiling pattern; a future recorder refuses A1/A2 if "
                "either control meets or clears it on the identical positions"
            ),
            "bar_value": native_position,
            "a1_clears_bar": english is not None
            and english["position_accuracy"] >= native_position,
            "a2_clears_bar": broken_position >= native_position,
            "native_ties_lookup": native_predictions == lookup_predictions,
            "note": (
                "ratios are reported, not thresholds: no phone-arm factor is "
                "carried over, and a tie between same-task mechanisms is a "
                "reported outcome, not a failure"
            ),
        },
        "policies": {
            "smoothing": SMOOTHING_POLICY,
            "unseen": UNSEEN_POLICY,
            "decoder": DECODER_POLICY,
            "a1_english_skip_disclosure": (
                "the A1 control fits the shipped engine on the CMUdict stride "
                "split, whose non-1:1 pairs are skipped by that engine's own "
                "logged policy; this artifact's pairs are 1:1 and its validator "
                "refuses misalignment instead of skipping"
            ),
        },
    }
    return report


def probe_prompt(word: str) -> str:
    """The arm's prompt: the shipped builder, a pure function of the spelling.

    L1 (prompt purity): the prompt carries the spelling and the declared
    canonical-inventory phrase only — no correspondence constant, no English
    G2P hint, no gold phone string.  The structural tests pin this.
    """

    return build_prompt("g2p", word)


def probe_request_payloads(words: list[str], *, response_tokens: int = 32) -> list[dict[str, Any]]:
    """Build the probe request payloads for evaluation words — gold never enters.

    L2 (gold isolation): evaluation gold strings live only in the scorer; the
    payloads below are exactly what a probe call would serialize, and the
    tests replay them against the scorer's gold to assert nothing leaked.
    """

    return [
        {
            "task": "g2p",
            "prompt": probe_prompt(word),
            "response_tokens": response_tokens,
        }
        for word in words
    ]


def demonstration_pairs(
    train_pool: list[tuple[str, list[str]]],
    k: int,
    position_index: int,
    *,
    seed_base: int = ALIGNED_G2P_DEMONSTRATION_SEED_BASE,
    attempts_cap: int = ALIGNED_G2P_DEMONSTRATION_ATTEMPTS_CAP,
    arm: str = "aligned-g2p-demo",
) -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]]]:
    """Draw a train-only demonstration set and its A4 deranged control.

    Demonstrations come exclusively from train pairs, keyed by a per-position
    derived seed (the ``bench_ctx_*`` pattern), never test gold.  The A4
    control permutes the gold pronunciations among the shown spellings —
    the same character-level phone multiset, a broken assignment — so a
    probe-level non-separation cannot be misread as a vacuous stimulus.  A
    pool that admits no derangement within the attempt cap is refused.
    """

    if k < 0:
        raise ValueError("k must be non-negative")
    if attempts_cap <= 0:
        raise ValueError("attempts_cap must be positive")
    if k > len(train_pool):
        raise ValueError(f"cannot draw {k} demonstrations from a pool of {len(train_pool)}")
    if k == 0:
        return [], []
    for _attempt in range(1, attempts_cap + 1):
        rng = random.Random(f"{seed_base}:{k}:{position_index}:{_attempt}")
        examples = rng.sample(train_pool, k)
        deranged = _derange_pair_gold(examples)
        if deranged is not None:
            return examples, deranged
    raise ValueError(
        f"aligned-g2p demonstration refusal: no derangeable draw of {k} pairs "
        f"appeared in {attempts_cap} attempts at position {position_index} in arm "
        f"{arm!r}. No claims written."
    )


def _derange_pair_gold(
    examples: list[tuple[str, list[str]]],
) -> list[tuple[str, list[str]]] | None:
    """Rotate the gold pronunciations among the shown spellings.

    The phone-list multiset is preserved exactly (the A4 property) and no
    spelling keeps its own gold, whenever at least two distinct golds exist.
    """

    golds = [tuple(phones) for _, phones in examples]
    distinct = sorted(set(golds))
    if len(distinct) < 2:
        return None
    successor = {gold: distinct[(index + 1) % len(distinct)] for index, gold in enumerate(distinct)}
    deranged = [(word, list(successor[tuple(phones)])) for word, phones in examples]
    if all(new == old for (_, new), (_, old) in zip(deranged, examples, strict=True)):
        return None
    return deranged
