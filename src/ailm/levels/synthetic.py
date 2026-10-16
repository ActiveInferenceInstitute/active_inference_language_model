"""Stipulated synthetic letter language for an exact-artifact-overlap control.

Why this module exists
----------------------
The external Ollama benchmark scores AILM against registered reference probes on
Brown-corpus letter statistics and CMUdict pronunciations. Both corpora are
public and longstanding, while the probes' training corpora are not documented
at the granularity needed to assess overlap. A win or loss on those arms cannot
separate inference from possible corpus recall.

This module narrows that confound. It defines a letter language whose bigram
transition is **stipulated in code** and sampled directly rather than estimated
from natural text. Local generation reduces exact-artifact overlap as an explanation;
it cannot prove that an undisclosed pretraining corpus lacks the same sequence,
a similar alphabet, or a related transition rule. AILM trains offline on a draw
from the chain; no registered external probe is fitted offline. In the
no-demonstration arm the probes receive no separately labeled transitions, while
the companion in-context protocol supplies exact one-transition labeled windows
and their deranged controls. Even the no-demonstration query is not
observation-free: its bounded sequence prefix exposes transitions from the same
chain. That protocol records the window and measures a one-character
sensitivity arm, so the opportunity asymmetry is measured rather than erased in
prose.

The designed chain
------------------
Alphabet: the 26 lowercase letters. The 30-symbol :mod:`ailm.levels.letter`
ALPHABET stays the indexing basis for every artifact and matrix built from this
language, so the existing letter machinery (``build_letter_transition``, the benchmark's
comparator path) consumes it unchanged; the prompt, response schema, and parser
instead expose this exact 26-symbol subset. The four non-letter symbols simply
never occur, which is a property the tests pin rather than assume.

Each column of the transition is an explicit three-part mixture:

    P(next = j | current = i)
        = W_DOMINANT * [j == successor(i)]
        + W_OPPOSITE * (1 / |opposite_class(i)|  if class(j) != class(i) else 0)
        + W_UNIFORM  * (1 / 26)

with weights summing to one. Three structures are layered deliberately:

1. **Vowel/consonant alternation** (``W_OPPOSITE``) — a coarse regularity a
   language model might guess at from English priors.
2. **A designed dominant successor** (``W_DOMINANT``) — a fixed map specified
   independently of English corpus counts. This is the part the AILM comparator
   estimates from the generated corpus; no claim is made that similar local
   structure is absent from external-model training.
3. **A uniform floor** (``W_UNIFORM``) — keeps every bigram reachable, so the
   sampled corpus has support everywhere and the task is not degenerate.

Because ``successor(i)`` always lies in the opposite class, it carries strictly
more mass than any other symbol, so the order-1 argmax over a well-estimated
column is exactly ``successor(i)``. That gives the probe a known ceiling: a
perfect order-1 learner scores the marginal probability of the dominant
successor, a little under one half, and the uniform baseline is 1/26.

Orientation
-----------
:func:`designed_transition` returns ``T[next, current]`` — **column-stochastic**,
matching the convention of :func:`ailm.levels.letter.build_letter_transition`. The
transpose is also column-stochastic here only by accident of shape, never by
content, and ``tests/test_synthetic_benchmark.py`` carries a negative control
against the transposed wiring for the same reason ``test_oracle.py`` does.

Determinism
-----------
Sampling draws its uniforms in one bulk call and walks precomputed cumulative
columns, so the stream is a pure function of ``(seed, n_chars)`` and does not
depend on NumPy's sampling-algorithm internals the way ``Generator.choice``
would. Two runs of ``scripts/generate_synthetic_counts.py`` therefore produce
byte-identical artifacts, which is the only thing that makes the vendored
sha256 worth recording.

This module reads and computes. It never writes, prints, parses argv, or touches
the network; ``scripts/generate_synthetic_counts.py`` owns the artifact and
``scripts/record_synthetic_benchmark_claims.py`` owns the measurement.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ailm.levels._stochastic import validate_column_stochastic
from ailm.levels.letter import ALPHA_LIST, ALPHA_TO_IDX, build_letter_transition
from ailm.levels.phoneme import PHONE_TO_IDX, PHONES, build_phone_transition
from ailm.resources import resolve_vendored_artifact

# ── The designed language ────────────────────────────────────────────────────

#: The 26 lowercase letters the synthetic language emits. A strict subset of the
#: letter level's 30-symbol ALPHABET: space, period, apostrophe and hyphen never
#: occur, so a prompt built from this stream contains letters only.
SYNTHETIC_LETTERS: tuple[str, ...] = tuple("abcdefghijklmnopqrstuvwxyz")

#: Vowels and consonants partition the alphabet; the chain alternates between them.
VOWELS: tuple[str, ...] = tuple("aeiou")
CONSONANTS: tuple[str, ...] = tuple(c for c in SYNTHETIC_LETTERS if c not in VOWELS)

#: Mixture weights of the designed transition. They must sum to 1.0.
W_DOMINANT: float = 0.45
W_OPPOSITE: float = 0.35
W_UNIFORM: float = 0.20

#: Uniform-guess accuracy on this language — the floor any comparator must clear.
UNIFORM_BASELINE: float = 1.0 / len(SYNTHETIC_LETTERS)

#: Version of the stipulated chain and corpus-generation contract. Measurement
#: recorders carry their own protocol versions: a zero-shot response contract
#: and an in-context sweep are not one protocol merely because they share this
#: language.
SYNTHETIC_CHAIN_VERSION = "synthetic-chain-v1"

#: Training-corpus generation parameters. Changing either changes the vendored
#: artifact's sha256 and requires updating docs/provenance/sources.md.
DEFAULT_CORPUS_SEED: int = 20260808
DEFAULT_CORPUS_CHARS: int = 200_000

#: Held-out independent-position seed: same chain, distinct seed domain.
DEFAULT_HELD_OUT_SEED: int = 20260809


#: The stipulated phone language. The design mirrors the letter chain above
#: with the same contract shape; its predictions and refutation conditions
#: are pre-registered in
#: ``scratchpad/fleet_2026-09-02/ailm-synth-phoneme/decision.md``.

#: The 39 canonical stress-free ARPAbet phones the synthetic phone language
#: emits — exactly the phoneme level's inventory, so the phoneme task's
#: prompt, response schema, parser and comparator consume this stream
#: unchanged.
SYNTHETIC_PHONES: tuple[str, ...] = PHONES

#: The ARPAbet vowel class: the monophthongs, diphthongs and r-colored vowel
#: of the canonical inventory, stress digits already stripped. The consonant
#: class is the complement. The partition is stipulated with the chain, not
#: learned and not derived from any corpus, and the chain alternates between
#: the classes.
_ARPABET_VOWEL_SET: frozenset[str] = frozenset(
    {
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "EH",
        "ER",
        "EY",
        "IH",
        "IY",
        "OW",
        "OY",
        "UH",
        "UW",
    }
)
PHONE_VOWELS: tuple[str, ...] = tuple(p for p in SYNTHETIC_PHONES if p in _ARPABET_VOWEL_SET)
PHONE_CONSONANTS: tuple[str, ...] = tuple(
    p for p in SYNTHETIC_PHONES if p not in _ARPABET_VOWEL_SET
)

#: Mixture weights of the designed phone transition, identical to the letter
#: arm's. They must sum to 1.0.
W_DOMINANT_PHONE: float = 0.45
W_OPPOSITE_PHONE: float = 0.35
W_UNIFORM_PHONE: float = 0.20

#: Uniform-guess accuracy on this language — the floor any comparator must
#: clear.
PHONE_UNIFORM_BASELINE: float = 1.0 / len(SYNTHETIC_PHONES)

#: Version of the stipulated phone chain and corpus-generation contract. The
#: measurement recorder carries its own protocol version, as the letter arm
#: does.
SYNTHETIC_PHONE_CHAIN_VERSION = "synthetic-phone-chain-v1"

#: Training-corpus generation parameters, mirroring the letter arm's. Changing
#: either changes the vendored artifact's sha256 and requires updating
#: docs/provenance/sources.md.
DEFAULT_PHONE_CORPUS_SYMBOLS: int = 200_000
DEFAULT_PHONE_CORPUS_SEED: int = 20260808

#: Held-out independent-position seed: same chain, distinct seed domain — the
#: letter arm's held-out seed base.
DEFAULT_PHONE_HELD_OUT_SEED: int = 20260809


def dominant_successor(letter: str) -> str:
    """Return the designed high-probability successor of ``letter``.

    The map is stipulated, not learned and not English: a vowel's successor is a
    consonant chosen by a fixed stride through :data:`CONSONANTS`, and a
    consonant's successor is a vowel chosen by a fixed stride through
    :data:`VOWELS`. Both strides are coprime with their cycle length, so the five
    vowels take five distinct consonant successors rather than collapsing onto a
    few.

    Args:
        letter: One symbol from :data:`SYNTHETIC_LETTERS`.

    Returns:
        The successor symbol, always of the opposite class.

    Raises:
        ValueError: If ``letter`` is outside the synthetic alphabet.
    """
    if letter in VOWELS:
        return CONSONANTS[(4 * VOWELS.index(letter) + 2) % len(CONSONANTS)]
    if letter in CONSONANTS:
        return VOWELS[(2 * CONSONANTS.index(letter) + 1) % len(VOWELS)]
    raise ValueError(f"{letter!r} is not one of the {len(SYNTHETIC_LETTERS)} synthetic letters")


def designed_transition() -> np.ndarray:
    """Build the stipulated column-stochastic transition of the synthetic language.

    Returns:
        ``T`` of shape (26, 26) with ``T[j, i] = P(next = j | current = i)``.
        Every column sums to 1 and every entry is strictly positive.
    """
    size = len(SYNTHETIC_LETTERS)
    transition = np.zeros((size, size), dtype=np.float64)
    for i, current in enumerate(SYNTHETIC_LETTERS):
        opposite = CONSONANTS if current in VOWELS else VOWELS
        column = np.full(size, W_UNIFORM / size, dtype=np.float64)
        for symbol in opposite:
            column[SYNTHETIC_LETTERS.index(symbol)] += W_OPPOSITE / len(opposite)
        column[SYNTHETIC_LETTERS.index(dominant_successor(current))] += W_DOMINANT
        transition[:, i] = column
    return transition


def generate_sequence(n_chars: int, seed: int) -> str:
    """Sample ``n_chars`` symbols from the designed chain.

    Deterministic in ``(seed, n_chars)``: one bulk uniform draw is walked against
    precomputed cumulative columns, so the stream does not depend on NumPy's
    ``Generator.choice`` internals. The chain starts at the first symbol of
    :data:`SYNTHETIC_LETTERS` and mixes within a few steps.

    Args:
        n_chars: Length of the returned string. Must be at least 2 so the stream
            contains at least one bigram.
        seed: Non-negative seed for ``numpy.random.default_rng``.

    Returns:
        A string of ``n_chars`` symbols drawn from :data:`SYNTHETIC_LETTERS`.

    Raises:
        ValueError: If ``n_chars < 2`` or ``seed`` is negative.
    """
    if n_chars < 2:
        raise ValueError(f"n_chars must be at least 2 to contain a bigram, got {n_chars}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    cumulative = np.cumsum(designed_transition(), axis=0)
    draws = np.random.default_rng(seed).random(n_chars)
    state = 0
    symbols = [SYNTHETIC_LETTERS[state]]
    for draw in draws[1:]:
        state = int(np.searchsorted(cumulative[:, state], draw, side="right"))
        state = min(state, len(SYNTHETIC_LETTERS) - 1)
        symbols.append(SYNTHETIC_LETTERS[state])
    return "".join(symbols)


def count_bigrams(text: str) -> dict[str, int]:
    """Count adjacent symbol pairs as the flat ``{"<prev>,<curr>": count}`` mapping.

    The key format and ALPHABET-order sort match the ``counts`` mapping inside
    ``vendor/brown_counts.json``, so
    :func:`ailm.levels.letter.build_letter_transition` consumes either mapping with no
    numerical adapter.  The synthetic artifact itself remains a flat mapping;
    Brown's external-source artifact has a provenance envelope.

    Args:
        text: A string over :data:`SYNTHETIC_LETTERS`.

    Returns:
        Bigram counts keyed ``"prev,curr"``, ordered by the letter level's
        ALPHABET index of the pair.

    Raises:
        ValueError: If ``text`` contains a symbol outside the synthetic alphabet.
    """
    outside = sorted(set(text) - set(SYNTHETIC_LETTERS))
    if outside:
        raise ValueError(f"text contains symbols outside the synthetic alphabet: {outside}")
    counts: dict[str, int] = {}
    for index in range(len(text) - 1):
        key = f"{text[index]},{text[index + 1]}"
        counts[key] = counts.get(key, 0) + 1
    return {
        key: counts[key]
        for key in sorted(counts, key=lambda k: (ALPHA_TO_IDX[k[0]], ALPHA_TO_IDX[k[2]]))
    }


def build_synthetic_counts(
    n_chars: int = DEFAULT_CORPUS_CHARS, seed: int = DEFAULT_CORPUS_SEED
) -> dict[str, int]:
    """Sample a training corpus and return its bigram counts.

    This is the whole payload of ``vendor/synthetic_counts.json``.

    Args:
        n_chars: Training-corpus length in symbols.
        seed: Seed for the training stream.

    Returns:
        The flat bigram-count mapping described in :func:`count_bigrams`.
    """
    return count_bigrams(generate_sequence(n_chars, seed))


def independent_held_out_positions(
    positions: int,
    *,
    window: int,
    seed: int = DEFAULT_HELD_OUT_SEED,
) -> list[tuple[str, str]]:
    """Generate one independently seeded chain trajectory per scored position.

    A stride through one held-out stream leaves adjacent scored blocks dependent
    when the stride is close to the prefix width.  Exact paired tests and Wilson
    intervals require independent sampling units, so the v2 synthetic protocols
    use one ``window + 1`` trajectory per item.  Consecutive integer seeds name
    the frozen trajectories; no result-dependent resampling occurs.

    Args:
        positions: Number of independent scored items. Must be positive.
        window: Prefix symbols exposed to the model. Must be positive.
        seed: Seed of the first trajectory. Each later item adds its index.

    Returns:
        ``(prefix, next_symbol)`` pairs in seed order.
    """

    if positions <= 0:
        raise ValueError(f"positions must be positive, got {positions}")
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    pairs: list[tuple[str, str]] = []
    for index in range(positions):
        sequence = generate_sequence(window + 1, seed + index)
        pairs.append((sequence[:-1], sequence[-1]))
    return pairs


def independent_target_start_tvd(window: int) -> float:
    """TVD between the scored target marginal and the chain's stationary law."""

    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    transition = designed_transition()
    size = len(SYNTHETIC_LETTERS)
    system = np.vstack((transition - np.eye(size), np.ones((1, size))))
    rhs = np.concatenate((np.zeros(size), np.ones(1)))
    stationary, *_ = np.linalg.lstsq(system, rhs, rcond=None)
    stationary = np.maximum(stationary, 0.0)
    stationary /= stationary.sum()
    start = np.zeros(size, dtype=np.float64)
    start[0] = 1.0
    target = np.linalg.matrix_power(transition, window) @ start
    return float(0.5 * np.abs(target - stationary).sum())


def load_synthetic_counts(vendor_dir: Path | None = None) -> dict[str, int]:
    """Load the vendored synthetic bigram counts.

    Mirrors :func:`ailm.levels.letter.load_brown_counts`, including naming the
    generator in the failure message so a missing artifact is actionable.

    Args:
        vendor_dir: Path to ``vendor/``. ``None`` reads the embedded wheel
            resource when installed and the canonical top-level copy in a
            source checkout.

    Returns:
        The flat ``"prev,curr" -> count`` mapping.

    Raises:
        FileNotFoundError: If the artifact has not been generated.
    """
    counts_file = resolve_vendored_artifact("synthetic_counts.json", directory=vendor_dir)
    if not counts_file.is_file():
        raise FileNotFoundError(
            f"Synthetic counts not found at {counts_file}. "
            "Run scripts/generate_synthetic_counts.py to generate it."
        )
    return json.loads(counts_file.read_text(encoding="utf-8"))


# ── The AILM comparator on this language ─────────────────────────────────────


def synthetic_letter_predictions(
    counts: dict[str, int], positions: list[tuple[str, str]]
) -> list[str]:
    """Order-1 argmax prediction for each position, trained on ``counts``.

    Deliberately identical in behavior to the external benchmark's private
    letter comparator — the smoothed bigram matrix, argmax over the column of the
    last observed symbol, and the mean column when a prefix is empty. The
    equivalence is asserted in ``tests/test_synthetic_benchmark.py`` rather than
    assumed, so the two cannot drift apart silently.

    Args:
        counts: Bigram counts in the vendored flat format.
        positions: ``(prefix, gold)`` pairs, as produced by
            :func:`ailm.benchmarks.letter_positions`.

    Returns:
        One predicted symbol per position, in order.
    """
    matrix = np.asarray(build_letter_transition(counts))[0]
    return [
        ALPHA_LIST[int(np.argmax(matrix[:, ALPHA_TO_IDX[prefix[-1]]]))]
        if prefix
        else ALPHA_LIST[int(np.argmax(matrix.mean(axis=1)))]
        for prefix, _ in positions
    ]


def correctness_flags(predictions: list[str], positions: list[tuple[str, str]]) -> list[bool]:
    """Per-position correctness, the paired-test input.

    Args:
        predictions: One prediction per position, same order.
        positions: ``(prefix, gold)`` pairs.

    Returns:
        ``True`` where the prediction equals the gold symbol.

    Raises:
        ValueError: If the two sequences have different lengths — a length
            mismatch would silently truncate the pairing under ``zip``.
    """
    if len(predictions) != len(positions):
        raise ValueError(
            f"{len(predictions)} predictions for {len(positions)} positions; "
            "a paired comparison cannot be formed from unequal sequences"
        )
    return [
        prediction == gold for prediction, (_, gold) in zip(predictions, positions, strict=True)
    ]


# ── The stipulated phone language ────────────────────────────────────────────
#
# A second stipulated language beside the letter chain above, sharing its
# contract shape: a transition written down in code, a deterministic sampler,
# a vendored count artifact with a named generator, an independently seeded
# evaluation population, and an order-1 comparator whose decoding rule is the
# one the external benchmark's own phoneme comparator uses. The design,
# predictions, and refutation conditions are pre-registered in
# ``scratchpad/fleet_2026-09-02/ailm-synth-phoneme/decision.md``.


def phone_dominant_successor(phone: str) -> str:
    """Return the designed high-probability successor of ``phone``.

    The map is stipulated, not learned and not English phonotactics: a vowel's
    successor is a consonant chosen by a fixed stride through
    :data:`PHONE_CONSONANTS`, and a consonant's successor is a vowel chosen by
    a fixed stride through :data:`PHONE_VOWELS`. The strides are coprime with
    their cycle lengths, so the vowels take distinct consonant successors
    rather than collapsing onto a few. Both strides and the offsets mirror the
    letter arm's ``dominant_successor`` construction.

    Args:
        phone: One symbol from :data:`SYNTHETIC_PHONES`.

    Returns:
        The successor phone, always of the opposite class.

    Raises:
        ValueError: If ``phone`` is outside the canonical ARPAbet inventory.
    """
    if phone in PHONE_VOWELS:
        return PHONE_CONSONANTS[(5 * PHONE_VOWELS.index(phone) + 2) % len(PHONE_CONSONANTS)]
    if phone in PHONE_CONSONANTS:
        return PHONE_VOWELS[(2 * PHONE_CONSONANTS.index(phone) + 1) % len(PHONE_VOWELS)]
    raise ValueError(f"{phone!r} is not one of the {len(SYNTHETIC_PHONES)} synthetic phones")


def phone_designed_transition() -> np.ndarray:
    """Build the stipulated column-stochastic transition of the phone language.

    Returns:
        ``T`` of shape ``(39, 39)`` with ``T[j, i] = P(next = j | current = i)``
        over :data:`SYNTHETIC_PHONES` order. Every column sums to one and every
        entry is strictly positive.
    """
    size = len(SYNTHETIC_PHONES)
    transition = np.zeros((size, size), dtype=np.float64)
    for i, current in enumerate(SYNTHETIC_PHONES):
        opposite = PHONE_CONSONANTS if current in PHONE_VOWELS else PHONE_VOWELS
        column = np.full(size, W_UNIFORM_PHONE / size, dtype=np.float64)
        for symbol in opposite:
            column[SYNTHETIC_PHONES.index(symbol)] += W_OPPOSITE_PHONE / len(opposite)
        column[SYNTHETIC_PHONES.index(phone_dominant_successor(current))] += W_DOMINANT_PHONE
        transition[:, i] = column
    return validate_column_stochastic(transition, "synthetic phone transition")


def generate_phone_sequence(n_symbols: int, seed: int) -> tuple[str, ...]:
    """Sample ``n_symbols`` phones from the designed phone chain.

    Deterministic in ``(seed, n_symbols)`` by the same construction as
    :func:`generate_sequence`: one bulk uniform draw walked against
    precomputed cumulative columns, so the stream does not depend on NumPy's
    ``Generator.choice`` internals. The chain starts at the first phone of
    :data:`SYNTHETIC_PHONES` and mixes within a few steps.

    Args:
        n_symbols: Length of the returned sequence. Must be at least 2 so the
            stream contains at least one bigram.
        seed: Non-negative seed for ``numpy.random.default_rng``.

    Returns:
        A tuple of ``n_symbols`` phones drawn from :data:`SYNTHETIC_PHONES`.

    Raises:
        ValueError: If ``n_symbols < 2`` or ``seed`` is negative.
    """
    if n_symbols < 2:
        raise ValueError(f"n_symbols must be at least 2 to contain a bigram, got {n_symbols}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    cumulative = np.cumsum(phone_designed_transition(), axis=0)
    draws = np.random.default_rng(seed).random(n_symbols)
    state = 0
    symbols = [SYNTHETIC_PHONES[state]]
    for draw in draws[1:]:
        state = int(np.searchsorted(cumulative[:, state], draw, side="right"))
        state = min(state, len(SYNTHETIC_PHONES) - 1)
        symbols.append(SYNTHETIC_PHONES[state])
    return tuple(symbols)


def count_phone_bigrams(sequence: Sequence[str]) -> dict[str, int]:
    """Count adjacent phone pairs as the flat ``{"<prev>,<next>": count}`` mapping.

    The key format matches the ``phone_bigrams`` mapping inside
    ``vendor/cmudict_counts.json``, so
    :func:`ailm.levels.phoneme.build_phone_transition` consumes either mapping
    with no numerical adapter: ARPAbet symbols never contain a comma, so the
    single split is unambiguous.

    Args:
        sequence: An ordered sequence of phones over :data:`SYNTHETIC_PHONES`.

    Returns:
        Bigram counts keyed ``"prev,next"``, ordered by the inventory index of
        the pair.

    Raises:
        ValueError: If ``sequence`` contains a symbol outside the canonical
            ARPAbet inventory.
    """
    outside = sorted(set(sequence) - set(SYNTHETIC_PHONES))
    if outside:
        raise ValueError(f"sequence contains symbols outside the ARPAbet inventory: {outside}")
    counts: dict[str, int] = {}
    for index in range(len(sequence) - 1):
        key = f"{sequence[index]},{sequence[index + 1]}"
        counts[key] = counts.get(key, 0) + 1

    def key_index(key: str) -> tuple[int, int]:
        previous, current = key.split(",", 1)
        return PHONE_TO_IDX[previous], PHONE_TO_IDX[current]

    return {key: counts[key] for key in sorted(counts, key=key_index)}


def build_synthetic_phone_counts(
    n_symbols: int = DEFAULT_PHONE_CORPUS_SYMBOLS, seed: int = DEFAULT_PHONE_CORPUS_SEED
) -> dict[str, int]:
    """Sample a training corpus and return its phone-bigram counts.

    This is the whole payload of ``vendor/synthetic_phone_counts.json``.

    Args:
        n_symbols: Training-corpus length in phones.
        seed: Seed for the training stream.

    Returns:
        The flat bigram-count mapping described in :func:`count_phone_bigrams`.
    """
    return count_phone_bigrams(generate_phone_sequence(n_symbols, seed))


def independent_phone_held_out_positions(
    positions: int,
    *,
    window: int,
    seed: int = DEFAULT_PHONE_HELD_OUT_SEED,
) -> list[tuple[str, str]]:
    """Generate one independently seeded chain trajectory per scored position.

    Mirrors :func:`independent_held_out_positions` with the same
    independent-sampling-unit guarantee: one ``window + 1`` trajectory per
    item, consecutive integer seeds naming the frozen trajectories, no
    result-dependent resampling. The prefix is the space-joined phone string
    the phoneme task's prompt and comparator already consume.

    Args:
        positions: Number of independent scored items. Must be positive.
        window: Prefix phones exposed to the model. Must be positive.
        seed: Seed of the first trajectory. Each later item adds its index.

    Returns:
        ``(space-joined prefix, next phone)`` pairs in seed order.
    """

    if positions <= 0:
        raise ValueError(f"positions must be positive, got {positions}")
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    pairs: list[tuple[str, str]] = []
    for index in range(positions):
        sequence = generate_phone_sequence(window + 1, seed + index)
        pairs.append((" ".join(sequence[:-1]), sequence[-1]))
    return pairs


def phone_target_start_tvd(window: int) -> float:
    """TVD between the scored target marginal and the chain's stationary law.

    Same estimand as :func:`independent_target_start_tvd` over the phone
    transition: the target marginal after ``window`` steps from the fixed
    start state, against the stationary distribution solved from the
    transition.
    """

    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    transition = phone_designed_transition()
    size = len(SYNTHETIC_PHONES)
    system = np.vstack((transition - np.eye(size), np.ones((1, size))))
    rhs = np.concatenate((np.zeros(size), np.ones(1)))
    stationary, *_ = np.linalg.lstsq(system, rhs, rcond=None)
    stationary = np.maximum(stationary, 0.0)
    stationary /= stationary.sum()
    start = np.zeros(size, dtype=np.float64)
    start[0] = 1.0
    target = np.linalg.matrix_power(transition, window) @ start
    return float(0.5 * np.abs(target - stationary).sum())


def load_synthetic_phone_counts(vendor_dir: Path | None = None) -> dict[str, int]:
    """Load the vendored synthetic phone-bigram counts.

    Mirrors :func:`load_synthetic_counts`, including naming the generator in
    the failure message so a missing artifact is actionable.

    Args:
        vendor_dir: Path to ``vendor/``. ``None`` reads the embedded wheel
            resource when installed and the canonical top-level copy in a
            source checkout.

    Returns:
        The flat ``"prev,next" -> count`` mapping.

    Raises:
        FileNotFoundError: If the artifact has not been generated.
    """
    counts_file = resolve_vendored_artifact("synthetic_phone_counts.json", directory=vendor_dir)
    if not counts_file.is_file():
        raise FileNotFoundError(
            f"Synthetic phone counts not found at {counts_file}. "
            "Run scripts/generate_synthetic_phone_counts.py to generate it."
        )
    return json.loads(counts_file.read_text(encoding="utf-8"))


def synthetic_phone_predictions(
    counts: dict[str, int], positions: list[tuple[str, str]]
) -> list[str]:
    """Order-1 argmax prediction for each position, trained on ``counts``.

    Deliberately identical in behavior to the external benchmark's phoneme
    comparator on nonempty prefixes — the add-1 smoothed phone transition from
    :func:`ailm.levels.phoneme.build_phone_transition`, argmax over the column
    of the prefix's final phone. An empty prefix never occurs in the executed
    protocol (the window is positive by registry validation); its fallback
    mirrors the letter comparator's mean-column convention so the two
    stipulated arms stay shape-identical. The equivalence on nonempty
    prefixes is asserted in ``tests/test_synthetic_benchmark.py`` rather than
    assumed.

    Args:
        counts: Phone-bigram counts in the vendored flat format.
        positions: ``(space-joined prefix, gold phone)`` pairs.

    Returns:
        One predicted phone per position, in order.
    """
    matrix = np.asarray(build_phone_transition(counts))[0, :, :, 0]
    return [
        PHONES[int(np.argmax(matrix[:, PHONE_TO_IDX[prefix.split()[-1]]]))]
        if prefix
        else PHONES[int(np.argmax(matrix.mean(axis=1)))]
        for prefix, _gold in positions
    ]
