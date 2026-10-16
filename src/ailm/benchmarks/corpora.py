"""Drawing evaluation positions out of the vendored corpora."""

from __future__ import annotations

import math

import numpy as np

from ailm.benchmarks.constants import LETTER_WINDOW_SHORT
from ailm.levels.letter import (
    ALPHA_LIST,
    ALPHA_TO_IDX,
    ALPHABET,
    build_letter_transition,
    normalize_text,
)


def letter_positions(text: str, window: int = LETTER_WINDOW_SHORT) -> list[tuple[str, str]]:
    """Prefix/gold pairs at every letter position, with a bounded prefix window.

    ``window`` changes only how much left context the prompt carries; the
    positions and their gold letters are identical at every window, which is
    what makes the context sweep a paired comparison.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    chars = [c for c in normalize_text(text) if c in ALPHABET]
    return [("".join(chars[max(0, i - window) : i]), chars[i]) for i in range(1, len(chars))]


def letter_passage_period(text: str) -> int:
    """Length of the shortest repeating unit of ``text`` in linear time.

    Detects *whole-string* periodicity only — a fixture built by repeating one
    passage, where a prefix window at or beyond the period puts the literal
    answer in the prompt at the aligned offset. That was the real hazard when the
    letter fixture was one passage repeated three times.

    It is not a general contamination control and must not be described as one.
    On a non-repeating fixture the period equals the string length, so the guard
    that compares a window against it can never fire, while ordinary local
    repetition still places a target after a verbatim earlier context on a small,
    window-dependent fraction of positions in the current fixture. Widening the
    window adds literal answer availability, and this function neither measures
    nor bounds it.
    """
    chars = [c for c in normalize_text(text) if c in ALPHABET]
    n = len(chars)
    if n == 0:
        return 0
    prefix = [0] * n
    matched = 0
    for index in range(1, n):
        while matched and chars[index] != chars[matched]:
            matched = prefix[matched - 1]
        if chars[index] == chars[matched]:
            matched += 1
        prefix[index] = matched
    candidate = n - prefix[-1]
    return candidate if n % candidate == 0 else n


def unique_letter_positions(
    text: str, *, window: int = LETTER_WINDOW_SHORT, long_window: int | None = None
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Deduplicated letter positions, optionally paired with a wider window.

    ``HELD_OUT_TEXT`` is a passage repeated three times, so the raw position
    list contains each ``(prefix, gold)`` pair up to three times. Both scored
    systems are deterministic, so a repeated position is not a second
    observation: it is the same observation counted again. Leaving the repeats
    in narrows every Wilson interval and makes the exact McNemar test
    anti-conservative, because the discordant pairs are perfectly correlated in
    blocks. Deduplication keeps the first occurrence of each distinct pair, in
    order, so the survivor set is deterministic.

    The wider window is taken at the *same* surviving positions rather than
    deduplicated independently, which is what keeps the context sweep paired.

    Returns:
        ``(short_positions, long_positions)``; the second list is empty when
        ``long_window`` is ``None``.
    """
    if long_window is not None:
        period = letter_passage_period(text)
        if long_window >= period:
            raise ValueError(
                f"long_window {long_window} reaches the passage period {period}, which "
                "would put the gold symbol itself in the prompt at the aligned offset"
            )
    short = letter_positions(text, window)
    wide = letter_positions(text, long_window) if long_window is not None else []
    seen: set[tuple[str, str]] = set()
    kept: list[int] = []
    for index, pair in enumerate(short):
        if pair in seen:
            continue
        seen.add(pair)
        kept.append(index)
    return ([short[i] for i in kept], [wide[i] for i in kept] if wide else [])


def phone_statistics(pairs: list[tuple[str, list[str]]]) -> tuple[dict[str, int], dict[str, int]]:
    """Phone unigram and bigram counts over one pronunciation split.

    The vendored ``phone_unigrams``/``phone_bigrams`` are accumulated over every
    CMUdict base word, which includes the words the phoneme task is scored on.
    Recomputing them from an explicit split is what makes the phoneme
    comparator held-out rather than fitted to its own test set.
    """
    unigrams: dict[str, int] = {}
    bigrams: dict[str, int] = {}
    for _word, phones in pairs:
        for phone in phones:
            unigrams[phone] = unigrams.get(phone, 0) + 1
        for previous, following in zip(phones, phones[1:], strict=False):
            key = f"{previous},{following}"
            bigrams[key] = bigrams.get(key, 0) + 1
    return unigrams, bigrams


def letter_posterior(counts: dict, prefix: str) -> dict[str, float]:
    """AILM's exact order-1 predictive distribution at one letter position.

    This is the same matrix column :func:`_letter_comparator` takes its argmax
    of, returned in full so the comparator can be scored for calibration rather
    than only for accuracy. The JAX-backed matrix is float32, but proper-score
    validation operates on Python floats. Re-normalize those converted values
    with :func:`math.fsum`: dividing by the float32 reduction can otherwise
    leave the converted row more than ``1e-7`` away from unit mass even though
    the source column passed pymdp's normalization check.
    """
    matrix = np.asarray(build_letter_transition(counts))[0]
    column = matrix[:, ALPHA_TO_IDX[prefix[-1]]] if prefix else matrix.mean(axis=1)
    values = [float(column[index]) for index in range(len(ALPHA_LIST))]
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"letter posterior has invalid total mass {total!r}")
    return {symbol: value / total for symbol, value in zip(ALPHA_LIST, values, strict=True)}


def phoneme_positions(
    counts: dict, pairs: list[tuple[str, list[str]]] | None = None
) -> list[tuple[str, str]]:
    """Next-phone positions over a pronunciation split.

    ``pairs`` defaults to every vendored G2P pair. The benchmark passes the
    held-out test split so the scored words are not the comparator's training
    words.
    """
    source = (
        [(word, list(phones)) for word, phones in pairs]
        if pairs is not None
        else [(word, phones_text.split()) for word, phones_text in counts["g2p_pairs"]]
    )
    result: list[tuple[str, str]] = []
    for _word, phones in source:
        result.extend((" ".join(phones[:i]), phones[i]) for i in range(1, len(phones)))
    return result
