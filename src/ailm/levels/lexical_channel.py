"""L3 → L2B: a lexical prefix message into the grapheme-to-phoneme prior.

The experiment this module exists to run
----------------------------------------
``docs/decisions.md`` entry 18 states the **novel-information criterion** — an
inter-level message helps iff it carries information the receiver does not
already have — and derives a prediction from it. This is the test.

The receiver is held fixed. Entry 9's channel sent L1's letter predictive into
L2B and measured significantly *harmful*; this one sends a lexical statistic into
the same level. Same receiver, different sender, opposite predictions from one
criterion. Changing only the sender is what makes the comparison controlled.

The message
-----------
``P(phone at position t | the word's letter prefix 0..t)``, estimated from the
G2P **train** split and applied to held-out pairs. It is mixed linearly into the
phone predictive::

    pred_t  <-  (1 - alpha) * pred_t  +  alpha * message_t

Linear mixing is chosen over a geometric/precision-weighted form for one reason:
at ``alpha = 0`` it is *exactly* the uncoupled decode, bit for bit. That gives an
identity anchor rather than an approximate one, and entry 9 established that an
anchor is what licenses computing a sweep in numpy at all.

Why it is conditioned on the prefix and not on word identity
------------------------------------------------------------
Conditioning on the full word would hand over the answer: the lexicon maps word
to pronunciation, so G2P would become a dictionary lookup and any "improvement"
would be the Stage 4 circularity trap wearing a new costume
(``docs/decisions.md`` entry 15). A prefix statistic estimated on train pairs and
evaluated on held-out ones is a genuine lexical generalization instead — it can
be wrong, which is the property that makes it a measurement.

Backoff
-------
A held-out word's exact context may be unseen in training, so the message backs
off. The rule is **longest suffix at a fixed position**, and it is stated here
because the tables and the lookup have to agree on it or the message describes a
different word than the one being decoded.

A table key is ``f"{position}|{context}"`` where ``context`` is the run of
letters *ending at* ``position`` — for ``through`` at position 2 the contexts are
``thr``, ``hr`` and ``r``, never ``th`` or ``t``. Lookup therefore shortens the
context **from the left**, keeping the current letter in every candidate::

    for L in range(min(len(context), max_prefix), 0, -1):
        key = f"{position}|{context[-L:]}"

and reports ``L``, the number of context letters matched, as the depth. Depth 0
means no context matched and the context-free phone distribution was used.

Shortening from the right instead would drop the newest letters and match a key
whose current letter differs from the observed one, which is a different
estimator wearing this one's name. The backoff *depth actually used* is recorded,
because a channel that always backs off to the unigram is not carrying lexical
information and would produce a null for a reason that has nothing to do with the
criterion under test.

``max_prefix`` is carried on the message rather than passed at call time for the
same reason: a table built with one bound and read with another silently ignores
its own longest keys.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

__all__ = [
    "LEXICAL_SWEEP_ALPHAS",
    "LexicalMessage",
    "build_lexical_message",
    "coupled_g2p_positions",
    "coupled_g2p_posteriors",
    "lexical_message_for",
    "lexical_sweep_claim_suffix",
    "scramble_phone_distribution",
    "sweep_lexical_alpha",
]

logger = logging.getLogger(__name__)

#: Canonical mixing-weight protocol for the L3 -> L2B lexical intervention.
#: The recorder, verifier, figure, and manuscript generator all consume this
#: tuple so an interior rung cannot disappear from one surface unnoticed.
LEXICAL_SWEEP_ALPHAS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)


def lexical_sweep_claim_suffix(alpha: float) -> str:
    """Return the canonical one-decimal ledger suffix for a sweep weight."""
    if alpha not in LEXICAL_SWEEP_ALPHAS:
        raise ValueError(f"alpha {alpha!r} is not on the registered lexical sweep grid")
    return f"{alpha:.1f}".replace(".", "_")


class LexicalMessage:
    """``P(phone_t | letter context)`` tables with backoff, estimated from train pairs.

    Attributes:
        by_prefix: ``"position|context" -> (n_phones,)`` distribution, for keys
            seen at least ``min_count`` times. ``context`` ends at ``position``.
        fallback: The context-free phone distribution used when no key matches.
        n_phones: Phone inventory size.
        max_prefix: The longest context the tables were built over. Lookup uses
            the same bound, so the tables and the reader cannot disagree.
    """

    def __init__(
        self,
        by_prefix: dict[str, NDArray[np.float64]],
        fallback: NDArray[np.float64],
        n_phones: int,
        max_prefix: int = 6,
    ) -> None:
        self.by_prefix = by_prefix
        self.fallback = fallback
        self.n_phones = n_phones
        self.max_prefix = max_prefix

    def lookup(self, position: int, context: str) -> tuple[NDArray[np.float64], int]:
        """Longest-**suffix** distribution at ``position``, with the depth used.

        Tries ``f"{position}|{context[-L:]}"`` for ``L`` from
        ``min(len(context), self.max_prefix)`` down to ``1`` and returns the first
        hit. Shortening the context from the left is what keeps the current
        letter — ``context[-1]`` — inside every candidate key; the table keys were
        built the same way, so a match always describes a word whose letter at
        ``position`` is the observed one.

        Args:
            position: The 0-based position being decoded.
            context: The run of letters ending at ``position`` (inclusive). The
                caller may pass the whole prefix ``word[: position + 1]``; the
                ``max_prefix`` bound is applied here.

        Returns:
            ``(distribution, depth)`` where ``depth`` is the number of **context
            letters** matched — never the length of the joined key string, which
            includes the ``"position|"`` stem — or ``0`` when the fallback was
            used. The depth is returned rather than discarded so a caller can
            report how often the channel had anything lexical to say.
        """
        for length in range(min(len(context), self.max_prefix), 0, -1):
            found = self.by_prefix.get(f"{position}|{context[-length:]}")
            if found is not None:
                return found, length
        return self.fallback, 0


def build_lexical_message(
    train_pairs: list[tuple[str, list[str]]],
    phones: list[str],
    *,
    max_prefix: int = 6,
    min_count: int = 3,
    smoothing: float = 1.0,
) -> LexicalMessage:
    """Estimate the prefix→phone tables from the G2P training split.

    Args:
        train_pairs: ``(word, phone_sequence)`` pairs, length-matched.
        phones: The phone inventory, indexing the returned distributions.
        max_prefix: Longest prefix conditioned on. Longer prefixes are sparser
            and back off more often, so this bounds the table rather than the
            information.
        min_count: Prefixes seen fewer times than this are dropped, because a
            distribution estimated from one or two words is memorisation rather
            than a lexical generalization.
        smoothing: Add-k pseudocount.

    Returns:
        The estimated message.

    Raises:
        ValueError: If ``train_pairs`` is empty or ``min_count`` is below 1.
    """
    if not train_pairs:
        raise ValueError("train_pairs is empty; there is nothing to estimate")
    if min_count < 1:
        raise ValueError(
            f"min_count must be at least 1, got {min_count}. At 0 a prefix seen "
            f"once would contribute a point-mass distribution, which memorises "
            f"that word rather than generalizing."
        )

    index = {p: i for i, p in enumerate(phones)}
    counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(phones), dtype=np.float64))
    seen: dict[str, int] = defaultdict(int)
    overall = np.zeros(len(phones), dtype=np.float64)

    for word, sequence in train_pairs:
        if len(word) != len(sequence):
            continue
        for position, phone in enumerate(sequence):
            slot = index.get(phone)
            if slot is None:
                continue
            overall[slot] += 1.0
            for length in range(1, min(position + 1, max_prefix) + 1):
                key = f"{position}|{word[position + 1 - length : position + 1]}"
                counts[key][slot] += 1.0
                seen[key] += 1

    by_prefix = {
        key: (row + smoothing) / (row + smoothing).sum()
        for key, row in counts.items()
        if seen[key] >= min_count
    }
    fallback = (overall + smoothing) / (overall + smoothing).sum()
    return LexicalMessage(by_prefix, fallback, len(phones), max_prefix)


def lexical_message_for(
    message: LexicalMessage, word: str
) -> tuple[NDArray[np.float64], list[int]]:
    """Per-position message for one word, plus the backoff depth used at each.

    Position ``t`` is looked up on ``word[: t + 1]``, the letters ending at ``t``,
    so :meth:`LexicalMessage.lookup` applies the longest-suffix rule to the full
    available context. The ``max_prefix`` bound comes from the message, which is
    the same bound its tables were built under.

    Returns:
        ``(T, n_phones)`` array of distributions and the per-position depths, each
        depth being the number of context letters matched (0 for the fallback).
    """
    rows = np.empty((len(word), message.n_phones), dtype=np.float64)
    depths: list[int] = []
    for position in range(len(word)):
        distribution, depth = message.lookup(position, word[: position + 1])
        rows[position] = distribution
        depths.append(depth)
    return rows, depths


def scramble_phone_distribution(
    rows: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Reverse the phone axis — the entropy-preserving negative control.

    A fixed involutive permutation, so the scrambled message has identical
    entropy, peakedness and normalization to the real one; only its
    correspondence to the phone inventory is destroyed. Replacing the message
    with a uniform one instead would confound content with entropy, which is the
    same argument entry 9 records.
    """
    return np.asarray(rows, dtype=np.float64)[:, ::-1].copy()


def coupled_g2p_posteriors(
    emission: NDArray[np.float64],
    transition: NDArray[np.float64],
    start: NDArray[np.float64],
    letter_indices: NDArray[np.int64],
    message_rows: NDArray[np.float64] | None,
    alpha: float,
) -> NDArray[np.float64]:
    """Per-position phone posteriors of the predictive-site linear recursion.

    This is the single implementation of the recursion:
    :func:`coupled_g2p_positions` derives its hit vector from these rows, and the
    proper-loss scoring pass reads the gold phone's mass from the same rows, so
    the accuracy axis and the loss axis cannot silently be computed from two
    recursions that have drifted apart.

    At ``alpha = 0`` this is *exactly* the uncoupled decode — the mixture term
    vanishes identically, not approximately — which is the identity anchor the
    sweep rests on.

    Args:
        emission: ``(n_letters, n_phones)`` ``P(letter | phone)``.
        transition: ``(n_phones, n_phones)`` ``P(next phone | current)``.
        start: ``(n_phones,)`` first-phone distribution.
        letter_indices: The word's letters.
        message_rows: ``(T, n_phones)`` lexical message, or None when ``alpha``
            is zero.
        alpha: Mixing weight in ``[0, 1]``.

    Returns:
        ``(T, n_phones)`` posterior rows, one per position.

    Raises:
        ValueError: If ``alpha`` is outside ``[0, 1]`` or a message is required
            and absent.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1], got {alpha!r}")
    if alpha > 0.0 and message_rows is None:
        raise ValueError("alpha > 0 requires a message")

    predictive = np.asarray(start, dtype=np.float64).copy()
    posteriors = np.empty((len(letter_indices), predictive.shape[0]), dtype=np.float64)
    for position, letter in enumerate(letter_indices):
        if alpha > 0.0:
            if message_rows is None:
                raise RuntimeError("message validation was bypassed for alpha > 0")
            predictive = (1.0 - alpha) * predictive + alpha * message_rows[position]
            total = predictive.sum()
            if total > 0.0:
                predictive = predictive / total
        likelihood = emission[letter, :]
        posterior = likelihood * predictive
        mass = posterior.sum()
        if mass <= 0.0:
            # A letter the emission gives no phone any mass for. Fall back to the
            # predictive rather than dividing by zero; recorded as a miss unless
            # it happens to be right.
            posterior = predictive.copy()
            mass = posterior.sum()
        posterior = posterior / mass
        posteriors[position] = posterior
        predictive = transition @ posterior
    return posteriors


def coupled_g2p_positions(
    emission: NDArray[np.float64],
    transition: NDArray[np.float64],
    start: NDArray[np.float64],
    letter_indices: NDArray[np.int64],
    truth: list[int],
    message_rows: NDArray[np.float64] | None,
    alpha: float,
) -> list[bool]:
    """Forward-filter G2P with the lexical message mixed into the phone predictive.

    The recursion lives in :func:`coupled_g2p_posteriors`; this reads the argmax
    off each row.

    Args:
        emission: ``(n_letters, n_phones)`` ``P(letter | phone)``.
        transition: ``(n_phones, n_phones)`` ``P(next phone | current)``.
        start: ``(n_phones,)`` first-phone distribution.
        letter_indices: The word's letters.
        truth: Gold phone indices, aligned with ``letter_indices``.
        message_rows: ``(T, n_phones)`` lexical message, or None when ``alpha``
            is zero.
        alpha: Mixing weight in ``[0, 1]``.

    Returns:
        One boolean per position: whether the argmax posterior matched the gold
        phone. Position-aligned so two runs can be paired for a McNemar test.

    Raises:
        ValueError: If ``alpha`` is outside ``[0, 1]`` or a message is required
            and absent.
    """
    posteriors = coupled_g2p_posteriors(
        emission, transition, start, letter_indices, message_rows, alpha
    )
    return [int(np.argmax(row)) == gold for row, gold in zip(posteriors, truth, strict=True)]


def sweep_lexical_alpha(
    pairs: list[tuple[str, list[str]]],
    emission: NDArray[np.float64],
    transition: NDArray[np.float64],
    start: NDArray[np.float64],
    message: LexicalMessage,
    phones: list[str],
    alpha_to_idx: dict[str, int],
    grid: tuple[float, ...],
    *,
    scrambled: bool = False,
) -> dict[float, list[bool]]:
    """Position-aligned hit vectors for every alpha in ``grid``.

    Alignment across alphas is what makes the paired test possible; a sweep whose
    position vectors differ in length cannot be compared (entry 5).

    Raises:
        ValueError: If ``grid`` lacks ``0.0`` or fewer than three positive values,
            which would make "no alpha beat the anchor" uninterpretable.
    """
    positives = {a for a in grid if a > 0.0}
    if 0.0 not in grid or len(positives) < 3:
        raise ValueError(
            f"grid must contain 0.0 plus at least three distinct positive alphas "
            f"so a null cannot come from a collapsed sweep; got {grid!r}"
        )

    phone_index = {p: i for i, p in enumerate(phones)}
    results: dict[float, list[bool]] = {a: [] for a in grid}
    for word, sequence in pairs:
        if len(word) != len(sequence):
            continue
        letters = [alpha_to_idx[c] for c in word if c in alpha_to_idx]
        if len(letters) != len(sequence):
            continue
        truth = [phone_index[p] for p in sequence if p in phone_index]
        if len(truth) != len(sequence):
            continue
        rows, _ = lexical_message_for(message, word)
        if scrambled:
            rows = scramble_phone_distribution(rows)
        for a in grid:
            results[a].extend(
                coupled_g2p_positions(
                    emission,
                    transition,
                    start,
                    np.asarray(letters, dtype=np.int64),
                    truth,
                    rows if a > 0.0 else None,
                    a,
                )
            )
    return results
