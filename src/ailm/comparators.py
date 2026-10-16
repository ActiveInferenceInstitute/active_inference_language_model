"""The strongest member of the comparator class, searched rather than assumed.

This project's staged-gate method requires that every gate be stated against a
comparator that is *context-free but not uniform*, because a model beating uniform
has shown only that it knows which symbols are frequent. That requirement is
necessary and, it turns out, not sufficient.

Nothing in the method says to look for the **strongest** comparator that is still
not the model. So the gates were stated against the first defensible comparator
rather than the best one, and at two levels a first-order lookup over the
*observed* symbols -- no latent state, no filtering, no inference at decode time,
and strictly less information than the filtering decode uses -- scores above the
shipped model.

That is the gate-audit defect class recurring inside the repair. The taxonomy in
:mod:`ailm.gates` is sound; the *check* is not yet a procedure, because a
procedure would search the comparator class instead of picking from it.

What this module does NOT establish. It withdraws no recorded number: the model
accuracies, the context-free bars and the margins all stand and reproduce exactly.
It withdraws an *interpretation* -- that clearing the declared bar shows the
latent-state machinery is doing the work. And it says nothing against the
project's actual claim, which is interpretability rather than accuracy: a lookup
table beating a generative model on accuracy leaves untouched the fact that the
generative model's behavior is attributable to a named emission and a named
transition, which is what the lookup cannot offer.

The comparators here are deliberately austere. Each is an argmax over counts of
``(previous observed symbol, current observed symbol) -> label``, fitted on the
training split only, backing off to the per-symbol argmax where a pair is unseen.
They are the obvious thing, which is the point: a comparator that takes ingenuity
to construct would be a weaker criticism than one that does not.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

#: Symbol standing in for "no previous symbol" at sequence start. Chosen outside
#: every alphabet in the project so it can never collide with a real symbol.
START_CONTEXT = "^"


@dataclass(frozen=True, slots=True)
class ComparatorScore:
    """One comparator's accuracy over an evaluation set."""

    accuracy_pct: float
    parameters: int
    positions: int
    backoff_positions: int


def fit_left_context_lookup(
    sequences: Sequence[Sequence[tuple[str, str]]],
) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    """Fit ``(previous, current) -> label`` and its per-symbol backoff.

    Args:
        sequences: Training sequences of ``(observed_symbol, label)`` pairs.

    Returns:
        The bigram map and the unigram backoff map, both argmax over counts.

    Raises:
        ValueError: If no sequence carries a pair. An empty fit would score as a
            constant predictor and read as a comparator.
    """
    bigram: dict[tuple[str, str], Counter] = defaultdict(Counter)
    unigram: dict[str, Counter] = defaultdict(Counter)
    for sequence in sequences:
        for index, (symbol, label) in enumerate(sequence):
            previous = sequence[index - 1][0] if index else START_CONTEXT
            bigram[(previous, symbol)][label] += 1
            unigram[symbol][label] += 1
    if not unigram:
        raise ValueError("no (symbol, label) pairs to fit; the comparator would be constant")
    return (
        {key: counts.most_common(1)[0][0] for key, counts in bigram.items()},
        {key: counts.most_common(1)[0][0] for key, counts in unigram.items()},
    )


def score_left_context_lookup(
    sequences: Sequence[Sequence[tuple[str, str]]],
    bigram: dict[tuple[str, str], str],
    unigram: dict[str, str],
) -> ComparatorScore:
    """Score the fitted lookup on held-out sequences.

    The backoff count is returned rather than hidden: a comparator that backs off
    at most positions is a unigram model wearing a bigram's name, and the reader
    needs the number to tell the two apart.
    """
    hits = total = backoff = 0
    for sequence in sequences:
        for index, (symbol, label) in enumerate(sequence):
            previous = sequence[index - 1][0] if index else START_CONTEXT
            key = (previous, symbol)
            predicted: str | None
            if key in bigram:
                predicted = bigram[key]
            else:
                predicted = unigram.get(symbol)
                backoff += 1
            hits += int(predicted == label)
            total += 1
    if total == 0:
        raise ValueError("no evaluation positions; an empty score is not a measurement")
    return ComparatorScore(
        accuracy_pct=100.0 * hits / total,
        parameters=len(bigram),
        positions=total,
        backoff_positions=backoff,
    )


def score_unigram_lookup(
    sequences: Sequence[Sequence[tuple[str, str]]], unigram: dict[str, str]
) -> ComparatorScore:
    """Score the per-symbol argmax alone.

    This is the positive control for :func:`score_left_context_lookup`: the bigram
    comparator uses strictly more context, so it must not score *below* this. If
    it does, the fit or the alignment is wrong rather than the finding being
    interesting.
    """
    hits = total = 0
    for sequence in sequences:
        for symbol, label in sequence:
            hits += int(unigram.get(symbol) == label)
            total += 1
    if total == 0:
        raise ValueError("no evaluation positions; an empty score is not a measurement")
    return ComparatorScore(
        accuracy_pct=100.0 * hits / total,
        parameters=len(unigram),
        positions=total,
        backoff_positions=0,
    )


def deranged_lookup(bigram: dict[tuple[str, str], str]) -> dict[tuple[str, str], str]:
    """The same map with its labels rotated off their own keys.

    The negative control. A comparator built this way retains the exact key set,
    the exact label multiset and the exact number of parameters, and differs only
    in which label sits at which key. It must score at or below the per-symbol
    argmax; if a derangement still beats the model, the comparison is measuring
    the label distribution rather than the mapping.

    Rotation rather than a random shuffle so the control is deterministic and
    needs no seed. Rotation by one leaves a fixed point only where two adjacent
    keys share a label, which is exactly the case where the "derangement" is
    indistinguishable from the original anyway.
    """
    keys = sorted(bigram)
    labels = [bigram[key] for key in keys]
    rotated = labels[1:] + labels[:1]
    return dict(zip(keys, rotated, strict=True))


def derange_transition_columns(matrix: NDArray[np.floating]) -> NDArray[np.floating]:
    """The same transition with its conditionals rotated off their own contexts.

    The Level 1 context-destruction control (decision 73). Column ``j`` of a
    column-stochastic transition is the conditional distribution over successors
    of symbol ``j``; rotating the columns by one over the index order preserves
    the exact multiset of conditionals, every column's normalization, and the
    parameter count, while mis-assigning every conditional to the wrong context.
    A rotation over distinct indices has no fixed points, so no context keeps its
    own conditional — the transition-shaped analogue of :func:`deranged_lookup`,
    deterministic for the same reason.

    Args:
        matrix: ``(n, n)`` column-stochastic numpy array.

    Returns:
        ``(n, n)`` array with ``out[:, j] = matrix[:, (j + 1) % n]``.

    Raises:
        ValueError: If the matrix is not square with at least two columns — a
            one-column matrix admits no derangement and returning it unchanged
            would hand back a control identical to its subject.
    """
    import numpy as np

    arr = np.asarray(matrix)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[1] < 2:
        raise ValueError(
            f"derange_transition_columns needs a square matrix with >= 2 columns, "
            f"got shape {arr.shape}"
        )
    return arr[:, list(range(1, arr.shape[1])) + [0]]


def context_marginal_distribution(
    matrix: NDArray[np.floating], weights: NDArray[np.floating]
) -> NDArray[np.floating]:
    """The transition's conditionals averaged out under fixed context weights.

    A context-free distribution *derived from the trained transition*: the class
    of comparator decision 73 measures because it is exactly what the "a
    different context-free distribution can beat the training-estimated unigram"
    objection invokes. Scoring with this vector at every position uses the same
    trained parameters as the bigram model and no context at all.

    Args:
        matrix: ``(n, n)`` column-stochastic numpy array.
        weights: ``(n,)`` probability vector over contexts (columns).

    Returns:
        ``(n,)`` probability vector ``matrix @ weights``.

    Raises:
        ValueError: If shapes disagree or the weights are not a distribution.
    """
    import numpy as np

    arr = np.asarray(matrix, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[1] != w.shape[0]:
        raise ValueError(f"shape mismatch: matrix {arr.shape} against weights {w.shape}")
    if w.min() < 0.0 or abs(w.sum() - 1.0) > 1e-9:
        raise ValueError("weights must be a probability distribution over contexts")
    marginal = arr @ w
    return marginal / marginal.sum()


__all__ = [
    "START_CONTEXT",
    "ComparatorScore",
    "context_marginal_distribution",
    "derange_transition_columns",
    "deranged_lookup",
    "fit_left_context_lookup",
    "score_left_context_lookup",
    "score_unigram_lookup",
]
