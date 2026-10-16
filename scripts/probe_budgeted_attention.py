#!/usr/bin/env python3
"""Kill-check for budgeted attention: is adaptive selection worth its inference?

The proposal
------------
Under a budget of ``B`` observations for a word of length ``L``, choose *which*
letters to look at. The active-inference answer is to pick adaptively, by
expected free energy: observe next wherever the observation is expected to be
most informative about the hidden phone sequence. The proposal's headline was a
margin over "uniform stride".

Why this is a probe and not a level
-----------------------------------
That margin is a property of the comparator. "Uniform stride" is not one arm:
its *phase* is an unstated free parameter, and a proposal that reports the best
phase for itself against an arbitrary phase for the comparator is reporting a
choice as a measurement. The comparator this script uses instead is the best
fixed per-length schedule **fitted on the train split** — for each ``(L, B)``,
the observation set that maximizes train accuracy, searched exhaustively.
That
comparator uses zero inference at decode time and strictly less data-dependence
than the adaptive arm, so it is the honest bar: if adaptivity cannot beat a
lookup table chosen in advance, adaptivity is not what is buying the accuracy.

This script records nothing. It answers one question and the answer determines
whether a level gets built; a null arm in the ledger would imply a level exists
to be null about.

Usage
-----
    .venv/bin/python scripts/probe_budgeted_attention.py [--budget-fraction 0.5]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ailm.levels.letter import ALPHA_TO_IDX  # noqa: E402
from ailm.levels.phoneme import (  # noqa: E402  (path setup must precede the import)
    _word_letters,
    build_g2p_emission,
    build_g2p_start,
    build_g2p_transition,
    g2p_split,
    load_phoneme_counts,
)

#: Word lengths the sweep covers. Short enough that the exhaustive
#: best-fixed-schedule search over ``C(L, B)`` subsets is tractable, which is
#: what makes the comparator *optimal* rather than merely good — a greedy
#: comparator would leave the adaptive arm a margin it did not earn.
LENGTHS: tuple[int, ...] = (4, 5, 6, 7)


def _phone_index(counts: dict) -> dict[str, int]:
    phones = counts["phones"] if "phones" in counts else sorted(counts["phone_unigram"])
    return {phone: index for index, phone in enumerate(phones)}


def _filter_with_observed(
    letters: list[int],
    observed: frozenset[int],
    emission: np.ndarray,
    transition: np.ndarray,
    start: np.ndarray,
) -> np.ndarray:
    """Forward-filter a word, using the letter only at observed positions.

    At an unobserved position the likelihood is flat, so the posterior there is
    the transition prior alone. This is exactly what a budget means: the model
    still infers a phone at every position, and only some positions carry
    evidence.

    Returns:
        ``(L, n_phones)`` array of filtered posteriors.
    """
    n_positions = len(letters)
    n_phones = transition.shape[0]
    posteriors = np.zeros((n_positions, n_phones), dtype=np.float64)
    prior = start.copy()
    for position in range(n_positions):
        likelihood = emission[letters[position], :] if position in observed else np.ones(n_phones)
        posterior = prior * likelihood
        total = posterior.sum()
        posterior = posterior / total if total > 0 else np.full(n_phones, 1.0 / n_phones)
        posteriors[position] = posterior
        prior = transition @ posterior
    return posteriors


def _accuracy(
    schedule_for: dict[tuple[int, int], frozenset[int]] | None,
    pairs: list[tuple[list[int], list[int]]],
    budget_fraction: float,
    emission: np.ndarray,
    transition: np.ndarray,
    start: np.ndarray,
    *,
    adaptive: bool,
    peek: bool = False,
) -> tuple[int, int]:
    """Positions correct and positions scored, over the given pairs."""
    correct = 0
    scored = 0
    for letters, phones in pairs:
        n_positions = len(letters)
        budget = max(1, int(round(budget_fraction * n_positions)))
        if adaptive:
            observed = _adaptive_schedule(letters, budget, emission, transition, start, peek=peek)
        else:
            if schedule_for is None:
                raise ValueError("schedule_for is required when adaptive is false")
            observed = schedule_for[(n_positions, budget)]
        posteriors = _filter_with_observed(letters, observed, emission, transition, start)
        predictions = posteriors.argmax(axis=1)
        correct += int((predictions == np.asarray(phones)).sum())
        scored += n_positions
    return correct, scored


def _adaptive_schedule(
    letters: list[int],
    budget: int,
    emission: np.ndarray,
    transition: np.ndarray,
    start: np.ndarray,
    *,
    peek: bool,
) -> frozenset[int]:
    """Greedily choose the next position to observe.

    Two policies live here because the difference between them is the whole
    result, and the first version of this probe accidentally measured the wrong
    one.

    ``peek=False`` is expected free energy as an agent could actually compute
    it. A candidate position is scored by the entropy it is *expected* to leave,
    marginalizing over the observations it might yield under the model's own
    predictive distribution. The letter at that position is never read.

    ``peek=True`` scores a candidate by the entropy left after conditioning on
    the letter actually there. That is not a policy — it consumes the
    observation to decide whether to pay for it, which makes the budget
    fictional — and it is retained only as an upper bound, so that a null on the
    honest arm can be shown to be a property of the policy rather than of the
    task. Reporting the peeking number as "expected free energy" would be the
    sharpest kind of error this project makes: it produces an impressive margin
    rather than a degraded one.
    """
    n_phones = transition.shape[0]
    observed: set[int] = set()
    for _ in range(budget):
        best_position, best_score = None, np.inf
        for candidate in range(len(letters)):
            if candidate in observed:
                continue
            if peek:
                trial = frozenset(observed | {candidate})
                score = _total_entropy(
                    _filter_with_observed(letters, trial, emission, transition, start)
                )
            else:
                score = _expected_entropy(
                    letters, observed, candidate, emission, transition, start, n_phones
                )
            if score < best_score:
                best_position, best_score = candidate, score
        if best_position is None:
            break
        observed.add(best_position)
    return frozenset(observed)


def _total_entropy(posteriors: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(-(posteriors * np.log(np.where(posteriors > 0, posteriors, 1))).sum())


def _expected_entropy(
    letters: list[int],
    observed: set[int],
    candidate: int,
    emission: np.ndarray,
    transition: np.ndarray,
    start: np.ndarray,
    n_phones: int,
) -> float:
    """Entropy expected after observing ``candidate``, over what it might show.

    The predictive distribution over the candidate's letter comes from the
    model's own filtered belief at that position, so no part of this reads the
    letter that is actually there.
    """
    current = _filter_with_observed(letters, frozenset(observed), emission, transition, start)
    predictive = current[candidate]
    letter_probabilities = emission @ predictive
    total = letter_probabilities.sum()
    if total <= 0:
        return float(np.inf)
    letter_probabilities = letter_probabilities / total

    expected = 0.0
    trial = frozenset(observed | {candidate})
    for letter, probability in enumerate(letter_probabilities):
        if probability <= 0.0:
            continue
        hypothetical = list(letters)
        hypothetical[candidate] = letter
        expected += probability * _total_entropy(
            _filter_with_observed(hypothetical, trial, emission, transition, start)
        )
    return expected


def _best_fixed_schedules(
    pairs: list[tuple[list[int], list[int]]],
    budget_fraction: float,
    emission: np.ndarray,
    transition: np.ndarray,
    start: np.ndarray,
) -> dict[tuple[int, int], frozenset[int]]:
    """The exhaustively optimal fixed schedule per (length, budget), on train."""
    by_length: dict[int, list[tuple[list[int], list[int]]]] = defaultdict(list)
    for letters, phones in pairs:
        by_length[len(letters)].append((letters, phones))

    schedules: dict[tuple[int, int], frozenset[int]] = {}
    for length, items in by_length.items():
        budget = max(1, int(round(budget_fraction * length)))
        best_set, best_score = None, -1
        for candidate in combinations(range(length), budget):
            observed = frozenset(candidate)
            score = 0
            for letters, phones in items:
                posteriors = _filter_with_observed(letters, observed, emission, transition, start)
                score += int((posteriors.argmax(axis=1) == np.asarray(phones)).sum())
            if score > best_score:
                best_set, best_score = observed, score
        if best_set is None:
            raise RuntimeError(
                f"no fixed attention schedule exists for length={length}, budget={budget}"
            )
        schedules[(length, budget)] = best_set
    return schedules


def _uniform_phases(length: int, budget: int) -> list[frozenset[int]]:
    """Every phase of a uniform stride at this length and budget.

    The point of enumerating them is that "uniform stride" names a *family*, and
    the spread across its members is the size of the free parameter a single-arm
    comparison would have hidden.
    """
    if budget <= 0:
        return [frozenset()]
    stride = length / budget
    return [
        frozenset(min(length - 1, int(phase + index * stride)) for index in range(budget))
        for phase in range(max(1, int(round(stride))))
    ]


def measure_budgeted_attention(budget_fraction: float = 0.5) -> dict[str, object]:
    """Measure every arm of the budgeted-attention comparison.

    Split out of ``main`` so the numbers have a caller other than stdout:
    ``scripts/record_budgeted_attention_claims.py`` is the single writer that
    puts them in the ledger, and five of them were previously typed into
    `docs/manuscript/05_discussion.md` by hand.

    Args:
        budget_fraction: Share of positions the model may observe.

    Returns:
        Accuracies for the three arms, the adaptivity margin, the peeking
        headroom, and the uniform-stride phase spread per word length.
    """
    counts = load_phoneme_counts()
    train_pairs, test_pairs = g2p_split(counts)
    phone_index = _phone_index(counts)
    # pymdp wraps each matrix in a per-factor object array. Unwrapping to plain
    # 2-D numpy here keeps the probe's arithmetic legible and, more importantly,
    # keeps it the SAME arithmetic the oracle uses; a probe that filtered
    # differently from the shipped decoder would be measuring its own code.
    emission = np.asarray(build_g2p_emission(train_pairs)[0], dtype=np.float64)
    transition = np.asarray(build_g2p_transition(train_pairs)[0], dtype=np.float64)[:, :, 0]
    start = np.asarray(build_g2p_start(train_pairs)[0], dtype=np.float64)

    def encode(items):
        out = []
        for word, phones in items:
            if len(word) != len(phones) or len(word) not in LENGTHS:
                continue
            if any(phone not in phone_index for phone in phones):
                continue
            symbols = _word_letters(word)
            if len(symbols) != len(phones):
                continue
            letters = [ALPHA_TO_IDX[ch] for ch in symbols]
            out.append((letters, [phone_index[phone] for phone in phones]))
        return out

    train, test = encode(train_pairs), encode(test_pairs)

    schedules = _best_fixed_schedules(train, budget_fraction, emission, transition, start)
    fixed_correct, scored = _accuracy(
        schedules, test, budget_fraction, emission, transition, start, adaptive=False
    )
    adaptive_correct, _ = _accuracy(
        None, test, budget_fraction, emission, transition, start, adaptive=True, peek=False
    )
    peeking_correct, _ = _accuracy(
        None, test, budget_fraction, emission, transition, start, adaptive=True, peek=True
    )
    fixed_pct = 100.0 * fixed_correct / scored
    adaptive_pct = 100.0 * adaptive_correct / scored
    peeking_pct = 100.0 * peeking_correct / scored

    # The phase spread, which is the number the original proposal's comparator
    # concealed. Kept per length so it cannot be averaged away.
    by_length: dict[int, list] = defaultdict(list)
    for letters, phones in test:
        by_length[len(letters)].append((letters, phones))

    spreads: dict[int, dict[str, float]] = {}
    for length in sorted(by_length):
        budget = max(1, int(round(budget_fraction * length)))
        scores = []
        for observed in _uniform_phases(length, budget):
            hit = sum(
                int(
                    (
                        _filter_with_observed(le, observed, emission, transition, start).argmax(
                            axis=1
                        )
                        == np.asarray(ph)
                    ).sum()
                )
                for le, ph in by_length[length]
            )
            scores.append(100.0 * hit / (length * len(by_length[length])))
        if scores:
            spreads[length] = {
                "phases": len(scores),
                "min_pct": min(scores),
                "max_pct": max(scores),
                "spread_pts": max(scores) - min(scores),
            }

    return {
        "budget_fraction": budget_fraction,
        "lengths": list(LENGTHS),
        "train_pairs": len(train),
        "test_pairs": len(test),
        "scored_positions": scored,
        "schedules": {f"L{k[0]}_B{k[1]}": sorted(v) for k, v in sorted(schedules.items())},
        "fixed_pct": fixed_pct,
        "adaptive_pct": adaptive_pct,
        "peeking_pct": peeking_pct,
        "adaptivity_margin_pts": adaptive_pct - fixed_pct,
        "peeking_headroom_pts": peeking_pct - adaptive_pct,
        "phase_spreads": spreads,
        "max_phase_spread_pts": max((v["spread_pts"] for v in spreads.values()), default=0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-fraction", type=float, default=0.5)
    args = parser.parse_args()

    r = measure_budgeted_attention(args.budget_fraction)
    print(f"train pairs {r['train_pairs']:,}  test pairs {r['test_pairs']:,}  lengths {LENGTHS}")
    print(f"budget fraction {r['budget_fraction']}\n")
    for key, observed in r["schedules"].items():
        print(f"  fitted schedule  {key}  observe {observed}")
    print(f"\n  train-fitted fixed schedule       {r['fixed_pct']:8.4f}%")
    print(f"  adaptive, expected free energy    {r['adaptive_pct']:8.4f}%")
    print(f"  adaptive, PEEKING (not a policy)  {r['peeking_pct']:8.4f}%")
    print(
        f"\n  margin for adaptivity {r['adaptivity_margin_pts']:+8.4f} points "
        f"over {r['scored_positions']:,} positions"
    )
    print(f"  the peeking upper bound sits {r['peeking_headroom_pts']:+.4f} above the honest arm")
    print("\n  uniform-stride phase spread (the free parameter a single arm hides):")
    for length, sp in sorted(r["phase_spreads"].items()):
        print(
            f"    L={length}  phases {sp['phases']}  "
            f"{sp['min_pct']:.4f}%–{sp['max_pct']:.4f}%  spread {sp['spread_pts']:.4f} pts"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
