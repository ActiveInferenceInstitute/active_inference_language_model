"""Interval estimates, paired tests, and proper scores over benchmark answers."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, TypedDict

from ailm.benchmarks.constants import CALIBRATION_BINS, G2P_BUCKETS
from ailm.levels.letter import ALPHA_LIST

# wilson_interval, paired_difference_interval, and paired_power moved to
# ailm.statistics (generic interval/power primitives, no ailm.benchmarks
# dependency). Re-exported here, unchanged, so every existing
# ``from ailm.benchmarks.scoring import ...`` and the
# ``ailm.benchmarks`` lazy-import table keep working. ``noqa`` rather than
# deletion: these names are re-exports, not internal uses.
from ailm.statistics import (  # noqa: F401
    REFERENCE_SHARE,
    clopper_pearson_interval,
    exact_binomial_power,
    exact_two_sided_binomial_p,
    paired_difference_interval,
    paired_power,
    wilson_interval,
)


class PairedDiscordance(TypedDict):
    """Three counts and a probability, under one name.

    A plain ``dict`` cannot say that ``b`` is a count and ``p_value`` is not.
    Typing it as ``dict[str, int]`` was false about the p-value, and widening to
    ``dict[str, int | float]`` made every count read as possibly fractional at
    the call sites that feed them to exact tests. The recorders read this
    structure by key and write ``b`` and ``c`` straight into the ledger, where an
    int must stay an int, so the shape is kept exactly and only the types are
    made honest.
    """

    b: int
    c: int
    discordant: int
    p_value: float


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant-pair counts.

    This is the single implementation used by benchmark recorders and by the
    manuscript's semantic reconciliation.  Keeping the count-level operation
    public avoids a verifier rebuilding the binomial tail independently (or
    manufacturing boolean vectors merely to call :func:`paired_discordance`).
    """
    wins = _validate_nonnegative_int("b", b)
    losses = _validate_nonnegative_int("c", c)
    return exact_two_sided_binomial_p(wins, losses)


def paired_discordance(
    model_correct: list[bool], reference_correct: list[bool]
) -> PairedDiscordance:
    """Count paired wins/losses and two-sided exact McNemar inputs."""
    if len(model_correct) != len(reference_correct):
        raise ValueError("paired inputs must have equal length")
    b = sum(a and not r for a, r in zip(model_correct, reference_correct, strict=True))
    c = sum((not a) and r for a, r in zip(model_correct, reference_correct, strict=True))
    n = b + c
    return PairedDiscordance(b=b, c=c, discordant=n, p_value=exact_mcnemar_p(b, c))


def _validate_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _validate_positive_int(name: str, value: object) -> int:
    validated = _validate_nonnegative_int(name, value)
    if validated == 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return validated


def _validate_finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return numeric


def _validate_positive_finite(name: str, value: object) -> float:
    numeric = _validate_finite_number(name, value)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return numeric


def _validate_probability(name: str, value: object) -> float:
    numeric = _validate_finite_number(name, value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value!r}")
    return numeric


def _validate_distributions(
    distributions: list[dict[str, float]], gold: list[str]
) -> list[dict[str, float]]:
    """Validate proper-score inputs as probability distributions.

    Sparse rows are accepted: omitted inventory symbols have probability zero.
    Extra symbols are not accepted because their unscored mass would make both
    the log and Brier scores cease to be scores over the declared inventory.
    """
    if len(distributions) != len(gold):
        raise ValueError("distributions and gold must have equal length")
    inventory = set(ALPHA_LIST)
    validated: list[dict[str, float]] = []
    for index, (distribution, target) in enumerate(zip(distributions, gold, strict=True)):
        if target not in inventory:
            raise ValueError(f"gold[{index}] is outside the declared inventory: {target!r}")
        if not isinstance(distribution, dict) or not distribution:
            raise ValueError(f"distribution[{index}] must be a non-empty mapping")
        unknown = sorted(set(distribution) - inventory)
        if unknown:
            raise ValueError(
                f"distribution[{index}] contains symbols outside the declared inventory: {unknown}"
            )
        row = {
            symbol: _validate_probability(f"distribution[{index}][{symbol!r}]", probability)
            for symbol, probability in distribution.items()
        }
        mass = math.fsum(row.values())
        if not math.isclose(mass, 1.0, rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError(f"distribution[{index}] must sum to 1, got {mass!r}")
        validated.append(row)
    return validated


def g2p_length_bucket(gold_length: int) -> str:
    """Name the gold-phone-count bucket a word falls in."""
    length = _validate_nonnegative_int("gold_length", gold_length)
    for name, low, high in G2P_BUCKETS:
        if low <= length <= high:
            return name
    raise ValueError(f"no G2P bucket covers length {length}")


def expected_calibration_error(
    confidences: list[float], correct: list[bool], bins: int = CALIBRATION_BINS
) -> float:
    """Equal-width binned |accuracy - confidence|, weighted by bin occupancy.

    The confidences are whatever the caller stated. For an externally served
    probe they are read off a reconstructed inventory distribution rather than
    off the probe's constrained output distribution, so the error below is the
    calibration of the reconstruction; see :func:`negative_log_likelihood` for
    why that distinction is not cosmetic and
    :func:`reconstructed_calibration_metrics` for where the reconstruction is
    built.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correctness must have equal length")
    bin_count = _validate_positive_int("bins", bins)
    probabilities = [
        _validate_probability(f"confidences[{index}]", value)
        for index, value in enumerate(confidences)
    ]
    for index, hit in enumerate(correct):
        if not isinstance(hit, bool):
            raise ValueError(f"correct[{index}] must be boolean, got {hit!r}")
    if not confidences:
        return 0.0
    edges = [i / bin_count for i in range(bin_count + 1)]
    total = len(probabilities)
    error = 0.0
    for index in range(bin_count):
        low, high = edges[index], edges[index + 1]
        members = [
            (p, hit)
            for p, hit in zip(probabilities, correct, strict=True)
            if (low < p <= high) or (index == 0 and p <= high)
        ]
        if not members:
            continue
        mean_confidence = sum(p for p, _ in members) / len(members)
        accuracy = sum(int(hit) for _, hit in members) / len(members)
        error += (len(members) / total) * abs(accuracy - mean_confidence)
    return error


def negative_log_likelihood(
    distributions: list[dict[str, float]], gold: list[str], floor: float = 1e-12
) -> float:
    """Mean ``-log p(gold)`` over the distributions actually passed in.

    The log score is strictly proper, and sharper than the Brier score: it
    punishes confident error without bound, where Brier saturates. The floor
    keeps a zero-probability gold symbol finite; it is reached only when a
    system assigns the truth no mass at all, which is itself worth seeing in
    the number rather than as an exception.

    WHAT PROPRIETY BUYS HERE, AND WHAT IT DOES NOT. Strict propriety is a
    property of the rule with respect to *the distribution submitted to it*: it
    says the score is minimized in expectation by reporting one's true beliefs
    over the sampled outcome. It says nothing about a distribution that was
    reconstructed before submission. For the comparator that distinction is
    empty — :func:`letter_posterior` hands over the exact predictive column the
    comparator decodes from. For an externally served probe it is the whole
    story: :func:`_distribution_from_candidates` receives at most
    ``top_logprobs`` raw candidates for a thirty-symbol inventory, discards the
    candidates outside it, bounds every unreported symbol and renormalizes, so
    what reaches this function is a reconstructed inventory distribution and not
    the probe's constrained output distribution. The number below is a proper
    score of that reconstruction. Read as a proper score of the probe, it would
    be attributing to the model a quantity the completion rule helped decide.
    """
    minimum = _validate_positive_finite("floor", floor)
    if minimum > 1.0:
        raise ValueError(f"floor must not exceed 1, got {floor!r}")
    validated = _validate_distributions(distributions, gold)
    if not validated:
        return 0.0
    return sum(
        -math.log(max(distribution.get(target, 0.0), minimum))
        for distribution, target in zip(validated, gold, strict=True)
    ) / len(validated)


def brier_score(distributions: list[dict[str, float]], gold: list[str]) -> float:
    """Multiclass Brier score over the declared symbol inventory.

    ``sum_k (p_k - y_k)^2`` averaged over positions, so 0 is a perfect
    one-hot forecast and 2 is a confident wrong one.

    Strictly proper for the distributions passed in, with the same caveat spelled
    out in :func:`negative_log_likelihood`: for an externally served probe those
    distributions are reconstructions of a truncated candidate list, so this is a
    proper score of a reconstructed inventory distribution rather than of the
    probe's constrained output distribution. How much of the resulting figure the
    reconstruction decides is measured directly, per probe, by
    :func:`dual_rule_calibration`.
    """
    validated = _validate_distributions(distributions, gold)
    if not validated:
        return 0.0
    total = 0.0
    for distribution, target in zip(validated, gold, strict=True):
        total += sum(
            (distribution.get(symbol, 0.0) - (1.0 if symbol == target else 0.0)) ** 2
            for symbol in ALPHA_LIST
        )
    return total / len(validated)


def levenshtein_align(reference: list[str], hypothesis: list[str]) -> dict[str, Any]:
    """Return edit distance, aligned position accuracy, and exact-match status."""
    d = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    for i in range(len(reference) + 1):
        d[i][0] = i
    for j in range(len(hypothesis) + 1):
        d[0][j] = j
    for i, ref in enumerate(reference, 1):
        for j, hyp in enumerate(hypothesis, 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref != hyp))
    i, j, correct = len(reference), len(hypothesis), 0
    while i and j:
        if d[i][j] == d[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            correct += int(reference[i - 1] == hypothesis[j - 1])
            i -= 1
            j -= 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return {
        "distance": d[-1][-1],
        "correct_positions": correct,
        "reference_length": len(reference),
        "position_accuracy": correct / len(reference) if reference else 0.0,
        "exact": reference == hypothesis,
    }
