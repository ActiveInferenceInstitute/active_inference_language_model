"""Paired-count power and bound statistics used by several producers."""

from __future__ import annotations

import math

from ailm.benchmarks import (
    paired_difference_interval,
    paired_power,
)
from ailm.statistics import (
    DIFFERENCE_CONFIDENCE,
    SUBSAMPLE_LARGER_ARM,
    SUBSAMPLE_MATCHED_ARM,
)
from ailm.value_formatting import ManuscriptVariableError


def paired_power_from_counts(b: float, c: float) -> float:
    """Retrospective power of a paired benchmark arm, from its discordant counts.

    ``ailm.benchmarks.paired_power`` is a pure function of ``b`` and ``c`` — the
    recorder passes it nothing else — so the power of an arm whose counts are
    already in the ledger needs no new measurement, only the same call at render
    time. Three benchmark arms (alphabetic-only letter, prompt format, context
    window) recorded ``b`` and ``c`` but no power, and the supplement had to say
    detectability was unstated for them.

    This wraps ``paired_power`` rather than reimplementing it. A reimplementation
    would be the exact defect
    ``tests/test_manuscript_variables.py::TestDerivedPairedPower`` exists to
    catch: a derived number that looks like the recorded one but is computed a
    different way, so the manuscript and the ledger drift apart silently.

    Args:
        b: Discordant pairs favoring the probe, as read from the ledger. Ledger
            values arrive as floats through ``num``, so a whole-valued float is
            accepted and narrowed.
        c: Discordant pairs favoring the comparator, same convention.

    Returns:
        Power at the shared reference share, on ``[0, 1]``.

    Raises:
        ValueError: If a count is not a whole number. A fractional discordant
            count means a rate label was read where a count label was meant, and
            truncating it would hand ``paired_power`` a plausible wrong integer.
    """
    counts: list[int] = []
    for name, value in (("b", b), ("c", c)):
        if value != int(value):
            raise ValueError(
                f"discordant count {name} must be a whole number, got {value!r}; "
                "this looks like a rate label read where a count label was meant"
            )
        counts.append(int(value))
    return paired_power(counts[0], counts[1])


def _whole(name: str, value: float) -> int:
    """Narrow a ledger float to the integer count it represents.

    Ledger values arrive as floats through ``num``. A fractional value here means
    a rate label was read where a count label was meant, and truncating it would
    hand the interval machinery a plausible wrong integer — a bound that renders
    fine and is silently about the wrong arm.
    """
    if value != int(value):
        raise ValueError(
            f"count {name} must be a whole number, got {value!r}; this looks like a "
            "rate label read where a count label was meant"
        )
    return int(value)


def _integer_success_count(name: str, rate: float, total: int) -> int:
    """Recover an exact binary success count from a recorded rate."""

    scaled = rate * total
    successes = round(scaled)
    if not math.isclose(scaled, successes, rel_tol=0.0, abs_tol=1e-9):
        raise ManuscriptVariableError(
            f"{name} cannot come from an integer success count at n={total}: {rate!r}"
        )
    return successes


def _require_paired_table(
    name: str,
    *,
    first_successes: int,
    second_successes: int,
    b: int,
    c: int,
    total: int,
) -> None:
    """Require that paired marginals and discordances form a realizable 2x2 table.

    ``b`` counts first-only successes and ``c`` second-only successes.  Checking
    only ``b + c <= n`` and the marginal difference misses negative concordant
    cells, so reconstruct both diagonal cells explicitly.
    """

    if any(value < 0 for value in (first_successes, second_successes, b, c)):
        raise ManuscriptVariableError(f"{name} has a negative paired-table cell")
    if first_successes > total or second_successes > total or b + c > total:
        raise ManuscriptVariableError(
            f"{name} exceeds its paired population n={total}: "
            f"first={first_successes}, second={second_successes}, b={b}, c={c}"
        )
    both_correct_from_first = first_successes - b
    both_correct_from_second = second_successes - c
    if both_correct_from_first != both_correct_from_second:
        raise ManuscriptVariableError(
            f"{name} has incompatible paired marginals: "
            f"first-b={both_correct_from_first}, second-c={both_correct_from_second}"
        )
    both_wrong = total - both_correct_from_first - b - c
    if both_correct_from_first < 0 or both_wrong < 0:
        raise ManuscriptVariableError(
            f"{name} implies a negative concordant cell: "
            f"both_correct={both_correct_from_first}, both_wrong={both_wrong}"
        )


DERIVED_BOUND_ARMS: tuple[tuple[str, str], ...] = (
    ("fitted", "fitted_positions"),
    ("geometric_lexicon", "channel_factorial_positions"),
    ("geometric_letter", "channel_factorial_positions"),
    ("channel_letter_evidence", "channel_factorial_positions"),
    ("channel_letter_predictive", "channel_factorial_positions"),
    ("channel_lexicon_evidence", "channel_factorial_positions"),
    ("channel_lexicon_predictive", "channel_factorial_positions"),
)
SUBSAMPLE_BOUND_ARMS: tuple[str, ...] = (
    f"subsample_{SUBSAMPLE_MATCHED_ARM}",
    f"subsample_{SUBSAMPLE_LARGER_ARM}",
)
DERIVED_POWER_ARM_PREFIXES: tuple[str, ...] = (
    "bench_letter_az",
    "bench_letter_prompt",
    "bench_letter_ctx",
)
AUTHORED_MANUSCRIPT_PROBE_KEYS: tuple[str, ...] = ("gemma3", "qwen25", "smollm")
PARAMETER_MODEL_DISPLAY_NAMES: dict[str, str] = {
    "L1_letter": "L1 letters",
    "L2A_phoneme": "L2A phonemes",
    "L2B_g2p": "L2B letter-to-phone",
    "L3_word": "L3 words",
    "L4_case": "L4 case roles",
    "L5_clause": "L5 widened clause types",
    "L6_narrative": "L6 document-position terciles",
}


def paired_difference_bound_from_counts(
    b: float, c: float, n: float, *, confidence: float = DIFFERENCE_CONFIDENCE
) -> tuple[float, float]:
    """Interval bounding a paired arm's effect, from its recorded counts.

    ``ailm.benchmarks.paired_difference_interval`` is a pure function of ``b``,
    ``c`` and ``n``, all three of which several arms already have in the ledger.
    The recorder wrote ``*_delta_low_*`` / ``*_delta_high_*`` for the four
    external-benchmark arms and for nothing else, so every other paired result —
    including all four of this project's nulls — could state a contrast but not
    how large the effect can be. Those bounds need no new measurement, only the
    same call at render time.

    This wraps the recorder's own function rather than reimplementing it. A
    reimplementation is the exact defect
    ``tests/test_statistics.py::TestDerivedPairedBound`` exists to catch: a
    derived interval that looks like a recorded one but is built a different way
    — a different confidence level, a Wald interval instead of the exact
    conditional one, or the unconditional difference instead of the conditional
    mapping — so the manuscript reports two kinds of interval under one name.

    The wrapped function receives the recorded denominator directly. It validates
    ``b + c <= n`` before constructing the exact conditional interval, so an
    internally impossible ledger fails during manuscript generation instead of
    producing a plausible-looking bound. The equivalence test proves this path
    reproduces the recorder exactly on every arm that has a recorded bound.

    Args:
        b: Discordant pairs favoring the probe, as read from the ledger.
        c: Discordant pairs favoring the comparator, same convention.
        n: Total paired positions — the denominator of the difference. Getting
            this wrong is the one error the counts cannot reveal, so every arm
            derived below is checked against a recorded point estimate.
        confidence: Two-sided coverage. Defaults to the level the recorded
            bounds were built at, so derived and recorded intervals in the same
            table mean the same thing.

    Returns:
        ``(low, high)`` on the paired difference, probe minus comparator.

    Raises:
        ValueError: If a count is fractional, or if ``n`` is not positive.
    """
    wins, losses, total = _whole("b", b), _whole("c", c), _whole("n", n)
    if total <= 0:
        raise ValueError(f"n must be positive, got {n!r}")
    return paired_difference_interval(wins, losses, total, confidence=confidence)
