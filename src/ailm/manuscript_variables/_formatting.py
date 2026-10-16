"""Prose-formatting helpers (counts, ordinals, timestamps, clauses)."""

from __future__ import annotations

import os
from datetime import (
    UTC,
    datetime,
)

from ailm.value_formatting import ManuscriptVariableError


def _build_timestamp() -> str:
    """UTC build stamp, honouring ``SOURCE_DATE_EPOCH`` for reproducible builds."""
    raw_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if raw_epoch is None or not raw_epoch.strip():
        moment = datetime.now(UTC)
    else:
        try:
            epoch = int(raw_epoch.strip())
        except ValueError as exc:
            raise ManuscriptVariableError(
                f"SOURCE_DATE_EPOCH must be a non-negative integer, got {raw_epoch!r}"
            ) from exc
        if epoch < 0:
            raise ManuscriptVariableError(
                f"SOURCE_DATE_EPOCH must be non-negative, got {raw_epoch!r}"
            )
        try:
            moment = datetime.fromtimestamp(epoch, tz=UTC)
        except (OverflowError, OSError, ValueError) as exc:
            raise ManuscriptVariableError(
                f"SOURCE_DATE_EPOCH is outside the supported timestamp range: {raw_epoch!r}"
            ) from exc
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


_COUNT_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
)
_ORDINAL_WORDS = (
    "zeroth",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
)
_FAMILY_MEMBER_PHRASES: dict[str, str] = {
    "stage6_sign_test_p": "the narrative level's sentence-sign test",
    "stage2_g2p_mcnemar_p": "the letter-to-phone decoder against its per-letter bar",
    "stage4_derived_mcnemar_p": "the case level against the derived oracle",
    "lexical_best_mcnemar_p": "the lexical channel at its best weight",
    "coupling_mcnemar_p": "the letter coupling harm",
    "fitted_paired_p": "the fitted lexical arm",
    "geometric_lexicon_paired_p": "the geometric lexical arm",
    "channel_lexicon_evidence_paired_p": "the lexical message at the evidence site",
    "subsample_prefix_mcnemar_p": "the biased prefix re-analysis",
}


def _join_prose(items: list[str]) -> str:
    """Join a list as English prose: ``a``, ``a and b``, ``a, b and c``."""
    if not items:
        raise ManuscriptVariableError("cannot join an empty list as prose")
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _count_word(value: int) -> str:
    """Spell a small architecture/outcome count for grammatical prose."""
    if not 0 <= value < len(_COUNT_WORDS):
        raise ManuscriptVariableError(
            f"no manuscript count-word rendering is defined for {value}; "
            "extend _COUNT_WORDS deliberately"
        )
    return _COUNT_WORDS[value]


def _quantified_clause(
    count: int,
    *,
    singular_verb: str,
    plural_verb: str,
    complement: str,
) -> str:
    """Render a count phrase with live zero/singular/plural agreement.

    Zero is rendered as ``none`` rather than ``zero`` so generated scientific
    prose cannot produce constructions such as ``zero are``.  The helper is
    deliberately generic enough for planted zero/one/two controls.
    """
    if count < 0:
        raise ManuscriptVariableError(f"a grammatical count cannot be negative: {count}")
    subject = "none" if count == 0 else _count_word(count)
    verb = singular_verb if count == 1 else plural_verb
    return f"{subject} {verb} {complement}"


def _separation_text(rows: list[tuple[int, str]], comparator: str) -> str:
    if not rows:
        return f"no treatment rung separated from {comparator}"
    above = [str(k) for k, direction in rows if direction == "above"]
    below = [str(k) for k, direction in rows if direction == "below"]
    parts: list[str] = []
    if above:
        parts.append(f"treatment was above {comparator} at k = {', '.join(above)}")
    if below:
        parts.append(f"treatment was below {comparator} at k = {', '.join(below)}")
    return " and ".join(parts)


def _ordinal_word(value: int) -> str:
    """Spell a small positive protocol stride as an ordinal."""
    if not 0 < value < len(_ORDINAL_WORDS):
        raise ManuscriptVariableError(
            f"no manuscript ordinal rendering is defined for {value}; "
            "extend _ORDINAL_WORDS deliberately"
        )
    return _ORDINAL_WORDS[value]
