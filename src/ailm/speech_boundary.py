"""The speech boundary (Stage 7, L7): where DSP ends and pymdp begins.

``docs/design/speech_boundary.md`` fixes the architecture: pymdp starts at the
phoneme level. Audio → phoneme conversion (Whisper-class STT) and phoneme →
audio synthesis (TTS) are DSP, deliberately outside the generative model, and
**their accuracy is not this project's claim to measure**. The declared Stage
7 gate — "Whisper → phoneme → TTS round trip" — would measure someone else's
model, and is re-scoped accordingly (`scripts/audit_gates.py`): what this
project owns is the **interface contract**, and that is what this module
implements and `tests/test_speech_boundary.py` gates.

The contract
------------
* The boundary vocabulary IS the Level 2 phone inventory — ``PHONES``, all 39,
  stress-free ARPABET. Asserted, not copied: the interface can never drift
  from the alphabet the agents infer over.
* Serialisation is space-joined ARPABET; deserialisation strips optional
  trailing stress digits (``AH0`` → ``AH``), because STT output carries them
  and stress is not part of the Level 2 state space.
* Everything else is **rejected, fail-closed**. An out-of-inventory symbol
  raises with the offending token; it is never smoothed, skipped, or guessed
  into the nearest phone, because a silently coerced observation would enter
  the hierarchy as a measurement that never happened.
* Round-trip identity: ``from_boundary(to_boundary(seq)) == seq`` for every
  valid sequence — checked in tests over every vendored CMUdict
  pronunciation, not a synthetic sample.
"""

from __future__ import annotations

import re

from ailm.levels.phoneme import PHONE_TO_IDX, PHONES

__all__ = ["BoundaryError", "from_boundary", "to_boundary"]

#: Optional trailing ARPABET stress digit, as produced by CMU-style STT output.
_STRESS_SUFFIX = re.compile(r"^(?P<phone>[A-Z]{1,3})(?P<stress>[0-2])?$")


class BoundaryError(ValueError):
    """A symbol crossed the boundary that the phone inventory cannot carry."""


def to_boundary(phones: list[str]) -> str:
    """Serialise a phone sequence for the DSP side of the boundary.

    Args:
        phones: Symbols from the 39-phone inventory.

    Returns:
        Space-joined ARPABET.

    Raises:
        BoundaryError: On an empty sequence or an out-of-inventory symbol —
            the generative side must never emit what the boundary cannot name.
    """
    if not phones:
        raise BoundaryError("empty phone sequence; there is nothing to serialise")
    for symbol in phones:
        if symbol not in PHONE_TO_IDX:
            raise BoundaryError(
                f"{symbol!r} is not in the {len(PHONES)}-phone inventory; the "
                f"generative side emitted a symbol the boundary cannot carry"
            )
    return " ".join(phones)


def from_boundary(text: str) -> list[str]:
    """Parse DSP-side output into Level 2 observations, fail-closed.

    Trailing stress digits are stripped (``AH0`` → ``AH``): STT output carries
    them, and stress is not part of the Level 2 state space. Anything else
    out of inventory raises rather than being coerced — a guessed observation
    would enter the hierarchy as a measurement that never happened.

    Args:
        text: Space-separated ARPABET, possibly stress-marked.

    Returns:
        Symbols from the 39-phone inventory, one per input token.

    Raises:
        BoundaryError: On empty input or an unrecognizable token, naming it.
    """
    tokens = text.split()
    if not tokens:
        raise BoundaryError("empty boundary input; there is nothing to observe")
    phones: list[str] = []
    for token in tokens:
        match = _STRESS_SUFFIX.match(token)
        candidate = match.group("phone") if match else None
        if candidate is None or candidate not in PHONE_TO_IDX:
            raise BoundaryError(
                f"{token!r} does not name a phone in the {len(PHONES)}-phone "
                f"inventory; refusing to coerce it. A silently guessed "
                f"observation would be a measurement that never happened."
            )
        phones.append(candidate)
    return phones
