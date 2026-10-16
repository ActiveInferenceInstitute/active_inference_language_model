"""Stage 7: the speech boundary's interface contract (ISC-24).

The declared gate — "Whisper → phoneme → TTS round trip" — would measure
Whisper's and the TTS engine's accuracy, which are someone else's models. What
this project owns is the boundary itself, and this file gates exactly that:

* the boundary vocabulary IS the Level 2 phone inventory, by identity;
* round-trip fidelity over every vendored CMUdict pronunciation — real data,
  the population the G2P level is estimated from, not a synthetic sample;
* stress digits are stripped on the way in (STT output carries them; the
  Level 2 state space does not);
* everything else is rejected fail-closed, with the offending token named —
  a coerced observation would enter the hierarchy as a measurement that never
  happened;
* the design document stating the boundary (ISC-24's actual requirement)
  exists and says so.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ailm.levels.phoneme import PHONE_TO_IDX, PHONES, load_phoneme_counts
from ailm.speech_boundary import BoundaryError, from_boundary, to_boundary

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_the_boundary_vocabulary_is_the_level2_inventory_by_identity():
    assert len(PHONES) == 39
    assert set(PHONE_TO_IDX) == set(PHONES)


def test_round_trip_over_every_vendored_pronunciation():
    """The whole CMUdict-derived population, not a sample."""
    counts = load_phoneme_counts()
    checked = 0
    for _word, phones_string in counts["g2p_pairs"]:
        phones = phones_string.split()
        assert from_boundary(to_boundary(phones)) == phones
        checked += 1
    assert checked > 30000, (
        f"only {checked} pronunciations round-tripped; the population shrank "
        f"and this gate may be examining a sliver"
    )


def test_stress_digits_are_stripped_on_the_way_in():
    assert from_boundary("AH0 M") == ["AH", "M"]
    assert from_boundary("K AE1 T") == ["K", "AE", "T"]


def test_out_of_inventory_symbols_are_rejected_and_named():
    with pytest.raises(BoundaryError, match="'XX'"):
        from_boundary("K XX T")
    with pytest.raises(BoundaryError, match="'schwa'"):
        from_boundary("schwa")
    with pytest.raises(BoundaryError, match="'AH3'"):
        from_boundary("AH3")  # 3 is not an ARPABET stress level
    with pytest.raises(BoundaryError, match="not in the 39-phone"):
        to_boundary(["K", "ZZ"])


def test_empty_input_is_refused_in_both_directions():
    with pytest.raises(BoundaryError, match="empty"):
        to_boundary([])
    with pytest.raises(BoundaryError, match="empty"):
        from_boundary("   ")


def test_the_design_document_states_the_boundary():
    """ISC-24: the DSP boundary is STATED as outside pymdp, in the design doc."""
    doc = PROJECT_ROOT / "docs" / "design" / "speech_boundary.md"
    text = doc.read_text(encoding="utf-8")
    assert "pymdp starts at the phoneme level" in text
    assert "Whisper" in text and "TTS" in text
