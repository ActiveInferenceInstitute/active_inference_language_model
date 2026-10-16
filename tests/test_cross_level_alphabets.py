"""Collected cross-level alphabet compatibility checks (P-08)."""

from __future__ import annotations


def test_cross_level_alphabets_are_compatible():
    """Adjacent levels expose the exact alphabets their contracts declare.

    Level 2B observes Level 1 letters, Level 2A's identity emission observes the
    canonical phone state space, and Level 4 must withhold fields used to derive
    its evaluation labels.
    """
    from ailm.corpus.cmudict import CANONICAL_ARPABET
    from ailm.levels.case import FORBIDDEN_OBSERVATIONS
    from ailm.levels.letter import ALPHA_TO_IDX
    from ailm.levels.phoneme import PHONES

    letter_alphabet = set(ALPHA_TO_IDX)
    assert len(letter_alphabet) > 20, "Letter alphabet suspiciously small"

    phone_set = set(PHONES)
    canonical = set(CANONICAL_ARPABET)
    assert len(PHONES) == len(phone_set), "Phoneme alphabet contains duplicate states"
    assert phone_set == canonical, (
        "Level 2A's phone states must equal canonical ARPAbet exactly; "
        f"extra={sorted(phone_set - canonical)}, missing={sorted(canonical - phone_set)}"
    )

    assert isinstance(FORBIDDEN_OBSERVATIONS, frozenset), (
        "FORBIDDEN_OBSERVATIONS must be a frozenset"
    )
    assert {"deprel", "feats"} <= FORBIDDEN_OBSERVATIONS, (
        "Level 4 must withhold both sources used to derive its evaluation labels"
    )
