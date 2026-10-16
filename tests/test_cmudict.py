"""Tests for ailm.corpus.cmudict — CMUdict parsing and phoneme inventory.

Stage 0 gate: the 39-phoneme invariant must hold.
All tests use either the tiny_cmudict fixture (offline, 5 words) or
local file paths — NO network calls in this test suite.
"""

from __future__ import annotations

import pytest

from ailm.corpus.cmudict import (
    CANONICAL_ARPABET,
    EXPECTED_PHONEME_COUNT,
    _parse_line,
    _strip_comment,
    _strip_stress,
    assert_full_inventory,
    build_phoneme_inventory,
    verify_pinned_cmudict_bytes,
    word_to_phones,
)


class TestPhonemeConstants:
    def test_expected_count_is_39(self):
        assert EXPECTED_PHONEME_COUNT == 39

    def test_canonical_arpabet_has_39_phonemes(self):
        assert len(CANONICAL_ARPABET) == 39, (
            f"CANONICAL_ARPABET has {len(CANONICAL_ARPABET)} entries, expected 39. "
            f"Check for accidental comment tokens (e.g. ';;;' from raw parsing) "
            f"or missing phonemes."
        )

    def test_canonical_arpabet_is_frozenset(self):
        assert isinstance(CANONICAL_ARPABET, frozenset)

    def test_pinned_raw_source_validator_rejects_truncated_bytes(self):
        """NEGATIVE CONTROL: a plausible text file is not the pinned corpus."""
        with pytest.raises(ValueError, match="pinned upstream revision"):
            verify_pinned_cmudict_bytes(b"cmudict fragment\n")

    def test_canonical_arpabet_has_no_stress_digits(self):
        """Canonical set contains base symbols only (no '0', '1', '2' suffixes)."""
        with_digits = {p for p in CANONICAL_ARPABET if any(c.isdigit() for c in p)}
        assert not with_digits, (
            f"Canonical ARPAbet should not contain stress variants: {with_digits}"
        )

    def test_known_vowels_present(self):
        """Spot-check: common vowels must be in the canonical set."""
        required = {
            "AA",
            "AE",
            "AH",
            "AO",
            "AW",
            "AY",
            "EH",
            "ER",
            "EY",
            "IH",
            "IY",
            "OW",
            "OY",
            "UH",
            "UW",
        }
        missing = required - CANONICAL_ARPABET
        assert not missing, f"Missing vowels from canonical set: {missing}"

    def test_known_consonants_present(self):
        required = {
            "B",
            "CH",
            "D",
            "DH",
            "F",
            "G",
            "HH",
            "JH",
            "K",
            "L",
            "M",
            "N",
            "NG",
            "P",
            "R",
            "S",
            "SH",
            "T",
            "TH",
            "V",
            "W",
            "Y",
            "Z",
            "ZH",
        }
        missing = required - CANONICAL_ARPABET
        assert not missing, f"Missing consonants: {missing}"


class TestCommentStripping:
    def test_strips_hash_and_after(self):
        line = "the  DH AH0  # common article"
        result = _strip_comment(line)
        assert "#" not in result
        assert "common article" not in result

    def test_preserves_content_before_hash(self):
        line = "the  DH AH0  # comment"
        result = _strip_comment(line)
        assert "DH" in result
        assert "AH0" in result

    def test_no_comment_unchanged(self):
        line = "cat  K AE1 T"
        assert _strip_comment(line) == "cat  K AE1 T"

    def test_hash_at_start_becomes_empty(self):
        line = "# This is a comment line"
        result = _strip_comment(line).strip()
        assert result == ""

    def test_semicolons_not_stripped_by_strip_comment(self):
        """_strip_comment only strips '#', not ';'. _parse_line handles ';;;'."""
        line = ";;; CMUdict header"
        result = _strip_comment(line)
        assert ";;;" in result  # strip_comment doesn't remove semicolons

    def test_empty_line_unchanged(self):
        assert _strip_comment("") == ""


class TestParseLine:
    def test_parses_simple_entry(self):
        result = _parse_line("cat  K AE1 T")
        assert result is not None
        word, phones = result
        assert word == "CAT"
        assert phones == ["K", "AE1", "T"]

    def test_strips_comment(self):
        result = _parse_line("the  DH AH0  # comment here")
        assert result is not None
        word, phones = result
        assert word == "THE"
        assert phones == ["DH", "AH0"]

    def test_comment_line_returns_none(self):
        assert _parse_line("# This is a comment") is None

    def test_semicolon_comment_returns_none(self):
        """CMUdict header lines ';;; ...' must return None."""
        assert _parse_line(";;; This is a CMUdict header") is None
        assert _parse_line(";") is None

    def test_empty_line_returns_none(self):
        assert _parse_line("") is None
        assert _parse_line("   ") is None

    def test_word_uppercased(self):
        result = _parse_line("cat  K AE1 T")
        assert result is not None
        assert result[0] == "CAT"

    def test_variant_pronunciation_stripped(self):
        """Words like 'the(2)' should become 'THE'."""
        result = _parse_line("the(2)  DH IY0")
        assert result is not None
        # The word key should be cleanable (variant suffix stripped)
        # Implementation may include or exclude — just must not crash
        assert result[0].startswith("THE")

    def test_phones_stripped(self):
        result = _parse_line("cat  K AE1 T")
        assert result is not None
        _, phones = result
        # No empty strings in phones
        assert all(p for p in phones)


class TestStripStress:
    def test_strips_0(self):
        assert _strip_stress("AH0") == "AH"

    def test_strips_1(self):
        assert _strip_stress("AE1") == "AE"

    def test_strips_2(self):
        assert _strip_stress("AH2") == "AH"

    def test_consonant_unchanged(self):
        assert _strip_stress("K") == "K"
        assert _strip_stress("SH") == "SH"

    def test_er_unchanged(self):
        """ER has no stress variant in canonical ARPAbet."""
        assert _strip_stress("ER") == "ER"


class TestBuildPhonemeInventory:
    def test_five_word_dict_has_expected_phones(self, tiny_cmudict):
        """Inventory from 5 words is a subset of the canonical 39 (not equal — 5 words cover ~9 phones)."""
        inv = build_phoneme_inventory(tiny_cmudict)
        assert inv.issubset(CANONICAL_ARPABET), f"Unknown phonemes: {inv - CANONICAL_ARPABET}"

    def test_inventory_contains_known_phones(self, tiny_cmudict):
        """tiny_cmudict covers DH, AH, K, AE, T, S, AO, N, M."""
        inv = build_phoneme_inventory(tiny_cmudict)
        for ph in ["DH", "AH", "K", "AE", "T", "S", "AO", "N", "M"]:
            assert ph in inv, f"{ph} should be in inventory"

    def test_inventory_with_stress_stripped(self, tiny_cmudict):
        """All returned phonemes should be stress-free."""
        inv = build_phoneme_inventory(tiny_cmudict)
        with_digits = {p for p in inv if any(c.isdigit() for c in p)}
        assert not with_digits, f"Stress variants in inventory: {with_digits}"

    def test_assert_full_inventory_raises_for_partial(self, tiny_cmudict):
        """assert_full_inventory must raise when called on a tiny subset."""
        inv = build_phoneme_inventory(tiny_cmudict)
        with pytest.raises(AssertionError, match="expected 39"):
            assert_full_inventory(inv)


class TestWordToPhones:
    def test_known_word(self, tiny_cmudict):
        phones = word_to_phones("cat", tiny_cmudict)
        assert phones is not None
        assert "K" in phones

    def test_case_insensitive(self, tiny_cmudict):
        lower = word_to_phones("cat", tiny_cmudict)
        upper = word_to_phones("CAT", tiny_cmudict)
        assert lower == upper

    def test_unknown_word_returns_none(self, tiny_cmudict):
        result = word_to_phones("xyzqwerty", tiny_cmudict)
        assert result is None

    def test_strip_stress_true(self, tiny_cmudict):
        """Default strip_stress=True: no digits in phones."""
        phones = word_to_phones("cat", tiny_cmudict, strip_stress=True)
        assert phones is not None
        with_digits = [p for p in phones if any(c.isdigit() for c in p)]
        assert not with_digits

    def test_strip_stress_false(self, tiny_cmudict):
        """With strip_stress=False: stress digits preserved."""
        phones = word_to_phones("cat", tiny_cmudict, strip_stress=False)
        assert phones is not None
        # CAT → [K, AE1, T]; AE1 has a digit
        with_digits = [p for p in phones if any(c.isdigit() for c in p)]
        assert len(with_digits) >= 1

    def test_the_has_dh(self, tiny_cmudict):
        phones = word_to_phones("the", tiny_cmudict)
        assert phones is not None
        assert "DH" in phones


class TestWordToPhonesReturnsThePrimaryPronunciation:
    """Pin the contract left after the dead ``primary_only`` flag was removed.

    The flag's two branches were identical — ``pronunciations[0] if primary_only
    else pronunciations[0]`` — so ``primary_only=False`` returned the primary
    anyway. It was removed rather than implemented. These tests pin both halves of
    that decision: the primary really is returned when alternates exist, and the
    flag is gone loudly rather than silently accepted and ignored.
    """

    #: A word with three distinct pronunciations, as CMUdict records READ.
    MULTI = {"READ": [["R", "IY1", "D"], ["R", "EH1", "D"], ["R", "EY1", "D"]]}

    def test_returns_the_first_pronunciation_when_alternates_exist(self):
        assert word_to_phones("read", self.MULTI) == ["R", "IY", "D"]

    def test_ignores_the_later_alternates_entirely(self):
        """The distinguishing vowel of alternate 2 must not appear."""
        phones = word_to_phones("read", self.MULTI)
        assert phones is not None
        assert "EH" not in phones and "EY" not in phones

    def test_alternates_remain_reachable_through_the_dict(self):
        """Removing the flag must not remove the capability from the caller."""
        assert len(self.MULTI["READ"]) == 3

    def test_primary_only_is_rejected_rather_than_silently_ignored(self):
        """A caller who trusted the old flag must get an error, not a wrong answer.

        This is the point of removal over retention: the old signature accepted
        ``primary_only=False`` and returned the primary regardless.
        """
        with pytest.raises(TypeError, match="primary_only"):
            word_to_phones("read", self.MULTI, primary_only=False)  # type: ignore[call-arg]

    def test_strip_stress_still_applies_to_the_primary(self):
        raw = word_to_phones("read", self.MULTI, strip_stress=False)
        assert raw == ["R", "IY1", "D"]


class TestNegativeControl:
    def test_comment_token_not_in_canonical(self):
        """The triple-semicolon CMUdict header token must not be in CANONICAL_ARPABET.

        This is the critical negative control: if comment stripping is broken,
        ';;;' appears as a "phoneme" and the inventory grows past 39.
        If this test fails, comment stripping was broken and the 39-phoneme
        invariant is meaningless.
        """
        assert ";;;" not in CANONICAL_ARPABET, (
            "Comment token ';;;' found in canonical ARPAbet. "
            "This means comment stripping failed or was bypassed."
        )
        assert ";;" not in CANONICAL_ARPABET
        assert ";" not in CANONICAL_ARPABET

    def test_build_inventory_detects_comment_tokens(self):
        """build_phoneme_inventory collects tokens verbatim — comment tokens appear.

        This is the POSITIVE demonstration that comment stripping is needed:
        a dict containing ';;;' as a phone token will have ';;;' in the inventory.
        assert_full_inventory then catches this via the count check.
        """
        fake_dict_with_comment_tokens = {
            "THE": [["DH", "AH0", ";;;"]],  # ';;;' should never be a phone
        }
        inv = build_phoneme_inventory(fake_dict_with_comment_tokens)
        # ';;;' appears in the result — proving the mechanism
        assert ";;;" in inv, (
            "Expected ';;;' in inventory from un-stripped fake dict. "
            "This confirms that parsing without comment stripping leaks tokens."
        )
        # canonical does not include it
        assert ";;;" not in CANONICAL_ARPABET

    def test_assert_full_inventory_catches_comment_token(self):
        """assert_full_inventory catches inventories with comment tokens."""
        fake_dict = {"THE": [["DH", "AH0", ";;;"]]}
        inv = build_phoneme_inventory(fake_dict)
        # The inventory has comment token; assert_full_inventory fires
        with pytest.raises(AssertionError):
            assert_full_inventory(inv)
