"""Coverage for case-level artifact validation paths.

``_validated_case_sequences`` is the gate between a vendored artifact and the
level's matrices: every rejection reason below is a distinct way a malformed
or hostile artifact could otherwise reach a published number.
"""

from __future__ import annotations

import pytest

from ailm.cases import CaseRole
from ailm.levels.case import UD_UPOS_TAGS, _validated_case_sequences


def _valid_sequence() -> list[list[str]]:
    upos = sorted(UD_UPOS_TAGS)[0]
    role = sorted(role.name for role in CaseRole)[0]
    return [[upos, role], [upos, role]]


def test_accepts_a_well_formed_sequence() -> None:
    sequence = _valid_sequence()
    out = _validated_case_sequences([sequence], context="test")
    assert out == [[tuple(pair) for pair in sequence]]


def test_rejects_a_non_list_container() -> None:
    with pytest.raises(ValueError, match="must be a list of token sequences"):
        _validated_case_sequences("not-a-list", context="test")


def test_rejects_a_non_list_inner_sequence() -> None:
    with pytest.raises(ValueError, match=r"test\[0\] must be a token sequence"):
        _validated_case_sequences(["not-a-sequence"], context="test")


def test_rejects_a_malformed_token_pair() -> None:
    with pytest.raises(ValueError, match=r"test\[0\]\[0\] must be \[UPOS, case\]"):
        _validated_case_sequences([[["NOUN"]]], context="test")


def test_rejects_a_three_element_pair() -> None:
    upos = sorted(UD_UPOS_TAGS)[0]
    role = sorted(r.name for r in CaseRole)[0]
    with pytest.raises(ValueError, match=r"must be \[UPOS, case\]"):
        _validated_case_sequences([[[upos, role, "extra"]]], context="test")


def test_rejects_a_non_canonical_upos_tag() -> None:
    role = sorted(r.name for r in CaseRole)[0]
    with pytest.raises(ValueError, match="not a canonical UD UPOS tag"):
        _validated_case_sequences([[["NOT_A_UPOS", role]]], context="test")


def test_rejects_a_non_caserole_label() -> None:
    upos = sorted(UD_UPOS_TAGS)[0]
    with pytest.raises(ValueError, match="is not a CaseRole"):
        _validated_case_sequences([[[upos, "NOT_A_ROLE"]]], context="test")


def test_rejects_a_bad_label_past_the_first_token() -> None:
    """The rejection message must index the true position, not collapse to the
    first token - a misindexed diagnostic sends the artifact doctor to the
    wrong line."""
    upos = sorted(UD_UPOS_TAGS)[0]
    role = sorted(r.name for r in CaseRole)[0]
    good = [upos, role]
    with pytest.raises(ValueError, match=r"test\[0\]\[3\]") as excinfo:
        _validated_case_sequences([[good, good, good, [upos, "BAD"]]], context="test")
    assert "test[0][3]" in str(excinfo.value)


def test_pairs_are_copied_not_aliased() -> None:
    """The validator must return checked tuples; a list escaping through would
    let a caller mutate a validated artifact in place."""
    sequence = _valid_sequence()
    out = _validated_case_sequences([sequence], context="test")
    assert all(isinstance(pair, tuple) for seq in out for pair in seq)
