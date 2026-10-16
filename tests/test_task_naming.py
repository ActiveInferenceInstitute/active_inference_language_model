"""Level 2B's task must be named for what it is, everywhere a reader looks.

The Level 2 methods subsection of `docs/manuscript/03_methods.md` argues at
length that the level performs
equal-length position-aligned letter-to-phone decoding and *not*
grapheme-to-phoneme conversion: the index pairing is a stipulation of the
evaluation, not a recovered alignment, and `use` = `Y UW S` is scored as u→Y,
s→UW, e→S with all three wrong.

`docs/manuscript/04_results.md` nevertheless opened its Level 2 discussion with "Part
B is grapheme-to-phoneme conversion". Two sections of one manuscript asserted
opposite things about the same level, and a reader who read only the results
section would take away the claim the phoneme section exists to refute.

The guard permits the phrase in exactly the two sections whose job is to
*reject* it, and forbids it everywhere else. It matches on whitespace-joined
text because the phrase is routinely split across a line break, which is how
several occurrences survived an earlier line-anchored sweep.

Identifiers are deliberately out of scope. `build_g2p_emission`, the
`stage2_g2p_*` claim labels and the `L2b_grapheme_to_phoneme.gnn.md` filename
are keys that the ledger, the tests and the specification registry resolve;
renaming them would break the single-writer discipline in exchange for a word.
`docs/manuscript/03_methods.md` states that distinction, and this file asserts
the statement is present.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MANUSCRIPT = _ROOT / "docs" / "manuscript"

#: The two sections that explain why the task is not grapheme-to-phoneme. They
#: must be able to say the words in order to reject them.
# 05_discussion.md states the task boundary in a rejecting frame ("stops short
# of many-to-many grapheme-to-phoneme conversion"), so it joins the sections
# that must be able to say the words in order to reject them.
_MAY_USE_THE_PHRASE = frozenset({"03_methods.md", "06_related_work.md", "05_discussion.md"})

_PHRASE = re.compile(r"grapheme[-–— ]to[-–— ]phoneme|grapheme[-–—]phoneme", re.I)

#: Contexts in which naming the phrase is a rejection rather than a use. A bare
#: substring check would fail the sections that correctly distinguish the tasks.
_REJECTING_CONTEXT = re.compile(
    r"(?:not|rather than|instead of|never|no)\s+(?:a\s+|an\s+|the\s+)?"
    r"(?:recovered\s+|genuine\s+|published\s+|real\s+)?"
    r"grapheme[-–— ]to[-–— ]phoneme",
    re.I,
)


def _prose(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def _sections() -> list[Path]:
    return sorted(p for p in _MANUSCRIPT.glob("*.md") if p.name not in _MAY_USE_THE_PHRASE)


@pytest.mark.parametrize("path", _sections(), ids=lambda p: p.name)
def test_no_section_names_the_task_grapheme_to_phoneme(path: Path) -> None:
    text = _prose(path)
    # Blank out the GNN filename, which is an identifier and stays.
    text = text.replace("L2b_grapheme_to_phoneme.gnn.md", "")
    offenders = []
    for match in _PHRASE.finditer(text):
        window = text[max(0, match.start() - 60) : match.end()]
        if _REJECTING_CONTEXT.search(window):
            continue
        offenders.append(text[max(0, match.start() - 70) : match.end() + 40])
    assert not offenders, (
        f"{path.name} names Level 2B's task 'grapheme-to-phoneme':\n"
        + "\n".join(f"  …{o}…" for o in offenders)
        + "\n\nmanuscript/03_methods.md argues the task is equal-length "
        "position-aligned letter-to-phone decoding and is explicitly not this. "
        "Use 'letter-to-phone' for the level and 'position-aligned letter-to-phone "
        "decoding' for the task."
    )


def test_the_phoneme_section_still_makes_the_distinction() -> None:
    """POSITIVE CONTROL: deleting the argument would satisfy the guard above."""
    phoneme = _prose(_MANUSCRIPT / "03_methods.md")
    assert "equal-length position-aligned letter-to-phone decoding" in phoneme
    assert "It is not grapheme-to-phoneme conversion" in phoneme
    assert "carry the string `g2p`" in phoneme, (
        "the identifier-versus-construct paragraph was removed; without it the "
        "retained `g2p` code and claim labels read as the task name"
    )


def test_the_specification_of_record_names_the_task_and_the_stipulation() -> None:
    """The GNN file is signposted as authoritative, so it must agree."""
    spec = _prose(_ROOT / "gnn" / "models" / "L2b_grapheme_to_phoneme.gnn.md")
    assert "Position-Aligned Letter-to-Phone Level" in spec
    assert "NOT" in spec and "grapheme-to-phoneme conversion" in spec, (
        "the specification does not state that the task is not grapheme-to-phoneme conversion"
    )
    assert "stipulation of the evaluation rather than a recovered alignment" in spec


def test_the_guard_detects_the_assertion_it_exists_to_forbid() -> None:
    """NEGATIVE CONTROL: including the rejecting-context escape hatch."""
    asserting = "Part B is grapheme-to-phoneme conversion — the first genuinely"
    assert _PHRASE.search(asserting)
    assert not _REJECTING_CONTEXT.search(asserting)

    rejecting = "which is a stipulated pairing rather than a recovered grapheme-to-phoneme"
    assert _REJECTING_CONTEXT.search(rejecting), (
        "the exemption for rejecting contexts stopped working, so correct prose "
        "would now fail the guard"
    )

    # Hard-wrapped across a line break, which is how occurrences hid before.
    wrapped = " ".join("at the letter and grapheme-to-phoneme\nlevels".split())
    assert _PHRASE.search(wrapped)
