"""The decisions index must name every decision, and link to it.

`docs/decisions.md` opens with a hand-maintained table of contents. It had
silently drifted: seventeen entries existed in the file and not in the table, so
the document that is "the single record of the decisions behind AILM's numbers"
did not list a third of them. Nothing failed, because nothing checked.

Two failure modes, and they are different. A *missing* row hides a decision from
the only index a reader has. A *broken anchor* is worse in one specific way: the
row is present, the reader clicks it, and the page does not move — which reads
as the decision being absent rather than as the link being wrong.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DECISIONS = Path(__file__).resolve().parent.parent / "docs" / "decisions.md"


def _slug(heading: str) -> str:
    """GitHub's anchor rule for the decision headings used in this document.

    Lowercase, retain literal underscores alongside ASCII letters, digits, spaces
    and hyphens, then
    map each remaining space to one hyphen. Collapsing runs of whitespace is the
    trap: removing the em-dash from ``50 — Foo`` leaves two spaces, and GitHub
    emits two hyphens for them. A slug function that collapses would generate
    links one hyphen short of every anchor in this file.
    """
    return re.sub(r"[^a-z0-9 _\-]", "", heading.lower()).strip().replace(" ", "-")


def _document(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _headings(path: Path) -> list[str]:
    return [
        h for h in re.findall(r"^## (.+)$", _document(path), flags=re.M) if re.match(r"^\d+", h)
    ]


def _index_rows(path: Path) -> list[tuple[str, str]]:
    return re.findall(r"^\| \[([^\]]+)\]\(#([^)]+)\) \|", _document(path), flags=re.M)


def _assert_index_complete(path: Path) -> None:
    headings = _headings(path)
    assert headings, "no numbered decision headings found; this check is vacuous"
    labels = [label for label, _anchor in _index_rows(path)]
    listed = set(labels)
    assert len(labels) == len(listed), "duplicate decision index labels"
    expected = set()
    for heading in headings:
        match = re.match(r"^(\d+)(?:\s*\(([^)]*)\))?\s*—", heading)
        assert match, f"decision heading does not follow '<n> [(qualifier)] — title': {heading!r}"
        number, qualifier = match.groups()
        expected.add(f"{number} ({qualifier})" if qualifier else number)
    assert listed == expected, (
        f"the decisions index and the decisions disagree. Missing from the index: "
        f"{sorted(expected - listed)}. In the index with no such decision: "
        f"{sorted(listed - expected)}."
    )


def _assert_index_targets(path: Path) -> None:
    """Each row's anchor must be the anchor of the decision the row names.

    Set containment was the previous check: the anchors were compared to the set
    of real anchors, and the labels to the set of real numbers, with nothing
    pairing a row's label to a row's anchor. Under that check an index in which
    every row linked to the same valid heading passed, and so did a row reading
    ``| [52](#51-outcome--a-thing) |``. A dead link reads to a reader as a
    missing decision; a link to the WRONG decision reads as the right one.
    """
    by_label = {}
    for heading in _headings(path):
        match = re.match(r"^(\d+)(?:\s*\(([^)]*)\))?\s*—", heading)
        assert match, f"malformed heading: {heading!r}"
        number, qualifier = match.groups()
        by_label[f"{number} ({qualifier})" if qualifier else number] = _slug(heading)

    rows = _index_rows(path)
    assert rows, "no index rows found; this check is vacuous"
    wrong = [
        (label, anchor, by_label.get(label))
        for label, anchor in rows
        if by_label.get(label) != anchor
    ]
    assert wrong == [], (
        f"index rows whose link does not point at the decision they name "
        f"(label, linked anchor, correct anchor): {wrong}"
    )


def test_the_index_lists_every_decision() -> None:
    _assert_index_complete(DECISIONS)


def test_every_index_row_links_to_its_own_decision() -> None:
    _assert_index_targets(DECISIONS)


def test_real_file_index_controls_accept_an_unchanged_copy(tmp_path: Path) -> None:
    path = tmp_path / "decisions.md"
    path.write_bytes(DECISIONS.read_bytes())
    _assert_index_complete(path)
    _assert_index_targets(path)


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("missing", "Missing from the index"),
        ("wrong-target", "does not point at the decision"),
        ("stripped-underscore", "does not point at the decision"),
        ("duplicate", "duplicate decision index labels"),
    ],
)
def test_real_file_index_controls_reject_damaged_rows(
    tmp_path: Path, defect: str, message: str
) -> None:
    text = DECISIONS.read_text(encoding="utf-8")
    row = next(line for line in text.splitlines(keepends=True) if line.startswith("| [91]"))
    anchor = dict(_index_rows(DECISIONS))["91"]
    assert "_" in anchor
    if defect == "missing":
        replacement = ""
    elif defect == "wrong-target":
        replacement = row.replace(anchor, dict(_index_rows(DECISIONS))["95"], 1)
    elif defect == "stripped-underscore":
        replacement = row.replace(anchor, anchor.replace("_", ""), 1)
    else:
        assert defect == "duplicate"
        replacement = row + row
    damaged = text.replace(row, replacement, 1)
    assert damaged != text
    path = tmp_path / "decisions.md"
    path.write_text(damaged, encoding="utf-8")
    with pytest.raises(AssertionError, match=message):
        _assert_index_complete(path)
        _assert_index_targets(path)


def test_the_slug_rule_does_not_collapse_the_gap_the_em_dash_leaves() -> None:
    """NON-VACUITY for the anchor check, on the exact case that breaks it.

    Every heading in this file carries an em-dash. If ``_slug`` collapsed
    whitespace, it would produce a single hyphen where the file's anchors carry
    two, and the check above would fail on all fifty-odd rows at once — or, if
    the index were regenerated with the same wrong rule, pass while every link
    was dead.
    """
    assert _slug("50 — The combination rule") == "50--the-combination-rule"
    assert _slug("51 (outcome) — A thing") == "51-outcome--a-thing"
    assert _slug("91 — The `external_lexicon` population") == "91--the-external_lexicon-population"
    assert _slug("95 — Agreement with `infer_policies`: recorded!") == (
        "95--agreement-with-infer_policies-recorded"
    )
