"""The figure-caption ratchet: new captions must name their falsifier.

The standard (``docs/style_guide.md``) asks every caption for what is plotted,
over what population and *n*, and **what would falsify the panel**. The third
clause has a mechanical proxy — a falsification statement in the caption text —
and this file enforces it over every shipped figure without exception. It was
once a ratchet against a frozen allowlist of captions predating the standard;
that allowlist reached empty and was removed rather than kept as a dormant
mechanism nobody needs and anybody could use.

Guards on the guard itself: the scan must actually find figures (an empty scan
passes any rule), and a planted non-compliant caption must be caught (the rule
can fail).
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANUSCRIPT = PROJECT_ROOT / "docs" / "manuscript"

#: Files excluded from the scan: documentation about the manuscript, not of it.
_EXCLUDED = {"SYNTAX.md", "AGENTS.md", "README.md", "preamble.md"}

#: The mechanical proxy for "states its falsifier". Any one phrase suffices.
_FALSIFIER_RE = re.compile(r"falsif|refuses\s+to\s+draw|would\s+(?:\w+\s+){0,4}fail", re.IGNORECASE)

_CAPTION_RE = re.compile(r"!\[(.*?)\]\([^)]*\)\{#(fig:[a-z_0-9]+)", re.DOTALL)


def _captions() -> dict[str, str]:
    found: dict[str, str] = {}
    for path in sorted(MANUSCRIPT.glob("*.md")):
        if path.name in _EXCLUDED:
            continue
        for match in _CAPTION_RE.finditer(path.read_text(encoding="utf-8")):
            found[match.group(2)] = match.group(1)
    return found


#: Every ``{#fig:...}`` anchor in the manuscript, however it was written.
#: Deliberately a different pattern from :data:`_CAPTION_RE`: the point is to
#: find figures the caption scanner cannot parse, so a shared regex would make
#: the comparison agree with itself by construction.
_ANCHOR_RE = re.compile(r"\{#(fig:[a-z_0-9]+)")


def _anchors() -> set[str]:
    return {
        label
        for path in sorted(MANUSCRIPT.glob("*.md"))
        if path.name not in _EXCLUDED
        for label in _ANCHOR_RE.findall(path.read_text(encoding="utf-8"))
    }


def test_the_scan_covers_every_figure_the_manuscript_declares():
    """NON-VACUITY, stated as coverage rather than as a floor.

    This check used to assert the scanner found at least fifteen captions. A
    floor is satisfied by any subset: the scanner could silently stop parsing
    half the figures and the gate would still read green, which is precisely the
    incident this repository already paid for when a prefix heuristic swept five
    of nine figures and reported completeness.

    So the population is derived instead. Every ``fig:`` anchor in the
    manuscript must have a caption the scanner parsed, and the two sets must be
    equal — an anchor with no parsed caption is a figure this gate is not
    examining, and a parsed caption with no anchor means the scanner invented a
    label. The anchor pattern is deliberately not the caption pattern, or the
    comparison would agree with itself.
    """
    captions = set(_captions())
    anchors = _anchors()
    assert anchors, "no figure anchors found at all; the manuscript layout changed"
    assert captions == anchors, (
        f"the caption scanner and the manuscript disagree about which figures "
        f"exist. Anchors with no parsed caption: {sorted(anchors - captions)}. "
        f"Parsed captions with no anchor: {sorted(captions - anchors)}. Either "
        f"way this gate is not examining what it reports on."
    )
    assert "fig:letter_transition" in captions


def test_every_declared_figure_has_a_file_on_disk():
    """A caption for a figure nobody generated is a broken image in the PDF.

    The caption gate reads Markdown and the layout gate reads PNGs, so a figure
    present in one and absent from the other is checked by neither. Extracted
    text does not show a missing image, which is why this is mechanical rather
    than left to reading the render.
    """
    referenced = {
        Path(match).name
        for path in sorted(MANUSCRIPT.glob("*.md"))
        if path.name not in _EXCLUDED
        for match in re.findall(r"!\[[^\]]*\]\(([^)]*\.png)\)", path.read_text(), re.DOTALL)
    }
    figures = {p.name for p in (PROJECT_ROOT / "output" / "figures").glob("*.png")}
    assert referenced, "no figure files referenced; the scan pattern stopped matching"
    missing = sorted(referenced - figures)
    assert missing == [], (
        f"the manuscript references {missing}, which do not exist under "
        f"output/figures. Each is a broken image in the rendered PDF, and "
        f"extracted text will not show it."
    )


def test_every_caption_states_its_falsifier():
    """The rule binds every figure, with no exception list to be added to.

    This began as a ratchet with a frozen allowlist of captions predating the
    standard. The allowlist reached empty, so it was removed: an exception
    mechanism that exists but is unused is an invitation, and the rule is
    stronger stated absolutely than stated as "everything except this set".
    """
    offenders = {
        label: caption[:80]
        for label, caption in _captions().items()
        if not _FALSIFIER_RE.search(caption)
    }
    assert offenders == {}, (
        f"captions without a falsification statement: {sorted(offenders)}. "
        f"State what observation would refute the panel (docs/style_guide.md). "
        f"There is deliberately no exception list to add the label to."
    )


def test_a_planted_noncompliant_caption_is_caught():
    """POSITIVE CONTROL: the falsifier proxy can itself fail."""
    planted = "A lovely panel showing several interesting quantities."
    assert not _FALSIFIER_RE.search(planted)
    compliant = "A marker off its bar would falsify the closed form."
    assert _FALSIFIER_RE.search(compliant)
