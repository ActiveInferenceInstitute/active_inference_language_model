"""No manuscript prose may report statistical power at the observed share.

Power computed at the effect size the data happened to produce is a monotone
transform of that arm's p-value: it is high exactly when p is small, so it adds
no evidence and cannot make a null informative. `src/ailm/statistics.py` says
this in the `REFERENCE_SHARE` docstring and cites Hoenig and Heisey; the
`fig:multiplicity_forest` caption says it too; and `docs/manuscript/04_results.md`
draws the correct Stage 4 conclusion from the reference-share number.

The caption of `tbl:multiplicity_family` nevertheless printed
`STAT_STAGE2_POWER_CORRECTED` and `STAT_STAGE4_POWER_CORRECTED` — the observed-share
values — while asserting they were measured against the reference share and that
a failure at either would have been informative. Both halves were false. Stage 4's
reference-share power is under four percent, so a Stage 4 null would have bounded
nothing, which is the opposite of what the caption claimed.

The guard is written against the general class — any token whose name marks it as
power at the observed share — rather than against the two labels that were wrong.
A guard naming `STAT_STAGE4_POWER_CORRECTED` would reprint the string it exists to
remove and would not catch a sixth effect added later.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MANUSCRIPT = _ROOT / "docs" / "manuscript"

#: Any `{{...}}` token that resolves to power at the observed share. The suffix
#: is the class marker: the recorder writes `_power_corrected` for observed-share
#: power and `_power_at_reference` for the reportable one.
_OBSERVED_POWER_TOKEN = re.compile(r"\{\{\s*STAT_[A-Z0-9_]*POWER_CORRECTED\s*\}\}")

#: `SYNTAX.md` documents the token vocabulary, so it must be able to name the
#: forbidden form in order to forbid it. It is prose about tokens, not a
#: rendered claim; the exemption is narrow and deliberate.
_VOCABULARY_DOC = "SYNTAX.md"


def _prose_files() -> list[Path]:
    return sorted(p for p in _MANUSCRIPT.glob("*.md") if p.name != _VOCABULARY_DOC)


@pytest.mark.parametrize("path", _prose_files(), ids=lambda p: p.name)
def test_no_section_reports_observed_share_power(path: Path) -> None:
    text = " ".join(path.read_text(encoding="utf-8").split())
    found = _OBSERVED_POWER_TOKEN.findall(text)
    assert not found, (
        f"{path.name} reports power at the observed share: {found}. That quantity is "
        "a monotone transform of the p-value and cannot make a null informative. "
        "Use the corresponding STAT_<EFFECT>_POWER_AT_REFERENCE token, and check "
        "that the surrounding sentence still holds — for Stage 4 the two numbers "
        "support opposite conclusions."
    )


def test_the_reference_share_power_is_actually_reported() -> None:
    """POSITIVE CONTROL: deleting every power number would pass the guard above.

    The point is not to remove power from the paper but to report the one that
    means something, so its presence is asserted rather than assumed.
    """
    # Scanned across the manuscript rather than in one named file: the
    # multiplicity material moved from the verification apparatus into
    # `98_supplement_statistics.md`, and a check bound to a filename reports a
    # reorganization as a missing result.
    prose = " ".join(
        " ".join(path.read_text(encoding="utf-8").split())
        for path in sorted(_MANUSCRIPT.glob("[0-9]*.md"))
    )
    for token in (
        "{{STAT_STAGE2_POWER_AT_REFERENCE}}",
        "{{STAT_STAGE4_POWER_AT_REFERENCE}}",
        "{{STAT_REFERENCE_SHARE}}",
    ):
        assert token in prose, f"{token} left the manuscript"


def test_the_two_stage_gates_still_support_opposite_conclusions() -> None:
    """The caption's claim is checked against the ledger, not just spelled.

    If a re-measurement moved Stage 4's reference-share power above Stage 2B's,
    or above the conventional 0.8, the caption's "could not" would be wrong and
    this test fails rather than letting the sentence go stale.
    """
    from ailm.claims import load_ledger

    ledger = load_ledger(_ROOT)

    def value(label: str) -> float:
        entry = ledger.get(label)
        assert entry is not None, f"{label} is not in the ledger"
        return float(entry["value"])

    stage2 = value("stat_stage2_power_at_reference")
    stage4 = value("stat_stage4_power_at_reference")
    assert stage2 > 0.8, (
        f"Stage 2B reference-share power fell to {stage2}; the caption says it "
        "could have detected the smallest interesting effect"
    )
    assert stage4 < 0.8, (
        f"Stage 4 reference-share power rose to {stage4}; the caption says it "
        "could not, and [@sec:gate_design] says so independently"
    )


def test_the_guard_matches_the_form_it_forbids() -> None:
    """NEGATIVE CONTROL: a regex matching nothing would pass every file."""
    assert _OBSERVED_POWER_TOKEN.search("reaches {{STAT_STAGE4_POWER_CORRECTED}} against")
    assert _OBSERVED_POWER_TOKEN.search("{{STAT_NEWLY_ADDED_ARM_POWER_CORRECTED}}")
    assert not _OBSERVED_POWER_TOKEN.search("{{STAT_STAGE4_POWER_AT_REFERENCE}}")
