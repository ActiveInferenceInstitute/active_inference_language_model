"""The order-1 entropy bound must never be stated as an impossibility theorem.

`docs/manuscript/02_background.md` states, correctly and at length, that the
training-distribution conditional entropy is *not* a lower bound on held-out
cross-entropy: the held-out sample is a different empirical distribution whose
own first-order optimum can sit either side of it.

Six places in the manuscript, figure text, and scripts nevertheless used that
quantity to assert that no letter bigram model *could* have passed the Stage 1
gate. A peer reviewer found the contradiction. Both statements cannot be true,
and the weaker one is the true one: what the measurement establishes is that the
fitting corpus does not hold the first-order information the bar demanded, which
is a finding about threshold calibration and is all the gate audit needs.

This guard is prose-level because that is where the defect lives. The arithmetic
was never wrong.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]

#: Text that asserts impossibility for *any* order-1 model without naming the
#: distribution the bound holds under. Matched against whitespace-joined text so
#: hard-wrapped prose cannot hide a phrase from a line-anchored search — the
#: failure mode that let three of these survive an earlier sweep.
_UNQUALIFIED = (
    re.compile(r"no letter bigram model (?:of any quality )?(?:could|can)\b", re.I),
    re.compile(r"no order-1 letter model (?:of any quality )?(?:could|can)\b", re.I),
    re.compile(r"could never (?:have )?pass", re.I),
    re.compile(r"impossible for any (?:order-1|bigram)", re.I),
)

#: Every file that has ever carried the claim, plus the ones that carry the
#: correct qualified form. Listing them explicitly rather than globbing keeps a
#: new file from silently escaping the guard only if someone also forgets to add
#: it here — which the repository-wide sweep below catches.
_KNOWN_SITES = (
    "docs/manuscript/02_background.md",
    "docs/manuscript/03_methods.md",
    "docs/manuscript/04_results.md",
    "src/ailm/figures/results.py",
    "scripts/audit_gates.py",
    "scripts/README.md",
)


def _joined(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


@pytest.mark.parametrize("relative", _KNOWN_SITES)
def test_known_sites_do_not_assert_impossibility(relative: str) -> None:
    text = _joined(_ROOT / relative)
    for pattern in _UNQUALIFIED:
        assert not pattern.search(text), (
            f"{relative} asserts that no order-1 model could pass the Stage 1 gate. "
            "docs/manuscript/02_background.md says the training-distribution conditional "
            "entropy is not a lower bound on held-out cross-entropy, so this is an "
            "internal contradiction. State the bound under p_train instead."
        )


def test_the_sweep_covers_every_tracked_text_file() -> None:
    """Repository-wide, so a new file cannot carry the claim unnoticed."""
    offenders: list[str] = []
    for suffix in ("*.md", "*.py"):
        for directory in ("manuscript", "src", "scripts", "docs"):
            for path in (_ROOT / directory).rglob(suffix):
                text = _joined(path)
                if any(pattern.search(text) for pattern in _UNQUALIFIED):
                    offenders.append(str(path.relative_to(_ROOT)))
    assert not offenders, (
        "these files assert order-1 impossibility without naming the distribution: "
        f"{sorted(offenders)}"
    )


def test_the_correct_qualified_statement_is_present() -> None:
    """POSITIVE CONTROL: the guard would be satisfied by deleting the argument.

    Removing every mention of the ceiling would pass all three checks above while
    destroying the Stage 1 finding, so the qualified claim is asserted present.
    """
    background = _joined(_ROOT / "docs" / "manuscript" / "02_background.md")
    assert "not* a lower bound on held-out cross-entropy" in background

    gate_design = _joined(_ROOT / "docs" / "manuscript" / "04_results.md")
    assert "The population qualifier is not a hedge" in gate_design, (
        "the gate-design section no longer explains why the ceiling is stated under "
        "the training distribution"
    )
    assert "{{STAGE1_CEILING_SHORTFALL_RATIO}}" in gate_design, (
        "the shortfall ratio left the gate-design section; the Stage 1 finding is "
        "no longer quantified where it is argued"
    )


def test_the_guard_detects_the_phrasing_it_exists_to_forbid() -> None:
    """NEGATIVE CONTROL: a regex that matches nothing would pass silently."""
    reintroduced = " ".join(
        [
            "The",
            "conditional",
            "entropy",
            "caps",
            "the",
            "gain,",
            "so",
            "no",
            "letter",
            "bigram",
            "model",
            "could",
            "have",
            "passed",
            "it.",
        ]
    )
    assert any(pattern.search(reintroduced) for pattern in _UNQUALIFIED)
