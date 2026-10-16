"""The incomplete literature search must not license a priority claim.

"The first fully active inference language model" is false. Friston, Parr,
Yufik, Sajid, Price and Holmes built a deep discrete active inference model of
linguistic communication in 2020, simulating both generation and comprehension,
six years before this work. Any unqualified priority sentence here is retired by
that paper on sight, and the paper is by the authors of the formalism, so a
reviewer will know it.

The bounded search found no earlier system meeting the joint corpus-fit and
held-out-metric criterion, but two broader indexes were incomplete or blocked.
That evidence supports a no-counterexample report, not a "first" claim. One
earlier corpus-fitted system with a held-out metric remains the explicit
counterexample condition.

This guard keeps the no-priority disclosure and closest-prior-work citation
present. It also forbids the formerly scoped priority wording: a qualifier
cannot turn an incomplete search into complete evidence.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MANUSCRIPT = _ROOT / "docs" / "manuscript"
_SEARCH_RECORD = _ROOT / "data" / "literature_search.json"

#: Any priority sentence attaching "first" to this model class. The search
#: receipt currently forbids both the unqualified and formerly scoped forms.
_FORBIDDEN_PRIORITY = re.compile(
    r"\bfirst\s+(?:\S+\s+){0,4}?"
    r"(?:fully\s+)?active[- ]inference\s+language\s+model\b",
    re.I,
)


def _prose(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


@pytest.mark.parametrize("path", sorted(_MANUSCRIPT.glob("*.md")), ids=lambda p: p.name)
def test_no_section_claims_priority_for_an_incomplete_search(path: Path) -> None:
    text = _prose(path)
    offenders = [
        text[max(0, m.start() - 90) : m.end() + 90] for m in _FORBIDDEN_PRIORITY.finditer(text)
    ]
    assert not offenders, (
        f"{path.name} claims priority over active inference language models "
        f"although the search receipt forbids it:\n"
        + "\n".join(f"  …{o}…" for o in offenders)
        + "\n\nReport the bounded search and counterexample condition instead."
    )


def test_bounded_no_counterexample_report_and_no_priority_disclosure_are_present() -> None:
    """POSITIVE CONTROL: silence would otherwise satisfy the prohibition above."""
    payload = json.loads(_SEARCH_RECORD.read_text(encoding="utf-8"))
    policy = payload["claim_policy"]
    assert policy["priority_claim_allowed"] is False
    assert "bounded no-counterexample report" in policy["reason"]
    assert "counterexample" in policy["retirement_condition"]

    abstract = _prose(_MANUSCRIPT / "01_introduction.md")
    # The abstract's phrasing wraps phrases across lines, so normalize
    # whitespace before matching; the abstract now says "index searches were
    # incomplete (rate-limited)" rather than the older singular form.
    normalized = " ".join(abstract.split())
    assert "earlier system satisfies the joint corpus-fit" in normalized
    assert "broader index searches were incomplete" in normalized
    assert "broader index searches were incomplete" in normalized
    assert "makes no priority claim" in abstract
    assert "@friston2020linguistic" in abstract, (
        "the introduction's bounded search report no longer cites the closest prior art"
    )


def test_the_prior_art_is_cited_and_described() -> None:
    """The closest prior work must be in the bibliography and in related work."""
    bibliography = (_MANUSCRIPT / "references.bib").read_text(encoding="utf-8")
    assert "friston2020linguistic" in bibliography
    assert "10.1016/j.neubiorev.2020.07.005" in bibliography, (
        "the Friston 2020 entry lost its DOI; an uncheckable citation of the "
        "work a priority claim is scoped against is worse than none"
    )

    related = _prose(_MANUSCRIPT / "06_related_work.md")
    assert "@friston2020linguistic" in related
    assert "Twenty Questions" in related, (
        "related work no longer says what the prior art actually did, so a "
        "reader cannot check the scoping for themselves"
    )


def test_the_guard_detects_the_sentence_it_forbids() -> None:
    """NEGATIVE CONTROL: a regex matching nothing would pass every file."""
    for banned in (
        "This is the first fully active inference language model.",
        "AILM is the first active inference language model built in pymdp.",
        "the first complete active-inference language model",
        (
            "the first discrete active inference language model fitted to "
            "natural-language corpora and evaluated with held-out quantitative metrics"
        ),
    ):
        assert _FORBIDDEN_PRIORITY.search(banned), banned

    allowed = "The targeted search found no eligible counterexample and remains incomplete."
    assert not _FORBIDDEN_PRIORITY.search(allowed)
