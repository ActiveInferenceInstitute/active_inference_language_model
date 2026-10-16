"""Bounded literature-search evidence and priority-wording guards."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ailm.manuscript_variables import _load_literature_variables

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEARCH_RECORD = PROJECT_ROOT / "data" / "literature_search.json"


def _normalized(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def _priority_wording_defects(text: str, *, priority_allowed: bool) -> list[str]:
    """Retire categorical priority language whenever the evidence policy forbids it."""
    if priority_allowed:
        return []
    normalized = " ".join(text.split()).lower()
    patterns = {
        "first discrete active inference language model": (
            r"\bfirst discrete active inference language model\b"
        ),
        "author-knowledge priority": r"to the author(?:'|’)s knowledge,? the first",
        "defensible-first claim": r"claim this project can defend.{0,80}\bfirst\b",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, normalized)]


def test_incomplete_search_policy_retires_priority_wording_with_live_control() -> None:
    payload = json.loads(SEARCH_RECORD.read_text(encoding="utf-8"))
    allowed = payload["claim_policy"]["priority_claim_allowed"]
    assert allowed is False

    authored = "\n".join(
        (PROJECT_ROOT / "docs" / "manuscript" / name).read_text(encoding="utf-8")
        for name in ("01_introduction.md", "06_related_work.md")
    )
    assert _priority_wording_defects(authored, priority_allowed=allowed) == []
    assert "makes no priority claim" in authored
    assert "explicitly forbids a priority claim" in authored

    planted = "To the author's knowledge, the first discrete active inference language model."
    assert set(_priority_wording_defects(planted, priority_allowed=False)) == {
        "first discrete active inference language model",
        "author-knowledge priority",
    }
    assert _priority_wording_defects(planted, priority_allowed=True) == []


def test_search_cutoff_has_one_evidence_owner_and_is_injected() -> None:
    literals = (PROJECT_ROOT / "data" / "manuscript_literals.yaml").read_text(encoding="utf-8")
    assert "LITERATURE_SEARCH_CUTOFF:" not in literals
    variables = _load_literature_variables(PROJECT_ROOT)
    assert variables["LITERATURE_SEARCH_CUTOFF"] == "August 2026"
    assert variables["LITERATURE_PRIORITY_CLAIM_ALLOWED"] == "false"


def test_every_cited_screened_neighbor_is_bound_to_bibliography_identity() -> None:
    payload = json.loads(SEARCH_RECORD.read_text(encoding="utf-8"))
    bibliography = _normalized(PROJECT_ROOT / "docs" / "manuscript" / "references.bib").lower()
    related = _normalized(PROJECT_ROOT / "docs" / "manuscript" / "06_related_work.md")
    cited_ids = {
        "doi:10.1016/j.neubiorev.2017.04.009",
        "doi:10.1016/j.neubiorev.2020.07.005",
        "doi:10.1016/j.heares.2020.107998",
        "arxiv:2403.19995",
        "arxiv:2308.04941",
        "arxiv:2510.05013",
    }
    candidates = {candidate["id"]: candidate for candidate in payload["screened_candidates"]}
    assert cited_ids <= candidates.keys()
    for candidate_id in cited_ids:
        identifier = candidate_id.removeprefix("doi:").removeprefix("arxiv:").lower()
        candidate_title = candidates[candidate_id]["title"].lower()
        assert identifier in bibliography or candidate_title in bibliography, candidate_id
        title_words = [word for word in candidates[candidate_id]["title"].split() if len(word) > 7]
        assert any(word.lower().strip("—:,.()") in bibliography for word in title_words)

    for citation_key in (
        "@friston2020linguistic",
        "@friston2017deep",
        "@friston2021listening",
        "@donnarumma2025reading",
        "@vijayaraghavan2024compositionality",
        "@tinker2025curiosity",
    ):
        assert citation_key in related


def test_search_receipt_preserves_external_failures_instead_of_empty_success() -> None:
    payload = json.loads(SEARCH_RECORD.read_text(encoding="utf-8"))
    searches = {entry["source"]: entry for entry in payload["searches"]}
    assert searches["arXiv API"]["status"] == "complete"
    assert searches["arXiv API"]["returned_records"] == searches["arXiv API"]["reported_total"]
    for source in ("OpenAlex API", "Semantic Scholar Graph API"):
        entry = searches[source]
        assert entry["status"].startswith(("incomplete_", "blocked_"))
        assert "429" in entry["failure"]
