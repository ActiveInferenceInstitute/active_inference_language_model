"""Tests for scripts/probe_l4_case_marking_treebank.py."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.probe_l4_case_marking_treebank import (  # noqa: E402
    CEREBRUM_CASE_INVENTORY,
    UD_CASE_FEATURE_MAP,
    audit_english_gold_vs_caserich_properties,
)


def test_cerebrum_8_case_inventory_completeness():
    """Verify Sanskrit/CEREBRUM 8-case ontology mapping."""
    assert len(CEREBRUM_CASE_INVENTORY) == 8
    assert len(UD_CASE_FEATURE_MAP) == 8
    assert set(UD_CASE_FEATURE_MAP.keys()) == {
        "Nom",
        "Acc",
        "Gen",
        "Dat",
        "Ins",
        "Loc",
        "Abl",
        "Voc",
    }


def test_english_gold_alphabet_collapse_and_power_requirement():
    """Verify that English gold is 100% PRON with only 3 cases present."""
    res = audit_english_gold_vs_caserich_properties()
    assert res["english_gold_upos_types"] == ["PRON"]
    assert set(res["english_gold_cases_present"]) == {"Nom", "Gen", "Acc"}
    assert res["minimum_discordant_pairs_required"] == 420
