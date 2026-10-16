"""Checkout completeness: a partial checkout must not read as a green suite.

W1-05: gate tests ``pytest.skip()`` when a vendored artifact is absent, so a
checkout missing ``vendor/`` passes the suite while testing nothing. This file
closes that hole: it FAILS when any artifact the gate tests depend on is
missing, converting a silent skip into a loud failure.

The single-source-of-truth claim: a green suite here means every gate actually
exercised its artifact. If a new stage adds a vendored artifact, add it to
``REQUIRED_ARTIFACTS`` in the same change — a gate that skips on a missing
artifact not listed here is the exact W1-05 defect this file exists to catch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Every vendored/generated artifact a gate test needs. A missing entry is a
#: FAILURE, not a skip — the gates must actually run.
REQUIRED_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("vendor/brown_counts.json", "Stage 1 letter gate"),
    ("vendor/cmudict_counts.json", "Stage 2 phoneme/G2P gates"),
    ("vendor/ud_ewt_case_counts.json", "Stage 4 case gate"),
    ("vendor/brown_word_counts.json", "Stage 3 word gate"),
    ("vendor/ud_ewt_clause_counts.json", "Stage 5 clause gate"),
    ("vendor/ud_ewt_narrative_counts.json", "Stage 6 narrative gate"),
    ("data/claims.json", "claims ledger and prose discipline"),
    ("output/reports/connected_gum.json", "authenticated connected-GUM report consumers"),
    ("output/reports/gate_audit.json", "cover, hierarchy, and Stage 6 finding checks"),
    ("output/reports/gate_audit_ablation.json", "retained case-only ablation figure checks"),
    # The Lean layer's two committed generated artifacts. Neither is vendored,
    # but both are generated-and-committed inputs a gate test reads, which is the
    # same contract.
    #
    # ``LedgerData.lean`` is belt-and-braces: ``test_formal_isomorphism.py::
    # test_the_generated_module_exists`` already FAILS on its absence rather than
    # skipping. It is listed anyway because that guard is one edit away from
    # becoming a skip, and this file is where the inventory lives.
    #
    # ``axiom_report.txt`` is NOT redundant. Every check that reads it is gated on
    # ``shutil.which("lake")``, so on a machine without the toolchain a missing
    # axiom report is currently invisible — the incomplete-proof and
    # forbidden-axiom checks skip. Its existence needs no toolchain, so it is
    # asserted here instead.
    ("formal/AILM/LedgerData.lean", "Lean ledger bridge (byte-identity regeneration)"),
    ("formal/axiom_report.txt", "Lean axiom policy: sorryAx / ofReduceBool scan"),
)


def test_every_gate_artifact_is_present() -> None:
    """A missing vendored artifact fails loudly instead of skipping gates.

    Without this, ``git clone``-ing without LFS (or a stale checkout) produces
    a green suite at ``python -m pytest -q`` — the gate tests skip, the
    discipline tests skip, and the project looks healthy while measuring
    nothing. This converts that silent skip into a hard failure naming the
    artifact and the stage it starves.
    """
    missing = [
        (rel, stage) for rel, stage in REQUIRED_ARTIFACTS if not (PROJECT_ROOT / rel).exists()
    ]
    assert missing == [], (
        "Gate artifacts missing — the suite would skip them, not test them "
        "(W1-05). Regenerate with the scripts/ generators, or restore from "
        "upstream:\n"
        + "\n".join(f"  {rel}  ({stage})" for rel, stage in missing)
        + "\nA green suite on this checkout is a lie: the stage gate tests "
        "pytest.skip() when their artifact is absent."
    )


def test_every_required_artifact_is_used_by_a_gate() -> None:
    """Each required artifact must be referenced by at least one test file.

    Guards against listing an artifact that nothing depends on (decoration)
    while some unlisted artifact is actually load-bearing.
    """
    import re

    # Read each consumer once, rather than walking and rereading the test tree
    # for each artifact. The inventory file itself is not a consumer: counting
    # it would make any new entry certify its own relevance.
    consumers = {
        path: path.read_text(encoding="utf-8")
        for path in sorted((PROJECT_ROOT / "tests").glob("*.py"))
        if path.name != Path(__file__).name
        and (path.name.startswith("test_") or path.name == "conftest.py")
    }
    assert consumers, "no gate consumers were discovered"
    uncovered = []
    for rel, _stage in REQUIRED_ARTIFACTS:
        stem = Path(rel).name.replace(".json", "")
        # Search test files for a reference to the artifact's stem.
        referenced = False
        for text in consumers.values():
            if re.search(rf"[\"\']?{re.escape(stem)}[\"\'.]", text):
                referenced = True
                break
        if not referenced:
            uncovered.append(rel)
    assert uncovered == [], (
        f"Acquired artifacts never referenced by any test: {uncovered}. "
        "Either a gate lost its dependency or the artifact is decoration — "
        "keep REQUIRED_ARTIFACTS in sync with what gates actually load."
    )


def test_gate_report_reader_accepts_the_complete_committed_reports(gate_audit_rows):
    assert "stage5_clause" in gate_audit_rows
    assert "stage5_clause_case_only" in gate_audit_rows
    assert "stage6_narrative" in gate_audit_rows


@pytest.mark.parametrize("name", ["gate_audit.json", "gate_audit_ablation.json"])
@pytest.mark.parametrize(
    "defect, message",
    [
        ("missing", "Required gate audit report missing"),
        ("empty", "gate inventory differs"),
        ("partial", "gate inventory differs"),
        ("duplicate", "duplicate gate_id"),
        ("unknown", "gate inventory differs"),
        ("object", "expected a list"),
        ("non_object_row", "expected an object"),
        ("missing_id", "missing or invalid gate_id"),
        ("outcome", "invalid outcome"),
        ("verdict", "invalid verdict"),
    ],
)
def test_gate_report_reader_rejects_missing_or_incomplete_evidence(
    tmp_path, gate_audit_report_reader, name, defect, message
):
    reports = tmp_path / "output" / "reports"
    reports.mkdir(parents=True)
    for source_name in ("gate_audit.json", "gate_audit_ablation.json"):
        source = PROJECT_ROOT / "output" / "reports" / source_name
        (reports / source_name).write_bytes(source.read_bytes())
    target = reports / name
    rows = json.loads(target.read_text(encoding="utf-8"))
    if defect == "missing":
        target.unlink()
    else:
        if defect == "empty":
            rows = []
        elif defect == "partial":
            rows.pop()
        elif defect == "duplicate":
            rows.append(rows[0])
        elif defect == "unknown":
            rows[0]["gate_id"] = "unregistered_gate"
        elif defect == "object":
            rows = {}
        elif defect == "non_object_row":
            rows = [None]
        elif defect == "missing_id":
            del rows[0]["gate_id"]
        elif defect == "outcome":
            rows[0]["outcome"] = "UNKNOWN"
        elif defect == "verdict":
            rows[0]["verdict"] = "CERTIFIED"
        target.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(AssertionError, match=message):
        gate_audit_report_reader(tmp_path)
