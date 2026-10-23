"""Every ledger claim carries a machine-readable epistemic status, honestly.

Why this file exists
--------------------
The external audit asked for a claims table that says what *kind* of evidence
each number is — ``confirmatory`` / ``exploratory`` / ``descriptive`` /
``analytic`` — so a reader (or a test) can tell a pre-registered result from a
test-selected sweep without reading prose. Schema ``ailm-claims-v2`` adds the
optional per-entry ``status`` field; ``scripts/assign_claim_status.py`` holds
the mechanical assignment rule, and ``data/registration_provenance.json`` binds
any promotion to immutable Git history rather than a prose citation alone.

Two failure modes are worth naming, and each gets a gate here:

1. **Erosion.** The field is optional on write (recorders predate it, and a
   forced field would break every concurrent writer), so nothing structural
   stops the ledger from silently accumulating unclassified entries. The
   ledger-wide gate below fails while any claim lacks a status and names the
   assignment script as the fix.
2. **Inflation.** ``confirmatory`` is the one status worth lying about, and the
   lie is cheap: a recorder types the word. So the string "confirmatory" may
   not appear in a recorder module unless that module's docstring cites the
   pre-registration decision entry that earns it — grep-based, in the style of
   the split-sequence scan in ``tests/test_claims_single_writer.py`` — and
   every ledger entry carrying the status must trace to a source whose
   docstring cites a decision number. Ambiguity resolves downward: the
   assignment script's default is ``descriptive``, never up.

Statuses survive replay by inheritance: a re-record that names no status keeps
the stored one, because the status classifies the measurement's design, not the
run. Without that, ``test_rerunning_the_recorders_is_a_no_op`` would fail on
every replay of a recorder that does not pass ``status=``.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from ailm.claims import CLAIM_STATUSES, ClaimBatch, ClaimsLedger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEDGER_FILE = PROJECT_ROOT / "data" / "claims.json"
ASSIGN_SCRIPT = PROJECT_ROOT / "scripts" / "assign_claim_status.py"
REGISTRATION_PROVENANCE = PROJECT_ROOT / "data" / "registration_provenance.json"

_SKIP_NO_LEDGER = pytest.mark.skipif(
    not LEDGER_FILE.exists(), reason="data/claims.json absent; run the recorders"
)

#: A docstring citation of a decisions.md entry: "decision 29", "entry 49",
#: "decisions 25-26". Case-insensitive so sentence position does not matter.
_CITATION = re.compile(r"\b(?:decision|entry|entries)s?\s+\d+", re.IGNORECASE)


def _claims() -> dict:
    return json.loads(LEDGER_FILE.read_text(encoding="utf-8"))["claims"]


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _registration_provenance_defects(payload: dict) -> list[str]:
    """Verify temporal classes against immutable Git objects and file history."""

    defects: list[str] = []
    if payload.get("_schema") != "ailm-registration-provenance-v1":
        return ["schema"]
    protocols = payload.get("protocols")
    if not isinstance(protocols, dict) or set(protocols) != {
        "18",
        "25",
        "27",
        "29",
        "46",
        "49",
        "51",
        "73",
        "74",
        "77",
        "78",
        "79",
        "111",
    }:
        return ["protocol inventory"]
    for decision, record in protocols.items():
        registration = record.get("registration_commit")
        outcome = record.get("outcome_commit")
        temporal_class = record.get("temporal_class")
        eligible = record.get("confirmatory_status_eligible")
        if not isinstance(registration, str) or re.fullmatch(r"[0-9a-f]{40}", registration) is None:
            defects.append(f"decision {decision} registration commit")
            continue
        exists = _git("cat-file", "-e", f"{registration}^{{commit}}")
        if exists.returncode != 0:
            defects.append(f"decision {decision} missing registration commit")
            continue
        registered_text = _git("show", f"{registration}:docs/decisions.md")
        if registered_text.returncode != 0 or f"## {decision} " not in registered_text.stdout:
            defects.append(f"decision {decision} absent from registration commit")
        registration_artifact = record.get("registration_artifact")
        if registration_artifact != "docs/decisions.md":
            defects.append(f"decision {decision} registration artifact")
        first_registration = _git(
            "log",
            "--all",
            "--reverse",
            "-S",
            f"## {decision} —",
            "--format=%H",
            "--",
            "docs/decisions.md",
        )
        first_registration_commit = (
            first_registration.stdout.splitlines()[0]
            if first_registration.stdout.splitlines()
            else None
        )
        if first_registration.returncode != 0 or first_registration_commit != registration:
            defects.append(f"decision {decision} registration provenance")

        if temporal_class == "independently_timestamped_pending_outcome":
            if outcome is not None or eligible is not False:
                defects.append(f"decision {decision} pending classification")
            continue
        if not isinstance(outcome, str) or re.fullmatch(r"[0-9a-f]{40}", outcome) is None:
            defects.append(f"decision {decision} outcome commit")
            continue
        outcome_artifact = record.get("outcome_artifact")
        if not isinstance(outcome_artifact, str):
            defects.append(f"decision {decision} outcome artifact")
            continue
        first_add = _git(
            "log",
            "--all",
            "--reverse",
            "--diff-filter=A",
            "--format=%H",
            "--",
            outcome_artifact,
        )
        first_commit = first_add.stdout.splitlines()[0] if first_add.stdout.splitlines() else None
        if first_add.returncode != 0 or first_commit != outcome:
            defects.append(f"decision {decision} outcome provenance")

        if temporal_class == "independently_timestamped_prior_commit":
            ancestor = _git("merge-base", "--is-ancestor", registration, outcome)
            if registration == outcome or ancestor.returncode != 0 or eligible is not True:
                defects.append(f"decision {decision} prior-commit classification")
        elif temporal_class == "prospective_in_session_not_independently_timestamped":
            if registration != outcome or eligible is not False:
                defects.append(f"decision {decision} same-commit classification")
        else:
            defects.append(f"decision {decision} temporal class")
    return defects


@pytest.mark.timeout(1800)
@pytest.mark.publication_context
def test_registration_provenance_is_bound_to_git_history() -> None:
    payload = json.loads(REGISTRATION_PROVENANCE.read_text(encoding="utf-8"))
    assert _registration_provenance_defects(payload) == []

    # NEGATIVE CONTROL: a same-commit plan cannot be promoted by editing prose.
    planted = json.loads(json.dumps(payload))
    planted["protocols"]["51"]["confirmatory_status_eligible"] = True
    assert "decision 51 same-commit classification" in _registration_provenance_defects(planted)


def test_same_commit_protocols_are_not_titled_preregistrations() -> None:
    """Same-commit intent is prospective, but its Git chronology is not prior."""

    payload = json.loads(REGISTRATION_PROVENANCE.read_text(encoding="utf-8"))["protocols"]
    decisions = (PROJECT_ROOT / "docs" / "decisions.md").read_text(encoding="utf-8")
    for decision in ("18", "46", "51", "73", "74", "77", "78"):
        assert payload[decision]["temporal_class"] == (
            "prospective_in_session_not_independently_timestamped"
        )
        heading = next(
            line for line in decisions.splitlines() if line.startswith(f"## {decision} —")
        )
        assert "PROSPECTIVE DEVELOPMENT-SESSION SPECIFICATION" in heading
        assert "PRE-REGISTRATION" not in heading

    # NEGATIVE CONTROL: restoring the historical title is detected by the same rule.
    planted = decisions.replace(
        "## 73 — PROSPECTIVE DEVELOPMENT-SESSION SPECIFICATION:",
        "## 73 — PRE-REGISTRATION:",
    )
    planted_heading = next(line for line in planted.splitlines() if line.startswith("## 73 —"))
    assert "PRE-REGISTRATION" in planted_heading


# ── The vocabulary itself ────────────────────────────────────────────────────


def test_the_status_vocabulary_is_exactly_the_audit_taxonomy():
    """The four words are load-bearing: docs and the assignment script use them
    literally, so a renamed member would silently orphan both."""
    assert {"confirmatory", "exploratory", "descriptive", "analytic"} == CLAIM_STATUSES


# ── Writer behavior ──────────────────────────────────────────────────────────


class TestStatusWrites:
    def test_record_stores_and_persists_a_status(self, tmp_path):
        path = tmp_path / "claims.json"
        ClaimsLedger(path).record("floor", 2.375, "run", "source", status="analytic")
        assert ClaimsLedger(path).get("floor")["status"] == "analytic"

    def test_record_without_status_records_no_status(self, claims_ledger_tmp):
        """Absent means unclassified — never a guessed default."""
        claims_ledger_tmp.record("bare", 1, "run", "source")
        assert "status" not in claims_ledger_tmp.get("bare")

    def test_record_many_applies_one_status_to_the_whole_batch(self, claims_ledger_tmp):
        claims_ledger_tmp.record_many(
            {"a": 1, "b": 2}, run_id="run", source="source", status="descriptive"
        )
        assert claims_ledger_tmp.get("a")["status"] == "descriptive"
        assert claims_ledger_tmp.get("b")["status"] == "descriptive"

    def test_replace_claims_batches_carry_their_own_status(self, claims_ledger_tmp):
        claims_ledger_tmp.replace_claims(
            batches=[
                ClaimBatch({"measured": 1}, "run_a", "source", "descriptive"),
                ClaimBatch({"registered": 2}, "run_b", "source", status="confirmatory"),
                ClaimBatch({"untagged": 3}, "run_c", "source"),
            ]
        )
        assert claims_ledger_tmp.get("measured")["status"] == "descriptive"
        assert claims_ledger_tmp.get("registered")["status"] == "confirmatory"
        assert "status" not in claims_ledger_tmp.get("untagged")

    def test_an_invalid_status_is_rejected_before_anything_is_written(self, tmp_path):
        path = tmp_path / "claims.json"
        ledger = ClaimsLedger(path)
        ledger.record("existing", 1, "r", "s")
        before = path.read_bytes()

        with pytest.raises(ValueError, match="status must be one of"):
            ledger.record("bad", 1, "r", "s", status="speculative")

        assert path.read_bytes() == before
        assert list(tmp_path.glob(".claims.json.*.tmp")) == []

    def test_an_invalid_status_on_disk_fails_closed(self, tmp_path):
        """A junk status in the file is a corrupt entry, not a soft note."""
        path = tmp_path / "claims.json"
        path.write_text(
            json.dumps(
                {
                    "_schema": "ailm-claims-v2",
                    "claims": {
                        "broken": {"value": 1, "run_id": "r", "source": "s", "status": "vibes"}
                    },
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="status must be one of"):
            ClaimsLedger(path)


class TestStatusInheritance:
    """A re-record that names no status keeps the stored one.

    The status classifies the measurement's *design* — pre-registered,
    sweep-selected, descriptive, closed-form — which does not change when the
    same recorder re-runs. Without inheritance every replay of a recorder that
    does not pass ``status=`` would strip the field, and the ledger-wide gate
    below would flip red on exactly the runs that prove the ledger stable.
    """

    def test_a_statusless_rerecord_keeps_the_status(self, claims_ledger_tmp):
        claims_ledger_tmp.record("kept", 1.0, "r1", "s", status="exploratory")
        claims_ledger_tmp.record("kept", 2.0, "r2", "s")
        entry = claims_ledger_tmp.get("kept")
        assert entry == {"value": 2.0, "run_id": "r2", "source": "s", "status": "exploratory"}

    def test_an_explicit_status_overrides_the_stored_one(self, claims_ledger_tmp):
        claims_ledger_tmp.record("promoted", 1.0, "r1", "s", status="descriptive")
        claims_ledger_tmp.record("promoted", 1.0, "r2", "s", status="confirmatory")
        assert claims_ledger_tmp.get("promoted")["status"] == "confirmatory"

    def test_replace_claims_inherits_across_a_retire_and_rerecord(self, claims_ledger_tmp):
        """A label in both halves is the same arm re-measured, not a new claim."""
        claims_ledger_tmp.record("refreshed", 1.0, "r1", "s", status="analytic")
        claims_ledger_tmp.replace_claims(
            remove=["refreshed"],
            batches=[ClaimBatch({"refreshed": 2.0}, "r2", "s")],
        )
        assert claims_ledger_tmp.get("refreshed")["status"] == "analytic"

    def test_inheritance_survives_a_reload_from_disk(self, tmp_path):
        path = tmp_path / "claims.json"
        ClaimsLedger(path).record("durable", 1.0, "r1", "s", status="descriptive")
        ClaimsLedger(path).record("durable", 2.0, "r2", "s")
        assert ClaimsLedger(path).get("durable")["status"] == "descriptive"


class TestAssignStatus:
    def test_assigns_without_touching_provenance(self, claims_ledger_tmp):
        claims_ledger_tmp.record("classified", 7.5, "run_x", "recorder.py")
        claims_ledger_tmp.assign_status({"classified": "descriptive"})
        assert claims_ledger_tmp.get("classified") == {
            "value": 7.5,
            "run_id": "run_x",
            "source": "recorder.py",
            "status": "descriptive",
        }

    def test_result_survives_a_reload_from_disk(self, tmp_path):
        path = tmp_path / "claims.json"
        ledger = ClaimsLedger(path)
        ledger.record("persisted", 1, "r", "s")
        ledger.assign_status({"persisted": "analytic"})
        assert ClaimsLedger(path).get("persisted")["status"] == "analytic"

    @pytest.mark.parametrize(
        ("statuses", "reason"),
        [
            ({}, "non-empty mapping"),
            ({"absent": "analytic"}, "unrecorded claims"),
            ({"present": "speculative"}, "must be one of"),
        ],
    )
    def test_a_rejected_call_leaves_the_file_byte_identical(self, tmp_path, statuses, reason):
        path = tmp_path / "claims.json"
        ledger = ClaimsLedger(path)
        ledger.record("present", 1, "r", "s")
        before = path.read_bytes()

        with pytest.raises(ValueError, match=reason):
            ledger.assign_status(statuses)

        assert path.read_bytes() == before
        assert "status" not in ledger.get("present")


# ── Unsupported schema refusal ────────────────────────────────────────────────


class TestUnsupportedSchema:
    def test_a_v1_ledger_is_refused_without_rewriting_it(self, tmp_path):
        path = tmp_path / "claims.json"
        path.write_text(
            json.dumps(
                {
                    "_schema": "ailm-claims-v1",
                    "claims": {"kept": {"value": 3, "run_id": "r", "source": "s"}},
                }
            ),
            encoding="utf-8",
        )
        before = path.read_bytes()

        with pytest.raises(ValueError, match="expected 'ailm-claims-v2'"):
            ClaimsLedger(path)

        assert path.read_bytes() == before


# ── The ledger-wide gate ─────────────────────────────────────────────────────


@_SKIP_NO_LEDGER
def test_every_claim_carries_a_valid_status():
    """THE gate: no claim in the real ledger may be unclassified.

    Recorders may legitimately write without a status (the field is optional so
    concurrent writers never break), which is exactly why this has to be
    enforced here rather than in the writer. Fix by classifying, never by
    deleting: ``.venv/bin/python scripts/assign_claim_status.py``.
    """
    claims = _claims()
    assert len(claims) > 20, "a near-empty ledger would make this gate vacuous"
    unclassified = sorted(label for label, entry in claims.items() if "status" not in entry)
    assert unclassified == [], (
        f"{len(unclassified)} claim(s) carry no epistemic status: "
        f"{unclassified[:10]}{'...' if len(unclassified) > 10 else ''}. Run "
        f"`.venv/bin/python scripts/assign_claim_status.py` — its mapping is the "
        f"assignment rule, and its default is descriptive on purpose."
    )
    invalid = {
        label: entry["status"]
        for label, entry in claims.items()
        if entry.get("status") not in CLAIM_STATUSES
    }
    assert invalid == {}, f"claims with a status outside {sorted(CLAIM_STATUSES)}: {invalid}"


@_SKIP_NO_LEDGER
def test_the_classification_is_not_degenerate():
    """NON-VACUITY: all four statuses exist in the real ledger, and
    ``confirmatory`` stays a minority. A classifier that stamped one word on
    everything would satisfy the gate above while informing nobody, and a
    ledger drifting toward majority-confirmatory is the inflation this field
    exists to make visible."""
    claims = _claims()
    present = {entry["status"] for entry in claims.values() if "status" in entry}
    assert present == CLAIM_STATUSES, (
        f"expected all four statuses in the real ledger, found {sorted(present)}. "
        f"If a whole category was legitimately retired, update this test in the "
        f"same commit that retired it."
    )
    confirmatory = sum(1 for entry in claims.values() if entry.get("status") == "confirmatory")
    assert confirmatory < 0.2 * len(claims), (
        f"{confirmatory} of {len(claims)} claims are confirmatory. Pre-registered "
        f"arms are the exception by construction; this ratio says the label is "
        f"being handed out."
    )


@_SKIP_NO_LEDGER
def test_every_confirmatory_claim_traces_to_a_cited_preregistration():
    """A confirmatory entry must come from a source whose docstring cites the
    decisions.md entry that registered the prediction.

    This is the ledger-side half of the inflation guard: the source-scan below
    stops a recorder from *typing* the word without a citation, and this stops
    an entry from *carrying* it on the strength of a source that never earned
    it — including one assigned by a future edit to the migration mapping.
    """
    offenders: dict[str, str] = {}
    for label, entry in _claims().items():
        if entry.get("status") != "confirmatory":
            continue
        source_file = PROJECT_ROOT / entry["source"].split("::")[0]
        if not source_file.exists():
            offenders[label] = f"{entry['source']} (file missing)"
            continue
        docstring = ast.get_docstring(ast.parse(source_file.read_text(encoding="utf-8"))) or ""
        if not _CITATION.search(docstring):
            offenders[label] = entry["source"]
    assert offenders == {}, (
        f"confirmatory claims whose source cites no pre-registration decision "
        f"entry in its module docstring: {offenders}. Either the arm was "
        f"registered — then cite the entry — or it was not, and the status is "
        f"exploratory or descriptive, never confirmatory."
    )


# ── The source-scan guard, in the split-sequence-scan style ──────────────────


def _status_guard_scan_set() -> list[Path]:
    """Every module that writes claim statuses: the recorders plus the
    assignment script, which is itself a writer of the word."""
    recorders = sorted((PROJECT_ROOT / "scripts").glob("record_*_claims.py"))
    return [*recorders, ASSIGN_SCRIPT]


def _confirmatory_without_citation(paths: list[Path]) -> list[str]:
    """Modules containing the string "confirmatory" whose module docstring
    cites no decision entry. Grep for the trigger, AST for the docstring — a
    citation buried in a code comment is not the visible justification the
    guard is for."""
    offenders = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if "confirmatory" not in text:
            continue
        docstring = ast.get_docstring(ast.parse(text)) or ""
        if not _CITATION.search(docstring):
            offenders.append(path.name)
    return offenders


def test_no_module_records_confirmatory_without_citing_a_decision():
    """The word is the escalation: any module that can put "confirmatory" into
    the ledger must say, in its docstring, which pre-registration earned it."""
    offenders = _confirmatory_without_citation(_status_guard_scan_set())
    assert offenders == [], (
        f"{offenders} mention 'confirmatory' but their module docstrings cite no "
        f"docs/decisions.md entry (pattern: 'decision N' / 'entry N'). A "
        f"confirmatory status is a claim about pre-registration; cite the entry "
        f"or use exploratory/descriptive."
    )


def test_the_confirmatory_scan_set_is_not_vacuous():
    """POSITIVE CONTROL half 1: the scan must read real, sufficient sources,
    and at least one scanned module must contain the trigger word — otherwise
    the guard above is green because nothing is being examined."""
    paths = _status_guard_scan_set()
    assert len(paths) >= 15, f"only {len(paths)} modules scanned"
    assert all(path.is_file() for path in paths)
    assert any("confirmatory" in path.read_text(encoding="utf-8") for path in paths), (
        "no scanned module mentions 'confirmatory'; the guard is checking nothing"
    )


def test_the_confirmatory_guard_fires_on_a_planted_violation(tmp_path):
    """POSITIVE CONTROL half 2: the classifier itself, on planted modules."""
    cited = tmp_path / "cited.py"
    cited.write_text(
        '"""Pre-registered in docs/decisions.md entry 12."""\nSTATUS = "confirmatory"\n',
        encoding="utf-8",
    )
    uncited = tmp_path / "uncited.py"
    uncited.write_text('"""Records numbers."""\nSTATUS = "confirmatory"\n', encoding="utf-8")
    comment_only = tmp_path / "comment_only.py"
    comment_only.write_text(
        '"""Records numbers."""\n# decision 12\nSTATUS = "confirmatory"\n', encoding="utf-8"
    )
    silent = tmp_path / "silent.py"
    silent.write_text('"""Records numbers."""\nSTATUS = "descriptive"\n', encoding="utf-8")

    flagged = _confirmatory_without_citation([cited, uncited, comment_only, silent])
    assert flagged == ["uncited.py", "comment_only.py"], (
        f"the classifier flagged {flagged}; it must flag the uncited module and "
        f"the comment-buried citation, and neither the cited nor the silent one."
    )


# ── The assignment script's own contract ─────────────────────────────────────


def _assignment_module() -> ModuleType:
    """Import scripts/assign_claim_status.py by path, as the thing that runs."""
    spec = importlib.util.spec_from_file_location("_assign_claim_status", ASSIGN_SCRIPT)
    assert spec is not None and spec.loader is not None, f"cannot load {ASSIGN_SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_assignment_rule_defaults_downward_and_stays_in_vocabulary():
    """An unknown source is descriptive — the mapping can only *raise* a status
    by naming a recorder, never by omission — and every mapped status is real."""
    module = _assignment_module()
    assert set(module.STATUS_BY_SOURCE.values()) <= CLAIM_STATUSES
    assert module.status_for("anything", "scripts/some_future_recorder.py") == "descriptive"
    assert module.status_for("stage1_uniform_baseline", "scripts/record_stage1_claims.py") == (
        "analytic"
    )
    assert (
        module.status_for("stage2_g2p_famsplit_accuracy", "scripts/record_g2p_famsplit_claims.py")
        == "exploratory"
    )
    for protocol_label, source in (
        ("stage2_g2p_split_stride", "scripts/record_stage2_claims.py"),
        ("stage2_g2p_famsplit_prefix_length", "scripts/record_g2p_famsplit_claims.py"),
        ("stage2_g2p_famsplit_family_stride", "scripts/record_g2p_famsplit_claims.py"),
    ):
        assert module.status_for(protocol_label, source) == "descriptive"
    assert module.status_for("stage6_mean_delta", "scripts/record_stage6_claims.py") == (
        "descriptive"
    )
    assert (
        module.status_for(
            "bench_ctx_protocol_version", "scripts/record_incontext_benchmark_claims.py"
        )
        == "descriptive"
    )
    assert (
        module.status_for("generation_best_gamma", "scripts/record_generation_claims.py")
        == "exploratory"
    )
    assert (
        module.status_for("generation_sampled_baseline_tv", "scripts/record_generation_claims.py")
        == "exploratory"
    )
    assert (
        module.status_for(
            "generation_shipped_information_gain_nats", "scripts/record_generation_claims.py"
        )
        == "analytic"
    )
    # Decision 77 has no independently timestamped prior artifact.  Its
    # document-clustered effect and uncertainty are exploratory, while frozen
    # method details and population counts remain descriptive.
    for label in (
        "cluster_stage4_czech_effect",
        "cluster_stage4_czech_ci_low",
        "cluster_stage4_czech_ci_high",
        "cluster_stage4_czech_p",
        "cluster_stage4_czech_z",
        "cluster_stage4_czech_p_at_resolution_floor",
    ):
        assert module.status_for(label, "scripts/record_cluster_claims.py") == "exploratory"
    for label in (
        "cluster_stage4_czech_cluster_unit",
        "cluster_stage4_czech_observations",
        "cluster_stage4_czech_clusters",
        "cluster_stage4_czech_resamples",
        "cluster_stage4_czech_seed",
    ):
        assert module.status_for(label, "scripts/record_cluster_claims.py") == "descriptive"


#: The S7 repair reclassified the four direction-agreement markers as
#: descriptive in the authoritative writer (scripts/assign_claim_status.py)
#: while the production ledger still carries their original recorded status.
#: The final executor's regeneration -- never a hand edit -- closes that gap.
#: Until then this map documents the deliberate interim state: entries here
#: are exempt from the ledger-agreement assertion below and must still carry
#: exactly the pre-repair status, so anything else that drifts fails loudly.
PENDING_STATUS_REGENERATION = {
    "confirmatory_stage4_derived_mcnemar_direction_agrees": "confirmatory",
    "confirmatory_stage4_gold_mcnemar_direction_agrees": "confirmatory",
    "confirmatory_stage5_mcnemar_direction_agrees": "confirmatory",
    "confirmatory_stage6_sign_test_direction_agrees": "confirmatory",
}


def _assert_authoritative_statuses(claims: dict) -> None:
    """A stale status is as misleading as a missing one.

    The first migration filled only absent fields, so changing a mapping could
    leave an explicitly exploratory recorder tagged confirmatory forever. This
    binds every on-disk entry to the same rule the reconciliation script runs.
    """
    module = _assignment_module()
    mismatches = {
        label: (entry.get("status"), module.status_for(label, entry.get("source", "")))
        for label, entry in claims.items()
        if entry.get("status") != module.status_for(label, entry.get("source", ""))
    }
    unexpected = {
        label: statuses
        for label, statuses in mismatches.items()
        if label not in PENDING_STATUS_REGENERATION
    }
    assert not unexpected, (
        f"ledger statuses disagree with scripts/assign_claim_status.py: "
        f"{sorted(unexpected.items())}. Run "
        "`.venv/bin/python scripts/assign_claim_status.py`; do not hand-edit the ledger."
    )
    stale_pending = {
        label: mismatches[label]
        for label in set(mismatches) & set(PENDING_STATUS_REGENERATION)
        if mismatches[label][0] != PENDING_STATUS_REGENERATION[label]
    }
    assert not stale_pending, (
        f"these labels are marked pending regeneration but no longer carry the "
        f"pre-repair status: {stale_pending}. The pending map in this file is "
        "stale: regenerate, then remove the entries it describes."
    )


@_SKIP_NO_LEDGER
def test_every_ledger_status_matches_the_authoritative_assignment_rule():
    _assert_authoritative_statuses(_claims())


def test_bad_status_reports_the_claim_and_expected_assignment(tmp_path):
    module = _assignment_module()
    label = "bench_synth_openrouter_model"
    source = "scripts/record_synthetic_benchmark_claims.py"
    expected = module.status_for(label, source)
    assert expected == "descriptive"
    path = tmp_path / "claims.json"
    claims = {label: {"value": "served-model", "source": source, "status": expected}}
    path.write_text(json.dumps({"claims": claims}), encoding="utf-8")
    _assert_authoritative_statuses(json.loads(path.read_text(encoding="utf-8"))["claims"])
    claims[label]["status"] = "confirmatory"
    path.write_text(json.dumps({"claims": claims}), encoding="utf-8")
    with pytest.raises(AssertionError, match=label) as caught:
        _assert_authoritative_statuses(json.loads(path.read_text(encoding="utf-8"))["claims"])
    assert "confirmatory" in str(caught.value)
    assert "descriptive" in str(caught.value)
    assert "assign_claim_status.py" in str(caught.value)


def test_every_confirmatory_mapping_names_an_existing_cited_recorder():
    """The confirmatory sources must exist and carry their citations —
    checked against the files, not against the ledger, so the mapping cannot
    rot between migrations."""
    module = _assignment_module()
    confirmatory_sources = sorted(
        source for source, status in module.STATUS_BY_SOURCE.items() if status == "confirmatory"
    )
    # Two arms carry confirmatory status, and each is named here so adding a
    # third stays a deliberate act rather than a default. Decision 29 is the
    # latent-state G2P learning arm; decision 111 is the one-run replication of
    # the four ud_ewt_test members of the frozen family, whose registration
    # commit strictly predates its outcome commit.
    assert confirmatory_sources == [
        "scripts/record_confirmatory_replication.py",
        "scripts/record_g2p_learning_claims.py",
    ], (
        f"the confirmatory list changed ({confirmatory_sources}); adding an arm "
        f"is a deliberate act that needs its pre-registration entry cited here "
        f"and in the recorder."
    )
    for source in confirmatory_sources:
        path = PROJECT_ROOT / source
        assert path.is_file(), f"{source} is mapped confirmatory but does not exist"
        docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
        assert _CITATION.search(docstring), (
            f"{source} is mapped confirmatory but its docstring cites no decisions.md entry"
        )
    provenance = json.loads(REGISTRATION_PROVENANCE.read_text(encoding="utf-8"))["protocols"]
    assert {
        source: decision
        for source, decision in module.CONFIRMATORY_DECISION_BY_SOURCE.items()
        if provenance[decision]["confirmatory_status_eligible"] is True
    } == module.CONFIRMATORY_DECISION_BY_SOURCE
