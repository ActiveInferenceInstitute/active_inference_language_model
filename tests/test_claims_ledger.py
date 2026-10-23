"""Tests for ailm.claims — the claims ledger."""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import pytest

from ailm.claims import ClaimBatch, ClaimsLedger, load_ledger


class TestClaimsLedger:
    def test_empty_ledger_has_no_claims(self, claims_ledger_tmp):
        assert len(claims_ledger_tmp) == 0

    def test_record_and_retrieve(self, claims_ledger_tmp):
        claims_ledger_tmp.record("test_claim_42", 42.0, "run_001", "test_function")
        assert claims_ledger_tmp.verify_label("test_claim_42")
        entry = claims_ledger_tmp.get("test_claim_42")
        assert entry is not None
        assert entry["value"] == 42.0
        assert entry["run_id"] == "run_001"
        assert entry["source"] == "test_function"

    def test_missing_label_fails_verification(self, claims_ledger_tmp):
        assert not claims_ledger_tmp.verify_label("nonexistent_claim")

    def test_get_missing_returns_none(self, claims_ledger_tmp):
        assert claims_ledger_tmp.get("not_here") is None

    def test_persistence_across_loads(self, tmp_path):
        path = tmp_path / "claims.json"
        ledger1 = ClaimsLedger(path)
        ledger1.record("persisted_value", 99.9, "r1", "test_persistence")
        # Reload from disk
        ledger2 = ClaimsLedger(path)
        assert ledger2.verify_label("persisted_value")
        assert ledger2.get("persisted_value")["value"] == 99.9

    def test_overwrite_existing_claim(self, claims_ledger_tmp):
        claims_ledger_tmp.record("changing_claim", 1.0, "r1", "test")
        claims_ledger_tmp.record("changing_claim", 2.0, "r2", "test")
        assert claims_ledger_tmp.get("changing_claim")["value"] == 2.0

    def test_scan_prose_detects_unbacked_claim(self, tmp_path, claims_ledger_tmp):
        prose = tmp_path / "readme.md"
        prose.write_text("The phoneme count is claim:phoneme_count_39 canonical phonemes.")
        unbacked = claims_ledger_tmp.scan_prose(prose)
        assert "phoneme_count_39" in unbacked

    def test_scan_prose_clears_when_backed(self, tmp_path, claims_ledger_tmp):
        claims_ledger_tmp.record("phoneme_count_39", 39, "r1", "test_cmudict")
        prose = tmp_path / "readme.md"
        prose.write_text("claim:phoneme_count_39 phonemes verified.")
        unbacked = claims_ledger_tmp.scan_prose(prose)
        assert "phoneme_count_39" not in unbacked

    def test_scan_prose_multiple_files(self, tmp_path, claims_ledger_tmp):
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_text("claim:alpha_claim here")
        f2.write_text("claim:beta_claim there")
        unbacked = claims_ledger_tmp.scan_prose(f1, f2)
        assert "alpha_claim" in unbacked
        assert "beta_claim" in unbacked

    def test_scan_prose_ignores_non_claim_patterns(self, tmp_path, claims_ledger_tmp):
        prose = tmp_path / "doc.md"
        prose.write_text("This is a normal sentence with no claim tags.")
        unbacked = claims_ledger_tmp.scan_prose(prose)
        assert unbacked == []

    def test_all_labels_returns_list(self, claims_ledger_tmp):
        claims_ledger_tmp.record("a", 1, "r", "s")
        claims_ledger_tmp.record("b", 2, "r", "s")
        labels = claims_ledger_tmp.all_labels()
        assert set(labels) == {"a", "b"}

    def test_labels_owned_by_selects_only_that_sources_labels(self, claims_ledger_tmp):
        claims_ledger_tmp.record("alpha_metric", 1, "r", "scripts/record_alpha_claims.py")
        claims_ledger_tmp.record("beta_metric", 2, "r", "scripts/record_beta_claims.py")
        alpha = claims_ledger_tmp.labels_owned_by("scripts/record_alpha_claims.py")
        beta = claims_ledger_tmp.labels_owned_by("scripts/record_beta_claims.py")
        assert alpha == ["alpha_metric"]
        assert beta == ["beta_metric"]

    def test_labels_owned_by_matches_an_absolute_path_source(self, claims_ledger_tmp):
        """An entry recorded under an absolute path still belongs to its script."""
        absolute = "/home/someone/checkout/scripts/record_alpha_claims.py"
        claims_ledger_tmp.record("absolute_metric", 3, "r", absolute)
        claims_ledger_tmp.record("alpha_metric", 1, "r", "scripts/record_alpha_claims.py")
        claims_ledger_tmp.record("beta_metric", 2, "r", "scripts/record_beta_claims.py")
        owned = claims_ledger_tmp.labels_owned_by("scripts/record_alpha_claims.py")
        assert owned == ["absolute_metric", "alpha_metric"]

    def test_labels_owned_by_an_unknown_source_is_empty(self, claims_ledger_tmp):
        claims_ledger_tmp.record("alpha_metric", 1, "r", "scripts/record_alpha_claims.py")
        assert claims_ledger_tmp.labels_owned_by("scripts/record_nobody_claims.py") == []

    @pytest.mark.parametrize("source", ["", "   ", "/"])
    def test_labels_owned_by_rejects_a_source_without_a_file_name(self, claims_ledger_tmp, source):
        """A nameless owner would match nothing and read as a clean retirement."""
        with pytest.raises(ValueError):
            claims_ledger_tmp.labels_owned_by(source)

    def test_remove_labels_persists(self, claims_ledger_tmp):
        claims_ledger_tmp.record("keep", 1, "r", "s")
        claims_ledger_tmp.record("drop", 2, "r", "s")
        claims_ledger_tmp.remove_labels(["drop"])
        assert claims_ledger_tmp.all_labels() == ["keep"]

    def test_len_counts_claims(self, claims_ledger_tmp):
        assert len(claims_ledger_tmp) == 0
        claims_ledger_tmp.record("x", 1, "r", "s")
        assert len(claims_ledger_tmp) == 1
        claims_ledger_tmp.record("y", 2, "r", "s")
        assert len(claims_ledger_tmp) == 2

    def test_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "deep" / "path" / "claims.json"
        ledger = ClaimsLedger(nested)
        assert nested.exists()
        ledger.record("test", 1, "r", "s")
        ledger2 = ClaimsLedger(nested)
        assert ledger2.verify_label("test")

    def test_record_many_commits_one_complete_batch(self, claims_ledger_tmp):
        claims_ledger_tmp.record_many(
            {"first": 1, "second": {"nested": [2, 3]}},
            run_id="batch_run",
            source="tests/test_claims_ledger.py",
        )
        assert claims_ledger_tmp.get("first")["run_id"] == "batch_run"
        assert claims_ledger_tmp.get("second")["value"] == {"nested": [2, 3]}

    def test_invalid_batch_does_not_partially_commit(self, tmp_path):
        path = tmp_path / "claims.json"
        ledger = ClaimsLedger(path)
        ledger.record("existing", 1, "r", "s")
        before = path.read_bytes()

        with pytest.raises(ValueError, match="finite JSON"):
            ledger.record_many(
                {"valid": 2, "invalid": math.nan},
                run_id="bad_batch",
                source="tests/test_claims_ledger.py",
            )

        assert path.read_bytes() == before
        assert ledger.get("valid") is None
        assert list(tmp_path.glob(".claims.json.*.tmp")) == []

    @pytest.mark.parametrize(
        ("label", "run_id", "source"),
        [
            ("bad label", "run", "source"),
            ("good_label", "", "source"),
            ("good_label", "run", " "),
        ],
    )
    def test_labels_and_provenance_are_validated(self, claims_ledger_tmp, label, run_id, source):
        with pytest.raises(ValueError):
            claims_ledger_tmp.record(label, 1, run_id, source)

    def test_atomic_replace_preserves_the_existing_file_mode(self, tmp_path):
        path = tmp_path / "claims.json"
        ledger = ClaimsLedger(path)
        path.chmod(0o640)
        ledger.record("mode", 1, "run", "source")
        assert path.stat().st_mode & 0o777 == 0o640


def _seeded_ledger(path: Path) -> ClaimsLedger:
    """A ledger holding two retirable labels plus one a refresh must not disturb."""
    ledger = ClaimsLedger(path)
    ledger.record_many({"stale_one": 1, "stale_two": 2}, run_id="prior_run", source="prior.py")
    ledger.record_many({"untouched": 3}, run_id="other_run", source="other.py")
    return ledger


def _observe_atomic_replacements(path: Path) -> list[tuple[dict[str, Any], bytes]]:
    """Observe real ``os.replace`` commits through Python's process audit stream.

    The audit event fires immediately before the filesystem replacement, so it
    exposes both the complete temporary payload and the durable destination
    bytes without replacing the ledger's writer.
    """
    observed: list[tuple[dict[str, Any], bytes]] = []
    destination = path.resolve()

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event != "os.rename" or len(args) < 2:
            return
        source_raw, destination_raw = args[:2]
        if Path(destination_raw).resolve() != destination:
            return
        source = Path(source_raw)
        payload = json.loads(source.read_text(encoding="utf-8"))
        before = path.read_bytes() if path.exists() else b""
        observed.append((payload, before))

    sys.addaudithook(audit)
    return observed


#: Calls ``replace_claims`` must reject, paired with a fragment of the reason.
#: Each one is checked to leave the ledger file byte-identical: validation of
#: every argument has to complete before the single write, or a recorder that
#: passes one bad value in its second batch commits its first batch anyway.
_REJECTED_CALLS: list[Any] = [
    pytest.param(
        {"batches": [ClaimBatch({"ok": 1, "bad": math.nan}, "run", "source")]},
        "finite JSON",
        id="non_finite_value",
    ),
    pytest.param(
        {
            "remove": ["stale_one"],
            "batches": [
                ClaimBatch({"good": 1}, "run_a", "source"),
                ClaimBatch({"bad": math.inf}, "run_b", "source"),
            ],
        },
        "finite JSON",
        id="non_finite_value_in_second_batch",
    ),
    pytest.param(
        {"batches": [ClaimBatch({"bad label": 1}, "run", "source")]},
        "claim labels must match",
        id="unusable_label",
    ),
    pytest.param(
        {"batches": [ClaimBatch({"ok": 1}, "", "source")]},
        "run_id must be a non-empty string",
        id="blank_run_id",
    ),
    pytest.param(
        {"batches": [ClaimBatch({"ok": 1}, "run", "  ")]},
        "source must be a non-empty string",
        id="blank_source",
    ),
    pytest.param(
        {"batches": [ClaimBatch({}, "run", "source")]},
        "non-empty mapping",
        id="empty_batch",
    ),
    pytest.param(
        {
            "batches": [
                ClaimBatch({"shared": 1}, "run_a", "source"),
                ClaimBatch({"shared": 2}, "run_b", "source"),
            ]
        },
        "repeats labels",
        id="two_runs_claim_one_label",
    ),
    pytest.param(
        {"remove": "stale_one"},
        "remove must be a list or tuple",
        id="remove_is_a_bare_string",
    ),
    pytest.param(
        {"remove": ["stale_one", 7]},
        "claim labels must be strings",
        id="remove_holds_a_non_string",
    ),
    pytest.param(
        {"batches": "not_a_sequence_of_batches"},
        "batches must be a list or tuple",
        id="batches_is_a_bare_string",
    ),
    pytest.param(
        {"batches": [("values", "run", "source")]},
        "must be a ClaimBatch",
        id="batch_is_a_plain_tuple",
    ),
    pytest.param(
        {},
        "at least one label",
        id="call_that_would_do_nothing",
    ),
]


class TestReplaceClaims:
    """`replace_claims` — a recorder's retire-and-record as ONE whole-file write.

    The defect this guards: spelling a namespace refresh as `remove_labels`
    followed by `record_many` is two independent whole-file replacements. A
    crash between them commits the deletions without the replacements, and the
    result is a ledger short by exactly the retired *and* the new labels that
    is internally consistent, so nothing downstream can tell it is wrong.
    """

    def test_retires_records_and_leaves_the_rest_alone(self, tmp_path):
        """Happy path, and the two ways a merged write quietly gets it wrong:
        collapsing per-batch provenance onto one run id, and treating the
        refresh as a wholesale replacement that drops another recorder's labels.
        """
        ledger = _seeded_ledger(tmp_path / "claims.json")

        ledger.replace_claims(
            remove=["stale_one", "stale_two"],
            batches=[
                ClaimBatch({"fresh_a": 10, "fresh_b": {"nested": [1, 2]}}, "run_x", "recorder.py"),
                ClaimBatch({"fresh_c": 30}, "run_y", "other_recorder.py"),
            ],
        )

        assert set(ledger.all_labels()) == {"untouched", "fresh_a", "fresh_b", "fresh_c"}
        assert ledger.get("fresh_a") == {
            "value": 10,
            "run_id": "run_x",
            "source": "recorder.py",
        }
        assert ledger.get("fresh_b")["value"] == {"nested": [1, 2]}
        # Each batch carries its own provenance; they must not bleed together.
        assert ledger.get("fresh_c") == {
            "value": 30,
            "run_id": "run_y",
            "source": "other_recorder.py",
        }
        # A label named in neither half keeps its value AND its provenance.
        assert ledger.get("untouched") == {
            "value": 3,
            "run_id": "other_run",
            "source": "other.py",
        }

    def test_result_survives_a_reload_from_disk(self, tmp_path):
        """In-memory state is not the deliverable; the file the next run reads is."""
        path = tmp_path / "claims.json"
        _seeded_ledger(path).replace_claims(
            remove=["stale_one", "stale_two"],
            batches=[ClaimBatch({"fresh": 10}, "run_x", "recorder.py")],
        )

        reloaded = ClaimsLedger(path)
        assert set(reloaded.all_labels()) == {"untouched", "fresh"}
        assert reloaded.get("fresh")["run_id"] == "run_x"

    def test_a_label_in_both_halves_ends_up_recorded(self, tmp_path):
        """Removals apply first, so re-recording a retired label is not a delete."""
        ledger = _seeded_ledger(tmp_path / "claims.json")

        ledger.replace_claims(
            remove=["stale_one"],
            batches=[ClaimBatch({"stale_one": 99}, "run_x", "recorder.py")],
        )

        assert ledger.get("stale_one") == {
            "value": 99,
            "run_id": "run_x",
            "source": "recorder.py",
        }

    def test_removing_an_absent_label_is_not_an_error(self, tmp_path):
        """A recorder's namespace shrinks over time; the first refresh after a
        label is dropped from the code must not have to know it is already gone."""
        ledger = _seeded_ledger(tmp_path / "claims.json")

        ledger.replace_claims(
            remove=["never_recorded", "stale_one"],
            batches=[ClaimBatch({"fresh": 10}, "run_x", "recorder.py")],
        )

        assert set(ledger.all_labels()) == {"stale_two", "untouched", "fresh"}

    def test_the_removals_and_the_additions_reach_disk_in_the_same_write(self, tmp_path):
        """The property the method exists for: no observable half-applied state.

        `_write_atomic` is the only thing that changes the file, so the whole
        claim reduces to two facts about it — it is called exactly once, and the
        payload it is handed is already the complete post-run ledger. Snapshotting
        the file at call time proves nothing was committed ahead of it: a
        remove-then-record implementation would show a ledger here that has lost
        the retired labels and does not yet have the new ones.
        """
        path = tmp_path / "claims.json"
        ledger = _seeded_ledger(path)
        before = path.read_bytes()

        handed = _observe_atomic_replacements(path)

        ledger.replace_claims(
            remove=["stale_one", "stale_two"],
            batches=[ClaimBatch({"fresh": 10}, "run_x", "recorder.py")],
        )

        assert len(handed) == 1, "retire-and-record must not be two file replacements"
        payload, on_disk_when_called = handed[0]
        assert on_disk_when_called == before, "the file changed before the single write"
        assert set(payload["claims"]) == {"untouched", "fresh"}
        assert payload["_schema"] == "ailm-claims-v2"

    def test_positive_control_the_split_sequence_exposes_a_half_applied_ledger(self, tmp_path):
        """Positive control for the test above: the same probe catches the bug.

        `remove_labels` then `record_many` is the sequence `replace_claims`
        replaces. Driving it through the identical spy must show two writes and
        a real on-disk state that has lost the retired labels without having
        gained the new ones — the window a crash lands in. Without this, the
        one-write assertion could be measuring nothing.
        """
        path = tmp_path / "claims.json"
        ledger = _seeded_ledger(path)

        handed = _observe_atomic_replacements(path)

        ledger.remove_labels(["stale_one", "stale_two"])
        ledger.record_many({"fresh": 10}, run_id="run_x", source="recorder.py")

        assert len(handed) == 2, "the split sequence is supposed to be two writes"
        # The second replacement's pre-event bytes are the durable result of
        # the first replacement: the real half-applied crash window.
        between = json.loads(handed[1][1].decode("utf-8"))["claims"]
        assert set(between) == {"untouched"}, (
            "the durable state between the two writes is missing the retired "
            "labels and the new ones at once — exactly what replace_claims exists "
            "to make unobservable"
        )

    @pytest.mark.parametrize(("kwargs", "reason"), _REJECTED_CALLS)
    def test_a_rejected_call_leaves_the_file_byte_identical(self, tmp_path, kwargs, reason):
        """Every argument is validated before anything is written.

        The defect: validating batch N+1 only after committing batch N. A
        recorder that fails on one non-finite value would then leave the ledger
        holding its retirements and an arbitrary prefix of its new claims.
        """
        path = tmp_path / "claims.json"
        ledger = _seeded_ledger(path)
        before = path.read_bytes()

        with pytest.raises(ValueError, match=reason):
            ledger.replace_claims(**kwargs)

        # Bytes, not a re-read dict: a re-read would pass even if the method had
        # rewritten the file with equivalent content, which is a different (and
        # weaker) promise than never having touched it.
        assert path.read_bytes() == before
        assert set(ledger.all_labels()) == {"stale_one", "stale_two", "untouched"}
        assert list(tmp_path.glob(".claims.json.*.tmp")) == []

    def test_positive_control_a_committed_replace_changes_those_bytes(self, tmp_path):
        """Positive control for the test above.

        If `replace_claims` never wrote anything at all, every byte-identity
        assertion in this class would pass vacuously. This proves the same
        comparison detects a real commit.
        """
        path = tmp_path / "claims.json"
        ledger = _seeded_ledger(path)
        before = path.read_bytes()

        ledger.replace_claims(
            remove=["stale_one"],
            batches=[ClaimBatch({"fresh": 10}, "run_x", "recorder.py")],
        )

        assert path.read_bytes() != before

    def test_a_no_op_call_neither_truncates_nor_corrupts_the_file(self, tmp_path):
        """Nothing to remove and nothing to record fails closed.

        The failure mode worth naming: a writer that built an empty candidate
        and committed it, or that opened the file for writing before deciding
        it had no work, would leave a truncated or empty ledger behind.
        """
        path = tmp_path / "claims.json"
        ledger = _seeded_ledger(path)
        before = path.read_bytes()

        with pytest.raises(ValueError, match="at least one label"):
            ledger.replace_claims(remove=[], batches=[])

        assert path.read_bytes() == before
        assert path.stat().st_size > 0
        # The file is still a loadable ledger, not just unchanged bytes.
        assert set(ClaimsLedger(path).all_labels()) == {"stale_one", "stale_two", "untouched"}

    def test_no_op_call_on_an_empty_ledger_also_fails_closed(self, tmp_path):
        """Boundary: an empty ledger has no claims to make the guard look busy."""
        path = tmp_path / "claims.json"
        ledger = ClaimsLedger(path)
        before = path.read_bytes()

        with pytest.raises(ValueError, match="at least one label"):
            ledger.replace_claims()

        assert path.read_bytes() == before
        assert json.loads(path.read_text(encoding="utf-8"))["claims"] == {}

    def test_removal_only_and_record_only_calls_are_both_accepted(self, tmp_path):
        """Either half alone is real work; only both being empty is the no-op."""
        path = tmp_path / "claims.json"
        ledger = _seeded_ledger(path)

        ledger.replace_claims(remove=["stale_one"])
        assert set(ledger.all_labels()) == {"stale_two", "untouched"}

        ledger.replace_claims(batches=[ClaimBatch({"fresh": 10}, "run_x", "recorder.py")])
        assert set(ledger.all_labels()) == {"stale_two", "untouched", "fresh"}


class TestMalformedLedgerOnDisk:
    """Loading a file that exists but is not the current ledger shape.

    The distinction under test: a file missing only the schema *skeleton* has
    nothing to lose, so it is repaired. A file whose content cannot be
    interpreted might hold recorded claims, so it must raise rather than be
    silently replaced — otherwise a stale seed would quietly destroy the ledger.
    """

    def test_bare_empty_object_is_repaired(self, tmp_path):
        """A `{}` seed must not brick the ledger with a bare KeyError."""
        path = tmp_path / "claims.json"
        path.write_text("{}")

        ledger = ClaimsLedger(path)

        # Every accessor indexes _data["claims"]; all must work.
        assert len(ledger) == 0
        assert ledger.all_labels() == []
        assert ledger.get("anything") is None
        assert not ledger.verify_label("anything")

        # And recording still persists correctly.
        ledger.record("after_repair", 7.5, "r", "s")
        assert ClaimsLedger(path).get("after_repair")["value"] == 7.5

    def test_repair_preserves_existing_claims(self, tmp_path):
        """Backfilling the skeleton must not discard claims already present."""
        path = tmp_path / "claims.json"
        # Has "claims" with real data, but no _schema/_description.
        path.write_text(
            json.dumps({"claims": {"kept": {"value": 3, "run_id": "r", "source": "s"}}})
        )

        ledger = ClaimsLedger(path)

        assert ledger.get("kept")["value"] == 3
        assert ledger._data["_schema"] == "ailm-claims-v2"

    def test_invalid_json_raises_and_does_not_overwrite(self, tmp_path):
        """Unparseable JSON must be preserved on disk, not replaced."""
        path = tmp_path / "claims.json"
        original = '{"claims": {"half_written": '
        path.write_text(original)

        with pytest.raises(ValueError, match="not valid JSON"):
            ClaimsLedger(path)

        # The load-time failure must be non-destructive.
        assert path.read_text() == original

    def test_non_object_toplevel_raises(self, tmp_path):
        """Valid JSON that is not an object is not a ledger."""
        path = tmp_path / "claims.json"
        path.write_text("[1, 2, 3]")

        with pytest.raises(ValueError, match="must contain a JSON object"):
            ClaimsLedger(path)

        assert path.read_text() == "[1, 2, 3]"

    def test_claims_key_of_wrong_type_raises(self, tmp_path):
        """A 'claims' key that is not an object must not be silently reset."""
        path = tmp_path / "claims.json"
        path.write_text(json.dumps({"_schema": "ailm-claims-v2", "claims": ["not", "a", "dict"]}))

        with pytest.raises(ValueError, match="expected an object"):
            ClaimsLedger(path)

    def test_unknown_schema_fails_closed(self, tmp_path):
        path = tmp_path / "claims.json"
        path.write_text(json.dumps({"_schema": "future-v9", "claims": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="implicit migration"):
            ClaimsLedger(path)

    def test_malformed_claim_entry_fails_closed(self, tmp_path):
        path = tmp_path / "claims.json"
        path.write_text(
            json.dumps(
                {
                    "_schema": "ailm-claims-v2",
                    "claims": {"broken": {"value": 1, "run_id": "r"}},
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="missing fields"):
            ClaimsLedger(path)

    def test_repair_is_logged(self, tmp_path, caplog):
        """The skeleton backfill must be visible, not silent."""
        path = tmp_path / "claims.json"
        path.write_text("{}")

        with caplog.at_level(logging.INFO, logger="ailm.claims"):
            ClaimsLedger(path)

        assert any("Repaired ledger skeleton" in r.message for r in caplog.records)

    def test_wellformed_ledger_is_not_reported_as_repaired(self, tmp_path, caplog):
        """Negative control: a complete file must take no repair path."""
        path = tmp_path / "claims.json"
        ClaimsLedger(path).record("x", 1, "r", "s")  # writes a full schema

        caplog.clear()
        with caplog.at_level(logging.INFO, logger="ailm.claims"):
            ledger = ClaimsLedger(path)

        assert ledger.verify_label("x")
        assert not any("Repaired ledger skeleton" in r.message for r in caplog.records)


class TestLoadLedger:
    def test_load_ledger_creates_file(self, tmp_path):
        ledger = load_ledger(tmp_path)
        assert (tmp_path / "data" / "claims.json").exists()
        assert len(ledger) == 0

    def test_load_ledger_loads_existing(self, tmp_path):
        ledger1 = load_ledger(tmp_path)
        ledger1.record("k", 42, "r", "s")
        ledger2 = load_ledger(tmp_path)
        assert ledger2.verify_label("k")

    def test_absolute_environment_override_isolates_the_ledger(self, tmp_path, monkeypatch):
        project = tmp_path / "project"
        override = tmp_path / "isolated" / "claims.json"
        monkeypatch.setenv("AILM_CLAIMS_PATH", str(override))
        load_ledger(project).record("isolated", 1, "run", "source")
        assert ClaimsLedger(override).verify_label("isolated")
        assert not (project / "data" / "claims.json").exists()

    def test_relative_environment_override_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AILM_CLAIMS_PATH", "relative/claims.json")
        with pytest.raises(ValueError, match="must be absolute"):
            load_ledger(tmp_path)


def test_atomic_writer_never_leaves_temporary_files(tmp_path: Path):
    ledger = ClaimsLedger(tmp_path / "claims.json")
    for index in range(5):
        ledger.record(f"claim_{index}", index, "run", "source")
    assert [
        path
        for path in tmp_path.iterdir()
        if path.name.startswith(".claims.json.") and path.suffix == ".tmp"
    ] == []
