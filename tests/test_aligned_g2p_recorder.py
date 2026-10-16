"""Contracts for the P2 comparator-arm recorder (protocol §3.4).

The recorder's whole value is that its gates are live arithmetic on the
frozen artifact, not decoration: a control that clears the native bar must
stop the run before the ledger is touched, and the happy path must write
exactly the recorder's own namespace.  Every test here drives the real
functions — real engine fits on real pair sets, real ledger writes into a
tmp ledger through ``AILM_CLAIMS_PATH`` — with planted fixtures only where a
gate's boundary needs a comparator the shipped artifact cannot produce.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from ailm.benchmarks.aligned_g2p import (
    ALIGNED_G2P_ARTIFACT_NAME,
    artifact_pairs,
    broken_correspondence,
    build_artifact_payload,
    build_splits,
    load_aligned_g2p,
    stipulated_correspondence,
)
from ailm.benchmarks.comparators import _ailm_g2p_metrics
from ailm.levels.phoneme import PHONES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = PROJECT_ROOT / "vendor" / ALIGNED_G2P_ARTIFACT_NAME
EMPTY_LEDGER = {"_schema": "ailm-claims-v2", "claims": {}}

#: The recorder's complete label inventory.  A label recorded outside this
#: set is a namespace leak; a missing one is a silent contract change.
RECORD_LABELS = frozenset(
    {
        "bench_synthg2p_comparator_native_position_accuracy",
        "bench_synthg2p_comparator_native_exact_accuracy",
        "bench_synthg2p_comparator_t_lookup_position_accuracy",
        "bench_synthg2p_comparator_t_lookup_exact_accuracy",
        "bench_synthg2p_comparator_context_free_position_accuracy",
        "bench_synthg2p_comparator_context_free_exact_accuracy",
        "bench_synthg2p_comparator_uniform_position_floor",
        "bench_synthg2p_comparator_uniform_exact_floor",
        "bench_synthg2p_comparator_a1_english_position_accuracy",
        "bench_synthg2p_comparator_a1_english_exact_accuracy",
        "bench_synthg2p_comparator_a2_broken_position_accuracy",
        "bench_synthg2p_comparator_a2_broken_exact_accuracy",
        "bench_synthg2p_comparator_native_ties_t_lookup",
        "bench_synthg2p_comparator_test_word_count",
        "bench_synthg2p_comparator_test_position_count",
        "bench_synthg2p_comparator_artifact_sha256",
    }
)
FLOOR_LABELS = frozenset(
    {
        "bench_synthg2p_comparator_uniform_position_floor",
        "bench_synthg2p_comparator_uniform_exact_floor",
    }
)

#: The deterministic A3 fixture: a minimal valid artifact (the generator's
#: own functions, a fixture-sized population recorded in its own provenance)
#: whose native arm decodes no test word exactly right.  Its A1/A2 controls
#: sit strictly below the native bar, so A3 is the gate that fires.
FIXTURE_CORRESPONDENCE_SEED = 20260912
FIXTURE_TRAIN_SEED_BASE = 900000
FIXTURE_N_TRAIN = 1
FIXTURE_N_TEST = 12


def _recorder():
    return importlib.import_module("scripts.record_aligned_g2p_comparator_claims")


def _fixture_vendor_dir(tmp_path: Path) -> Path:
    """Write a minimal valid artifact assembled by the shipped builder."""

    correspondence = stipulated_correspondence(seed=FIXTURE_CORRESPONDENCE_SEED)
    train, test, stats = build_splits(
        correspondence,
        n_train=FIXTURE_N_TRAIN,
        n_test=FIXTURE_N_TEST,
        train_seed_base=FIXTURE_TRAIN_SEED_BASE,
        held_out_seed_base=FIXTURE_TRAIN_SEED_BASE + FIXTURE_N_TRAIN,
    )
    payload = build_artifact_payload(
        correspondence,
        train,
        test,
        stats,
        train_seed_base=FIXTURE_TRAIN_SEED_BASE,
        held_out_seed_base=FIXTURE_TRAIN_SEED_BASE + FIXTURE_N_TRAIN,
    )
    vendor_dir = tmp_path / "vendor"
    vendor_dir.mkdir()
    (vendor_dir / ALIGNED_G2P_ARTIFACT_NAME).write_text(json.dumps(payload), encoding="utf-8")
    return vendor_dir


def test_refusals_pass_on_the_frozen_artifact() -> None:
    """POSITIVE CONTROL: the pre-registered refusals hold on the executed arms."""

    recorder = _recorder()
    materials = recorder._gate_materials()
    recorder._comparator_refusals(**{key: materials[key] for key in recorder.GATE_KEYS})


def test_a1_refusal_fires_on_a_planted_transfer_comparator_that_clears_the_bar() -> None:
    """A transfer fit that meets or clears the native bar refuses publication.

    The planted values are boundary arithmetic on the real native bar (from a
    real engine fit on the frozen artifact's train split, quantized at the
    real position grid): what an English CMUdict fit would look like if
    English statistics coincided with the stipulated correspondence.  The
    English control's actual measured value on this artifact is pinned by the
    happy-path test.
    """

    recorder = _recorder()
    _correspondence, train, test = artifact_pairs(load_aligned_g2p())
    bar, _bar_exact, _predictions = _ailm_g2p_metrics(train, test)
    step = 1 / sum(len(gold) for _word, gold in test)
    floors = {
        "position_floor": 1.0 / len(PHONES),
        "exact_floor": (1.0 / len(PHONES)) ** len(test[0][0]),
    }
    # A fit that merely meets the bar is already the fired condition.
    with pytest.raises(SystemExit, match="A1"):
        recorder._comparator_refusals(
            native_position=bar,
            native_exact=1.0,
            english_position=bar,
            broken_position=0.0,
            **floors,
        )
    # So is one that clears it by a quantization step.
    with pytest.raises(SystemExit, match="A1"):
        recorder._comparator_refusals(
            native_position=bar,
            native_exact=1.0,
            english_position=min(1.0, bar + step),
            broken_position=0.0,
            **floors,
        )
    # Boundary: one quantization step below the bar is not a refusal.
    recorder._comparator_refusals(
        native_position=bar,
        native_exact=1.0,
        english_position=bar - step,
        broken_position=0.0,
        **floors,
    )


def test_a2_refusal_fires_for_a_broken_map_comparator_that_beats_the_bar() -> None:
    """A broken-correspondence fit that clears the native bar refuses too.

    The planted value is a real engine fit on ``sigma . T`` pairs evaluated
    where the broken map generates the evaluation population — the shape the
    A2 control exists to catch when it happens against ``T`` gold.
    """

    recorder = _recorder()
    correspondence, train, test = artifact_pairs(load_aligned_g2p())
    bar, _bar_exact, _predictions = _ailm_g2p_metrics(train, test)
    step = 1 / sum(len(gold) for _word, gold in test)
    broken = broken_correspondence(correspondence)
    broken_train = [(word, [broken[letter] for letter in word]) for word, _ in train]
    broken_test = [(word, [broken[letter] for letter in word]) for word, _ in test]
    broken_position, _exact, _predictions = _ailm_g2p_metrics(broken_train, broken_test)
    assert broken_position >= bar, "the planted broken-map fixture must clear the bar"
    with pytest.raises(SystemExit, match="A2"):
        recorder._comparator_refusals(
            native_position=bar,
            native_exact=1.0,
            english_position=0.0,
            broken_position=broken_position,
            position_floor=1.0 / len(PHONES),
            exact_floor=(1.0 / len(PHONES)) ** len(test[0][0]),
        )
    # Boundary: strictly below the bar passes.
    recorder._comparator_refusals(
        native_position=bar,
        native_exact=1.0,
        english_position=0.0,
        broken_position=bar - step,
        position_floor=1.0 / len(PHONES),
        exact_floor=(1.0 / len(PHONES)) ** len(test[0][0]),
    )


def test_a3_refusal_fires_for_a_native_arm_at_the_uniform_floor() -> None:
    """A native arm performing at the uniform predictor's level refuses."""

    recorder = _recorder()
    position_floor = 1.0 / len(PHONES)
    exact_floor = position_floor**6
    with pytest.raises(SystemExit, match="A3"):
        recorder._comparator_refusals(
            native_position=position_floor,
            native_exact=1.0,
            english_position=0.0,
            broken_position=0.0,
            position_floor=position_floor,
            exact_floor=exact_floor,
        )
    # The exact-word leg refuses on its own: zero whole-word matches does not
    # beat the analytic uniform floor.
    with pytest.raises(SystemExit, match="A3"):
        recorder._comparator_refusals(
            native_position=position_floor + 0.01,
            native_exact=0.0,
            english_position=0.0,
            broken_position=0.0,
            position_floor=position_floor,
            exact_floor=exact_floor,
        )
    # Boundary: one quantization step above both floors passes.
    recorder._comparator_refusals(
        native_position=position_floor + 1 / 360,
        native_exact=1 / 60,
        english_position=0.0,
        broken_position=0.0,
        position_floor=position_floor,
        exact_floor=exact_floor,
    )


def test_refused_run_writes_no_ledger_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fired gate stops the run before the ledger is touched."""

    recorder = _recorder()
    ledger_path = tmp_path / "claims.json"
    ledger_path.write_text(json.dumps(EMPTY_LEDGER), encoding="utf-8")
    before = ledger_path.read_text(encoding="utf-8")
    monkeypatch.setenv("AILM_CLAIMS_PATH", str(ledger_path))
    with pytest.raises(SystemExit, match="A3"):
        recorder.main(["--vendor-dir", str(_fixture_vendor_dir(tmp_path))])
    assert ledger_path.read_text(encoding="utf-8") == before


def test_happy_path_records_the_expected_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full run records exactly the recorder's namespace into the tmp ledger."""

    recorder = _recorder()
    ledger_path = tmp_path / "claims.json"
    ledger_path.write_text(json.dumps(EMPTY_LEDGER), encoding="utf-8")
    monkeypatch.setenv("AILM_CLAIMS_PATH", str(ledger_path))
    assert recorder.main([]) == 0

    written = json.loads(ledger_path.read_text(encoding="utf-8"))["claims"]
    assert set(written) == RECORD_LABELS
    for entry in written.values():
        assert entry["source"] == recorder.SOURCE
        assert entry["run_id"] == recorder.RUN_ID
    written = {label: entry["value"] for label, entry in written.items()}

    # Closed-form and identity claims, pinned by construction rather than by
    # measurement: the T-lookup is exactly right on T-generated data, the
    # floors are the canonical-inventory arithmetic, the digest binds the
    # claims to the artifact bytes, and the counts mirror the artifact's own
    # provenance fields.
    assert written["bench_synthg2p_comparator_t_lookup_position_accuracy"] == 1.0
    assert written["bench_synthg2p_comparator_t_lookup_exact_accuracy"] == 1.0
    assert written["bench_synthg2p_comparator_uniform_position_floor"] == pytest.approx(
        1.0 / len(PHONES)
    )
    assert written["bench_synthg2p_comparator_uniform_exact_floor"] == pytest.approx(
        (1.0 / len(PHONES)) ** 6, rel=1e-6
    )
    assert (
        written["bench_synthg2p_comparator_artifact_sha256"]
        == hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    )
    payload = load_aligned_g2p()
    assert (
        written["bench_synthg2p_comparator_test_word_count"] == payload["provenance"]["test_words"]
    )
    assert (
        written["bench_synthg2p_comparator_test_position_count"]
        == payload["provenance"]["test_words"] * payload["provenance"]["word_length_L"]
    )
    # The shipped engine cannot beat the exact map on T-generated data, and
    # the gates' outcome is part of the recorded contract.
    assert (
        written["bench_synthg2p_comparator_native_position_accuracy"]
        <= written["bench_synthg2p_comparator_t_lookup_position_accuracy"]
    )
    assert (
        written["bench_synthg2p_comparator_native_position_accuracy"]
        > written["bench_synthg2p_comparator_uniform_position_floor"]
    )
    assert (
        written["bench_synthg2p_comparator_a1_english_position_accuracy"]
        < written["bench_synthg2p_comparator_native_position_accuracy"]
    )
    assert (
        written["bench_synthg2p_comparator_a2_broken_position_accuracy"]
        < written["bench_synthg2p_comparator_native_position_accuracy"]
    )


def test_claim_status_assignment_maps_the_recorder_to_exploratory() -> None:
    """The status rule knows the recorder, and its analytic floor exception."""

    recorder = _recorder()
    assign = importlib.import_module("scripts.assign_claim_status")
    assert assign.STATUS_BY_SOURCE[recorder.SOURCE] == "exploratory"
    for label in RECORD_LABELS - FLOOR_LABELS:
        assert assign.status_for(label, recorder.SOURCE) == "exploratory"
    for label in FLOOR_LABELS:
        assert assign.status_for(label, recorder.SOURCE) == "analytic"
