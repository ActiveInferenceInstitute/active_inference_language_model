"""No claim label may be written by two producers from different measurements.

## The defect this catches

The ledger stores one value and one ``source`` per label. If two code paths write
the same label from *different* measurements, the stored value is decided by
whichever ran last. Nothing errors, both writers are individually correct, and
prose bound to that label silently changes meaning between runs.

That happened here twice, and the second time was self-inflicted:

1. ``scripts/record_stage2_claims.py`` wrote ``stage2_phoneme_perplexity``
   = 16.7445 (full dictionary, closed form) while
   ``tests/test_stage2_gate.py`` wrote 15.2882 (length-matched subset) to the same
   label. A full-suite run put the subset value on disk, hydration propagated it,
   and the rendered PDF asserted the wrong number. Every test passed.
2. The fix gave the test a corpus-explicit label — and the recorder also wrote
   *that* label, from a stride-13 sample rather than the full subset. Two different
   populations under one name is the same defect wearing a disguise.

## The principle

**Recorders write; tests verify.** A test that measures the same quantity over a
different population deserves its own label naming that population, and exactly one
producer may write any given label.

## Why this file compares values rather than grepping call sites

A static scan for ``.record(`` would have to resolve which labels each call site
writes, and the labels are built in dict literals and loops. Instead this runs the
recorders, snapshots the ledger, and checks that the values the tests would write
agree with what is stored. A disagreement is the defect itself, not a proxy for it.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from ailm._publication_inventory import (
    publication_source_bytes,
    validate_publication_execution_binding,
)
from ailm._publication_types import DerivativeError
from ailm.replay_provenance import NETWORK_CALL_ENTRY_POINTS, discover_network_recorders
from ailm.verification.connected_provenance import (
    ExecutionBindingError,
    canonical_execution_binding,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEDGER = PROJECT_ROOT / "data" / "claims.json"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def _discover_recorders() -> tuple[str, ...]:
    """Default-deny discovery, with the aggregate recorder running last."""
    discovered = sorted(
        str(path.relative_to(PROJECT_ROOT))
        for path in (PROJECT_ROOT / "scripts").glob("record_*_claims.py")
    )
    aggregate = "scripts/record_statistical_claims.py"
    if aggregate in discovered:
        discovered.remove(aggregate)
        discovered.append(aggregate)
    return tuple(discovered)


RECORDERS = _discover_recorders()

#: Every entry point in ``ailm.benchmarks`` that reaches the server. One
#: definition, used both to decide which recorders are excluded from replay and
#: to check that each exclusion is earned — two lists would be the same
#: judgement written twice, free to drift apart, and they did: the exclusion set
#: was keyed on the availability probe while the earned-check was keyed on two
#: names, and ``record_logprob_drift_claims.py``, which reaches Ollama through
#: ``post_generate_uncached`` and mentions neither, fell through both. The
#: predicate has to be "calls the server", not "asks whether the server is
#: there".
_SERVER_ENTRY_POINTS = tuple(sorted(NETWORK_CALL_ENTRY_POINTS))


def _offline_anchors(
    project_root: Path = PROJECT_ROOT, recorders: tuple[str, ...] = RECORDERS
) -> tuple[str, ...]:
    """Recorders that import from ``ailm.benchmarks`` but reach no server.

    Discovered, so the anchor set cannot go stale, and asserted non-empty,
    because an empty anchor set is a check that passes by having nothing to
    check.
    """
    found = []
    for script in recorders:
        tree = ast.parse((project_root / script).read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and (node.module == "ailm.benchmarks" or node.module.startswith("ailm.benchmarks."))
            for alias in node.names
        }
        if imported and not (imported & set(_SERVER_ENTRY_POINTS)):
            found.append(script)
    return tuple(found)


_OFFLINE_ANCHORS = _offline_anchors()


def _declares_publication_replay_configuration(script: str) -> bool:
    """Whether an excluded recorder implements the independent receipt API."""
    tree = ast.parse((PROJECT_ROOT / script).read_text(encoding="utf-8"))
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "publication_replay_configuration"
        for node in ast.walk(tree)
    )


#: Recorders the replay test cannot run, and why. Each queries a local Ollama
#: server for the registered probes: they are not hermetic (a clean checkout has
#: no server and no models), they are not fast (the external benchmark scores
#: ``DEFAULT_TASK_N`` positions per task per model, which is deliberately sized
#: for statistical resolution rather than for a test budget), and they are not
#: bit-reproducible in the way the replay assertion requires, since the value
#: depends on a server the repository does not pin.
#:
#: This is an exclusion from *replay*, not from verification. The recorded
#: values are still guarded: ``tests/test_benchmarks.py`` checks the protocol,
#: the statistics, and the ledger contents, and the recorders remain
#: single-writer-locked like every other. Shrinking the sample so the replay
#: fits a timeout was tried and is the wrong trade — it silently voided the
#: only two contrasts the benchmark can resolve while every assertion still
#: passed.
#:
#: Derived, not listed. As a hand-written set this named its instances, and the
#: failure that follows is not a red test: a new live-server recorder is simply
#: *not excluded*, so the replay test launches it as a subprocess against a real
#: Ollama server, and in an environment with no route to the server it fails on
#: a socket error that says nothing about the recorder.
NETWORK_RECORDERS = frozenset(discover_network_recorders(PROJECT_ROOT))
EXPECTED_NETWORK_RECORDERS = frozenset(
    {
        "scripts/record_aligned_g2p_probe_claims.py",
        "scripts/record_external_benchmark_claims.py",
        "scripts/record_incontext_benchmark_claims.py",
        "scripts/record_logprob_drift_claims.py",
        "scripts/record_synthetic_benchmark_claims.py",
    }
)

#: Labels the manuscript's central claims rest on. These must be produced by a
#: recorder, never by a test — a test is free to *verify* them.
#:
#: This list is a readable spot-check, not the boundary. Enumerating the labels
#: that must not be test-written is the wrong polarity: it caught 4 of the 14
#: labels that were in fact dual-written, because the other 10 were simply never
#: listed. The boundary is ``test_no_test_writes_a_label_it_does_not_own``
#: below, which is default-deny over the whole ledger.
CANONICAL_LABELS = (
    "stage1_letter_perplexity",
    "stage1_unigram_baseline",
    "stage1_bigram_floor_perplexity",
    "stage2_phoneme_perplexity",
    "stage2_phoneme_unigram_baseline",
    "stage2_phoneme_all_position_floor_perplexity",
    "stage2_phoneme_all_position_max_gain_nats",
    "stage2_g2p_accuracy",
    "stage2_g2p_baseline_accuracy",
    "stage2_g2p_mcnemar_p",
    "stage2_g2p_split_stride",
    "stage2_g2p_famsplit_prefix_length",
    "stage2_g2p_famsplit_family_stride",
    # Decision 83's inferential correction is part of the Stage 5 publication
    # contract and must be owned by the real recorder, never by a gate test.
    "stage5_oracle_gain_nats_per_sentence",
    "stage5_oracle_gain_sentence_ci_low",
    "stage5_oracle_gain_sentence_ci_high",
    "stage5_oracle_gain_nats_per_position",
    "stage5_oracle_gain_position_ci_low",
    "stage5_oracle_gain_position_ci_high",
    "stage5_oracle_gain_ci_confidence_pct",
    "stage5_oracle_gain_bootstrap_resamples",
    "stage5_oracle_gain_bootstrap_seed",
    # P1 core-inference lane (2026-08-31): recorder-owned pymdp
    # inference-equivalence evidence (scripts/record_pymdp_equivalence_claims.py).
    # Decision 94 adds the exhaustive L4 count and the L5 posterior tolerance,
    # which is the statistic that actually carries the L5 equivalence.
    "stage4_pymdp_argmax_disagreements",
    "stage5_pymdp_type_argmax_disagreements",
    "stage5_pymdp_posterior_max_abs_dev",
    "stage6_pymdp_live_vs_observed_delta_nats",
    # L3 → L2B lexical-channel labels (decisions 18-19). These rest the central
    # manuscript claims of ISC-41, so they must be produced by
    # scripts/record_lexical_channel_claims.py, never by a test. The high-alpha
    # row and every per-alpha McNemar row are included with the summary labels.
    "lexical_anchor_accuracy",
    "lexical_best_alpha",
    "lexical_best_accuracy",
    "lexical_best_margin_pts",
    "lexical_best_mcnemar_b",
    "lexical_best_mcnemar_c",
    "lexical_best_mcnemar_p",
    "lexical_high_alpha",
    "lexical_high_alpha_accuracy",
    "lexical_high_alpha_margin_pts",
    "lexical_sweep_alpha_0_0_mcnemar_b",
    "lexical_sweep_alpha_0_0_mcnemar_c",
    "lexical_sweep_alpha_0_0_mcnemar_p",
    "lexical_sweep_alpha_0_1_mcnemar_b",
    "lexical_sweep_alpha_0_1_mcnemar_c",
    "lexical_sweep_alpha_0_1_mcnemar_p",
    "lexical_sweep_alpha_0_2_mcnemar_b",
    "lexical_sweep_alpha_0_2_mcnemar_c",
    "lexical_sweep_alpha_0_2_mcnemar_p",
    "lexical_sweep_alpha_0_3_mcnemar_b",
    "lexical_sweep_alpha_0_3_mcnemar_c",
    "lexical_sweep_alpha_0_3_mcnemar_p",
    "lexical_sweep_alpha_0_5_mcnemar_b",
    "lexical_sweep_alpha_0_5_mcnemar_c",
    "lexical_sweep_alpha_0_5_mcnemar_p",
    "lexical_sweep_alpha_0_7_mcnemar_b",
    "lexical_sweep_alpha_0_7_mcnemar_c",
    "lexical_sweep_alpha_0_7_mcnemar_p",
    "lexical_sweep_alpha_0_9_mcnemar_b",
    "lexical_sweep_alpha_0_9_mcnemar_c",
    "lexical_sweep_alpha_0_9_mcnemar_p",
    # Clause -> word channel (decision entry 27). The manuscript's central
    # claim that the first clause of the novel-information criterion is
    # testable at the word level rests on these, so they must be produced
    # by scripts/record_clause_channel_claims.py, never by a test. The
    # per-alpha sweep rows and every headline summary label are included.
    "clause_word_anchor_ppl",
    "clause_word_anchor_ppl_oracle",
    "clause_word_best_alpha",
    "clause_word_best_degraded_tokens",
    "clause_word_best_delta_ppl",
    "clause_word_best_improved_tokens",
    "clause_word_best_pair_log10_p",
    "clause_word_best_ppl",
    "clause_word_confirms_harm",
    "clause_word_cluster_confidence_pct",
    "clause_word_cluster_documents",
    "clause_word_cluster_loggain_ci_high",
    "clause_word_cluster_loggain_ci_low",
    "clause_word_cluster_loggain_emissions",
    "clause_word_cluster_loggain_largest_document",
    "clause_word_cluster_loggain_nats_per_emission",
    "clause_word_cluster_loggain_negative_documents",
    "clause_word_cluster_loggain_p",
    "clause_word_cluster_loggain_p_at_resolution_floor",
    "clause_word_cluster_loggain_p_floor",
    "clause_word_cluster_loggain_positive_documents",
    "clause_word_cluster_resamples",
    "clause_word_cluster_seed",
    "clause_word_cluster_token_ci_high",
    "clause_word_cluster_token_ci_low",
    "clause_word_cluster_token_largest_document",
    "clause_word_cluster_token_negative_documents",
    "clause_word_cluster_token_net_share",
    "clause_word_cluster_token_observations",
    "clause_word_cluster_token_p",
    "clause_word_cluster_token_p_at_resolution_floor",
    "clause_word_cluster_token_p_floor",
    "clause_word_cluster_token_positive_documents",
    "clause_word_cluster_unit",
    "clause_word_document_overlap",
    "clause_word_eval_emissions",
    "clause_word_eval_tokens",
    "clause_word_high_alpha_ppl",
    "clause_word_maxdiff_declarative",
    "clause_word_maxdiff_interrogative",
    "clause_word_monotone_decline",
    "clause_word_optimum_is_interior",
    "clause_word_real_beats_swapped_type",
    "clause_word_real_beats_typefree",
    "clause_word_scrambled_ppl_at_best",
    "clause_word_swapped_type_ppl_at_best",
    "clause_word_source_categories",
    "clause_word_source_documents",
    "clause_word_sweep_alpha_0_0_degraded",
    "clause_word_sweep_alpha_0_0_delta",
    "clause_word_sweep_alpha_0_0_improved",
    "clause_word_sweep_alpha_0_0_pair_log10_p",
    "clause_word_sweep_alpha_0_0_ppl",
    "clause_word_sweep_alpha_0_0_scrambled_ppl",
    "clause_word_sweep_alpha_0_1_degraded",
    "clause_word_sweep_alpha_0_1_delta",
    "clause_word_sweep_alpha_0_1_improved",
    "clause_word_sweep_alpha_0_1_pair_log10_p",
    "clause_word_sweep_alpha_0_1_ppl",
    "clause_word_sweep_alpha_0_1_scrambled_ppl",
    "clause_word_sweep_alpha_0_2_degraded",
    "clause_word_sweep_alpha_0_2_delta",
    "clause_word_sweep_alpha_0_2_improved",
    "clause_word_sweep_alpha_0_2_pair_log10_p",
    "clause_word_sweep_alpha_0_2_ppl",
    "clause_word_sweep_alpha_0_2_scrambled_ppl",
    "clause_word_sweep_alpha_0_3_degraded",
    "clause_word_sweep_alpha_0_3_delta",
    "clause_word_sweep_alpha_0_3_improved",
    "clause_word_sweep_alpha_0_3_pair_log10_p",
    "clause_word_sweep_alpha_0_3_ppl",
    "clause_word_sweep_alpha_0_3_scrambled_ppl",
    "clause_word_sweep_alpha_0_5_degraded",
    "clause_word_sweep_alpha_0_5_delta",
    "clause_word_sweep_alpha_0_5_improved",
    "clause_word_sweep_alpha_0_5_pair_log10_p",
    "clause_word_sweep_alpha_0_5_ppl",
    "clause_word_sweep_alpha_0_5_scrambled_ppl",
    "clause_word_sweep_alpha_0_7_degraded",
    "clause_word_sweep_alpha_0_7_delta",
    "clause_word_sweep_alpha_0_7_improved",
    "clause_word_sweep_alpha_0_7_pair_log10_p",
    "clause_word_sweep_alpha_0_7_ppl",
    "clause_word_sweep_alpha_0_7_scrambled_ppl",
    "clause_word_sweep_alpha_0_9_degraded",
    "clause_word_sweep_alpha_0_9_delta",
    "clause_word_sweep_alpha_0_9_improved",
    "clause_word_sweep_alpha_0_9_pair_log10_p",
    "clause_word_sweep_alpha_0_9_ppl",
    "clause_word_sweep_alpha_0_9_scrambled_ppl",
    "clause_word_test_sentences",
    "clause_word_test_categories",
    "clause_word_test_documents",
    "clause_word_train_sentences",
    "clause_word_train_categories",
    "clause_word_train_documents",
    "clause_word_typefree_ppl_at_best",
    # Online learning on non-stationary text (decision entry 28). The
    # manuscript's claim that a level which keeps learning beats a frozen
    # control on a drifted stream rests on these, so they must be produced
    # by scripts/record_online_learning_claims.py, never by a test.
    "online_confirmed",
    "online_drift_tolerance",
    "online_frozen_vs_post_gap_ppl",
    "online_max_drift",
    "online_p1_passes",
    "online_p2_passes",
    "online_p3_passes",
    "online_post_hold_positions",
    "online_post_learn_positions",
    "online_ppl_frozen_post",
    "online_ppl_frozen_pre",
    "online_ppl_online_post",
    "online_ppl_refit_post",
    "online_pre_hold_positions",
    "online_pre_learn_positions",
    "online_pseudocount",
)

#: Ordinary tests are verify-only. Every project-ledger value has a recorder
#: that takes the single-writer lock; temporary-ledger unit tests exercise the
#: persistence API through ``tmp_path`` and never appear in this inventory.
TEST_OWNED_LABELS: frozenset[str] = frozenset()

_SKIP_NO_LEDGER = pytest.mark.skipif(
    not LEDGER.exists(), reason="data/claims.json absent; run the recorders"
)


def _claims() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8"))["claims"]


# ── Structure ────────────────────────────────────────────────────────────────


@_SKIP_NO_LEDGER
def test_every_entry_records_its_producer():
    """Without a source there is no way to detect a second writer at all."""
    missing = sorted(k for k, v in _claims().items() if not v.get("source"))
    assert missing == [], f"claims with no source: {missing}"


@_SKIP_NO_LEDGER
def test_ledger_is_not_empty():
    """Guard vacuity: an empty ledger would satisfy every check below."""
    assert len(_claims()) > 20


@_SKIP_NO_LEDGER
@pytest.mark.parametrize("label", CANONICAL_LABELS)
def test_canonical_labels_are_produced_by_a_recorder(label: str):
    """A canonical value last written by a test is the order-dependence defect.

    This is the assertion that would have caught the original conflict on the run
    that introduced it, rather than after it reached a rendered PDF.
    """
    claims = _claims()
    if label not in claims:
        pytest.skip(f"{label} not recorded in this checkout")
    source = claims[label]["source"]
    assert source.startswith("scripts/"), (
        f"{label} was last written by {source!r}. Canonical claim values must come "
        f"from a recorder script; tests must verify them, not overwrite them. If the "
        f"test measures a different population, give it a label naming that "
        f"population (see stage2_phoneme_subset_perplexity)."
    )


@_SKIP_NO_LEDGER
def test_no_test_writes_a_label_it_does_not_own():
    """THE boundary: default-deny. No shared-ledger label may be test-written.

    ``CANONICAL_LABELS`` above enumerates what must not be test-written, and an
    enumeration only constrains what someone remembered to enumerate. When this
    check was added, fourteen labels were being written by both
    ``scripts/record_stage2_claims.py`` and ``tests/test_stage2_gate.py``; the
    canonical list named four of them. The values agreed, so nothing was wrong
    on disk — which is precisely why the ordering dependence survived a green
    suite and a rendered PDF.

    Inverting the polarity removes the memory requirement: every entry sourced
    to a test is a failure. Measurement populations that differ from the
    headline get distinct labels, but still belong to an authoritative recorder.
    """
    offenders = {
        label: entry["source"]
        for label, entry in _claims().items()
        if entry.get("source", "").startswith("tests/")
    }
    assert offenders == {}, (
        f"These labels were last written by a test: {offenders}. Recorders take "
        f"the single-writer lock and write; ordinary tests only verify."
    )


@_SKIP_NO_LEDGER
def test_the_test_owned_allowlist_has_no_exemptions():
    """The default-deny boundary has no shared-ledger exception path."""
    assert TEST_OWNED_LABELS == frozenset()


def test_the_ownership_detector_fires_on_a_planted_violation():
    """POSITIVE CONTROL for the default-deny scan.

    The real scan passes on a healthy tree, so the comparison itself is exercised
    here against a planted test-written label. Without this, a bug that made
    ``offenders`` always empty would read as a clean ledger.
    """
    planted = {
        "former_exception": {"source": "tests/test_stage2_gate.py::x"},
        "rogue": {"source": "tests/test_rogue.py::y"},
        "fine": {"source": "scripts/record_stage1_claims.py"},
    }
    offenders = {
        label: entry["source"]
        for label, entry in planted.items()
        if entry.get("source", "").startswith("tests/")
    }
    assert offenders == {
        "former_exception": "tests/test_stage2_gate.py::x",
        "rogue": "tests/test_rogue.py::y",
    }


# ── The behavioral check: rerunning the recorders must not change anything ──


def _connected_report_bytes(report: dict) -> bytes:
    return (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _connected_replay_preimage(root: Path, ledger_bytes: bytes, report_bytes: bytes) -> dict:
    """Authenticate the original, then predict current provenance before any replay.

    Publication URLs change the raw package digest, not measured values. Only the
    temporary replay ledger gets those two predetermined provenance values. The
    complete current inventory and the original package blob are authenticated by
    the same strict bridge used by publication readers; no replay output is an
    input to this prediction.
    """
    declaration = json.loads((root / "data/publication_derivative.json").read_bytes())
    assert hashlib.sha256(ledger_bytes).hexdigest() == declaration["inputs"]["data/claims.json"], (
        "retained ledger differs from the frozen publication input"
    )
    ledger = json.loads(ledger_bytes)
    claims = ledger["claims"]
    report = json.loads(report_bytes)
    assert report["schema"] == "ailm-connected-gum-report-v1"
    assert _connected_report_bytes(report) == report_bytes, "noncanonical original report"
    assert (
        hashlib.sha256(report_bytes).hexdigest() == claims["connected_gum_report_sha256"]["value"]
    ), "original report differs from the retained ledger"
    recorded = report["execution_binding"]
    execution_digest = hashlib.sha256(_connected_report_bytes(recorded)).hexdigest()
    assert execution_digest == report["execution_sha256"]
    assert execution_digest == claims["connected_gum_execution_sha256"]["value"]
    assert recorded["src/ailm/levels/connected.py"] == report["runtime_sha256"]
    assert report["runtime_sha256"] == claims["connected_gum_runtime_sha256"]["value"]
    current = canonical_execution_binding(root)
    validate_publication_execution_binding(root, recorded, current)

    expected_report = deepcopy(report)
    expected_report["execution_binding"] = current
    expected_report["execution_sha256"] = hashlib.sha256(
        _connected_report_bytes(current)
    ).hexdigest()
    expected = deepcopy(ledger)
    for label, value in (
        ("connected_gum_execution_sha256", expected_report["execution_sha256"]),
        (
            "connected_gum_report_sha256",
            hashlib.sha256(_connected_report_bytes(expected_report)).hexdigest(),
        ),
    ):
        entry = expected["claims"][label]
        assert (entry["run_id"], entry["source"], entry["status"]) == (
            "connected_gum_unexamined_v1",
            "scripts/record_connected_gum_claims.py",
            "descriptive",
        )
        entry["value"] = value
    return expected


def _assert_replay_no_op(before: dict, after: dict) -> None:
    changed = {
        label: (before.get(label), after.get(label))
        for label in sorted(set(before) | set(after))
        if before.get(label) != after.get(label)
    }
    assert changed == {}, (
        "Re-running the recorders changed stored entries, so another producer had "
        f"overwritten them: {changed}. Fix the duplicate writer — do not simply "
        "re-run the recorder, which only papers over the ordering dependence."
    )


@pytest.fixture
def connected_replay_source(tmp_path: Path):
    """Real files for metadata controls, not a scored corpus or a model substitute."""
    report = json.loads((PROJECT_ROOT / "output/reports/connected_gum.json").read_bytes())
    ledger = json.loads(LEDGER.read_bytes())
    root = tmp_path / "metadata-source"
    for relative in report["execution_binding"]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"metadata control fixture\n")
    for relative in ("data/publication_derivative.json", "data/publication_sources.json"):
        shutil.copy2(PROJECT_ROOT / relative, root / relative)
    declaration = json.loads((root / "data/publication_derivative.json").read_bytes())
    original_package = publication_source_bytes(
        root, declaration["measurement_commit"], "pyproject.toml"
    )
    (root / "pyproject.toml").write_bytes(original_package)
    report["execution_binding"] = canonical_execution_binding(root)
    report["execution_sha256"] = hashlib.sha256(
        _connected_report_bytes(report["execution_binding"])
    ).hexdigest()
    report["runtime_sha256"] = report["execution_binding"]["src/ailm/levels/connected.py"]
    content = _connected_report_bytes(report)
    for suffix, value in (
        ("execution_sha256", report["execution_sha256"]),
        ("runtime_sha256", report["runtime_sha256"]),
        ("report_sha256", hashlib.sha256(content).hexdigest()),
    ):
        ledger["claims"]["connected_gum_" + suffix]["value"] = value
    declaration["inputs"]["data/claims.json"] = hashlib.sha256(
        json.dumps(ledger).encode()
    ).hexdigest()
    (root / "data/publication_derivative.json").write_text(json.dumps(declaration))
    (root / "pyproject.toml").write_bytes((PROJECT_ROOT / "pyproject.toml").read_bytes())
    return root, ledger, content, original_package


def test_connected_replay_preimage_preserves_original_metadata_identity(connected_replay_source):
    root, ledger, report, original_package = connected_replay_source
    (root / "pyproject.toml").write_bytes(original_package)
    assert _connected_replay_preimage(root, json.dumps(ledger).encode(), report) == ledger


def test_connected_replay_preimage_predicts_only_publication_hashes(connected_replay_source):
    root, ledger, report, _ = connected_replay_source
    before = deepcopy(ledger)
    expected = _connected_replay_preimage(root, json.dumps(ledger).encode(), report)
    changed = {
        label for label in ledger["claims"] if expected["claims"][label] != ledger["claims"][label]
    }
    assert changed == {"connected_gum_execution_sha256", "connected_gum_report_sha256"}
    for label in changed:
        assert {k: v for k, v in expected["claims"][label].items() if k != "value"} == {
            k: v for k, v in ledger["claims"][label].items() if k != "value"
        }
    assert ledger == before


def test_connected_replay_preimage_agrees_with_actual_recorder_serializer(connected_replay_source):
    root, ledger, report, _ = connected_replay_source
    expected = _connected_replay_preimage(root, json.dumps(ledger).encode(), report)
    code = """
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / 'scripts'))
import record_connected_gum_claims as recorder
root = Path(sys.argv[2])
report = json.loads(sys.stdin.buffer.read())
report['execution_binding'] = recorder.execution_binding(root)
report['execution_sha256'] = hashlib.sha256(recorder._json_bytes(report['execution_binding'])).hexdigest()
print(json.dumps({
    'execution': report['execution_sha256'],
    'report': hashlib.sha256(recorder._json_bytes(report)).hexdigest(),
}))
"""
    result = subprocess.run(
        [str(VENV_PYTHON), "-B", "-c", code, str(PROJECT_ROOT), str(root)],
        input=report,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode()
    actual = json.loads(result.stdout)
    assert expected["claims"]["connected_gum_execution_sha256"]["value"] == actual["execution"]
    assert expected["claims"]["connected_gum_report_sha256"]["value"] == actual["report"]


@pytest.mark.parametrize("mutation", ("non-url", "lock", "core", "model", "missing", "extra"))
def test_connected_replay_preimage_refuses_changed_inputs(connected_replay_source, mutation):
    root, ledger, report, _ = connected_replay_source
    paths = {
        "non-url": "pyproject.toml",
        "lock": "uv.lock",
        "core": "src/ailm/levels/connected.py",
        "model": "gnn/models/L1_letter.gnn.md",
        "missing": "src/ailm/levels/connected.py",
        "extra": "src/ailm/levels/extra_dependency.py",
    }
    path = root / paths[mutation]
    if mutation == "missing":
        path.rename(path.with_suffix(".retained"))
    elif mutation == "non-url":
        content = path.read_bytes()
        assert b"inferactively-pymdp==1.0.4" in content
        path.write_bytes(
            content.replace(b"inferactively-pymdp==1.0.4", b"inferactively-pymdp==1.0.3")
        )
    else:
        path.write_bytes(b"changed metadata control input\n")
    with pytest.raises((DerivativeError, ExecutionBindingError)):
        _connected_replay_preimage(root, json.dumps(ledger).encode(), report)


@pytest.mark.parametrize(
    "mutation", ("forged-report", "forged-hash", "numeric-report", "numeric-report-refreshed-hash")
)
def test_connected_replay_preimage_refuses_forged_original(connected_replay_source, mutation):
    root, ledger, report, _ = connected_replay_source
    if mutation == "forged-hash":
        ledger["claims"]["connected_gum_report_sha256"]["value"] = "0" * 64
    else:
        altered = json.loads(report)
        if mutation in ("numeric-report", "numeric-report-refreshed-hash"):
            altered["uncertainty"]["document_mean_gain"] += 1
        else:
            altered["scope"] = "forged scope"
        report = _connected_report_bytes(altered)
        if mutation == "numeric-report-refreshed-hash":
            ledger["claims"]["connected_gum_report_sha256"]["value"] = hashlib.sha256(
                report
            ).hexdigest()
    message = (
        "retained ledger differs"
        if mutation in ("forged-hash", "numeric-report-refreshed-hash")
        else "original report differs"
    )
    with pytest.raises(AssertionError, match=message):
        _connected_replay_preimage(root, json.dumps(ledger).encode(), report)


@pytest.mark.parametrize("suffix", ("execution_sha256", "report_sha256"))
def test_connected_replay_no_op_refuses_wrong_derived_hash(connected_replay_source, suffix):
    root, ledger, report, _ = connected_replay_source
    expected = _connected_replay_preimage(root, json.dumps(ledger).encode(), report)["claims"]
    after = deepcopy(expected)
    after["connected_gum_" + suffix]["value"] = "0" * 64
    with pytest.raises(AssertionError, match="Re-running the recorders changed"):
        _assert_replay_no_op(expected, after)


@pytest.mark.parametrize(
    "mutation", ("numerical-value", "run-id", "source", "status", "missing-label", "extra-label")
)
def test_connected_replay_no_op_preserves_all_claim_fields(connected_replay_source, mutation):
    root, ledger, report, _ = connected_replay_source
    expected = _connected_replay_preimage(root, json.dumps(ledger).encode(), report)["claims"]
    after = deepcopy(expected)
    label = "connected_gum_document_mean_gain"
    if mutation == "missing-label":
        del after[label]
    elif mutation == "extra-label":
        after["planted_extra_claim"] = deepcopy(after[label])
    elif mutation == "numerical-value":
        after[label]["value"] += 1
    else:
        field = {"run-id": "run_id", "source": "source", "status": "status"}[mutation]
        after[label][field] = "planted changed claim field"
    with pytest.raises(AssertionError, match="Re-running the recorders changed"):
        _assert_replay_no_op(expected, after)


@_SKIP_NO_LEDGER
@pytest.mark.skipif(not VENV_PYTHON.exists(), reason="project venv absent")
# The replay re-runs every hermetic recorder end to end, so its cost is the sum of
# the project's measurements rather than of its code. One hour was enough until the
# calibration-floor recorder joined the set; the run then exceeded the ceiling and
# reported as a failed test, which reads as "a recorded value moved" — the one thing
# this test exists to say. Two hours restores the margin. The alternatives were both
# worse: excluding a recorder would silently drop it from the only check that it is a
# no-op on replay, and shrinking a measurement to fit a clock is the trade this
# project has already refused twice. This test is marked ``recorder_replay`` and is
# excluded from the release gate's quick profile, so the cost lands on the full suite.
@pytest.mark.timeout(7200)
@pytest.mark.recorder_replay
def test_rerunning_the_recorders_is_a_no_op(tmp_path: Path):
    """The authenticated replay preimage must equal what the recorders produce.

    If a value changes, some other producer wrote it after the recorder did — which
    is exactly the conflict. Comparing values catches it even when two producers
    share a plausible-looking label.

    Slow (it re-measures Stage 2), so it is a single test rather than one per label.
    ``NETWORK_RECORDERS`` are skipped here and guarded elsewhere; see that
    constant for why a live-server recorder cannot be a replay target.
    The canonical ledger stays the original measurement; only the temporary
    preimage predicts its two URL-dependent provenance hashes before execution.
    """
    isolated_ledger = tmp_path / "claims.json"
    before_bytes = LEDGER.read_bytes()
    report_path = PROJECT_ROOT / "output/reports/connected_gum.json"
    report_before_bytes = report_path.read_bytes()
    preimage = _connected_replay_preimage(PROJECT_ROOT, before_bytes, report_before_bytes)
    isolated_ledger.write_text(json.dumps(preimage, indent=2) + "\n", encoding="utf-8")
    before = preimage["claims"]
    env = {
        **os.environ,
        "AILM_CLAIMS_PATH": str(isolated_ledger.resolve()),
    }
    for script in RECORDERS:
        if script in NETWORK_RECORDERS:
            continue
        path = PROJECT_ROOT / script
        if not path.exists():
            continue
        result = subprocess.run(
            [str(VENV_PYTHON), str(path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        assert result.returncode == 0, (
            f"{script} exited {result.returncode}. A recorder that refuses to run "
            f"cannot be the single source of truth.\n{result.stderr[-1500:]}"
        )
    after = json.loads(isolated_ledger.read_text(encoding="utf-8"))["claims"]

    _assert_replay_no_op(before, after)
    assert LEDGER.read_bytes() == before_bytes, (
        "Recorder replay touched the tracked ledger despite AILM_CLAIMS_PATH isolation."
    )
    assert report_path.read_bytes() == report_before_bytes, (
        "Recorder replay touched the original report"
    )


def test_the_network_exclusion_is_derived_and_not_vacuous():
    """The live-server exclusion must cover every live-server recorder.

    A lower bound rather than an exact set: adding a live-server recorder should
    extend this automatically, and that is the point. The upper-bound half
    matters just as much — a recorder that never touches a server must NOT be
    excluded, or the replay test would quietly stop checking it.
    """
    assert NETWORK_RECORDERS == frozenset(discover_network_recorders(PROJECT_ROOT))
    assert NETWORK_RECORDERS == EXPECTED_NETWORK_RECORDERS
    assert NETWORK_RECORDERS < set(RECORDERS), "every recorder was excluded from replay"
    # The upper-bound half, which is what stops the predicate from widening
    # until replay covers nothing. The anchors must be recorders that DO import
    # from ``ailm.benchmarks`` while importing no server entry point — a
    # recorder that imports nothing from that module is insensitive to the
    # predicate by construction, so anchoring on one pins nothing. That was the
    # first version of this check: with ``record_stage1_claims.py`` as the
    # anchor, appending a pure helper like ``subsample_positions`` to
    # ``_SERVER_ENTRY_POINTS`` silently dropped the hermetic
    # ``record_calibration_floor_claims.py`` out of replay while all four guards
    # here stayed green — and that recorder is precisely the slow one somebody
    # has an incentive to exclude.
    assert _OFFLINE_ANCHORS, (
        "no recorder imports from ailm.benchmarks without importing a server entry "
        "point, so the upper-bound half of this check has nothing to anchor on and "
        "a widened predicate would go undetected"
    )
    for offline in _OFFLINE_ANCHORS:
        assert offline not in NETWORK_RECORDERS, (
            f"{offline} computes offline but was excluded from replay; the "
            f"predicate has widened past the class it names"
        )


def test_hosted_import_boundary_keeps_pure_helpers_in_offline_replay(tmp_path):
    """Real source files distinguish hosted submodule imports from pure helpers."""
    sources = {
        "record_hosted_claims.py": "from ailm.benchmarks.openrouter import ask_openrouter as query\n",
        "record_public_hosted_claims.py": "from ailm.benchmarks import ask_openrouter\n",
        "record_mixed_claims.py": (
            "from ailm.benchmarks import subsample_positions\n"
            "from ailm.benchmarks.openrouter import ask_openrouter\n"
        ),
        "record_pure_claims.py": "from ailm.benchmarks import subsample_positions\n",
        "record_pure_submodule_claims.py": (
            '"""The phrase ask_openrouter here is documentation, not a server import."""\n'
            "from ailm.benchmarks.prompts import subsample_positions\n"
        ),
    }
    directory = tmp_path / "scripts"
    directory.mkdir()
    for name, content in sources.items():
        (directory / name).write_text(content)
    recorders = tuple("scripts/" + name for name in sources)
    offline = frozenset(_offline_anchors(tmp_path, recorders))
    assert offline == {
        "scripts/record_pure_claims.py",
        "scripts/record_pure_submodule_claims.py",
    }
    assert set(discover_network_recorders(tmp_path)) == set(recorders) - offline


def test_recorder_discovery_covers_every_recorder_script():
    expected = {
        str(path.relative_to(PROJECT_ROOT))
        for path in (PROJECT_ROOT / "scripts").glob("record_*_claims.py")
    }
    assert set(RECORDERS) == expected
    assert len(RECORDERS) >= 14


def test_every_recorder_takes_the_single_writer_lock():
    """A recorder that writes the ledger without the lock defeats the lock.

    ``src/ailm/ledger_lock.py`` exists because two overlapping recorders lose one
    of the two results silently. That guarantee is only as strong as its least
    disciplined writer: for most of this file's life four of twenty-one recorders
    took the lock and the other seventeen wrote the same ledger unguarded, so the
    mutual exclusion was decorative on any run that did not happen to involve one
    of the four.

    The rule is stated over every recorder that loads the ledger, so a new
    recorder is covered the moment it is added rather than when someone
    remembers to list it.
    """
    unguarded = []
    for script in RECORDERS:
        source = (PROJECT_ROOT / script).read_text(encoding="utf-8")
        if "load_ledger" not in source:
            continue
        if "single_writer_lock" not in source:
            unguarded.append(script)
    assert unguarded == [], (
        f"{unguarded} write the claims ledger without taking the single-writer "
        f"lock. Wrap the entrypoint: "
        f"`with single_writer_lock(ledger_lock_path(PROJECT_ROOT)):`. An "
        f"unguarded writer makes the lock decorative for every run it takes part "
        f"in."
    )


def test_the_lock_rule_is_not_vacuous():
    """POSITIVE CONTROL: the rule above passes trivially if nothing is scanned."""
    scanned = [
        script
        for script in RECORDERS
        if "load_ledger" in (PROJECT_ROOT / script).read_text(encoding="utf-8")
    ]
    assert len(scanned) >= 14, (
        f"only {len(scanned)} recorders load the ledger; the lock rule is checking almost nothing"
    )


def test_no_recorder_reintroduces_the_split_clear_and_write():
    """A retire and a record that are two writes leave a window; one write does not.

    ``ClaimsLedger.replace_claims`` exists because clearing with ``remove_labels``
    and then writing with ``record_many`` is two whole-file replacements: a crash
    between them commits the deletion without the replacement, and the ledger is
    then short by both the retired labels and the new ones while looking
    internally consistent. Every recorder has been converted.

    Decision 35 named "a second recorder adopting the split sequence" as the
    symptom to watch for. Watching is weaker than scanning, so this is the scan.
    It binds every recorder rather than a list, because the edit that adds a
    recorder to a list is the same edit that forgets why the list exists.
    """
    offenders = sorted(
        script
        for script in RECORDERS
        if "remove_labels(" in (PROJECT_ROOT / script).read_text(encoding="utf-8")
    )
    assert offenders == [], (
        f"{offenders} clear the ledger with remove_labels() before writing, which "
        f"reopens the crash window replace_claims() closes. Pass the retire set "
        f"and the batches to replace_claims() so the transition is one write."
    )


def test_the_split_sequence_scan_is_not_vacuous():
    """POSITIVE CONTROL: the scan must be reading real recorder source."""
    sources = [(PROJECT_ROOT / script).read_text(encoding="utf-8") for script in RECORDERS]
    assert len(sources) >= 14, f"only {len(sources)} recorders scanned"
    users = sum(1 for text in sources if "replace_claims(" in text)
    assert users >= 1, (
        "no recorder calls replace_claims(), so the rule above is satisfied by "
        "the method being unused rather than by the window being closed"
    )


def test_the_replay_exclusion_names_only_recorders_that_exist():
    """Every excluded recorder is a file on disk that the replay would have run.

    The previous form asserted ``NETWORK_RECORDERS - set(RECORDERS) == []``.
    ``NETWORK_RECORDERS`` is built by a comprehension over ``RECORDERS``, so that
    difference is empty for every possible predicate — the assertion was true by
    construction and survived a mutation that widened the predicate enough to
    drop a hermetic recorder out of replay. What is worth checking is that each
    excluded path resolves to a real, readable recorder, since the exclusion is
    consumed as a set of paths elsewhere.
    """
    assert NETWORK_RECORDERS, "the exclusion is empty; every recorder would be replayed"
    for script in sorted(NETWORK_RECORDERS):
        path = PROJECT_ROOT / script
        assert path.is_file(), f"{script} is excluded from replay but is not a file"
        assert path.read_text(encoding="utf-8").strip(), f"{script} is excluded but empty"


def test_every_excluded_recorder_implements_the_independent_receipt_contract():
    """A discovery result cannot become an unauthenticated replay escape hatch.

    Without this the list is a general-purpose escape hatch: any recorder that
    became inconvenient could be added to it and would stop being replayed, with
    the docstring above still claiming the replay is exhaustive.

    The signal is executable, not textual. Grepping the file for "ollama" or a
    port number matches a docstring, a comment, or a rejected alternative
    described in prose, so a hermetic-but-slow recorder that merely *mentions*
    the server would satisfy it — the escape hatch would still be open. What is
    checked instead is that the module imports the client entry points from
    ``ailm.benchmarks``, which a recorder that never contacts a server has no
    reason to do.

    This once carried its own two-name copy of that list while the exclusion set
    was built from a different one, so a recorder could be excluded by one
    judgement and rejected by the other. Both now read ``_SERVER_ENTRY_POINTS``,
    which makes this a non-vacuity check on that predicate rather than a second
    opinion about it — the real work is done by the tests below, which pin what
    must be in the set and what must stay out.
    """
    missing = [
        script
        for script in sorted(NETWORK_RECORDERS)
        if not _declares_publication_replay_configuration(script)
    ]
    assert missing == [], (
        f"{missing} are excluded from hermetic replay but do not declare the "
        "configuration required to authenticate their network claim receipts"
    )


def test_the_replay_still_covers_most_recorders():
    """POSITIVE CONTROL: an exclusion that swallowed the suite would pass above.

    Stated as a proportion, not as a literal count. ``len(RECORDERS) - 2`` was
    the previous form and it broke the moment a third live-server recorder was
    legitimately added — the same defect as the power anchor pinned to a
    family size, where a test fails on growth it was never about. What this is
    about is that replay stays the rule and the live-server exclusion stays the
    exception, and a share says that directly.
    """
    replayed = set(RECORDERS) - NETWORK_RECORDERS
    assert len(replayed) >= 0.8 * len(RECORDERS), (
        f"only {len(replayed)} of {len(RECORDERS)} recorders are replayed; the "
        f"exclusion has stopped being an exception"
    )


@_SKIP_NO_LEDGER
def test_the_conflict_detector_would_notice_a_planted_disagreement(tmp_path):
    """POSITIVE CONTROL for the comparison logic above.

    The real check is slow and passes when the tree is healthy, so the comparison
    itself is exercised here on a planted disagreement. Without this, a bug making
    the diff always empty would look like a clean ledger.
    """
    before = {"a": {"value": 1.0}, "b": {"value": 2.0}}
    after = {"a": {"value": 1.0}, "b": {"value": 99.0}}
    changed = {
        label: (before[label]["value"], after[label]["value"])
        for label in before
        if label in after and before[label]["value"] != after[label]["value"]
    }
    assert changed == {"b": (2.0, 99.0)}


def test_no_recorded_claim_carries_a_sentinel_value() -> None:
    """A sentinel is a request, never a measurement, and must not reach the ledger.

    ``LETTER_CENSUS`` is -1: a flag meaning "score the whole pool", resolved
    inside the measurement. Two recorder sites wrote the *requested* value where
    the *scored* count belonged. One was visible — the ledger said the letter arm
    scored -1 positions, and the manuscript would have printed it. The other was
    not: ``-1 + 400 + 400 = 799`` is a plausible denominator, so it produced no
    absurd number, just every rate computed against it inflated by the whole
    census. A sentinel that survives arithmetic is worse than one that does not.

    The first version of this guard asserted that no claim is negative. That was
    wrong, and instructively so: this project records signed deltas, a negative
    intracluster correlation, and lower interval bounds below zero, several of
    which are findings. A rule broad enough to catch the sentinel by its sign
    also deletes real results.

    So the rule is stated over the sentinels themselves, discovered from the
    module rather than listed here. Every quantity ``ailm.benchmarks`` declares
    is a count, so a negative module-level integer constant in it *is* a
    sentinel; no ledger value may equal one. That catches the next sentinel as
    well as this one, and names none of them.
    """
    import json

    from ailm import benchmarks

    sentinels = {
        name: value
        for name, value in vars(benchmarks).items()
        if name.isupper() and isinstance(value, int) and not isinstance(value, bool) and value < 0
    }
    assert sentinels, (
        "no sentinel constants found in ailm.benchmarks; either they were "
        "renamed or this scan is now vacuous"
    )

    ledger = json.loads((PROJECT_ROOT / "data" / "claims.json").read_text())["claims"]
    offenders = {
        label: entry["value"]
        for label, entry in ledger.items()
        for value in sentinels.values()
        if entry.get("value") == value and isinstance(entry.get("value"), int)
    }
    assert offenders == {}, (
        f"these claims hold a value equal to a declared sentinel "
        f"({sentinels}): {offenders}. A sentinel is what was asked for, not what "
        f"was measured; record the resolved quantity the measurement reports."
    )


def _comprehensions_feeding_remove(tree: ast.Module) -> list[ast.ListComp | ast.GeneratorExp]:
    """Every comprehension whose result reaches a ``remove=`` keyword argument.

    The value is followed directly, through wrapping calls such as ``sorted``
    or ``list``, through a name assigned anywhere in the module, and through a
    call to a module-level function via that function's ``return`` expressions.
    A comprehension that never reaches ``remove=`` is not a retirement and is
    not reported.
    """
    assignments: dict[str, list[ast.expr]] = {}
    functions: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments.setdefault(target.id, []).append(node.value)
        elif isinstance(node, ast.FunctionDef):
            functions[node.name] = node

    found: list[ast.ListComp | ast.GeneratorExp] = []
    visited: set[int] = set()

    def follow(expr: ast.expr) -> None:
        if id(expr) in visited:
            return
        visited.add(id(expr))
        if isinstance(expr, (ast.ListComp, ast.GeneratorExp)):
            found.append(expr)
        elif isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Name) and expr.func.id in functions:
                for node in ast.walk(functions[expr.func.id]):
                    if isinstance(node, ast.Return) and node.value is not None:
                        follow(node.value)
            for arg in expr.args:
                follow(arg)
        elif isinstance(expr, ast.Name):
            for value in assignments.get(expr.id, []):
                follow(value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "remove":
                    follow(keyword.value)
    return found


def _iterates_all_labels(comp: ast.ListComp | ast.GeneratorExp) -> bool:
    return any(
        isinstance(gen.iter, ast.Call)
        and isinstance(gen.iter.func, ast.Attribute)
        and gen.iter.func.attr == "all_labels"
        for gen in comp.generators
    )


def _startswith_prefixes(comp: ast.ListComp | ast.GeneratorExp) -> list[str]:
    """String-constant ``startswith`` prefixes anywhere inside the ``if`` clauses.

    ``ast.walk`` descends through ``and``/``or``/``not`` so a carve-out such as
    ``a.startswith("x") and not a.startswith("xy")`` reports both prefixes.
    """
    prefixes: list[str] = []
    for gen in comp.generators:
        for condition in gen.ifs:
            for node in ast.walk(condition):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "startswith"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    prefixes.append(node.args[0].value)
    return prefixes


def _prefix_scoped_removals(source_text: str) -> list[str]:
    """The prefixes of every retirement in ``source_text`` scoped by label prefix.

    A retirement is a comprehension over ``<ledger>.all_labels()`` that feeds a
    ``remove=`` keyword; it is prefix-scoped when any of its ``if`` clauses tests
    ``startswith`` against a string constant.
    """
    tree = ast.parse(source_text)
    return [
        prefix
        for comp in _comprehensions_feeding_remove(tree)
        if _iterates_all_labels(comp)
        for prefix in _startswith_prefixes(comp)
    ]


def test_no_recorder_retires_labels_by_prefix():
    """A recorder's ``remove=[...]`` may only name labels it itself owns.

    DEFECT (shipped 2026-08-21, caught by the replay no-op test):
    ``record_channel_factorial_claims.py`` removed every ``channel_*`` label,
    but the proper-loss labels (``channel_proper_loss_*``) belong to
    ``record_channel_predictive_loss_claims.py``. The prefix-wide removal
    deleted them wholesale; the loss recorder's replay then re-created them as
    brand-new entries with no prior status to inherit, silently unclassifying
    every one of them — and on a fresh ledger it would have silently changed
    their provenance instead.

    The first guard compared each prefix against the shipped ledger, which
    passes exactly until a foreign label appears under that prefix, and then
    fails only if the foreign recorder happened to run first. The rule is now
    structural: no retirement that iterates ``all_labels()`` may select by
    ``startswith`` at all. Ownership is by recorded ``source`` through
    ``ClaimsLedger.labels_owned_by``, and a shared prefix is a naming
    convention, not an ownership boundary.
    """
    offenders = {
        script: prefixes
        for script in RECORDERS
        if (
            prefixes := _prefix_scoped_removals((PROJECT_ROOT / script).read_text(encoding="utf-8"))
        )
    }
    assert offenders == {}, (
        f"These recorders retire labels by prefix: {offenders}. Retire "
        f"`[label for label in ledger.labels_owned_by(SOURCE) if label not in values]` "
        f"instead — a shared prefix is not an ownership boundary."
    )


def test_the_prefix_removal_detector_fires_on_each_planted_shape():
    """POSITIVE CONTROL: every way a prefix-scoped retirement has been written."""
    planted = {
        "bare call": (
            "ledger.replace_claims("
            '    remove=[label for label in ledger.all_labels() if label.startswith("x_")],'
            "    batches=[],"
            ")"
        ),
        "boolop with a carve-out": (
            "stale = [\n"
            "    label\n"
            "    for label in ledger.all_labels()\n"
            '    if label.startswith("a_") and not label.startswith("a_b_")\n'
            "]\n"
            "ledger.replace_claims(remove=stale, batches=[])\n"
        ),
        "sorted genexpr": (
            "ledger.replace_claims("
            '    remove=sorted(label for label in ledger.all_labels() if label.startswith("s_")),'
            "    batches=[],"
            ")"
        ),
        "function return": (
            "def stale_labels(ledger, produced):\n"
            "    return sorted(\n"
            "        label\n"
            "        for label in ledger.all_labels()\n"
            '        if label.startswith("f_") and label not in produced\n'
            "    )\n"
            "ledger.replace_claims(remove=stale_labels(ledger, values), batches=[])\n"
        ),
    }
    detected = {shape: _prefix_scoped_removals(snippet) for shape, snippet in planted.items()}
    assert detected == {
        "bare call": ["x_"],
        "boolop with a carve-out": ["a_", "a_b_"],
        "sorted genexpr": ["s_"],
        "function return": ["f_"],
    }


def test_the_prefix_removal_detector_accepts_source_scoped_retirement():
    """NEGATIVE CONTROL: the sanctioned form, and a prefix that never reaches ``remove=``."""
    sanctioned = (
        "stale = [label for label in ledger.labels_owned_by(SOURCE) if label not in values]\n"
        "ledger.replace_claims(remove=stale, batches=[])\n"
    )
    assert _prefix_scoped_removals(sanctioned) == []
    unrelated = (
        'owned = [label for label in ledger.all_labels() if label.startswith("bench_")]\n'
        "print(len(owned))\n"
    )
    assert _prefix_scoped_removals(unrelated) == []
