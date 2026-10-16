"""Real-file and rendered controls for the connected schedule and paired result."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ailm.figures._common import (
    ExportFigure,
    effective_on_page_dpi,
    on_page_point_size,
    save_figure,
)
from ailm.figures.connected import (
    channel_edges,
    connected_core_data,
    generate_connected_core_figure,
)
from ailm.levels.connected_evaluation import document_uncertainty
from ailm.levels.connected_types import Channel
from ailm.required_claims import CONNECTED_GUM_REQUIRED_CLAIMS
from ailm.value_formatting import ManuscriptVariableError

ROOT = Path(__file__).resolve().parents[1]


def _write_evidence(directory: Path, direction: str = "positive") -> tuple[Path, Path]:
    """Transform existing measured pairs only for explicitly synthetic render controls."""
    report = json.loads((ROOT / "output/reports/connected_gum.json").read_text())
    source = json.loads((ROOT / "data/claims.json").read_text())
    claims = {key: source["claims"][key] for key in CONNECTED_GUM_REQUIRED_CLAIMS}
    for row in report["documents"]:
        if direction == "harmful":
            row["integrated"], row["disconnected"] = row["disconnected"], row["integrated"]
        elif direction == "null":
            row["disconnected"] = copy.deepcopy(row["integrated"])
    integrated = [row["integrated"]["letter_log_loss"] for row in report["documents"]]
    disconnected = [row["disconnected"]["letter_log_loss"] for row in report["documents"]]
    uncertainty = document_uncertainty(np.subtract(disconnected, integrated))
    report["uncertainty"] = asdict(uncertainty)
    for suffix, value in {
        "document_integrated_log_loss": float(np.mean(integrated)),
        "document_disconnected_log_loss": float(np.mean(disconnected)),
        "document_mean_gain": uncertainty.document_mean_gain,
        "document_ci_low": uncertainty.ci_low,
        "document_ci_high": uncertainty.ci_high,
        "outcome": uncertainty.outcome,
    }.items():
        claims["connected_gum_" + suffix]["value"] = value
    ledger = {"_schema": "ailm-claims-v2", "claims": claims}
    paths = directory / "claims.json", directory / "report.json"
    _save_pair(paths, ledger, report)
    return paths


def _save_pair(paths: tuple[Path, Path], ledger: dict, report: dict) -> None:
    paths[1].write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    ledger["claims"]["connected_gum_report_sha256"]["value"] = hashlib.sha256(
        paths[1].read_bytes()
    ).hexdigest()
    paths[0].write_text(json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + "\n")


@pytest.mark.parametrize(
    "direction,outcome",
    [
        ("positive", "lower integrated loss"),
        ("null", "an interval spanning zero"),
        ("harmful", "higher integrated loss"),
    ],
)
def test_real_measured_pairs_and_transformed_controls_render_without_direction_assumption(
    tmp_path, direction, outcome
):
    from tests.test_figure_layout import assert_layout_is_clean

    paths = _write_evidence(tmp_path, direction)
    data = connected_core_data(*paths)
    assert data.outcome == outcome
    figure = generate_connected_core_figure(*paths)
    try:
        np.testing.assert_allclose(figure.get_size_inches(), (6.4, 7.0))
        assert_layout_is_clean(figure, "connected_core")
        first = save_figure(figure, "connected_core", tmp_path / "first")
        second = save_figure(figure, "connected_core", tmp_path / "second")
        assert first.read_bytes() == second.read_bytes()
        assert first.stat().st_size > 0
        arrows = {
            patch.get_gid()
            for patch in figure.axes[0].patches
            if (patch.get_gid() or "").startswith("channel:")
        }
        assert arrows == {"channel:" + channel.value for channel in Channel}
        labels = "\n".join(text.get_text() for text in figure.texts)
        assert "retained in both arms" in labels
        assert "outputs can differ" in labels
        assert outcome in labels
    finally:
        plt.close(figure)


def test_live_channel_enum_is_the_complete_directed_inventory():
    assert tuple(edge[0] for edge in channel_edges()) == tuple(Channel)
    assert all(
        channel.value == sender + "_" + receiver for channel, sender, receiver in channel_edges()
    )


def test_current_claims_and_frozen_report_reconcile_exactly():
    values = connected_core_data()
    claims = json.loads((ROOT / "data/claims.json").read_text())["claims"]
    for attribute, suffix in (
        ("gain", "document_mean_gain"),
        ("low", "document_ci_low"),
        ("high", "document_ci_high"),
        ("documents", "documents"),
    ):
        assert getattr(values, attribute) == claims["connected_gum_" + suffix]["value"]


def test_connected_export_preserves_layout_and_meets_print_resolution(tmp_path):
    manuscript = tmp_path / "docs/manuscript"
    manuscript.mkdir(parents=True)
    (manuscript / "preamble.md").write_bytes((ROOT / "docs/manuscript/preamble.md").read_bytes())
    (manuscript / "config.yaml").write_bytes((ROOT / "docs/manuscript/config.yaml").read_bytes())
    (manuscript / "04_results.md").write_text(
        "![Connected](../../output/figures/connected_core.png){#fig:connected_core width=100%}\n"
    )
    figure = generate_connected_core_figure()
    try:
        assert isinstance(figure, ExportFigure)
        assert figure.export_dpi == 400
        np.testing.assert_array_equal(figure.get_size_inches(), (6.4, 7.0))
        save_figure(figure, "connected_core", tmp_path / "output/figures")
        assert effective_on_page_dpi("connected_core", tmp_path) >= 300
        assert on_page_point_size("connected_core", 9.5, tmp_path) >= 8
    finally:
        plt.close(figure)


def test_every_missing_typed_claim_refuses_a_figure(tmp_path):
    paths = _write_evidence(tmp_path)
    full = json.loads(paths[0].read_text())
    for label in CONNECTED_GUM_REQUIRED_CLAIMS:
        missing = copy.deepcopy(full)
        del missing["claims"][label]
        paths[0].write_text(json.dumps(missing))
        with pytest.raises(ManuscriptVariableError, match="exact connected claim family"):
            generate_connected_core_figure(*paths)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_report",
        "digest",
        "gain",
        "mixed_run",
        "unpaired",
        "bootstrap",
        "source",
        "duplicate",
        "nonfinite",
        "omitted_binding",
        "extra_binding",
        "renamed_binding",
        "document_identity",
        "document_order",
    ],
)
def test_inconsistent_evidence_refuses_before_rendering(tmp_path, defect):
    paths = _write_evidence(tmp_path)
    ledger = json.loads(paths[0].read_text())
    report = json.loads(paths[1].read_text())
    if defect == "missing_report":
        paths[1].unlink()
    elif defect == "digest":
        paths[1].write_text(paths[1].read_text() + "\n")
    elif defect == "gain":
        ledger["claims"]["connected_gum_document_mean_gain"]["value"] += 1
        _save_pair(paths, ledger, report)
    elif defect == "mixed_run":
        ledger["claims"]["connected_gum_outcome"]["run_id"] = "other"
        _save_pair(paths, ledger, report)
    elif defect == "unpaired":
        report["documents"][0]["disconnected"]["normalized_tokens"] += 1
        _save_pair(paths, ledger, report)
    elif defect == "bootstrap":
        report["uncertainty"]["ci_low"] += 1
        _save_pair(paths, ledger, report)
    elif defect == "source":
        report["execution_binding"]["src/ailm/levels/connected.py"] = "0" * 64
        encoded = (
            json.dumps(report["execution_binding"], indent=2, sort_keys=True) + "\n"
        ).encode()
        report["execution_sha256"] = hashlib.sha256(encoded).hexdigest()
        ledger["claims"]["connected_gum_execution_sha256"]["value"] = report["execution_sha256"]
        _save_pair(paths, ledger, report)
    elif defect in {"omitted_binding", "extra_binding", "renamed_binding"}:
        if defect != "extra_binding":
            del report["execution_binding"]["src/ailm/corpus/connected_gum.py"]
        if defect != "omitted_binding":
            report["execution_binding"]["src/ailm/figures/connected.py"] = hashlib.sha256(
                (ROOT / "src/ailm/figures/connected.py").read_bytes()
            ).hexdigest()
        encoded = (
            json.dumps(report["execution_binding"], indent=2, sort_keys=True) + "\n"
        ).encode()
        report["execution_sha256"] = hashlib.sha256(encoded).hexdigest()
        ledger["claims"]["connected_gum_execution_sha256"]["value"] = report["execution_sha256"]
        _save_pair(paths, ledger, report)
    elif defect == "document_identity":
        report["documents"][0]["document_id"] = "not-in-frozen-population"
        _save_pair(paths, ledger, report)
    elif defect == "document_order":
        report["documents"].reverse()
        _save_pair(paths, ledger, report)
    elif defect == "nonfinite":
        ledger["claims"]["connected_gum_document_mean_gain"]["value"] = float("nan")
        paths[0].write_text(json.dumps(ledger))
    else:
        paths[1].write_text('{"schema":"duplicate",' + paths[1].read_text()[1:])
        ledger["claims"]["connected_gum_report_sha256"]["value"] = hashlib.sha256(
            paths[1].read_bytes()
        ).hexdigest()
        paths[0].write_text(json.dumps(ledger))
    before = plt.get_fignums()
    with pytest.raises((ManuscriptVariableError, FileNotFoundError)):
        generate_connected_core_figure(*paths)
    assert plt.get_fignums() == before
