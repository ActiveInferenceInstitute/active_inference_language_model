"""The level gate-verdict summary: one row per built level, checked two ways.

``ailm.figures.level_gate_summary`` draws every built level's placement between its
own baseline and its own ceiling on one shared fractional axis. It computes nothing
a measurement — every baseline/model/ceiling/verdict is read from the committed
``output/reports/gate_audit.json`` — but it does compute the fraction and re-derive
the verdict, and it refuses to draw when either disagrees with what it read. No
mocks: every test reads the real committed report and the real
:data:`ailm.figures.architecture.LEVELS` table, and the negative controls write a
real tampered copy of the report to ``tmp_path`` rather than patching a mock.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure
from matplotlib.text import Text

from ailm.figures._common import MissingVariableError
from ailm.figures.architecture import LEVELS
from ailm.figures.level_gate_summary import (
    _gate_audit_report_path,
    build_level_gate_summary_figure,
    generate_level_gate_summary,
    level_gate_summary_data,
)


def _built_level_codes() -> set[str]:
    return {level.level for level in LEVELS if level.built}


def test_the_real_report_exists() -> None:
    """A guard on the guard: every test below is vacuous if this is absent."""
    assert _gate_audit_report_path().is_file()


def test_one_row_per_built_level(gate_audit_rows) -> None:
    """Every built level gets a row; the one unbuilt level (L7, speech) does not."""
    rows = level_gate_summary_data()
    levels = {row["level"] for row in rows}
    assert levels == _built_level_codes()
    assert "L7" not in levels


def test_every_row_names_an_audited_gate(gate_audit_rows) -> None:
    """Each row's gate id is one the committed report actually audited."""
    rows = level_gate_summary_data()
    audited = set(gate_audit_rows)
    for row in rows:
        assert row["gate_id"] in audited


def test_the_fraction_is_computed_from_the_row_s_own_numbers(gate_audit_rows) -> None:
    """No hidden rescaling: fraction is exactly (model - baseline) / (ceiling - baseline)."""
    rows = level_gate_summary_data()
    for row in rows:
        record = gate_audit_rows[row["gate_id"]]
        expected = (float(record["model"]) - float(record["baseline"])) / (
            float(record["ceiling"]) - float(record["baseline"])
        )
        assert row["fraction"] == pytest.approx(expected, rel=1e-12)


def test_stage4_and_stage6_exceed_their_own_ceiling(gate_audit_rows) -> None:
    """The two documented over-ceiling rows (decision-noted in the source) still are.

    ``ailm.gates.ComparatorType.REFERENCE_MODEL``'s docstring names Stage 4's
    shipped agent as scoring above its own smoothing reference; the committed
    report's Stage 6 row does the same against its fitted-on-test comparator. A
    fraction that stopped exceeding 1 here would mean the report changed under
    this figure without anyone updating the one sentence in ``ailm.gates`` that
    explains why exceeding a reference ceiling is possible at all.
    """
    rows = {row["level"]: row for row in level_gate_summary_data()}
    assert rows["L4"]["fraction"] > 1.0
    assert rows["L6"]["fraction"] > 1.0


def test_every_row_s_verdict_agrees_with_the_hierarchy_table(gate_audit_rows) -> None:
    """The cross-check the data function already performs, re-asserted from outside."""
    rows = {row["level"]: row for row in level_gate_summary_data()}
    for level in LEVELS:
        if level.level not in rows:
            continue
        expected = level.status.split("\n", 1)[0]
        assert rows[level.level]["verdict"] == expected


def test_missing_report_raises(tmp_path) -> None:
    with pytest.raises(MissingVariableError, match="Gate audit report not found"):
        level_gate_summary_data(tmp_path / "does_not_exist.json")


def test_a_report_that_is_not_a_json_array_raises(tmp_path) -> None:
    bad = tmp_path / "gate_audit.json"
    bad.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(MissingVariableError, match="JSON array"):
        level_gate_summary_data(bad)


def test_a_record_missing_a_required_field_raises(tmp_path) -> None:
    bad = tmp_path / "gate_audit.json"
    bad.write_text(json.dumps([{"gate_id": "stage1_letter", "baseline": 1.0}]), encoding="utf-8")
    with pytest.raises(MissingVariableError, match="missing field"):
        level_gate_summary_data(bad)


def _tamper(rows: list[dict], gate_id: str, **overrides) -> list[dict]:
    """A deep-enough copy of the real report with one record's fields overridden."""
    out = []
    for record in rows:
        record = dict(record)
        if record["gate_id"] == gate_id:
            record.update(overrides)
        out.append(record)
    return out


def _real_rows() -> list[dict]:
    return json.loads(_gate_audit_report_path().read_text(encoding="utf-8"))


def test_an_internally_inconsistent_report_refuses_to_draw(tmp_path) -> None:
    """POSITIVE CONTROL — a verdict that no longer matches its own row's numbers.

    Stage 1 is BRACKETED_BOUND on the real report (baseline fails a threshold
    the ceiling clears). Restating the threshold far above the baseline, with
    baseline/ceiling/verdict left untouched, makes the baseline pass on its own
    — reclassifying VACUOUS — which the untouched ``"verdict"`` field must not
    match.
    """
    tampered = _tamper(_real_rows(), "stage1_letter", threshold=1000.0)
    path = tmp_path / "gate_audit.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="internally inconsistent"):
        level_gate_summary_data(path)


def test_a_report_that_disagrees_with_the_hierarchy_table_refuses_to_draw(tmp_path) -> None:
    """POSITIVE CONTROL — a self-consistent row whose verdict the table doesn't share.

    Flip Stage 4's ceiling type from a reference model to an analytic bound and
    restate its verdict to match: baseline/ceiling/threshold reclassify as
    BRACKETED_BOUND under that type exactly as recorded, so the first
    (self-consistency) check has nothing to catch. But
    ``ailm.figures.architecture.LEVELS`` draws BRACKETED_REFERENCE for L4 — the
    drift the second check exists to catch even when the first one cannot.
    """
    tampered = _tamper(
        _real_rows(), "stage4_case", ceiling_type="analytic_bound", verdict="BRACKETED_BOUND"
    )
    path = tmp_path / "gate_audit.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="drifted apart"):
        level_gate_summary_data(path)


def test_a_row_with_no_ceiling_refuses_to_draw(tmp_path) -> None:
    """This figure only draws bracketed rows; an unaudited ceiling has nothing to plot."""
    tampered = _tamper(_real_rows(), "stage1_letter", ceiling=None, verdict="UNAUDITED")
    path = tmp_path / "gate_audit.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="only draws bracketed rows"):
        level_gate_summary_data(path)


class TestGenerate:
    def test_returns_a_figure_carrying_every_level_and_verdict(self, gate_audit_rows) -> None:
        fig = generate_level_gate_summary()
        try:
            assert isinstance(fig, Figure)
            drawn = " ".join(text.get_text() for text in fig.findobj(Text))
            for level in LEVELS:
                if level.built:
                    assert level.level in drawn
            assert "BRACKETED_BOUND" in drawn
            assert "BRACKETED_REFERENCE" in drawn
            assert "baseline" in drawn.lower()
            assert "ceiling" in drawn.lower()
        finally:
            plt.close(fig)

    def test_is_legible_and_clean(self, gate_audit_rows) -> None:
        """The same visual gate every other figure passes before it ships."""
        from ailm.figures._common import (
            assert_legible,
            assert_no_text_crosses_rules,
            assert_within_canvas,
        )

        fig = build_level_gate_summary_figure()
        try:
            assert_legible(fig)
            assert_within_canvas(fig)
            assert_no_text_crosses_rules(fig)
        finally:
            plt.close(fig)

    def test_generate_closes_cleanly_and_is_deterministic(self, tmp_path) -> None:
        from ailm.figures._common import save_figure

        plt.close("all")
        first = save_figure(generate_level_gate_summary(), "level_gate_summary", tmp_path)
        plt.close("all")
        second_dir = tmp_path / "again"
        second = save_figure(generate_level_gate_summary(), "level_gate_summary", second_dir)
        plt.close("all")
        assert first.read_bytes() == second.read_bytes()
