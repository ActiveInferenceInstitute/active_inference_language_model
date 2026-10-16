"""One panel: every built level's shipped model against its own baseline and ceiling.

Six figures already carry six separate, unit-specific arguments — perplexity ladders,
an accuracy comparison, a case-agreement bar, a clause perplexity pair, a narrative
log-loss pair — each on its own axis, in its own unit, in its own section. None of them
answers the question a reader forms after seeing all six: across the whole stack, how
far toward the achievable ceiling did the shipped model actually get, and where does
that placement's epistemic strength differ level to level?

This figure answers exactly that, on one shared axis, by normalizing every row to the
same span: 0 is that level's own trivial baseline (which is also its gate's threshold —
every audited gate here is baseline-relative, decision-for-decision the same retrospective
design), 1 is that level's own achievable ceiling. A model at 0.5 closed half the gap
between doing nothing and the best this model order can do; a model past 1.0 beat its own
ceiling comparator outright, which is possible for a REFERENCE_MODEL or PRIVILEGED_ORACLE
ceiling (never for a MODEL_CLASS_BOUND or ANALYTIC_BOUND one — beating a real bound would
falsify it) and happens twice here, at Stage 4 and Stage 6.

Six numbers with six different units cannot share an axis by rescaling alone without
losing the reader's ability to check the claim, so both the fraction AND the four raw
numbers a fraction is computed from (baseline, model, ceiling, and which comparator type
the ceiling is) are drawn on every row: the fraction is the summary, the numbers beside it
are the audit trail.

Determinism and the falsifier
------------------------------
Every row's baseline, model, ceiling, ceiling type and verdict are read from
``output/reports/gate_audit.json`` — the same committed, re-runnable artifact
:mod:`ailm.figures.architecture` already binds its Stage-column badges to (see
``tests/test_figures_architecture.py``). Nothing here re-derives those four numbers by a
second measurement; this figure's only computation is the fraction and the verdict
classification, both of which are cheap enough to fall out of ``ailm.gates.classify_gate``
rather than being trusted from the report.

Two independent checks gate the draw, and either raises ``ValueError`` rather than
silently drawing a stale or invented number:

1. Reclassifying each row's own ``(baseline, ceiling, threshold, direction,
   ceiling_type)`` through :func:`ailm.gates.classify_gate` must reproduce the verdict
   the report recorded for it — a report edited to change one field without the others
   would otherwise draw a verdict badge the four numbers beside it no longer support.
2. That reclassified verdict must equal the verdict half of the same level's row in
   :data:`ailm.figures.architecture.LEVELS`, which is the table
   ``tests/test_figures_architecture.py`` already holds to the live audit. Two figures
   drawing two different verdicts for the same level would be a silent contradiction no
   reader could see from either figure alone; this makes it a build failure instead.

Every generator returns an unwritten ``Figure``; ``scripts/generate_figures.py``
discovers them by the ``generate_`` prefix and handles saving.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ailm.figures._common import MissingVariableError, project_root
from ailm.figures.architecture import LEVELS
from ailm.figures.results import role_color
from ailm.gates import GateId, classify_gate
from ailm.viz_config import FIGSIZES, FONTS, PALETTE, apply_style

__all__ = [
    "generate_level_gate_summary",
    "level_gate_summary_data",
]

#: The level's own row in :data:`ailm.figures.architecture.LEVELS` names the gate it
#: draws a badge for only when the stage audits more than one gate (Stage 2 and
#: Stage 5 — see ``LevelRow.gate_note``). A level whose row leaves ``gate_id`` unset
#: audits exactly one gate, and this is which: the id ``scripts/audit_gates.py``
#: assigns that stage. Keeping the map here rather than reading it off "the only
#: entry in a filtered ``AUDITED_GATE_IDS`` list" makes an accidental second gate
#: for one of these stages a ``KeyError`` this module raises immediately, rather
#: than a wrong guess.
_IMPLIED_GATE_ID: dict[str, GateId] = {
    "L1": GateId.STAGE1_LETTER,
    "L3": GateId.STAGE3_WORD,
    "L4": GateId.STAGE4_CASE,
    "L6": GateId.STAGE6_NARRATIVE,
}

#: Display unit for each audited gate's own metric. Not measured — these are the
#: metric's units, fixed by what ``scripts/audit_gates.py`` computes for that stage,
#: the same way every other figure's axis label is a fixed unit string rather than a
#: ledger token. Kept short: this string sits in a y-tick label, the one part of
#: the figure whose width sets the whole canvas's left margin, and the full unit
#: name (e.g. "per-emission perplexity") is already spelled out in the sections
#: each of these six numbers is native to.
_UNIT_BY_GATE: dict[GateId, str] = {
    GateId.STAGE1_LETTER: "ppl",
    GateId.STAGE2_G2P: "acc",
    GateId.STAGE3_WORD: "ppl",
    GateId.STAGE4_CASE: "pts",
    GateId.STAGE5_CLAUSE: "ppl",
    GateId.STAGE6_NARRATIVE: "nats",
}

#: Comparator-type tag drawn beside the ceiling value — the finer distinction
#: ``classify_gate`` collapses into one verdict (BRACKETED_REFERENCE covers both
#: REFERENCE_MODEL and PRIVILEGED_ORACLE) but that a reader auditing exceedability
#: still needs: only a bound can never be beaten; a reference or an oracle can.
_COMPARATOR_TAG: dict[str, str] = {
    "model_class_bound": "bound, cannot be beaten",
    "analytic_bound": "bound, cannot be beaten",
    "reference_model": "reference, exceedable",
    "privileged_oracle": "oracle, exceedable",
}


def _gate_audit_report_path() -> Path:
    """``<project root>/output/reports/gate_audit.json`` — committed, not generated here."""
    return project_root() / "output" / "reports" / "gate_audit.json"


def _load_gate_audit(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the committed gate-audit report, keyed by ``gate_id``.

    Args:
        path: Explicit path, for tests. Defaults to :func:`_gate_audit_report_path`.

    Returns:
        Mapping from gate id string to that gate's audit record.

    Raises:
        MissingVariableError: If the report is absent, not a JSON array, or a
            record is missing the fields every row of it must carry.
    """
    target = path if path is not None else _gate_audit_report_path()
    if not target.exists():
        raise MissingVariableError(
            f"Gate audit report not found at {target}. This figure reads every "
            "level's baseline/model/ceiling from it rather than re-measuring them — "
            "run scripts/audit_gates.py --full first."
        )
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise MissingVariableError(f"{target} must contain a JSON array of gate records.")
    records: dict[str, dict[str, Any]] = {}
    required = {
        "gate_id",
        "stage",
        "baseline",
        "baseline_name",
        "model",
        "ceiling",
        "ceiling_name",
        "ceiling_type",
        "threshold",
        "verdict",
        "outcome",
    }
    for record in payload:
        missing = required - set(record)
        if missing:
            raise MissingVariableError(
                f"{target}: a record is missing field(s) {sorted(missing)}: {record!r}"
            )
        records[str(record["gate_id"])] = record
    return records


def level_gate_summary_data(report_path: Path | None = None) -> list[dict[str, Any]]:
    """One row per built level, with its fraction-of-the-gap position and its audit trail.

    Args:
        report_path: Explicit path to the gate-audit JSON, for tests. Defaults to
            the committed ``output/reports/gate_audit.json``.

    Returns:
        Rows in :data:`ailm.figures.architecture.LEVELS` order (top to bottom in the
        hierarchy figure), each carrying the level id/name, the gate id, the stage
        label the audit itself uses, baseline/model/ceiling with their names and unit,
        the comparator type, the fraction of the baseline-to-ceiling gap the model
        closed (``None`` only if ``ceiling == baseline``, which no audited gate here
        has), and the verdict/status text.

    Raises:
        MissingVariableError: If the gate-audit report is absent or malformed.
        ValueError: If a row's own recorded verdict cannot be reproduced by
            reclassifying its own baseline/ceiling/threshold/comparator-type, or if
            that verdict disagrees with the level's own row in
            :data:`ailm.figures.architecture.LEVELS`. Either would mean this figure
            and the hierarchy table, or this figure and the audit it reads, no longer
            agree about the same level.
    """
    audit = _load_gate_audit(report_path)
    rows: list[dict[str, Any]] = []
    for level_row in LEVELS:
        if not level_row.built:
            continue  # Level 7 (speech): a contract, not an audited gate.
        gate_id = level_row.gate_id or _IMPLIED_GATE_ID.get(level_row.level)
        if gate_id is None:
            raise KeyError(
                f"{level_row.level}: built with no gate_id and no entry in "
                "_IMPLIED_GATE_ID — a new level needs one or the other."
            )
        gate_id = GateId(gate_id)
        record = audit.get(str(gate_id))
        if record is None:
            raise ValueError(f"{level_row.level}: gate audit report has no record for {gate_id!r}.")
        baseline = float(record["baseline"])
        model = float(record["model"])
        ceiling = None if record["ceiling"] is None else float(record["ceiling"])
        threshold = float(record["threshold"])
        if ceiling is None:
            raise ValueError(
                f"{level_row.level} ({gate_id}): this figure only draws bracketed "
                "rows (a measured ceiling), and the report carries none."
            )
        direction: Literal["higher_is_better", "lower_is_better"] = (
            "lower_is_better" if ceiling < baseline else "higher_is_better"
        )
        recomputed = str(
            classify_gate(
                baseline, ceiling, threshold, direction, ceiling_type=record["ceiling_type"]
            )
        )
        if recomputed != record["verdict"]:
            raise ValueError(
                f"{level_row.level} ({gate_id}): reclassifying baseline={baseline!r}, "
                f"ceiling={ceiling!r}, threshold={threshold!r} gives {recomputed}, but "
                f"{_gate_audit_report_path()} records {record['verdict']!r}. The report "
                "is internally inconsistent — reconcile it before drawing."
            )
        expected_verdict = level_row.status.split("\n", 1)[0]
        if recomputed != expected_verdict:
            raise ValueError(
                f"{level_row.level}: the gate audit classifies {gate_id} as "
                f"{recomputed}, but ailm.figures.architecture.LEVELS draws "
                f"{expected_verdict!r} for this level. The hierarchy table and this "
                "figure have drifted apart."
            )
        fraction = (model - baseline) / (ceiling - baseline)
        rows.append(
            {
                "level": level_row.level,
                "name": level_row.name,
                "gate_id": str(gate_id),
                "stage_label": record["stage"],
                "baseline": baseline,
                "baseline_name": record["baseline_name"],
                "model": model,
                "ceiling": ceiling,
                "ceiling_name": record["ceiling_name"],
                "ceiling_type": record["ceiling_type"],
                "unit": _UNIT_BY_GATE[gate_id],
                "fraction": fraction,
                "verdict": recomputed,
                "status": level_row.status,
            }
        )
    return rows


def build_level_gate_summary_figure(report_path: Path | None = None) -> Figure:
    """The one-panel gate-verdict summary: every level on one fractional axis."""
    apply_style()
    rows = level_gate_summary_data(report_path)

    fig, ax = plt.subplots(figsize=FIGSIZES["level_summary"])
    ys = list(range(len(rows) - 1, -1, -1))

    color_baseline = role_color("honest")
    color_model = role_color("model")
    color_ceiling = role_color("ceiling")

    ax.axvline(0.0, color=PALETTE["MUTED"], linewidth=1.0, linestyle=":", zorder=1)
    ax.axvline(1.0, color=PALETTE["MUTED"], linewidth=1.0, linestyle=":", zorder=1)
    _label_bbox = {
        "boxstyle": "square,pad=0.15",
        "facecolor": PALETTE["BACKGROUND"],
        "edgecolor": "none",
    }
    top_y = ys[0] + 0.55
    ax.text(
        0.0,
        top_y,
        "0: baseline\n(= this gate's threshold)",
        ha="center",
        va="bottom",
        fontsize=FONTS["annotation"],
        color=PALETTE["MUTED"],
        bbox=_label_bbox,
        zorder=7,
    )
    ax.text(
        1.0,
        top_y,
        "1: ceiling",
        ha="center",
        va="bottom",
        fontsize=FONTS["annotation"],
        color=PALETTE["MUTED"],
        bbox=_label_bbox,
        zorder=7,
    )

    y_labels: list[str] = []
    for y, row in zip(ys, rows, strict=True):
        hatch = "" if row["verdict"] == "BRACKETED_BOUND" else "OO"
        ax.barh(
            y,
            width=1.0,
            left=0.0,
            height=0.42,
            color=color_ceiling,
            alpha=0.22,
            edgecolor=PALETTE["OUTLINE"],
            linewidth=0.8,
            hatch=hatch,
            zorder=2,
        )
        ax.scatter(
            [0.0],
            [y],
            marker="D",
            s=90,
            color=color_baseline,
            zorder=4,
            edgecolors=PALETTE["INK"],
            linewidths=0.6,
        )
        ax.scatter(
            [1.0],
            [y],
            marker="^",
            s=110,
            color=color_ceiling,
            zorder=4,
            edgecolors=PALETTE["INK"],
            linewidths=0.6,
        )
        beyond = row["fraction"] > 1.0
        ax.scatter(
            [row["fraction"]],
            [y],
            marker="*" if beyond else "o",
            s=190 if beyond else 120,
            color=color_model,
            zorder=5,
            edgecolors=PALETTE["INK"],
            linewidths=0.7,
        )
        # The raw numbers a fraction is computed from live in the row's own
        # y-tick label rather than as floating plot-area text: a label at a
        # fixed data-x coordinate has to guess how wide the widest note is and
        # set the axis limit to match, which is exactly backwards when six
        # rows carry six different units and string lengths. The tick label
        # column has no such constraint — matplotlib sizes the left margin to
        # whatever it holds.
        verdict_line, outcome_line = row["status"].split("\n", 1)
        numbers = (
            f"{row['baseline']:.3g}→{row['model']:.3g}→{row['ceiling']:.3g} "
            f"{row['unit']}, {_COMPARATOR_TAG[row['ceiling_type']]}"
        )
        # Four short lines rather than one or two long ones: the widest single
        # line, not the label's total height, is what a canvas-containment
        # check on a fixed left margin actually bounds, and "BRACKETED_
        # REFERENCE" beside "PASS (DEV-SESSION)" on one line was wider than six
        # rows of tick-label margin can hold without shrinking type.
        y_labels.append(f"{row['level']} {row['name']}\n{verdict_line}\n{outcome_line}\n{numbers}")

    ax.set_yticks(ys)
    ax.set_yticklabels(y_labels, fontsize=FONTS["tick_label"])
    ax.set_ylim(-0.6, ys[0] + 1.35)
    ax.set_xlim(-0.10, 1.32)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    ax.set_xlabel(
        "fraction of baseline→ceiling closed\n(beyond 1: beat an exceedable ceiling)",
        fontsize=FONTS["axis_label"],
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="D",
                linestyle="None",
                color=color_baseline,
                label="baseline (= threshold)",
            ),
            Line2D([], [], marker="o", linestyle="None", color=color_model, label="shipped model"),
            Line2D(
                [],
                [],
                marker="*",
                linestyle="None",
                color=color_model,
                markersize=11,
                label="model, beyond its own ceiling",
            ),
            Line2D([], [], marker="^", linestyle="None", color=color_ceiling, label="ceiling"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.34),
        ncol=2,
        fontsize=FONTS["legend"],
        frameon=True,
        facecolor=PALETTE["BACKGROUND"],
        edgecolor="none",
    )
    ax.set_title(
        "Every level, placed between its own baseline and ceiling",
        fontsize=FONTS["title"],
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.30, right=0.92, top=0.92, bottom=0.24)
    return fig


def generate_level_gate_summary(report_path: Path | None = None) -> Figure:
    """Generate the level gate-verdict summary; the pipeline owns saving."""
    return build_level_gate_summary_figure(report_path)
