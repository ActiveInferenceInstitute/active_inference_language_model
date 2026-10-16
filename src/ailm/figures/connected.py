"""The conditional channel schedule beside its report-bound GUM comparison.

The plot authenticates the recorded report and reconciles its summaries with
the connected manuscript claim reader. Bootstrap replay is a consistency check
using the registered estimator, not a new measurement or independent oracle.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ailm._publication_inventory import validate_publication_execution_binding
from ailm._publication_types import DerivativeError
from ailm.corpus.connected_gum import EXCLUDED_DOCUMENTS, MANIFEST_SHA256, load_frozen_population
from ailm.figures._common import ExportFigure, project_root
from ailm.levels.connected_evaluation import document_uncertainty
from ailm.levels.connected_types import Channel
from ailm.manuscript_variables._claims import _ClaimReaders, require_close
from ailm.manuscript_variables._io import _StrictVariableMap
from ailm.manuscript_variables.connected import _produce_connected_gum
from ailm.required_claims import CONNECTED_GUM_REQUIRED_CLAIMS
from ailm.value_formatting import ManuscriptVariableError
from ailm.verification.connected_provenance import (
    ExecutionBindingError,
    canonical_execution_binding,
)
from ailm.viz_config import FONTS, PALETTE, apply_style


@dataclass(frozen=True)
class ConnectedFigureData:
    """Validated figure inputs from the existing paired document measurement."""

    integrated: float
    disconnected: float
    gain: float
    low: float
    high: float
    document_gains: tuple[float, ...]
    documents: int
    oov_tokens: int
    coverage: float
    strength: float
    confidence: float
    outcome: str
    word_loss: str


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManuscriptVariableError(f"duplicate evidence key: {key}")
        result[key] = value
    return result


def _object(content: bytes) -> dict[str, Any]:
    value = json.loads(content, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ManuscriptVariableError("connected figure evidence must be a JSON object")
    return value


def _digest_object(value: object) -> str:
    content = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return hashlib.sha256(content).hexdigest()


def connected_core_data(
    ledger_path: Path | None = None, report_path: Path | None = None
) -> ConnectedFigureData:
    """Refuse missing, mixed-run, changed-source or inconsistent measured evidence."""
    root = project_root()
    ledger = _object((ledger_path or root / "data/claims.json").read_bytes())
    claims = ledger.get("claims")
    if ledger.get("_schema") != "ailm-claims-v2" or not isinstance(claims, dict):
        raise ManuscriptVariableError("connected figure requires a current claims ledger")
    selected = {key: value for key, value in claims.items() if key.startswith("connected_gum_")}
    if set(selected) != set(CONNECTED_GUM_REQUIRED_CLAIMS):
        raise ManuscriptVariableError("connected figure requires the exact connected claim family")
    readers = _ClaimReaders(selected)
    variables = _StrictVariableMap()
    _produce_connected_gum(variables, readers)
    report_bytes = (report_path or root / "output/reports/connected_gum.json").read_bytes()
    readers.require_exact_claim(
        "connected_gum_report_sha256", hashlib.sha256(report_bytes).hexdigest()
    )
    report = _object(report_bytes)
    try:
        if report["schema"] != "ailm-connected-gum-report-v1":
            raise ManuscriptVariableError("unsupported connected report schema")
        for suffix in ("corpus_sha256", "protocol_sha256", "execution_sha256", "runtime_sha256"):
            readers.require_exact_claim("connected_gum_" + suffix, report[suffix])
        if report["input_manifest_sha256"] != MANIFEST_SHA256 or report[
            "excluded_documents"
        ] != list(EXCLUDED_DOCUMENTS):
            raise ManuscriptVariableError("connected report population contract differs")
        binding = report["execution_binding"]
        if (
            not isinstance(binding, dict)
            or not binding
            or _digest_object(binding) != report["execution_sha256"]
        ):
            raise ManuscriptVariableError("connected report execution binding differs")
        try:
            validate_publication_execution_binding(root, binding, canonical_execution_binding(root))
        except (DerivativeError, ExecutionBindingError, OSError, ValueError) as error:
            raise ManuscriptVariableError(
                f"connected report execution inventory is incomplete or changed: {error}"
            ) from error
        if binding["src/ailm/levels/connected.py"] != report["runtime_sha256"]:
            raise ManuscriptVariableError("connected runtime identity differs")
        documents = report["documents"]
        if not isinstance(documents, list) or len(documents) != readers.num(
            "connected_gum_documents"
        ):
            raise ManuscriptVariableError("connected report document count differs")
        identities = tuple(row["document_id"] for row in documents)
        population, _ = load_frozen_population(root)
        if identities != population.document_ids:
            raise ManuscriptVariableError("connected report document identities differ")
        integrated: list[float] = []
        disconnected: list[float] = []
        totals = dict.fromkeys(
            (
                "normalized_tokens",
                "oov_tokens",
                "normalized_letters",
                "scored_characters",
                "source_sentences",
                "source_characters",
            ),
            0,
        )
        for row in documents:
            for name in (
                "normalized_tokens",
                "oov_tokens",
                "normalized_letters",
                "scored_characters",
            ):
                value = row["integrated"][name]
                if type(value) is not int or value < 0 or value != row["disconnected"][name]:
                    raise ManuscriptVariableError("connected report has unpaired counts")
                totals[name] += value
            for name in ("source_sentences", "source_characters"):
                value = row[name]
                if type(value) is not int or value < 0:
                    raise ManuscriptVariableError("connected report has invalid source counts")
                totals[name] += value
            for arm, losses in (("integrated", integrated), ("disconnected", disconnected)):
                value = row[arm]["letter_log_loss"]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(value)
                    or value < 0
                ):
                    raise ManuscriptVariableError("connected report has invalid document loss")
                losses.append(float(value))
        for suffix, count in totals.items():
            readers.require_exact_claim("connected_gum_" + suffix, count)
        gains = tuple(
            control - full for full, control in zip(integrated, disconnected, strict=True)
        )
        uncertainty = document_uncertainty(gains)
        if report["uncertainty"] != asdict(uncertainty):
            raise ManuscriptVariableError("connected report bootstrap does not reproduce")
        for suffix, value in (
            ("document_integrated_log_loss", float(np.mean(integrated))),
            ("document_disconnected_log_loss", float(np.mean(disconnected))),
            ("document_mean_gain", uncertainty.document_mean_gain),
            ("document_ci_low", uncertainty.ci_low),
            ("document_ci_high", uncertainty.ci_high),
        ):
            require_close(suffix, readers.num("connected_gum_" + suffix), value)
        expected_word_status = "infinite_oov" if totals["oov_tokens"] else "finite"
        if report["full_stream_word_loss_status"] != expected_word_status:
            raise ManuscriptVariableError("connected report word-loss status differs")
    except (KeyError, TypeError) as error:
        raise ManuscriptVariableError("incomplete connected report evidence") from error
    return ConnectedFigureData(
        readers.num("connected_gum_document_integrated_log_loss"),
        readers.num("connected_gum_document_disconnected_log_loss"),
        uncertainty.document_mean_gain,
        uncertainty.ci_low,
        uncertainty.ci_high,
        gains,
        len(documents),
        totals["oov_tokens"],
        readers.percentage("connected_gum_vocab_coverage_pct"),
        readers.num("connected_gum_strength"),
        readers.num("connected_gum_bootstrap_confidence_pct"),
        str(variables["CONNECTED_GUM_OUTCOME_TEXT"]),
        "infinite (OOV retained)" if totals["oov_tokens"] else "finite",
    )


def channel_edges() -> tuple[tuple[Channel, str, str], ...]:
    """Derive directed endpoints from the runtime's named channel contract."""
    edges = []
    for channel in Channel:
        sender, receiver = channel.value.split("_", 1)
        edges.append((channel, sender, receiver))
    return tuple(edges)


def _arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    name: str,
    shared: bool = False,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.5,
        linestyle="--" if shared else "-",
        color=PALETTE["MUTED"] if shared else PALETTE["MODEL"],
    )
    arrow.set_gid(name)
    ax.add_patch(arrow)


def generate_connected_core_figure(
    ledger_path: Path | None = None, report_path: Path | None = None
) -> Figure:
    """Draw causal schedule and paired gain only after authenticating their evidence."""
    return _draw_connected_core(connected_core_data(ledger_path, report_path))


def _draw_connected_core(data: ConnectedFigureData) -> Figure:
    """Lay out already validated values; public generation authenticates them first."""
    apply_style()
    fig = plt.figure(figsize=(6.4, 7.0), FigureClass=ExportFigure, export_dpi=400)
    diagram = fig.add_axes((0.055, 0.51, 0.89, 0.44))
    diagram.set(xlim=(0, 1), ylim=(0, 1))
    diagram.axis("off")
    diagram.set_title(
        "A  Conditional schedule · named channels", loc="left", fontsize=FONTS["title"], pad=10
    )
    nodes = {
        "phase": (0.48, 0.90),
        "clause": (0.48, 0.72),
        "case": (0.48, 0.54),
        "word": (0.48, 0.36),
        "phone": (0.26, 0.15),
        "letter": (0.70, 0.15),
    }
    labels = {
        "phase": "L6 · document phase",
        "clause": "L5 · widened clause",
        "case": "L4 · case",
        "word": "L3 · word + position",
        "phone": "L2B · phone",
        "letter": "L1 · letter",
    }
    for name, (x, y) in nodes.items():
        diagram.add_patch(
            FancyBboxPatch(
                (x - 0.145, y - 0.04),
                0.29,
                0.08,
                boxstyle="round,pad=0.005",
                facecolor=PALETTE["PANEL"],
                edgecolor=PALETTE["OUTLINE"],
            )
        )
        diagram.text(x, y, labels[name], ha="center", va="center", fontsize=FONTS["annotation"])
    for channel, sender, receiver in channel_edges():
        sx, sy = nodes[sender]
        tx, ty = nodes[receiver]
        if sender == "word":
            start, end = (sx + (tx - sx) * 0.22, sy - 0.045), (tx, ty + 0.045)
            label_x, label_y, align = (sx + tx) / 2, (sy + ty) / 2 + 0.014, "center"
        elif sy > ty:
            start, end = (sx - 0.04, sy - 0.045), (tx - 0.04, ty + 0.045)
            label_x, label_y, align = sx - 0.075, (sy + ty) / 2, "right"
        else:
            start, end = (sx + 0.04, sy + 0.045), (tx + 0.04, ty - 0.045)
            label_x, label_y, align = sx + 0.075, (sy + ty) / 2, "left"
        _arrow(diagram, start, end, name="channel:" + channel.value)
        diagram.text(
            label_x,
            label_y,
            channel.value,
            ha=align,
            va="center",
            fontsize=FONTS["cell"],
            color=PALETTE["MODEL"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )
    diagram.text(
        0.84,
        0.44,
        "Train-fitted\nUPOS encoder\nafter the word",
        ha="center",
        va="center",
        fontsize=FONTS["cell"],
        color=PALETTE["MUTED"],
    )
    _arrow(diagram, (0.63, 0.36), (0.83, 0.36), name="shared:word_upos", shared=True)
    _arrow(diagram, (0.83, 0.53), (0.63, 0.54), name="shared:upos_case", shared=True)
    diagram.text(
        0.48,
        0.022,
        "Observed spelling → L1, L2B, L3 · retained in both arms",
        ha="center",
        fontsize=FONTS["annotation"],
        color=PALETTE["MUTED"],
    )
    fig.text(
        0.055,
        0.445,
        "Solid: named channels disabled together in the comparator.\nDashed: UPOS encoder retained in both arms; outputs can differ.",
        fontsize=FONTS["footnote"],
        linespacing=1.5,
    )
    effect = fig.add_axes((0.22, 0.24, 0.72, 0.085))
    fig.text(
        0.055,
        0.397,
        "B  Paired GUM character-loss gain",
        fontsize=FONTS["title"],
        fontweight="bold",
    )
    fig.text(
        0.22,
        0.342,
        f"Connected {data.integrated:.3f} · channels disabled {data.disconnected:.3f}\nnats/character, averaged equally over documents",
        fontsize=FONTS["annotation"],
    )
    effect.axvline(0, color=PALETTE["MUTED"], linestyle="--", linewidth=1)
    effect.scatter(
        data.document_gains,
        np.full(data.documents, 0.75),
        s=20,
        facecolors="none",
        edgecolors=PALETTE["MUTED"],
        label="document pairs",
    )
    effect.hlines(0.30, data.low, data.high, color=PALETTE["MODEL"], linewidth=2)
    effect.plot([data.low, data.high], [0.30, 0.30], "|", color=PALETTE["MODEL"], markersize=10)
    effect.plot(data.gain, 0.30, "D", color=PALETTE["MODEL"], markersize=7)
    values = (0.0, data.low, data.high, *data.document_gains)
    span = max(values) - min(values)
    margin = max(span * 0.12, 0.025)
    effect.set_xlim(min(values) - margin, max(values) + margin)
    effect.set_ylim(0, 1.05)
    effect.set_yticks((0.30, 0.75), (f"Mean + {data.confidence:g}% CI", "Documents"))
    effect.set_xlabel(
        "Gain = disabled − connected (nats / character)", fontsize=FONTS["axis_label"]
    )
    effect.spines[["top", "right", "left"]].set_visible(False)
    effect.tick_params(axis="y", length=0)
    fig.text(
        0.22,
        0.115,
        f"{data.gain:+.3f} [{data.low:+.3f}, {data.high:+.3f}] · {data.outcome}",
        fontsize=FONTS["annotation"],
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.060,
        f"{data.documents} documents · strength {data.strength:g} · vocabulary coverage {data.coverage:.1f}%\n{data.oov_tokens:,} OOV tokens retained · word loss {data.word_loss}",
        fontsize=FONTS["annotation"],
        linespacing=1.5,
    )
    fig.text(
        0.055,
        0.007,
        "Local prospective comparison; no global joint posterior.\nAggregate ablation, not individual channel effect attribution.",
        fontsize=FONTS["footnote"],
        color=PALETTE["MUTED"],
    )
    return fig
