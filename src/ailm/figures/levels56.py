"""Figures for the Stage 5 non-passing result and narrative level (Stage 6).

``clause_null``
    The project's first above-reference-and-failed gate, drawn honestly: the EWT
    measurement (type-free bar, the mixture that sits *worse*, the type-ORACLE
    privileged-label reference that also sits worse) beside the detectability
    control - the same harness on genuinely different synthetic types, where
    the mixture wins decisively. The juxtaposition is the argument: the
    instrument fires; the EWT gate is non-passing and the paired
    privileged-label gain remains unresolved at this model order and population.
``narrative_channel``
    Stage 6's distinct estimands on one canvas: the registered observed-position
    point score, its non-confirmatory document-clustered primary interval, the
    descriptive live-Agent score, and the independent L6-to-L5 prior
    intervention with its zero-weight anchor and phase-permuted control.

Both panels draw every number from the ledger and refuse to draw if the
relations the captions assert no longer hold.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ailm.figures._common import (
    resolve_variables as _v,
)
from ailm.figures._common import (
    token_float,
    token_int,
)
from ailm.viz_config import FONTS, PALETTE, apply_style

__all__ = [
    "clause_null_data",
    "generate_clause_null",
    "generate_narrative_channel",
    "narrative_channel_data",
]

COLOR_MODEL = PALETTE["MODEL"]
COLOR_BASELINE = PALETTE["UNIGRAM"]
COLOR_ANTI = PALETTE["ANTI"]
COLOR_ACCENT = PALETTE["ACCENT"]
#: The color this paper gives a reference the model is measured against rather
#: than a bound nothing can pass: the same green as fig:perplexity_ladder's
#: empirical order-1 optimum and fig:g2p_accuracy's smoothing ceiling.
COLOR_REFERENCE = PALETTE["FLOOR"]


def clause_null_data(variables: dict[str, str] | None = None) -> dict[str, Any]:
    """The Stage 5 point result, paired bounds and control, re-checked.

    Raises:
        ValueError: If the mixture or fitted privileged reference no longer
            sits at-or-above the type-free bar, if the control stops
            firing, or if either oracle-gain interval stops spanning zero — each
            means the recorded finding changed and decision 83
            must change before any figure does.
    """
    v = _v(variables)
    data: dict[str, Any] = {
        "type_free": token_float(v, "STAGE5_TYPEFREE_PERPLEXITY"),
        "mixture": token_float(v, "STAGE5_MIXTURE_PERPLEXITY"),
        "privileged_reference": token_float(v, "STAGE5_TYPE_ORACLE_PERPLEXITY"),
        "privileged_label_gain_pct": token_float(v, "STAGE5_PRIVILEGED_LABEL_GAIN_PCT"),
        "gain_nats_per_sentence": token_float(v, "STAGE5_ORACLE_GAIN_NATS_PER_SENTENCE"),
        "gain_sentence_ci_low": token_float(v, "STAGE5_ORACLE_GAIN_SENTENCE_CI_LOW"),
        "gain_sentence_ci_high": token_float(v, "STAGE5_ORACLE_GAIN_SENTENCE_CI_HIGH"),
        "gain_nats_per_position": token_float(v, "STAGE5_ORACLE_GAIN_NATS_PER_POSITION"),
        "gain_position_ci_low": token_float(v, "STAGE5_ORACLE_GAIN_POSITION_CI_LOW"),
        "gain_position_ci_high": token_float(v, "STAGE5_ORACLE_GAIN_POSITION_CI_HIGH"),
        "gain_ci_confidence_pct": token_float(v, "STAGE5_ORACLE_GAIN_CI_CONFIDENCE_PCT"),
        "gain_documents": token_int(v, "STAGE5_ORACLE_GAIN_BOOTSTRAP_DOCUMENTS"),
        "control_type_free": token_float(v, "STAGE5_CONTROL_TYPEFREE_PERPLEXITY"),
        "control_mixture": token_float(v, "STAGE5_CONTROL_MIXTURE_PERPLEXITY"),
        "control_classification_pct": token_float(v, "STAGE5_CONTROL_CLASSIFICATION_PCT"),
        # Baked into the PNG until now, where no prose audit could reach it.
        "control_majority_pct": token_float(v, "STAGE5_CONTROL_MAJORITY_BASELINE_PCT"),
        "positions": token_int(v, "STAGE5_TEST_POSITIONS"),
        "auc": token_float(v, "STAGE5_AUC"),
    }
    if not data["mixture"] >= data["type_free"]:
        raise ValueError("the mixture now beats the bar — the Stage 5 outcome changed")
    if not data["privileged_reference"] >= data["type_free"]:
        raise ValueError(
            "the privileged-label reference now clears the bar — the Stage 5 gate placement changed"
        )
    if not data["privileged_label_gain_pct"] <= 0.0:
        raise ValueError("the privileged-label reference point gain is now positive")
    if not data["control_mixture"] < data["control_type_free"] * 0.95:
        raise ValueError("the detectability control stopped firing")
    if not data["gain_sentence_ci_low"] <= 0.0 <= data["gain_sentence_ci_high"]:
        raise ValueError("the sentence-level privileged-label gain is no longer unresolved")
    if not data["gain_position_ci_low"] <= 0.0 <= data["gain_position_ci_high"]:
        raise ValueError("the position-weighted privileged-label gain is no longer unresolved")
    return data


def generate_clause_null(variables: dict[str, str] | None = None) -> Figure:
    """The above-reference failed gate beside the instrument that could refute it."""
    apply_style()
    data = clause_null_data(variables)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 5.0))

    labels = ["type-free bar", "mixture\n(the model)", "type known\n(privileged ref.)"]
    values = [data["type_free"], data["mixture"], data["privileged_reference"]]
    # The mixture IS the shipped model, so it takes MODEL in this panel and in the
    # control panel alike; it previously took ANTI here and MODEL there, asking a
    # reader to track one quantity across two encodings inside one figure. ANTI is
    # reserved for a mis-wiring and no mis-wiring is drawn here. The privileged
    # label reference takes the paper's reference green.
    colors = [COLOR_BASELINE, COLOR_MODEL, COLOR_REFERENCE]
    left.bar(labels, values, color=colors, width=0.55)
    span = max(values) - min(values)
    # Headroom above the bars holds the inset that draws the paired interval.
    left.set_ylim(min(values) - 2.5 * span, max(values) + 4.0 * span)
    for x, value in enumerate(values):
        # The oracle bar's label would land on the dashed bar line at the
        # default offset; every label clears it at half a span.
        left.text(x, value + 0.5 * span, f"{value:g}", ha="center", fontsize=FONTS["footnote"])
    left.axhline(data["type_free"], color=PALETTE["OUTLINE"], lw=1.2, linestyle="--")
    left.set_ylabel("per-position perplexity (note the axis span)")

    # The paired gain and its interval are in nats per position, a different
    # unit from the perplexity axis, so they are drawn as a mark in an inset
    # rather than only quoted in the title: an interval a reader can see spanning
    # the zero line is the finding, and title text is not a mark.
    inset = left.inset_axes((0.20, 0.62, 0.42, 0.26))
    gain = data["gain_nats_per_position"]
    inset.errorbar(
        [0.0],
        [gain],
        yerr=[[gain - data["gain_position_ci_low"]], [data["gain_position_ci_high"] - gain]],
        fmt="o",
        color=COLOR_MODEL,
        ecolor=COLOR_MODEL,
        capsize=5,
        markersize=6,
        linewidth=1.6,
    )
    inset.axhline(0.0, color=PALETTE["OUTLINE"], lw=1.0, linestyle="--")
    inset.set_xlim(-1.0, 1.0)
    inset.set_xticks([])
    inset.set_yticks([data["gain_position_ci_low"], 0.0, data["gain_position_ci_high"]])
    inset.tick_params(axis="y", labelsize=FONTS["cell"])
    inset.set_title(
        f"paired gain, nats/position, {data['gain_ci_confidence_pct']:g}% CI",
        fontsize=FONTS["cell"],
        pad=3,
    )
    inset.spines[["top", "right"]].set_visible(False)
    left.set_title(
        f"EWT split: privileged-label gain unresolved\n"
        f"{data['gain_nats_per_position']:g} nats/position, "
        f"{data['gain_ci_confidence_pct']:g}% CI "
        f"[{data['gain_position_ci_low']:g}, {data['gain_position_ci_high']:g}]\n"
        f"whole-document bootstrap ({data['gain_documents']} documents)",
        fontsize=FONTS["subtitle"],
    )
    left.spines[["top", "right"]].set_visible(False)

    control_labels = ["type-free bar", "mixture"]
    control_values = [data["control_type_free"], data["control_mixture"]]
    right.bar(control_labels, control_values, color=[COLOR_BASELINE, COLOR_MODEL], width=0.55)
    for x, value in enumerate(control_values):
        right.text(x, value + 0.05, f"{value:g}", ha="center", fontsize=FONTS["footnote"])
    right.set_ylim(0, max(control_values) * 1.25)
    # Two categories on a default categorical axis leave a gap between the bars
    # half again as wide as a bar, which reads as a third bar that failed to
    # draw. These limits give the pair the same bar-to-gap ratio as the
    # three-category panel on the left.
    right.set_xlim(-0.6, 1.6)
    right.set_ylabel("per-position perplexity")
    right.set_title(
        f"the control: planted types, same harness\n"
        f"(classification {data['control_classification_pct']:g}% vs "
        f"{data['control_majority_pct']:g}%)",
        fontsize=FONTS["subtitle"],
    )
    right.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig


def narrative_channel_data(variables: dict[str, str] | None = None) -> dict[str, Any]:
    """Stage 6's separate descriptive and primary results, re-checked.

    Raises:
        ValueError: If the recorded point result, clustered primary verdict,
            live-Agent score, or prior-intervention controls no longer support
            the figure's wording.
    """
    v = _v(variables)
    data: dict[str, Any] = {
        "phase_logloss": token_float(v, "STAGE6_PHASE_LOGLOSS"),
        "unigram_logloss": token_float(v, "STAGE6_UNIGRAM_LOGLOSS"),
        "improvement": token_float(v, "STAGE6_LOGLOSS_IMPROVEMENT_NATS"),
        "sign_p": token_float(v, "STAGE6_SIGN_TEST_P"),
        "cluster_effect": token_float(v, "CLUSTER_STAGE6_LOGLOSS_EFFECT"),
        "cluster_ci_low": token_float(v, "CLUSTER_STAGE6_LOGLOSS_CI_LOW"),
        "cluster_ci_high": token_float(v, "CLUSTER_STAGE6_LOGLOSS_CI_HIGH"),
        "cluster_p": token_float(v, "CLUSTER_STAGE6_LOGLOSS_P"),
        "cluster_documents": token_int(v, "CLUSTER_STAGE6_LOGLOSS_CLUSTERS"),
        "live_logloss": token_float(v, "STAGE6_LIVE_FILTER_LOGLOSS"),
        "live_improvement": token_float(v, "STAGE6_LIVE_FILTER_IMPROVEMENT_NATS"),
        "live_sign_p": token_float(v, "STAGE6_LIVE_FILTER_SIGN_P"),
        "static": token_float(v, "STAGE6_STATIC_INTERROGATIVE_MASS"),
        "shifts": {
            "opening": token_float(v, "STAGE6_OPENING_INTERROGATIVE_SHIFT"),
            "middle": token_float(v, "STAGE6_MIDDLE_INTERROGATIVE_SHIFT"),
            "closing": token_float(v, "STAGE6_CLOSING_INTERROGATIVE_SHIFT"),
        },
        "scrambled_opening": token_float(v, "STAGE6_SCRAMBLED_OPENING_SHIFT"),
        "n_sentences": token_int(v, "STAGE6_TEST_SENTENCES"),
    }
    if not data["improvement"] > 0:
        raise ValueError("the registered observed-position point is no longer positive")
    if not data["cluster_ci_low"] < 0 < data["cluster_ci_high"]:
        raise ValueError("the document-clustered primary interval is no longer non-confirmatory")
    if not data["cluster_p"] > 0.05:
        raise ValueError("the document-clustered primary p-value is now confirmatory")
    if not np.isclose(data["cluster_effect"], data["improvement"], atol=1e-4):
        raise ValueError("the registered point and clustered effect describe different estimands")
    if not data["live_improvement"] > 0 or not np.isclose(
        data["unigram_logloss"] - data["live_logloss"],
        data["live_improvement"],
        atol=1e-4,
    ):
        raise ValueError("the live-Agent score and its recorded improvement disagree")
    if not data["shifts"]["opening"] > 0:
        raise ValueError("the opening-phase shift is no longer positive")
    if data["scrambled_opening"] == data["shifts"]["opening"]:
        raise ValueError("the scramble is indistinguishable from the message")
    return data


def generate_narrative_channel(variables: dict[str, str] | None = None) -> Figure:
    """Observed-position, live-filter, and L6-to-L5 prior-intervention results."""
    apply_style()
    v = _v(variables)
    data = narrative_channel_data(v)

    # The sentence-level sign p crosses Holm but not the fixed Bonferroni
    # threshold. It remains secondary because the registered mean's
    # document-clustered interval spans zero and a reversed-position falsifier
    # also produces favorable sign counts.
    alpha_bonferroni = token_float(v, "STAT_BONFERRONI_ALPHA")
    if data["sign_p"] < alpha_bonferroni:
        raise ValueError(
            "the Stage 6 sign test now crosses the repository-wide Bonferroni "
            "threshold; the 'nominal only' footnote no longer holds and must be "
            "rewritten, not redrawn."
        )

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 5.0))

    losses = [data["unigram_logloss"], data["phase_logloss"], data["live_logloss"]]
    left.bar(
        ["type unigram\n(the bar)", "observed position\n(registered)", "live Agent\n(descriptive)"],
        losses,
        # MODEL belongs to the live pymdp Agent, which is the only L6 quantity
        # drawn through pymdp's inference path; the registered observed-position
        # score is a conditional table, not inference, and takes ACCENT. These
        # two were swapped here relative to fig:inference_evidence, which draws
        # the same three quantities, so one reader's color meant two things.
        color=[COLOR_BASELINE, COLOR_ACCENT, COLOR_MODEL],
        width=0.45,
    )
    span = losses[0] - min(losses[1:])
    left.set_ylim(min(losses[1:]) - 2.5 * span, losses[0] + 2.5 * span)
    for x, value in enumerate(losses):
        left.text(x, value + 0.25 * span, f"{value:g}", ha="center", fontsize=FONTS["footnote"])
    left.set_ylabel("held-out type log-loss, nats per sentence\n(note the axis span)")
    left.set_title(
        "two distinct descriptive scores; registered mean is non-confirmatory",
        fontsize=FONTS["subtitle"],
    )
    left.text(
        0.5,
        0.97,
        f"registered delta +{data['improvement']:g}; document-cluster 95% CI\n"
        f"[{data['cluster_ci_low']:g}, {data['cluster_ci_high']:g}], "
        f"randomization p = {data['cluster_p']:g} "
        f"({data['cluster_documents']} documents)\n"
        "sentence sign is secondary; reversed-position control also favors model",
        transform=left.transAxes,
        ha="center",
        va="top",
        fontsize=FONTS["cell"],
        color=PALETTE["MUTED"],
    )
    left.spines[["top", "right"]].set_visible(False)

    phases = ["opening", "middle", "closing"]
    blended = [data["static"] + data["shifts"][phase] for phase in phases]
    phase_x = np.arange(len(phases))
    right.bar(phase_x, blended, color=COLOR_MODEL, width=0.5, zorder=2)
    right.axhline(
        data["static"],
        color=PALETTE["OUTLINE"],
        lw=1.4,
        linestyle="--",
        label=f"static clause prior ({data['static']:g}) = α0 anchor",
    )
    right.scatter(
        [0],
        [data["static"] + data["scrambled_opening"]],
        marker="x",
        s=90,
        color=COLOR_ANTI,
        zorder=3,
        label="phase-permuted control (opening)",
    )
    # A value label that would sit on the static-prior dashed line is lifted just
    # above the line instead, so the line never strikes through the digits.
    label_gap = 0.004
    line_clearance = 0.012
    for xi, value in zip(phase_x, blended, strict=True):
        label_y = value + label_gap
        if abs(label_y - data["static"]) < line_clearance:
            label_y = data["static"] + label_gap * 1.5
        right.text(xi, label_y, f"{value:.3f}", ha="center", fontsize=FONTS["footnote"])
    right.set_xticks(phase_x, phases)
    right.set_ylim(0, max(blended) * 1.45)
    right.set_ylabel("clause-level P(interrogative) after prior intervention")
    right.set_title("independent L6 -> L5 D intervention", fontsize=FONTS["subtitle"])
    right.legend(frameon=False, fontsize=FONTS["cell"], loc="upper right")
    right.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig
