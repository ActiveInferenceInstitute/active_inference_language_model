"""The case-word bridge's influence sweep across the shipped mixing-weight grid.

One panel for the decision-121 sweep of the L4-to-L3 case-word inlet on the
EWT-dev development population: the mean per-token log-gain of the mixed L3
word prior against the bit-identical alpha = 0 anchor at every grid weight,
the whole-document clustered bootstrap per-emission effect with its interval
at each weight, the scrambled-word negative control, and the zero anchor
line. The recorded best weight is marked from its own token, not from
geometry re-derived here.

Every plotted value is read from the token map. The alpha grid is discovered
by prefix scan over the token namespace rather than listed, so a re-record on
a different grid changes the panel instead of silently plotting stale stems;
the discovery is anchored on the full token shape so the provenance tokens
(``_RUN`` / ``_SOURCE``) that shadow every ledger label cannot be mistaken for
sweep weights.
"""

from __future__ import annotations

import re
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ailm.figures._common import (
    MissingVariableError,
    load_manuscript_variables,
    token_float,
)
from ailm.viz_config import FONTS, PALETTE, apply_style

__all__ = ["case_word_sweep_data", "generate_case_word_sweep"]


#: The full token shape one grid weight's mean log-gain carries. Matching the
#: whole shape is what keeps the ``_RUN``/``_SOURCE`` provenance companions out
#: of the discovered grid.
_ALPHA_STEM_RE = re.compile(r"^CASE_WORD_SWEEP_ALPHA_(\d+)_(\d+)_MEAN_LOGGAIN$")


class SweepPoint(TypedDict):
    """One grid weight's series, in nats per scored token and per emission."""

    alpha: float
    mean_loggain: float
    effect: float
    ci_low: float
    ci_high: float
    scrambled: float


class CaseWordSweepData(TypedDict):
    """Everything the panel draws, assembled exclusively from manuscript tokens."""

    grid: list[SweepPoint]
    best_alpha: float
    confidence_pct: float


def _decode_alpha(stem_low: str, stem_frac: str) -> float:
    """Decode one token stem's alpha weight (the stem ``0_3`` is 0.3)."""
    return float(f"{int(stem_low)}.{int(stem_frac)}")


def _alpha_stem(alpha: float) -> str:
    """Token stem for one grid weight, the inverse of :func:`_decode_alpha`."""
    return f"{alpha:.1f}".replace(".", "_")


def _discover_grid(tokens: dict[str, str]) -> list[float]:
    """Grid weights present in the token map, sorted ascending.

    Raises:
        MissingVariableError: If the scan finds nothing. An empty sweep is a
            missing measurement, not an empty panel.
    """
    alphas = sorted(
        {
            _decode_alpha(match.group(1), match.group(2))
            for name in tokens
            if (match := _ALPHA_STEM_RE.match(name)) is not None
        }
    )
    if not alphas:
        raise MissingVariableError(
            "no CASE_WORD_SWEEP_ALPHA_*_MEAN_LOGGAIN tokens found; regenerate the "
            "token map from a ledger that recorded the decision-121 sweep"
        )
    return alphas


def case_word_sweep_data(variables: dict[str, str] | None = None) -> CaseWordSweepData:
    """The sweep's grid, all series per weight, and the coverage statement.

    Raises:
        MissingVariableError: If the alpha = 0 anchor, the recorded best weight,
            or any weight's clustered interval, clustered effect, or scrambled
            control is absent. One series without its control is the reading
            this panel exists to prevent, and the anchor is the estimand's own
            zero point.
    """
    tokens = variables if variables is not None else load_manuscript_variables()
    best_alpha = token_float(tokens, "CASE_WORD_SWEEP_BEST_ALPHA")
    confidence_pct = token_float(tokens, "CASE_WORD_SWEEP_CLUSTER_CONFIDENCE_PCT")

    grid: list[SweepPoint] = []
    for alpha in _discover_grid(tokens):
        stem = f"CASE_WORD_SWEEP_ALPHA_{_alpha_stem(alpha)}"
        grid.append(
            {
                "alpha": alpha,
                "mean_loggain": token_float(tokens, f"{stem}_MEAN_LOGGAIN"),
                "effect": token_float(tokens, f"{stem}_CLUSTER_LOGGAIN_EFFECT"),
                "ci_low": token_float(tokens, f"{stem}_CLUSTER_LOGGAIN_CI_LOW"),
                "ci_high": token_float(tokens, f"{stem}_CLUSTER_LOGGAIN_CI_HIGH"),
                "scrambled": token_float(tokens, f"{stem}_SCRAMBLED_MEAN_LOGGAIN"),
            }
        )

    zero = next((point for point in grid if point["alpha"] == 0.0), None)
    if zero is None:
        raise MissingVariableError(
            "the sweep grid has no alpha = 0 anchor; the estimand is a log-gain "
            "against that anchor, so a panel without it draws a difference "
            "against a model nobody ran"
        )
    if zero["mean_loggain"] != 0.0 or zero["effect"] != 0.0:
        raise MissingVariableError(
            "the alpha = 0 anchor is not zero on every drawn series; the "
            "bit-identical anchor identity is part of the estimand (decision 121)"
        )
    for point in grid:
        if not point["ci_low"] <= point["effect"] <= point["ci_high"]:
            raise MissingVariableError(
                f"at alpha = {point['alpha']!r} the clustered interval does not "
                "bracket its own recorded effect; the interval tokens belong to "
                "the per-emission clustered estimand, not to the per-token mean"
            )
    if not any(point["alpha"] == best_alpha for point in grid):
        raise MissingVariableError(
            f"the recorded best weight {best_alpha!r} is not on the discovered grid"
        )
    return {"grid": grid, "best_alpha": best_alpha, "confidence_pct": confidence_pct}


def build_case_word_sweep_figure(variables: dict[str, str] | None = None) -> Figure:
    apply_style()
    data = case_word_sweep_data(variables)
    grid = data["grid"]
    weights = [point["alpha"] for point in grid]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))

    # The clustered bootstrap interval belongs to the per-emission clustered
    # effect, not to the per-token mean: the estimand expands each token's
    # difference across its word's emissions, so the two series carry
    # different units and are drawn as two series rather than one bracketed
    # curve.
    means = [point["mean_loggain"] for point in grid]
    effects = [point["effect"] for point in grid]
    lower = [point["effect"] - point["ci_low"] for point in grid]
    upper = [point["ci_high"] - point["effect"] for point in grid]
    ax.plot(
        weights,
        means,
        color=PALETTE["MODEL"],
        marker="o",
        markersize=4,
        linewidth=1.6,
        label="mean per-token log-gain",
    )
    ax.errorbar(
        weights,
        effects,
        yerr=[lower, upper],
        color=PALETTE["MODEL"],
        marker="o",
        markersize=3,
        linewidth=1.0,
        alpha=0.85,
        capsize=3,
        label=f"clustered per-emission effect ({data['confidence_pct']:g}% interval)",
    )
    ax.plot(
        weights,
        [point["scrambled"] for point in grid],
        color=PALETTE["ANTI"],
        marker="s",
        markersize=4,
        linewidth=1.2,
        linestyle=":",
        label="scrambled-word negative control",
    )
    ax.axhline(0.0, color=PALETTE["INK"], linewidth=1.0, label="alpha = 0 anchor")
    ax.axvline(
        data["best_alpha"],
        color=PALETTE["FLOOR"],
        linewidth=1.2,
        linestyle="--",
        label="recorded best weight",
    )
    ax.set_xlabel("Mixing weight α", fontsize=FONTS["axis_label"])
    ax.set_ylabel("Log-gain (nats)", fontsize=FONTS["axis_label"])
    ax.set_title(
        "Case-word bridge influence sweep\n(EWT-dev development population)",
        fontsize=FONTS["title"],
    )
    ax.legend(fontsize=FONTS["legend"], loc="lower left")
    fig.tight_layout()
    return fig


def generate_case_word_sweep(variables: dict[str, str] | None = None) -> Figure:
    """The case-word sweep panel; the pipeline owns saving."""
    return build_case_word_sweep_figure(variables)
