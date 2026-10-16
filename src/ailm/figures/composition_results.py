"""Figures for Stages 3–4, generation, and the inter-level composition result.

Four panels, each drawn from the claims ledger rather than from typed numbers, so
a figure cannot drift from the measurement that produced it:

``word_miswirings``
    Stage 3's four mis-wirings against the letter-bigram bar. Three of them clear
    it. The panel exists to make "beating the baseline is necessary, not
    sufficient" visible rather than argued.
``case_both_defects``
    Stage 4's single declared threshold against two populations, landing as
    VACUOUS on one and ABOVE_REFERENCE on the other.
``generation_sweep``
    Distance to the true word distribution against policy precision — U-shaped,
    with both endpoints failing for opposite reasons.
``channel_comparison``
    The headline composition result: two inter-level channels into the *same*
    receiving level, one monotone-declining and one peaking in the middle.

Every generator returns an unwritten ``Figure``; ``scripts/generate_figures.py``
discovers them by the ``generate_`` prefix and handles saving.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ailm.benchmarks import paired_difference_interval
from ailm.figures._common import (
    resolve_variables as _v,
)
from ailm.figures._common import (
    token_float,
    token_int,
)
from ailm.levels.lexical_channel import LEXICAL_SWEEP_ALPHAS, lexical_sweep_claim_suffix
from ailm.levels.word import validate_stage3_miswirings
from ailm.statistics import DIFFERENCE_CONFIDENCE
from ailm.viz_config import FONTS, PALETTE, apply_style

__all__ = [
    "case_both_defects_data",
    "channel_comparison_data",
    "generate_case_both_defects",
    "generate_channel_comparison",
    "generate_generation_sweep",
    "generate_word_miswirings",
    "generation_sweep_data",
    "word_miswirings_data",
]

# The project palette carries semantics (see ailm.viz_config): ANTI is the
# mis-wiring color, MODEL the shipped agent, UNIGRAM the honest baseline. Reusing
# those roles keeps a reader's color associations intact across every figure
# rather than inventing a private scheme for these four panels.
COLOR_MODEL = PALETTE["MODEL"]
COLOR_BASELINE = PALETTE["UNIGRAM"]
COLOR_SURVIVES = PALETTE["ANTI"]
#: The mis-wiring the bar catches. Previously UNIFORM grey, which said two wrong
#: things at once: UNIFORM is the crippled *comparator*, not a mis-wiring, and
#: #999999 renders this panel's load-bearing verdict word at a contrast no reader
#: can use — the same failure decision 100 removed from the inference-evidence
#: notes. OUTLINE separates from ANTI by lightness, so the distinction survives
#: every form of color blindness, and it reads as ink rather than as a comparator.
COLOR_CAUGHT = PALETTE["OUTLINE"]
COLOR_CONTROL = PALETTE["ACCENT"]


def _styled() -> None:
    """Apply the project's shared matplotlib style before building a figure."""
    apply_style()


# ── Stage 3: the mis-wirings that clear the bar ───────────────────────────────


def word_miswirings_data(variables: dict[str, str] | None = None) -> dict[str, Any]:
    """Stage 3 mis-wirings and the bar each is scored against.

    Raises:
        ValueError: If the count of bar-clearing mis-wirings stops matching the
            three the project reports. That number is the panel's entire claim,
            so a change in it is a change in the finding, not a redraw.
    """
    v = _v(variables)
    bar = token_float(v, "STAGE3_BIGRAM_BASELINE")
    finite_arms = tuple(arm for arm in validate_stage3_miswirings() if arm.claim_label is not None)
    rows = [("shipped model", token_float(v, "STAGE3_WORD_PERPLEXITY"), COLOR_MODEL)]
    rows.extend(
        (
            arm.display_name,
            token_float(v, arm.claim_label.upper()),
            COLOR_CAUGHT if arm.key == "time_reversed" else COLOR_SURVIVES,
        )
        for arm in finite_arms
        if arm.claim_label is not None
    )
    survivors = sum(1 for name, value, _ in rows[1:] if value < bar)
    expected_survivors = token_int(v, "STAGE3_MISWIRINGS_CLEARING_BAR")
    if survivors != expected_survivors:
        raise ValueError(
            f"figure computes {survivors} bar-clearing mis-wirings but the generated "
            f"manuscript contract records {expected_survivors}; regenerate the token "
            "map or correct the registry/claims mismatch"
        )
    return {"bar": bar, "rows": rows, "survivors": survivors}


def generate_word_miswirings(variables: dict[str, str] | None = None) -> Figure:
    """Plot the finite Stage 3 mis-wirings against their accuracy bar."""
    _styled()
    data = word_miswirings_data(variables)
    bar = data["bar"]
    rows = data["rows"]

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    labels = [name for name, _, _ in rows]
    values = [value for _, value, _ in rows]
    colors = [color for _, _, color in rows]
    positions = range(len(rows))

    # A log axis, because the time-reversed emission is two orders of magnitude
    # worse than everything else. Clipping it to a linear axis would either hide
    # the one mis-wiring the bar DOES catch or expand the canvas to fit it -- and
    # the contrast between "caught" and "clears the bar" is the panel's content.
    ax.barh(list(positions), values, color=colors, height=0.62, log=True)
    ax.set_xscale("log")
    ax.set_xlim(1.0, max(values) * 3.0)
    ax.axvline(bar, color=PALETTE["OUTLINE"], linestyle="--", linewidth=1.8, zorder=3)
    # The threshold's own label sits in headroom above the top bar (the y-axis is
    # inverted below, so negative y is above row 0), ending just left of the
    # dashed line — not straddling the x-axis spine at the bottom.
    ax.text(
        bar * 0.94,
        -0.72,
        f"letter-bigram bar = {bar:.2f}  ",
        va="center",
        ha="right",
        fontsize=FONTS["cell"],
        color=PALETTE["OUTLINE"],
        fontweight="bold",
    )
    for index, (value, color) in enumerate(zip(values, colors, strict=True)):
        verdict = "clears the bar" if value < bar else "CAUGHT"
        # Bar-clearing values sit close enough to the threshold that a label
        # starting at the bar's end is struck through by the dashed line; start
        # those labels just right of the line instead.
        x_text = value * 1.12 if value >= bar else max(value * 1.12, bar * 1.12)
        ax.text(
            x_text,
            index,
            f"{value:.2f}   {verdict}",
            va="center",
            ha="left",
            fontsize=FONTS["cell"],
            color=color,
            fontweight="bold" if value >= bar else "normal",
        )

    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels, fontsize=FONTS["cell"])
    ax.invert_yaxis()
    # Extra headroom above row 0 for the threshold label placed at y = -0.72.
    ax.set_ylim(len(rows) - 0.5, -1.05)
    ax.set_xlabel("per-emission perplexity, log scale (lower is better)", fontsize=FONTS["cell"])
    ax.set_title(
        f"Stage 3: {data['survivors']} of {len(rows) - 1} mis-wirings BEAT the letter-bigram bar",
        fontsize=FONTS["subtitle"],
        fontweight="bold",
    )
    ax.text(
        0.5,
        -0.30,
        # "Everything left of the dashed line" read as a claim about the labels,
        # which are deliberately aligned in a column to the RIGHT of the line
        # because a label starting at its own bar's end is struck through by it.
        # The sentence is about the bars, so it now says so.
        "Every bar ending left of the dashed line clears it — including a prior whose frequency\n"
        "ranking is REVERSED, so the rarest words are predicted most confidently. The bar is a\n"
        "necessary condition, never a sufficient one; the closed-form path identity is what\n"
        "actually pins this level.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=FONTS["cell"],
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ── Stage 4: one threshold, two population-dependent verdicts ────────────────


def case_both_defects_data(variables: dict[str, str] | None = None) -> dict[str, Any]:
    """Stage 4's two populations against one declared threshold.

    Raises:
        ValueError: If the threshold stops straddling the two populations in
            opposite directions, which is the panel's whole claim.
    """
    v = _v(variables)
    threshold = token_float(v, "STAGE4_DECLARED_GATE")
    # The two straddle values are bound before the records are built. Read back
    # out of a heterogeneous dict they are typed ``object``, and the comparison
    # that carries the panel's whole claim would not be checked at all.
    gold_majority = token_float(v, "STAGE4_GOLD_MAJORITY")
    derived_model = token_float(v, "STAGE4_DERIVED_AGREEMENT")
    gold = {
        "name": "gold morphology",
        "n": token_int(v, "STAGE4_GOLD_TEST_TOKENS"),
        "majority": gold_majority,
        "baseline": token_float(v, "STAGE4_GOLD_BASELINE"),
        "model": token_float(v, "STAGE4_GOLD_AGREEMENT"),
        "verdict": "VACUOUS",
    }
    derived = {
        "name": "derived oracle",
        "n": token_int(v, "STAGE4_DERIVED_TEST_TOKENS"),
        "majority": token_float(v, "STAGE4_DERIVED_MAJORITY"),
        "baseline": token_float(v, "STAGE4_DERIVED_BASELINE"),
        "model": derived_model,
        "verdict": "ABOVE_REFERENCE",
    }
    if not (gold_majority >= threshold > derived_model):
        raise ValueError(
            f"The declared {threshold}% no longer sits above the derived model "
            f"({derived_model}%) and below the gold majority "
            f"({gold_majority}%). The two-defect finding has changed."
        )
    return {"threshold": threshold, "populations": [gold, derived]}


def generate_case_both_defects(variables: dict[str, str] | None = None) -> Figure:
    """Stage 4: the same threshold is vacuous on one population, above-reference on the other."""
    _styled()
    data = case_both_defects_data(variables)
    threshold = data["threshold"]

    fig, axes = plt.subplots(2, 1, figsize=(9.6, 4.6), sharex=True)
    for row, (ax, population) in enumerate(zip(axes, data["populations"], strict=True)):
        marks = [
            ("majority class", population["majority"], COLOR_BASELINE, "o"),
            ("context-free bar", population["baseline"], COLOR_CONTROL, "s"),
            ("model", population["model"], COLOR_MODEL, "D"),
        ]
        ax.axvspan(0, threshold, color=PALETTE["PANEL"], zorder=0)
        ax.axvline(threshold, color=PALETTE["GATE"], linewidth=2.0, zorder=4)
        # On the gold population the three markers COINCIDE, because the
        # observation alphabet has size 1 and every predictor collapses to the
        # prior's argmax. Drawing three overlapping labels would turn the panel's
        # sharpest finding into an unreadable smudge, so it is stated instead.
        values = [value for _, value, _, _ in marks]
        coincident = max(values) - min(values) < 0.01
        if coincident:
            ax.scatter(values[0], 0, s=150, color=COLOR_MODEL, marker="D", zorder=5)
            # Anchored on the gate line, not on the marker. Anchored on the
            # marker with a fixed 14pt lead-in, the block's right edge landed a
            # point and a half PAST the threshold, so the gate rule was drawn
            # through the words "63.80%", "observation" and "differ". Right
            # alignment against the rule itself cannot cross it at any threshold
            # or marker position the ledger later records.
            ax.annotate(
                f"majority = bar = model = {values[0]:.2f}%\n"
                f"all three IDENTICAL — the observation\n"
                f"alphabet has size 1, so nothing can differ",
                (threshold, 0),
                textcoords="offset points",
                xytext=(-10, 20),
                ha="right",
                fontsize=FONTS["cell"],
            )
        else:
            for offset, (label, value, color, marker) in zip((48, 30, 12), marks, strict=True):
                ax.scatter(
                    value,
                    0,
                    s=110,
                    color=color,
                    marker=marker,
                    zorder=5,
                    label=label,
                )
                ax.annotate(
                    f"{label}  {value:.2f}%",
                    (value, 0),
                    textcoords="offset points",
                    xytext=(0, offset),
                    ha="center",
                    fontsize=FONTS["cell"],
                )
        ax.set_yticks([])
        # Content-driven, not the old fixed (-0.5, 1.5): every marker sits at
        # y=0 and nothing is ever drawn below it, so a symmetric -0.5 floor was
        # pure dead space. -0.15 leaves headroom only for the marker glyph
        # itself. The top differs by row: row 0 carries the fixed "declared
        # gate" label at data y=1.20 (added once, after this loop, on
        # ``axes[0]``) and needs room above it; the other row's tallest content
        # is its own annotation stack, which tops out well under 1.0.
        ax.set_ylim(-0.15, 1.35 if row == 0 else 1.05)
        ax.set_title(
            f"{population['name']}  (n = {population['n']:,})   →   {population['verdict']}",
            fontsize=FONTS["tick_label"],
            fontweight="bold",
            loc="left",
        )
        ax.spines[["top", "right", "left"]].set_visible(False)

    axes[0].text(
        threshold,
        1.20,
        f"  declared gate: ≥{threshold:.0f}%",
        color=PALETTE["GATE"],
        fontsize=FONTS["cell"],
        fontweight="bold",
        va="center",
    )
    axes[-1].set_xlabel("agreement with the oracle labeling, %", fontsize=FONTS["cell"])
    axes[-1].set_xlim(0, 100)
    fig.suptitle(
        "Stage 4: one declared threshold, two opposite defects",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.045,
        "On gold morphology a constant majority predictor already clears the gate. On the derived\n"
        "oracle nothing at this model order reaches it. The only difference is which population it\n"
        "is scored against — which is why a threshold without a named population is not a threshold.",
        ha="center",
        fontsize=FONTS["cell"],
    )
    fig.tight_layout(rect=(0, 0.16, 1, 0.90))
    return fig


# ── Generation: the U-shaped precision sweep ──────────────────────────────────


def generation_sweep_data(variables: dict[str, str] | None = None) -> dict[str, Any]:
    """The three recorded points of the generation precision sweep.

    Raises:
        ValueError: If the optimum stops being interior.
    """
    v = _v(variables)
    low = (
        token_float(v, "GENERATION_LOW_GAMMA"),
        token_float(v, "GENERATION_LOW_GAMMA_TV"),
    )
    best = (
        token_float(v, "GENERATION_BEST_GAMMA"),
        token_float(v, "GENERATION_BEST_TV"),
    )
    high = (
        token_float(v, "GENERATION_DEGENERATE_GAMMA"),
        token_float(v, "GENERATION_DEGENERATE_TV"),
    )
    baseline = token_float(v, "GENERATION_BASELINE_TV")
    if not (best[1] < low[1] and best[1] < high[1]):
        raise ValueError(
            "The generation optimum is no longer interior; the U-shape claim has changed."
        )
    return {"low": low, "best": best, "high": high, "baseline": baseline}


def generate_generation_sweep(variables: dict[str, str] | None = None) -> Figure:
    """Generation: both extremes of policy precision generate badly."""
    _styled()
    data = generation_sweep_data(variables)
    points = [data["low"], data["best"], data["high"]]
    gammas = [p[0] for p in points]
    distances = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(gammas, distances, "-o", color=COLOR_MODEL, linewidth=2.0, markersize=8)
    ax.axhline(
        data["baseline"],
        color=COLOR_BASELINE,
        linestyle="--",
        linewidth=1.6,
    )
    # Lifted a few points off the dashed rule so the line does not strike the
    # text's baseline.
    ax.annotate(
        f"baseline: sampling from B  ({data['baseline']:.3f})",
        (gammas[1], data["baseline"]),
        textcoords="offset points",
        xytext=(0, 7),
        va="bottom",
        ha="center",
        fontsize=FONTS["cell"],
        color=COLOR_BASELINE,
    )
    ax.set_xscale("log")
    # Extra right-hand margin so the strong-goal callout clears the canvas edge.
    ax.set_xlim(gammas[0] * 0.7, gammas[2] * 2.0)
    # The low-γ label sits above and to the right of its point, where the
    # descending segment cannot cross it; the high-γ label drops far enough
    # that the steeply rising data segment passes above it rather than through.
    annotations = [
        ("goal too weak:\nreverts to the habit", (14, 16), "left"),
        ("shipped\noperating point", (0, -62), "center"),
        ("goal too strong:\ncollapses to one word", (-14, -90), "right"),
    ]
    for (gamma, distance), (note, offset, align) in zip(points, annotations, strict=True):
        ax.annotate(
            f"γ={gamma:g}   {distance:.3f}\n{note}",
            (gamma, distance),
            textcoords="offset points",
            xytext=offset,
            ha=align,
            fontsize=FONTS["cell"],
        )
    ax.set_xlabel("policy precision γ  (log scale)", fontsize=FONTS["cell"])
    ax.set_ylabel(
        "distance to the true word\ndistribution (lower is better)", fontsize=FONTS["cell"]
    )
    ax.set_title(
        "Generation: the optimum is INTERIOR — both extremes fail",
        fontsize=FONTS["subtitle"],
        fontweight="bold",
    )
    # Bottom headroom so the "shipped\noperating point" callout clears the axis.
    ax.set_ylim(0.08, max(distances + [data["baseline"]]) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ── The composition result: two channels, one receiver ────────────────────────


#: The recorded lexical sweep grid: ledger token suffix per mixing weight. The
#: alpha values are the sweep's design points (structural, not measured); every
#: delta drawn against them is a recorded claim.
_LEXICAL_SWEEP_GRID: tuple[tuple[float, str], ...] = tuple(
    (alpha, lexical_sweep_claim_suffix(alpha)) for alpha in LEXICAL_SWEEP_ALPHAS
)


def channel_comparison_data(variables: dict[str, str] | None = None) -> dict[str, Any]:
    """The two inter-level channels into L2B, with their opposite shapes — and
    the pre-registered shape verdict recomputed from the recorded sweeps.

    The pre-registration (the working-hypothesis box in
    ``docs/manuscript/09_hierarchy.md``) named a shape class per channel before the
    measurement: an *interior optimum* for the novel lexical message, *no
    interior gain* for the redundant letter predictive. Both verdicts are
    re-derived here from the recorded claims at draw time, so the overlay in
    :func:`generate_channel_comparison` physically cannot survive data that
    breaks the prediction.

    Raises:
        ValueError: If the two channels stop having opposite signs (the
            controlled comparison the panel exists to show); if the argmax of
            the recorded lexical sweep is no longer strictly interior to the
            recorded alpha grid, or disagrees with the recorded best alpha
            (tie/plateau rule: *every* delta within rounding tolerance of the
            maximum must sit at an interior alpha, so a plateau reaching a
            boundary refutes "interior" rather than slipping past on argmax
            order); or if any recorded coupling anchor improves on the
            uncoupled alpha=0 anchor, which would refute the pre-registered
            "no interior gain" for the redundant channel. The coupling check
            deliberately uses only the three recorded anchors — the curve
            between them is schematic and can witness nothing.
    """
    v = _v(variables)
    harmful = token_float(v, "COUPLING_MARGIN_PTS")
    helpful = token_float(v, "LEXICAL_BEST_MARGIN_PTS")
    if not (harmful < 0 < helpful):
        raise ValueError(
            f"The two channels no longer have opposite signs (coupling "
            f"{harmful}, lexical {helpful}). The controlled comparison in "
            f"docs/decisions.md entry 19 has changed."
        )

    alphas = [alpha for alpha, _ in _LEXICAL_SWEEP_GRID]
    deltas = [
        token_float(v, f"LEXICAL_SWEEP_ALPHA_{suffix}_DELTA") for _, suffix in _LEXICAL_SWEEP_GRID
    ]
    best_alpha = token_float(v, "LEXICAL_BEST_ALPHA")

    # ── Pre-registered lexical verdict: interior argmax over the recorded grid.
    # Tie/plateau rule: the recorded deltas carry ~3 decimals, so any delta
    # within half a unit in the last recorded decimal of the maximum counts as
    # attaining it; "interior" requires every attaining alpha to be interior.
    peak = max(deltas)
    tolerance = 5e-4
    attaining = [index for index, delta in enumerate(deltas) if peak - delta <= tolerance]
    if any(index in (0, len(deltas) - 1) for index in attaining):
        raise ValueError(
            f"The recorded lexical sweep no longer has a strictly interior optimum "
            f"(maximum {peak} attained at alpha "
            f"{[alphas[i] for i in attaining]}). The pre-registered 'interior "
            f"optimum' band would misstate the data; refusing to draw."
        )
    if best_alpha not in {alphas[index] for index in attaining}:
        raise ValueError(
            f"The recorded best alpha {best_alpha} is not where the recorded sweep "
            f"peaks (alpha {[alphas[i] for i in attaining]}). The ledger's sweep "
            f"and best-point claims have diverged; re-run the recorder."
        )
    if abs(deltas[alphas.index(best_alpha)] - helpful) > 5e-4:
        raise ValueError(
            f"The sweep delta at the best alpha ({deltas[alphas.index(best_alpha)]}) "
            f"disagrees with the recorded best margin ({helpful})."
        )

    # ── Pre-registered coupling verdict, from the three recorded anchors only:
    # no recorded alpha improves on the uncoupled alpha=0 anchor.
    anchor_accuracy = token_float(v, "COUPLING_ALPHA0_ACCURACY")
    coupled_accuracies = {
        token_float(v, "COUPLING_BEST_ALPHA"): token_float(v, "COUPLING_BEST_ACCURACY"),
        1.0: token_float(v, "COUPLING_ALPHA1_ACCURACY"),
    }
    improving = {a: acc for a, acc in coupled_accuracies.items() if acc > anchor_accuracy}
    if improving:
        raise ValueError(
            f"A recorded coupling anchor improves on the uncoupled anchor "
            f"({anchor_accuracy}): {improving}. The pre-registered 'no interior "
            f"gain' band for the redundant channel is refuted; refusing to draw."
        )

    # ── The uncertainty on every drawn point ────────────────────────────────
    # The sweep was drawn as a bare polyline: seven point estimates and no
    # statement of how precisely any of them is known, on the figure whose whole
    # claim is about the SHAPE those points trace. Every point already has its
    # discordant counts in the ledger, so the interval needs no new measurement,
    # only the recorder's own function at draw time — the same wrapping
    # ``paired_difference_bound_from_counts`` performs for the arms that get a
    # rendered bound. Re-deriving the point estimate from the same counts and
    # requiring it to match the recorded delta is what stands between this and a
    # plausible wrong denominator, which the counts alone cannot reveal.
    positions = token_int(v, "LEXICAL_EVAL_POSITIONS")
    lows: list[float] = []
    highs: list[float] = []
    discordants: list[int] = []
    p_values: list[float] = []
    for alpha, suffix in _LEXICAL_SWEEP_GRID:
        b = token_int(v, f"LEXICAL_SWEEP_ALPHA_{suffix}_MCNEMAR_B")
        c = token_int(v, f"LEXICAL_SWEEP_ALPHA_{suffix}_MCNEMAR_C")
        if b + c > positions:
            raise ValueError(
                f"alpha {alpha}: {b + c} discordant pairs over {positions} evaluated "
                "positions is impossible; the sweep counts and the position claim "
                "came from different runs."
            )
        recomputed_delta_pts = 100.0 * (b - c) / positions
        recorded_delta = deltas[alphas.index(alpha)]
        if abs(recomputed_delta_pts - recorded_delta) > 5e-4:
            raise ValueError(
                f"alpha {alpha}: the delta recomputed from the recorded counts "
                f"({b}, {c}) over {positions} positions is {recomputed_delta_pts}, "
                f"but the ledger records {recorded_delta}. Either the denominator is "
                "wrong or the counts and the delta came from different runs; refusing "
                "to draw an interval around a point the counts do not reproduce."
            )
        low, high = paired_difference_interval(b, c, positions, confidence=DIFFERENCE_CONFIDENCE)
        lows.append(100.0 * low)
        highs.append(100.0 * high)
        discordants.append(b + c)
        p_values.append(token_float(v, f"LEXICAL_SWEEP_ALPHA_{suffix}_MCNEMAR_P"))

    return {
        "lexical": {
            "alphas": alphas,
            "deltas": deltas,
            "lows": lows,
            "highs": highs,
            "discordants": discordants,
            "p_values": p_values,
            "best_alpha": best_alpha,
        },
        "positions": positions,
        "confidence": DIFFERENCE_CONFIDENCE,
        "harmful_margin": harmful,
        "harmful_alpha": token_float(v, "COUPLING_BEST_ALPHA"),
        "verdicts": {
            "lexical": "interior optimum",
            "coupling": "no recorded alpha improves on the anchor",
        },
    }


def generate_channel_comparison(variables: dict[str, str] | None = None) -> Figure:
    """The composition result: novelty predicts the SHAPE, not just the sign.

    The pre-registered shape class for each channel is drawn as a faint band
    behind the measured curves. The bands are not decoration: the data function
    recomputes both verdicts from the recorded claims and refuses to draw if
    either contradicts its band, so a drawn band is a prediction the recorded
    data still supports.
    """
    _styled()
    data = channel_comparison_data(variables)
    lexical = data["lexical"]

    # Taller in proportion to the y range the interval endpoints added. Keeping
    # the old height compressed the two annotations in the harmful band until
    # they overlapped, which tests/test_figure_layout.py caught and
    # assert_within_canvas cannot see: both labels stayed on the canvas.
    fig, ax = plt.subplots(figsize=(9.6, 6.3))
    ax.axhline(0.0, color=PALETTE["OUTLINE"], linewidth=1.2)
    # The interval on every point, drawn before the line so the markers stay on
    # top. A shape argument made from point estimates alone asks the reader to
    # trust a curve whose precision it never states.
    ax.errorbar(
        lexical["alphas"],
        lexical["deltas"],
        yerr=[
            [d - lo for d, lo in zip(lexical["deltas"], lexical["lows"], strict=True)],
            [hi - d for d, hi in zip(lexical["deltas"], lexical["highs"], strict=True)],
        ],
        fmt="none",
        ecolor=COLOR_MODEL,
        elinewidth=1.6,
        capsize=4,
        zorder=3,
    )
    ax.plot(
        lexical["alphas"],
        lexical["deltas"],
        "-o",
        color=COLOR_MODEL,
        linewidth=2.2,
        markersize=7,
        zorder=4,
        label="L3 → L2B  (lexical prefix: NOVEL to the receiver)",
    )
    # The harmful channel is drawn only across the range where it has recorded
    # anchors. It previously ran to alpha=1 as a schematic power curve whose
    # endpoint was the alpha=0.1 measurement placed at x=1, so the caption's
    # promise that "only its endpoints are measurements" was false and the drawn
    # harm was understated by more than two orders of magnitude: the recorded
    # alpha=1 accuracy collapse is about -53.9 points against the -0.26 the curve
    # showed. Extrapolating a shape between measurements is what put a number on
    # the page that no run produced, so the series now stops where the data does
    # and the alpha=1 collapse is reported in the prose from its own tokens.
    end = data["harmful_margin"]
    harmful_alphas = [0.0, data["harmful_alpha"]]
    harmful_deltas = [0.0, end]
    ax.plot(
        harmful_alphas,
        harmful_deltas,
        "--s",
        color=COLOR_SURVIVES,
        linewidth=2.0,
        markersize=8,
        markerfacecolor="none",
        markeredgewidth=1.5,
        zorder=3,
        label="L1 → L2B  (letter predictive: REDUNDANT — receiver already sees it)",
    )
    # Both channels share the exact alpha=0 anchor. Offset only the redundant
    # marker for visibility and label that cosmetic offset explicitly; the
    # dashed series itself still starts at the measured coordinate (0, 0).
    ax.scatter(
        [0.035],
        [0.0],
        marker="s",
        s=72,
        facecolors="none",
        edgecolors=COLOR_SURVIVES,
        linewidths=1.8,
        zorder=6,
    )
    ax.annotate(
        "shared α=0 anchor\n(square offset for visibility)",
        (0.035, 0.0),
        textcoords="offset points",
        xytext=(22, 10),
        fontsize=FONTS["cell"],
        color=COLOR_SURVIVES,
        ha="left",
    )

    # Dropped clear of the descending limb. At -30pt the block sat in the band
    # the curve falls through between alpha 0.3 and 0.6, so the series ran
    # straight through the words "interior optimum". Nothing is drawn in the
    # band below it before the curve crosses zero near alpha 0.85.
    ax.annotate(
        f"observed: interior argmax\nα={lexical['best_alpha']:g} (ordering, not a\ndistinguished optimum)",
        (lexical["best_alpha"], max(lexical["deltas"])),
        textcoords="offset points",
        xytext=(14, -66),
        fontsize=FONTS["cell"],
        fontweight="bold",
    )
    # Anchored on the series' own last drawn point. It previously sat at
    # (0.75, end * 0.75) — a coordinate on the extrapolated curve that no longer
    # exists, leaving the label floating in empty space next to a different
    # series, where a reader would attach the claim to the wrong channel.
    ax.annotate(
        "observed: monotone decline —\nno weight rescues it",
        (data["harmful_alpha"], end),
        textcoords="offset points",
        xytext=(10, -10),
        fontsize=FONTS["cell"],
        color=COLOR_SURVIVES,
        ha="left",
        arrowprops={"arrowstyle": "-", "color": COLOR_SURVIVES, "linewidth": 1.0},
    )

    ax.set_xlabel("mixing weight α", fontsize=FONTS["cell"])
    ax.set_ylabel(
        "Δ L2B accuracy vs the uncoupled\nanchor (percentage points)\n"
        f"whiskers: exact conditional {data['confidence'] * 100:g}% paired\n"
        f"interval over {data['positions']:,} held-out positions",
        fontsize=FONTS["cell"],
    )
    ax.set_title(
        "Same receiving level, different sender: the SHAPE diagnoses novelty",
        fontsize=FONTS["subtitle"],
        fontweight="bold",
    )
    # The y range covers the series minimum with ticks on the negative stretch —
    # the lexical curve's last point would otherwise sit clipped at the floor —
    # and leaves headroom at the top for the legend, which the dashed harmful
    # series ran through at its old lower-left position.
    # The extremes are now the interval endpoints, not the point estimates. Taking
    # them from the points alone left the top whisker inside the legend, where a
    # drawn mark ran through the words explaining it — invisible to a
    # text-versus-text sweep, and the reason the headroom is derived rather than
    # tuned.
    y_min = min(lexical["lows"] + harmful_deltas)
    y_max = max(lexical["highs"] + harmful_deltas)
    ax.set_ylim(y_min - 0.62, y_max + 1.05)

    # The pre-registration, drawn as faint bands BEHIND the measured curves —
    # gain territory for the channel predicted to peak inside the grid, decline
    # territory for the one predicted never to gain. channel_comparison_data
    # recomputed both verdicts from the recorded claims before this point, so
    # these bands cannot be drawn over data that refutes them.
    ax.axhspan(0.0, y_max + 1.05, color=COLOR_MODEL, alpha=0.045, zorder=0)
    ax.axhspan(y_min - 0.62, 0.0, color=COLOR_SURVIVES, alpha=0.045, zorder=0)
    ax.text(
        0.02,
        y_max + 0.14,
        "pre-registered: interior optimum (novel channel)",
        fontsize=FONTS["cell"],
        fontstyle="italic",
        color=COLOR_MODEL,
        alpha=0.8,
        ha="left",
    )
    ax.text(
        0.02,
        y_min - 0.38,
        "pre-registered: no interior gain (redundant channel; curve between anchors is schematic)",
        fontsize=FONTS["cell"],
        fontstyle="italic",
        color=COLOR_SURVIVES,
        alpha=0.8,
        ha="left",
    )
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.legend(fontsize=FONTS["cell"], loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.5,
        -0.24,
        "A monotone decline says the message is redundant, and no amount of tuning will rescue\n"
        "it. A sample argmax inside the grid is consistent with novel information but, on this\n"
        "sweep, is not distinguished from a plateau (decision 110). Both channels enter the same\n"
        "level; only novelty differs — and it predicted the sign in advance.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=FONTS["cell"],
    )
    fig.tight_layout()
    return fig
