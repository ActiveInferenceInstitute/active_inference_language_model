"""Matrix figures: the letter transition B, and the wiring comparison.

``fig:letter_transition``
    The 30x30 Brown letter bigram matrix that parameterizes ``B`` at Level 1,
    built through :func:`ailm.levels.letter.build_letter_transition` on the vendored
    counts — the same call the agent uses, so the figure cannot show a matrix the
    model does not have. Axes are labeled with the actual ``ALPHABET`` symbols
    and a handful of high-probability cells are marked, because the structure that
    makes this matrix informative (``q`` almost always followed by ``u``, ``t`` by
    ``h``) is the structure that vanishes when the matrix is put in the wrong slot.

``fig:wiring_comparison``
    The load-bearing figure. Correct wiring (``A`` identity, ``B`` bigram) beside
    the anti-informative wiring (``A`` bigram, ``B`` identity), with the measured
    perplexity of each. Both matrices are column-stochastic, both pass pymdp's
    ``_validate()``, and the panels look equally plausible — which is the entire
    problem. The bar chart underneath places the mis-wiring against the honest
    comparators and shows that it lands *worse than the context-free unigram
    baseline*, i.e. worse than using no context at all.

Every number that appears in these figures is either computed from the vendored
corpus artifact at generation time or read from
``output/data/manuscript_variables.json``. None is written into this module. A
perplexity typed into plotting code would be exactly as fabricated as one typed
into prose, and considerably harder to notice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from ailm.figures._common import (
    MissingVariableError,
    load_manuscript_variables,
    save_figure,
)
from ailm.levels.letter import (
    ALPHA_LIST,
    ALPHA_TO_IDX,
    N_SYMBOLS,
    build_letter_transition,
    load_brown_counts,
)
from ailm.viz_config import FONTS, PALETTE, SEQUENTIAL_CMAP, apply_style

__all__ = [
    "MARKED_BIGRAMS",
    "generate_letter_transition_figure",
    "generate_phoneme_matrices_figure",
    "generate_wiring_comparison_figure",
    "letter_transition_matrix",
    "write_matrix_figures",
]

#: Bigrams marked on the transition heatmap. A fixed, sorted-by-nothing literal
#: list rather than "the top k cells": the point is to label the constraints a
#: reader can check against their own knowledge of English orthography, and a
#: top-k selection would silently change identity when the corpus changes.
MARKED_BIGRAMS: tuple[tuple[str, str], ...] = (
    ("q", "u"),
    ("t", "h"),
    ("h", "e"),
    ("e", " "),
    (" ", "t"),
)

#: Floor of the shared logarithmic color scale. Chosen below the smallest
#: add-1-smoothed probability the Brown matrix contains, so the bigram panels show
#: a continuous gradient rather than a scatter of white "under" cells. Structural
#: zeros — the off-diagonal of an identity matrix — still fall below it and render
#: white, which is what makes an identity panel read as a bare diagonal.
_PROB_VMIN: float = 1e-6


def _display(symbol: str) -> str:
    """Render an ALPHABET symbol as a visible tick label.

    Args:
        symbol: A single ALPHABET character.

    Returns:
        The symbol, with the space character replaced by U+2423 OPEN BOX so a
        blank tick is distinguishable from a missing one.
    """
    return "␣" if symbol == " " else symbol


def letter_transition_matrix(vendor_dir: Path | None = None) -> np.ndarray:
    """Load the Brown letter bigram matrix as a plain ``(30, 30)`` numpy array.

    Goes through :func:`ailm.levels.letter.build_letter_transition`, the same constructor
    :func:`ailm.levels.letter.build_letter_agent` consumes, so the figure and the
    agent cannot disagree.

    Args:
        vendor_dir: Directory holding ``brown_counts.json``. Defaults to the
            project's ``vendor/``.

    Returns:
        ``matrix[i, j] = P(next letter = i | current letter = j)``, column-
        stochastic down axis 0.
    """
    counts = load_brown_counts(vendor_dir)
    return np.asarray(build_letter_transition(counts))[0].astype(np.float64)


def _token_text(tokens: dict[str, str], name: str) -> str:
    """Return a token's value verbatim, or fail with a directive message.

    Verbatim matters: the manuscript prints ``11.9215``, so the figure must print
    ``11.9215`` and not a locally reformatted ``11.92``. Two spellings of one
    measurement in one paper is a defect a reader cannot resolve.

    Args:
        tokens: Manuscript token map.
        name: UPPERCASE token name.

    Returns:
        The recorded string value.

    Raises:
        MissingVariableError: If the token is absent.
    """
    if name not in tokens:
        raise MissingVariableError(
            f"Token {name!r} is not in the manuscript variable map, so this "
            "figure cannot show it. Record the claim from the run that measured "
            "it — do not put the number in the figure code."
        )
    return tokens[name]


def _probability_cmap():
    """Return the sequential colormap with an explicit ``under`` color.

    Cells below the log-scale floor render white rather than clamping to the
    darkest hue, so "structurally absent" is visually distinct from "very small".
    """
    return plt.get_cmap(SEQUENTIAL_CMAP).with_extremes(under="white")


def generate_letter_transition_figure(vendor_dir: Path | None = None) -> Figure:
    """Build ``fig:letter_transition`` — the 30x30 Brown letter bigram B.

    Args:
        vendor_dir: Directory holding ``brown_counts.json``. Defaults to the
            project's ``vendor/``.

    Returns:
        A matplotlib Figure. The caller owns it.
    """
    apply_style()
    matrix = letter_transition_matrix(vendor_dir)

    fig = plt.figure(figsize=(13.0, 9.0))
    ax = fig.add_axes((0.052, 0.095, 0.500, 0.810))
    cax = fig.add_axes((0.572, 0.095, 0.014, 0.810))

    image = ax.imshow(
        matrix,
        cmap=_probability_cmap(),
        norm=LogNorm(vmin=_PROB_VMIN, vmax=1.0),
        origin="upper",
        aspect="equal",
        interpolation="nearest",
    )

    labels = [_display(symbol) for symbol in ALPHA_LIST]
    ax.set_xticks(range(N_SYMBOLS))
    ax.set_yticks(range(N_SYMBOLS))
    ax.set_xticklabels(labels, fontsize=FONTS["tick_label"])
    ax.set_yticklabels(labels, fontsize=FONTS["tick_label"])
    ax.set_xlabel("current letter — state $j$", fontsize=FONTS["axis_label"], labelpad=8)
    ax.set_ylabel("next letter — state $i$", fontsize=FONTS["axis_label"], labelpad=8)
    ax.set_title(
        f"Level 1 transition $B$: the {N_SYMBOLS}x{N_SYMBOLS} Brown letter bigram",
        fontsize=FONTS["title"],
        pad=14,
    )
    ax.tick_params(length=3, width=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(PALETTE["MUTED"])

    bar = fig.colorbar(image, cax=cax, extend="min")
    bar.set_label(
        "$P(\\mathrm{next} \\mid \\mathrm{current})$, log scale",
        fontsize=FONTS["axis_label"],
        labelpad=6,
    )
    bar.ax.tick_params(labelsize=FONTS["tick_label"])

    # ── Mark a few cells a reader can check by eye ────────────────────────────
    marks: list[tuple[int, str, str, float]] = []
    for order, (current, following) in enumerate(MARKED_BIGRAMS, start=1):
        j, i = ALPHA_TO_IDX[current], ALPHA_TO_IDX[following]
        probability = float(matrix[i, j])
        marks.append((order, current, following, probability))
        ax.add_patch(
            Rectangle(
                (j - 0.5, i - 0.5),
                1.0,
                1.0,
                facecolor="none",
                edgecolor=PALETTE["ACCENT"],
                linewidth=1.8,
                zorder=4,
            )
        )
        ax.text(
            j,
            i,
            str(order),
            fontsize=FONTS["cell"],
            fontweight="bold",
            va="center",
            ha="center",
            color=PALETTE["ACCENT"],
            zorder=5,
        )

    # ── Right-hand key, values read from the matrix ───────────────────────────
    note_x = 0.672
    fig.text(
        note_x,
        0.905,
        "Marked cells",
        fontsize=FONTS["title"],
        fontweight="bold",
        va="top",
        ha="left",
        color=PALETTE["INK"],
    )
    lines = [
        f"{order}   {_display(current)} → {_display(following)}      P = {probability:.3f}"
        for order, current, following, probability in marks
    ]
    fig.text(
        note_x,
        0.855,
        "\n".join(lines),
        fontsize=FONTS["annotation"],
        va="top",
        ha="left",
        color=PALETTE["INK"],
        linespacing=1.9,
        family="monospace",
    )

    peak_i, peak_j = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
    fig.text(
        note_x,
        0.630,
        "Strongest transition in the matrix\n"
        f"    {_display(ALPHA_LIST[peak_j])} → {_display(ALPHA_LIST[peak_i])}"
        f"      P = {matrix[peak_i, peak_j]:.3f}",
        fontsize=FONTS["annotation"],
        va="top",
        ha="left",
        color=PALETTE["ACCENT"],
        linespacing=1.9,
        family="monospace",
    )

    # The white-cell note is derived, not asserted: whether any cell falls under
    # the color floor depends on the corpus, and a stale "nothing renders white"
    # sentence beside a white cell is exactly the kind of caption drift the
    # claims discipline exists to prevent.
    under = np.argwhere(matrix < _PROB_VMIN)
    if len(under) == 0:
        floor_note = (
            "Add-1 Laplace smoothing, so no cell is exactly\n"
            "zero, and every cell sits above the color floor."
        )
    else:
        named = ", ".join(
            f"{_display(ALPHA_LIST[j])} → {_display(ALPHA_LIST[i])}" for i, j in under[:3]
        )
        plural = "" if len(under) == 1 else "s"
        verb_s = "s" if len(under) == 1 else ""
        floor_note = (
            "Add-1 Laplace smoothing,\n"
            "so no cell is exactly zero.\n"
            f"{len(under)} cell{plural} ({named})\n"
            f"still fall{verb_s} below the color floor\n"
            f"and render{verb_s} white:\n"
            "the corpus holds no such bigram,\n"
            "so the value is pure smoothing mass."
        )

    fig.text(
        note_x,
        0.545,
        "Orientation\n"
        "B[0, i, j, 0] = P(next = i | current = j).\n"
        "Every column j sums to 1, which is the\n"
        "axis pymdp validates. The same matrix is\n"
        "column-stochastic when placed in A, so\n"
        "normalization cannot tell the two wirings\n"
        "apart — see the wiring-comparison figure.\n"
        "\n"
        f"{floor_note}\n"
        "\n"
        "The space symbol is drawn ␣ throughout.",
        fontsize=FONTS["annotation"],
        va="top",
        ha="left",
        color=PALETTE["INK"],
        linespacing=1.5,
    )

    fig.text(
        0.055,
        0.012,
        "Built by ailm.levels.letter.build_letter_transition on vendor/brown_counts.json "
        "— the same constructor the Level 1 agent consumes.",
        fontsize=FONTS["cell"],
        va="bottom",
        ha="left",
        color=PALETTE["MUTED"],
    )

    return fig


# ── Wiring comparison ────────────────────────────────────────────────────────

#: Comparators drawn in the bar panel: (token, label, palette key, hatch).
#: Ordered worst-to-best so the bars read downward toward the information floor.
_COMPARATORS: tuple[tuple[str, str, str, str], ...] = (
    (
        "STAGE1_UNIFORM_BASELINE",
        f"uniform over {N_SYMBOLS} symbols",
        "UNIFORM",
        "",
    ),
    (
        "STAGE1_ANTI_INFORMATIVE_PERPLEXITY",
        "anti-informative: bigram in $A$",
        "ANTI",
        "//",
    ),
    (
        "STAGE1_UNIGRAM_BASELINE",
        "unigram letter frequencies",
        "UNIGRAM",
        "",
    ),
    (
        "STAGE1_LETTER_PERPLEXITY",
        "correct: identity $A$, bigram $B$",
        "MODEL",
        "",
    ),
    (
        "STAGE1_BIGRAM_FLOOR_PERPLEXITY",
        "empirical order-1 optimum",
        "FLOOR",
        "..",
    ),
)


def _mini_matrix(
    fig: Figure,
    rect: tuple[float, float, float, float],
    matrix: np.ndarray,
    title: str,
    accent: str,
) -> None:
    """Draw one small matrix panel with sparse symbol ticks."""
    ax = fig.add_axes(rect)
    ax.imshow(
        matrix,
        cmap=_probability_cmap(),
        norm=LogNorm(vmin=_PROB_VMIN, vmax=1.0),
        origin="upper",
        aspect="equal",
        interpolation="nearest",
    )
    ticks = list(range(0, N_SYMBOLS, 6))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([_display(ALPHA_LIST[t]) for t in ticks], fontsize=FONTS["cell"])
    ax.set_yticklabels([_display(ALPHA_LIST[t]) for t in ticks], fontsize=FONTS["cell"])
    ax.set_title(title, fontsize=FONTS["axis_label"], pad=8, color=accent)
    ax.tick_params(length=2.5, width=0.7)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(accent)
        spine.set_linewidth(1.6)


@dataclass(frozen=True)
class _WiringGroup:
    """One wiring column of ``fig:wiring_comparison``.

    A dict literal here erases the field types to ``object`` — the panel
    geometry, the accent color, and the matrix pairs have nothing in common —
    so every use site had to re-assert them with ``float()``/``str()`` and the
    type checker could not tell a coordinate from a heading. The fields are
    declared once instead.
    """

    center: float
    x0: float
    heading: str
    accent: str
    panels: tuple[tuple[str, np.ndarray], ...]
    reading: str
    token: str


def generate_wiring_comparison_figure(
    variables: dict[str, str] | None = None,
    vendor_dir: Path | None = None,
) -> Figure:
    """Build ``fig:wiring_comparison`` — correct versus anti-informative wiring.

    Args:
        variables: Manuscript token map. Defaults to loading
            ``output/data/manuscript_variables.json``. Every perplexity shown is
            read from here; none is written into this module.
        vendor_dir: Directory holding ``brown_counts.json``.

    Returns:
        A matplotlib Figure. The caller owns it.

    Raises:
        MissingVariableError: If the token map is absent or lacks a comparator.
    """
    apply_style()
    tokens = load_manuscript_variables() if variables is None else dict(variables)

    bigram = letter_transition_matrix(vendor_dir)
    identity = np.eye(N_SYMBOLS, dtype=np.float64)

    # 10.6 in tall, not 9.6: the "What the bars say" block is 22 lines of fixed
    # point-size type, so its height in pixels does not shrink with the canvas. At
    # 9.6 in it overran the bottom edge by 129 px and was cropped. Type size is a
    # build-time invariant here, so the figure gets more room rather than smaller
    # text. Everything else is placed in figure fractions and simply gains space.
    fig = plt.figure(figsize=(15.0, 10.6))

    ax_title_y = 0.982
    fig.text(
        0.030,
        ax_title_y,
        "The same matrix, two slots: wiring orientation is load-bearing and pymdp cannot check it",
        fontsize=FONTS["suptitle"],
        fontweight="bold",
        va="top",
        ha="left",
        color=PALETTE["INK"],
    )

    panel_y, panel_h, panel_w = 0.600, 0.250, 0.170
    groups = (
        _WiringGroup(
            center=0.245,
            x0=0.048,
            heading="CORRECT — the shipped Level 1 agent",
            accent=PALETTE["MODEL"],
            panels=(
                ("$A$ = identity  (emission)", identity),
                ("$B$ = Brown bigram  (transition)", bigram),
            ),
            reading=(
                "Observing a letter reveals the state exactly;\n$B$ then predicts the NEXT letter."
            ),
            token="STAGE1_LETTER_PERPLEXITY",
        ),
        _WiringGroup(
            center=0.745,
            x0=0.548,
            heading="ANTI-INFORMATIVE — what Stage 1 first shipped",
            accent=PALETTE["ANTI"],
            panels=(
                ("$A$ = Brown bigram  (emission)", bigram),
                ("$B$ = identity  (transition)", identity),
            ),
            reading=(
                "Observing a letter yields a posterior over the letter\n"
                "that PRECEDED it; identity $B$ carries that forward."
            ),
            token="STAGE1_ANTI_INFORMATIVE_PERPLEXITY",
        ),
    )

    for group in groups:
        accent = group.accent
        fig.text(
            group.center,
            0.930,
            group.heading,
            fontsize=FONTS["title"],
            fontweight="bold",
            va="center",
            ha="center",
            color=accent,
        )
        for slot, (label, matrix) in enumerate(group.panels):
            x0 = group.x0 + slot * (panel_w + 0.054)
            _mini_matrix(fig, (x0, panel_y, panel_w, panel_h), matrix, label, accent)
        fig.text(
            group.center,
            0.562,
            group.reading,
            fontsize=FONTS["annotation"],
            va="top",
            ha="center",
            color=PALETTE["INK"],
            linespacing=1.5,
        )
        fig.text(
            group.center,
            0.472,
            f"measured perplexity   {_token_text(tokens, group.token)}",
            fontsize=FONTS["title"],
            fontweight="bold",
            va="center",
            ha="center",
            color=accent,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": PALETTE["BACKGROUND"],
                "edgecolor": accent,
                "linewidth": 1.6,
            },
        )

    # Divider between the two wirings.
    fig.add_artist(
        plt.Line2D(
            (0.495, 0.495),
            (0.440, 0.915),
            color=PALETTE["MUTED"],
            linewidth=1.0,
            linestyle=":",
        )
    )

    # Shared color scale for all four panels.
    # x is 0.940, not 0.958: the colorbar's label sits outboard of its tick
    # labels, so at 0.958 the rotated "probability (log scale)" ran 27 px past the
    # canvas and was cropped out of the written PNG.
    cax = fig.add_axes((0.940, panel_y, 0.012, panel_h))
    bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=LogNorm(vmin=_PROB_VMIN, vmax=1.0), cmap=_probability_cmap()),
        cax=cax,
        extend="min",
    )
    bar.set_label("probability (log scale)", fontsize=FONTS["annotation"])
    bar.ax.tick_params(labelsize=FONTS["cell"])

    # ── Comparator bars ──────────────────────────────────────────────────────
    ax = fig.add_axes((0.520, 0.100, 0.430, 0.280))
    shown = [_token_text(tokens, token) for token, _l, _c, _h in _COMPARATORS]
    values = [float(text.replace(",", "")) for text in shown]
    labels = [label for _t, label, _c, _h in _COMPARATORS]
    colors = [PALETTE[key] for _t, _l, key, _h in _COMPARATORS]
    hatches = [hatch for _t, _l, _c, hatch in _COMPARATORS]

    positions = list(range(len(values)))[::-1]
    bars = ax.barh(
        positions,
        values,
        height=0.62,
        color=colors,
        edgecolor=PALETTE["INK"],
        linewidth=1.0,
    )
    for patch, hatch in zip(bars, hatches, strict=True):
        if hatch:
            patch.set_hatch(hatch)

    for position, value, text in zip(positions, values, shown, strict=True):
        ax.text(
            value + 0.35,
            position,
            text,
            fontsize=FONTS["annotation"],
            fontweight="bold",
            va="center",
            ha="left",
            color=PALETTE["INK"],
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=FONTS["annotation"])
    ax.set_xlim(0.0, max(values) * 1.22)
    ax.set_xlabel(
        "per-letter perplexity on held-out prose — lower is better",
        fontsize=FONTS["axis_label"],
        labelpad=8,
    )
    ax.set_title(
        "Both wirings validate; only one of them is informative",
        fontsize=FONTS["title"],
        pad=10,
        loc="left",
    )
    ax.grid(axis="x", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)

    # ── The reading of the bar panel, stated in words ────────────────────────
    # Heading at 0.430 and body at 0.402 (was 0.390/0.345): the body is 792 px of
    # fixed-point-size type, so its top must sit at least that far above the
    # bottom edge. On the 10.6 in canvas 0.402 leaves ~60 px of clearance.
    fig.text(
        0.030,
        0.430,
        "What the bars say",
        fontsize=FONTS["title"],
        fontweight="bold",
        va="top",
        ha="left",
        color=PALETTE["INK"],
    )
    fig.text(
        0.030,
        0.402,
        "The mis-wiring is not loudly broken. It sits\n"
        "between uniform and the context-free unigram\n"
        "bar — worse than using no context at all, yet\n"
        "comfortably under a naive perplexity threshold\n"
        "stated against uniform. On a small in-sample\n"
        "corpus it scored worse than uniform outright.\n"
        "\n"
        "pymdp raises nothing either way: _validate()\n"
        "checks normalization, and a bigram matrix is\n"
        "column-stochastic in the emission slot and the\n"
        "transition slot alike. Shape does not separate\n"
        f"them here either — both are {N_SYMBOLS}x{N_SYMBOLS}.\n"
        "\n"
        "What caught it was an independent closed-form\n"
        "numpy chain that disagreed with pymdp, now\n"
        "kept as ailm.verification.oracle. Gate on a\n"
        "measured baseline and a measured ceiling, and\n"
        "cross-check the number against a second\n"
        "implementation — not against a threshold.",
        fontsize=FONTS["annotation"],
        va="top",
        ha="left",
        color=PALETTE["INK"],
        linespacing=1.5,
    )

    return fig


# ── Orchestration hook ───────────────────────────────────────────────────────


def generate_phoneme_matrices_figure(vendor_dir: Path | None = None) -> Figure:
    """Build ``fig:phoneme_matrices`` — Level 2's two learned matrices, side by side.

    Left: the G2P emission ``A[0, l, p] = P(letter | phone)`` — a genuine
    emission, letter observations over the phone inventory, column-stochastic
    over letters. Right: the phone-bigram transition
    ``B[0, i, j, 0] = P(next phone | current phone)``. Both are built through
    the exact calls the shipped agent uses (:func:`build_g2p_emission` on the
    vendored train split, :func:`build_phone_transition` on the vendored
    bigrams), so the figure cannot show a matrix the model does not have — and
    their different shapes (letters x phones vs phones x phones) is the visual
    form of the Part-A lesson that an emission and a transition must never be
    confusable by slot alone.

    Args:
        vendor_dir: Directory holding ``cmudict_counts.json``. Defaults to the
            project's ``vendor/``.

    Returns:
        A matplotlib Figure. The caller owns it.
    """
    from ailm.levels.phoneme import (
        N_PHONES,
        PHONES,
        build_g2p_emission,
        build_phone_transition,
        g2p_split,
        load_phoneme_counts,
    )

    apply_style()
    counts = load_phoneme_counts(vendor_dir)
    train, _test = g2p_split(counts)
    emission = np.asarray(build_g2p_emission(train), dtype=np.float64)[0]
    transition = np.asarray(build_phone_transition(counts["phone_bigrams"]), dtype=np.float64)[
        0, :, :, 0
    ]

    fig = plt.figure(figsize=(15.0, 8.2))
    ax_a = fig.add_axes((0.050, 0.130, 0.400, 0.720))
    ax_b = fig.add_axes((0.530, 0.130, 0.400, 0.720))
    cax = fig.add_axes((0.936, 0.130, 0.012, 0.720))

    cmap = _probability_cmap()
    norm = LogNorm(vmin=_PROB_VMIN, vmax=1.0)
    phone_labels = list(PHONES)

    # aspect="equal", deliberately: the shape contrast — a 30x39 emission beside a
    # 39x39 transition — is the panel's point, and "auto" would stretch both to
    # the same box and erase it. Letterboxing inside ax_a's box is fine.
    image = ax_a.imshow(
        emission,
        cmap=cmap,
        norm=norm,
        origin="upper",
        aspect="equal",
        interpolation="nearest",
    )
    ax_a.set_xticks(range(N_PHONES))
    ax_a.set_xticklabels(phone_labels, fontsize=FONTS["tick_label"], rotation=90)
    ax_a.set_yticks(range(N_SYMBOLS))
    ax_a.set_yticklabels([_display(symbol) for symbol in ALPHA_LIST], fontsize=FONTS["tick_label"])
    ax_a.set_xlabel("phone — hidden state $p$", fontsize=FONTS["axis_label"], labelpad=8)
    ax_a.set_ylabel("letter — observation $l$", fontsize=FONTS["axis_label"], labelpad=8)
    ax_a.set_title(
        f"L2B emission $A$: {N_SYMBOLS} letters x {N_PHONES} phones\n"
        "$P(\\mathrm{letter} \\mid \\mathrm{phone})$, columns sum to 1",
        fontsize=FONTS["title"],
        pad=12,
    )

    ax_b.imshow(
        transition,
        cmap=cmap,
        norm=norm,
        origin="upper",
        aspect="equal",
        interpolation="nearest",
    )
    ax_b.set_xticks(range(N_PHONES))
    ax_b.set_xticklabels(phone_labels, fontsize=FONTS["tick_label"], rotation=90)
    ax_b.set_yticks(range(N_PHONES))
    ax_b.set_yticklabels(phone_labels, fontsize=FONTS["tick_label"])
    ax_b.set_xlabel("current phone — state $j$", fontsize=FONTS["axis_label"], labelpad=8)
    ax_b.set_ylabel("next phone — state $i$", fontsize=FONTS["axis_label"], labelpad=8)
    ax_b.set_title(
        f"Phone transition $B$: {N_PHONES} x {N_PHONES} CMUdict bigram\n"
        "$P(\\mathrm{next} \\mid \\mathrm{current})$, columns sum to 1",
        fontsize=FONTS["title"],
        pad=12,
    )

    for ax in (ax_a, ax_b):
        ax.tick_params(length=3, width=0.8)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(PALETTE["MUTED"])

    bar = fig.colorbar(image, cax=cax, extend="min")
    bar.set_label("probability, log scale", fontsize=FONTS["axis_label"], labelpad=6)
    bar.ax.tick_params(labelsize=FONTS["tick_label"])

    # ── Peak cells, read from the matrices rather than typed ──────────────────
    peak_l, peak_p = np.unravel_index(int(np.argmax(emission)), emission.shape)
    peak_i, peak_j = np.unravel_index(int(np.argmax(transition)), transition.shape)
    fig.text(
        0.050,
        0.030,
        f"Strongest emission: phone {phone_labels[peak_p]} → letter "
        f"{_display(ALPHA_LIST[peak_l])!r}  (P = {emission[peak_l, peak_p]:.3f}).   "
        f"Strongest transition: {phone_labels[peak_j]} → {phone_labels[peak_i]}  "
        f"(P = {transition[peak_i, peak_j]:.3f}). Both computed from the vendored "
        "artifact at generation time.",
        fontsize=FONTS["annotation"],
        va="bottom",
        ha="left",
        color=PALETTE["INK"],
    )

    return fig


def write_matrix_figures(
    output_dir: Path | None = None,
    vendor_dir: Path | None = None,
) -> list[Path]:
    """Generate and write both matrix figures.

    Args:
        output_dir: Destination directory. Defaults to
            :func:`ailm.figures._common.default_output_dir`.
        vendor_dir: Directory holding ``brown_counts.json``.

    Returns:
        Paths written, in generation order.
    """
    paths: list[Path] = []
    builders = (
        ("letter_transition", lambda: generate_letter_transition_figure(vendor_dir)),
        (
            "wiring_comparison",
            lambda: generate_wiring_comparison_figure(None, vendor_dir),
        ),
    )
    for name, builder in builders:
        fig = builder()
        try:
            paths.append(save_figure(fig, name, output_dir))
        finally:
            plt.close(fig)
    return paths
