"""Composition figures: spec to running agent, and how the three levels relate.

Two figures, drawn entirely with matplotlib patches and text.

``fig:gnn_pipeline``
    The path from a GNN spec to a verified running agent: the spec sections on
    the left, the builder functions in the middle, the constructed
    ``pymdp.agent.Agent`` on the right, and the conformance check as a closing
    arrow back to the spec. The declared dimensions and roles are read out of
    ``gnn/models/L2b_grapheme_to_phoneme.gnn.md`` by :func:`ailm.gnn.load_spec` at generation time, so the
    panel cannot show a declaration the spec does not carry.

``fig:composition``
    The three shipped levels side by side, the one alphabet identity that
    genuinely joins two of them, and — as a drawn element rather than a footnote
    — the absence of any channel between the agents. The measured cost of the
    coupling attempt is loaded from the claims ledger.

**What "composition" means here.** The levels do not pass messages to each other.
``update_empirical_prior`` is used only for within-level temporal recursion. A
real Level 1 to Level 2 soft-observation channel was implemented
(:mod:`ailm.levels.coupling`) and measured harmful, so it ships gated off. What
composes is the alphabet — L2B's observation space *is* L1's state space — and
the GNN section grammar the three specs share. ``docs/gnn_composition.md`` is the
long form; ``docs/decisions.md`` holds the measurement.

Neither figure may contain a sentence implying agent-to-agent inference.
``tests/test_figures_composition.py`` scans the rendered text artists for that
and requires :data:`NO_MESSAGE_MARKER` to be present, so the property survives
future edits to the layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from ailm.figures._common import load_manuscript_variables, save_figure, token_float
from ailm.gnn import GnnSpec, load_spec
from ailm.levels.letter import N_SYMBOLS
from ailm.levels.phoneme import N_PHONES
from ailm.viz_config import FIGSIZES, FONTS, PALETTE, apply_style

__all__ = [
    "COUPLING_TOKENS",
    "NO_MESSAGE_MARKER",
    "generate_composition_figure",
    "generate_gnn_pipeline_figure",
    "write_composition_figures",
]


#: The sentence fig:composition draws across the top of the level cards. It is a
#: module constant rather than an inline literal because the honesty test binds to
#: it: a reader who only looks at the figure must not come away believing the
#: agents exchange beliefs, and the marker is what carries that.
NO_MESSAGE_MARKER: str = "No message channel: the levels never exchange beliefs"

#: Ledger tokens fig:composition prints. Listed here so the test can assert the
#: figure fails loudly when any one of them is missing rather than filling in a
#: plausible number.
COUPLING_TOKENS: tuple[str, ...] = (
    "COUPLING_BEST_ALPHA",
    "COUPLING_BEST_ACCURACY",
    "COUPLING_ALPHA0_ACCURACY",
    "COUPLING_MARGIN_PTS",
    "COUPLING_MCNEMAR_P",
    "COUPLING_EVAL_POSITIONS",
    "COUPLING_SUPPORTED",
)


# ── Small drawing helpers ────────────────────────────────────────────────────


def _panel(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    facecolor: str,
    edgecolor: str,
    linestyle: str = "-",
    linewidth: float = 1.4,
    zorder: float = 2.0,
) -> None:
    """Draw a rounded panel spanning ``(x0, y0)`` to ``(x1, y1)`` in axes units."""
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle="round,pad=0.003,rounding_size=0.010",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
        )
    )


def _lines(
    ax: plt.Axes,
    x: float,
    y_top: float,
    lines: list[tuple[str, float, str, str]],
    *,
    step: float,
    ha: str = "left",
) -> float:
    """Stack individually styled lines downward from ``y_top``.

    Each line becomes its own text artist rather than one multi-line block: the
    layout sweep measures a multi-line block at the width of its widest line, so
    a single tall block is charged more horizontal extent than it occupies and
    reads as a collision with its neighbor.

    Args:
        ax: Target axes, in 0..1 axes-fraction coordinates.
        x: Anchor x.
        y_top: y of the first line.
        lines: ``(text, fontsize, color, weight)`` per line. An empty text is a
            spacer and is not drawn.
        step: Vertical pitch between lines.
        ha: Horizontal alignment for every line.

    Returns:
        The y of the line after the last one drawn.
    """
    y = y_top
    for text, size, color, weight in lines:
        if text:
            ax.text(
                x,
                y,
                text,
                fontsize=size,
                fontweight=weight,
                va="center",
                ha=ha,
                color=color,
                zorder=4,
            )
        y -= step
    return y


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    linewidth: float = 2.0,
    linestyle: str = "-",
    style: str = "-|>",
    mutation_scale: float = 16.0,
    connectionstyle: str | None = None,
) -> None:
    """Draw one arrow between two axes-fraction points."""
    extra: dict[str, Any] = {"connectionstyle": connectionstyle} if connectionstyle else {}
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=3,
            **extra,
        )
    )


def _blank_canvas(size: str) -> tuple[Figure, plt.Axes]:
    """Return a styled full-bleed figure and its 0..1 axes."""
    apply_style()
    width, height = FIGSIZES[size]
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    return fig, ax


def _dims(spec: GnnSpec, name: str) -> str:
    """Format a variable's declared dimensions as they appear in the spec."""
    return ",".join(str(d) for d in spec.variable_dims(name))


# ── fig:gnn_pipeline ─────────────────────────────────────────────────────────

# Column extents in axes fraction. The gaps are wide enough to hold the arrow
# labels without either column's text reaching into them.
_SPEC_X = (0.020, 0.320)
_BUILD_X = (0.370, 0.630)
_AGENT_X = (0.680, 0.980)
_PIPE_TOP, _PIPE_BOTTOM = 0.880, 0.212
_FLOW_Y = 0.540


def generate_gnn_pipeline_figure(spec: GnnSpec | None = None) -> Figure:
    """Build ``fig:gnn_pipeline`` — GNN spec to a conformance-checked agent.

    Args:
        spec: Parsed L2B spec. Defaults to ``load_spec("L2b_grapheme_to_phoneme")``, so the
            declared dimensions and roles on the panel are the ones the shipped
            file actually carries.

    Returns:
        A matplotlib Figure. The caller owns it.
    """
    if spec is None:
        spec = load_spec("L2b_grapheme_to_phoneme")

    a_dims, b_dims = _dims(spec, "A"), _dims(spec, "B")
    c_dims, d_dims = _dims(spec, "C"), _dims(spec, "D")
    num_states = spec.parameters["num_hidden_states"]
    num_obs = spec.parameters["num_obs"]

    fig, ax = _blank_canvas("table")

    ax.text(
        0.0,
        0.990,
        "From a GNN spec to a conformance-checked agent",
        fontsize=FONTS["suptitle"],
        fontweight="bold",
        va="top",
        ha="left",
        color=PALETTE["INK"],
    )
    # The repo-relative path from "gnn" down, derived from spec.origin rather than
    # typed: spec.origin is an absolute path (whose prefix depends on where the
    # checkout lives and would overrun the canvas), and the real file lives under
    # gnn/models/, so printing "gnn/<name>" would name a nonexistent path.
    origin_parts = Path(spec.origin).parts
    spec_path = (
        "/".join(origin_parts[origin_parts.index("gnn") :])
        if "gnn" in origin_parts
        else Path(spec.origin).name
    )
    ax.text(
        0.0,
        0.935,
        f"Worked on the L2B grapheme-to-phoneme level, "
        f"{spec_path}. Declared dimensions and roles below are "
        "read out of that file at generation time.",
        fontsize=FONTS["annotation"],
        va="top",
        ha="left",
        color=PALETTE["MUTED"],
    )

    for (x0, _x1), header, color in (
        (_SPEC_X, "1 — the GNN spec declares", PALETTE["MODEL"]),
        (_BUILD_X, "2 — the builders construct", PALETTE["FLOOR"]),
        (_AGENT_X, "3 — the live pymdp object", PALETTE["INK"]),
    ):
        ax.text(
            x0,
            0.902,
            header,
            fontsize=FONTS["title"],
            fontweight="bold",
            va="center",
            ha="left",
            color=color,
        )

    # ── Column 1: the spec sections ──────────────────────────────────────────
    spec_rows: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "StateSpaceBlock",
            (
                "every variable and its dimensions",
                f"A[{a_dims}]   B[{b_dims}]   C[{c_dims}]   D[{d_dims}]",
            ),
        ),
        (
            "Connections",
            (
                "which variables touch which",
                "D>s_t   s_t-A   A-o_t   s_t-B   B>s_prime",
            ),
        ),
        (
            "InitialParameterization",
            (
                "structural form, or the generator that",
                "reproduces a learned block byte for byte",
            ),
        ),
        (
            "ActInfOntologyAnnotation",
            (
                "the ROLE of every slot — the guard",
                "A=RecognitionMatrix   B=TransitionMatrix",
            ),
        ),
        (
            "ModelParameters",
            (
                "the sizes both sides must agree on",
                f"num_hidden_states: {num_states}   num_obs: {num_obs}",
            ),
        ),
    )
    _stacked_boxes(
        ax,
        _SPEC_X,
        spec_rows,
        edgecolor=PALETTE["MODEL"],
        header_color=PALETTE["MODEL"],
    )

    # ── Column 2: the builders ───────────────────────────────────────────────
    build_rows: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "build_g2p_emission()",
            (f"returns A, shape (1, {N_SYMBOLS}, {N_PHONES})",),
        ),
        (
            "build_g2p_transition()",
            (f"returns B, shape (1, {N_PHONES}, {N_PHONES}, 1)",),
        ),
        ("build_g2p_start()", (f"returns D, shape (1, {N_PHONES})",)),
        (
            "build_g2p_agent(A, B, D)",
            (
                "asserts each shape, then constructs the",
                f"Agent with a flat C of {N_SYMBOLS} zeros",
            ),
        ),
    )
    _stacked_boxes(
        ax,
        _BUILD_X,
        build_rows,
        edgecolor=PALETTE["FLOOR"],
        header_color=PALETTE["FLOOR"],
    )

    # ── Column 3: the constructed agent ──────────────────────────────────────
    _panel(
        ax,
        _AGENT_X[0] - 0.008,
        _PIPE_BOTTOM - 0.010,
        _AGENT_X[1] + 0.008,
        _PIPE_TOP + 0.006,
        facecolor=PALETTE["PANEL"],
        edgecolor=PALETTE["INK"],
        linewidth=1.8,
        zorder=1.2,
    )
    agent_rows: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            f"agent.A[0] — (1, {N_SYMBOLS}, {N_PHONES})",
            (
                "P(letter | phone), the learned emission",
                "size pinned by StateSpaceBlock + ModelParameters",
                "role pinned by ActInfOntologyAnnotation",
            ),
        ),
        (
            f"agent.B[0] — (1, {N_PHONES}, {N_PHONES}, 1)",
            (
                "P(next phone | current phone), TRAIN split",
                "size pinned by StateSpaceBlock + ModelParameters",
                "role pinned by ActInfOntologyAnnotation",
            ),
        ),
        (
            f"agent.D[0] — (1, {N_PHONES})",
            (
                "distribution over a word's first phone",
                "size pinned by StateSpaceBlock",
                "role pinned by ActInfOntologyAnnotation",
            ),
        ),
        (
            f"agent.C[0] — (1, {N_SYMBOLS})",
            (
                "flat: no preference over observed letters",
                "size pinned by StateSpaceBlock",
                "declared Zeros in InitialParameterization",
            ),
        ),
    )
    _stacked_boxes(
        ax,
        _AGENT_X,
        agent_rows,
        edgecolor=PALETTE["INK"],
        header_color=PALETTE["INK"],
        facecolor="white",
    )

    # ── Flow arrows ──────────────────────────────────────────────────────────
    for (x_from, x_to), label in (
        ((_SPEC_X[1] + 0.006, _BUILD_X[0] - 0.006), "declares"),
        ((_BUILD_X[1] + 0.006, _AGENT_X[0] - 0.014), "builds"),
    ):
        _arrow(ax, (x_from, _FLOW_Y), (x_to, _FLOW_Y), color=PALETTE["INK"])
        ax.text(
            (x_from + x_to) / 2,
            _FLOW_Y + 0.028,
            label,
            fontsize=FONTS["annotation"],
            fontweight="bold",
            va="center",
            ha="center",
            color=PALETTE["INK"],
        )

    # ── The closing arrow: conformance, right back to left ───────────────────
    band_y = 0.128
    _panel(
        ax,
        0.020,
        0.070,
        0.980,
        0.188,
        facecolor="white",
        edgecolor=PALETTE["ACCENT"],
        linewidth=2.0,
        zorder=1.2,
    )
    _arrow(
        ax,
        (_AGENT_X[1] - 0.010, band_y),
        (_SPEC_X[0] + 0.010, band_y),
        color=PALETTE["ACCENT"],
        linewidth=2.2,
        mutation_scale=18.0,
    )
    ax.text(
        0.500,
        0.168,
        "conformance — the spec is checked against the LIVE object, not merely filed beside it",
        fontsize=FONTS["axis_label"],
        fontweight="bold",
        va="center",
        ha="center",
        color=PALETTE["ACCENT"],
    )
    ax.text(
        0.500,
        band_y - 0.032,
        "describe_shapes(agent) reads A/B/C/D off the constructed Agent  ->  "
        "check_conformance(spec, ...) returns every violation  ->  "
        "assert_conforms(spec, ...) raises",
        fontsize=FONTS["annotation"],
        va="center",
        ha="center",
        color=PALETTE["INK"],
    )

    ax.text(
        0.0,
        0.010,
        "The spec does not generate the builders: both are hand-written. Two "
        "things bind them: this arrow, and ailm.spec_agent.build_agent_from_spec, "
        "whose spec-only agent is bit-identical to the builder-built one at every "
        "level\n(tests/test_spec_agent.py). src/ailm/gnn/ is stdlib plus numpy "
        "and never imports pymdp, so the checker cannot inherit the defect it "
        "audits for; tests/test_gnn_conformance.py must REJECT a transposed "
        "emission and an A/B swap.",
        fontsize=FONTS["cell"],
        va="bottom",
        ha="left",
        color=PALETTE["MUTED"],
        linespacing=1.5,
    )

    return fig


def _stacked_boxes(
    ax: plt.Axes,
    x_span: tuple[float, float],
    rows: tuple[tuple[str, tuple[str, ...]], ...],
    *,
    edgecolor: str,
    header_color: str,
    facecolor: str = "white",
) -> None:
    """Lay ``rows`` out as evenly spaced boxes filling the pipeline column."""
    x0, x1 = x_span
    count = len(rows)
    span = _PIPE_TOP - _PIPE_BOTTOM
    pitch = span / count
    box_h = pitch - 0.022
    for index, (header, body) in enumerate(rows):
        top = _PIPE_TOP - index * pitch
        _panel(
            ax,
            x0,
            top - box_h,
            x1,
            top,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.4,
        )
        lines: list[tuple[str, float, str, str]] = [
            (header, FONTS["axis_label"], header_color, "bold")
        ]
        lines.extend((text, FONTS["cell"], PALETTE["INK"], "normal") for text in body)
        # Center the stack vertically inside its box.
        pitch_in_box = 0.030
        block = (len(lines) - 1) * pitch_in_box
        first_y = (top - box_h / 2) + block / 2
        _lines(ax, x0 + 0.012, first_y, lines, step=pitch_in_box)


# ── fig:composition ──────────────────────────────────────────────────────────

_CARDS_X: tuple[tuple[float, float], ...] = (
    (0.020, 0.315),
    (0.345, 0.640),
    (0.670, 0.965),
)
_CARD_TOP, _CARD_BOTTOM = 0.870, 0.565
_PANELS_TOP, _PANELS_BOTTOM = 0.400, 0.048


def _card(
    ax: plt.Axes,
    index: int,
    title: str,
    spec_file: str,
    body: tuple[tuple[str, str], ...],
    *,
    edgecolor: str,
    badge: str,
    badge_color: str,
    badge_hatch: str,
) -> None:
    """Draw one level card: title, spec file, an A/B badge, then labeled rows."""
    x0, x1 = _CARDS_X[index]
    _panel(
        ax,
        x0,
        _CARD_BOTTOM,
        x1,
        _CARD_TOP,
        facecolor="white",
        edgecolor=edgecolor,
        linewidth=1.8,
    )
    ax.text(
        x0 + 0.012,
        _CARD_TOP - 0.030,
        title,
        fontsize=FONTS["title"],
        fontweight="bold",
        va="center",
        ha="left",
        color=edgecolor,
    )
    ax.text(
        x0 + 0.012,
        _CARD_TOP - 0.062,
        spec_file,
        fontsize=FONTS["cell"],
        va="center",
        ha="left",
        color=PALETTE["MUTED"],
    )

    badge_y = _CARD_TOP - 0.096
    ax.add_patch(
        Rectangle(
            (x0 + 0.012, badge_y - 0.009),
            0.010,
            0.018,
            facecolor="white" if badge_hatch else badge_color,
            edgecolor=badge_color,
            hatch=badge_hatch or None,
            linewidth=1.3,
            zorder=4,
        )
    )
    ax.text(
        x0 + 0.030,
        badge_y,
        badge,
        fontsize=FONTS["annotation"],
        fontweight="bold",
        va="center",
        ha="left",
        color=badge_color,
        zorder=4,
    )

    lines = [
        (f"{label}  {value}", FONTS["cell"], PALETTE["INK"], "normal") for label, value in body
    ]
    _lines(ax, x0 + 0.012, _CARD_TOP - 0.132, lines, step=0.030)


def _bottom_panel(
    ax: plt.Axes,
    index: int,
    header: str,
    header_color: str,
    body: list[tuple[str, float, str, str]],
    *,
    edgecolor: str,
) -> None:
    """Draw one of the three explanatory panels below the cards."""
    x0, x1 = _CARDS_X[index]
    _panel(
        ax,
        x0,
        _PANELS_BOTTOM,
        x1,
        _PANELS_TOP,
        facecolor="white",
        edgecolor=edgecolor,
        linewidth=1.6,
    )
    ax.text(
        x0 + 0.012,
        _PANELS_TOP - 0.030,
        header,
        fontsize=FONTS["axis_label"],
        fontweight="bold",
        va="center",
        ha="left",
        color=header_color,
    )
    _lines(ax, x0 + 0.012, _PANELS_TOP - 0.064, body, step=0.024)


def _coupling_strings(variables: dict[str, str] | None) -> dict[str, str]:
    """Read the coupling measurement out of the generated token map.

    Args:
        variables: An explicit token map, or None to load the generated one.

    Returns:
        Token name to the string the manuscript prints, so the figure and the
        prose cannot disagree about a digit.

    Raises:
        MissingVariableError: If any token in :data:`COUPLING_TOKENS` is absent
            or non-numeric. A figure that made up a number would be worse than a
            figure that failed to build.
    """
    resolved = load_manuscript_variables() if variables is None else variables
    for token in COUPLING_TOKENS:
        token_float(resolved, token)  # validates presence and numeracy
    return {token: resolved[token] for token in COUPLING_TOKENS}


def generate_composition_figure(variables: dict[str, str] | None = None) -> Figure:
    """Build ``fig:composition`` — how the three shipped levels relate.

    Args:
        variables: Manuscript token map. Defaults to the generated one; passing a
            map without the coupling tokens raises rather than substituting a
            placeholder.

    Returns:
        A matplotlib Figure. The caller owns it.
    """
    values = _coupling_strings(variables)
    fig, ax = _blank_canvas("table")

    ax.text(
        0.0,
        0.990,
        "How the AILM levels compose — and how they do not",
        fontsize=FONTS["suptitle"],
        fontweight="bold",
        va="top",
        ha="left",
        color=PALETTE["INK"],
    )

    # ── The absence, drawn where an inter-agent arrow would go ───────────────
    left_center = sum(_CARDS_X[0]) / 2
    right_center = sum(_CARDS_X[2]) / 2
    mid = (left_center + right_center) / 2
    channel_y = 0.905
    for x_from, x_to in ((left_center, mid - 0.016), (mid + 0.016, right_center)):
        _arrow(
            ax,
            (x_from, channel_y),
            (x_to, channel_y),
            color=PALETTE["ANTI"],
            linewidth=1.8,
            linestyle="--",
            style="-",
        )
    ax.text(
        mid,
        channel_y,
        "✗",
        fontsize=FONTS["suptitle"],
        fontweight="bold",
        va="center",
        ha="center",
        color=PALETTE["ANTI"],
        zorder=4,
    )
    ax.text(
        mid,
        channel_y + 0.038,
        NO_MESSAGE_MARKER,
        fontsize=FONTS["axis_label"],
        fontweight="bold",
        va="center",
        ha="center",
        color=PALETTE["ANTI"],
    )

    # ── The three shipped levels ─────────────────────────────────────────────
    _card(
        ax,
        0,
        "L1 — Letter",
        "gnn/models/L1_letter.gnn.md",
        (
            ("hidden state:", f"letter ({N_SYMBOLS} symbols)"),
            ("observation:", f"letter ({N_SYMBOLS} symbols)"),
            ("A:", f"({N_SYMBOLS}, {N_SYMBOLS}) identity"),
            ("B:", "Brown letter bigram"),
            ("D:", f"uniform over {N_SYMBOLS} letters"),
            ("built by:", "levels.letter.build_letter_agent"),
        ),
        edgecolor=PALETTE["INK"],
        badge="A = I — the observation is the state",
        badge_color=PALETTE["IDENTITY"],
        badge_hatch="///",
    )
    _card(
        ax,
        1,
        "L2A — Phoneme sequence",
        "gnn/models/L2a_phoneme_sequence.gnn.md",
        (
            ("hidden state:", f"phone ({N_PHONES} ARPAbet)"),
            ("observation:", f"phone ({N_PHONES} ARPAbet)"),
            ("A:", f"({N_PHONES}, {N_PHONES}) identity"),
            ("B:", "CMUdict phone bigram"),
            ("D:", "phone unigram"),
            ("built by:", "levels.phoneme.build_phoneme_agent"),
        ),
        edgecolor=PALETTE["INK"],
        badge="A = I — the observation is the state",
        badge_color=PALETTE["IDENTITY"],
        badge_hatch="///",
    )
    _card(
        ax,
        2,
        "L2B — Grapheme to phoneme",
        "gnn/models/L2b_grapheme_to_phoneme.gnn.md",
        (
            ("hidden state:", f"phone ({N_PHONES} ARPAbet) — L2A's space"),
            ("observation:", f"LETTER ({N_SYMBOLS} symbols) — L1's space"),
            ("A:", f"({N_SYMBOLS}, {N_PHONES}) learned emission"),
            ("B:", "phone bigram, TRAIN split only"),
            ("D:", "first phone of a training word"),
            ("built by:", "levels.phoneme.build_g2p_agent"),
        ),
        edgecolor=PALETTE["ACCENT"],
        badge="A learned — a genuine emission",
        badge_color=PALETTE["LEARNED"],
        badge_hatch="",
    )

    # ── The one relation that genuinely joins two levels ──────────────────────
    # Drawn as a plain connector with an identity glyph, never as an arrow: an
    # arrowhead is the visual idiom for a value flowing, and nothing flows here.
    link_y = 0.508
    for x in (left_center, right_center):
        _arrow(
            ax,
            (x, _CARD_BOTTOM - 0.004),
            (x, link_y),
            color=PALETTE["MODEL"],
            linewidth=1.6,
            style="-",
        )
    for x_from, x_to in (
        (left_center, mid - 0.014),
        (mid + 0.014, right_center),
    ):
        _arrow(
            ax,
            (x_from, link_y),
            (x_to, link_y),
            color=PALETTE["MODEL"],
            linewidth=1.6,
            style="-",
        )
    ax.text(
        mid,
        link_y,
        "≡",
        fontsize=FONTS["suptitle"],
        fontweight="bold",
        va="center",
        ha="center",
        color=PALETTE["MODEL"],
        zorder=4,
    )
    ax.text(
        mid,
        link_y - 0.030,
        f"L1's {N_SYMBOLS}-symbol STATE alphabet IS L2B's {N_SYMBOLS}-symbol OBSERVATION alphabet",
        fontsize=FONTS["axis_label"],
        fontweight="bold",
        va="center",
        ha="center",
        color=PALETTE["MODEL"],
    )
    ax.text(
        mid,
        link_y - 0.060,
        "The levels meet in a shared alphabet — a type identity fixed at build "
        "time, not a value computed at run time.",
        fontsize=FONTS["annotation"],
        va="center",
        ha="center",
        color=PALETTE["INK"],
    )
    ax.text(
        mid,
        link_y - 0.086,
        f"L2A and L2B likewise share the {N_PHONES}-phone state space. Sharing an "
        "alphabet is all the coupling there is.",
        fontsize=FONTS["annotation"],
        va="center",
        ha="center",
        color=PALETTE["MUTED"],
    )

    # Local shorthand: every panel body line is 9 pt and differs only in
    # color and weight.
    def _row(
        text: str, color: str = PALETTE["INK"], weight: str = "normal"
    ) -> tuple[str, float, str, str]:
        return (text, FONTS["cell"], color, weight)

    gap = _row("")
    anti, muted = PALETTE["ANTI"], PALETTE["MUTED"]

    # ── Panel 1: the measured null ───────────────────────────────────────────
    _bottom_panel(
        ax,
        0,
        "Off, because it was measured harmful",
        anti,
        [
            _row("ailm.levels.coupling mixes L1's predictive over letters"),
            _row("into L2B's likelihood by weight alpha. It is kept and"),
            _row("gated, alpha defaulting to 0 — the uncoupled decode."),
            gap,
            _row(
                f"best positive alpha {values['COUPLING_BEST_ALPHA']}: "
                f"{values['COUPLING_BEST_ACCURACY']}% against",
                anti,
                "bold",
            ),
            _row(
                f"{values['COUPLING_ALPHA0_ACCURACY']}% uncoupled — a margin of "
                f"{values['COUPLING_MARGIN_PTS']} points, a LOSS,",
                anti,
                "bold",
            ),
            _row(
                f"at paired McNemar p = {values['COUPLING_MCNEMAR_P']} over "
                f"{values['COUPLING_EVAL_POSITIONS']} held-out",
                anti,
                "bold",
            ),
            _row(
                f"positions. Recorded support flag = {values['COUPLING_SUPPORTED']}.",
                anti,
                "bold",
            ),
            gap,
            _row("Structural cause: at position t the G2P level already", muted),
            _row("observes letter t, while L1's predictive is a function", muted),
            _row("of letters 0..t-1 only.", muted),
        ],
        edgecolor=anti,
    )

    # ── Panel 2: what inference is not done ──────────────────────────────────
    _bottom_panel(
        ax,
        1,
        "What neither Part-A model does",
        PALETTE["INK"],
        [
            _row("No policy selection and no expected free energy. C is"),
            _row("zero at both built levels and each has a single trivial"),
            _row("action, so there is nothing to choose between."),
            gap,
            _row("No agent-to-agent inference of any kind. Each level"),
            _row("runs its own recursion over its own observations:"),
            _row("infer_states, then update_empirical_prior for the NEXT"),
            _row("position of the SAME level. That call never crosses a"),
            _row("level boundary."),
            gap,
            _row("Filtering, not smoothing: position t conditions on", muted),
            _row("symbols 1..t only.", muted),
        ],
        edgecolor=PALETTE["MUTED"],
    )

    # ── Panel 3: structural composition, and where the rest is drawn ─────────
    _bottom_panel(
        ax,
        2,
        "Shared grammar; full stack drawn elsewhere",
        PALETTE["INK"],
        [
            _row("All three specs share one section grammar, one ontology"),
            _row("vocabulary (RecognitionMatrix, TransitionMatrix, Prior)"),
            _row("and one naming convention, so one parser reads all of"),
            _row("them and one checker binds each to its live agent."),
            _row("That is composition a reader or a renderer can use now."),
            gap,
            _row("This figure details only the three Part-A models. The", muted),
            _row("full six-level stack is in fig:hierarchy; all five", muted),
            _row("measured channels are reported in sec:hierarchy.", muted),
            _row("Top-down C is implemented at one seam: the word→letter", muted),
            _row("generation preference. The L6→L5 channel is a", muted),
            _row("blend into the clause level's start prior D, not a C.", muted),
        ],
        edgecolor=PALETTE["MUTED"],
    )

    ax.text(
        0.0,
        0.010,
        "Composition here is representational and structural, never a message "
        "pass. docs/gnn_composition.md is the long form; docs/decisions.md entry "
        "9 records the coupling null and its controls.",
        fontsize=FONTS["cell"],
        va="bottom",
        ha="left",
        color=PALETTE["MUTED"],
    )

    return fig


# ── Orchestration hook ───────────────────────────────────────────────────────


def write_composition_figures(output_dir: Path | None = None) -> list[Path]:
    """Generate and write both composition figures.

    Args:
        output_dir: Destination directory. Defaults to
            :func:`ailm.figures._common.default_output_dir`.

    Returns:
        Paths written, in generation order.
    """
    paths: list[Path] = []
    for name, builder in (
        ("composition", generate_composition_figure),
        ("gnn_pipeline", generate_gnn_pipeline_figure),
    ):
        fig = builder()
        try:
            paths.append(save_figure(fig, name, output_dir))
        finally:
            plt.close(fig)
    return paths
