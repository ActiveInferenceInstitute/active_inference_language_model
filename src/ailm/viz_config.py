"""Shared figure styling for AILM.

One place for the palette, the type scale, the canvas sizes, and the legibility
floor, so every figure in the manuscript looks like it came from the same paper.

Three properties are load-bearing and the reasons are worth stating.

**Colorblind-safe by construction.** :data:`PALETTE` is the Okabe--Ito
eight-color qualitative set plus two neutrals. No distinction anywhere in this
project may be carried by red versus green alone: the two comparators that sit
closest together in the argument — the *correct* wiring and the
*anti-informative* wiring — are blue and vermillion, and they are additionally
separated by fill pattern and by an explicit text label. A reader with
deuteranopia loses none of the argument.

**Colors are named for what they mean, not for what they look like.** ``MODEL``
is the shipped agent, ``UNIGRAM`` the honest context-free baseline, ``UNIFORM``
the crippled comparator, ``FLOOR`` the information-theoretic ceiling on the
achievable gain, ``ANTI`` the mis-wired variant. Renaming a color therefore
requires deciding that the *meaning* changed, which is the intended friction.
For panels comparing unrelated categories, :func:`qualitative_colors` supplies
these same hues as a distinct qualitative sequence; the legend then identifies
the categories rather than inheriting the palette's model/comparator roles.

**Legibility is a build-time invariant.** :data:`MIN_FONT_PT` is the smallest
type any AILM figure may contain, and
:func:`ailm.figures._common.assert_legible` enforces it against the produced
figure rather than trusting the code that wrote it. A runtime clamp that shrinks
text to fit is not a floor; it is a way of silently producing an unreadable
panel. Nothing here clamps.

``src/`` stays standalone: this module imports matplotlib and nothing from
``infrastructure``.
"""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

__all__ = [
    "DPI",
    "FIGSIZES",
    "FONTS",
    "MIN_FONT_PT",
    "PALETTE",
    "PALETTE_ORDER",
    "SEQUENTIAL_CMAP",
    "VIZ_CONFIG",
    "apply_style",
    "qualitative_colors",
]

# ── Legibility floor ─────────────────────────────────────────────────────────

#: Smallest font size, in points, permitted in any AILM figure. Enforced against
#: the rendered figure by ``assert_legible``; figures do not clamp to it.
MIN_FONT_PT: float = 9.0

#: Raster resolution for every saved PNG. 200 dpi keeps a 16-inch canvas legible
#: on screen and at print size without producing files too large to commit.
DPI: int = 200


# ── Palette ──────────────────────────────────────────────────────────────────
# Okabe & Ito (2008), "Color Universal Design" — the eight-color qualitative set
# that stays distinguishable under deuteranopia, protanopia and tritanopia, plus
# two greys for structure. Keys are semantic roles in the AILM argument.
PALETTE: dict[str, str] = {
    # Comparators, in the order the argument uses them.
    "UNIFORM": "#999999",  # neutral grey — the crippled comparator
    "UNIGRAM": "#E69F00",  # orange — the honest context-free baseline
    "MODEL": "#0072B2",  # blue — the shipped pymdp agent
    "FLOOR": "#009E73",  # bluish green — the information-theoretic ceiling
    "ANTI": "#D55E00",  # vermillion — the anti-informative mis-wiring
    # Structural roles.
    "ACCENT": "#CC79A7",  # reddish purple — annotations and callouts
    "IDENTITY": "#56B4E9",  # sky blue — A = I, a state read-off
    "LEARNED": "#0072B2",  # blue — A or B carrying corpus statistics
    "HIGHLIGHT": "#F0E442",  # yellow — the one cell or row being pointed at
    # Neutrals.
    "INK": "#1A1A1A",  # body text and strong strokes
    "OUTLINE": "#22223B",  # marker edges, hlines and annotation ink
    "MUTED": "#6E6E6E",  # secondary text, pending/not-yet-built content
    "PANEL": "#F2F2F2",  # panel fill behind a group of rows
    "GATE": "#D1495B",  # the declared gate threshold line and its annotation
    "BACKGROUND": "#FFFFFF",
}

#: Deterministic cycle order for series colors. Only the eight Okabe--Ito hues
#: appear here — the neutrals are structure, never data.
PALETTE_ORDER: tuple[str, ...] = (
    PALETTE["MODEL"],
    PALETTE["UNIGRAM"],
    PALETTE["FLOOR"],
    PALETTE["ANTI"],
    PALETTE["ACCENT"],
    PALETTE["IDENTITY"],
    PALETTE["HIGHLIGHT"],
    PALETTE["UNIFORM"],
)

#: Sequential colormap for probability heatmaps. Perceptually uniform and
#: monotone in lightness, so a greyscale print keeps the ordering, and it avoids
#: the red/green axis entirely.
SEQUENTIAL_CMAP: str = "viridis"


# ── Type scale ───────────────────────────────────────────────────────────────
# Every size is at or above MIN_FONT_PT. ``cell`` is the smallest role in use —
# the body text of the hierarchy table — and it sits exactly on the floor, so any
# future shrink is caught by assert_legible rather than by a reader squinting.
FONTS: dict[str, float] = {
    "suptitle": 17.0,
    "title": 14.0,
    "axis_label": 12.0,
    "tick_label": 10.0,
    "legend": 10.0,
    "annotation": 10.0,
    "cell": 9.0,
    "footnote": 9.5,
    "subtitle": 10.5,
}


# ── Canvas sizes (inches) ────────────────────────────────────────────────────
FIGSIZES: dict[str, tuple[float, float]] = {
    "confirmatory": (9.4, 10.5),
    "single": (9.0, 6.5),
    "wide": (14.0, 8.0),
    "table": (16.0, 9.0),
    "square": (8.5, 8.5),
    "comparison": (15.0, 9.0),
    "level_summary": (13.5, 8.2),
}


#: Aggregate view, for callers that want one object rather than five imports.
VIZ_CONFIG: dict[str, Any] = {
    "palette": PALETTE,
    "palette_order": PALETTE_ORDER,
    "sequential_cmap": SEQUENTIAL_CMAP,
    "fonts": FONTS,
    "figsizes": FIGSIZES,
    "dpi": DPI,
    "min_font_pt": MIN_FONT_PT,
}


def qualitative_colors(n: int) -> list[str]:
    """Return ``n`` distinct qualitative colors in the canonical order.

    Args:
        n: How many series need a color.

    Returns:
        The first ``n`` entries of :data:`PALETTE_ORDER`.

    Raises:
        ValueError: If ``n`` is negative, or exceeds the number of
            colorblind-safe hues available. Cycling past the end would hand two
            series the same color, which is a silent chart defect; the caller
            must instead re-encode with markers, hatching, or facets.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}.")
    if n > len(PALETTE_ORDER):
        raise ValueError(
            f"Requested {n} qualitative colors but only {len(PALETTE_ORDER)} "
            "colorblind-safe hues are defined. Do not cycle — re-encode the "
            "extra series with markers or hatching, or split the panel."
        )
    return list(PALETTE_ORDER[:n])


def apply_style() -> None:
    """Apply the AILM matplotlib style globally and deterministically.

    Pins the font family to the DejaVu family that ships inside matplotlib, so a
    figure renders identically on a machine with a different set of system fonts —
    a substituted font changes glyph widths, which changes the ``tight`` bounding
    box, which breaks byte-identical regeneration.

    Idempotent: calling it twice is the same as calling it once.
    """
    mpl.rcParams.update(
        {
            # ── Determinism ──────────────────────────────────────────────────
            # Bundled fonts only. No system-font fallback, no font hashing in the
            # PNG, and SVG text stays text rather than becoming paths.
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "svg.hashsalt": "ailm",
            # ── Canvas ───────────────────────────────────────────────────────
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "figure.facecolor": PALETTE["BACKGROUND"],
            "figure.edgecolor": "none",
            "savefig.facecolor": PALETTE["BACKGROUND"],
            "savefig.edgecolor": "none",
            # ── Type ─────────────────────────────────────────────────────────
            "font.size": FONTS["tick_label"],
            "axes.titlesize": FONTS["title"],
            "axes.labelsize": FONTS["axis_label"],
            "xtick.labelsize": FONTS["tick_label"],
            "ytick.labelsize": FONTS["tick_label"],
            "legend.fontsize": FONTS["legend"],
            "figure.titlesize": FONTS["suptitle"],
            "axes.titleweight": "bold",
            "text.color": PALETTE["INK"],
            "axes.labelcolor": PALETTE["INK"],
            "xtick.color": PALETTE["INK"],
            "ytick.color": PALETTE["INK"],
            # ── Marks ────────────────────────────────────────────────────────
            "axes.prop_cycle": cycler(color=list(PALETTE_ORDER)),
            "image.cmap": SEQUENTIAL_CMAP,
            "lines.linewidth": 2.0,
            "lines.markersize": 7.0,
            "patch.linewidth": 1.2,
            # ── Frame ────────────────────────────────────────────────────────
            "axes.edgecolor": PALETTE["MUTED"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": PALETTE["MUTED"],
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.edgecolor": PALETTE["MUTED"],
            "legend.fancybox": False,
        }
    )
    # Touch pyplot so a caller that only imported this module still gets a
    # configured pyplot state machine.
    plt.rcParams["figure.dpi"] = DPI
