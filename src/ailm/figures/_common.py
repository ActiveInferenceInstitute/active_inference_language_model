"""Shared helpers for AILM figure generators.

Three jobs, each of which exists because of a specific failure mode.

:func:`save_figure` writes PNGs **deterministically**. matplotlib stamps a
``Software`` key into PNG text metadata by default, which embeds the matplotlib
version and makes two byte-comparisons of the same plot disagree across
environments. Overriding it with a constant, pinning dpi and pad, and never
writing a timestamp into the image means regenerating a figure twice produces
identical bytes — so "the figures are up to date" is a checkable claim rather
than a hope.

:func:`assert_legible` enforces :data:`ailm.viz_config.MIN_FONT_PT` against the
**produced figure**, walking every text artist matplotlib actually holds. This is
deliberately not a clamp. A clamp that shrinks type to make it fit converts an
overflowing label into an unreadable one and reports success; the check here
fails the build instead. It also refuses a figure with no text at all, because a
legibility check that passes trivially on an empty canvas is not a check.
Both geometry checks below extend the same guarantee to layout:
:func:`assert_within_canvas` refuses text that leaves the canvas, and
:func:`assert_no_text_crosses_rules` refuses a drawn reference rule that
strikes through a label.

:func:`load_manuscript_variables` reads the generated token map. Figure code that
needs a measured number — the wiring-comparison panel needs two perplexities —
loads it from ``output/data/manuscript_variables.json`` at generation time. A
number typed into plotting code is exactly as fabricated as a number typed into
prose, and harder to notice.

``src/`` stays standalone: nothing here imports ``infrastructure``.
"""

from __future__ import annotations

import json
import math
import os
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg", force=False)

from matplotlib import image as matplotlib_image
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from ailm.viz_config import DPI, MIN_FONT_PT, PALETTE

#: Paper widths in inches for the LaTeX size options a preamble may name. The
#: text block is the paper width less both margins, so getting this wrong scales
#: every on-page measurement below.
_PAPER_WIDTHS_IN: dict[str, float] = {
    "letterpaper": 8.5,
    "legalpaper": 8.5,
    "executivepaper": 7.25,
    "a4paper": 210.0 / 25.4,
    "a5paper": 148.0 / 25.4,
    "b5paper": 176.0 / 25.4,
}

#: Used when the preamble names no size. Pandoc's template declares none, so
#: LaTeX's ``article`` default — US Letter — applies. Stated rather than typed
#: into an expression, because an ``a4paper`` switch has to change it.
_DEFAULT_PAPER_WIDTH_IN = 8.5


class ExportFigure(Figure):
    """A figure with an explicit raster-export density and unchanged layout DPI."""

    def __init__(self, *args: Any, export_dpi: int = DPI, **kwargs: Any) -> None:
        if isinstance(export_dpi, bool) or not isinstance(export_dpi, int) or export_dpi <= 0:
            raise ValueError("export_dpi must be a positive integer")
        self._ailm_export_dpi = export_dpi
        super().__init__(*args, **kwargs)

    @property
    def export_dpi(self) -> int:
        return self._ailm_export_dpi


def _png_width_and_density(path: Path) -> tuple[int, float]:
    """Read the PNG's actual physical density; missing metadata is not a default."""
    content = path.read_bytes()
    if content[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: expected a PNG")
    cursor = 8
    width: int | None = None
    density: float | None = None
    while cursor + 12 <= len(content):
        length, kind = struct.unpack_from(">I4s", content, cursor)
        end = cursor + 8 + length
        if end + 4 > len(content):
            raise ValueError(f"{path}: truncated PNG chunk")
        body = content[cursor + 8 : end]
        expected_crc = struct.unpack_from(">I", content, end)[0]
        if zlib.crc32(kind + body) & 0xFFFFFFFF != expected_crc:
            raise ValueError(f"{path}: invalid PNG chunk checksum")
        if kind == b"IHDR":
            if width is not None or length != 13:
                raise ValueError(f"{path}: invalid PNG image header")
            width, height = struct.unpack_from(">II", body)
            if width == 0 or height == 0:
                raise ValueError(f"{path}: empty PNG dimensions")
        elif kind == b"pHYs":
            if density is not None or length != 9:
                raise ValueError(f"{path}: invalid PNG density metadata")
            x, y, unit = struct.unpack(">IIB", body)
            if unit != 1 or x == 0 or y == 0:
                raise ValueError(f"{path}: PNG must declare a positive physical density")
            density = x * 0.0254
        elif kind == b"IEND":
            if width is None or density is None:
                raise ValueError(f"{path}: missing PNG dimensions or physical density")
            return width, density
        cursor = end + 4
    raise ValueError(f"{path}: incomplete PNG")


def _paper_width_in(preamble: str) -> float:
    """Paper width in inches, read from the preamble's own declaration."""
    import re

    explicit = re.search(r"paperwidth\s*=\s*(\d+(?:\.\d+)?)(in|mm|cm)", preamble)
    if explicit:
        per_inch = {"in": 1.0, "mm": 1.0 / 25.4, "cm": 1.0 / 2.54}[explicit.group(2)]
        return float(explicit.group(1)) * per_inch
    for name, width in _PAPER_WIDTHS_IN.items():
        if re.search(rf"\b{re.escape(name)}\b", preamble):
            return width
    return _DEFAULT_PAPER_WIDTH_IN


def _on_page_width_in(figure_name: str, root: Path) -> float | None:
    """Declared print width, or None for an export with no manuscript include."""
    import re

    include_pct: float | None = None
    pattern = re.compile(
        r"\((?:\.\./)+output/figures/"
        + re.escape(figure_name)
        + r"\.png\)\{#fig:[a-z_0-9]+\s*(?:width=(\d+(?:\.\d+)?)%)?"
    )
    for section in sorted((root / "docs" / "manuscript").glob("*.md")):
        found = pattern.search(section.read_text(encoding="utf-8"))
        if found:
            include_pct = float(found.group(1)) if found.group(1) else 100.0
            break

    if include_pct is None:
        return None

    preamble = (root / "docs" / "manuscript" / "preamble.md").read_text(encoding="utf-8")
    from ailm.manuscript_contract import article_typography_bytes, article_typography_required

    if article_typography_required(root):
        from ailm.manuscript_variables.literature import _load_rendering_variables

        preamble += "\n" + article_typography_bytes(root, _load_rendering_variables(root)).decode(
            "utf-8"
        )
    if "{{" in preamble or "}}" in preamble:
        from ailm.manuscript_contract import resolve_manuscript_text
        from ailm.manuscript_variables.literature import _load_rendering_variables

        preamble = resolve_manuscript_text(preamble, _load_rendering_variables(root))
    # Explicit sides override an all-sides margin, as in LaTeX geometry.
    # Absence retains the historical default; a present malformed side refuses.
    margins: dict[str, float] = {}
    for match in re.finditer(r"\b(margin|left|right)\s*=\s*([^,}\n]+)", preamble):
        name, raw = match.groups()
        dimension = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(in|mm|cm)\s*", raw)
        if dimension is None:
            raise ValueError(f"invalid page geometry {name}={raw!r}")
        unit_scale = {"in": 1.0, "mm": 1.0 / 25.4, "cm": 1.0 / 2.54}
        margins[name] = float(dimension.group(1)) * unit_scale[dimension.group(2)]
    if (margins or re.search(r"\\geometry\s*\{", preamble)) and (
        "margin" not in margins and not {"left", "right"} <= margins.keys()
    ):
        raise ValueError("page geometry must specify both horizontal margins")
    default_margin = margins.get("margin", 1.0)
    linewidth_in = (
        _paper_width_in(preamble)
        - margins.get("left", default_margin)
        - margins.get("right", default_margin)
    )
    if linewidth_in <= 0:
        raise ValueError("page geometry must leave a positive text width")

    return linewidth_in * include_pct / 100.0


def on_page_point_size(
    figure_name: str,
    canvas_pt: float = MIN_FONT_PT,
    project: Path | None = None,
) -> float:
    """Printed size of ``canvas_pt`` text for one figure, in points.

    The canvas floor says nothing about the page. A figure is included at some
    ``width=NN%`` of the text width, so its on-page size is the canvas size times
    ``(linewidth * NN%) / canvas_width``. Everything is read from artifacts — the
    include width from the manuscript, the canvas width from the saved PNG, the
    page geometry from the preamble — so the number tracks what is actually
    shipped rather than what the figure code intended.

    A figure with no manuscript include is an error rather than a figure assumed
    to print at full text width: it is not on any page, so there is no on-page
    size to report, and silently assuming full width would let an unincluded figure
    pass the resolution gate that calls this.

    Raises:
        ValueError: If no manuscript section includes ``figure_name``.
    """
    root = project or project_root()
    printed_width = _on_page_width_in(figure_name, root)
    if printed_width is None:
        raise ValueError(
            f"{figure_name}.png is not included by any docs/manuscript/*.md section, so it "
            f"has no on-page size. Add the include, or delete the PNG if the figure "
            f"was retired — a figure nothing ships cannot be measured against the page."
        )

    png = root / "output" / "figures" / f"{figure_name}.png"
    width_px, raster_dpi = _png_width_and_density(png)
    canvas_in = width_px / raster_dpi
    return canvas_pt * printed_width / canvas_in


def effective_on_page_dpi(figure_name: str, project: Path | None = None) -> float:
    """Resolution a figure actually resolves at on the page, in dots per inch.

    A figure's own PNG density defines its rasterised canvas, and it is then
    then scaled to a fraction of the text width, so its pixels land closer
    together than the nominal DPI suggests: a wide canvas shrunk to the column
    resolves *higher*, not lower. This is the export-quality figure that matters
    when a reader zooms, and it moves opposite to on-page point size — shrinking
    a canvas enlarges the printed type while coarsening the raster.
    """
    root = project or project_root()
    png = root / "output" / "figures" / f"{figure_name}.png"
    width_px, raster_dpi = _png_width_and_density(png)
    scale = on_page_point_size(figure_name, MIN_FONT_PT, root) / MIN_FONT_PT
    return width_px / ((width_px / raster_dpi) * scale)


__all__ = [
    "ExportFigure",
    "FigureOverflowError",
    "IllegibleFigureError",
    "MissingVariableError",
    "TextRuleCollisionError",
    "assert_legible",
    "assert_no_text_crosses_rules",
    "assert_within_canvas",
    "default_output_dir",
    "load_manuscript_variables",
    "effective_on_page_dpi",
    "on_page_point_size",
    "manuscript_variables_path",
    "project_root",
    "rendered_texts",
    "resolve_variables",
    "save_figure",
    "text_extent",
    "token_float",
    "token_int",
]

#: Constant written into the PNG ``Software`` field, replacing matplotlib's
#: version string. Two builds on different matplotlib patch releases must produce
#: the same bytes for a byte-identity check to mean anything.
_PNG_SOFTWARE: str = "ailm"

#: Whitespace padding, in inches, around the tight bounding box.
_PAD_INCHES: float = 0.25


# ── Paths ────────────────────────────────────────────────────────────────────


def project_root() -> Path:
    """Resolve the AILM project root.

    Honours ``AILM_PROJECT_ROOT`` when set — the render pipeline may invoke a
    generator from a different working directory — and otherwise walks up from
    this file: ``src/ailm/figures/`` → ``src/ailm/`` → ``src/`` → root.

    Returns:
        Absolute path to the directory holding ``output/``, ``vendor/`` and
        ``docs/manuscript/``.
    """
    override = os.environ.get("AILM_PROJECT_ROOT", "").strip()
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parent.parent.parent.parent


def default_output_dir() -> Path:
    """Return ``<project root>/output/figures``, without creating it.

    Returns:
        The canonical figure output directory.
    """
    return project_root() / "output" / "figures"


def manuscript_variables_path() -> Path:
    """Return the path of the generated ``{{TOKEN}}`` map.

    Returns:
        ``<project root>/output/data/manuscript_variables.json``.
    """
    return project_root() / "output" / "data" / "manuscript_variables.json"


# ── Measured values ──────────────────────────────────────────────────────────


class MissingVariableError(RuntimeError):
    """Raised when a figure needs a measured token the ledger has not recorded.

    Failing loudly is the point. The alternative — a plausible default, or a
    literal left in the plotting code — puts an unmeasured number on a published
    axis, where no reader can tell it apart from a measured one.
    """


def load_manuscript_variables(path: Path | None = None) -> dict[str, str]:
    """Load the generated manuscript token map.

    Args:
        path: Explicit path to ``manuscript_variables.json``. Defaults to
            :func:`manuscript_variables_path`.

    Returns:
        Flat mapping of UPPERCASE token name to its string value, exactly as the
        manuscript renderer sees it.

    Raises:
        MissingVariableError: If the file does not exist, or does not contain a
            JSON object.
    """
    target = manuscript_variables_path() if path is None else Path(path)
    if not target.exists():
        raise MissingVariableError(
            f"Manuscript variables not found at {target}. Figures read measured "
            "values from the claims ledger rather than hardcoding them — run "
            "scripts/z_generate_manuscript_variables.py first."
        )
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MissingVariableError(
            f"{target} must contain a JSON object of token -> value, got {type(payload).__name__}."
        )
    return {str(k): str(v) for k, v in payload.items()}


def resolve_variables(variables: dict[str, str] | None) -> dict[str, str]:
    """Return ``variables`` unchanged, or load the ledger's token map if omitted.

    Every ``*_data()`` figure function accepts an explicit ``variables`` mapping
    for the manuscript renderer and callers who already have one, and falls back
    to :func:`load_manuscript_variables` when none is passed. Ten figure modules
    each defined this exact one-liner as a private ``_v`` helper; this is the
    single implementation they now import instead.
    """
    return variables if variables is not None else load_manuscript_variables()


def token_float(variables: dict[str, str], token: str) -> float:
    """Read a numeric token out of the manuscript variable map.

    Handles the two presentation forms the renderer emits: thousands separators
    (``"42,820"``) and scientific notation (``"3.47e-06"``).

    Args:
        variables: Mapping from :func:`load_manuscript_variables`.
        token: UPPERCASE token name, e.g. ``"STAGE1_LETTER_PERPLEXITY"``.

    Returns:
        The value as a float.

    Raises:
        MissingVariableError: If the token is absent, or its value does not parse
            as a number.
    """
    if token not in variables:
        raise MissingVariableError(
            f"Token {token!r} is not in the manuscript variable map. Record the "
            "claim from the run that measured it — do not put the number in the "
            f"figure code. Available tokens: {len(variables)}."
        )
    raw = variables[token].replace(",", "").strip().rstrip("%")
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise MissingVariableError(
            f"Token {token!r} has non-numeric value {variables[token]!r}."
        ) from exc


def token_int(variables: dict[str, str], token: str) -> int:
    """Read an integral token, rejecting a value that is not whole.

    Args:
        variables: Mapping from :func:`load_manuscript_variables`.
        token: UPPERCASE token name.

    Returns:
        The value as an int.

    Raises:
        MissingVariableError: If the token is absent, unparseable, or fractional.
    """
    value = token_float(variables, token)
    if value != int(value):
        raise MissingVariableError(f"Token {token!r} is {value}, which is not an integer.")
    return int(value)


# ── Legibility ───────────────────────────────────────────────────────────────


class IllegibleFigureError(AssertionError):
    """Raised when a figure contains type below the project's legibility floor.

    Subclasses ``AssertionError`` so it reads naturally in a test, but it is
    raised by production figure code, not only by tests: ``save_figure`` refuses
    to write a figure that fails this check.
    """


def _visible_texts(fig: Figure) -> list[Text]:
    """Collect every text artist on ``fig`` that a reader would actually see.

    Empty and whitespace-only labels are excluded — matplotlib creates one per
    unused tick, axis title and offset-text slot, and they carry no information.

    Args:
        fig: The figure to walk.

    Returns:
        Visible, non-empty text artists.
    """
    return [
        artist for artist in fig.findobj(Text) if artist.get_visible() and artist.get_text().strip()
    ]


class FigureOverflowError(AssertionError):
    """Raised when rendered text falls outside the figure canvas.

    Subclasses ``AssertionError`` so it reads naturally in a test.
    """


def _axis_text(axis) -> list[Text]:
    """Every text artist owned by one :class:`~matplotlib.axis.Axis`."""
    owned: list[Text] = [axis.label, axis.offsetText]
    for minor in (False, True):
        owned.extend(axis.get_ticklabels(minor=minor))
    return [artist for artist in owned if artist is not None]


def _undrawn_axis_text_ids(fig: Figure) -> set[int]:
    """Identify axis text that matplotlib holds but never rasterises.

    ``get_visible()`` is not sufficient to decide whether a label is drawn, so a
    geometric check that trusts it reports defects in figures that render
    correctly. Three mechanisms produce retained-but-undrawn text, all observed in
    this project's own figures:

    1. **The axis is switched off.** ``ax.set_axis_off()`` clears ``ax.axison``,
       and :meth:`matplotlib.axes.Axes.draw` then omits both ``Axis`` objects
       wholesale. Their tick labels still report ``get_visible() is True``. The
       ``hierarchy`` and ``inference_cycle`` schematics are full-bleed
       ``axis("off")`` canvases, and each retained 12 phantom ``0.0``–``1.0``
       labels sitting ~47 px below the canvas.
    2. **One axis is hidden**, via ``ax.xaxis.set_visible(False)``.
    3. **The tick lies outside the view interval.** A ``Locator`` proposes ticks
       for the whole data range, not the visible window, so a zoomed or inverted
       axis keeps an artist per proposed tick. ``information_ceiling`` shows
       ylim 1.12–3.70 while the locator proposes 1.0 and 4.0; ``gate_taxonomy``
       zooms to 57.4–75.4 while the locator proposes 56 and 76.

    The exemption is deliberately confined to text owned by an ``Axis``. Titles,
    suptitles, legend entries and ``fig.text`` annotations are never exempt, and
    an in-view tick label on a live axis is never exempt — so a real overflow is
    still reported. The positive controls in ``tests/test_figure_layout.py`` and
    ``tests/test_figures_results.py`` pin that.

    Args:
        fig: The figure to walk.

    Returns:
        ``id()`` of every axis-owned text artist that will not be drawn.
    """
    undrawn: set[int] = set()
    for ax in fig.axes:
        axis_off = not getattr(ax, "axison", True)
        for axis in (ax.xaxis, ax.yaxis):
            if axis_off or not axis.get_visible():
                undrawn.update(id(artist) for artist in _axis_text(axis))
                continue

            lo, hi = sorted(axis.get_view_interval())
            # A tick exactly on the boundary is drawn; allow for float error.
            span = hi - lo
            tol = abs(span) * 1e-9 if span else 1e-9
            for minor in (False, True):
                locs = list(axis.get_ticklocs(minor=minor))
                labels = list(axis.get_ticklabels(minor=minor))
                # Locator and formatter are expected to agree in length; zip
                # truncates rather than guessing if a custom axis disagrees.
                for loc, label in zip(locs, labels, strict=False):
                    if not (lo - tol <= loc <= hi + tol):
                        undrawn.add(id(label))
    return undrawn


def rendered_texts(fig: Figure) -> list[Text]:
    """Collect the text artists on ``fig`` that are actually rasterised.

    :func:`_visible_texts` minus the axis text identified by
    :func:`_undrawn_axis_text_ids`. Use this — not ``fig.findobj(Text)`` — for any
    geometric assertion about where text sits, because position is only meaningful
    for text that gets drawn.

    The figure must already be drawn for artist positions to be current;
    :func:`assert_within_canvas` draws before calling this.

    Args:
        fig: The figure to walk.

    Returns:
        Visible, non-empty, actually-drawn text artists.
    """
    undrawn = _undrawn_axis_text_ids(fig)
    return [artist for artist in _visible_texts(fig) if id(artist) not in undrawn]


def text_extent(artist: Text, renderer=None):
    """Display-space bounding box of an artist's **glyphs only**.

    :class:`~matplotlib.text.Annotation` overrides ``get_window_extent`` to return
    the union of its text box and its arrow, so measuring an annotation the usual
    way charges the arrow to the text. A callout that points at a labeled bar
    then always reads as a collision with that bar's label: ``subsample_bias``
    reported its ``'+1.691'`` value label overlapping the callout at a narrow
    display scale while the
    two glyph boxes were 209 px apart. Moving the text did not clear it, because
    the arrow tail is anchored to the bar and never moved.

    Calling the base :class:`~matplotlib.text.Text` implementation returns the
    glyph box for annotations and for plain text alike.

    Args:
        artist: The text artist to measure.
        renderer: Renderer to measure with; matplotlib's cached one when omitted.

    Returns:
        The glyph bounding box in display coordinates.
    """
    return Text.get_window_extent(artist, renderer=renderer)


def assert_within_canvas(fig: Figure, tol_px: float = 1.0) -> None:
    """Assert that every rendered label lies inside the figure canvas.

    Text that overruns the canvas is a layout defect: with
    ``bbox_inches="tight"`` it silently enlarges the written PNG, changing the
    image dimensions, and with a fixed bbox it is cropped away entirely.

    Args:
        fig: The figure to check. It is drawn first so text extents are current.
        tol_px: Slack in device pixels, absorbing antialiasing and rounding.

    Raises:
        FigureOverflowError: If any drawn label crosses the canvas edge, or if
            the figure has no drawn text at all — a containment check that passes
            on an empty canvas is not a check.
    """
    fig.canvas.draw()
    texts = rendered_texts(fig)
    if not texts:
        raise FigureOverflowError(
            "Figure has no drawn text, which would make this containment check "
            "vacuously true. Every AILM figure carries labeled axes."
        )

    bounds = fig.bbox
    offenders: list[str] = []
    for artist in texts:
        extent = text_extent(artist)
        overruns = []
        if extent.x0 < bounds.x0 - tol_px:
            overruns.append(f"left by {bounds.x0 - extent.x0:.0f}px")
        if extent.x1 > bounds.x1 + tol_px:
            overruns.append(f"right by {extent.x1 - bounds.x1:.0f}px")
        if extent.y0 < bounds.y0 - tol_px:
            overruns.append(f"bottom by {bounds.y0 - extent.y0:.0f}px")
        if extent.y1 > bounds.y1 + tol_px:
            overruns.append(f"top by {extent.y1 - bounds.y1:.0f}px")
        if overruns:
            offenders.append(f"{artist.get_text().strip()[:50]!r} overruns {', '.join(overruns)}")

    if offenders:
        listing = "; ".join(sorted(offenders))
        raise FigureOverflowError(
            f"{len(offenders)} label(s) fall outside the figure canvas: {listing}. "
            "Give the figure more room or shorten the label — do not shrink the "
            "type below the legibility floor."
        )


#: A rule counts as crossing a label when it penetrates the label's glyph box
#: deeper than this fraction of the box height. The full box includes ascent
#: and descent padding a flat rule can skirt without touching a glyph.
_RULE_BOX_SHRINK_FRACTION = 0.12

#: Minimum penetration, in device pixels, before a graze counts as a crossing.
_RULE_PENETRATION_PX = 1.0

#: Display-space tolerance, in device pixels, within which a probe point must
#: land on the axes edge for the stroke to count as declared in axes coords.

_RULE_ALIGNMENT_PX = 0.5


class TextRuleCollisionError(AssertionError):
    """Raised when a drawn reference rule runs through a rendered label.

    Subclasses ``AssertionError`` so it reads naturally in a test, but it is
    raised by production figure code: ``save_figure`` refuses to write a
    figure whose reference rules strike through their own annotations.
    """


def _axes_including_children(fig: Figure) -> list[Axes]:
    """Every axes on ``fig``, the inset children included.

    ``Figure.axes`` holds only the axes added at the top level; ``inset_axes``
    children hang off their parent's ``child_axes``. A geometric gate that
    walks ``fig.axes`` alone would leave every zoom inset a blind spot.
    """
    found: list[Axes] = []
    stack: list[Axes] = list(fig.axes)
    while stack:
        axes = stack.pop()
        found.append(axes)
        # ``child_axes`` is typed down to the private base class, but every
        # child an inset creates is an ``Axes``; narrow at this one boundary.
        stack.extend(child for child in axes.child_axes if isinstance(child, Axes))
    return found


def _effective_alpha(artist) -> float:
    """The artist's own alpha times the alpha carried by its facecolor."""
    alpha = artist.get_alpha()
    facecolor = artist.get_facecolor()
    face_alpha = float(facecolor[3]) if len(facecolor) == 4 else 1.0
    own = 1.0 if alpha is None else float(alpha)
    return own * face_alpha


def _declared_orientation(line: Line2D, axes: Axes) -> str | None:
    """Identify an explicit axes-coordinate rule from its transform.

    Compare the declared transforms, not transformed sample coordinates: a data
    view with limits 0..1 maps onto the axes edges too, without turning its data
    curves into reference rules. Matplotlib's horizontal-rule transform uses
    axes fractions for x and data for y; the vertical transform does the reverse.
    A wholly axes-relative separator can have either orientation.
    """
    transform = line.get_transform()
    if transform == axes.get_yaxis_transform():
        return "horizontal"
    if transform == axes.get_xaxis_transform():
        return "vertical"
    if transform == axes.transAxes:
        return "frame"
    return None


def _reference_rules(axes: Axes) -> list[Line2D]:
    """The declared reference rules drawn in one axes.

    The gate's scope is **explicit annotation and separator rules**: strokes
    using Matplotlib's standard x-axis, y-axis, or frame transform — including
    ``axhline``/``axvline`` and ``plot`` calls with those same transforms. Custom
    compositions (for example an additional pixel translation) are outside this
    detector; transformed coordinates alone do not establish a rule's semantics.
    A stroke in plain data coordinates is a data curve, however flat or however
    far it happens to span, and is out of scope by construction. Grid lines never
    reach here either: matplotlib owns them on the ``Axis``, not in
    ``axes.lines``, and they are structure behind the data by design.
    """
    rules: list[Line2D] = []
    for line in axes.lines:
        if not line.get_visible():
            continue
        if str(line.get_linestyle()).strip().lower() in {"", "none"}:
            continue
        if line.get_marker() not in ("", "None", "none", None):
            continue
        if (line.get_linewidth() or 0.0) <= 0.0 or line.get_alpha() == 0.0:
            continue
        orientation = _declared_orientation(line, axes)
        if orientation is None:
            continue
        points = line.get_transform().transform(np.asarray(line.get_xydata(), dtype=float))
        if len(points) < 2:
            continue
        x0, x1 = float(points[:, 0].min()), float(points[:, 0].max())
        y0, y1 = float(points[:, 1].min()), float(points[:, 1].max())
        if orientation in {"horizontal", "frame"} and y1 - y0 <= _RULE_ALIGNMENT_PX:
            rules.append(line)
        elif orientation in {"vertical", "frame"} and x1 - x0 <= _RULE_ALIGNMENT_PX:
            rules.append(line)
    return rules


def _opaque_background_zorder(text: Text, axes: Axes) -> float | None:
    """Zorder of the fully opaque background behind ``text``, if it has one.

    Two backgrounds can hide a rule: the text's own ``bbox`` patch, drawing
    with the text, and an opaque legend frame around a legend entry. A
    semi-transparent frame is not a shield — the rule shows through — and a
    background drawn *below* the rule does not hide it either, so the caller
    still compares zorders.
    """
    patch = text.get_bbox_patch()
    if patch is not None and patch.get_visible() and _effective_alpha(patch) >= 0.999:
        return text.get_zorder()
    legend = axes.get_legend()
    if legend is not None and legend.get_frame_on():
        if any(candidate is text for candidate in legend.get_texts()):
            frame = getattr(legend, "legendPatch", None)
            if frame is not None and frame.get_visible() and _effective_alpha(frame) >= 0.999:
                return legend.get_zorder()
    return None


def _rule_penetration(rule: Line2D, box: Bbox, pad: float, axes: Axes) -> tuple[str, float] | None:
    """How far ``rule`` penetrates the glyph box ``box``, or ``None``.

    ``box`` is the text's display-space glyph box already reduced to its
    visible (clipped) extent; ``pad`` is the edge allowance that keeps a rule
    skirting the box's ascent/descent padding from counting. The rule's drawn
    extent is clipped to the axes, because that is all a reader can see.
    """
    points = rule.get_transform().transform(np.asarray(rule.get_xydata(), dtype=float))
    x0, x1 = float(points[:, 0].min()), float(points[:, 0].max())
    y0, y1 = float(points[:, 1].min()), float(points[:, 1].max())
    bbox = axes.bbox
    if y1 - y0 <= _RULE_ALIGNMENT_PX:
        y = 0.5 * (y0 + y1)
        # A line the axes never draws — its level sits outside the view —
        # cannot strike anything, whatever an unclipped label does there.
        if not bbox.y0 - _RULE_ALIGNMENT_PX <= y <= bbox.y1 + _RULE_ALIGNMENT_PX:
            return None
        inner_lo, inner_hi = box.y0 + pad, box.y1 - pad
        if inner_lo <= y <= inner_hi:
            drawn_x0, drawn_x1 = max(x0, bbox.x0), min(x1, bbox.x1)
            if min(drawn_x1, box.x1) - max(drawn_x0, box.x0) > _RULE_PENETRATION_PX:
                depth = min(y - inner_lo, inner_hi - y)
                value = axes.transData.inverted().transform((0.0, y))[1]
                return (f"horizontal rule at y={value:.6g}", depth)
    elif x1 - x0 <= _RULE_ALIGNMENT_PX:
        x = 0.5 * (x0 + x1)
        if not bbox.x0 - _RULE_ALIGNMENT_PX <= x <= bbox.x1 + _RULE_ALIGNMENT_PX:
            return None
        inner_lo, inner_hi = box.x0 + pad, box.x1 - pad
        if inner_lo <= x <= inner_hi:
            drawn_y0, drawn_y1 = max(y0, bbox.y0), min(y1, bbox.y1)
            if min(drawn_y1, box.y1) - max(drawn_y0, box.y0) > _RULE_PENETRATION_PX:
                depth = min(x - inner_lo, inner_hi - x)
                value = axes.transData.inverted().transform((x, 0.0))[0]
                return (f"vertical rule at x={value:.6g}", depth)
    return None


def find_text_rule_crossings(fig: Figure) -> list[tuple[str, str, float]]:
    """Return ``(label, rule, penetration_px)`` for every rule through a label.

    The 2026-09-05 visual audit shipped five figures whose reference rules ran
    straight through their own annotation text. Neither
    :func:`assert_within_canvas` (the canvas edge) nor the text-on-text sweep
    can see that class, because a rule is not text; this is the drawn-rule
    sibling of the border check.

    Scope and exemptions, so intentional graphics do not fire:

    - a rule is compared only with text owned by the same axes — an inset's
      opaque patch covers the parent's rules, and figure-level text carries no
      axes and is out of scope;
    - text backed by a fully opaque background — its own ``bbox`` patch or an
      opaque legend frame — is exempt while that background draws above the
      rule, because the rule is hidden behind it; semi-transparent backgrounds
      are not shields;
    - clipped text counts by its visible extent, so a label clipped away from
      the rule is not reported;
    - the rule must *penetrate* the glyph box, not graze its padded edge.

    The figure is drawn first so artist positions are current.
    """
    fig.canvas.draw()

    texts_by_axes: dict[int, dict[int, Text]] = {}
    for text in rendered_texts(fig):
        if text.axes is None:
            continue
        bucket = texts_by_axes.setdefault(id(text.axes), {})
        bucket[id(text)] = text

    crossings: list[tuple[str, str, float]] = []
    for axes in _axes_including_children(fig):
        rules = _reference_rules(axes)
        if not rules:
            continue
        for text in texts_by_axes.get(id(axes), {}).values():
            box = text_extent(text)
            if text.get_clip_on():
                clip = text.get_clip_box()
                if clip is None:
                    clip = axes.bbox
                box = Bbox.intersection(box, clip)
            if box is None or box.width <= 0.0 or box.height <= 0.0:
                continue
            pad = max(_RULE_PENETRATION_PX, _RULE_BOX_SHRINK_FRACTION * box.height)
            shield = _opaque_background_zorder(text, axes)
            for rule in rules:
                if shield is not None and shield > rule.get_zorder():
                    continue
                hit = _rule_penetration(rule, box, pad, axes)
                if hit is not None:
                    crossings.append((text.get_text().strip()[:55], hit[0], hit[1]))
    return crossings


def assert_no_text_crosses_rules(fig: Figure) -> None:
    """Assert that no drawn reference rule runs through a rendered label.

    A rule striking through its own annotation reads as a strikethrough, and
    the 2026-09-05 visual audit found five shipped figures carrying exactly
    that. Legibility is a build-time invariant, so ``save_figure`` runs this
    beside the font-size and canvas-containment checks.

    Args:
        fig: The figure to check. It is drawn first so text extents are
            current.

    Raises:
        TextRuleCollisionError: If any reference rule crosses a drawn label.
    """
    crossings = find_text_rule_crossings(fig)
    if crossings:
        listing = "; ".join(
            f"{label!r} x {rule} ({depth:.0f}px deep)" for label, rule, depth in crossings
        )
        raise TextRuleCollisionError(
            f"{len(crossings)} label(s) struck through by a drawn reference rule: "
            f"{listing}. Move the label off the rule, shorten it, or give it an "
            "opaque background box — do not shrink the type below the legibility floor."
        )


def assert_legible(fig: Figure, min_pt: float = MIN_FONT_PT) -> None:
    """Assert that every visible label on ``fig`` is at least ``min_pt`` points.

    Point size is resolution-independent, so this is a statement about the
    CANVAS and not about the pixel grid: a 9 pt label is 9 pt on the canvas
    whether the PNG was rasterised at 100 or 200 dpi.

    It is **not** a statement about the printed page, and it was previously
    documented as if it were. LaTeX scales each figure to a fraction of the text
    width, so even a compliant canvas label can print much smaller when a wide
    canvas is scaled to fit a line. On-page size is canvas points times that
    include scale; :func:`on_page_point_size` is the diagnostic that computes
    it. What ``tests/test_figure_layout.py`` ratchets is the *resolution* the
    same scale implies — :func:`effective_on_page_dpi`, floored at 300 dpi —
    because these figures are meant to be read at zoom. No test constrains the
    printed point size, so do not read this paragraph as a second gate.

    Args:
        fig: A rendered or unrendered figure.
        min_pt: Legibility floor in points. Defaults to
            :data:`ailm.viz_config.MIN_FONT_PT`.

    Raises:
        IllegibleFigureError: If any visible label is smaller than ``min_pt``, or
            if the figure has no visible text at all. The second case matters
            because a check that cannot fail on an empty canvas is not a check —
            a generator that silently stopped emitting labels would otherwise
            pass.
    """
    texts = _visible_texts(fig)
    if not texts:
        raise IllegibleFigureError(
            "Figure has no visible text. Every AILM figure carries labeled "
            "axes or annotations, so an empty text set means the generator "
            "stopped emitting labels — and it would make this legibility check "
            "vacuously true."
        )

    offenders = sorted(
        {
            (round(float(artist.get_fontsize()), 3), artist.get_text().strip()[:60])
            for artist in texts
            if float(artist.get_fontsize()) < min_pt
        }
    )
    if offenders:
        listing = "; ".join(f"{size:g}pt {text!r}" for size, text in offenders)
        raise IllegibleFigureError(
            f"{len(offenders)} label(s) below the {min_pt:g}pt legibility floor: "
            f"{listing}. Legibility is a build-time invariant — enlarge the type "
            "or give the figure more room. Do not clamp."
        )


# ── Writing ──────────────────────────────────────────────────────────────────


_MIN_ON_PAGE_EXPORT_DPI = 300.0
_MAX_PRINT_DENSITY_ATTEMPTS = 3


def _export_project_root(target_dir: Path) -> Path:
    if target_dir.name == "figures" and target_dir.parent.name == "output":
        candidate = target_dir.parent.parent
        if (candidate / "docs" / "manuscript").is_dir():
            return candidate
    return project_root()


def save_figure(
    fig: Figure,
    name: str,
    output_dir: Path | None = None,
    *,
    check_legible: bool = True,
) -> Path:
    """Write ``fig`` to ``<output_dir>/<name>.png`` deterministically.

    Density starts at the configured export value and is raised when actual
    tight-cropped pixels do not meet the declared manuscript print width. Each
    attempt redraws the vector artists, preserving their canvas, type and data.
    Unreferenced diagnostic exports retain their configured density because they
    have no declared print width. Equal inputs produce byte-identical PNGs;
    padding and Software metadata are pinned and no timestamp is written.

    Args:
        fig: The figure to write.
        name: Base filename without extension, e.g. ``"hierarchy"``. A ``.png``
            suffix is accepted and stripped.
        output_dir: Destination directory, created if absent. Defaults to
            :func:`default_output_dir`.
        check_legible: Run :func:`assert_legible`, :func:`assert_within_canvas`
            and :func:`assert_no_text_crosses_rules` before writing. Leave it
            on — the point of the visual gate is that an illegible, clipped or
            rule-struck figure never reaches ``output/``. Turning it off is
            only for exercising the gate.

    Returns:
        The path written.

    Raises:
        ValueError: If ``name`` is empty or contains a path separator.
        IllegibleFigureError: If ``check_legible`` is set and the figure contains
            type below the legibility floor.
        TextRuleCollisionError: If ``check_legible`` is set and a drawn
            reference rule strikes through a rendered label.
    """
    stem = name[:-4] if name.lower().endswith(".png") else name
    if not stem.strip():
        raise ValueError("Figure name must be a non-empty stem, e.g. 'hierarchy'.")
    if "/" in stem or "\\" in stem:
        raise ValueError(
            f"Figure name {name!r} must be a bare stem, not a path — the "
            "destination is chosen by output_dir."
        )

    if check_legible:
        assert_legible(fig)
        assert_within_canvas(fig)
        assert_no_text_crosses_rules(fig)

    target_dir = default_output_dir() if output_dir is None else Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{stem}.png"
    descriptor, scratch_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.png", dir=target_dir
    )
    scratch = Path(scratch_name)
    try:
        # The image is a public artifact, not a credential-bearing temporary.
        # Preserve the conventional read permission after os.replace publishes
        # the mkstemp-created inode.
        os.fchmod(descriptor, 0o644)
        os.close(descriptor)
        descriptor = -1
        printed_width = _on_page_width_in(stem, _export_project_root(target_dir))
        export_dpi = fig.export_dpi if isinstance(fig, ExportFigure) else DPI
        for _attempt in range(_MAX_PRINT_DENSITY_ATTEMPTS):
            # Rerender the original vector artists; never upscale saved pixels.
            fig.savefig(
                scratch,
                dpi=export_dpi,
                bbox_inches="tight",
                pad_inches=_PAD_INCHES,
                facecolor=PALETTE["BACKGROUND"],
                edgecolor="none",
                metadata={"Software": _PNG_SOFTWARE},
            )
            width_pixels, actual_density = _png_width_and_density(scratch)
            if printed_width is None or width_pixels / printed_width >= _MIN_ON_PAGE_EXPORT_DPI:
                break
            required_pixels = math.ceil(printed_width * _MIN_ON_PAGE_EXPORT_DPI)
            export_dpi = max(
                export_dpi + 1,
                math.ceil(actual_density * (required_pixels + 1) / width_pixels),
            )
        else:
            raise RuntimeError(
                f"{stem}.png cannot meet {_MIN_ON_PAGE_EXPORT_DPI:g} dpi at its "
                f"declared print width after vector rerendering"
            )

        # A successful savefig return is necessary but not sufficient: verify
        # the complete image can be decoded and has a non-empty PNG canvas
        # before it is allowed to replace a previously published figure.
        header = scratch.read_bytes()[:24]
        if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
            raise RuntimeError(f"matplotlib did not produce a valid PNG for {path.name}")
        width, height = struct.unpack(">II", header[16:24])
        if width <= 0 or height <= 0:
            raise RuntimeError(f"matplotlib produced an empty PNG canvas for {path.name}")
        pixels = matplotlib_image.imread(scratch)
        if pixels.ndim not in {2, 3} or any(size <= 0 for size in pixels.shape[:2]):
            raise RuntimeError(f"matplotlib could not decode the complete PNG for {path.name}")

        with scratch.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(scratch, path)
        directory_fd = os.open(target_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if scratch.exists():
            scratch.unlink()
    return path
