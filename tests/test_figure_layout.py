"""Geometric legibility: no overlapping text, nothing clipped out of frame.

``ailm.figures._common.assert_legible`` checks font *size* and that some text
exists. Both are necessary and neither is sufficient: the two defects that
actually make a rendered figure unreadable are text drawn on top of other text,
and text pushed outside the canvas. A 10 pt label is not legible if another 10 pt
label sits on it, so a size-only gate is a partially vacuous legibility gate.

This module closes that hole for every figure the project generates, and carries
positive controls proving each check can fail — a geometric assertion that never
fires would be worse than none, because it would license the belief that layout
had been verified.

The checks operate on rendered display coordinates, so they measure what a reader
sees rather than what the code intended.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox, ScaledTranslation

from ailm.figures._common import (
    MissingVariableError,
    TextRuleCollisionError,
    assert_no_text_crosses_rules,
    find_text_rule_crossings,
    rendered_texts,
    text_extent,
)

#: Permitted overlap between two text bounding boxes, as a fraction of the area of
#: the smaller box. Rendered glyph boxes include a little padding and adjacent
#: tick labels legitimately touch, so a hard zero produces false positives.
OVERLAP_TOLERANCE = 0.22

#: How far outside the figure a text box may extend, in display points. Anti-
#: aliasing and descenders make an exact bound too strict.
CLIP_TOLERANCE_PT = 2.0


def _rendered_text_boxes(fig: Figure) -> list[tuple[object, object]]:
    """Return (artist, display-space bbox) for every text artist that is drawn.

    Membership comes from :func:`ailm.figures._common.rendered_texts` rather than a
    local ``findobj`` walk. A ``Locator`` proposes ticks for the whole data range,
    so a zoomed or inverted axis — ``information_ceiling`` shows ylim 1.12–3.70
    while the locator proposes 1.0 and 4.0 — retains a ``Text`` artist per
    proposed tick, positioned off-canvas and never rasterised. Counting those
    reports clipping for a figure that renders correctly, so geometry must be
    measured over drawn text only. ``TestChecksCanFail`` keeps both detectors
    live under that narrowing.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for artist in rendered_texts(fig):
        try:
            # Glyph box only. An Annotation's own get_window_extent unions its
            # text with its arrow, so a callout pointing at a labeled bar reads
            # as colliding with that label however far the text is moved.
            bbox = text_extent(artist, renderer=renderer)
        except (RuntimeError, ValueError):  # unrenderable artist
            continue
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        boxes.append((artist, bbox))
    return boxes


def _line_boxes(boxes):
    """Split each multi-line text box into one box per rendered line.

    Args:
        boxes: ``(artist, block bbox)`` pairs as returned by
            :func:`_rendered_text_boxes`.

    Returns:
        ``(artist, line bbox)`` pairs — one entry per line of each artist. The
        horizontal extent of a line is not known from the block box, so each band
        keeps the block's full width. That is conservative in the safe direction:
        it can over-report a clash between two labels that only *nearly* touch, and
        the tolerance absorbs that, but it cannot under-report a real one.
    """
    split = []
    for artist, box in boxes:
        lines = artist.get_text().split("\n")
        count = max(len(lines), 1)
        if count == 1:
            split.append((artist, box))
            continue
        height = box.height / count
        for index in range(count):
            # Line 0 is the topmost band in display space.
            y1 = box.y1 - index * height
            split.append((artist, Bbox.from_extents(box.x0, y1 - height, box.x1, y1)))
    return split


def find_text_overlaps(fig: Figure, tolerance: float = OVERLAP_TOLERANCE):
    """Return a list of (text_a, text_b, overlap_fraction) for colliding labels.

    Overlap is measured against the smaller of the two boxes, so a small label
    buried under a large block of prose is reported at close to 1.0 rather than
    being diluted by the large box's area.

    Multi-line labels are charged **per line**, not per block. A three-line tick
    label whose first line prints through its neighbor's first line overlaps only
    a third of the block, and ``subsample_bias`` slipped under this tolerance at
    14% while "alphabetical prefix[:1200]" was visibly overprinting
    "stride[::5] truncated[:1200]". Dividing each block into equal horizontal
    bands — matplotlib lays multi-line text out on a uniform baseline pitch — and
    comparing bands makes the fraction reflect what a reader sees. The tolerance
    itself is unchanged; only the denominator stops being diluted.
    """
    boxes = _line_boxes(_rendered_text_boxes(fig))
    collisions = []
    for i in range(len(boxes)):
        artist_a, box_a = boxes[i]
        for j in range(i + 1, len(boxes)):
            artist_b, box_b = boxes[j]
            if artist_a is artist_b:
                continue  # two bands of the same label are not a collision
            dx = min(box_a.x1, box_b.x1) - max(box_a.x0, box_b.x0)
            dy = min(box_a.y1, box_b.y1) - max(box_a.y0, box_b.y0)
            if dx <= 0 or dy <= 0:
                continue
            smaller = min(box_a.width * box_a.height, box_b.width * box_b.height)
            if smaller <= 0:
                continue
            fraction = (dx * dy) / smaller
            if fraction > tolerance:
                collisions.append(
                    (
                        artist_a.get_text().strip()[:55],
                        artist_b.get_text().strip()[:55],
                        round(fraction, 3),
                    )
                )
    return collisions


def find_clipped_text(fig: Figure, tolerance_pt: float = CLIP_TOLERANCE_PT):
    """Return a list of (text, overhang_pt) for labels falling outside the canvas."""
    fig.canvas.draw()
    figure_box = fig.bbox
    clipped = []
    for artist, box in _rendered_text_boxes(fig):
        overhang = max(
            figure_box.x0 - box.x0,
            box.x1 - figure_box.x1,
            figure_box.y0 - box.y0,
            box.y1 - figure_box.y1,
        )
        if overhang > tolerance_pt:
            clipped.append((artist.get_text().strip()[:55], round(float(overhang), 1)))
    return clipped


def assert_layout_is_clean(fig: Figure, name: str) -> None:
    """Assert a figure has neither overlapping nor clipped text."""
    overlaps = find_text_overlaps(fig)
    clipped = find_clipped_text(fig)
    problems = []
    if overlaps:
        listing = "; ".join(f"{a!r} x {b!r} ({f:.0%})" for a, b, f in overlaps[:6])
        problems.append(f"{len(overlaps)} overlapping label pair(s): {listing}")
    if clipped:
        listing = "; ".join(f"{t!r} by {o}pt" for t, o in clipped[:6])
        problems.append(f"{len(clipped)} clipped label(s): {listing}")
    assert not problems, (
        f"Figure {name!r} has layout defects — legibility is a build-time "
        f"invariant, so give the figure more room or shorten the text rather than "
        f"shrinking type. " + " | ".join(problems)
    )


# ── POSITIVE CONTROLS: each check must be able to fail ───────────────────────


class TestChecksCanFail:
    """Without these, a clean report would prove nothing about the checks."""

    def test_overlap_detector_catches_stacked_labels(self):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "COLLIDING TEXT ONE", ha="center")
        ax.text(0.5, 0.5, "COLLIDING TEXT TWO", ha="center")
        try:
            overlaps = find_text_overlaps(fig)
            assert overlaps, "two labels drawn at the same point were not flagged"
        finally:
            plt.close(fig)

    def test_overlap_detector_accepts_separated_labels(self):
        """NEGATIVE CONTROL — a clean figure must not be reported."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.1, 0.9, "top left")
        ax.text(0.1, 0.1, "bottom left")
        try:
            assert find_text_overlaps(fig) == []
        finally:
            plt.close(fig)

    def test_clip_detector_catches_out_of_frame_text(self):
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.text(0.5, 0.5, "X" * 200, ha="center")
        try:
            clipped = find_clipped_text(fig)
            assert clipped, "a label far wider than the canvas was not flagged"
        finally:
            plt.close(fig)

    def test_clip_detector_accepts_in_frame_text(self):
        """NEGATIVE CONTROL — text inside the canvas must not be reported."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "short", ha="center")
        try:
            assert find_clipped_text(fig) == []
        finally:
            plt.close(fig)

    def test_clip_detector_still_catches_an_in_view_tick_label(self):
        """The undrawn-tick narrowing must not exempt tick labels that are drawn.

        Restricting the sweep to rendered text could have silently exempted every
        tick label. This pins that only *out-of-view* ticks are dropped.
        """
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.set_xlim(0, 1)
        ax.set_xticks([0.0])
        ax.set_xticklabels(["W" * 140], fontsize=12)
        try:
            assert find_clipped_text(fig), (
                "an in-view tick label far wider than the canvas was not flagged"
            )
        finally:
            plt.close(fig)

    def test_out_of_view_tick_labels_are_not_reported_as_clipped(self):
        """NEGATIVE CONTROL — the specific false positive this narrowing removes."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xticks([0.0, 0.5, 1.0, 40.0])  # 40.0 is far outside the view
        ax.set_xlim(0.0, 1.0)
        try:
            assert find_clipped_text(fig) == [], "an undrawn tick label was reported as clipped"
        finally:
            plt.close(fig)

    def test_an_arrow_is_not_charged_to_the_annotation_text(self):
        """NEGATIVE CONTROL — a callout arrow reaching a label is not a collision.

        ``Annotation.get_window_extent`` unions the text box with the arrow, so
        measuring it that way makes any callout pointing at a labeled feature
        overlap that label permanently, no matter where the text is placed.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.text(2.0, 5.0, "target label", ha="center", va="center")
        ax.annotate(
            "callout text",
            xy=(2.0, 5.0),
            xytext=(8.0, 8.0),
            arrowprops={"arrowstyle": "->"},
        )
        try:
            assert find_text_overlaps(fig) == [], (
                "the arrow spanning the gap was charged to the annotation's text"
            )
        finally:
            plt.close(fig)

    def test_annotation_text_itself_is_still_checked_for_overlap(self):
        """POSITIVE CONTROL — excluding the arrow must not exempt annotation text.

        Otherwise the previous control would license ignoring annotations
        entirely, and a callout printed on top of another label would pass.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.text(5.0, 5.0, "UNDERNEATH", ha="center", va="center")
        ax.annotate(
            "UNDERNEATH",
            xy=(1.0, 1.0),
            xytext=(5.0, 5.0),
            ha="center",
            va="center",
            arrowprops={"arrowstyle": "->"},
        )
        try:
            assert find_text_overlaps(fig), (
                "annotation text drawn on top of another label was not flagged"
            )
        finally:
            plt.close(fig)

    def test_one_line_clash_in_a_multiline_block_is_not_diluted(self):
        """POSITIVE CONTROL for the per-line split.

        Two three-line labels whose top lines overprint each other overlap only a
        third of either block. Measured per block that is ~33% at full horizontal
        overlap and far less at partial overlap — which is how ``subsample_bias``
        shipped a visible collision at 14%, under the tolerance. Measured per line
        the clash is charged at its true severity.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        # Only the first lines are wide enough to reach each other.
        ax.text(3.0, 5.0, "WIDE TOP LINE AAAA\nnarrow\nnarrow", ha="left", va="top")
        ax.text(6.0, 5.0, "WIDE TOP LINE BBBB\nnarrow\nnarrow", ha="right", va="top")
        try:
            assert find_text_overlaps(fig), (
                "a first-line collision was diluted by the lines that do not clash"
            )
        finally:
            plt.close(fig)

    def test_disjoint_multiline_blocks_are_not_reported(self):
        """NEGATIVE CONTROL — splitting into lines must not invent collisions."""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.text(1.0, 8.0, "left one\nleft two\nleft three", ha="left", va="top")
        ax.text(9.0, 8.0, "right one\nright two\nright three", ha="right", va="top")
        try:
            assert find_text_overlaps(fig) == [], (
                "two well-separated multi-line blocks were reported as colliding"
            )
        finally:
            plt.close(fig)

    def test_assert_layout_is_clean_raises_on_a_bad_figure(self):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "OVERLAP A", ha="center")
        ax.text(0.5, 0.5, "OVERLAP B", ha="center")
        try:
            with pytest.raises(AssertionError, match="layout defects"):
                assert_layout_is_clean(fig, "deliberately_bad")
        finally:
            plt.close(fig)


# ── EVERY PROJECT FIGURE MUST HAVE CLEAN LAYOUT ──────────────────────────────


def _discover_builder_candidates():
    """Discover every zero-argument figure builder exported by ailm.figures.

    Discovery rather than a hardcoded list, so a newly added figure is covered
    automatically instead of silently escaping the check.
    """
    import importlib
    import inspect
    import pkgutil

    import ailm.figures as figures_pkg

    builders = []
    for module_info in pkgutil.iter_modules(figures_pkg.__path__):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"ailm.figures.{module_info.name}")
        for attr_name, attr in vars(module).items():
            # `generate_` is the convention this project actually uses
            # (generate_hierarchy_figure, generate_wiring_comparison_figure...).
            # It was missing from this list at first, which silently excluded four
            # of the nine figures — including two with confirmed defects. The
            # completeness assertion below exists so that class of omission fails
            # loudly instead of reading as a clean sweep.
            if not attr_name.startswith(("build_", "make_", "plot_", "figure_", "generate_")):
                continue
            if not callable(attr):
                continue
            signature = inspect.signature(attr)
            required = [
                p
                for p in signature.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            if required:
                continue
            builders.append((module_info.name, attr_name, attr))
    return sorted(builders)


def _figure_stem(attr_name: str) -> str:
    """Derive the output stem the way scripts/generate_figures.py does.

    ``generate_perplexity_ladder`` → ``perplexity_ladder`` and
    ``generate_hierarchy_figure`` → ``hierarchy``. The sweep compares these
    stems against the PNGs actually shipped in output/figures, so it must be
    the same derivation the publication pipeline writes with.
    """
    for prefix in ("generate_", "build_", "make_", "plot_", "figure_"):
        if attr_name.startswith(prefix):
            return attr_name[len(prefix) :].removesuffix("_figure")
    return attr_name


_SAVE_CALL_NAMES = frozenset({"save_figure", "savefig"})


def _save_writer_calls(module, function_name: str) -> bool:
    """Whether the named function's own body calls ``save_figure``/``savefig``.

    The project convention is that a builder returns a Figure and only a save
    writer persists one, through ``save_figure``. A callable the sweep is about
    to invoke must visibly contain no such call; anything else would make a
    test write a publication artifact. This is a static read of the real
    module source, so it runs without building anything. An indirect save
    through a helper is not visible here — the module-scoped
    publication-write guard below is the behavioral net for that case.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        is_target = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name == function_name
        )
        if not is_target:
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            called = sub.func
            if isinstance(called, ast.Name) and called.id in _SAVE_CALL_NAMES:
                return True
            if isinstance(called, ast.Attribute) and called.attr in _SAVE_CALL_NAMES:
                return True
    return False


def _resolve_builder(module_name: str, module, attr_name: str, attr):
    """Return the callable that builds this entry's Figure in memory.

    A ``generate_*`` function that persists its figure is never called by this
    sweep: calling it would write into output/figures. Its ``build_*_figure``
    (or ``build_*``) sibling is used instead. A save writer with no sibling is
    a defect and fails discovery loudly — never silently called, never
    silently dropped. Whatever callable the sweep will invoke is statically
    checked (``_save_writer_calls``) to contain no save call.
    """
    resolved = attr
    if attr_name.startswith("generate_"):
        stem = attr_name[len("generate_") :]
        for candidate_name in (f"build_{stem}_figure", f"build_{stem}"):
            candidate = getattr(module, candidate_name, None)
            if callable(candidate):
                resolved = candidate
                break
    if _save_writer_calls(module, resolved.__name__):
        raise AssertionError(
            f"{module_name}.{resolved.__name__} persists a figure "
            "(save_figure/savefig) and the layout sweep refuses to run publication "
            "writers. Give the module an in-memory build_*_figure sibling and "
            "point generate_* at it."
        )
    return resolved


def _figure_inventory():
    """One entry per figure: ``(stem, display_name, builder)``.

    Resolution happens here, once, so the whole module builds each figure
    exactly once: the same figure reached through both a ``generate_*``
    wrapper and its ``build_*`` sibling collapses to one builder. Two
    different callables claiming one stem is a defect, not a dedupe. The
    result is sorted by stem for deterministic parametrization.
    """
    import importlib

    inventory: dict[str, tuple[str, object]] = {}
    for module_name, attr_name, attr in _discover_builder_candidates():
        module = importlib.import_module(f"ailm.figures.{module_name}")
        builder = _resolve_builder(module_name, module, attr_name, attr)
        stem = _figure_stem(attr_name)
        display = f"{module_name}.{builder.__name__}"
        existing = inventory.get(stem)
        if existing is not None:
            if existing[1] is not builder:
                raise AssertionError(
                    f"two different callables claim figure stem {stem!r}: "
                    f"{existing[0]} and {display}"
                )
            continue
        inventory[stem] = (display, builder)
    return sorted((stem, display, builder) for stem, (display, builder) in inventory.items())


_FIGURE_INVENTORY = _figure_inventory()

_FIGURES_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"


def _layout_figure(stem, builder):
    """Build the live Figure for one inventory entry.

    The inventory has already resolved every entry to an in-memory builder, so
    this never runs a publication writer. A builder that still returns
    anything but a Figure (or a ``(Figure, ...)`` tuple) is surfaced, not
    invoked harder.
    """
    result = builder()
    fig = result[0] if isinstance(result, tuple) else result
    if isinstance(fig, Figure):
        return fig
    raise AssertionError(
        f"{stem}: builder {builder.__name__} returned {type(fig).__name__}, not a "
        "Figure — the entry must resolve to an in-memory build_*_figure sibling, "
        "not a save writer."
    )


def _missing_shipped_figures(shipped, discovered_stems):
    """Shipped PNG stems with no discovered builder — the escape list."""
    return sorted(set(shipped) - set(discovered_stems))


@pytest.fixture(scope="module", autouse=True)
def _no_publication_writes():
    """No build in this module may write into output/figures.

    The inventory resolves every save-writing ``generate_*`` to its in-memory
    sibling, so a write here means a builder saves somewhere the structural
    check cannot see (an indirect call). The published PNGs are snapshotted
    before the first build and compared after the last teardown;
    byte-identical regeneration still replaces the file and moves its mtime,
    so any rewrite — or any leftover ``.tmp.png`` scratch file — fails.
    """

    def _snapshot():
        if not _FIGURES_DIR.is_dir():
            return {}
        return {
            p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in _FIGURES_DIR.glob("*.png")
        }

    before = _snapshot()
    yield
    after = _snapshot()
    assert after == before, (
        "a figure build wrote into output/figures during this module; tests must "
        "not save publication artifacts. Created: "
        f"{sorted(set(after) - set(before))}, removed: "
        f"{sorted(set(before) - set(after))}, modified: "
        f"{sorted(name for name in set(before) & set(after) if after[name] != before[name])}."
    )


@pytest.fixture(
    scope="module",
    params=_FIGURE_INVENTORY,
    ids=[display for _, display, _ in _FIGURE_INVENTORY],
)
def built_figure(request, _no_publication_writes):
    """One real build per shipped figure, shared by the three layout sweeps.

    ``test_figure_layout_is_clean``, ``test_legend_colors_are_distinguishable``
    and ``test_no_label_is_struck_through_by_a_reference_rule`` examine the same
    figure three ways; the fixture builds each entry once and closes it at
    teardown, and the assertions remain independent tests. Save-writing
    ``generate_*`` wrappers are resolved to their build siblings at discovery,
    so a build here never persists a PNG. A builder whose manuscript tokens are
    absent skips here — once, for all three — exactly as each sweep skipped on
    its own before.
    """
    stem, _display, builder = request.param
    try:
        fig = _layout_figure(stem, builder)
    except MissingVariableError as exc:
        pytest.skip(f"{stem} requires manuscript tokens not yet in the ledger: {exc}")
    try:
        yield stem, fig
    finally:
        plt.close(fig)


def test_at_least_one_figure_builder_was_discovered():
    """Guard the discovery: an empty inventory would make the sweep vacuous."""
    assert _FIGURE_INVENTORY, (
        "No zero-argument figure builders discovered in ailm.figures. The layout "
        "sweep below would pass without examining anything."
    )


def test_discovery_covers_every_shipped_figure():
    """Completeness, not just non-emptiness — by stem identity, not by count.

    A name-prefix heuristic silently under-collects the moment someone names a
    builder differently, and a sweep over a subset reads exactly like a sweep
    over everything. A *count* comparison is weaker still: extra or duplicate
    builders can satisfy ``len(checkable) >= len(shipped)`` while a real figure
    escapes unchecked. The shipped PNG stems and the discovered stems are
    therefore compared as sets — every figure on disk must have its own
    builder, so an omission is named, not diluted.

    This is the stronger sibling of the assertion above, which only checked
    that *something* was found. That weaker form passed while four of nine
    figures went unexamined.
    """
    if not _FIGURES_DIR.is_dir():
        pytest.skip("output/figures/ absent; run scripts/generate_figures.py")
    shipped = sorted(p.stem for p in _FIGURES_DIR.glob("*.png"))
    if not shipped:
        pytest.skip("no PNGs generated yet")

    stems = {stem for stem, _, _ in _FIGURE_INVENTORY}
    missing = _missing_shipped_figures(shipped, stems)
    assert not missing, (
        f"{len(missing)} shipped figure(s) have no discovered builder and escape "
        f"the layout sweep entirely: {missing}. Shipped: {shipped}. "
        f"Discovered stems: {sorted(stems)}."
    )


# ── COMPLETENESS AND RESOLUTION CONTROLS ─────────────────────────────────────


def test_an_omitted_builder_is_caught_despite_unrelated_extra_builders():
    """POSITIVE CONTROL — a missing figure cannot hide behind extra builders.

    Take the real shipped set and the real inventory, drop one figure's
    builder, and add an unrelated extra builder. The old count comparison
    (``len(checkable) >= len(shipped)``) would still pass that state; the
    stem-set comparison must name the exact escaped figure.
    """
    if not _FIGURES_DIR.is_dir():
        pytest.skip("output/figures/ absent; run scripts/generate_figures.py")
    shipped = {p.stem for p in _FIGURES_DIR.glob("*.png")}
    if not shipped:
        pytest.skip("no PNGs generated yet")
    stems = {stem for stem, _, _ in _FIGURE_INVENTORY}

    assert _missing_shipped_figures(shipped, stems) == [], (
        "the real inventory is already incomplete; the control below would be meaningless"
    )
    victim = sorted(stems & shipped)[0]
    poisoned = (stems - {victim}) | {"a_totally_unrelated_builder"}
    assert _missing_shipped_figures(shipped, poisoned) == [victim], (
        f"dropping {victim!r} behind an unrelated extra builder went undetected"
    )


def test_a_save_writer_with_a_build_sibling_is_resolved_to_the_sibling(tmp_path: Path):
    """POSITIVE CONTROL — the writer is never invoked; the sibling is used instead."""
    import importlib

    (tmp_path / "ailm_layout_probe_pair.py").write_text(
        "from ailm.figures._common import save_figure\n"
        "\n"
        "\n"
        "WAS_CALLED = False\n"
        "\n"
        "\n"
        "def build_widget():\n"
        "    return 'FIGURE'\n"
        "\n"
        "\n"
        "def generate_widget(output_dir=None):\n"
        "    global WAS_CALLED\n"
        "    WAS_CALLED = True\n"
        "    save_figure(object(), 'widget')\n"
        "    return output_dir\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    try:
        probe = importlib.import_module("ailm_layout_probe_pair")
        resolved = _resolve_builder(
            "ailm_layout_probe_pair", probe, "generate_widget", probe.generate_widget
        )
        assert resolved is probe.build_widget
        assert probe.WAS_CALLED is False, "the save-writing generate_* wrapper was invoked"
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("ailm_layout_probe_pair", None)


def test_a_save_writer_without_a_build_sibling_is_refused(tmp_path: Path):
    """POSITIVE CONTROL — a writer with no in-memory sibling fails discovery.

    Refusing here is what keeps the sweep from ever running a publication
    writer: the module is real, the check is structural, and the failure
    names the missing sibling instead of writing output/figures/widget.png.
    """
    import importlib

    (tmp_path / "ailm_layout_probe_writer.py").write_text(
        "from ailm.figures._common import save_figure\n"
        "\n"
        "\n"
        "def generate_widget(output_dir=None):\n"
        "    save_figure(object(), 'widget')\n"
        "    return output_dir\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    try:
        probe = importlib.import_module("ailm_layout_probe_writer")
        with pytest.raises(AssertionError, match="save_figure"):
            _resolve_builder(
                "ailm_layout_probe_writer", probe, "generate_widget", probe.generate_widget
            )
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("ailm_layout_probe_writer", None)


@pytest.mark.parametrize(
    ("attr_name", "expected"),
    [
        ("generate_hierarchy_figure", "hierarchy"),
        ("generate_incontext_sweep", "incontext_sweep"),
        ("build_belief_dynamics_figure", "belief_dynamics"),
        ("make_panel", "panel"),
    ],
)
def test_figure_stem_derivation_matches_the_pipeline_naming(attr_name, expected):
    """The sweep's stems must be the stems the publication pipeline writes."""
    assert _figure_stem(attr_name) == expected


@pytest.mark.timeout(180)
def test_figure_layout_is_clean(built_figure):
    """No project figure may ship with overlapping or clipped text."""
    name, fig = built_figure
    assert_layout_is_clean(fig, name)


# ── LEGEND DISTINGUISHABILITY ────────────────────────────────────────────────
# A legend whose entries differ only in their label text is unreadable in the
# drawing itself: the reader cannot tell which curve is which. belief_dynamics
# shipped exactly this — PALETTE["MODEL"] and PALETTE["LEARNED"] are the same
# blue, so two entropy traces were identical in color and linestyle.


def find_indistinguishable_legend_entries(fig: Figure):
    """Return (legend_title_or_axes, label_a, label_b) for visually identical entries.

    Two legend entries are charged when their handles share both color and
    linestyle (and, for lines, marker; for patches, hatch) — everything that
    could tell them apart in the drawing besides position.
    """
    from matplotlib.colors import to_hex
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    clashes = []
    legends = [ax.get_legend() for ax in fig.axes] + list(fig.legends)
    for index, legend in enumerate(legends):
        if legend is None:
            continue
        seen: dict[tuple, str] = {}
        for handle, text in zip(legend.legend_handles, legend.get_texts(), strict=False):
            if isinstance(handle, Line2D):
                key = (
                    "line",
                    to_hex(handle.get_color(), keep_alpha=True),
                    str(handle.get_linestyle()),
                    str(handle.get_marker()),
                )
            elif isinstance(handle, Patch):
                key = (
                    "patch",
                    to_hex(handle.get_facecolor(), keep_alpha=True),
                    str(handle.get_linestyle()),
                    handle.get_hatch() or "",
                )
            else:
                continue  # collection handles carry marker identity of their own
            label = text.get_text()
            if key in seen:
                clashes.append((f"legend[{index}]", seen[key], label))
            else:
                seen[key] = label
    return clashes


@pytest.mark.timeout(180)
def test_legend_colors_are_distinguishable(built_figure):
    """No legend may show two entries sharing both color and linestyle."""
    name, fig = built_figure
    clashes = find_indistinguishable_legend_entries(fig)
    assert not clashes, (
        f"Figure {name!r} has legend entries a reader cannot tell apart — "
        f"same color and linestyle: {clashes}. Give each series a distinct "
        f"hue or a distinct linestyle."
    )


def test_legend_check_catches_identical_entries():
    """POSITIVE CONTROL — the check must be able to fail."""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1], color="#0072B2", linestyle="-", label="series A")
    ax.plot([0, 1], [1, 0], color="#0072B2", linestyle="-", label="series B")
    ax.legend()
    try:
        assert find_indistinguishable_legend_entries(fig), (
            "two same-color same-style legend entries were not flagged"
        )
    finally:
        plt.close(fig)


def test_legend_check_accepts_distinct_entries():
    """NEGATIVE CONTROL — distinct color or linestyle must not be flagged."""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1], color="#0072B2", linestyle="-", label="blue solid")
    ax.plot([0, 1], [1, 0], color="#0072B2", linestyle="--", label="blue dashed")
    ax.plot([0, 1], [0.5, 0.5], color="#E69F00", linestyle="-", label="orange solid")
    ax.legend()
    try:
        assert find_indistinguishable_legend_entries(fig) == []
    finally:
        plt.close(fig)


# ── TEXT VS DRAWN REFERENCE RULES ────────────────────────────────────────────
# A dotted rule running through its own annotation reads as a strikethrough.
# ``assert_within_canvas`` checks the canvas edge and the sweep above checks
# text against text; neither sees an axhline drawn through a label. Five
# figures shipped that defect and the 2026-09-05 visual audit caught it only
# by looking. The geometric check lives in ``ailm.figures._common`` beside the
# border check, and ``save_figure`` enforces it on every write.


@pytest.mark.timeout(180)
def test_no_label_is_struck_through_by_a_reference_rule(built_figure):
    """No drawn reference rule may run through a rendered label."""
    _name, fig = built_figure
    assert_no_text_crosses_rules(fig)


def test_rule_check_catches_a_rule_through_a_label():
    """POSITIVE CONTROL — the check must be able to fail."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axhline(5.0)
    ax.text(5.0, 5.0, "STRUCK THROUGH", ha="center", va="center")
    try:
        crossings = find_text_rule_crossings(fig)
        assert crossings, "an axhline drawn through a label was not flagged"
    finally:
        plt.close(fig)


def test_rule_check_catches_a_vertical_rule_through_a_label():
    """POSITIVE CONTROL — the vertical sibling of the same defect."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axvline(5.0)
    ax.text(5.0, 5.0, "STRUCK THROUGH", ha="center", va="center")
    try:
        assert find_text_rule_crossings(fig), "an axvline through a label was not flagged"
    finally:
        plt.close(fig)


@pytest.mark.parametrize("orientation", ["horizontal", "vertical", "frame"])
@pytest.mark.parametrize("inverted", [False, True])
def test_rule_check_catches_an_axes_declared_segment(orientation, inverted):
    """Explicit separator rules are checked even when they span only part of an axis."""
    fig, ax = plt.subplots(figsize=(8, 6))
    limits = (10, 0) if inverted else (0, 10)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    if orientation == "horizontal":
        ax.plot([0.4, 0.6], [5, 5], transform=ax.get_yaxis_transform())
    elif orientation == "vertical":
        ax.plot([5, 5], [0.4, 0.6], transform=ax.get_xaxis_transform())
    else:
        ax.plot([0.4, 0.6], [0.5, 0.5], transform=ax.transAxes)
    ax.text(5.0, 5.0, "STRUCK THROUGH", ha="center", va="center")
    try:
        assert find_text_rule_crossings(fig), "an axes-declared separator was not flagged"
    finally:
        plt.close(fig)


def test_rule_check_ignores_a_full_span_flat_data_line_through_a_label():
    """A data curve stays outside this gate even when flat and nearly full-width."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.plot([0.5, 9.5], [5.0, 5.0])
    ax.text(5.0, 5.0, "DATA", ha="center", va="center")
    try:
        assert find_text_rule_crossings(fig) == []
    finally:
        plt.close(fig)


def test_rule_check_ignores_flat_data_on_unit_limits():
    """Data coordinates must not become rules when their limits happen to be 0..1."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0.5, 0.5])
    ax.text(0.5, 0.5, "DATA", ha="center", va="center")
    try:
        assert find_text_rule_crossings(fig) == []
    finally:
        plt.close(fig)


def test_rule_check_keeps_custom_compositions_outside_its_declared_scope():
    """Matching pixels cannot certify the semantics of an arbitrary transform chain."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    (line,) = ax.plot([0.5, 0.5], [0.4, 0.6], transform=ax.transAxes)
    ax.text(5, 5, "SEPARATOR", ha="center", va="center")
    try:
        assert find_text_rule_crossings(fig), "the standard frame transform must be checked"
        line.set_transform(ax.transAxes + ScaledTranslation(0, 0, fig.dpi_scale_trans))
        assert find_text_rule_crossings(fig) == []
    finally:
        plt.close(fig)


def test_rule_check_accepts_a_label_clear_of_the_rule():
    """NEGATIVE CONTROL — a label off the rule must not be reported."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axhline(5.0)
    ax.text(5.0, 7.5, "clear above", ha="center", va="bottom")
    try:
        assert find_text_rule_crossings(fig) == []
    finally:
        plt.close(fig)


def test_rule_check_respects_an_opaque_label_background():
    """NEGATIVE CONTROL — an opaque box hides the rule, so nothing crosses glyphs."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axhline(5.0)
    ax.text(
        5.0,
        5.0,
        "BOXED LABEL",
        ha="center",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none"},
    )
    try:
        assert find_text_rule_crossings(fig) == []
    finally:
        plt.close(fig)


def test_rule_check_still_flags_a_box_when_the_rule_draws_above_it():
    """POSITIVE CONTROL — the opaque exemption must not outrank draw order."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axhline(5.0, zorder=5)
    ax.text(
        5.0,
        5.0,
        "BOXED LABEL",
        ha="center",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none"},
    )
    try:
        assert find_text_rule_crossings(fig), (
            "a rule drawn above an opaque label box was wrongly exempted"
        )
    finally:
        plt.close(fig)


def test_rule_check_flags_a_semi_transparent_label_background():
    """POSITIVE CONTROL — a translucent box is not a shield; the rule shows."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axhline(5.0)
    ax.text(
        5.0,
        5.0,
        "TINTED LABEL",
        ha="center",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.5},
    )
    try:
        assert find_text_rule_crossings(fig), "a translucent box hid the rule from the check"
    finally:
        plt.close(fig)


def test_rule_check_flags_frameless_legend_text_over_a_rule():
    """POSITIVE CONTROL — legend text is text; a bare legend hides nothing."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.plot([1, 9], [8.0, 8.0], label="series")
    legend = ax.legend(loc="center", frameon=False)
    fig.canvas.draw()
    box = text_extent(legend.get_texts()[0], renderer=fig.canvas.get_renderer())
    y = ax.transData.inverted().transform((0.0, 0.5 * (box.y0 + box.y1)))[1]
    ax.axhline(y)
    try:
        assert find_text_rule_crossings(fig), "a rule through bare legend text was not flagged"
    finally:
        plt.close(fig)


def test_rule_check_respects_an_opaque_legend_frame():
    """NEGATIVE CONTROL — legend text on an opaque card sits above the rule."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.plot([1, 9], [8.0, 8.0], label="series")
    legend = ax.legend(loc="center", frameon=True, framealpha=1.0)
    fig.canvas.draw()
    box = text_extent(legend.get_texts()[0], renderer=fig.canvas.get_renderer())
    y = ax.transData.inverted().transform((0.0, 0.5 * (box.y0 + box.y1)))[1]
    ax.axhline(y)
    try:
        assert find_text_rule_crossings(fig) == [], (
            "the rule is hidden behind the legend's opaque frame"
        )
    finally:
        plt.close(fig)


def test_rule_check_ignores_gridlines_through_a_label():
    """NEGATIVE CONTROL — grid structure behind the data is not a rule."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_yticks([5.0])
    ax.grid(True)
    ax.text(5.0, 5.0, "ON A GRIDLINE", ha="center", va="center")
    try:
        assert find_text_rule_crossings(fig) == []
    finally:
        plt.close(fig)


def test_rule_check_ignores_a_marker_series_through_a_label():
    """NEGATIVE CONTROL — scoping: a data series, even flat, is not a rule."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.plot([0, 10], [5.0, 5.0], marker="o")
    ax.text(5.0, 5.0, "DATA", ha="center", va="center")
    try:
        assert find_text_rule_crossings(fig) == []
    finally:
        plt.close(fig)


def test_rule_check_ignores_a_short_segment_through_a_label():
    """NEGATIVE CONTROL — scoping: a short stroke is an element, not a rule."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.plot([4.5, 5.5], [5.0, 5.0])
    ax.text(5.0, 5.0, "SEGMENT", ha="center", va="center")
    try:
        assert find_text_rule_crossings(fig) == []
    finally:
        plt.close(fig)


def test_rule_check_ignores_a_slanted_line_through_a_label():
    """NEGATIVE CONTROL — scoping: an axis-aligned span is what makes a rule."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.plot([0, 10], [4.0, 6.0])
    ax.text(5.0, 5.0, "SLANTED", ha="center", va="center")
    try:
        assert find_text_rule_crossings(fig) == []
    finally:
        plt.close(fig)


def test_rule_check_ignores_a_rule_outside_the_drawn_view():
    """NEGATIVE CONTROL — a level the axes never draws cannot strike anything."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axhline(12.0)
    # Unclipped, so the text IS rendered — above the axes, where no rule is.
    ax.text(5.0, 12.0, "NOT DRAWN", ha="center", va="center")
    try:
        assert find_text_rule_crossings(fig) == []
    finally:
        plt.close(fig)


def test_rule_check_respects_text_clipped_away_from_the_rule():
    """NEGATIVE CONTROL — clipping: text counts by its visible extent only."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axhline(5.0)
    text = ax.text(5.0, 5.0, "CLIPPED AWAY", ha="center", va="center", clip_on=True)
    fig.canvas.draw()
    box = ax.bbox
    text.set_clip_box(
        Bbox.from_extents(box.x0, box.y1 - 0.25 * box.height, box.x0 + 0.25 * box.width, box.y1)
    )
    try:
        assert find_text_rule_crossings(fig) == []
    finally:
        plt.close(fig)


def test_assert_no_text_crosses_rules_raises_on_a_bad_figure():
    """POSITIVE CONTROL — the save-time gate refuses a rule-struck figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axhline(5.0)
    ax.text(5.0, 5.0, "STRUCK THROUGH", ha="center", va="center")
    try:
        with pytest.raises(TextRuleCollisionError, match="struck through"):
            assert_no_text_crosses_rules(fig)
    finally:
        plt.close(fig)


# ── FIGURE / CODE AGREEMENT ──────────────────────────────────────────────────
# A figure that contradicts the code is a defect even when it renders perfectly.
# Layout checks above cannot see this class of error at all.


def test_figure_alpha_matches_the_gate_alpha():
    """The α drawn in fig:subsample_bias must be the α the gate decides at.

    Two constants for one decision drift, and here the drift is not cosmetic: the
    panel exists to show that a representative subsample is *underpowered*, and the
    subsample's p-value sits between the conventional 0.05 and the Stage 2 gate's
    stricter α. Drawing the looser line would label that subsample significant —
    the exact conclusion the figure is there to deny.

    Since the 2026-08-04 hardening both sides derive from one constant,
    ``ailm.statistics.FAMILYWISE_ALPHA``. This check is structural: it asserts the
    gate test still *imports* that constant as its ALPHA (rather than shadowing it
    with a literal that could drift) and that the figure module's ALPHA_LEVEL still
    equals it. Either side dropping the shared source fails here.
    """
    import ast

    from ailm.figures.gates import ALPHA_LEVEL
    from ailm.statistics import FAMILYWISE_ALPHA

    gate_test = __import__("pathlib").Path(__file__).resolve().parent / "test_stage2_gate.py"
    if not gate_test.exists():
        pytest.skip("tests/test_stage2_gate.py absent")

    tree = ast.parse(gate_test.read_text(), filename=str(gate_test))
    imports_shared_alpha = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "ailm.statistics"
        and any(
            alias.name == "FAMILYWISE_ALPHA" and alias.asname == "ALPHA" for alias in node.names
        )
        for node in ast.walk(tree)
    )
    shadows_alpha = any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "ALPHA" for t in node.targets)
        for node in ast.walk(tree)
    )
    assert imports_shared_alpha and not shadows_alpha, (
        "tests/test_stage2_gate.py no longer binds ALPHA to "
        "ailm.statistics.FAMILYWISE_ALPHA (or shadows it with a literal), so the "
        "gate could decide at a different α than fig:subsample_bias draws."
    )
    assert ALPHA_LEVEL == FAMILYWISE_ALPHA, (
        f"fig:subsample_bias draws α = {ALPHA_LEVEL} but the shared gate α is "
        f"{FAMILYWISE_ALPHA}. The figure would report a different verdict than the gate."
    )


def test_alpha_is_strict_enough_to_reject_the_representative_stride():
    """Pins the substantive point, not merely the numeric agreement.

    The figure's argument requires that the representative stride subsample FAIL at
    the chosen α. If α ever loosened past its p-value the panel would still render,
    still agree with the gate, and still say the opposite of what it claims.
    """
    from ailm.figures.gates import ALPHA_LEVEL

    stride_p = 0.017  # stride[::3], measured; see fig:subsample_bias panel B
    assert stride_p > ALPHA_LEVEL, (
        f"α = {ALPHA_LEVEL} would make the representative stride (p = {stride_p}) "
        f"significant, contradicting the panel's own conclusion that a representative "
        f"subsample of this size is underpowered."
    )


#: Minimum resolution a figure must resolve at once scaled onto the page.
#: 300 dpi is the print standard; above it, zooming reveals detail rather than
#: pixels. This is the export-quality criterion these figures are held to.
#:
#: Note it moves OPPOSITE to on-page point size. A wide canvas shrunk to the
#: column prints small type but resolves at high dpi; shrinking the canvas
#: enlarges the type and coarsens the raster. Two figures were once shrunk
#: toward print width and fell to 219 and 238 dpi — below print standard — so
#: the change was reverted. Optimize resolution, not printed point size.
MIN_ON_PAGE_DPI = 300.0


def test_every_figure_resolves_at_print_resolution_on_the_page():
    """Export quality: no figure may land below print resolution once scaled.

    These figures are dense and are meant to be read at zoom, so small printed
    type is accepted. What is not accepted is a raster that goes soft when a
    reader zooms in — that is the failure this gate exists to prevent.
    """
    import pathlib

    from ailm.figures._common import effective_on_page_dpi

    root = pathlib.Path(__file__).resolve().parent.parent
    figures = sorted(p.stem for p in (root / "output" / "figures").glob("*.png"))
    if not figures:
        pytest.skip("output/figures/ empty; run scripts/generate_figures.py")

    measured = {name: effective_on_page_dpi(name, project=root) for name in figures}
    assert measured, "no figure measured; this gate would be vacuous"
    soft = {n: round(d) for n, d in measured.items() if d < MIN_ON_PAGE_DPI}
    assert soft == {}, (
        f"these figures resolve below {MIN_ON_PAGE_DPI:.0f} dpi on the page: {soft}. "
        f"Either widen the canvas (which raises effective dpi) or raise viz_config.DPI. "
        f"Do not shrink the canvas to enlarge the type — that lowers resolution."
    )


def test_the_resolution_gate_is_not_vacuous():
    """A gate every figure clears by a mile constrains nothing worth checking."""
    import pathlib

    from ailm.figures._common import effective_on_page_dpi

    root = pathlib.Path(__file__).resolve().parent.parent
    figures = sorted(p.stem for p in (root / "output" / "figures").glob("*.png"))
    if not figures:
        pytest.skip("output/figures/ empty")
    worst = min(effective_on_page_dpi(n, project=root) for n in figures)
    assert worst < MIN_ON_PAGE_DPI * 2, (
        f"the softest figure resolves at {worst:.0f} dpi, far above the "
        f"{MIN_ON_PAGE_DPI:.0f} dpi gate. Raise the gate to lock the margin in."
    )


def test_the_rendered_pdf_embeds_every_font_and_uses_no_bitmap_type3():
    """Font materials in the export must be complete and scalable.

    Small printed type is fine only if the glyphs behind it are real: fully
    embedded so no reader's machine substitutes a different face, subsetted with
    a Unicode map so the text stays selectable and searchable, and outline rather
    than Type 3 bitmaps, which go jagged exactly when a reader zooms in to read
    something small.
    """
    import pathlib
    import re

    pdf = (
        pathlib.Path(__file__).resolve().parent.parent
        / "output"
        / "pdf"
        / "active_inference_language_model_combined.pdf"
    )
    if not pdf.exists():
        pytest.skip("combined PDF absent; run the render pipeline")

    # Font dictionaries live inside compressed object streams in PDF 1.5+, so the
    # raw bytes must be inflated before anything can be found in them. Scanning
    # the file uncompressed silently finds nothing and reads as a pass.
    import zlib

    raw = pdf.read_bytes()
    inflated = [raw]
    for match in re.finditer(rb"stream\r?\n", raw):
        start = match.end()
        end = raw.find(b"endstream", start)
        if end < 0:
            continue
        try:
            inflated.append(zlib.decompress(raw[start:end]))
        except zlib.error:
            continue
    blob = b"".join(inflated)

    base_fonts = set(re.findall(rb"/BaseFont\s*/([A-Za-z0-9+#\-,_.]+)", blob))
    assert len(base_fonts) > 5, (
        f"only {len(base_fonts)} fonts found in the PDF; the scan is broken rather "
        f"than the document being clean, and an empty scan passes every rule below"
    )

    type3 = blob.count(b"/Subtype /Type3") + blob.count(b"/Subtype/Type3")
    assert type3 == 0, (
        f"{type3} Type 3 (bitmap) font(s) in the export. Type 3 glyphs are "
        f"rasterised outlines that degrade under the zoom these figures rely on."
    )

    not_subset = sorted(f.decode(errors="replace") for f in base_fonts if b"+" not in f)
    assert not_subset == [], (
        f"these fonts are not subset-embedded: {not_subset}. A non-embedded font "
        f"is substituted on machines that lack it, silently changing the glyphs."
    )
