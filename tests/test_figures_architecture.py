"""Tests for the figure infrastructure and the architecture / matrix figures.

No mocks. Every test builds a real matplotlib figure, writes a real PNG to a real
temporary directory, and reads the bytes back.

The load-bearing test in this file is
:meth:`TestAssertLegible.test_tiny_font_figure_is_rejected`. ``assert_legible``
is called by ``save_figure`` on every figure the project produces, so if it could
not fail, every other legibility assertion here would be vacuously true — the
green suite would prove only that the function returns. The negative control
constructs a figure whose label is deliberately below the floor and requires the
raise. The same reasoning covers the empty-figure case: a legibility check that
passes on a canvas with no text is not checking anything.

Determinism is tested by writing the same figure twice and comparing bytes, which
is the only formulation that actually catches an embedded timestamp or a
version-stamped metadata field. Comparing file sizes would not.
"""

from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path

from ailm.cases import CaseRole
from ailm.figures import architecture
from ailm.figures._common import (
    FigureOverflowError,
    IllegibleFigureError,
    MissingVariableError,
    assert_legible,
    assert_within_canvas,
    default_output_dir,
    load_manuscript_variables,
    manuscript_variables_path,
    project_root,
    rendered_texts,
    save_figure,
    text_extent,
    token_float,
    token_int,
)
from ailm.figures.architecture import (
    LEVELS,
    generate_hierarchy_figure,
    generate_inference_cycle_figure,
    write_architecture_figures,
)
from ailm.figures.matrices import (
    MARKED_BIGRAMS,
    generate_letter_transition_figure,
    generate_wiring_comparison_figure,
    letter_transition_matrix,
    write_matrix_figures,
)
from ailm.gates import GateId
from ailm.levels.letter import ALPHA_TO_IDX, N_SYMBOLS
from ailm.levels.widened import N_WIDENED_SYMBOLS
from ailm.viz_config import (
    DPI,
    FIGSIZES,
    FONTS,
    MIN_FONT_PT,
    PALETTE,
    PALETTE_ORDER,
    apply_style,
    qualitative_colors,
)

# pymdp/jax emit a benign UserWarning that would otherwise dominate the output.
warnings.filterwarnings("ignore", message=".*JAX array is being set as static.*")

#: Below this a PNG is a blank canvas or an error page, not a figure. The real
#: figures land in the hundreds of kilobytes; 20 KB is a floor, not a target.
_MIN_PNG_BYTES = 20_000

#: How far a rendered label may cross a table or column border, in display
#: points, before it counts as overflow. Anti-aliasing and glyph side bearings
#: make an exact bound too strict; the same allowance the layout sweep uses.
_CONTAINMENT_TOLERANCE_PT = 2.0


def _arrow_y_extent(arrow: FancyArrowPatch) -> tuple[float, float]:
    """Lowest and highest y, in axes fraction, of an arrow's drawn geometry.

    ``get_path`` returns the body and head as one compound path in data
    coordinates; its CLOSEPOLY placeholders sit at the origin and must be
    dropped or every arrow reads as reaching y = 0.
    """
    path = arrow.get_path()
    assert path.codes is not None
    ys = path.vertices[path.codes != Path.CLOSEPOLY, 1]
    return float(ys.min()), float(ys.max())


def _hierarchy_row_index(y: float) -> int | None:
    """Map an axes-fraction y to the LEVELS row containing it, or None."""
    for index in range(len(architecture.LEVELS)):
        y0, y1 = architecture._row_bounds(index)
        if y0 <= y <= y1:
            return index
    return None


def _hierarchy_containment_violations(
    fig: Figure,
    *,
    table_columns: tuple[tuple[str, float, float], ...] = architecture._COLUMNS,
    stage0_y0: float = architecture._STAGE0_Y0,
) -> list[str]:
    """Rendered text that leaves its column, the table, or intrudes on the band.

    Measured in display space on the drawn figure, so it reports what a reader
    sees. Three checks: every label whose baseline sits inside a level row must
    end inside the table's right border; every such label that starts inside a
    column must end inside that column; and the footnote block must sit below
    the Stage 0 band rather than printing through it.

    ``table_columns`` and ``stage0_y0`` default to the real module geometry
    and must match whatever geometry the figure was actually rendered with —
    the positive controls pass the same perturbed values here that they pass
    to :func:`generate_hierarchy_figure`.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    tolerance = _CONTAINMENT_TOLERANCE_PT * fig.dpi / 72.0
    table_x1 = architecture._TABLE_X1 * width
    columns = [
        (header, architecture._col_x(f0) * width, architecture._col_x(f1) * width)
        for header, f0, f1 in table_columns
    ]
    band_y0 = stage0_y0 * height

    violations: list[str] = []
    for artist in rendered_texts(fig):
        box = text_extent(artist, renderer=renderer)
        label = artist.get_text().replace("\n", " / ")[:40]
        row = _hierarchy_row_index(((box.y0 + box.y1) / 2) / height)
        if row is None:
            if artist.get_text().startswith("Audited") and box.y1 > band_y0 + tolerance:
                violations.append(
                    f"footnote intrudes on the Stage 0 band by {box.y1 - band_y0:.0f}px"
                )
            continue
        if box.x0 < architecture._TABLE_X0 * width or box.x0 >= table_x1:
            continue  # gutter notes and the level-span label live outside the table
        if box.x1 > table_x1 + tolerance:
            violations.append(f"{label!r} ends {box.x1 - table_x1:.0f}px past the table border")
        for header, x0, x1 in columns:
            if x0 <= box.x0 < x1 and box.x1 > x1 + tolerance:
                violations.append(
                    f"{label!r} ends {box.x1 - x1:.0f}px past its {header.split(chr(10))[0]!r} column"
                )
    return violations


@pytest.fixture(scope="module")
def brown_matrix() -> np.ndarray:
    """The Brown letter bigram matrix, loaded once for the whole module."""
    return letter_transition_matrix()


@pytest.fixture(scope="module")
def variables() -> dict[str, str]:
    """The generated manuscript token map."""
    return load_manuscript_variables()


def _tiny_font_figure(size_pt: float = 5.0) -> Figure:
    """Build a figure whose only label is below the legibility floor.

    This is the negative control for :func:`assert_legible`. It must exist and it
    must fail the check, otherwise a green suite says nothing about legibility.

    Args:
        size_pt: Font size for the offending label. Must be below
            :data:`ailm.viz_config.MIN_FONT_PT`.

    Returns:
        A figure containing one deliberately illegible label.
    """
    assert size_pt < MIN_FONT_PT, (
        f"The negative control must be below the {MIN_FONT_PT}pt floor to be a "
        f"control at all; got {size_pt}pt."
    )
    fig = plt.figure(figsize=(3.0, 2.0))
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, "deliberately too small to read", fontsize=size_pt)
    ax.set_title("legible title", fontsize=FONTS["title"])
    return fig


# ── viz_config ───────────────────────────────────────────────────────────────


class TestVizConfig:
    """The palette, the type scale, and the style application."""

    def test_palette_has_enough_distinct_colors(self) -> None:
        """The qualitative cycle must not reuse a hue."""
        assert len(PALETTE_ORDER) >= 6
        assert len(set(PALETTE_ORDER)) == len(PALETTE_ORDER), (
            f"Two series would be drawn in the same color: {PALETTE_ORDER}"
        )
        for color in PALETTE_ORDER:
            assert color.startswith("#") and len(color) == 7, color

    def test_semantic_roles_are_all_defined_and_distinct(self) -> None:
        """Each comparator in the Stage 1 argument gets its own color."""
        roles = ("UNIFORM", "UNIGRAM", "MODEL", "FLOOR", "ANTI", "ACCENT")
        values = [PALETTE[role] for role in roles]
        assert len(set(values)) == len(values), dict(zip(roles, values, strict=True))

    def test_correct_and_anti_wiring_are_not_a_red_green_pair(self) -> None:
        """The two closest-together comparators must differ in the blue channel.

        MODEL and ANTI carry the figure's central contrast. If they were a
        red/green pair, a reader with deuteranopia would lose it. Okabe-Ito blue
        versus vermillion differ strongly in blue, which is the channel
        red/green colorblindness leaves intact.
        """
        model_blue = int(PALETTE["MODEL"][5:7], 16)
        anti_blue = int(PALETTE["ANTI"][5:7], 16)
        assert abs(model_blue - anti_blue) > 100, (
            f"MODEL {PALETTE['MODEL']} and ANTI {PALETTE['ANTI']} differ by only "
            f"{abs(model_blue - anti_blue)} in the blue channel."
        )

    def test_every_font_role_is_at_or_above_the_floor(self) -> None:
        """No role in the type scale may sit below MIN_FONT_PT."""
        offenders = {k: v for k, v in FONTS.items() if v < MIN_FONT_PT}
        assert not offenders, offenders

    def test_min_font_pt_and_dpi(self) -> None:
        assert MIN_FONT_PT == 9.0
        assert DPI == 200

    def test_figsizes_are_positive(self) -> None:
        for name, (w, h) in FIGSIZES.items():
            assert w > 0 and h > 0, name

    def test_apply_style_is_idempotent(self) -> None:
        apply_style()
        first = dict(matplotlib.rcParams)
        apply_style()
        assert matplotlib.rcParams["figure.dpi"] == first["figure.dpi"]
        assert matplotlib.rcParams["font.sans-serif"] == first["font.sans-serif"]

    def test_apply_style_pins_a_bundled_font(self) -> None:
        """A system-font substitution would break byte-identical regeneration."""
        apply_style()
        assert matplotlib.rcParams["font.sans-serif"] == ["DejaVu Sans"]

    def test_qualitative_colors_returns_requested_count(self) -> None:
        assert qualitative_colors(0) == []
        assert qualitative_colors(3) == list(PALETTE_ORDER[:3])

    def test_qualitative_colors_refuses_to_cycle(self) -> None:
        """Cycling would silently give two series the same color."""
        with pytest.raises(ValueError, match="colorblind-safe"):
            qualitative_colors(len(PALETTE_ORDER) + 1)

    def test_qualitative_colors_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            qualitative_colors(-1)


# ── assert_legible: positive AND negative control ────────────────────────────


class TestAssertLegible:
    """The legibility gate, including the control that proves it can fail."""

    def test_tiny_font_figure_is_rejected(self) -> None:
        """NEGATIVE CONTROL — required.

        Without this, ``assert_legible`` could be a no-op and every other
        legibility assertion in this file would pass for the wrong reason.
        """
        fig = _tiny_font_figure(5.0)
        try:
            with pytest.raises(IllegibleFigureError, match="legibility floor"):
                assert_legible(fig)
        finally:
            plt.close(fig)

    def test_rejection_names_the_offending_label(self) -> None:
        """The error must be actionable, not just a failure."""
        fig = _tiny_font_figure(4.0)
        try:
            with pytest.raises(IllegibleFigureError) as excinfo:
                assert_legible(fig)
            assert "deliberately too small to read" in str(excinfo.value)
            assert "4pt" in str(excinfo.value)
        finally:
            plt.close(fig)

    def test_a_label_exactly_on_the_floor_passes(self) -> None:
        """The floor is inclusive; the smallest role in the type scale sits on it."""
        fig = plt.figure(figsize=(3.0, 2.0))
        try:
            fig.text(0.1, 0.5, "exactly on the floor", fontsize=MIN_FONT_PT)
            assert_legible(fig)
        finally:
            plt.close(fig)

    def test_a_label_just_under_the_floor_fails(self) -> None:
        """Boundary in the other direction — the check is not off by one."""
        fig = plt.figure(figsize=(3.0, 2.0))
        try:
            fig.text(0.1, 0.5, "just under", fontsize=MIN_FONT_PT - 0.5)
            with pytest.raises(IllegibleFigureError):
                assert_legible(fig)
        finally:
            plt.close(fig)

    def test_textless_figure_is_rejected(self) -> None:
        """A check that passes on an empty canvas is not a check."""
        fig = plt.figure(figsize=(2.0, 2.0))
        try:
            with pytest.raises(IllegibleFigureError, match="no visible text"):
                assert_legible(fig)
        finally:
            plt.close(fig)

    def test_invisible_small_text_does_not_trip_the_check(self) -> None:
        """Only what a reader can see counts."""
        fig = plt.figure(figsize=(3.0, 2.0))
        try:
            fig.text(0.1, 0.5, "readable", fontsize=FONTS["title"])
            hidden = fig.text(0.1, 0.2, "hidden and tiny", fontsize=3.0)
            hidden.set_visible(False)
            assert_legible(fig)
        finally:
            plt.close(fig)

    def test_save_figure_refuses_an_illegible_figure(self, tmp_path) -> None:
        """The gate is a build-time invariant, not a test-time nicety."""
        fig = _tiny_font_figure(5.0)
        try:
            with pytest.raises(IllegibleFigureError):
                save_figure(fig, "should_not_exist", tmp_path)
        finally:
            plt.close(fig)
        assert not (tmp_path / "should_not_exist.png").exists()

    def test_save_figure_can_be_forced_past_the_gate(self, tmp_path) -> None:
        """The escape hatch exists only so the gate itself can be exercised."""
        fig = _tiny_font_figure(5.0)
        try:
            path = save_figure(fig, "forced", tmp_path, check_legible=False)
        finally:
            plt.close(fig)
        assert path.exists()

    def test_save_figure_refuses_a_label_outside_the_canvas(self, tmp_path) -> None:
        """NEGATIVE CONTROL: the publication path must invoke containment."""
        fig = plt.figure(figsize=(2.0, 2.0))
        try:
            fig.text(0.99, 0.5, "deliberately clipped", fontsize=FONTS["title"], ha="left")
            with pytest.raises(FigureOverflowError, match="outside the figure canvas"):
                save_figure(fig, "should_not_exist", tmp_path)
        finally:
            plt.close(fig)
        assert not (tmp_path / "should_not_exist.png").exists()


# ── save_figure ──────────────────────────────────────────────────────────────


class TestSaveFigure:
    """Writing, naming, and determinism."""

    def test_writes_a_nontrivial_png(self, tmp_path) -> None:
        fig = generate_hierarchy_figure()
        try:
            path = save_figure(fig, "hierarchy", tmp_path)
        finally:
            plt.close(fig)
        assert path.exists()
        assert path.name == "hierarchy.png"
        assert path.stat().st_size > _MIN_PNG_BYTES
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_accepts_a_png_suffix_without_doubling_it(self, tmp_path) -> None:
        fig = plt.figure(figsize=(2.0, 2.0))
        try:
            fig.text(0.1, 0.5, "label", fontsize=FONTS["title"])
            path = save_figure(fig, "named.png", tmp_path)
        finally:
            plt.close(fig)
        assert path.name == "named.png"

    def test_creates_the_output_directory(self, tmp_path) -> None:
        target = tmp_path / "deep" / "nested"
        fig = plt.figure(figsize=(2.0, 2.0))
        try:
            fig.text(0.1, 0.5, "label", fontsize=FONTS["title"])
            path = save_figure(fig, "made", target)
        finally:
            plt.close(fig)
        assert path.parent == target and path.exists()

    @pytest.mark.parametrize("bad", ["", "   ", "sub/dir", "sub\\dir"])
    def test_rejects_a_bad_name(self, tmp_path, bad: str) -> None:
        fig = plt.figure(figsize=(2.0, 2.0))
        try:
            fig.text(0.1, 0.5, "label", fontsize=FONTS["title"])
            with pytest.raises(ValueError):
                save_figure(fig, bad, tmp_path)
        finally:
            plt.close(fig)

    def test_png_software_metadata_is_pinned(self, tmp_path) -> None:
        """matplotlib's default Software field embeds its own version number."""
        from PIL import Image

        fig = plt.figure(figsize=(2.0, 2.0))
        try:
            fig.text(0.1, 0.5, "label", fontsize=FONTS["title"])
            path = save_figure(fig, "meta", tmp_path)
        finally:
            plt.close(fig)
        with Image.open(path) as image:
            assert image.info.get("Software") == "ailm"
            assert "Creation Time" not in image.info
            assert "Date" not in image.info

    def test_render_failure_preserves_the_previous_png(self, tmp_path) -> None:
        """NEGATIVE CONTROL: a failed draw must never truncate publication output."""

        class ExplodingArtist(Artist):
            def draw(self, renderer) -> None:
                raise RuntimeError("planted draw failure")

        target = tmp_path / "published.png"
        previous = b"previous publication bytes"
        target.write_bytes(previous)
        fig = plt.figure(figsize=(2.0, 2.0))
        try:
            fig.text(0.1, 0.5, "label", fontsize=FONTS["title"])
            fig.add_artist(ExplodingArtist())
            with pytest.raises(RuntimeError, match="planted draw failure"):
                save_figure(fig, "published", tmp_path)
        finally:
            plt.close(fig)

        assert target.read_bytes() == previous
        assert list(tmp_path.glob(".published.png.*.tmp.png")) == []


# ── Determinism ──────────────────────────────────────────────────────────────


class TestDeterminism:
    """Regenerating a figure must reproduce it byte for byte."""

    @pytest.mark.parametrize(
        "builder",
        [
            generate_hierarchy_figure,
            generate_inference_cycle_figure,
            generate_letter_transition_figure,
            generate_wiring_comparison_figure,
        ],
        ids=["hierarchy", "inference_cycle", "letter_transition", "wiring_comparison"],
    )
    def test_two_runs_produce_identical_bytes(self, tmp_path, builder) -> None:
        written = []
        for run in ("first", "second"):
            fig = builder()
            try:
                written.append(save_figure(fig, run, tmp_path).read_bytes())
            finally:
                plt.close(fig)
        assert written[0] == written[1], (
            f"{builder.__name__} is not deterministic: "
            f"{len(written[0])} vs {len(written[1])} bytes."
        )

    def test_the_determinism_check_can_distinguish_two_figures(self, tmp_path) -> None:
        """Positive control for the byte comparison above.

        If ``save_figure`` were writing a constant — an empty file, say — the
        determinism tests would pass while proving nothing. Two genuinely
        different figures must produce different bytes.
        """
        first = generate_hierarchy_figure()
        try:
            a = save_figure(first, "a", tmp_path).read_bytes()
        finally:
            plt.close(first)
        second = generate_inference_cycle_figure()
        try:
            b = save_figure(second, "b", tmp_path).read_bytes()
        finally:
            plt.close(second)
        assert a != b


# ── Manuscript variables ─────────────────────────────────────────────────────


class TestManuscriptVariables:
    """Reading measured values instead of hardcoding them."""

    def test_the_token_map_exists_and_is_populated(self, variables) -> None:
        assert manuscript_variables_path().exists()
        assert len(variables) > 50

    def test_token_float_parses_thousands_separators(self, variables) -> None:
        """``42,820`` must not raise, and must not become 42.82."""
        positions = token_float(variables, "STAGE2_G2P_EVAL_POSITIONS")
        assert positions > 1000

    def test_token_float_parses_scientific_notation(self, variables) -> None:
        p_value = token_float(variables, "STAGE2_G2P_MCNEMAR_P")
        assert 0.0 < p_value < 1e-3

    def test_token_int_rejects_a_fractional_value(self, variables) -> None:
        with pytest.raises(MissingVariableError, match="not an integer"):
            token_int(variables, "STAGE1_LETTER_PERPLEXITY")

    def test_token_int_accepts_a_whole_value(self, variables) -> None:
        assert token_int(variables, "PHONEME_COUNT_39") == 39

    def test_missing_token_raises_a_directive_error(self, variables) -> None:
        with pytest.raises(MissingVariableError, match="do not put the number"):
            token_float(variables, "STAGE9_NOT_A_REAL_CLAIM")

    def test_missing_file_raises_rather_than_defaulting(self, tmp_path) -> None:
        with pytest.raises(MissingVariableError, match="not found"):
            load_manuscript_variables(tmp_path / "absent.json")

    def test_non_object_payload_is_rejected(self, tmp_path) -> None:
        bad = tmp_path / "list.json"
        bad.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(MissingVariableError, match="JSON object"):
            load_manuscript_variables(bad)

    def test_project_root_and_output_dir_resolve(self) -> None:
        root = project_root()
        assert (root / "vendor").is_dir()
        assert default_output_dir() == root / "output" / "figures"


# ── Architecture figures ─────────────────────────────────────────────────────


class TestArchitectureFigures:
    """The level stack and the inference cycle."""

    def test_hierarchy_returns_a_legible_figure(self) -> None:
        fig = generate_hierarchy_figure()
        try:
            assert isinstance(fig, Figure)
            assert_legible(fig)
            assert_within_canvas(fig)
        finally:
            plt.close(fig)

    def test_inference_cycle_returns_a_legible_figure(self) -> None:
        fig = generate_inference_cycle_figure()
        try:
            assert isinstance(fig, Figure)
            assert_legible(fig)
        finally:
            plt.close(fig)

    def test_hierarchy_text_stays_inside_its_column_and_the_table(self) -> None:
        """Stage-column verdicts once ran past the table border; nothing saw it.

        ``assert_within_canvas`` checks the canvas edge and the layout sweep
        checks text-on-text, so a label overflowing a *border* passed both.
        """
        fig = generate_hierarchy_figure()
        try:
            violations = _hierarchy_containment_violations(fig)
            assert not violations, "\n".join(violations)
        finally:
            plt.close(fig)

    def test_hierarchy_containment_check_can_fail(self) -> None:
        """POSITIVE CONTROL — the Stage column at its pre-fix width must be caught."""
        columns = list(architecture._COLUMNS)
        header_b, b0, _b1 = columns[4]
        columns[4] = (header_b, b0, 0.925)
        columns[5] = ("Stage", 0.925, 1.000)
        perturbed = tuple(columns)
        fig = generate_hierarchy_figure(columns=perturbed)
        try:
            violations = _hierarchy_containment_violations(fig, table_columns=perturbed)
            assert any("past the table border" in v for v in violations), violations
        finally:
            plt.close(fig)

    def test_hierarchy_footnote_check_can_fail(self) -> None:
        """POSITIVE CONTROL — a band lowered onto the footnote must be caught."""
        fig = generate_hierarchy_figure(stage0_y0=0.030)
        try:
            violations = _hierarchy_containment_violations(fig, stage0_y0=0.030)
            assert any("footnote intrudes" in v for v in violations), violations
        finally:
            plt.close(fig)

    def test_hierarchy_arrows_span_at_most_two_adjacent_rows_one_way(self) -> None:
        """An arrow is a claim about a relation between exactly the rows it touches.

        The alphabet-identity gutter is labeled for L1 and L2 and drawn one way,
        because L2's observation modality is L1's alphabet; ``docs/hierarchy.md``
        rules out any level-spanning cascade. A double-headed arrow from L1 to L6
        shipped for weeks under an L1/L2 label, which is what this pins.
        """
        fig = generate_hierarchy_figure()
        try:
            fig.canvas.draw()
            ax = fig.axes[0]
            arrows = [patch for patch in ax.patches if isinstance(patch, FancyArrowPatch)]
            assert arrows, "the hierarchy figure draws no arrows; the check would be vacuous"
            for arrow in arrows:
                style_name = type(arrow.get_arrowstyle()).__name__
                assert not style_name.endswith("AB"), f"bidirectional arrow style {style_name}"
                y_min, y_max = _arrow_y_extent(arrow)
                low = _hierarchy_row_index(y_min)
                high = _hierarchy_row_index(y_max)
                assert low is not None and high is not None, (
                    f"arrow end outside every level row: {y_min}, {y_max}"
                )
                assert high - low <= 1, f"arrow spans rows {low} to {high}"
        finally:
            plt.close(fig)

    def test_hierarchy_arrow_check_can_fail(self) -> None:
        """POSITIVE CONTROL — the retired L1-to-L6 double-headed arrow is caught."""
        fig = generate_hierarchy_figure()
        try:
            ax = fig.axes[0]
            top = next(i for i, r in enumerate(LEVELS) if r.name == "Doc-position")
            ax.add_patch(
                FancyArrowPatch(
                    (0.040, sum(architecture._row_bounds(0)) / 2),
                    (0.040, sum(architecture._row_bounds(top)) / 2),
                    arrowstyle="<|-|>",
                    mutation_scale=12,
                )
            )
            fig.canvas.draw()
            arrows = [patch for patch in ax.patches if isinstance(patch, FancyArrowPatch)]
            spans = []
            for arrow in arrows:
                y_min, y_max = _arrow_y_extent(arrow)
                lo = _hierarchy_row_index(y_min)
                hi = _hierarchy_row_index(y_max)
                spans.append((lo, hi, type(arrow.get_arrowstyle()).__name__))
            assert any(hi - lo > 1 for lo, hi, _s in spans if lo is not None and hi is not None)
            assert any(style.endswith("AB") for _lo, _hi, style in spans)
        finally:
            plt.close(fig)

    def test_levels_cover_every_stage_from_one_to_seven(self) -> None:
        stages = [row.stage for row in LEVELS]
        assert stages == [f"Stage {n}" for n in range(1, 8)]

    def test_built_rows_match_the_audited_stages(self) -> None:
        """The figure's built flags must track the audit, not a memory of it.

        Originally this asserted only Stages 1 and 2 — true when written,
        stale the day Stage 3 shipped (the finding changed, so the test
        changes; HANDOFF corollary). Stages 1–6 are built models; Stage 7 is
        deliberately NOT marked built, because it is an interface whose
        contract is tested rather than a model with a gate.
        """
        built = [row.stage for row in LEVELS if row.built]
        assert built == [f"Stage {n}" for n in range(1, 7)]
        unbuilt = [row for row in LEVELS if not row.built]
        assert [row.name for row in unbuilt] == ["Speech"]
        assert unbuilt[0].a_kind == "external"
        # Stage 5's qualified pass and Stage 6's non-confirmatory primary result
        # must be visible as distinct outcomes. Until decision 114 this asserted
        # that Stage 5 carried "GAIN UNRES." — true of the case-only model, which
        # that decision demoted to a retained ablation. The canonical Level 5 is
        # the widened-observation arm and it passes, so the assertion follows the
        # finding rather than the memory of it.
        by_stage = {row.stage: row for row in LEVELS}
        assert "PASS" in by_stage["Stage 5"].status
        # Decision 114 requires both limits stated wherever the verdict is
        # stated. A bare PASS would read as the same kind of pass Levels 1
        # through 4 carry, so the qualifier may not be dropped silently.
        assert "DEV-SESSION" in by_stage["Stage 5"].status
        # The ablation's non-passing outcome lives in the gate audit under
        # `stage5_clause_case_only`. Drawing it here would put the demoted arm's
        # verdict in the column the caption says carries the audited outcome.
        assert not any("GAIN UNRES" in row.status for row in LEVELS)
        assert not any("NULL" in row.status for row in LEVELS)
        nonconfirmatory = [row.stage for row in LEVELS if "NONCONF" in row.status]
        assert nonconfirmatory == ["Stage 6"]

    def test_corpus_alphabet_sharing_is_not_drawn_as_live_state_handoff(self) -> None:
        by_name = {row.name: row for row in LEVELS}
        assert "corpus labels" in by_name["Clause"].observation
        assert "corpus labels" in by_name["Doc-position"].observation
        assert "L4's states" not in by_name["Clause"].observation
        assert "L5's latent variable" not in by_name["Doc-position"].observation

    def test_the_clause_row_does_not_claim_level_fours_alphabet(self) -> None:
        """Containment is not identity, and the row may not say it is.

        The canonical Level 5 of decision 114 observes the eight case roles plus
        a distinguished symbol for VERB and one for AUX, so its alphabet strictly
        CONTAINS Level 4's rather than equalling it. The row said "shared L4
        alphabet", which asserts on the level stack the same identity the L1/L2
        seam has and the cover deliberately stopped drawing. Nothing else could
        catch it: the string is inside the canvas, clears the font floor, and
        overlaps nothing.
        """
        clause = next(row for row in LEVELS if row.name == "Clause")
        assert "shared L4 alphabet" not in clause.observation
        assert str(N_WIDENED_SYMBOLS) in clause.observation
        # The count is imported, not typed, so widening the alphabet again moves
        # the figure instead of leaving a stale number on it.
        assert len(CaseRole) < N_WIDENED_SYMBOLS

    def test_the_clause_row_names_the_canonical_gate_not_the_ablation(self) -> None:
        """Stage 5 audits two gates and the row must say which one it draws.

        `stage5_clause` is the canonical widened-observation arm; the case-only
        model keeps its own verdict in the ablation report under
        `stage5_clause_case_only`. Drawing the ablation's outcome in the column
        the caption says carries the audited outcome is the defect decision 114
        created the opportunity for.
        """
        clause = next(row for row in LEVELS if row.name == "Clause")
        assert clause.gate_id == GateId.STAGE5_CLAUSE

    def test_letter_level_a_is_identity_and_phoneme_level_a_is_learned(self) -> None:
        """The distinction the figure exists to make must hold in the data."""
        by_name = {row.name: row for row in LEVELS}
        assert by_name["Letter"].a_kind == "identity"
        assert by_name["Phoneme"].a_kind == "learned"

    def test_phoneme_is_the_first_learned_emission(self) -> None:
        first_learned = next(i for i, r in enumerate(LEVELS) if r.a_kind == "learned")
        assert LEVELS[first_learned].name == "Phoneme"
        assert all(r.a_kind == "identity" for r in LEVELS[:first_learned])

    def test_unbuilt_levels_label_their_matrices_as_planned(self) -> None:
        """A pending level must not describe design intent as a measurement."""
        for row in LEVELS:
            if row.built or row.a_kind == "external":
                continue
            assert row.emission.startswith("planned:"), row
            assert row.transition.startswith("planned:"), row

    def test_symbol_counts_come_from_the_code(self) -> None:
        """A hardcoded 30 or 39 would drift when the inventory changes."""
        by_name = {row.name: row for row in LEVELS}
        assert f"({N_SYMBOLS} symbols)" in by_name["Letter"].hidden
        assert str(N_SYMBOLS) in by_name["Phoneme"].emission

    def test_write_architecture_figures_writes_both(self, tmp_path) -> None:
        paths = write_architecture_figures(tmp_path)
        assert [p.name for p in paths] == ["hierarchy.png", "inference_cycle.png"]
        for path in paths:
            assert path.stat().st_size > _MIN_PNG_BYTES

    def test_write_architecture_figures_closes_its_figures(self, tmp_path) -> None:
        """A generator that leaks figures exhausts memory over a full render."""
        plt.close("all")
        write_architecture_figures(tmp_path)
        assert plt.get_fignums() == []


# ── Matrix figures ───────────────────────────────────────────────────────────


class TestMatrixFigures:
    """The letter transition heatmap and the wiring comparison."""

    def test_transition_matrix_shape_and_normalization(self, brown_matrix) -> None:
        assert brown_matrix.shape == (N_SYMBOLS, N_SYMBOLS)
        assert np.allclose(brown_matrix.sum(axis=0), 1.0, atol=1e-6), (
            "Columns must sum to 1 — this is the axis pymdp validates."
        )

    def test_transition_matrix_has_no_zero_cells(self, brown_matrix) -> None:
        """Add-1 smoothing means every transition has positive probability."""
        assert brown_matrix.min() > 0.0

    def test_transition_matrix_orientation_is_next_given_current(self, brown_matrix) -> None:
        """``matrix[i, j] = P(next=i | current=j)``, not the transpose.

        ``q`` is followed by ``u`` in almost all English; ``u`` is followed by
        ``q`` almost never. A transposed matrix would flip that, so this
        asymmetry pins the orientation the figure claims in its axis labels.
        """
        q, u = ALPHA_TO_IDX["q"], ALPHA_TO_IDX["u"]
        assert brown_matrix[u, q] > 0.9, brown_matrix[u, q]
        assert brown_matrix[q, u] < 0.01, brown_matrix[q, u]

    def test_marked_bigrams_are_all_in_the_alphabet(self) -> None:
        for current, following in MARKED_BIGRAMS:
            assert current in ALPHA_TO_IDX and following in ALPHA_TO_IDX

    def test_marked_bigrams_are_actually_high_probability(self, brown_matrix) -> None:
        """A "high-probability cell" annotation has to be true of the data.

        Each marked transition must beat the uniform 1/30 rate by a clear margin,
        otherwise the figure would be pointing at noise and calling it structure.
        """
        uniform = 1.0 / N_SYMBOLS
        for current, following in MARKED_BIGRAMS:
            probability = brown_matrix[ALPHA_TO_IDX[following], ALPHA_TO_IDX[current]]
            assert probability > 2 * uniform, (
                f"{current} -> {following} is {probability:.4f}, not clearly above "
                f"the uniform {uniform:.4f} rate."
            )

    def test_letter_transition_returns_a_legible_figure(self) -> None:
        fig = generate_letter_transition_figure()
        try:
            assert isinstance(fig, Figure)
            assert_legible(fig)
        finally:
            plt.close(fig)

    def test_letter_transition_labels_every_alphabet_symbol(self) -> None:
        """All 30 symbols must be on both axes, with space made visible."""
        fig = generate_letter_transition_figure()
        try:
            ax = fig.axes[0]
            x_labels = [t.get_text() for t in ax.get_xticklabels()]
            y_labels = [t.get_text() for t in ax.get_yticklabels()]
            assert len(x_labels) == N_SYMBOLS
            assert x_labels == y_labels
            assert "␣" in x_labels, "The space symbol must be drawn visibly."
            assert "q" in x_labels and "z" in x_labels
        finally:
            plt.close(fig)

    def test_wiring_comparison_returns_a_legible_figure(self) -> None:
        fig = generate_wiring_comparison_figure()
        try:
            assert isinstance(fig, Figure)
            assert_legible(fig)
        finally:
            plt.close(fig)

    def test_wiring_comparison_shows_the_ledger_values_verbatim(self, variables) -> None:
        """The figure must print the same string the manuscript prints."""
        fig = generate_wiring_comparison_figure()
        try:
            rendered = {t.get_text() for t in fig.findobj(matplotlib.text.Text)}
            blob = " | ".join(sorted(rendered))
            for token in (
                "STAGE1_LETTER_PERPLEXITY",
                "STAGE1_ANTI_INFORMATIVE_PERPLEXITY",
                "STAGE1_UNIGRAM_BASELINE",
                "STAGE1_UNIFORM_BASELINE",
                "STAGE1_BIGRAM_FLOOR_PERPLEXITY",
            ):
                assert variables[token] in blob, (
                    f"{token} = {variables[token]!r} does not appear in the figure."
                )
        finally:
            plt.close(fig)

    def test_wiring_comparison_needs_the_ledger(self) -> None:
        """A missing measurement must fail the figure, not be filled in."""
        with pytest.raises(MissingVariableError, match="do not put the number"):
            generate_wiring_comparison_figure({"SOMETHING_ELSE": "1.0"})

    def test_the_anti_informative_wiring_is_worse_than_the_unigram_bar(self, variables) -> None:
        """The claim the figure is built to convey, checked against the ledger.

        If this ordering ever inverted, the figure's argument — that the
        mis-wiring is worse than using no context at all — would be false, and
        the panel would be asserting something the numbers do not support.
        """
        anti = token_float(variables, "STAGE1_ANTI_INFORMATIVE_PERPLEXITY")
        unigram = token_float(variables, "STAGE1_UNIGRAM_BASELINE")
        uniform = token_float(variables, "STAGE1_UNIFORM_BASELINE")
        correct = token_float(variables, "STAGE1_LETTER_PERPLEXITY")
        floor = token_float(variables, "STAGE1_BIGRAM_FLOOR_PERPLEXITY")
        assert correct < unigram < anti < uniform
        assert floor < correct, "The model cannot beat its own information floor."

    def test_write_matrix_figures_writes_both(self, tmp_path) -> None:
        paths = write_matrix_figures(tmp_path)
        assert [p.name for p in paths] == [
            "letter_transition.png",
            "wiring_comparison.png",
        ]
        for path in paths:
            assert path.stat().st_size > _MIN_PNG_BYTES

    def test_write_matrix_figures_closes_its_figures(self, tmp_path) -> None:
        plt.close("all")
        write_matrix_figures(tmp_path)
        assert plt.get_fignums() == []


# ── The Stage-column badges against the re-runnable gate audit ───────────────


def test_every_drawn_verdict_is_one_the_audit_produces(gate_audit_rows) -> None:
    """A badge naming a placement the audit never computes is a typed claim.

    The Stage column says it carries each gate's audited placement verdict. This
    binds the words on the canvas to the words ``scripts/audit_gates.py`` writes,
    so a verdict cannot drift onto a row the audit does not support.
    """
    audited = {str(row["verdict"]) for row in gate_audit_rows.values()}
    for level in architecture.LEVELS:
        if not level.status:
            continue
        verdict = level.status.split("\n")[0].strip()
        if not verdict.isupper():
            # Level 7 is an interface contract, not an audited gate.
            continue
        assert verdict in audited, (
            f"{level.level} draws the verdict {verdict!r}, which "
            f"scripts/audit_gates.py does not produce for any gate: {sorted(audited)}"
        )


def test_a_row_naming_a_gate_draws_that_gate_s_audited_verdict(gate_audit_rows) -> None:
    """The badge must be the audit's verdict for the gate the row names.

    Two stages audit more than one gate, so a row that draws one of them has to
    say which. Stage 5 drew a failing ablation badge under a bare "Stage 5" label
    while the audit recorded its clause gate as passing, and nothing caught it
    because the badge was typed and unattributed.
    """
    audited = gate_audit_rows
    named = [level for level in architecture.LEVELS if getattr(level, "gate_id", None) is not None]
    assert named, "no level row names its gate id; the audit binding checks nothing"
    for level in named:
        gate_id = str(level.gate_id)
        assert gate_id in audited, (
            f"{level.level} names gate {gate_id!r}, which the audit does not report"
        )
        verdict, outcome = level.status.split("\n")
        assert verdict.strip() == str(audited[gate_id]["verdict"]), (
            f"{level.level} draws {verdict.strip()} but the audit records "
            f"{audited[gate_id]['verdict']} for {gate_id}"
        )
        # The outcome half may qualify a pass — NONCONF., DEV-SESSION — but a
        # bare PASS may not stand where the audit did not record one.
        if outcome.strip() == "PASS":
            assert str(audited[gate_id]["outcome"]) == "PASS", (
                f"{level.level} draws a bare PASS but the audit records "
                f"{audited[gate_id]['outcome']} for {gate_id}"
            )


def test_a_row_says_which_gate_it_draws_where_its_stage_audits_more_than_one(
    gate_audit_rows,
) -> None:
    """A badge under a bare stage label does not say which gate earned it.

    Stage 2 audits a phoneme-perplexity gate and a G2P accuracy gate; Stage 5
    audits the clause gate and its retained case-only ablation. Each of those
    rows draws exactly one verdict, and until the row names its gate the reader
    cannot tell which. That is not hypothetical: the Stage 5 row drew the
    ablation's failing verdict under a passing gate's stage label, and nothing
    caught it because the badge was unattributed.

    The rule is derived from the audit rather than from a list of stage names,
    so a stage that gains a second gate later is covered on the same commit.
    """
    audited = gate_audit_rows
    gates_per_stage: dict[str, set[str]] = {}
    for gate_id in audited:
        stage = gate_id.split("_", 1)[0]
        gates_per_stage.setdefault(stage, set()).add(gate_id)

    ambiguous = {stage for stage, ids in gates_per_stage.items() if len(ids) > 1}
    assert ambiguous, "no stage audits more than one gate; this guard is vacuous"

    for level in architecture.LEVELS:
        gate_id = getattr(level, "gate_id", None)
        if gate_id is None:
            continue
        stage = str(gate_id).split("_", 1)[0]
        if stage in ambiguous:
            assert level.gate_note, (
                f"{level.level} draws {gate_id} while {stage} audits "
                f"{sorted(gates_per_stage[stage])}; the row must say which one it draws"
            )


def test_the_drawn_stage_cell_carries_the_gate_note() -> None:
    """The note has to reach the canvas, not just the row object."""
    figure = generate_hierarchy_figure()
    try:
        drawn = " ".join(t.get_text() for t in figure.findobj(matplotlib.text.Text))
        for level in architecture.LEVELS:
            if level.gate_note:
                assert f"{level.stage} · {level.gate_note}" in drawn
    finally:
        plt.close(figure)
