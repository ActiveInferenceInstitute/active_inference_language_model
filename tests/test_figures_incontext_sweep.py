"""The zoom-inset contract of ``fig:incontext_sweep``.

The panel's whole point is a comparison the marks cannot support at full-axis
scale: the series band is a few points wide on an axis tens of points tall,
because the comparator rule at the AILM agent's accuracy sets the ceiling. The
layout keeps the full axis truthful — every mark, both reference rules — and
adds a zoom inset fitted to the drawn band. These tests pin that contract
against a deterministic synthetic token map: the inset exists, is a true zoom
(identical x-limits, narrower y, nothing drawn clipped away), the parent keeps
both reference levels, and the region the inset enlarges is marked on the
parent.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex
from matplotlib.inset import InsetIndicator

from ailm.benchmarks.constants import SERVED_PROBES
from ailm.benchmarks.synthetic_protocol import SYNTHETIC_INCONTEXT_PROTOCOL
from ailm.figures._common import (
    assert_legible,
    assert_no_text_crosses_rules,
    assert_within_canvas,
)
from ailm.figures.incontext_sweep import generate_incontext_sweep, incontext_sweep_data
from ailm.viz_config import PALETTE


def _tokens() -> dict[str, str]:
    """A deterministic token map whose series band is narrower than the axis."""
    tokens = {
        "BENCH_CTX_FIGURE_TITLE": "Registry title check",
        "BENCH_CTX_TOP1_AILM": "0.75",
        "BENCH_CTX_UNIFORM_BASELINE": "0.04",
        "BENCH_CTX_POSITIONS": "4",
    }
    for probe in SERVED_PROBES:
        suffix = probe.key.upper()
        for index, k in enumerate(SYNTHETIC_INCONTEXT_PROTOCOL.k_grid):
            value = 0.1 + index / 20
            tokens[f"BENCH_CTX_TOP1_K{k}_{suffix}"] = str(value)
            tokens[f"BENCH_CTX_TOP1_WILSON_LOW_K{k}_{suffix}"] = str(value / 2)
            tokens[f"BENCH_CTX_TOP1_WILSON_HIGH_K{k}_{suffix}"] = str(min(1.0, value * 1.5))
            if k:
                tokens[f"BENCH_CTX_TOP1_K{k}_DERANGED_{suffix}"] = str(value / 2)
                tokens[f"BENCH_CTX_TOP1_WILSON_LOW_K{k}_DERANGED_{suffix}"] = str(value / 4)
                tokens[f"BENCH_CTX_TOP1_WILSON_HIGH_K{k}_DERANGED_{suffix}"] = str(value * 0.75)
    return tokens


@pytest.fixture(scope="module")
def figure():
    """The built figure, generated once for the module."""
    built = generate_incontext_sweep(_tokens())
    yield built
    plt.close(built)


def _band() -> tuple[float, float]:
    """The union of every drawn Wilson interval, in percentage points."""
    data = incontext_sweep_data(_tokens())
    lows = [
        100.0 * value
        for probe in data["probes"]
        for arm in ("treatment_low", "control_low")
        for value in data["series"][probe][arm]
    ]
    highs = [
        100.0 * value
        for probe in data["probes"]
        for arm in ("treatment_high", "control_high")
        for value in data["series"][probe][arm]
    ]
    return min(lows), max(highs)


def test_every_panel_carries_exactly_one_zoom_inset(figure):
    """One panel per registered probe, each with one inset child."""
    assert len(figure.axes) == len(SERVED_PROBES)
    for axes in figure.axes:
        assert len(axes.child_axes) == 1


def test_the_inset_is_a_true_zoom_fitted_to_the_drawn_band(figure):
    """The inset narrows y onto the band, shares x, and hides nothing drawn."""
    band_low, band_high = _band()
    for axes in figure.axes:
        (inset,) = axes.child_axes
        lo, hi = inset.get_ylim()
        assert lo <= band_low and hi >= band_high, (
            "the inset's y-limits clip an interval the parent draws"
        )
        parent_lo, parent_hi = axes.get_ylim()
        assert hi - lo < parent_hi - parent_lo, "the inset spans the parent's axis"
        assert inset.get_xlim() == axes.get_xlim(), (
            "the inset re-plots x instead of zooming the panel"
        )


def test_the_parent_panels_keep_both_reference_levels(figure):
    """The zoom must not retire the comparators that set the axis ceiling."""
    data = incontext_sweep_data(_tokens())
    levels = {100.0 * data["ailm"], 100.0 * data["uniform"]}
    for axes in figure.axes:
        drawn = {
            float(value)
            for line in axes.lines
            if len(line.get_ydata()) >= 2
            and float(line.get_ydata()[0]) == float(line.get_ydata()[-1])
            for value in (line.get_ydata()[0],)
        }
        assert levels <= drawn, f"missing reference rules: {levels - drawn}"


def test_the_inset_redraws_both_arms_in_the_parent_encoding(figure):
    """Both series appear in the zoom, encoded as they are in the panel."""
    expected = {to_hex(PALETTE["MODEL"]), to_hex(PALETTE["UNIGRAM"])}
    for axes in figure.axes:
        (inset,) = axes.child_axes
        found = {to_hex(line.get_color()) for line in inset.lines}
        assert expected <= found


def test_the_zoom_region_is_marked_on_the_parent(figure):
    """Each parent carries the inset indicator marking the enlarged range."""
    for axes in figure.axes:
        assert any(isinstance(artist, InsetIndicator) for artist in axes.artists)


def test_the_figure_passes_every_layout_gate(figure):
    """The synthetic map must satisfy the same gates the sweep runs on tokens."""
    assert_legible(figure)
    assert_within_canvas(figure)
    assert_no_text_crosses_rules(figure)
