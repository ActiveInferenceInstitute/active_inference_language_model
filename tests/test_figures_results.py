"""Tests for the AILM results and methodology figures.

These tests generate the figures for real — no mocks, no stubbed matplotlib — and then
assert four properties that a figure can fail silently:

1. **It exists and is a real PNG** of publication size at 200 dpi.
2. **It is deterministic**: regenerating produces byte-identical output. A timestamp or
   an unsorted iteration would break reproducibility without breaking the image.
3. **It is legible**: every text artist is at or above ``viz_config.MIN_FONT_PT`` and no
   text escapes the figure. Legibility is a build-time invariant, not a runtime clamp.
4. **It agrees with the manuscript.** Each figure's data function is checked against
   ``output/data/manuscript_variables.json`` token by token, so a figure cannot show one
   number while the prose beside it shows another.

Property 4 is backed by negative controls. Every cross-check inside the data functions
(ladder monotonicity, the nat-scale identities, the G2P margin identity, the taxonomy
verdicts, the subsample constant) is fed a deliberately inconsistent token map here and
must raise. Without those, the checks would be decoration that happens to pass.
"""

from __future__ import annotations

import math
import os
import struct
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

# Headless before matplotlib is imported by anything below.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.text import Text

from ailm.figures import benchmarks as benchmarks_fig
from ailm.figures import gates, results
from ailm.figures import incontext_sweep as incontext_fig
from ailm.figures._common import (
    FigureOverflowError,
    MissingVariableError,
    assert_within_canvas,
    rendered_texts,
)
from ailm.viz_config import MIN_FONT_PT

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _PROJECT_ROOT / "scripts" / "generate_figures.py"

#: (figure name, build function, generate function) for every figure this module owns.
FIGURES: tuple[tuple[str, Callable[..., Any], Callable[..., Any]], ...] = (
    (
        "perplexity_ladder",
        results.build_perplexity_ladder_figure,
        results.generate_perplexity_ladder,
    ),
    (
        "information_ceiling",
        results.build_information_ceiling_figure,
        results.generate_information_ceiling,
    ),
    ("g2p_accuracy", results.build_g2p_accuracy_figure, results.generate_g2p_accuracy),
    ("gate_taxonomy", gates.build_gate_taxonomy_figure, gates.generate_gate_taxonomy),
    ("subsample_bias", gates.build_subsample_bias_figure, gates.generate_subsample_bias),
)

FIGURE_NAMES = [name for name, _b, _g in FIGURES]


# ── External benchmark figure: values must come from the ledger tokens ─────────


def _bench_token_rows() -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Every (panel, model, tokens) triple the benchmark figure plots.

    Built from the figure module's own key tuples rather than typed out, so a
    model added to the figure is covered here on the same commit instead of
    plotting an unchecked series. The token *shapes* are still spelled out, so a
    renamed label fails rather than silently following the rename.
    """
    tasks = ("LETTER_TOP1", "PHONEME_TOP1", "G2P_POS_ACC")
    deltas = ("LETTER", "PHONEME", "G2P")
    rows: list[tuple[str, str, tuple[str, ...]]] = []
    for model in benchmarks_fig.MODEL_KEYS:
        upper = model.upper()
        rows.append(("series", model, tuple(f"BENCH_{task}_{upper}" for task in tasks)))
        rows.append(("lows", model, tuple(f"BENCH_{task}_WILSON_LOW_{upper}" for task in tasks)))
        rows.append(("highs", model, tuple(f"BENCH_{task}_WILSON_HIGH_{upper}" for task in tasks)))
        rows.append(
            (
                "context",
                model,
                (
                    f"BENCH_LETTER_SHORT_TOP1_{upper}",
                    f"BENCH_LETTER_LONG_TOP1_{upper}",
                    f"BENCH_LETTER_AZ_TOP1_{upper}",
                ),
            )
        )
        context_fields = ("LETTER_SHORT_TOP1", "LETTER_LONG_TOP1", "LETTER_AZ_TOP1")
        rows.append(
            (
                "context_lows",
                model,
                tuple(f"BENCH_{field}_WILSON_LOW_{upper}" for field in context_fields),
            )
        )
        rows.append(
            (
                "context_highs",
                model,
                tuple(f"BENCH_{field}_WILSON_HIGH_{upper}" for field in context_fields),
            )
        )
        buckets = ("SHORT", "MID", "LONG")
        rows.append(("buckets", model, tuple(f"BENCH_G2P_POS_{b}_{upper}" for b in buckets)))
        rows.append(
            (
                "bucket_lows",
                model,
                tuple(f"BENCH_G2P_POS_{b}_WILSON_LOW_{upper}" for b in buckets),
            )
        )
        rows.append(
            (
                "bucket_highs",
                model,
                tuple(f"BENCH_G2P_POS_{b}_WILSON_HIGH_{upper}" for b in buckets),
            )
        )
        # Calibration rows are added per measurable model by the caller, since
        # membership depends on the recorded captured-mass verdict.

    for model in benchmarks_fig.PROBE_KEYS:
        upper = model.upper()
        rows.append(("deltas", model, tuple(f"BENCH_{task}_DELTA_{upper}" for task in deltas)))
        rows.append(
            ("delta_lows", model, tuple(f"BENCH_{task}_DELTA_LOW_{upper}" for task in deltas))
        )
        rows.append(
            ("delta_highs", model, tuple(f"BENCH_{task}_DELTA_HIGH_{upper}" for task in deltas))
        )
    for model in benchmarks_fig.SYNTH_KEYS:
        upper = model.upper()
        rows.append(
            (
                "synthetic",
                model,
                (
                    f"BENCH_SYNTH_LETTER_TOP1_{upper}",
                    f"BENCH_SYNTHPHONE_PHONE_TOP1_{upper}",
                ),
            )
        )
        rows.append(
            (
                "synthetic_lows",
                model,
                (
                    f"BENCH_SYNTH_LETTER_TOP1_WILSON_LOW_{upper}",
                    f"BENCH_SYNTHPHONE_PHONE_TOP1_WILSON_LOW_{upper}",
                ),
            )
        )
        rows.append(
            (
                "synthetic_highs",
                model,
                (
                    f"BENCH_SYNTH_LETTER_TOP1_WILSON_HIGH_{upper}",
                    f"BENCH_SYNTHPHONE_PHONE_TOP1_WILSON_HIGH_{upper}",
                ),
            )
        )
    return tuple(rows)


_BENCH_TOKEN_ROWS = _bench_token_rows()


def _calibration_rows(variables) -> list[tuple[str, str, tuple[str, ...]]]:
    """Calibration token rows for exactly the systems the figure plots."""
    data = benchmarks_fig.external_benchmarks_data(variables)
    return [
        (
            "calibration",
            model,
            (f"BENCH_CALIB_ECE_{model.upper()}", f"BENCH_CALIB_BRIER_{model.upper()}"),
        )
        for model in data["calibration_models"]
    ]


def test_calibration_is_plotted_only_where_it_was_measurable(variables) -> None:
    """A probe below the captured-mass floor must not appear on the panel.

    Below the floor the distribution is mostly completion rather than
    measurement, and completion lowers mean confidence, which on a low-accuracy
    model *lowers* the calibration error. An unmeasurable model would therefore
    plot as the best-calibrated one on the panel.
    """
    data = benchmarks_fig.external_benchmarks_data(variables)
    plotted = set(data["calibration_models"])
    assert "ailm" in plotted, "the comparator's exact posterior is always measurable"
    # The withheld systems are the registered probes the panel does not draw, and
    # the panel names them so the reader is told why it shows fewer systems.
    withheld = set(data["calibration_withheld"])
    assert withheld == set(benchmarks_fig.PROBE_KEYS) - plotted
    if withheld:
        fig = benchmarks_fig.build_external_benchmarks_figure(variables)
        try:
            calibration_ax = fig.axes[4]
            assert "calibration" in calibration_ax.get_title()
            panel_text = "\n".join(artist.get_text() for artist in calibration_ax.texts)
            assert "withheld" in panel_text
            for model in withheld:
                assert benchmarks_fig.MODEL_LABELS[model] in panel_text, model
        finally:
            plt.close(fig)
    for model in benchmarks_fig.PROBE_KEYS:
        measurable = bool(
            int(round(results.parse_token(variables[f"BENCH_CALIB_MEASURABLE_{model.upper()}"])))
        )
        assert (model in plotted) == measurable, (
            f"{model}: plotted={model in plotted} but measurable={measurable}"
        )
        if not measurable:
            assert f"BENCH_CALIB_ECE_{model.upper()}" not in variables, (
                f"{model} failed the captured-mass gate, so its ECE must not be recorded "
                f"at all — a resolvable token invites exactly the misreading the gate exists "
                f"to prevent"
            )


def test_bench_token_rows_cover_every_plotted_series() -> None:
    """NON-VACUITY: the drift check must examine every model the figure draws."""
    covered = {model for _kind, model, _tokens in _BENCH_TOKEN_ROWS}
    assert covered == set(benchmarks_fig.MODEL_KEYS)
    assert len(_BENCH_TOKEN_ROWS) >= 3 * len(benchmarks_fig.MODEL_KEYS)


def test_external_benchmarks_values_match_the_ledger(variables) -> None:
    """Every plotted benchmark value must equal its manuscript token."""
    data = benchmarks_fig.external_benchmarks_data(variables)
    for kind, model, tokens in (*_BENCH_TOKEN_ROWS, *_calibration_rows(variables)):
        for index, token in enumerate(tokens):
            assert data[kind][model][index] == results.parse_token(variables[token]), (
                f"benchmark {kind}[{model}][{index}] does not match {token}"
            )
    for index, model in enumerate(benchmarks_fig.PROBE_KEYS):
        assert data["invalid_rates"][index] == results.parse_token(
            variables[f"BENCH_UNPARSABLE_RATE_{model.upper()}"]
        )
    assert data["windows"] == [
        results.parse_token(variables["BENCH_LETTER_WINDOW_SHORT"]),
        results.parse_token(variables["BENCH_LETTER_WINDOW_LONG"]),
    ]

    drifted = results.load_variables()
    drifted["BENCH_LETTER_TOP1_GEMMA3"] = "0.999"
    drifted["BENCH_CALIB_ECE_AILM"] = "0.777"
    drifted["BENCH_LETTER_LONG_TOP1_QWEN25"] = "0.555"
    drifted["BENCH_SYNTH_LETTER_TOP1_AILM"] = "0.800"
    drifted["BENCH_SYNTH_UNIFORM_BASELINE"] = "0.050"
    drifted["BENCH_SYNTHPHONE_PHONE_TOP1_AILM"] = "0.900"
    drifted["BENCH_SYNTHPHONE_UNIFORM_BASELINE"] = "0.040"
    data = benchmarks_fig.external_benchmarks_data(drifted)
    assert data["series"]["gemma3"][0] == 0.999
    assert data["calibration"]["ailm"][0] == 0.777
    assert data["context"]["qwen25"][1] == 0.555
    assert data["synthetic"]["ailm"][0] == 0.800
    assert data["synthetic"]["ailm"][1] == 0.900
    assert data["synthetic_floors"] == [0.050, 0.040]


def test_incontext_figure_refuses_an_omitted_registered_probe(variables) -> None:
    """NEGATIVE CONTROL — a complete missing probe must not disappear quietly."""

    incomplete = {
        name: value
        for name, value in variables.items()
        if not (name.startswith("BENCH_CTX_") and name.endswith("_QWEN25"))
    }
    with pytest.raises(MissingVariableError, match="probe inventory"):
        incontext_fig.incontext_sweep_data(incomplete)


def test_incontext_figure_refuses_an_omitted_registered_rung(variables) -> None:
    """NEGATIVE CONTROL — a complete missing grid rung must fail the figure."""

    incomplete = {
        name: value
        for name, value in variables.items()
        if not (
            name.startswith("BENCH_CTX_TOP1_K48_") or ("TOP1_WILSON_" in name and "K48_" in name)
        )
    }
    with pytest.raises(MissingVariableError, match="rung inventory"):
        incontext_fig.incontext_sweep_data(incomplete)


@pytest.fixture(scope="module")
def variables() -> dict[str, str]:
    """The real manuscript token map."""
    return results.load_variables()


@pytest.fixture(params=FIGURES, ids=FIGURE_NAMES)
def figure_case(request: pytest.FixtureRequest) -> tuple[str, Callable, Callable]:
    """One (name, build, generate) triple."""
    return request.param


def _png_dimensions(path: Path) -> tuple[int, int]:
    """Read (width, height) in pixels straight from the PNG IHDR chunk."""
    raw = path.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"
    width, height = struct.unpack(">II", raw[16:24])
    return width, height


def _texts(fig) -> list[Text]:
    """Every non-empty text artist in a figure."""
    return [
        artist
        for artist in fig.findobj(Text)
        if isinstance(artist, Text) and artist.get_text().strip()
    ]


# ── 1. THE FIGURE EXISTS AND IS A REAL PNG ───────────────────────────────────


def test_generator_writes_a_publication_size_png(figure_case, tmp_path: Path) -> None:
    name, _build, generate = figure_case
    path = Path(generate(output_dir=tmp_path))

    assert path.exists(), f"{name}: generator returned {path}, which does not exist"
    assert path.parent == tmp_path, f"{name}: wrote outside the requested directory"
    assert path.name == f"{name}.png", f"{name}: unexpected filename {path.name}"

    width, height = _png_dimensions(path)
    # Every figure in this module is at least 9.5 in wide; at the project's 200 dpi
    # that is 1900 px. A figure materially smaller than that has lost its dpi setting,
    # which is exactly how text stops being legible at print size.
    assert width >= 1800, f"{name}: {width}px wide — dpi setting lost?"
    assert height >= 1000, f"{name}: {height}px tall — dpi setting lost?"
    assert path.stat().st_size > 20_000, f"{name}: {path.stat().st_size} bytes is empty"


# ── 2. DETERMINISM ───────────────────────────────────────────────────────────


def test_regeneration_is_byte_identical(figure_case, tmp_path: Path) -> None:
    name, _build, generate = figure_case
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    path_a = Path(generate(output_dir=first))
    path_b = Path(generate(output_dir=second))

    assert path_a.read_bytes() == path_b.read_bytes(), (
        f"{name}: two runs produced different bytes. Something non-deterministic is "
        "in the figure — an embedded timestamp, an unsorted iteration, or an RNG."
    )


# ── 3. LEGIBILITY ────────────────────────────────────────────────────────────


def test_no_text_below_the_legibility_floor(figure_case) -> None:
    name, build, _generate = figure_case
    fig, _data = build()
    try:
        offenders = [
            (artist.get_text()[:40], artist.get_fontsize())
            for artist in _texts(fig)
            if artist.get_fontsize() < MIN_FONT_PT
        ]
        assert not offenders, (
            f"{name}: {len(offenders)} text artists below MIN_FONT_PT={MIN_FONT_PT}: "
            f"{offenders[:5]}"
        )
    finally:
        plt.close(fig)


def test_no_text_escapes_the_figure(figure_case) -> None:
    """No *rendered* label may cross the canvas edge.

    Containment is asserted through :func:`assert_within_canvas`, which measures
    only text matplotlib actually rasterises. Walking ``fig.findobj(Text)``
    directly does not work here: a ``Locator`` proposes ticks for the whole data
    range, so a zoomed or inverted axis keeps ``Text`` artists for ticks outside
    its view interval, positioned off-canvas and never drawn. The positive
    controls below prove the narrower check still catches real overflow.
    """
    name, build, _generate = figure_case
    fig, _data = build()
    try:
        assert_within_canvas(fig)
    except FigureOverflowError as exc:  # pragma: no cover - failure path
        pytest.fail(f"{name}: {exc}")
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("desc", "x", "y", "ha", "va"),
    [
        ("right", 0.99, 0.5, "left", "center"),
        ("left", 0.01, 0.5, "right", "center"),
        ("top", 0.5, 0.995, "center", "bottom"),
        ("bottom", 0.5, 0.005, "center", "top"),
    ],
)
def test_canvas_check_catches_overflow_in_every_direction(
    desc: str, x: float, y: float, ha: str, va: str
) -> None:
    """Positive control: the containment check must fail on real overflow.

    Without this, filtering undrawn tick labels could have quietly turned
    :func:`assert_within_canvas` into a check that never fires.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    try:
        ax.plot([0, 1], [0, 1])
        fig.text(x, y, "OVERFLOW" * 12, ha=ha, va=va, fontsize=12)
        with pytest.raises(FigureOverflowError, match="outside the figure canvas"):
            assert_within_canvas(fig)
    finally:
        plt.close(fig)


def test_canvas_check_still_catches_an_in_view_tick_label() -> None:
    """The undrawn-tick filter must not exempt tick labels that *are* drawn."""
    fig, ax = plt.subplots(figsize=(3, 2))
    try:
        ax.set_xlim(0, 1)
        ax.set_xticks([0.0])
        ax.set_xticklabels(["W" * 120], fontsize=12)
        with pytest.raises(FigureOverflowError, match="outside the figure canvas"):
            assert_within_canvas(fig)
    finally:
        plt.close(fig)


def test_canvas_check_refuses_an_empty_canvas() -> None:
    """A containment check that passes with no text at all is not a check."""
    fig = plt.figure()
    try:
        with pytest.raises(FigureOverflowError, match="no drawn text"):
            assert_within_canvas(fig)
    finally:
        plt.close(fig)


def test_undrawn_tick_labels_are_the_only_thing_filtered() -> None:
    """Pin the exemption: out-of-view ticks are dropped, in-view ones retained.

    ``gate_taxonomy`` is the real case — it zooms an axis so the locator proposes
    ticks outside the window. This reproduces that shape in miniature so the
    exemption is bound to a stated reason rather than to whatever makes the suite
    green.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        ax.set_xticks([0.0, 0.5, 1.0, 5.0])  # 5.0 lies outside the view
        ax.set_xlim(0.0, 1.0)
        fig.canvas.draw()

        all_labels = {artist.get_text() for artist in _texts(fig) if artist.get_text().strip()}
        drawn = {artist.get_text() for artist in rendered_texts(fig) if artist.get_text()}

        assert "5.0" in all_labels, "expected an out-of-view tick label to exist"
        assert "5.0" not in drawn, "out-of-view tick label was not filtered"
        for kept in ("0.0", "0.5", "1.0"):
            assert kept in drawn, f"in-view tick label {kept!r} was wrongly filtered"
    finally:
        plt.close(fig)


def test_every_axes_has_readable_tick_labels(figure_case) -> None:
    """Tick labels inherit from rcParams, so they need their own floor check."""
    name, build, _generate = figure_case
    fig, _data = build()
    try:
        for ax in fig.axes:
            for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
                if not label.get_text().strip():
                    continue
                assert label.get_fontsize() >= MIN_FONT_PT, (
                    f"{name}: tick label {label.get_text()!r} at "
                    f"{label.get_fontsize()}pt is below MIN_FONT_PT={MIN_FONT_PT}"
                )
    finally:
        plt.close(fig)


# ── 4. THE FIGURES AGREE WITH THE MANUSCRIPT ─────────────────────────────────


def test_perplexity_ladder_values_match_the_ledger(variables) -> None:
    data = results.perplexity_ladder_data(variables)
    expected = (
        {
            "uniform": "STAGE1_UNIFORM_BASELINE",
            "unigram": "STAGE1_UNIGRAM_BASELINE",
            "model": "STAGE1_LETTER_PERPLEXITY",
            "floor": "STAGE1_BIGRAM_FLOOR_PERPLEXITY",
        },
        {
            "uniform": "STAGE2_PHONEME_UNIFORM_BASELINE",
            "unigram": "STAGE2_PHONEME_UNIGRAM_BASELINE",
            "model": "STAGE2_PHONEME_PERPLEXITY",
            "floor": "STAGE2_PHONEME_ALL_POSITION_FLOOR_PERPLEXITY",
        },
    )
    assert len(data["panels"]) == len(expected)
    for panel, token_map in zip(data["panels"], expected, strict=True):
        for key, token_name in token_map.items():
            assert panel["values"][key] == results.parse_token(variables[token_name])
            # The bar label is the token's own string, character for character.
            assert panel["displays"][key] == variables[token_name]


def test_each_ladder_panel_states_the_population_it_was_scored_over(variables) -> None:
    """Two panels scored three orders of magnitude apart must say so.

    Side by side and unlabeled, the panels invite a rung-for-rung comparison
    that the populations do not support.
    """
    data = results.perplexity_ladder_data(variables)
    populations = [
        results.parse_token(variables["STAGE1_HELDOUT_POSITIONS"]),
        results.parse_token(variables["STAGE2_PHONEME_EVAL_POSITIONS"]),
    ]
    for panel, population in zip(data["panels"], populations, strict=True):
        assert panel["population"] == int(population)
        assert panel["population_unit"]


def test_an_empty_ladder_population_refuses_to_draw(variables) -> None:
    """Positive control for the population check."""
    tampered = dict(variables)
    tampered["STAGE1_HELDOUT_POSITIONS"] = "0"
    with pytest.raises(ValueError, match="not a population"):
        results.perplexity_ladder_data(tampered)


def test_the_drawn_ladder_names_its_floor_rule_and_its_population(variables) -> None:
    figure, _data = results.build_perplexity_ladder_figure(variables)
    try:
        drawn = " ".join(text.get_text() for text in figure.findobj(Text))
        # The rule is the panel's threshold, so it is named on the canvas rather
        # than only in the caption a reader may never reach.
        assert "order-1 conditional-" in drawn
        assert "held-out letter positions" in drawn
        assert "scored phoneme positions" in drawn
    finally:
        plt.close(figure)


def test_information_ceiling_values_match_the_ledger(variables) -> None:
    data = results.information_ceiling_data(variables)
    assert data["model_nats"] == results.parse_token(variables["STAGE1_LETTER_NATS"])
    assert data["floor_nats"] == results.parse_token(variables["STAGE1_BIGRAM_FLOOR_NATS"])
    assert data["achievable_gain_nats"] == results.parse_token(
        variables["STAGE1_MAX_BIGRAM_GAIN_NATS"]
    )
    # The declared gate is drawn from the ledger token, not from a literal in the
    # figure module, and every derived quantity follows the token.
    declared = results.parse_token(variables[results.DECLARED_GATE_TOKEN])
    assert data["demanded_gain_nats"] == declared
    assert data["demanded_nats"] == data["uniform_nats"] - declared
    assert data["shortfall_ratio"] == declared / data["achievable_gain_nats"]
    # The whole argument: the demanded target is below the floor, so the model's
    # position inside the achievable band is irrelevant to whether it could pass.
    assert data["demanded_nats"] < data["floor_nats"]
    assert data["floor_nats"] <= data["model_nats"] <= data["uniform_nats"]
    assert data["shortfall_nats"] > 0


def test_g2p_values_match_the_ledger(variables) -> None:
    data = results.g2p_accuracy_data(variables)
    for key, token_name in (
        ("baseline_pct", "STAGE2_G2P_BASELINE_ACCURACY"),
        ("model_pct", "STAGE2_G2P_ACCURACY"),
        ("margin_pts", "STAGE2_G2P_MARGIN_PTS"),
        ("mcnemar_p", "STAGE2_G2P_MCNEMAR_P"),
        ("mcnemar_b", "STAGE2_G2P_MCNEMAR_B"),
        ("mcnemar_c", "STAGE2_G2P_MCNEMAR_C"),
        ("positions", "STAGE2_G2P_EVAL_POSITIONS"),
    ):
        assert data[key] == results.parse_token(variables[token_name]), key
    # Unclaimed by design: nothing records the forward-backward ceiling yet, and the
    # figure must not invent one.
    if results.SMOOTHING_CEILING_TOKEN not in variables:
        assert data["smoothing_ceiling_pct"] is None


def test_g2p_decoder_bars_carry_render_time_wilson_intervals(variables) -> None:
    """Decision 101 on fig:g2p_accuracy: each decoder bar's interval is derived at
    render time from its recorded accuracy on the recorded eval-position denominator,
    not stored anywhere. The recorder never records an interval, so the bounds must
    re-derive exactly from ``wilson_interval`` on the rounded success count."""
    from ailm.benchmarks.scoring import wilson_interval

    data = results.g2p_accuracy_data(variables)
    n = round(data["positions"])
    for pct_key, low_key, high_key in (
        ("baseline_pct", "baseline_wilson_low_pct", "baseline_wilson_high_pct"),
        ("model_pct", "model_wilson_low_pct", "model_wilson_high_pct"),
    ):
        successes = round(data[pct_key] / 100.0 * n)
        low, high = wilson_interval(successes, n)
        assert data[low_key] == pytest.approx(low * 100.0), low_key
        assert data[high_key] == pytest.approx(high * 100.0), high_key
        # A Wilson interval on its own denominator always brackets the bar it is
        # drawn on; if it did not, the derivation would be drift, not precision.
        assert data[low_key] < data[pct_key] < data[high_key], pct_key


def _g2p_errorbar_bounds(ax) -> dict[int, tuple[float, float]]:
    """The ``(bar index -> (low bound, high bound))`` an axes actually draws."""
    from matplotlib.container import ErrorbarContainer

    containers = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
    # One errorbar call covers both decoder bars; the smoothing reference must
    # draw bare, so exactly one intervalled container may exist per panel.
    assert len(containers) == 1, f"expected 1 intervalled bar group, found {len(containers)}"
    bounds: dict[int, tuple[float, float]] = {}
    (_data_line, _caps, bar_cols) = containers[0]  # fmt="none": no center line
    for bar_col in bar_cols:
        for segment in bar_col.get_segments():
            (x, y0), (_x, y1) = segment
            low, high = sorted((y0, y1))
            assert round(x) not in bounds, f"duplicate error bar at x={x}"
            bounds[round(x)] = (low, high)
    return bounds


def test_g2p_figure_draws_the_intervals_on_both_decoder_bars(variables) -> None:
    """Both panels must draw the derived bounds as error bars on bars 0 and 1,
    and no interval on the smoothing reference."""
    fig, data = results.build_g2p_accuracy_figure(variables)
    try:
        expected = {
            0: (data["baseline_wilson_low_pct"], data["baseline_wilson_high_pct"]),
            1: (data["model_wilson_low_pct"], data["model_wilson_high_pct"]),
        }
        for ax in fig.axes:  # panel A (full scale) and panel B (truncated)
            drawn = _g2p_errorbar_bounds(ax)
            assert set(drawn) == {0, 1}
            for bar_index, (low, high) in expected.items():
                assert drawn[bar_index] == pytest.approx((low, high)), bar_index
    finally:
        plt.close(fig)


def test_g2p_refuses_an_accuracy_with_no_integer_success_count(variables) -> None:
    """POSITIVE CONTROL — an accuracy that cannot come from an integer success
    count at the recorded denominator must refuse, not draw a wrong interval."""
    positions = round(results.parse_token(variables["STAGE2_G2P_EVAL_POSITIONS"]))
    baseline = results.parse_token(variables["STAGE2_G2P_BASELINE_ACCURACY"])
    # The first three-decimal percentage point above the recorded baseline that is
    # not consistent with any integer success count at the recorded denominator.
    bad_pct = next(
        candidate
        for candidate in (baseline + k / 1000 for k in range(1, 10))
        if abs(candidate / 100.0 * positions - round(candidate / 100.0 * positions))
        > 5e-6 * positions + 1e-9
    )
    broken = dict(variables)
    broken["STAGE2_G2P_BASELINE_ACCURACY"] = str(bad_pct)
    # Keep the margin identity consistent so the new guard, not the old one, fires.
    broken["STAGE2_G2P_MARGIN_PTS"] = str(
        results.parse_token(variables["STAGE2_G2P_ACCURACY"]) - bad_pct
    )
    with pytest.raises(ValueError, match="integer success count"):
        results.g2p_accuracy_data(broken)
    # And the guard is not vacuous: the untouched map carries a real interval.
    data = results.g2p_accuracy_data(variables)
    assert data["baseline_wilson_low_pct"] < data["baseline_wilson_high_pct"]


def test_gate_taxonomy_values_match_the_ledger(variables) -> None:
    data = gates.gate_taxonomy_data(variables)

    stage1 = data["stage1"]
    assert stage1["baseline"] == results.parse_token(variables["STAGE1_UNIGRAM_BASELINE"])
    assert stage1["model"] == results.parse_token(variables["STAGE1_LETTER_PERPLEXITY"])
    assert stage1["ceiling"] == results.parse_token(variables["STAGE1_BIGRAM_FLOOR_PERPLEXITY"])
    assert stage1["verdict"] == "UNREACHABLE"
    assert stage1["original_threshold"] < stage1["ceiling"]
    # The original threshold is uniform / e^(declared gain), with the gain read
    # from the ledger token rather than typed into the figure.
    declared = results.parse_token(variables[results.DECLARED_GATE_TOKEN])
    assert stage1["original_threshold"] == results.parse_token(
        variables["STAGE1_UNIFORM_BASELINE"]
    ) / math.exp(declared)

    stage2 = data["stage2b"]
    assert stage2["baseline"] == results.parse_token(variables["STAGE2_G2P_BASELINE_ACCURACY"])
    assert stage2["model"] == results.parse_token(variables["STAGE2_G2P_ACCURACY"])
    assert stage2["verdict"] == "VACUOUS"
    assert stage2["original_threshold"] < stage2["baseline"]


def test_the_retained_ablation_is_drawn_as_its_own_gate_row(variables) -> None:
    """The demoted clause arm is a second gate on Level 5, not a second level.

    fig:hierarchy is one row per level, so the ablation's non-passing verdict had
    nowhere to appear. It appears here, at its measured position, and its verdict
    and outcome must be the ones the audit records for
    ``stage5_clause_case_only``.
    """
    data = gates.gate_taxonomy_data(variables)
    row = data["stage5_ablation"]
    assert row["verdict"] == "ABOVE_REFERENCE"
    assert row["outcome"] == "FAIL"
    assert row["baseline"] == results.parse_token(variables["STAGE5_TYPEFREE_PERPLEXITY"])
    assert row["model"] == results.parse_token(variables["STAGE5_MIXTURE_PERPLEXITY"])
    assert row["ceiling"] == results.parse_token(variables["STAGE5_TYPE_ORACLE_PERPLEXITY"])
    # Its comparator loses to its own bar, so there is no band a threshold could
    # discriminate in and the figure must not state one.
    assert row["band_width"] is None
    assert not gates._comparator_beats_baseline(row)
    # The two passing rows keep theirs.
    for key in ("stage1", "stage2b"):
        assert gates._comparator_beats_baseline(data[key])
        assert data[key]["band_width"] is not None
        assert data[key]["outcome"] == "PASS"


def test_a_row_whose_comparator_loses_draws_no_bracketed_band(variables) -> None:
    """Two zones, not three, and no overlap between them.

    Emitting the bracketed span anyway produced two overlapping regions with both
    captions printed on top of each other in the overlap, advertising a verdict
    region no threshold on that row could ever land in.
    """
    data = gates.gate_taxonomy_data(variables)
    row = data["stage5_ablation"]
    spans = gates._zone_spans(row, row["model"] + 1.0, row["baseline"] - 1.0)
    verdicts = [verdict for _x0, _x1, verdict in spans]
    assert verdicts == ["VACUOUS", "ABOVE_REFERENCE"]
    # The figure orders zones from looser to stricter, which reverses numeric
    # order on a lower-is-better axis. Compare intervals in numeric order.
    ordered = sorted(sorted((x0, x1)) for x0, x1, _v in spans)
    assert ordered[0][1] <= ordered[1][0], f"zones overlap: {spans}"


def test_a_misreported_gate_outcome_refuses_to_draw(variables) -> None:
    """Positive control: the outcome half of every badge is checked."""
    tampered = dict(variables)
    # Make the ablation's model beat its bar; its recorded FAIL then contradicts
    # the arithmetic.
    tampered["STAGE5_MIXTURE_PERPLEXITY"] = str(
        results.parse_token(variables["STAGE5_TYPEFREE_PERPLEXITY"]) - 0.001
    )
    with pytest.raises(ValueError, match="but the row reports FAIL"):
        gates.gate_taxonomy_data(tampered)


def test_every_measured_gate_row_states_its_population_and_its_band(variables) -> None:
    """A verdict about a threshold is a verdict about a measurement.

    The row has to say what it was scored on, what it was fitted on, and how
    much room a threshold ever had between the trivial baseline and the
    achievable ceiling.
    """
    data = gates.gate_taxonomy_data(variables)
    for key in gates._MEASURED_ROWS:
        row = data[key]
        assert row["population"] > 0
        assert row["population_unit"]
        assert row["fitted_on"]
        if row["band_width"] is None:
            continue
        assert row["band_width"] == pytest.approx(abs(row["ceiling"] - row["baseline"]))
        # A passing row draws the shipped model inside the band; if it stopped
        # being inside, the picture would be asserting something false.
        low, high = sorted((row["baseline"], row["ceiling"]))
        assert low <= row["model"] <= high
    # Bind the populations to their tokens so a hardcoded copy cannot pass.
    assert data["stage1"]["population"] == int(
        results.parse_token(variables["STAGE1_HELDOUT_POSITIONS"])
    )
    assert data["stage2b"]["population"] == int(
        results.parse_token(variables["STAGE2_G2P_EVAL_POSITIONS"])
    )
    # Stage 2B's verdict rests on a paired per-position comparison and carries it;
    # Stage 1's perplexity does not have one and says so rather than implying one.
    assert data["stage1"]["test"] is None
    assert data["stage2b"]["test"]["p"] == results.parse_token(variables["STAGE2_G2P_MCNEMAR_P"])
    assert data["stage2b"]["test"]["discordant"] == int(
        results.parse_token(variables["STAT_STAGE2_DISCORDANT_N"])
    )


def test_a_model_outside_its_own_band_refuses_to_draw(variables) -> None:
    """Positive control: the band claim must be able to fail."""
    tampered = dict(variables)
    tampered["STAGE2_G2P_ACCURACY"] = str(
        results.parse_token(variables["STAGE2_G2P_SMOOTHING_CEILING"]) + 1.0
    )
    with pytest.raises(ValueError, match="outside the band"):
        gates.gate_taxonomy_data(tampered)


def test_a_paired_test_larger_than_its_population_refuses_to_draw(variables) -> None:
    """Positive control: the population and the paired test must agree."""
    tampered = dict(variables)
    tampered["STAGE2_G2P_EVAL_POSITIONS"] = str(
        int(results.parse_token(variables["STAT_STAGE2_DISCORDANT_N"])) - 1
    )
    with pytest.raises(ValueError, match="came from different runs"):
        gates.gate_taxonomy_data(tampered)


def test_the_drawn_gate_taxonomy_carries_its_band_and_its_population(variables) -> None:
    figure, _data = gates.build_gate_taxonomy_figure(variables)
    try:
        drawn = " ".join(text.get_text() for text in figure.findobj(Text))
        assert "the only band a threshold discriminates in" in drawn
        assert "wide" in drawn
        assert "scored on" in drawn
        assert "exact McNemar, model vs baseline" in drawn
        assert "no paired per-item test attaches" in drawn
    finally:
        plt.close(figure)


def test_subsample_full_row_comes_from_the_ledger(variables) -> None:
    """The full row reads the tokens ``SUBSAMPLE_COMPARISON`` names for it.

    Against ``SUBSAMPLE_FULL_*``, not ``STAGE2_G2P_*``. Those are different
    labels from a different recorder that hold equal values today, so asserting
    against them passes for the wrong reason and stops tracking the tokens the
    figure actually reads the moment the two recorders diverge.
    """
    data = gates.subsample_bias_data(variables)
    full = next(row for row in data["rows"] if row["key"] == "full")
    assert full["margin_pts"] == results.parse_token(variables["SUBSAMPLE_FULL_MARGIN_PTS"])
    assert full["mcnemar_p"] == results.parse_token(variables["SUBSAMPLE_FULL_MCNEMAR_P"])
    assert full["positions"] == int(results.parse_token(variables["SUBSAMPLE_FULL_POSITIONS"]))
    assert full["initials"] == int(results.parse_token(variables["SUBSAMPLE_FULL_INITIALS"]))


def test_subsample_comparison_tells_the_documented_story(variables) -> None:
    data = gates.subsample_bias_data(variables)
    by_key = {row["key"]: row for row in data["rows"]}

    # The prefix inflates the effect and narrows the alphabet.
    assert by_key["prefix"]["margin_pts"] > by_key["full"]["margin_pts"]
    assert by_key["prefix"]["initials"] < by_key["full"]["initials"]
    assert data["overstatement_ratio"] > 3.0

    # At the level the Stage 2 gate actually decides at (ALPHA_LEVEL, kept equal to
    # tests/test_stage2_gate.py::ALPHA by
    # tests/test_figure_layout.py::test_figure_alpha_matches_the_gate_alpha), ONLY the
    # full held-out set clears. Both representative strides are underpowered.
    #
    # This assertion previously read stride3 < ALPHA_LEVEL, which was true while the
    # figure was drawn at the conventional 0.05. That was a weaker and slightly
    # self-defeating story: at 0.05 the representative stride3 (p = 0.01696) reads as
    # adequate, which is the opposite of what the panel argues. The p-values did not
    # change; the decision level did, to match the gate.
    assert by_key["stride5"]["mcnemar_p"] > gates.ALPHA_LEVEL
    assert by_key["stride3"]["mcnemar_p"] > gates.ALPHA_LEVEL
    assert by_key["full"]["mcnemar_p"] < gates.ALPHA_LEVEL
    assert by_key["full"]["mcnemar_p"] < by_key["stride3"]["mcnemar_p"]
    assert by_key["stride5"]["positions"] < by_key["full"]["positions"]
    # The prefix is significant too — but for the wrong reason, on 4 initials. That
    # pairing is the panel's whole point, so pin it rather than leaving it implied.
    assert by_key["prefix"]["mcnemar_p"] < gates.ALPHA_LEVEL


# ── 4b. NEGATIVE CONTROLS FOR EVERY CROSS-CHECK ──────────────────────────────


def test_ladder_rejects_a_model_below_its_own_information_floor(variables) -> None:
    broken = dict(variables)
    broken["STAGE1_LETTER_PERPLEXITY"] = "10.0"  # below the 10.7506 floor
    with pytest.raises(ValueError, match="not monotone"):
        results.perplexity_ladder_data(broken)


def test_ladder_rejects_a_model_worse_than_the_unigram_bar(variables) -> None:
    broken = dict(variables)
    broken["STAGE1_LETTER_PERPLEXITY"] = "25.0"  # worse than the 19.9558 bar
    with pytest.raises(ValueError, match="not monotone"):
        results.perplexity_ladder_data(broken)


def test_ceiling_rejects_nats_claims_from_different_runs(variables) -> None:
    broken = dict(variables)
    broken["STAGE1_GAIN_OVER_UNIFORM_NATS"] = "0.5"
    with pytest.raises(ValueError, match="Ledger inconsistency"):
        results.information_ceiling_data(broken)

    broken = dict(variables)
    broken["STAGE1_MAX_BIGRAM_GAIN_NATS"] = "2.5"
    with pytest.raises(ValueError, match="Ledger inconsistency"):
        results.information_ceiling_data(broken)


def test_declared_gate_must_agree_with_the_code_owned_constant(variables) -> None:
    """NEGATIVE CONTROL — a re-recorded declaration the code does not own must refuse.

    Before the token was read, this perturbation was silently ignored: both
    figures kept drawing the literal in the module while the ledger said
    otherwise.
    """
    broken = dict(variables)
    broken[results.DECLARED_GATE_TOKEN] = "3"
    with pytest.raises(ValueError, match="Ledger inconsistency"):
        results.information_ceiling_data(broken)
    with pytest.raises(ValueError, match="Ledger inconsistency"):
        gates.gate_taxonomy_data(broken)


def test_declared_gate_is_read_from_the_ledger_not_typed(variables) -> None:
    """Without the token, neither figure may fall back to a literal."""
    missing = {k: v for k, v in variables.items() if k != results.DECLARED_GATE_TOKEN}
    with pytest.raises(MissingVariableError):
        results.information_ceiling_data(missing)
    with pytest.raises(MissingVariableError):
        gates.gate_taxonomy_data(missing)


def test_g2p_rejects_a_margin_that_is_not_model_minus_baseline(variables) -> None:
    broken = dict(variables)
    broken["STAGE2_G2P_MARGIN_PTS"] = "5.0"
    with pytest.raises(ValueError, match="Ledger inconsistency"):
        results.g2p_accuracy_data(broken)


def test_g2p_rejects_a_discordant_total_that_is_not_b_plus_c(variables) -> None:
    broken = dict(variables)
    broken["STAGE2_MCNEMAR_TOTAL_DISCORDANT"] = "999"
    with pytest.raises(ValueError, match="Ledger inconsistency"):
        results.g2p_accuracy_data(broken)


def test_taxonomy_rejects_a_ledger_that_no_longer_supports_the_verdict(
    variables,
) -> None:
    # Raise the baseline above the original 60% threshold and Stage 2B's gate would no
    # longer be VACUOUS; the figure must refuse rather than relabel the argument.
    broken = dict(variables)
    broken["STAGE2_G2P_BASELINE_ACCURACY"] = "40.0"
    broken["STAGE2_G2P_ACCURACY"] = "40.546"
    with pytest.raises(ValueError, match="classifies the original gate"):
        gates.gate_taxonomy_data(broken)


def test_subsample_values_are_loaded_from_the_ledger(variables) -> None:
    """A perturbed token must change the returned row.

    There is no second hardcoded measurement in ``SUBSAMPLE_COMPARISON`` for the
    ledger to disagree with, and this is what proves it: if the constant still
    carried the number, the perturbation would be ignored.
    """
    broken = dict(variables)
    broken["SUBSAMPLE_PREFIX_MARGIN_PTS"] = "0.900"
    data = gates.subsample_bias_data(broken)
    prefix = next(row for row in data["rows"] if row["key"] == "prefix")
    assert prefix["margin_pts"] == 0.900


def test_subsample_missing_token_raises(variables) -> None:
    broken = {key: value for key, value in variables.items() if key != "SUBSAMPLE_PREFIX_MCNEMAR_P"}
    with pytest.raises(MissingVariableError):
        gates.subsample_bias_data(broken)


def test_missing_token_raises_instead_of_substituting(variables) -> None:
    broken = {k: v for k, v in variables.items() if k != "STAGE1_LETTER_NATS"}
    with pytest.raises(MissingVariableError):
        results.information_ceiling_data(broken)


def test_missing_display_token_raises_instead_of_substituting(variables) -> None:
    """A value present for arithmetic but absent for labeling must still fail."""
    broken = {k: v for k, v in variables.items() if k != "STAGE2_G2P_TEST_PAIRS"}
    with pytest.raises(MissingVariableError):
        results.g2p_accuracy_data(broken)


def test_load_variables_reports_a_missing_token_map(tmp_path: Path) -> None:
    with pytest.raises(MissingVariableError, match="z_generate_manuscript_variables"):
        results.load_variables(tmp_path / "nope.json")


def test_parse_token_handles_prose_formatting() -> None:
    assert results.parse_token("42,820") == 42820.0
    assert results.parse_token("3.47e-06") == pytest.approx(3.47e-06)
    assert results.parse_token(" 70.801 ") == 70.801


def test_role_color_rejects_an_unknown_role() -> None:
    with pytest.raises(KeyError):
        results.role_color("chartreuse")


# ── 4b. CLASSIFIER BINDING ──────────────────────────────────────────────────


def test_gate_verdict_and_classify_agree() -> None:
    """The figure's wrapper must delegate to the shared typed classifier."""
    from ailm.gates import classify_gate

    _VECTORS: list[tuple[float, float, float, bool]] = [
        (10.0, 5.0, 10.0, True),  # baseline equality fails; ceiling clears
        (20.0, 10.0, 19.0, True),  # bracketed
        (20.0, 20.0, 19.0, True),  # ceiling cannot clear: unreachable/above-ref
        (38.0, 80.0, 39.0, False),  # bracketed
        (40.0, 60.0, 39.0, False),  # baseline passes: vacuous
        (40.0, 50.0, 50.0, False),  # ceiling equality: unreachable/above-ref
    ]

    for baseline, ceiling, threshold, lower_is_better in _VECTORS:
        direction = "lower_is_better" if lower_is_better else "higher_is_better"
        for kind, ceiling_type in (
            ("ceiling", "model_class_bound"),
            ("reference", "reference_model"),
        ):
            a = str(
                classify_gate(baseline, ceiling, threshold, direction, ceiling_type=ceiling_type)
            )
            f = gates._classify(baseline, ceiling, threshold, lower_is_better, kind)
            assert a == f, (
                f"classify_gate({baseline}, {ceiling}, {threshold}, "
                f"{lower_is_better}, {ceiling_type}) = {a!r} but gates._classify "
                f"returns {f!r}. The wrapper has drifted from the shared classifier."
            )


def test_gate_verdict_and_classify_agree_on_ceiling_none() -> None:
    """Missing ceiling data is UNAUDITED, with no kind required or accepted."""
    from ailm.gates import classify_gate

    f_none = gates._classify(20.0, None, 19.0, True)
    assert f_none == "UNAUDITED"
    assert str(classify_gate(20.0, None, 19.0, "lower_is_better")) == "UNAUDITED"


def test_classify_refuses_a_ceiling_without_a_kind() -> None:
    """A supplied ceiling with no kind cannot be classified; it must not be guessed."""
    with pytest.raises(ValueError, match="ceiling_kind"):
        gates._classify(20.0, 10.0, 19.0, True)


# ── 4c. THE TAXONOMY LEGEND NAMES ONLY DRAWN ZONES ──────────────────────────


def _taxonomy_legend_and_drawn_zones(fig) -> tuple[set[str], set[str]]:
    """Verdict names in the figure legend, and verdict captions drawn on a row."""
    verdicts = {verdict for verdict, _text in gates._LEGEND_ROWS}
    (legend,) = fig.legends
    in_legend = {label.get_text().split(" — ")[0] for label in legend.get_texts()}
    drawn = {
        artist.get_text() for ax in fig.axes for artist in ax.texts if artist.get_text() in verdicts
    }
    return in_legend, drawn


def test_gate_taxonomy_legend_names_exactly_the_drawn_zones(variables) -> None:
    """A legend entry for a zone no row draws advertises a region nobody can find."""
    fig, data = gates.build_gate_taxonomy_figure(variables)
    try:
        in_legend, drawn = _taxonomy_legend_and_drawn_zones(fig)
        assert drawn, "no verdict caption was drawn; the check would be vacuous"
        assert in_legend == drawn
        # Every ceiling in the current ledger is recorded, so no row is UNAUDITED.
        assert data["stage2b"]["ceiling"] is not None
        assert "UNAUDITED" not in in_legend
    finally:
        plt.close(fig)


def test_gate_taxonomy_legend_advertises_unaudited_only_when_it_is_drawn(variables) -> None:
    """POSITIVE CONTROL — the entry returns exactly when a row draws that zone."""
    without_ceiling = {
        key: value for key, value in variables.items() if key != results.SMOOTHING_CEILING_TOKEN
    }
    fig, data = gates.build_gate_taxonomy_figure(without_ceiling)
    try:
        assert data["stage2b"]["ceiling"] is None
        in_legend, drawn = _taxonomy_legend_and_drawn_zones(fig)
        assert "UNAUDITED" in drawn
        assert "UNAUDITED" in in_legend
        assert in_legend == drawn
    finally:
        plt.close(fig)


# ── 5. THE ORCHESTRATOR ──────────────────────────────────────────────────────


def _run_script(*argv: str) -> subprocess.CompletedProcess[str]:
    # Strictly below the marks on the tests that spawn this (1200 s), so a hang is
    # reported by the child with its output rather than by pytest-timeout.
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *argv],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        check=False,
        timeout=600,
    )


def _load_orchestrator():
    """Import ``scripts/generate_figures.py`` by path — scripts/ is not a package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("ailm_generate_figures", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_orchestrator_discovers_every_owned_figure() -> None:
    discovered = {g.name for g in _load_orchestrator().discover_generators()}
    missing = set(FIGURE_NAMES) - discovered
    assert not missing, (
        f"the orchestrator did not discover {sorted(missing)}; a figure the manuscript "
        "references would be missing from the build with no error"
    )


def test_orchestrator_discovery_is_deterministically_ordered() -> None:
    orchestrator = _load_orchestrator()
    first = [(g.module, g.name) for g in orchestrator.discover_generators()]
    assert first == sorted(first), "discovery order must be stable for the manifest"
    assert first == [(g.module, g.name) for g in orchestrator.discover_generators()]


@pytest.mark.timeout(1200)
def test_orchestrator_lists_all_owned_figures() -> None:
    result = _run_script("--list")
    assert result.returncode == 0, result.stderr
    for name in FIGURE_NAMES:
        assert name in result.stderr, f"{name} was not discovered by the orchestrator"


@pytest.mark.timeout(1200)
def test_orchestrator_generates_a_selected_figure(tmp_path: Path) -> None:
    result = _run_script("--only", "perplexity_ladder", "--output-dir", str(tmp_path))
    assert result.returncode == 0, result.stderr
    printed = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(printed) == 1, f"stdout must be a clean manifest, got {printed}"
    written = Path(printed[0])
    assert written.exists() and written.parent == tmp_path


@pytest.mark.timeout(1200)
def test_orchestrator_generates_a_whole_module(tmp_path: Path) -> None:
    result = _run_script("--only", "gates", "--output-dir", str(tmp_path))
    assert result.returncode == 0, result.stderr
    printed = sorted(Path(line).name for line in result.stdout.splitlines() if line)
    assert printed == ["gate_taxonomy.png", "subsample_bias.png"]


@pytest.mark.timeout(1200)
def test_orchestrator_fails_on_an_unknown_figure(tmp_path: Path) -> None:
    result = _run_script("--only", "not_a_figure", "--output-dir", str(tmp_path))
    assert result.returncode != 0
    assert "matched nothing" in result.stderr
