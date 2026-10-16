"""External benchmark figure, sourced only from ledger tokens."""

from __future__ import annotations

from typing import TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ailm.benchmarks.constants import (
    HOSTED_SYNTHETIC_DISPLAY,
    SERVED_PROBES,
    ProbeSpec,
    hosted_cohort_label,
)
from ailm.figures._common import load_manuscript_variables, token_float
from ailm.viz_config import FONTS, PALETTE, apply_style, qualitative_colors


def _model_presentation(
    probes: tuple[ProbeSpec, ...],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return registry-derived labels and distinct colors for every system."""

    keys = ("ailm",) + tuple(probe.key for probe in probes)
    labels = {"ailm": "AILM", **{probe.key: probe.display_name for probe in probes}}
    colors = dict(zip(keys, qualitative_colors(len(keys)), strict=True))
    return labels, colors


#: Comparator first, then each probe. These are separate model families, not
#: rungs of a scale ladder, so the panels order them and never connect them.
PROBE_KEYS = tuple(probe.key for probe in SERVED_PROBES)
MODEL_KEYS = ("ailm",) + PROBE_KEYS
#: The zero-shot synthetic producer and its in-context consumer both serve the
#: complete registered probe inventory; the figure follows that same registry.
SYNTH_KEYS = MODEL_KEYS
MODEL_LABELS, MODEL_COLORS = _model_presentation(SERVED_PROBES)

_TASKS = ("Letters", "Phonemes", "L2B position")
_FIELDS = ("LETTER_TOP1", "PHONEME_TOP1", "G2P_POS_ACC")
_DELTA_FIELDS = ("LETTER", "PHONEME", "G2P")
#: The third paired series is the G2P EXACT-match difference, not the position
#: accuracy the top-left panel plots, so the delta panel carries its own labels.
#: Reusing the accuracy labels named a quantity the bars do not show.
_DELTA_TASKS = ("Letters", "Phonemes", "L2B exact match")
_BUCKETS = ("SHORT", "MID", "LONG")


class ExternalBenchmarkData(TypedDict):
    """Typed plotting payload assembled exclusively from manuscript tokens."""

    tasks: list[str]
    series: dict[str, list[float]]
    lows: dict[str, list[float]]
    highs: dict[str, list[float]]
    deltas: dict[str, list[float]]
    delta_lows: dict[str, list[float]]
    delta_highs: dict[str, list[float]]
    context: dict[str, list[float]]
    context_lows: dict[str, list[float]]
    context_highs: dict[str, list[float]]
    buckets: dict[str, list[float]]
    bucket_lows: dict[str, list[float]]
    bucket_highs: dict[str, list[float]]
    bucket_labels: list[str]
    calibration: dict[str, list[float]]
    calibration_models: list[str]
    calibration_withheld: list[str]
    windows: list[float]
    invalid_rates: list[float]
    synthetic: dict[str, list[float]]
    synthetic_lows: dict[str, list[float]]
    synthetic_highs: dict[str, list[float]]
    synthetic_floors: list[float]
    bound_confidence_pct: float


def _token(v: dict[str, str], name: str) -> float:
    return token_float(v, name.upper())


def external_benchmarks_data(variables: dict[str, str]) -> ExternalBenchmarkData:
    """Pull every plotted value from the token map for token-drift tests."""
    series = {
        model: [_token(variables, f"BENCH_{field}_{model}") for field in _FIELDS]
        for model in MODEL_KEYS
    }
    lows = {
        model: [_token(variables, f"BENCH_{field}_WILSON_LOW_{model}") for field in _FIELDS]
        for model in MODEL_KEYS
    }
    highs = {
        model: [_token(variables, f"BENCH_{field}_WILSON_HIGH_{model}") for field in _FIELDS]
        for model in MODEL_KEYS
    }
    deltas = {
        model: [_token(variables, f"BENCH_{task}_DELTA_{model}") for task in _DELTA_FIELDS]
        for model in PROBE_KEYS
    }
    delta_lows = {
        model: [_token(variables, f"BENCH_{task}_DELTA_LOW_{model}") for task in _DELTA_FIELDS]
        for model in PROBE_KEYS
    }
    delta_highs = {
        model: [_token(variables, f"BENCH_{task}_DELTA_HIGH_{model}") for task in _DELTA_FIELDS]
        for model in PROBE_KEYS
    }
    context = {
        model: [
            _token(variables, f"BENCH_LETTER_SHORT_TOP1_{model}"),
            _token(variables, f"BENCH_LETTER_LONG_TOP1_{model}"),
            _token(variables, f"BENCH_LETTER_AZ_TOP1_{model}"),
        ]
        for model in MODEL_KEYS
    }
    # The three letter variants and the three phone-count strata carry Wilson
    # intervals on their own denominators, like the headline panel.
    _context_fields = ("LETTER_SHORT_TOP1", "LETTER_LONG_TOP1", "LETTER_AZ_TOP1")
    context_lows = {
        model: [_token(variables, f"BENCH_{field}_WILSON_LOW_{model}") for field in _context_fields]
        for model in MODEL_KEYS
    }
    context_highs = {
        model: [
            _token(variables, f"BENCH_{field}_WILSON_HIGH_{model}") for field in _context_fields
        ]
        for model in MODEL_KEYS
    }
    buckets = {
        model: [_token(variables, f"BENCH_G2P_POS_{bucket}_{model}") for bucket in _BUCKETS]
        for model in MODEL_KEYS
    }
    bucket_lows = {
        model: [
            _token(variables, f"BENCH_G2P_POS_{bucket}_WILSON_LOW_{model}") for bucket in _BUCKETS
        ]
        for model in MODEL_KEYS
    }
    bucket_highs = {
        model: [
            _token(variables, f"BENCH_G2P_POS_{bucket}_WILSON_HIGH_{model}") for bucket in _BUCKETS
        ]
        for model in MODEL_KEYS
    }
    # Only systems whose calibration was actually measurable are plotted. The
    # manuscript variables remove the ECE and Brier tokens of every probe the
    # direct dual-rule gate withholds (decision 93), so this filter is what keeps
    # ``_token`` resolvable and keeps the panel free of a bar a reader would rank.
    # The withheld systems are named in-panel rather than drawn, so the panel
    # itself says why it shows fewer systems than its neighbors.
    calibration_models = ["ailm"] + [
        model
        for model in PROBE_KEYS
        if int(round(_token(variables, f"BENCH_CALIB_MEASURABLE_{model}")))
    ]
    calibration_withheld = [model for model in PROBE_KEYS if model not in calibration_models]
    calibration = {
        model: [
            _token(variables, f"BENCH_CALIB_ECE_{model}"),
            _token(variables, f"BENCH_CALIB_BRIER_{model}"),
        ]
        for model in calibration_models
    }
    # The stipulated panel draws both chains beside each other: the letter
    # chain and the phone chain, each with its own uniform floor. Every value
    # comes from the recorded ledger tokens, per arm and per model.
    synthetic = {
        model: [
            _token(variables, f"BENCH_SYNTH_LETTER_TOP1_{model.upper()}"),
            _token(variables, f"BENCH_SYNTHPHONE_PHONE_TOP1_{model.upper()}"),
        ]
        for model in SYNTH_KEYS
    }
    synthetic_lows = {
        model: [
            _token(variables, f"BENCH_SYNTH_LETTER_TOP1_WILSON_LOW_{model.upper()}"),
            _token(variables, f"BENCH_SYNTHPHONE_PHONE_TOP1_WILSON_LOW_{model.upper()}"),
        ]
        for model in SYNTH_KEYS
    }
    synthetic_highs = {
        model: [
            _token(variables, f"BENCH_SYNTH_LETTER_TOP1_WILSON_HIGH_{model.upper()}"),
            _token(variables, f"BENCH_SYNTHPHONE_PHONE_TOP1_WILSON_HIGH_{model.upper()}"),
        ]
        for model in SYNTH_KEYS
    }
    return {
        "tasks": list(_TASKS),
        "series": series,
        "lows": lows,
        "highs": highs,
        "deltas": deltas,
        "delta_lows": delta_lows,
        "delta_highs": delta_highs,
        "context": context,
        "context_lows": context_lows,
        "context_highs": context_highs,
        "buckets": buckets,
        "bucket_lows": bucket_lows,
        "bucket_highs": bucket_highs,
        "bucket_labels": [variables[f"BENCH_G2P_BUCKET_{bucket}_RANGE"] for bucket in _BUCKETS],
        "calibration": calibration,
        "calibration_models": calibration_models,
        "calibration_withheld": calibration_withheld,
        "windows": [
            _token(variables, "BENCH_LETTER_WINDOW_SHORT"),
            _token(variables, "BENCH_LETTER_WINDOW_LONG"),
        ],
        "invalid_rates": [_token(variables, f"BENCH_UNPARSABLE_RATE_{m}") for m in PROBE_KEYS],
        "synthetic": synthetic,
        "synthetic_lows": synthetic_lows,
        "synthetic_highs": synthetic_highs,
        "synthetic_floors": [
            _token(variables, "BENCH_SYNTH_UNIFORM_BASELINE"),
            _token(variables, "BENCH_SYNTHPHONE_UNIFORM_BASELINE"),
        ],
        "bound_confidence_pct": _token(variables, "BOUND_CONFIDENCE_PCT"),
    }


def _grouped_bars(
    ax,
    categories: list[str] | tuple[str, ...],
    values_by_model: dict[str, list[float]],
    models: list[str] | tuple[str, ...],
    *,
    errors: tuple[dict[str, list[float]], dict[str, list[float]]] | None = None,
    width: float = 0.21,
    labels: dict[str, str] | None = None,
) -> None:
    x = list(range(len(categories)))
    offset = (len(models) - 1) / 2
    for index, model in enumerate(models):
        positions = [item + (index - offset) * width for item in x]
        bars = values_by_model[model]
        kwargs: dict[str, object] = {}
        if errors is not None:
            low, high = errors
            kwargs["yerr"] = [
                [b - value for b, value in zip(bars, low[model], strict=True)],
                [value - b for b, value in zip(bars, high[model], strict=True)],
            ]
            kwargs["capsize"] = 3
        ax.bar(
            positions,
            bars,
            width=width,
            label=(MODEL_LABELS if labels is None else labels)[model],
            color=MODEL_COLORS[model],
            edgecolor=PALETTE["OUTLINE"],
            **kwargs,
        )
    ax.set_xticks(x, categories)


def build_external_benchmarks_figure(variables: dict[str, str] | None = None) -> Figure:
    apply_style()
    values = variables or load_manuscript_variables()
    data = external_benchmarks_data(values)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5))
    (accuracy_ax, delta_ax, context_ax), (bucket_ax, calibration_ax, synthetic_ax) = axes

    _grouped_bars(
        accuracy_ax,
        data["tasks"],
        data["series"],
        MODEL_KEYS,
        errors=(data["lows"], data["highs"]),
    )
    confidence = data["bound_confidence_pct"]
    accuracy_ax.set_title(
        f"Same-position accuracy, {confidence:g}% Wilson",
        fontsize=FONTS["title"],
    )
    accuracy_ax.set_ylabel("Accuracy", fontsize=FONTS["axis_label"])
    accuracy_ax.set_ylim(0, 1)
    accuracy_ax.legend(fontsize=FONTS["legend"], loc="upper left")

    _grouped_bars(
        delta_ax,
        list(_DELTA_TASKS),
        data["deltas"],
        PROBE_KEYS,
        errors=(data["delta_lows"], data["delta_highs"]),
        width=0.26,
    )
    delta_ax.axhline(0.0, color=PALETTE["INK"], linewidth=1.0)
    delta_ax.set_title(
        f"Paired difference vs AILM, exact {confidence:g}%",
        fontsize=FONTS["title"],
    )
    delta_ax.set_ylabel("Probe - AILM", fontsize=FONTS["axis_label"])
    delta_ax.legend(fontsize=FONTS["legend"], loc="upper right", framealpha=0.95)

    windows = data["windows"]
    context_labels = [f"{int(windows[0])} chars", f"{int(windows[1])} chars", "a-z gold"]
    _grouped_bars(
        context_ax,
        context_labels,
        data["context"],
        MODEL_KEYS,
        errors=(data["context_lows"], data["context_highs"]),
    )
    context_ax.set_title(f"Letter task variants, {confidence:g}% Wilson", fontsize=FONTS["title"])
    context_ax.set_ylabel("Accuracy", fontsize=FONTS["axis_label"])
    context_ax.set_ylim(0, 1)
    context_ax.legend(fontsize=FONTS["legend"], loc="upper left")

    _grouped_bars(
        bucket_ax,
        data["bucket_labels"],
        data["buckets"],
        MODEL_KEYS,
        errors=(data["bucket_lows"], data["bucket_highs"]),
    )
    bucket_ax.set_title(
        f"L2B position accuracy by word length, {confidence:g}% Wilson", fontsize=FONTS["title"]
    )
    bucket_ax.set_ylabel("Accuracy", fontsize=FONTS["axis_label"])
    bucket_ax.set_ylim(0, 1)
    bucket_ax.legend(fontsize=FONTS["legend"], loc="upper right")

    # ECE is bounded at 1 and the multiclass Brier score at 2, so raw values on one
    # axis make every Brier bar look worse than every ECE bar for a reason that is
    # arithmetic rather than empirical. Each is drawn as a fraction of its own
    # worst case, which is what makes the two comparable. The data function still
    # returns the raw recorded values, so the token-drift check compares against
    # the ledger and not against a plotting transform.
    worst_case = (1.0, 2.0)
    scaled = {
        model: [value / bound for value, bound in zip(values, worst_case, strict=True)]
        for model, values in data["calibration"].items()
    }
    _grouped_bars(
        calibration_ax,
        ["ECE\n(worst 1)", "Brier\n(worst 2)"],
        scaled,
        data["calibration_models"],
    )
    calibration_ax.set_title("Letter-task calibration (lower better)", fontsize=FONTS["title"])
    calibration_ax.set_ylabel("Error / metric worst case", fontsize=FONTS["axis_label"])
    calibration_ax.set_ylim(0, 1)
    calibration_ax.legend(fontsize=FONTS["legend"], loc="upper left")
    if data["calibration_withheld"]:
        calibration_ax.text(
            0.98,
            0.97,
            # Three short lines: a single long line would run under the legend.
            "withheld — not measurable\n(decision 93):\n"
            + ", ".join(MODEL_LABELS[model] for model in data["calibration_withheld"]),
            transform=calibration_ax.transAxes,
            ha="right",
            va="top",
            fontsize=FONTS["legend"],
            color=PALETTE["INK"],
        )

    _stipulated_categories = ["letters", "phones"]
    _grouped_bars(
        synthetic_ax,
        _stipulated_categories,
        data["synthetic"],
        SYNTH_KEYS,
        errors=(data["synthetic_lows"], data["synthetic_highs"]),
        width=0.25,
        labels={"ailm": "AILM", **{key: hosted_cohort_label(key) for key in PROBE_KEYS}},
    )
    # Each stipulated chain has its own uniform floor, so one shared line would
    # assert a floor the other chain does not have. Each segment spans only its
    # own category, and the two dash patterns keep the legend entries
    # distinguishable at the same ink color.
    floor_linestyles = ("--", ":")
    for index, floor in enumerate(data["synthetic_floors"]):
        synthetic_ax.plot(
            [index - 0.42, index + 0.42],
            [floor, floor],
            color=PALETTE["INK"],
            linewidth=1.2,
            linestyle=floor_linestyles[index],
            label=f"uniform floor, {_stipulated_categories[index]} ({floor:.1%})",
        )
    synthetic_ax.set_title(
        f"{HOSTED_SYNTHETIC_DISPLAY} runs\nstipulated letters and phones",
        fontsize=FONTS["title"],
    )
    synthetic_ax.set_ylabel("Accuracy", fontsize=FONTS["axis_label"])
    synthetic_ax.set_ylim(0, 1)
    synthetic_ax.legend(fontsize=FONTS["legend"], loc="upper left")

    fig.suptitle(
        "Local reference models and hosted stipulated-language runs beside AILM",
        fontsize=FONTS["suptitle"],
    )
    fig.tight_layout()
    return fig


def generate_external_benchmarks(variables: dict[str, str] | None = None) -> Figure:
    """Generate the external benchmark comparison; the pipeline owns saving."""
    return build_external_benchmarks_figure(variables)
