"""What happens to an external probe given labeled chain transitions.

The supplement records that AILM is fitted offline while every external probe
starts without separately labeled demonstrations. This panel measures one
bounded remediation of that asymmetry. The probe is given ``k`` labeled
source-symbol/next-symbol pairs from the corpus AILM's counts come from and
``k`` is swept; the question is
whether the recorded margin narrows under this prompt format and grid.

Two series, not one, and the second is what makes the panel a check rather than
an illustration. The control shows the same examples with their prefix-to-symbol
mapping deranged and everything else — prefixes, symbol multiset, prompt length
— held fixed. A treatment curve alone cannot separate "the probe used the
chain from the examples" from "the probe was pulled toward the alphabet by
seeing a wall of plausible symbols". Where the two curves lie on top of each
other, the mapping-specific contribution is unresolved under this test; the
figure does not claim that the broader fitted-versus-unfitted asymmetry vanished.
"""

from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
from typing import TypedDict

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from ailm.benchmarks.constants import SERVED_PROBES, ProbeSpec, hosted_cohort_label
from ailm.benchmarks.synthetic_protocol import SYNTHETIC_INCONTEXT_PROTOCOL
from ailm.figures._common import (
    MissingVariableError,
    load_manuscript_variables,
    token_float,
    token_int,
)
from ailm.viz_config import FONTS, PALETTE, apply_style

__all__ = ["generate_incontext_sweep", "incontext_sweep_data"]


def _probe_titles(probes: tuple[ProbeSpec, ...]) -> dict[str, str]:
    """Map retained ledger suffixes to hosted run labels, not local models."""

    return {probe.key.upper(): hosted_cohort_label(probe.key) for probe in probes}


PROBE_TITLES = _probe_titles(SERVED_PROBES)


class ProbeSeries(TypedDict):
    """One probe's two curves, in rung order."""

    treatment: list[float]
    treatment_low: list[float]
    treatment_high: list[float]
    control: list[float]
    control_low: list[float]
    control_high: list[float]


class SweepData(TypedDict):
    """Everything the panel draws. Typed so a mis-keyed access fails at check time."""

    rungs: list[float]
    probes: list[str]
    series: dict[str, ProbeSeries]
    ailm: float
    uniform: float
    positions: float
    title: str


#: Probe label suffixes, discovered from the token namespace rather than listed.
#: This was a single hardcoded ``"GEMMA3"`` while one probe was registered, and a
#: figure that keeps plotting one curve after a second probe is recorded is the
#: quiet-drop defect this project has already paid for once in its figure sweep.
def _probes(tokens: dict[str, str]) -> list[str]:
    """Probe suffixes, matched on the token's full shape.

    Splitting on the last underscore was the first version and it discovered a
    probe called ``RUN``: the variable generator emits a ``_RUN`` and a
    ``_SOURCE`` provenance token beside every label, so ``rsplit`` read the
    provenance suffix as a system name. Anchoring the whole shape is what makes
    the discovery a rule rather than a guess.
    """
    pattern = re.compile(r"^BENCH_CTX_TOP1_K\d+_([A-Z0-9]+)$")
    return sorted(
        {
            match.group(1)
            for name in tokens
            if (match := pattern.match(name)) and match.group(1) not in {"DERANGED", "AILM"}
        }
    )


def incontext_sweep_data(variables: dict[str, str] | None = None) -> SweepData:
    """The rungs, both series per probe, and the two reference levels.

    Rungs and probes are discovered from the token namespace and then required
    to equal the frozen protocol and served-probe registries. A figure that
    quietly drops either is a defect this project has already paid for.

    Raises:
        MissingVariableError: If fewer than two example counts are present, or if
            no probe is found. One point cannot show a trend.
        ValueError: If a rung is missing its derangement control for any probe.
            One curve without its control is the reading this panel exists to
            prevent, and drawing it would assert in-context learning from a
            measurement that cannot distinguish it from display.
    """
    tokens = variables if variables is not None else load_manuscript_variables()
    title = tokens.get("BENCH_CTX_FIGURE_TITLE")
    if not isinstance(title, str) or not title.strip():
        raise MissingVariableError(
            "BENCH_CTX_FIGURE_TITLE is absent; regenerate manuscript variables "
            "after recording the complete in-context family."
        )
    probes = _probes(tokens)
    if not probes:
        raise MissingVariableError(
            "no in-context probe found in the ledger. Run "
            "scripts/record_incontext_benchmark_claims.py."
        )
    expected_probes = sorted(probe.key.upper() for probe in SERVED_PROBES)
    if probes != expected_probes:
        raise MissingVariableError(
            "in-context figure probe inventory does not match the served-probe "
            f"registry: found {probes}, expected {expected_probes}"
        )
    prefix = "BENCH_CTX_TOP1_K"
    first = f"_{probes[0]}"
    rungs = sorted(
        int(name[len(prefix) : -len(first)])
        for name in tokens
        if name.startswith(prefix) and name.endswith(first) and "_DERANGED" not in name
    )
    if len(rungs) < 2:
        raise MissingVariableError(
            f"found {len(rungs)} example count(s) in the ledger; a sweep needs at "
            f"least two, and one point cannot show a trend."
        )
    expected_rungs = list(SYNTHETIC_INCONTEXT_PROTOCOL.k_grid)
    if rungs != expected_rungs:
        raise MissingVariableError(
            "in-context figure rung inventory does not match the frozen protocol: "
            f"found {rungs}, expected {expected_rungs}"
        )

    series: dict[str, ProbeSeries] = {}
    for probe in probes:
        suffix = f"_{probe}"
        missing = [k for k in rungs if k and f"{prefix}{k}_DERANGED{suffix}" not in tokens]
        if missing:
            raise ValueError(
                f"probe {probe} is missing derangement controls at {missing}. The "
                f"treatment curve alone cannot separate learning from display."
            )
        treatment_names = [f"K{k}" for k in rungs]
        control_names = [f"K{k}_DERANGED" if k else "K0" for k in rungs]
        series[probe] = ProbeSeries(
            treatment=[token_float(tokens, f"{prefix}{k}{suffix}") for k in rungs],
            treatment_low=[
                token_float(tokens, f"BENCH_CTX_TOP1_WILSON_LOW_{name}{suffix}")
                for name in treatment_names
            ],
            treatment_high=[
                token_float(tokens, f"BENCH_CTX_TOP1_WILSON_HIGH_{name}{suffix}")
                for name in treatment_names
            ],
            # k=0 has no control by construction: there is nothing to derange, so
            # the control series starts at the no-demonstration value and the two curves
            # share an origin rather than one beginning in mid-air.
            control=[
                token_float(tokens, f"{prefix}{k}_DERANGED{suffix}" if k else f"{prefix}0{suffix}")
                for k in rungs
            ],
            control_low=[
                token_float(tokens, f"BENCH_CTX_TOP1_WILSON_LOW_{name}{suffix}")
                for name in control_names
            ],
            control_high=[
                token_float(tokens, f"BENCH_CTX_TOP1_WILSON_HIGH_{name}{suffix}")
                for name in control_names
            ],
        )

    return SweepData(
        rungs=[float(k) for k in rungs],
        probes=probes,
        series=series,
        ailm=token_float(tokens, "BENCH_CTX_TOP1_AILM"),
        uniform=token_float(tokens, "BENCH_CTX_UNIFORM_BASELINE"),
        positions=float(token_int(tokens, "BENCH_CTX_POSITIONS")),
        title=title,
    )


#: Vertical padding applied on each side of the zoom inset's y-limits, as a
#: fraction of the drawn series band — the union of every Wilson interval on
#: every arm of every probe. The zoom is fitted to the drawn evidence, not the
#: point estimates, so no interval the parent shows is clipped in the inset.
_BAND_PAD_FRACTION = 0.15

#: Floor on that padding, in percentage points of letter accuracy, so a
#: degenerate band — every interval identical — still yields a readable scale.
_MIN_BAND_PAD_PCT = 0.5

#: Placement of the zoom inset inside each panel, in parent-axes fraction:
#: (left, bottom, width, height). The comparator rule sets the ceiling, so the
#: series band tops out near the bottom fifth of the panel and the empty middle
#: band between the marks and the rule carries the inset.
_INSET_BOUNDS = (0.115, 0.295, 0.80, 0.475)


def _draw_sweep_series(
    axis: plt.Axes, rungs: list[float], rows: ProbeSeries, *, labeled: bool
) -> None:
    """Draw one probe's treatment and deranged-control arms with intervals.

    Shared by the panel and its zoom inset so the two cannot drift apart in
    encoding: the same colors, markers, linestyles and interval caps identify
    the same series in both places.

    Args:
        axis: Where to draw.
        rungs: The example counts on the x-axis.
        rows: The probe's two series with their Wilson bounds.
        labeled: Attach the legend labels. Only the panel carries them; the
            inset repeats the encoding, not the legend.
    """
    treatment = [100.0 * value for value in rows["treatment"]]
    control = [100.0 * value for value in rows["control"]]
    axis.errorbar(
        rungs,
        treatment,
        yerr=[
            [
                value - 100.0 * low
                for value, low in zip(treatment, rows["treatment_low"], strict=True)
            ],
            [
                100.0 * high - value
                for value, high in zip(treatment, rows["treatment_high"], strict=True)
            ],
        ],
        marker="o",
        color=PALETTE["MODEL"],
        linewidth=2,
        capsize=2.5,
        label="Labeled transitions shown" if labeled else None,
    )
    axis.errorbar(
        rungs,
        control,
        yerr=[
            [value - 100.0 * low for value, low in zip(control, rows["control_low"], strict=True)],
            [
                100.0 * high - value
                for value, high in zip(control, rows["control_high"], strict=True)
            ],
        ],
        marker="s",
        linestyle="--",
        color=PALETTE["UNIGRAM"],
        linewidth=2,
        capsize=2.5,
        label="Same examples, mapping deranged" if labeled else None,
    )


def generate_incontext_sweep(variables: dict[str, str] | None = None) -> Figure:
    """Probe accuracy against labeled one-transition examples, one panel each.

    Each panel keeps the full axis: the comparator rule at the AILM agent's
    measured accuracy sets the ceiling, and the distance between the marks and
    that rule is part of what the panel reports. But the treatment-versus-
    control separation the suptitle asserts is a few points wide on an axis
    tens of points tall, and cannot be read off the marks at that scale — the
    defect this layout exists to fix. So each panel also carries a zoom inset
    over the drawn series band: identical x-limits, its own labeled y-ticks, an
    indicator tying it to the region it enlarges, and no mark the parent does
    not also show. The parent keeps every mark and both reference rules.
    """
    apply_style()
    data = incontext_sweep_data(variables)
    rungs = data["rungs"]
    probes = data["probes"]
    ailm = 100.0 * data["ailm"]
    uniform = 100.0 * data["uniform"]

    figure, axes = plt.subplots(1, len(probes), figsize=(5.2 * len(probes), 4.8), sharey=True)
    if len(probes) == 1:
        axes = [axes]

    all_values = [ailm, uniform]
    all_values.extend(
        100.0 * value
        for probe in probes
        for arm in ("treatment", "control")
        for value in data["series"][probe][arm]
    )
    ceiling = min(100.0, max(all_values) * 1.15)

    band_low = min(
        100.0 * value
        for probe in probes
        for arm in ("treatment_low", "control_low")
        for value in data["series"][probe][arm]
    )
    band_high = max(
        100.0 * value
        for probe in probes
        for arm in ("treatment_high", "control_high")
        for value in data["series"][probe][arm]
    )
    pad = max((band_high - band_low) * _BAND_PAD_FRACTION, _MIN_BAND_PAD_PCT)

    for axis, probe in zip(axes, probes, strict=True):
        _draw_sweep_series(axis, rungs, data["series"][probe], labeled=True)
        axis.axhline(ailm, color=PALETTE["GATE"], linestyle=":", linewidth=1.6)
        # The floor rule carried no label and no legend entry, so it read as an
        # unexplained grey dash while the comparator rule directly above it has
        # always been named. Labeled here, and moved off a bare greyscale
        # literal onto the palette's secondary neutral.
        axis.axhline(uniform, color=PALETTE["MUTED"], linestyle=":", linewidth=1.2)
        axis.annotate(
            "uniform floor",
            xy=(max(rungs), uniform),
            xytext=(-4, 4),
            textcoords="offset points",
            fontsize=FONTS["cell"],
            ha="right",
            va="bottom",
            color=PALETTE["MUTED"],
        )
        axis.annotate(
            "AILM comparator",
            xy=(max(rungs), ailm),
            xytext=(-4, 4),
            textcoords="offset points",
            fontsize=FONTS["cell"],
            ha="right",
            va="bottom",
            color=PALETTE["GATE"],
        )
        axis.set_xlabel("Labeled one-transition examples supplied (k)")
        axis.set_ylabel("Letter accuracy (%)")
        axis.tick_params(axis="y", labelleft=True)
        axis.set_title(PROBE_TITLES[probe], fontsize=FONTS["tick_label"])
        axis.set_ylim(0.0, ceiling)
        axis.grid(True, alpha=0.3)

    # Settle the parent limits before anything reads them: an inset shares the
    # parent's exact x-limits or it is a re-plotting, not a zoom.
    figure.canvas.draw()
    inset_xlim = axes[0].get_xlim()
    for axis, probe in zip(axes, probes, strict=True):
        inset = axis.inset_axes(_INSET_BOUNDS)
        _draw_sweep_series(inset, rungs, data["series"][probe], labeled=False)
        inset.set_xlim(inset_xlim)
        inset.set_ylim(band_low - pad, band_high + pad)
        inset.set_xticks([rungs[0], rungs[len(rungs) // 2], rungs[-1]])
        inset.tick_params(axis="x", pad=-15)
        inset.yaxis.set_major_locator(MaxNLocator(nbins=3))
        inset.grid(True, alpha=0.3)
        inset.set_title("Treatment vs. control (zoom)", fontsize=FONTS["cell"], fontweight="normal")
        # The rectangle alone marks the region the inset enlarges. The default
        # corner connectors turn a full-width region into a funnel graphic
        # whose slanted sides read as data; the region outline and the inset's
        # own labeled axes carry the link without them.
        axis.indicate_inset(
            bounds=(
                inset_xlim[0],
                band_low - pad,
                inset_xlim[1] - inset_xlim[0],
                (band_high + pad) - (band_low - pad),
            ),
            facecolor="none",
            edgecolor=PALETTE["MUTED"],
            alpha=0.8,
            linewidth=0.8,
        )
    # The framed default (apply_style: an opaque white card) keeps the
    # comparator rule from striking through the legend text: the frameless
    # variant left the rule grazing the second entry.
    axes[0].legend(loc="upper left")

    figure.suptitle(data["title"], fontsize=FONTS["tick_label"])
    figure.tight_layout()
    return figure
