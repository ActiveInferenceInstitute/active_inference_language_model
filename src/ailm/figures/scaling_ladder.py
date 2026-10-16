"""The word level's coverage/perplexity trade across lexicon size.

Both numeric quantities rise as the lexicon grows: a bigger vocabulary covers
more of the held-out stream and has higher per-emission perplexity because it
faces a harder prediction population. Their utility directions oppose — more
coverage is desirable, more perplexity is not. A single operating point can be
quoted to support either reading, so the figure plots both against one lexicon
axis and reports the shared-denominator gain that widens across the ladder.

The third series is the one that makes the panel a check rather than an
illustration. The letter-bigram comparator is estimated once on the full training
frequencies and re-scored per rung, never re-fit, so its curve moves only with the
evaluation population. That population
shrinks with the lexicon, and it shrinks toward more frequent words, so the
per-emission gain is not comparable across rungs. The title carries both scales
for that reason: flat per emission, strictly rising once every rung shares one
denominator.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ailm.figures._common import load_manuscript_variables, token_float, token_int
from ailm.viz_config import FONTS, PALETTE, apply_style

__all__ = ["generate_scaling_ladder", "scaling_ladder_data"]

#: Where the shipped agent stops being instantiable. Not a property of the model:
#: the closed form is exact at any size, and the cap is a memory bound on the
#: dense ``pymdp`` transition. Drawn because the distinction between what can be
#: measured and what can be run is the section's point.
_CAP_LABEL = "pymdp agent cap"


def scaling_ladder_data(variables: dict[str, str] | None = None) -> dict[str, list[float]]:
    """Rungs read from the ledger tokens, smallest lexicon first.

    Discovers the rungs from the token namespace rather than from a list here.
    A hardcoded set of sizes would silently plot a subset the day the recorder's
    ``SCALING_RUNGS`` changed, and a figure that quietly drops a rung is the
    defect this project has already paid for in its figure-discovery sweep.

    Raises:
        ValueError: If fewer than two rungs are present. One point is not a
            ladder, and drawing it would assert a trend from a single
            measurement.
    """
    tokens = variables if variables is not None else load_manuscript_variables()
    sizes = sorted(
        int(name.split("_")[2])
        for name in tokens
        if name.startswith("STAGE3_SCALE_") and name.endswith("_COVERAGE_PCT")
    )
    if len(sizes) < 2:
        raise ValueError(
            f"found {len(sizes)} scaling rung(s) in the ledger; a ladder needs at "
            f"least two, and one point cannot show a trade"
        )
    return {
        "sizes": [float(size) for size in sizes],
        "coverage": [token_float(tokens, f"STAGE3_SCALE_{s}_COVERAGE_PCT") for s in sizes],
        "perplexity": [token_float(tokens, f"STAGE3_SCALE_{s}_WORD_PERPLEXITY") for s in sizes],
        "baseline": [token_float(tokens, f"STAGE3_SCALE_{s}_BIGRAM_BASELINE") for s in sizes],
        "positions": [float(token_int(tokens, f"STAGE3_SCALE_{s}_EVAL_POSITIONS")) for s in sizes],
        "cap": [float(token_int(tokens, "STAGE3_MAX_AGENT_VOCAB"))],
    }


def generate_scaling_ladder(variables: dict[str, str] | None = None) -> Figure:
    """Coverage and perplexity against lexicon size, with the comparator."""
    apply_style()
    data = scaling_ladder_data(variables)
    sizes = data["sizes"]

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    left.plot(sizes, data["coverage"], marker="o", color=PALETTE["MODEL"], linewidth=2)
    left.set_xscale("log")
    left.set_xlabel("Lexicon size (types, log scale)")
    left.set_ylabel("Held-out coverage (%)")
    left.set_title("Coverage rises with size")
    left.grid(True, alpha=0.3)

    right.plot(
        sizes,
        data["baseline"],
        marker="s",
        color=PALETTE["UNIGRAM"],
        linewidth=2,
        label="Letter-bigram bar",
    )
    right.plot(
        sizes,
        data["perplexity"],
        marker="o",
        color=PALETTE["MODEL"],
        linewidth=2,
        label="Word level",
    )
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("Lexicon size (types, log scale)")
    right.set_ylabel("Per-emission perplexity (log scale)")
    right.set_title("Both PPLs rise; shared gain widens")
    right.legend(frameon=False, loc="center left")
    right.grid(True, alpha=0.3)

    for axis in (left, right):
        cap = data["cap"][0]
        if min(sizes) <= cap <= max(sizes):
            axis.axvline(cap, color=PALETTE["GATE"], linestyle="--", linewidth=1.2)
            # Placed in axes coordinates rather than at the data floor: an
            # artist drawn at the limit is clipped by the layout gate, and an
            # artist drawn outside it silently expands the canvas under
            # bbox_inches="tight". Both failures have shipped here before.
            axis.annotate(
                _CAP_LABEL,
                xy=(cap, 0.06),
                xycoords=axis.get_xaxis_transform(),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=FONTS["cell"],
                rotation=90,
                va="bottom",
                ha="left",
                color=PALETTE["GATE"],
                clip_on=True,
            )

    gains = [
        math.log(b) - math.log(m) for b, m in zip(data["baseline"], data["perplexity"], strict=True)
    ]
    # Both scales in the title, because the per-emission range alone reads as a
    # flat gain and that flatness is the denominator rather than the model.
    shared = [g * p / data["positions"][-1] for g, p in zip(gains, data["positions"], strict=True)]
    figure.suptitle(
        f"Word level across lexicon size: per-emission gain {min(gains):.3f}–{max(gains):.3f} nats; "
        f"on a shared denominator {min(shared):.3f}\u2013{max(shared):.3f}",
        fontsize=FONTS["tick_label"],
    )
    figure.tight_layout()
    return figure
