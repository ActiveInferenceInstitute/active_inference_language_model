"""The generation sample panel: real output, not a summary statistic of it.

Three text columns from the same habit and the same lexicon, at the three
operating points the sweep of ``fig:generation_sweep`` characterizes
numerically: ancestral sampling from the habit (no goal), the shipped interior
optimum, and the degenerate high-precision endpoint. Word tokens are colored
by lexicon membership, so the qualitative signature of each regime is visible
in the text itself — the baseline drifts out of the lexicon, the operating
point stays in it while remaining diverse, and the degenerate column collapses
onto almost a single word.

Every sample is regenerated at draw time from fixed seeds through the shipped
generators (`generate_by_sampling`, `generate_by_policy_selection`), so the
panel shows text the model actually produced — never an illustrative mock-up.
The gamma values are read from the ledger, not typed here.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from ailm.figures._common import (
    resolve_variables as _v,
)
from ailm.figures._common import (
    token_float,
)
from ailm.viz_config import FONTS, PALETTE, apply_style

__all__ = ["generate_generation_samples", "generation_samples_data"]

_SEED = 0
_N_WORDS = 24
_WRAP = 34


def generation_samples_data(
    variables: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Generate the three columns' text through the shipped generators.

    Raises:
        ValueError: If the qualitative signature the sweep records fails to
            appear in the regenerated samples — the degenerate column must
            have strictly fewer distinct words than the operating point, and
            the operating point at least as high a membership rate as the
            baseline. A panel whose text contradicted the recorded sweep
            would be showing a different model.
    """
    from ailm.generation import (
        build_generative_letter_agent,
        generate_by_policy_selection,
        generate_by_sampling,
    )
    from ailm.levels.letter import build_letter_transition, load_brown_counts
    from ailm.levels.word import build_word_prior, load_word_artifact

    v = _v(variables)
    best_gamma = token_float(v, "GENERATION_BEST_GAMMA")
    degenerate_gamma = token_float(v, "GENERATION_DEGENERATE_GAMMA")

    # float32, exactly as the recorder builds it: numpy's rng.choice scales its
    # sum-to-one tolerance with the dtype, and a float64 upcast of the same
    # values fails the stricter check.
    habit = np.asarray(build_letter_transition(load_brown_counts()))[0]
    artifact = load_word_artifact()
    prior = build_word_prior(artifact)

    baseline = generate_by_sampling(habit, artifact, n_words=_N_WORDS, seed=_SEED)
    shipped = generate_by_policy_selection(
        build_generative_letter_agent(habit, gamma=best_gamma),
        artifact,
        prior,
        habit,
        n_words=_N_WORDS,
        seed=_SEED,
    )
    degenerate = generate_by_policy_selection(
        build_generative_letter_agent(habit, gamma=degenerate_gamma),
        artifact,
        prior,
        habit,
        n_words=_N_WORDS,
        seed=_SEED,
    )

    if not degenerate.distinct_words < shipped.distinct_words:
        raise ValueError(
            f"degenerate gamma produced {degenerate.distinct_words} distinct "
            f"words vs the operating point's {shipped.distinct_words} — the "
            f"collapse the sweep records did not appear in these samples."
        )
    if not shipped.membership_rate >= baseline.membership_rate:
        raise ValueError(
            "the operating point fell below the goalless baseline on lexicon "
            "membership; the top-down channel is not doing in these samples "
            "what the sweep says it does."
        )

    return {
        "columns": [
            ("ancestral sampling\n(habit only)", baseline),
            (f"policy selection\nγ = {best_gamma:g} (shipped)", shipped),
            (f"policy selection\nγ = {degenerate_gamma:g} (degenerate)", degenerate),
        ]
    }


def generate_generation_samples(
    variables: dict[str, str] | None = None,
) -> Figure:
    """Three real output streams, word-colored by lexicon membership."""
    apply_style()
    data = generation_samples_data(variables)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4))
    in_color = PALETTE["MODEL"]
    out_color = PALETTE["ANTI"]

    for axis, (title, result) in zip(axes, data["columns"], strict=True):
        axis.set_axis_off()
        axis.set_title(title, fontsize=FONTS["subtitle"])
        x, y = 0.02, 0.90
        for word, member in zip(result.words, result.in_lexicon, strict=True):
            token = word + " "
            if x + 0.026 * len(token) > 0.98:
                x, y = 0.02, y - 0.13
            axis.text(
                x,
                y,
                word,
                transform=axis.transAxes,
                fontsize=FONTS["tick_label"],
                family="monospace",
                color=in_color if member else out_color,
            )
            x += 0.026 * len(token)
        rate = 100.0 * result.membership_rate
        axis.text(
            0.02,
            0.045,
            f"{rate:.0f}% in lexicon · {result.distinct_words} distinct",
            transform=axis.transAxes,
            fontsize=FONTS["footnote"],
            color=PALETTE["OUTLINE"],
        )

    fig.legend(
        handles=[
            Patch(facecolor=in_color, edgecolor="none", label="in lexicon"),
            Patch(facecolor=out_color, edgecolor="none", label="out of lexicon"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
        fontsize=FONTS["annotation"],
    )

    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig
