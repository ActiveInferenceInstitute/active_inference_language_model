"""Figures that SHOW parameters as readable conditional probabilities.

The interpretability claim — every parameter is a conditional probability a
reader can inspect — was demonstrated only at Level 1 (the letter transition
heatmap). These panels extend the demonstration to the newest levels:

``case_matrices``
    Level 4's entire generative model on one canvas: the emission
    ``P(UPOS | case)``, the transition ``P(case_t | case_{t-1})`` and the
    start prior, with real UPOS tags and case names on the axes and every
    probability legible in its cell. The whole model is 200 numbers; showing
    all of them IS the interpretability claim for this level.
``word_matrices``
    Level 3's factors: the deterministic emission path of two example
    vocabulary words, one short and one long (letter × position), the ``+1``
    position shift, and the learned word prior's head — the one learned
    quantity in the level.
``trie_narrowing``
    The induced trie: how many vocabulary words remain consistent with a
    prefix, per prefix length, measured over the real held-out token stream.
    This is Stage 3's most striking property (the posterior's support IS the
    trie node) and previously had no figure.

Every matrix is built by the same shipped builder call the live agent uses,
so a panel cannot show a parameter the model does not have.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ailm.viz_config import FONTS, PALETTE, SEQUENTIAL_CMAP, apply_style

__all__ = [
    "generate_case_matrices",
    "generate_trie_narrowing",
    "generate_word_matrices",
    "trie_narrowing_data",
]

#: Example words for the emission-path panels: chosen from the vocabulary at
#: generation time (most frequent of each length), never hard-coded spellings.
#: Two lengths rather than three — a short word and a long one carry the same
#: argument (the path is a deterministic spelling lookup) as three did, and
#: dropping the middle length gives each remaining heatmap more canvas.
_EXAMPLE_LENGTHS = (3, 7)


def _annotate(axis, matrix: np.ndarray, *, fmt: str = "{:.2f}", threshold: float = 0.5) -> None:
    """Write each cell's probability into the cell.

    Text color switches on the *relative luminance of the sampled colormap
    color* (ITU-R BT.709 coefficients), not on the raw value: viridis crosses
    mid-luminance near 0.55–0.6, so a raw-value cutoff at 0.5 puts dark text on
    cells that still render dark. Every caller draws with ``vmin=0, vmax=1``, so
    the value indexes the colormap directly.
    """
    cmap = plt.get_cmap(SEQUENTIAL_CMAP)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            r, g, b, _a = cmap(float(value))
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            axis.text(
                col,
                row,
                fmt.format(value),
                ha="center",
                va="center",
                fontsize=FONTS["cell"],
                color=PALETTE["OUTLINE"] if luminance > threshold else "white",
            )


def generate_case_matrices() -> Figure:
    """Level 4, all of it: A, B and D with their real labels."""
    from ailm.levels.case import (
        CASE_TO_IDX,
        build_case_emission,
        build_case_start,
        build_case_transition,
        load_case_artifact,
    )

    apply_style()
    artifact = load_case_artifact()
    emission = build_case_emission(artifact)  # (n_obs, 8)
    transition = build_case_transition(artifact)  # (8, 8)
    start = build_case_start(artifact)  # (8,)

    case_names = [role.name for role, _ in sorted(CASE_TO_IDX.items(), key=lambda kv: kv[1])]
    upos_names = [tag for tag, _ in sorted(artifact.upos_to_idx.items(), key=lambda kv: kv[1])]

    fig, (ax_a, ax_b, ax_d) = plt.subplots(
        1, 3, figsize=(13.2, 5.4), width_ratios=[1.15, 1.0, 0.45]
    )

    ax_a.imshow(emission, cmap=SEQUENTIAL_CMAP, vmin=0.0, vmax=1.0, aspect="auto")
    _annotate(ax_a, emission)
    ax_a.set_xticks(range(len(case_names)), case_names, rotation=45, ha="right")
    ax_a.set_yticks(range(len(upos_names)), upos_names)
    ax_a.set_title("A — P(UPOS tag | case role)", fontsize=FONTS["subtitle"])

    ax_b.imshow(transition, cmap=SEQUENTIAL_CMAP, vmin=0.0, vmax=1.0, aspect="auto")
    _annotate(ax_b, transition)
    ax_b.set_xticks(range(len(case_names)), case_names, rotation=45, ha="right")
    ax_b.set_yticks(range(len(case_names)), case_names)
    ax_b.set_title("B — P(case$_t$ | case$_{t-1}$)", fontsize=FONTS["subtitle"])

    ax_d.imshow(start[:, None], cmap=SEQUENTIAL_CMAP, vmin=0.0, vmax=1.0, aspect="auto")
    _annotate(ax_d, start[:, None])
    ax_d.set_xticks([0], ["P(case$_0$)"])
    ax_d.set_yticks(range(len(case_names)), case_names)
    ax_d.set_title("D — start", fontsize=FONTS["subtitle"])

    for axis in (ax_a, ax_b, ax_d):
        axis.tick_params(length=0)
    fig.tight_layout()
    return fig


def generate_word_matrices() -> Figure:
    """Level 3's factors: deterministic spelling paths, the shift, the prior."""
    from ailm.levels.word import (
        END_INDEX,
        build_emission,
        build_position_transition,
        build_word_prior,
        load_word_artifact,
    )

    apply_style()
    artifact = load_word_artifact()
    emission = build_emission(artifact)  # (27, V, P)
    shift = build_position_transition(artifact)  # (P, P)
    prior = build_word_prior(artifact)  # (V,)

    examples = []
    for length in _EXAMPLE_LENGTHS:
        candidates = [w for w in artifact.vocab if len(w) == length]
        if candidates:
            examples.append(max(candidates, key=lambda w: artifact.word_freq[w]))

    n_positions = artifact.n_positions
    letter_labels = [chr(97 + i) for i in range(26)] + ["END"]

    fig = plt.figure(figsize=(10.6, 8.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.34, wspace=0.32)

    for column, word in enumerate(examples):
        axis = fig.add_subplot(grid[0, column])
        w_idx = artifact.word_to_index[word]
        path = emission[:, w_idx, :]  # (27, P)
        axis.imshow(path, cmap=SEQUENTIAL_CMAP, vmin=0.0, vmax=1.0, aspect="auto")
        # Every 5th position only: sixteen single-digit labels at this panel
        # width run together ("101112131415").
        position_ticks = list(range(0, n_positions, 5))
        axis.set_xticks(position_ticks, [str(p) for p in position_ticks], fontsize=FONTS["cell"])
        axis.set_yticks(range(27), letter_labels, fontsize=FONTS["cell"])
        axis.set_xlabel("position")
        axis.set_ylabel("letter")
        axis.set_title(f"A[·, {word!r}, ·] — its spelling, then END", fontsize=FONTS["tick_label"])
        axis.tick_params(length=0)

    ax_shift = fig.add_subplot(grid[1, 0])
    ax_shift.imshow(shift, cmap=SEQUENTIAL_CMAP, vmin=0.0, vmax=1.0, aspect="auto")
    ax_shift.set_xlabel("position$_{t-1}$")
    ax_shift.set_ylabel("position$_t$")
    ax_shift.set_title("B$_{f1}$ — the +1 shift, absorbing", fontsize=FONTS["tick_label"])

    ax_prior = fig.add_subplot(grid[1, 1])
    top = 20
    order = np.argsort(prior)[::-1][:top]
    ax_prior.barh(
        np.arange(top)[::-1],
        prior[order],
        color=PALETTE["MODEL"],
        height=0.7,
    )
    ax_prior.set_yticks(
        np.arange(top)[::-1], [artifact.vocab[i] for i in order], fontsize=FONTS["cell"]
    )
    ax_prior.set_xlabel("P(word) — the level's one learned quantity")
    ax_prior.set_title(
        f"D$_{{f0}}$ — head of the {len(artifact.vocab):,}-word prior",
        fontsize=FONTS["tick_label"],
    )
    ax_prior.spines[["top", "right"]].set_visible(False)

    # END_INDEX is load-bearing for the top row's labels; check rather than trust,
    # and with an explicit raise so `python -O` cannot delete the check.
    if END_INDEX != 26:
        raise AssertionError(f"END_INDEX is {END_INDEX}, expected 26")
    return fig


def trie_narrowing_data() -> dict[str, Any]:
    """Support size per prefix length over the real held-out stream.

    Returns:
        ``lengths`` (prefix lengths, 0..max), ``mean_support`` /
        ``median_support`` over all held-out token prefixes of that length,
        ``n_prefixes`` per length, and one worked ``example`` token with its
        per-prefix supports.

    Raises:
        ValueError: If support ever fails to narrow monotonically for the
            worked example — the induced-trie property this figure exists to
            show, checked here rather than assumed.
    """
    from ailm.levels.word import load_word_artifact

    artifact = load_word_artifact()

    prefix_support: dict[str, int] = {}
    for word in artifact.vocab:
        for cut in range(len(word) + 1):
            prefix = word[:cut]
            prefix_support[prefix] = prefix_support.get(prefix, 0) + 1

    max_len = 8
    sums = np.zeros(max_len + 1, dtype=np.float64)
    counts = np.zeros(max_len + 1, dtype=np.int64)
    values: list[list[int]] = [[] for _ in range(max_len + 1)]
    for token in artifact.eval_tokens:
        for cut in range(min(len(token), max_len) + 1):
            support = prefix_support[token[:cut]]
            sums[cut] += support
            counts[cut] += 1
            values[cut].append(support)

    example = max(
        (w for w in artifact.vocab if len(w) >= 6),
        key=lambda w: artifact.word_freq[w],
    )
    example_supports = [prefix_support[example[:cut]] for cut in range(len(example) + 1)]
    if any(b > a for a, b in pairwise(example_supports)):
        raise ValueError(
            f"support GREW along {example!r}: {example_supports}. A longer "
            f"prefix admitting more words is not a trie; the induced-trie "
            f"correspondence is broken."
        )

    return {
        "lengths": list(range(max_len + 1)),
        "mean_support": (sums / np.maximum(counts, 1)).tolist(),
        "median_support": [float(np.median(v)) if v else 0.0 for v in values],
        "n_prefixes": counts.tolist(),
        "example": example,
        "example_supports": example_supports,
        "vocab_size": len(artifact.vocab),
        "n_tokens": len(artifact.eval_tokens),
    }


def generate_trie_narrowing() -> Figure:
    """The induced trie: posterior support against prefix length."""
    apply_style()
    data = trie_narrowing_data()

    fig, axis = plt.subplots(figsize=(9.6, 5.4))
    axis.set_title("Trie narrowing from letters to valid word paths")
    lengths = data["lengths"]
    axis.plot(
        lengths,
        data["mean_support"],
        marker="o",
        color=PALETTE["MODEL"],
        lw=2.0,
        label=f"mean over {data['n_tokens']:,} held-out tokens",
    )
    axis.plot(
        lengths,
        data["median_support"],
        marker="s",
        color=PALETTE["UNIGRAM"],
        lw=1.6,
        label="median",
    )
    example_x = list(range(len(data["example_supports"])))
    axis.plot(
        example_x,
        data["example_supports"],
        marker="d",
        color=PALETTE["ACCENT"],
        lw=1.4,
        linestyle="--",
        label=f"the word {data['example']!r}",
    )
    for cut, support in enumerate(data["example_supports"]):
        axis.annotate(
            f"{data['example'][:cut] or 'ε'}: {support}",
            (cut, support),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=FONTS["cell"],
            color=PALETTE["ACCENT"],
        )
    axis.set_yscale("log")
    axis.set_xlabel("prefix length (letters observed)")
    axis.set_ylabel("words still consistent with the prefix (log scale)")
    axis.axhline(1.0, color=PALETTE["OUTLINE"], lw=1.0, linestyle=":")
    axis.text(0.1, 1.15, "unique word", fontsize=FONTS["cell"], color=PALETTE["OUTLINE"])
    axis.legend(frameon=False, fontsize=FONTS["footnote"])
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
