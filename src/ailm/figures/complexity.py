"""Figures for the computational-complexity section (``98_supplement_complexity.md``).

Two figures, both drawn from the claims ledger so they cannot drift from the
measurements — and both carrying their own falsifier inside the figure:

``complexity_ops``
    Three panels. Left: the per-position cost of each filtering level split
    into the terms the matrices contribute — the ``A`` row and its
    normalization, the ``B`` push, the decode argmax — recomputed here and
    required to sum to the recorded rate. Middle: the same recorded rates
    against the size of the state space, inside the analytic envelope
    ``[3S, 2S² + 2S + decode]`` every position must fall between. Right: the
    shape comparison the manuscript needs — what each system pays for context
    length, AILM flat and self-attention linear in ``L/d``.
``complexity_memory``
    Left: the memory ladder — every live agent's exact parameter bytes,
    spec-priced and byte-verified — with the closed form
    ``ailm.complexity.agent_memory_bytes_from_shape`` overlaid as an open
    marker. Right:
    the Stage 3 counterfactual — the factored transitions against the dense
    joint they replaced and the trie-node encoding both replace, with the pymdp
    instantiation budget and the vocabulary where it bites.

Every generator returns an unwritten ``Figure``; ``scripts/generate_figures.py``
discovers them by the ``generate_`` prefix and handles saving.

Why this module recomputes instead of only reading the ledger
-------------------------------------------------------------
Every closed form used here lives in ``ailm.complexity``, where it is tested as
arithmetic. What this module adds is the *second input*: it feeds those forms
the manuscript tokens alone — the state counts, factor sizes and observation
counts a reader can see in the prose — and raises rather than draws when the
result disagrees with the recorded measurement. The defect that guards against
is a figure that keeps illustrating a cost model the recorder has since stopped
measuring: an inventory change that moves ``S`` but leaves a stale bar in place
would otherwise render perfectly.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ailm.complexity import (
    FLOAT_BYTES,
    agent_memory_bytes_from_shape,
    dense_joint_transition_bytes,
    filtering_ops_theory,
    step_ops_terms,
    word_factored_transition_bytes,
)
from ailm.figures._common import (
    resolve_variables as _v,
)
from ailm.figures._common import (
    token_float,
    token_int,
)
from ailm.levels.word import MAX_AGENT_VOCAB_BYTES
from ailm.viz_config import FONTS, PALETTE, apply_style

__all__ = [
    "complexity_memory_data",
    "complexity_ops_data",
    "generate_complexity_memory",
    "generate_complexity_ops",
]

COLOR_MODEL = PALETTE["MODEL"]
COLOR_BASELINE = PALETTE["UNIGRAM"]
COLOR_COUNTERFACTUAL = PALETTE["ANTI"]
COLOR_ACCENT = PALETTE["ACCENT"]
COLOR_CEILING = PALETTE["FLOOR"]


def _format_bytes(value: float) -> str:
    """Human-readable bytes with the precision the ledger carries."""
    if value >= 1e9:
        return f"{value / 1e9:.2f} GB"
    if value >= 1e6:
        return f"{value / 1e6:.1f} MB"
    if value >= 1e3:
        return f"{value / 1e3:.1f} KB"
    return f"{value:.0f} B"


# ── OPERATIONS PER POSITION ──────────────────────────────────────────────────

#: The filtering levels drawn in the ops panels: (row label, ledger prefix,
#: decoding?, asymptotic form). The word level joins them separately because
#: its unit is a token, not a position, and its recursion collapsed.
#:
#: The state count is deliberately absent here. Each row's drawn note is built
#: from the same ``COMPLEXITY_<LEVEL>_STATES`` token the recomputation reads, so
#: an inventory change cannot leave the recorder and the closed form agreeing
#: with each other while the note beside the bar still says the old S.
_FILTER_ROWS: tuple[tuple[str, str, bool, str], ...] = (
    ("letters (L1)", "l1", False, "Θ(T·S²)"),
    ("phonemes (L2A)", "l2a", False, "Θ(T·S²)"),
    ("Letter-to-phone decode (L2B)", "l2b", True, "Θ(T·S²)"),
    ("cases (L4, decode)", "l4", True, "Θ(T·S²)"),
)

#: Rounding the ledger carries for an operations-per-position rate. Terms are
#: amortised over the workload, so the decomposition is compared at the same
#: precision the recorder stored rather than at float equality.
_RATE_DECIMALS = 1


def complexity_ops_data(variables: dict[str, str] | None = None) -> dict[str, Any]:
    """Recorded per-position costs, decomposed, beside the recomputed formula.

    For each filtering level two independent recomputations run here. The
    closed form is re-derived from the workload-shape tokens (states,
    sequences, positions) and compared against the recorded rate; and the same
    rate is split into the ``A``, ``B`` and decode terms of
    ``ailm.complexity.step_ops_terms``, amortised over the workload, and
    required to sum back to it.

    Raises:
        ValueError: If any recomputed rate disagrees with the recorded one, if
            a decomposition fails to sum to its own total, or if a level's
            recorded rate falls outside the analytic envelope
            ``[3S, 2S² + 2S + decode]`` that every scored position lies
            between. The recorded value came from an instrumented execution, so
            a disagreement means the formula or the workload accounting
            changed — a finding, not a redraw.
    """
    v = _v(variables)
    rows: list[dict[str, Any]] = []
    for label, level, decode, asymptotic in _FILTER_ROWS:
        states = token_int(v, f"COMPLEXITY_{level.upper()}_STATES")
        sequences = token_int(v, f"COMPLEXITY_{level.upper()}_SEQUENCES")
        positions = token_int(v, f"COMPLEXITY_{level.upper()}_POSITIONS")
        recorded = token_float(v, f"COMPLEXITY_{level.upper()}_OPS_PER_POSITION")

        # Re-derive the closed form from the workload shape alone. Every
        # sequence has at least one position; the per-sequence split of the
        # remainder does not change any category's total, so a two-length
        # reconstruction [1, ..., 1, remainder] carries the exact count.
        lengths = [1] * (sequences - 1) + [positions - (sequences - 1)]
        theory = filtering_ops_theory(states, lengths, decode=decode)
        predicted = round(theory.total / positions, _RATE_DECIMALS)
        if predicted != recorded:
            raise ValueError(
                f"{level}: closed form predicts {predicted} ops/position, the "
                f"ledger records {recorded}. The formula and the instrumented "
                f"count have diverged; re-run the recorder and find out which "
                f"is wrong."
            )

        # The push is withheld once per sequence, so its share of the workload
        # is (T - n)/T. Amortising here rather than assuming every position
        # pushes is what makes the stacked bar sum to the recorded rate; the
        # off-by-one that assumes T pushes overshoots every level.
        terms = step_ops_terms(states, decode=decode)
        push_share = (positions - sequences) / positions
        parts = {
            "emission": terms["emission"],
            "push": terms["push"] * push_share,
            "decode": float(terms["decode"]),
        }
        if round(sum(parts.values()), _RATE_DECIMALS) != recorded:
            raise ValueError(
                f"{level}: the A/B/decode decomposition sums to "
                f"{sum(parts.values()):.4f} ops/position against a recorded "
                f"{recorded}. The split drawn in the left panel would not be a "
                f"split of the bar it is drawn inside."
            )

        floor = float(terms["emission"])
        ceiling = float(terms["emission"] + terms["push"] + terms["decode"])
        if not floor <= recorded <= ceiling:
            raise ValueError(
                f"{level}: recorded {recorded} ops/position is outside the "
                f"analytic envelope [{floor}, {ceiling}] that every scored "
                f"position lies between at S = {states}. No workload shape can "
                f"produce that, so the cost model and the ledger disagree."
            )

        rows.append(
            {
                "label": label,
                "states": states,
                "recorded": recorded,
                "predicted": predicted,
                "parts": parts,
                "floor": floor,
                "ceiling": ceiling,
                "note": f"S = {states} · {asymptotic}",
            }
        )

    word_rate = token_float(v, "COMPLEXITY_L3_OPS_PER_TOKEN")
    if word_rate != 2.0:
        raise ValueError(
            f"the word level scores at {word_rate} ops/token, not the 2 the "
            f"Θ(1) collapse predicts. That is a change in the model, not in "
            f"the figure."
        )
    word_joint = token_int(v, "STAGE3_VOCAB_SIZE") * token_int(v, "COMPLEXITY_L3_POSITION_STATES")
    word_floor = 3 * word_joint
    if word_rate >= word_floor:
        raise ValueError(
            f"the word level costs {word_rate} ops/token against an "
            f"unfactored floor of {word_floor} at its {word_joint}-state "
            f"joint. The collapse is the whole point of the middle panel; if "
            f"it stops holding, the panel is drawing a claim that is false."
        )

    return {
        "rows": rows,
        "word_rate": word_rate,
        "word_joint": word_joint,
        "word_floor": word_floor,
    }


def _draw_ops_decomposition(ax: Any, rows: list[dict[str, Any]]) -> None:
    """Left panel: which matrix pays for which part of a scored position."""
    y = np.arange(len(rows))[::-1]
    segments = (
        ("emission", "A row + normalize · 3S", COLOR_BASELINE),
        ("push", "B push · 2S² − S, amortised", COLOR_MODEL),
        ("decode", "decode argmax · S − 1", COLOR_ACCENT),
    )
    left = np.zeros(len(rows))
    for key, label, color in segments:
        widths = np.array([row["parts"][key] for row in rows], dtype=np.float64)
        ax.barh(y, widths, left=left, color=color, height=0.6, label=label, zorder=2)
        left = left + widths
    for yi, row in zip(y, rows, strict=True):
        ax.text(
            row["recorded"] * 1.02,
            yi,
            f"{row['recorded']:g}   ·   S = {row['states']}",
            va="center",
            fontsize=FONTS["cell"],
        )
    ax.set_yticks(y)
    ax.set_yticklabels([row["label"] for row in rows])
    ax.set_xlim(0, max(row["recorded"] for row in rows) * 1.42)
    ax.set_xlabel("scalar operations per scored position")
    ax.set_title("where a position's operations go", fontsize=FONTS["subtitle"])
    ax.legend(loc="lower right", frameon=False, fontsize=FONTS["cell"])
    ax.spines[["top", "right"]].set_visible(False)


def _draw_ops_scaling(ax: Any, data: dict[str, Any]) -> None:
    """Middle panel: cost against the size of the enumerated state space."""
    rows = data["rows"]
    grid = np.geomspace(2.0, 1e5, 300)
    ax.fill_between(
        grid,
        3.0 * grid,
        2.0 * grid * grid + 2.0 * grid,
        color=COLOR_MODEL,
        alpha=0.16,
        zorder=1,
    )
    ax.plot(
        grid,
        2.0 * grid * grid + 2.0 * grid,
        color=COLOR_MODEL,
        lw=1.8,
        label="2S² + 2S · position with a push",
    )
    ax.plot(
        grid,
        3.0 * grid,
        color=COLOR_BASELINE,
        lw=1.8,
        ls="--",
        label="3S · position without one",
    )
    ax.scatter(
        [row["states"] for row in rows],
        [row["recorded"] for row in rows],
        s=64,
        facecolors="white",
        edgecolors=PALETTE["OUTLINE"],
        zorder=4,
        label="recorded rate",
    )
    ax.scatter(
        [data["word_joint"]],
        [data["word_rate"]],
        s=80,
        marker="D",
        color=COLOR_CEILING,
        edgecolors=PALETTE["OUTLINE"],
        zorder=4,
    )
    ax.annotate(
        "L3: Θ(1) per token,\ndeterminism removes\nthe recursion",
        (data["word_joint"], data["word_rate"]),
        textcoords="offset points",
        xytext=(-104, 16),
        fontsize=FONTS["cell"],
        color=COLOR_CEILING,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2, 2e5)
    ax.set_ylim(1, 1e10)
    # Fixed ticks, not the log locator's. Over a ten-decade span the locator
    # proposes a decade below the axis and matplotlib keeps an off-canvas Text
    # for it, which the layout sweep reads as a clipped label on a figure that
    # renders correctly.
    ax.set_xticks([1e1, 1e2, 1e3, 1e4, 1e5])
    ax.set_yticks([1e0, 1e2, 1e4, 1e6, 1e8, 1e10])
    ax.set_xlabel("hidden states S (joint, log scale)")
    ax.set_ylabel("operations per scored symbol (log scale)")
    ax.set_title("cost is set by the state space", fontsize=FONTS["subtitle"])
    ax.legend(loc="upper left", frameon=False, fontsize=FONTS["cell"])
    ax.spines[["top", "right"]].set_visible(False)


def _draw_context_scaling(ax: Any) -> None:
    """Right panel: what each mechanism pays as the context grows.

    Purely analytic, and labeled as such on the axis: the ordinate is each
    system's per-symbol cost divided by its own context-free term, so no
    constant from either system is needed and none is smuggled in.
    """
    ratio = np.geomspace(1e-2, 1e2, 300)
    ax.plot(
        ratio,
        1.0 + ratio,
        color=COLOR_COUNTERFACTUAL,
        lw=2.0,
        label="self-attention · 1 + L/d",
    )
    ax.plot(
        ratio,
        np.ones_like(ratio),
        color=COLOR_MODEL,
        lw=2.0,
        label="AILM · order 1, no L term",
    )
    ax.axvline(1.0, color=PALETTE["OUTLINE"], ls=":", lw=1.4)
    ax.text(1.15, 1.6, "L = d", fontsize=FONTS["cell"], color=PALETTE["OUTLINE"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-2, 1e2)
    ax.set_ylim(0.5, 2e2)
    ax.set_xlabel("context length in units of model width, L/d (log scale)")
    ax.set_ylabel("per-symbol cost ÷ its own context-free term")
    ax.set_title("what context length costs each mechanism", fontsize=FONTS["subtitle"])
    ax.legend(loc="upper left", frameon=False, fontsize=FONTS["cell"])
    ax.spines[["top", "right"]].set_visible(False)


def generate_complexity_ops(variables: dict[str, str] | None = None) -> Figure:
    """Per-position inference cost: decomposed, scaled, and set beside attention."""
    apply_style()
    data = complexity_ops_data(variables)

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.0))
    _draw_ops_decomposition(axes[0], data["rows"])
    _draw_ops_scaling(axes[1], data)
    _draw_context_scaling(axes[2])
    fig.tight_layout()
    return fig


# ── MEMORY ───────────────────────────────────────────────────────────────────

#: Memory-ladder rows: (row label, ledger prefix, observation-count token,
#: hidden-factor size tokens, action-count token or None).
#:
#: The shape tokens are what let the closed form be re-derived at draw time.
#: ``None`` for the action count means one action — the recognition levels do
#: not act, and pymdp still materialises the one-scalar policy prior.
_MEMORY_ROWS: tuple[tuple[str, str, str, tuple[str, ...], str | None], ...] = (
    ("cases (L4)", "L4", "STAGE4_UPOS_ALPHABET_SIZE", ("COMPLEXITY_L4_STATES",), None),
    ("letters (L1)", "L1", "STAGE0_LETTER_ALPHABET_SIZE", ("COMPLEXITY_L1_STATES",), None),
    ("L2B", "L2B", "STAGE0_LETTER_ALPHABET_SIZE", ("COMPLEXITY_L2B_STATES",), None),
    ("phonemes (L2A)", "L2A", "PHONEME_COUNT_39", ("COMPLEXITY_L2A_STATES",), None),
    (
        "generation (L1g)",
        "L1G",
        "STAGE0_LETTER_ALPHABET_SIZE",
        ("COMPLEXITY_L1_STATES",),
        "COMPLEXITY_L1_STATES",
    ),
    (
        "words (L3)",
        "L3",
        "STAGE0_WORD_ALPHABET_SIZE",
        ("STAGE3_VOCAB_SIZE", "COMPLEXITY_L3_POSITION_STATES"),
        None,
    ),
)


def complexity_memory_data(variables: dict[str, str] | None = None) -> dict[str, Any]:
    """The memory ladder plus the Stage 3 counterfactual curves.

    Raises:
        ValueError: If a level's recorded bytes stop equalling the closed form
            re-derived from its declared shape, if the recorded joint/factored
            ratio stops equalling the recomputed one, or if the trie's recorded
            dense size stops equalling its node count squared. All three are
            arithmetic, so a mismatch means the recorded configuration changed
            and the panel would be illustrating the previous one.
    """
    v = _v(variables)
    ladder: list[dict[str, Any]] = []
    for label, key, obs_token, factor_tokens, action_token in _MEMORY_ROWS:
        recorded = token_int(v, f"COMPLEXITY_{key}_MEMORY_BYTES")
        factor_sizes = [token_int(v, name) for name in factor_tokens]
        predicted = agent_memory_bytes_from_shape(
            token_int(v, obs_token),
            factor_sizes,
            n_actions=token_int(v, action_token) if action_token else 1,
        )
        if predicted != recorded:
            raise ValueError(
                f"{key}: the shape closed form prices the agent at {predicted} "
                f"bytes, the ledger records {recorded}. Memory is exact here — "
                f"one byte of disagreement is a finding, not a rounding."
            )
        ladder.append(
            {
                "label": label,
                "recorded": recorded,
                "predicted": predicted,
                "factor_sizes": factor_sizes,
            }
        )

    n_vocab = token_int(v, "STAGE3_VOCAB_SIZE")
    n_positions = token_int(v, "COMPLEXITY_L3_POSITION_STATES")
    ratio = token_float(v, "COMPLEXITY_L3_JOINT_OVER_FACTORED")
    recomputed = round(
        dense_joint_transition_bytes(n_vocab, n_positions)
        / word_factored_transition_bytes(n_vocab, n_positions),
        1,
    )
    if recomputed != ratio:
        raise ValueError(
            f"recorded joint/factored ratio {ratio} != recomputed {recomputed}; "
            f"the recorded Stage 3 configuration and this figure disagree."
        )

    # The trie encoding this level replaced, priced the same way: a dense pymdp
    # transition over trie nodes is nodes² float32 entries. Recomputing it from
    # the recorded node count catches the drift where the node count is
    # re-measured on a new vocabulary and the GB figure is left behind.
    trie_nodes = token_int(v, "STAGE3_TRIE_NODES_G2P")
    trie_bytes = trie_nodes * trie_nodes * FLOAT_BYTES
    recorded_trie_gb = token_float(v, "STAGE3_TRIE_DENSE_GB")
    if round(trie_bytes / 1e9, 1) != recorded_trie_gb:
        raise ValueError(
            f"{trie_nodes} trie nodes price a dense transition at "
            f"{trie_bytes / 1e9:.1f} GB, the ledger records {recorded_trie_gb} "
            f"GB. The node count and the size claim are out of step."
        )

    return {
        "ladder": ladder,
        "n_vocab": n_vocab,
        "n_positions": n_positions,
        "ratio": ratio,
        "full_vocab": token_int(v, "STAGE3_FULL_VOCAB_SIZE"),
        "max_agent_vocab": token_int(v, "STAGE3_MAX_AGENT_VOCAB"),
        "trie_nodes": trie_nodes,
        "trie_bytes": trie_bytes,
        "trie_gb": recorded_trie_gb,
    }


def generate_complexity_memory(variables: dict[str, str] | None = None) -> Figure:
    """Exact agent bytes per level, and the factored-vs-joint-vs-trie scaling."""
    apply_style()
    data = complexity_memory_data(variables)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.4, 5.4))

    # Left: the ladder. Log axis — the levels span five orders of magnitude.
    labels = [row["label"] for row in data["ladder"]]
    values = [row["recorded"] for row in data["ladder"]]
    y = np.arange(len(values))[::-1]
    left.barh(y, values, color=COLOR_MODEL, height=0.6, zorder=2)
    left.scatter(
        [row["predicted"] for row in data["ladder"]],
        y,
        facecolors="white",
        edgecolors=PALETTE["OUTLINE"],
        s=64,
        zorder=3,
        label="closed form from the declared shape",
    )
    left.set_xscale("log")
    left.set_xlim(100, max(values) * 40)
    left.set_yticks(y)
    left.set_yticklabels(labels)
    left.set_xlabel("live agent parameter bytes (log scale)")
    left.set_title("every level, priced by its spec", fontsize=FONTS["subtitle"])
    for yi, value in zip(y, values, strict=True):
        left.text(value * 1.5, yi, _format_bytes(value), va="center", fontsize=FONTS["cell"])
    left.legend(loc="upper right", frameon=False, fontsize=FONTS["cell"])
    left.spines[["top", "right"]].set_visible(False)

    # Right: transition memory against vocabulary size.
    n_positions = data["n_positions"]
    vocab_grid = np.geomspace(100, 2e5, 200)
    factored = np.array(
        [word_factored_transition_bytes(int(v_), n_positions) for v_ in vocab_grid],
        dtype=np.float64,
    )
    dense = np.array(
        [dense_joint_transition_bytes(int(v_), n_positions) for v_ in vocab_grid],
        dtype=np.float64,
    )
    right.plot(
        vocab_grid,
        dense,
        color=COLOR_COUNTERFACTUAL,
        lw=2.0,
        label="dense joint · Θ(V²P²)",
    )
    right.plot(
        vocab_grid,
        factored,
        color=COLOR_MODEL,
        lw=2.0,
        label="factored · Θ(V²)",
    )
    right.axhline(data["trie_bytes"], color=COLOR_BASELINE, lw=1.6, ls="-.")
    right.text(
        130,
        data["trie_bytes"] * 1.5,
        f"trie-node states: {data['trie_gb']:g} GB",
        fontsize=FONTS["cell"],
        color=COLOR_BASELINE,
    )
    right.axhline(MAX_AGENT_VOCAB_BYTES, color=PALETTE["OUTLINE"], linestyle="--", lw=1.6)
    right.text(
        130,
        MAX_AGENT_VOCAB_BYTES / 5.5,
        "pymdp instantiation budget",
        fontsize=FONTS["cell"],
        color=PALETTE["OUTLINE"],
    )
    shipped = data["n_vocab"]
    right.scatter(
        [shipped],
        [word_factored_transition_bytes(shipped, n_positions)],
        s=70,
        zorder=3,
        color=COLOR_MODEL,
        edgecolors=PALETTE["OUTLINE"],
    )
    right.annotate(
        f"shipped V = {shipped:,}",
        (shipped, word_factored_transition_bytes(shipped, n_positions)),
        textcoords="offset points",
        xytext=(8, -16),
        fontsize=FONTS["cell"],
    )
    cap_vocab = data["max_agent_vocab"]
    right.axvline(cap_vocab, color=COLOR_ACCENT, linestyle=":", lw=1.6)
    right.text(
        cap_vocab * 1.12,
        3e5,
        f"cap ≈ {cap_vocab:,} words",
        fontsize=FONTS["cell"],
        color=COLOR_ACCENT,
        rotation=90,
        va="bottom",
    )
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("vocabulary size V (log scale)")
    right.set_ylabel("transition memory, bytes (log scale)")
    right.set_title(f"the factorization buys P² = {data['ratio']:g}×", fontsize=FONTS["subtitle"])
    right.legend(loc="upper left", frameon=False, fontsize=FONTS["cell"])
    right.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig
