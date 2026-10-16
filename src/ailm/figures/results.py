"""Results figures: the perplexity ladder, Stage 1's information ceiling, and the
Stage 2 grapheme-to-phoneme comparison.

Every number these figures draw is read from ``output/data/manuscript_variables.json``
via :mod:`ailm.figures._common` — the same token map the manuscript renders from.
Nothing is hardcoded, so a figure cannot silently disagree with the prose beside it: if
the ledger changes the figure changes, and if a token is missing the generator raises
instead of guessing.

The three figures carry three separate arguments.

**fig:perplexity_ladder** — "beats uniform" and "beats unigram" are very different
claims. Uniform over the symbol inventory is a crippled comparator that a context-free
frequency table already clears by a wide margin; the honest bar is that frequency table.
The ladder puts both baselines, the model, and the empirical order-1 optimum on one axis
so the size of each margin is visible rather than asserted.

**fig:information_ceiling** — Stage 1's original gate demanded a gain of at least two
nats over uniform. The conditional entropy of the training distribution caps the gain
achievable *on that distribution* below the demand, so the target sits outside the band
the fitting corpus can supply — no amount of better smoothing supplies the shortfall from
that data. The bound is under p_train and not under the held-out evaluation sample, so
the panel is a statement about threshold calibration rather than an impossibility
theorem; the figure shades the achievable band and places the demand outside it.

**fig:g2p_accuracy** — the G2P margin over the context-free per-letter argmax bar is a
fraction of a point. Panel A shows the two decoders at full scale, where they are nearly
indistinguishable; panel B zooms in — labeled as a truncated axis — and carries the
paired McNemar statistic, which is what actually establishes the difference. Both
decoder bars carry a render-time Wilson interval on the paired eval-position
denominator (decision 101); the smoothing reference is drawn bare because it is a
reference, not an arm of the paired comparison. Filtering is what the online
architecture does; forward–backward smoothing is headroom it does not claim.

Ledger cross-checks
-------------------
Each data function re-derives an identity that must hold between separately recorded
claims: the ladder must be monotone, ``log(uniform perplexity)`` must equal the model's
cross-entropy plus its recorded gain, and the recorded G2P margin must equal model minus
baseline. A violation means two claims came from different runs, and the figure raises
rather than drawing a story the ledger no longer supports. ``tests/test_figures_results.py``
feeds each check a deliberately inconsistent token map, so none of them is decoration.

Determinism
-----------
No RNG, no timestamps, no filesystem iteration order: the figures are functions of the
token map alone. Regenerating twice produces byte-identical PNGs.

Layer discipline
----------------
``src/`` is standalone — no ``infrastructure`` import. Styling and saving come from
:mod:`ailm.viz_config` and :mod:`ailm.figures._common`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ailm.figures._common import (
    MissingVariableError,
    assert_legible,
    load_manuscript_variables,
    save_figure,
    token_float,
)
from ailm.gates import STAGE1_DECLARED_GATE_NATS
from ailm.levels.letter import N_SYMBOLS
from ailm.levels.phoneme import N_PHONES
from ailm.statistics import wilson_interval
from ailm.viz_config import FONTS, MIN_FONT_PT, PALETTE, apply_style

__all__ = [
    "DEMANDED_GAIN_NATS",
    "DECLARED_GATE_TOKEN",
    "SMOOTHING_CEILING_TOKEN",
    "build_g2p_accuracy_figure",
    "build_information_ceiling_figure",
    "build_perplexity_ladder_figure",
    "g2p_accuracy_data",
    "generate_g2p_accuracy",
    "generate_information_ceiling",
    "declared_gain_nats",
    "generate_perplexity_ladder",
    "information_ceiling_data",
    "load_variables",
    "parse_token",
    "perplexity_ladder_data",
    "role_color",
    "token",
]

#: The gain, in nats over uniform, that Stage 1's gate originally demanded. A
#: *specification* constant owned by :mod:`ailm.gates`, not a measurement — it is the
#: number the gate asked for, which is precisely what :func:`information_ceiling_data`
#: shows to be unreachable at this model order. The figures draw the *ledger token*
#: ``STAGE1_DECLARED_GATE_NATS`` and use this constant only to refuse when the ledger
#: and the code disagree.
DEMANDED_GAIN_NATS = STAGE1_DECLARED_GATE_NATS

#: Token under which ``scripts/record_stage1_claims.py`` records the declared gate.
DECLARED_GATE_TOKEN = "STAGE1_DECLARED_GATE_NATS"


def declared_gain_nats(variables: dict[str, str]) -> float:
    """Read Stage 1's declared gain target from the token map, checked against the code.

    Args:
        variables: Token map.

    Returns:
        The ledger's declared gain, in nats.

    Raises:
        MissingVariableError: If the token is absent.
        ValueError: If the ledger's declared gate no longer equals the constant the
            code owns — the figure would otherwise draw one target while the gate
            audit reasons about another.
    """
    declared = token(variables, DECLARED_GATE_TOKEN)
    if abs(declared - DEMANDED_GAIN_NATS) > 1e-9:
        raise ValueError(
            f"Ledger inconsistency: {DECLARED_GATE_TOKEN} = {declared:g} nats but "
            f"ailm.gates.STAGE1_DECLARED_GATE_NATS = {DEMANDED_GAIN_NATS:g}. The recorded "
            "declaration and the code-owned declaration have diverged."
        )
    return declared


#: Token for the forward–backward smoothing ceiling on G2P accuracy.
#: ``scripts/record_stage2_claims.py`` records it as ``stage2_g2p_smoothing_ceiling``,
#: so the ceiling is drawn. The absent-token branches are kept for a stage whose ceiling
#: has not been measured: a figure states the absence in words rather than substituting a
#: value, and ``tests/test_figures_results.py`` feeds a token map without it to keep that
#: path exercised.
SMOOTHING_CEILING_TOKEN = "STAGE2_G2P_SMOOTHING_CEILING"

#: Semantic color roles used by the results and methodology figures, mapped onto
#: :data:`ailm.viz_config.PALETTE` so every AILM figure shares one palette. The palette
#: is Okabe-Ito and therefore colorblind-safe; in addition, no distinction in these
#: figures is carried by hue alone — every band and bar is labeled in place, and the
#: gate-verdict zones differ by hatch as well as color.
_ROLES: dict[str, str] = {
    "crippled": "UNIFORM",  # uniform over the inventory — proves nothing
    "honest": "UNIGRAM",  # context-free unigram / per-letter argmax — the real bar
    "model": "MODEL",  # the shipped pymdp agent
    "ceiling": "FLOOR",  # empirical order-1 optimum / smoothing ceiling
    "threshold": "ACCENT",  # a demanded gate value
    "vacuous": "ANTI",  # a gate a no-op model already passes
    "unreachable": "MUTED",  # a region nothing of this order can enter
    "unaudited": "MUTED",  # no measured ceiling; reachability is unknown
    "ink": "INK",
}

# The type scale is shared so all AILM figures agree; the floor is asserted at import
# rather than trusted, because a shrunk scale would make every legibility check in this
# module quietly weaker.
_FS_CELL = FONTS["cell"]
_FS_ANNOT = FONTS["annotation"]
_FS_AXIS = FONTS["axis_label"]
_FS_TITLE = FONTS["title"]
_FS_SUPTITLE = FONTS["suptitle"]
if min(_FS_CELL, _FS_ANNOT, _FS_AXIS, _FS_TITLE, _FS_SUPTITLE) < MIN_FONT_PT:
    raise AssertionError(
        "ailm.viz_config.FONTS contains a size below MIN_FONT_PT; the figures in this "
        "module would be illegible by construction."
    )


def role_color(role: str) -> str:
    """Resolve a semantic color role to its palette hex value.

    Args:
        role: One of the keys of :data:`_ROLES`.

    Returns:
        A hex color string from :data:`ailm.viz_config.PALETTE`.

    Raises:
        KeyError: If ``role`` is not a known semantic role. Guessing a color for an
            unknown role would silently break the cross-figure encoding.
    """
    if role not in _ROLES:
        raise KeyError(f"Unknown color role {role!r}; known roles: {sorted(_ROLES)}")
    return PALETTE[_ROLES[role]]


# ── TOKEN ACCESS ─────────────────────────────────────────────────────────────


def parse_token(raw: object) -> float:
    """Parse a manuscript token string into a float.

    Tokens are rendered for prose, so integers above ten thousand carry thousands
    separators (``"42,820"``) and small p-values keep scientific notation
    (``"3.47e-06"``). Both must round-trip to a number for plotting.

    Args:
        raw: The token's string value.

    Returns:
        The numeric value.

    Raises:
        ValueError: If the token does not parse as a number.
    """
    return float(str(raw).replace(",", "").strip().rstrip("%"))


def load_variables(path: Path | None = None) -> dict[str, str]:
    """Load the manuscript token map.

    Thin pass-through to :func:`ailm.figures._common.load_manuscript_variables` so the
    figure modules and the manuscript renderer read one file through one code path.

    Args:
        path: Explicit path to ``manuscript_variables.json``; the project default when
            omitted.

    Returns:
        The flat token → string-value mapping.

    Raises:
        MissingVariableError: If the token map has not been generated.
    """
    return load_manuscript_variables(path)


def token(variables: dict[str, str], name: str) -> float:
    """Read one token as a float.

    Args:
        variables: Token map from :func:`load_variables`.
        name: UPPERCASE token name.

    Returns:
        The numeric value.

    Raises:
        MissingVariableError: If the token is absent. A figure must never substitute a
            plausible default for a measurement nobody took.
    """
    return token_float(variables, name)


def _display(variables: dict[str, str], name: str) -> str:
    """The token's own rendered string, used verbatim as the figure's value label.

    Labeling with the token string rather than a re-formatted float is what makes a
    figure label and the manuscript's prose the *same characters*, not merely the same
    quantity rounded twice in two places.

    Raises:
        MissingVariableError: If the token is absent.
    """
    if name not in variables:
        raise MissingVariableError(
            f"Token {name!r} is not in the manuscript variable map, so no label can be "
            "drawn for it. Record the claim from the run that measured it."
        )
    return str(variables[name])


# ── FIGURE 1: PERPLEXITY LADDER ──────────────────────────────────────────────

#: Ladder rungs: ``(row key, x-axis label, color role)``, in plot order — worst
#: comparator first. The order is asserted to be monotone in
#: :func:`perplexity_ladder_data`.
_LADDER_RUNGS: tuple[tuple[str, str, str], ...] = (
    ("uniform", "uniform\nover inventory", "crippled"),
    ("unigram", "context-free\nunigram", "honest"),
    ("model", "pymdp agent\n(bigram transition)", "model"),
    ("floor", "empirical order-1\noptimum", "ceiling"),
)

#: Horizontal extent of the ladder panels, in bar-index units. The floor rule
#: stops at ``_LADDER_RULE_RIGHT`` and the lane between there and
#: ``_LADDER_LANE_RIGHT`` holds the rule's label, so the label is never crossed
#: by the rule it names nor drawn over a bar. Layout only; no measurement.
_LADDER_RULE_LEFT: float = -0.6
_LADDER_RULE_RIGHT: float = 3.55
_LADDER_LANE_RIGHT: float = 4.85

_LADDER_PANELS: tuple[dict[str, Any], ...] = (
    {
        "stage": "Stage 1 — letters",
        "inventory": f"{N_SYMBOLS} symbols",
        # The population every rung on this panel was scored over. A ladder
        # without it invites the two panels to be compared rung for rung, and
        # they are scored over populations three orders of magnitude apart.
        "population_token": "STAGE1_HELDOUT_POSITIONS",
        "population_unit": "held-out letter positions",
        "tokens": {
            "uniform": "STAGE1_UNIFORM_BASELINE",
            "unigram": "STAGE1_UNIGRAM_BASELINE",
            "model": "STAGE1_LETTER_PERPLEXITY",
            "floor": "STAGE1_BIGRAM_FLOOR_PERPLEXITY",
        },
    },
    {
        "stage": "Stage 2 — phonemes",
        "inventory": f"{N_PHONES} canonical ARPAbet phones",
        "population_token": "STAGE2_PHONEME_EVAL_POSITIONS",
        "population_unit": "scored phoneme positions",
        "tokens": {
            "uniform": "STAGE2_PHONEME_UNIFORM_BASELINE",
            "unigram": "STAGE2_PHONEME_UNIGRAM_BASELINE",
            "model": "STAGE2_PHONEME_PERPLEXITY",
            "floor": "STAGE2_PHONEME_ALL_POSITION_FLOOR_PERPLEXITY",
        },
    },
)


def perplexity_ladder_data(variables: dict[str, str] | None = None) -> dict[str, Any]:
    """Assemble the four-rung perplexity ladder for both stages.

    Enforces the ordering the ladder asserts visually — floor ≤ model ≤ unigram ≤
    uniform. A ledger violating it would mean either that the model has beaten the
    empirical optimum of its own order (impossible) or that a comparator is mislabeled,
    and in both cases the figure would be drawing a false story.

    Args:
        variables: Token map; loaded from disk when omitted.

    Returns:
        ``{"panels": [{"stage", "inventory", "values", "displays",
        "margin_vs_uniform", "margin_vs_unigram", "above_floor",
        "unigram_gain_from_uniform"}, ...]}``

    Raises:
        MissingVariableError: If a required token is absent.
        ValueError: If the ledger values are not monotone across the ladder.
    """
    v = variables if variables is not None else load_variables()
    panels: list[dict[str, Any]] = []

    for spec in _LADDER_PANELS:
        values = {key: token(v, name) for key, name in spec["tokens"].items()}
        displays = {key: _display(v, name) for key, name in spec["tokens"].items()}

        order = [values[key] for key, _label, _role in _LADDER_RUNGS]
        if not (order[3] <= order[2] <= order[1] <= order[0]):
            raise ValueError(
                f"{spec['stage']}: ladder is not monotone. Expected "
                f"floor <= model <= unigram <= uniform, got floor={order[3]}, "
                f"model={order[2]}, unigram={order[1]}, uniform={order[0]}. Either the "
                "model beat the empirical optimum of its own order (impossible) or a "
                "comparator token is mislabeled."
            )

        population = int(token(v, spec["population_token"]))
        if population <= 0:
            raise ValueError(
                f"{spec['stage']}: the ladder is drawn over {population} "
                f"{spec['population_unit']}, which is not a population."
            )
        panels.append(
            {
                "stage": spec["stage"],
                "inventory": spec["inventory"],
                "population": population,
                "population_unit": spec["population_unit"],
                "values": values,
                "displays": displays,
                "margin_vs_uniform": values["uniform"] - values["model"],
                "margin_vs_unigram": values["unigram"] - values["model"],
                "above_floor": values["model"] - values["floor"],
                "unigram_gain_from_uniform": values["uniform"] - values["unigram"],
            }
        )

    return {"panels": panels}


def build_perplexity_ladder_figure(
    variables: dict[str, str] | None = None,
) -> tuple[Figure, dict[str, Any]]:
    """Build fig:perplexity_ladder.

    Args:
        variables: Token map; loaded from disk when omitted.

    Returns:
        ``(figure, data)`` where ``data`` is :func:`perplexity_ladder_data`'s output.
    """
    apply_style()
    data = perplexity_ladder_data(variables)

    # Width: the floor rule's label lane widened the x range, and the four
    # category labels have to keep their pitch — at the old width
    # "(bigram transition)" and "empirical order-1" abutted and read as one
    # string. Room, not smaller type.
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.8), sharey=True)
    colors = [role_color(role) for _k, _l, role in _LADDER_RUNGS]
    positions = list(range(len(_LADDER_RUNGS)))
    labels = [label for _k, label, _r in _LADDER_RUNGS]
    top = max(panel["values"]["uniform"] for panel in data["panels"])

    for ax, panel in zip(axes, data["panels"], strict=True):
        heights = [panel["values"][key] for key, _l, _r in _LADDER_RUNGS]
        ax.bar(
            positions,
            heights,
            width=0.66,
            color=colors,
            edgecolor=role_color("ink"),
            linewidth=0.8,
            zorder=3,
        )
        for x, height, (key, _label, _role) in zip(positions, heights, _LADDER_RUNGS, strict=True):
            ax.text(
                x,
                height + top * 0.015,
                panel["displays"][key],
                ha="center",
                va="bottom",
                fontsize=_FS_ANNOT,
                fontweight="bold",
                color=role_color("ink"),
                zorder=4,
            )

        # The floor rule stops short of the right edge so its label can sit in the
        # lane beyond it. Drawn as a full-width axhline the rule ran through
        # whatever named it, and a label a rule strikes through is a defect no
        # text-versus-text overlap sweep can see; a label placed anywhere inside
        # the bar region crosses a bar instead, which is worse.
        ax.hlines(
            panel["values"]["floor"],
            _LADDER_RULE_LEFT,
            _LADDER_RULE_RIGHT,
            color=role_color("ceiling"),
            linestyle=(0, (5, 3)),
            linewidth=1.4,
            zorder=2,
        )
        ax.text(
            _LADDER_RULE_RIGHT + 0.08,
            panel["values"]["floor"],
            f"order-1 conditional-\nentropy floor\n{panel['displays']['floor']}",
            ha="left",
            va="center",
            fontsize=_FS_CELL,
            color=role_color("ceiling"),
            linespacing=1.4,
            zorder=4,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=_FS_CELL)
        ax.tick_params(axis="y", labelsize=_FS_CELL)
        ax.set_title(
            f"{panel['stage']} · {panel['inventory']}\n"
            f"{panel['population']:,} {panel['population_unit']}",
            fontsize=_FS_TITLE,
            pad=10,
        )
        ax.set_xlim(_LADDER_RULE_LEFT, _LADDER_LANE_RIGHT)
        ax.set_ylim(0, top * 1.30)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.grid(axis="x", visible=False)

        summary = (
            f"vs uniform    −{panel['margin_vs_uniform']:6.2f}\n"
            f"  a context-free model\n"
            f"  already gets  −{panel['unigram_gain_from_uniform']:6.2f}\n"
            f"vs unigram    −{panel['margin_vs_unigram']:6.2f}  ← the gate\n"
            f"above floor   +{panel['above_floor']:6.2f}"
        )
        ax.text(
            0.97,
            0.97,
            summary,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=_FS_CELL,
            family="monospace",
            color=role_color("ink"),
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": PALETTE["BACKGROUND"],
                "edgecolor": role_color("crippled"),
                "alpha": 0.96,
            },
            zorder=5,
        )

    axes[0].set_ylabel("perplexity (lower is better)", fontsize=_FS_AXIS)
    fig.suptitle(
        "Beating uniform is a weak claim; beating the unigram is the gate",
        fontsize=_FS_SUPTITLE,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Dashed green line: the order-1 conditional-entropy floor, the best any model of "
        "this order can reach. The agent sits just above it in both stages.",
        ha="center",
        va="bottom",
        fontsize=_FS_CELL,
        color=role_color("ink"),
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    assert_legible(fig)
    return fig, data


def generate_perplexity_ladder(
    output_dir: Path | None = None,
    variables: dict[str, str] | None = None,
) -> Path:
    """Write ``output/figures/perplexity_ladder.png``.

    Args:
        output_dir: Destination directory; the project's ``output/figures`` by default.
        variables: Token map; loaded from disk when omitted.

    Returns:
        The written PNG path.
    """
    fig, _data = build_perplexity_ladder_figure(variables)
    try:
        return save_figure(fig, "perplexity_ladder", output_dir)
    finally:
        plt.close(fig)


# ── FIGURE 2: STAGE 1 INFORMATION CEILING ────────────────────────────────────


def information_ceiling_data(variables: dict[str, str] | None = None) -> dict[str, Any]:
    """Assemble Stage 1's cross-entropy levels, in nats per letter.

    Cross-checks the ledger against itself twice. ``log(uniform perplexity)`` must equal
    the model's cross-entropy plus its recorded gain over uniform, and the recorded
    maximum bigram gain must equal the uniform-to-floor distance. Both are arithmetic
    identities, so a mismatch means two claims were recorded from different runs —
    exactly the drift a figure would otherwise hide behind a tidy layout.

    Args:
        variables: Token map; loaded from disk when omitted.

    Returns:
        ``{"uniform_nats", "model_nats", "floor_nats", "demanded_nats",
        "achievable_gain_nats", "realized_gain_nats", "shortfall_nats",
        "shortfall_ratio", "captured_pct", "demanded_gain_nats", "displays"}``

    Raises:
        MissingVariableError: If a required token is absent.
        ValueError: If the ledger's nat-scale claims are mutually inconsistent.
    """
    v = variables if variables is not None else load_variables()

    uniform_ppl = token(v, "STAGE1_UNIFORM_BASELINE")
    model_nats = token(v, "STAGE1_LETTER_NATS")
    floor_nats = token(v, "STAGE1_BIGRAM_FLOOR_NATS")
    realized_gain = token(v, "STAGE1_GAIN_OVER_UNIFORM_NATS")
    achievable_gain = token(v, "STAGE1_MAX_BIGRAM_GAIN_NATS")
    captured_pct = token(v, "STAGE1_BIGRAM_SIGNAL_CAPTURED_PCT")

    uniform_nats = math.log(uniform_ppl)

    if abs((model_nats + realized_gain) - uniform_nats) > 1e-3:
        raise ValueError(
            "Ledger inconsistency: log(stage1_uniform_baseline) = "
            f"{uniform_nats:.4f} nats but stage1_letter_nats + "
            f"stage1_gain_over_uniform_nats = {model_nats + realized_gain:.4f}. "
            "These claims were recorded from different runs."
        )
    if abs((uniform_nats - floor_nats) - achievable_gain) > 1e-3:
        raise ValueError(
            "Ledger inconsistency: uniform − floor = "
            f"{uniform_nats - floor_nats:.4f} nats but stage1_max_bigram_gain_nats = "
            f"{achievable_gain:.4f}."
        )

    demanded_gain = declared_gain_nats(v)
    demanded_nats = uniform_nats - demanded_gain

    return {
        "uniform_nats": uniform_nats,
        "model_nats": model_nats,
        "floor_nats": floor_nats,
        "demanded_nats": demanded_nats,
        "achievable_gain_nats": achievable_gain,
        "realized_gain_nats": realized_gain,
        # How much of the demand the training distribution cannot supply to any
        # order-1 model. Under p_train, not under the held-out sample.
        "shortfall_nats": floor_nats - demanded_nats,
        "shortfall_ratio": demanded_gain / achievable_gain,
        "captured_pct": captured_pct,
        "demanded_gain_nats": demanded_gain,
        "displays": {
            "model_nats": _display(v, "STAGE1_LETTER_NATS"),
            "floor_nats": _display(v, "STAGE1_BIGRAM_FLOOR_NATS"),
            "achievable_gain_nats": _display(v, "STAGE1_MAX_BIGRAM_GAIN_NATS"),
            "realized_gain_nats": _display(v, "STAGE1_GAIN_OVER_UNIFORM_NATS"),
            "captured_pct": _display(v, "STAGE1_BIGRAM_SIGNAL_CAPTURED_PCT"),
            "shortfall_ratio": _display(v, "STAGE1_CEILING_SHORTFALL_RATIO"),
            "uniform_ppl": _display(v, "STAGE1_UNIFORM_BASELINE"),
        },
    }


def build_information_ceiling_figure(
    variables: dict[str, str] | None = None,
) -> tuple[Figure, dict[str, Any]]:
    """Build fig:information_ceiling — Stage 1's demanded gate, below the floor.

    Args:
        variables: Token map; loaded from disk when omitted.

    Returns:
        ``(figure, data)``.
    """
    apply_style()
    data = information_ceiling_data(variables)

    fig, ax = plt.subplots(figsize=(11.0, 7.0))

    y_lo = data["demanded_nats"] - 0.28
    y_hi = data["uniform_nats"] + 0.30
    ax.set_xlim(0, 1)
    ax.set_ylim(y_lo, y_hi)

    # The band any order-1 letter model can occupy, and the region below it that none
    # can enter. Hatching carries the distinction as well as hue.
    ax.axhspan(
        data["floor_nats"],
        data["uniform_nats"],
        facecolor=role_color("ceiling"),
        alpha=0.16,
        zorder=1,
    )
    ax.axhspan(
        y_lo,
        data["floor_nats"],
        facecolor=role_color("unreachable"),
        alpha=0.18,
        hatch="///",
        edgecolor=role_color("unreachable"),
        linewidth=0.0,
        zorder=1,
    )

    levels = (
        (
            "uniform_nats",
            "crippled",
            (0, (6, 3)),
            2.0,
            f"uniform over {N_SYMBOLS} symbols — {data['uniform_nats']:.4f} nats",
            False,
        ),
        (
            "model_nats",
            "model",
            "solid",
            3.0,
            f"pymdp bigram agent — {data['displays']['model_nats']} nats",
            True,
        ),
        (
            "floor_nats",
            "ceiling",
            "solid",
            2.4,
            f"order-1 conditional-entropy floor — {data['displays']['floor_nats']} nats",
            False,
        ),
        (
            "demanded_nats",
            "threshold",
            (0, (7, 2, 1, 2)),
            2.6,
            f"gate as originally demanded: ≥{data['demanded_gain_nats']:.0f} nats below "
            f"uniform — {data['demanded_nats']:.4f} nats",
            True,
        ),
    )
    for key, role, style, width, label, bold in levels:
        ax.axhline(data[key], color=role_color(role), linestyle=style, linewidth=width, zorder=3)
        ax.text(
            0.985,
            data[key] + (y_hi - y_lo) * 0.008,
            label,
            ha="right",
            va="bottom",
            fontsize=_FS_ANNOT,
            fontweight="bold" if bold else "normal",
            color=role_color(role if role != "crippled" else "ink"),
            zorder=4,
        )

    # Band width and the size of the shortfall.
    ax.annotate(
        "",
        xy=(0.09, data["floor_nats"]),
        xytext=(0.09, data["uniform_nats"]),
        arrowprops={"arrowstyle": "<->", "color": role_color("ceiling"), "linewidth": 1.8},
        zorder=4,
    )
    ax.text(
        0.12,
        (data["floor_nats"] + data["uniform_nats"]) / 2,
        f"achievable band: {data['displays']['achievable_gain_nats']} nats wide\n"
        f"agent realizes {data['displays']['realized_gain_nats']} nats "
        f"({data['displays']['captured_pct']}% of it)",
        ha="left",
        va="center",
        fontsize=_FS_ANNOT,
        color=role_color("ink"),
        zorder=4,
    )

    ax.annotate(
        "",
        xy=(0.09, data["demanded_nats"]),
        xytext=(0.09, data["floor_nats"]),
        arrowprops={
            "arrowstyle": "<->",
            "color": role_color("threshold"),
            "linewidth": 1.8,
        },
        zorder=4,
    )
    ax.text(
        0.12,
        (data["demanded_nats"] + data["floor_nats"]) / 2,
        f"demanded but unreachable: {data['shortfall_nats']:.4f} nats below the floor\n"
        f"the demand is {data['displays']['shortfall_ratio']}× the whole band",
        ha="left",
        va="center",
        fontsize=_FS_ANNOT,
        fontweight="bold",
        color=role_color("threshold"),
        zorder=4,
    )
    ax.text(
        0.985,
        y_lo + (y_hi - y_lo) * 0.02,
        "hatched: beyond the training distribution's order-1 budget",
        ha="right",
        va="bottom",
        fontsize=_FS_CELL,
        color=role_color("ink"),
        zorder=4,
    )

    ax.set_ylabel("cross-entropy, nats per letter (lower is better)", fontsize=_FS_AXIS)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=_FS_CELL)
    ax.set_title(
        "Stage 1's original gate was placed below the empirical order-1 optimum",
        fontsize=_FS_SUPTITLE,
        fontweight="bold",
        pad=14,
    )
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    assert_legible(fig)
    return fig, data


def generate_information_ceiling(
    output_dir: Path | None = None,
    variables: dict[str, str] | None = None,
) -> Path:
    """Write ``output/figures/information_ceiling.png``.

    Args:
        output_dir: Destination directory; the project's ``output/figures`` by default.
        variables: Token map; loaded from disk when omitted.

    Returns:
        The written PNG path.
    """
    fig, _data = build_information_ceiling_figure(variables)
    try:
        return save_figure(fig, "information_ceiling", output_dir)
    finally:
        plt.close(fig)


# ── FIGURE 3: G2P ACCURACY ───────────────────────────────────────────────────


def _wilson_pct_bounds(accuracy_pct: float, positions: float, label: str) -> tuple[float, float]:
    """Render-time Wilson bounds, in percent, for one recorded accuracy.

    The recorder never records an interval, so the bounds are derived from the
    recorded accuracy times the recorded denominator — the decision-101 contract
    for a benchmark bar. Fails closed when the accuracy cannot come from an
    integer success count at that denominator: the recorded accuracies are
    rounded to three decimals of a percentage point, so the check tolerates
    exactly that half-unit and nothing more. An accuracy that only fits some
    *other* position count is precisely the drift this guard exists to catch.
    """
    if positions != int(positions) or positions <= 0:
        raise ValueError(
            f"Ledger inconsistency: {label}'s denominator {positions!r} is not a "
            "positive integer count of evaluation positions."
        )
    n = int(positions)
    scaled = accuracy_pct / 100.0 * n
    successes = round(scaled)
    if abs(scaled - successes) > 5e-6 * n + 1e-9:
        raise ValueError(
            f"Ledger inconsistency: {label} = {accuracy_pct} cannot come from an "
            f"integer success count at its recorded denominator {n}. The accuracy "
            "and the position count were not recorded from the same evaluation."
        )
    low, high = wilson_interval(successes, n)
    return low * 100.0, high * 100.0


def g2p_accuracy_data(variables: dict[str, str] | None = None) -> dict[str, Any]:
    """Assemble the Stage 2 G2P comparison.

    Cross-checks that the recorded margin equals model − baseline and that the recorded
    discordant total equals ``b + c``. Both are arithmetic identities between ledger
    claims; a mismatch means the accuracy pair and the McNemar pair came from different
    evaluations, which would make this figure's central annotation false.

    Args:
        variables: Token map; loaded from disk when omitted.

    Returns:
        ``{"baseline_pct", "model_pct", "margin_pts", "mcnemar_p", "mcnemar_b",
        "mcnemar_c", "discordant", "positions", "test_pairs", "phone_count",
        "chance_pct", "smoothing_ceiling_pct" (None when unclaimed),
        "baseline_wilson_low_pct", "baseline_wilson_high_pct",
        "model_wilson_low_pct", "model_wilson_high_pct", "displays"}``

    Raises:
        MissingVariableError: If a required token is absent.
        ValueError: If the ledger's Stage 2 claims are mutually inconsistent.
    """

    v = variables if variables is not None else load_variables()

    baseline = token(v, "STAGE2_G2P_BASELINE_ACCURACY")
    model = token(v, "STAGE2_G2P_ACCURACY")
    margin = token(v, "STAGE2_G2P_MARGIN_PTS")
    mcnemar_b = token(v, "STAGE2_G2P_MCNEMAR_B")
    mcnemar_c = token(v, "STAGE2_G2P_MCNEMAR_C")
    discordant = token(v, "STAGE2_MCNEMAR_TOTAL_DISCORDANT")
    phone_count = token(v, "PHONEME_COUNT_39")
    positions = token(v, "STAGE2_G2P_EVAL_POSITIONS")

    if abs((model - baseline) - margin) > 5e-3:
        raise ValueError(
            "Ledger inconsistency: stage2_g2p_accuracy − "
            f"stage2_g2p_baseline_accuracy = {model - baseline:.4f} pts but "
            f"stage2_g2p_margin_pts = {margin}. The two accuracies and the margin were "
            "not recorded from the same evaluation."
        )
    if abs((mcnemar_b + mcnemar_c) - discordant) > 0.5:
        raise ValueError(
            f"Ledger inconsistency: mcnemar b + c = {mcnemar_b + mcnemar_c:.0f} but "
            f"stage2_mcnemar_total_discordant = {discordant:.0f}."
        )

    smoothing = token(v, SMOOTHING_CEILING_TOKEN) if SMOOTHING_CEILING_TOKEN in v else None

    # The baseline's denominator carries no label of its own — and needs none:
    # the recorder scores both decoders position-for-position over the same
    # held-out population and pairs them strictly for the McNemar counts, so
    # ``stage2_g2p_eval_positions`` is the denominator of both accuracies. The
    # integer-success check in ``_wilson_pct_bounds`` refuses to draw if that
    # ever stops holding.
    baseline_wilson = _wilson_pct_bounds(baseline, positions, "stage2_g2p_baseline_accuracy")
    model_wilson = _wilson_pct_bounds(model, positions, "stage2_g2p_accuracy")

    return {
        "baseline_pct": baseline,
        "model_pct": model,
        "margin_pts": margin,
        "mcnemar_p": token(v, "STAGE2_G2P_MCNEMAR_P"),
        "mcnemar_b": mcnemar_b,
        "mcnemar_c": mcnemar_c,
        "discordant": discordant,
        "positions": positions,
        "test_pairs": token(v, "STAGE2_G2P_TEST_PAIRS"),
        "phone_count": phone_count,
        # A uniform guess over the phone inventory: the comparator that would have made
        # the margin look enormous, shown so it cannot be mistaken for the bar.
        "chance_pct": 100.0 / phone_count,
        "smoothing_ceiling_pct": smoothing,
        "baseline_wilson_low_pct": baseline_wilson[0],
        "baseline_wilson_high_pct": baseline_wilson[1],
        "model_wilson_low_pct": model_wilson[0],
        "model_wilson_high_pct": model_wilson[1],
        "displays": {
            "baseline_pct": _display(v, "STAGE2_G2P_BASELINE_ACCURACY"),
            "model_pct": _display(v, "STAGE2_G2P_ACCURACY"),
            "margin_pts": _display(v, "STAGE2_G2P_MARGIN_PTS"),
            "mcnemar_p": _display(v, "STAGE2_G2P_MCNEMAR_P"),
            "mcnemar_b": _display(v, "STAGE2_G2P_MCNEMAR_B"),
            "mcnemar_c": _display(v, "STAGE2_G2P_MCNEMAR_C"),
            "discordant": _display(v, "STAGE2_MCNEMAR_TOTAL_DISCORDANT"),
            "positions": _display(v, "STAGE2_G2P_EVAL_POSITIONS"),
            "test_pairs": _display(v, "STAGE2_G2P_TEST_PAIRS"),
            "phone_count": _display(v, "PHONEME_COUNT_39"),
            "smoothing_ceiling_pct": (
                _display(v, SMOOTHING_CEILING_TOKEN) if smoothing is not None else None
            ),
        },
    }


def _g2p_bars(data: dict[str, Any]) -> list[tuple[str, float, str, str]]:
    """Bar specification: ``(x label, value, color role, value display)``.

    The smoothing-ceiling bar appears only when the claim exists. An unmeasured ceiling
    is stated in words on the figure rather than drawn at a guessed height.
    """
    bars = [
        (
            "context-free\nper-letter argmax",
            data["baseline_pct"],
            "honest",
            data["displays"]["baseline_pct"],
        ),
        (
            "pymdp forward\nfiltering",
            data["model_pct"],
            "model",
            data["displays"]["model_pct"],
        ),
    ]
    if data["smoothing_ceiling_pct"] is not None:
        bars.append(
            (
                "forward–backward\nsmoothing reference",
                data["smoothing_ceiling_pct"],
                "ceiling",
                data["displays"]["smoothing_ceiling_pct"],
            )
        )
    return bars


def build_g2p_accuracy_figure(
    variables: dict[str, str] | None = None,
) -> tuple[Figure, dict[str, Any]]:
    """Build fig:g2p_accuracy — a thin margin, honestly scaled and paired-tested.

    Args:
        variables: Token map; loaded from disk when omitted.

    Returns:
        ``(figure, data)``.
    """
    apply_style()
    data = g2p_accuracy_data(variables)
    bars = _g2p_bars(data)
    positions = list(range(len(bars)))
    values = [value for _l, value, _r, _d in bars]
    colors = [role_color(role) for _l, _v, role, _d in bars]
    tick_labels = [label for label, _v, _r, _d in bars]
    # The two decoder bars carry the decision-101 render-time Wilson interval on the
    # paired eval-position denominator; the optional smoothing bar is a whole-word
    # reference the streaming architecture does not claim, so it draws bare — the
    # same rule the aligned-G2P panel applies to its uniform floor.
    interval_lows = [
        max(pct - low, 0.0)
        for pct, low in (
            (data["baseline_pct"], data["baseline_wilson_low_pct"]),
            (data["model_pct"], data["model_wilson_low_pct"]),
        )
    ]
    interval_highs = [
        high - pct
        for pct, high in (
            (data["baseline_pct"], data["baseline_wilson_high_pct"]),
            (data["model_pct"], data["model_wilson_high_pct"]),
        )
    ]
    error_kw = {"ecolor": role_color("ink"), "elinewidth": 1.2, "capsize": 3, "zorder": 4}

    fig, (ax_full, ax_zoom) = plt.subplots(
        1, 2, figsize=(13.0, 7.0), gridspec_kw={"width_ratios": [1.0, 1.3]}
    )

    # Panel A — full scale. At this scale the two decoders are nearly identical, which
    # is the honest first impression: the margin is real but small.
    ax_full.bar(
        positions[:2],
        values[:2],
        width=0.6,
        color=colors[:2],
        edgecolor=role_color("ink"),
        linewidth=0.8,
        zorder=3,
        yerr=[interval_lows, interval_highs],
        error_kw=error_kw,
    )
    if len(bars) > 2:
        ax_full.bar(
            positions[2:],
            values[2:],
            width=0.6,
            color=colors[2:],
            edgecolor=role_color("ink"),
            linewidth=0.8,
            zorder=3,
        )
    for x, (_label, value, _role, display) in zip(positions, bars, strict=True):
        ax_full.text(
            x,
            value + 1.6,
            f"{display}%",
            ha="center",
            va="bottom",
            fontsize=_FS_ANNOT,
            fontweight="bold",
            zorder=4,
        )
    ax_full.axhline(
        data["chance_pct"],
        color=role_color("crippled"),
        linestyle=":",
        linewidth=1.8,
        zorder=2,
    )
    # Lifted into the clear band above the bars and tinted to match the dotted rule.
    # Sitting just above the 2.6% line it was ~2 data units wide across a 0.6-unit
    # bar pitch, so most of the sentence was dark grey ink printed over the blue and
    # green bars. Nothing flags that: the glyphs are inside the canvas, above the
    # font floor, and overlapping a *patch* rather than another label. Shared color
    # ties the caption to its rule, the device this module already uses for the
    # full-set margin caption in fig:subsample_bias.
    ax_full.text(
        len(bars) - 0.5,
        88.0,
        f"uniform guess over {data['displays']['phone_count']} phones = "
        f"{data['chance_pct']:.1f}%\na crippled comparator, not the bar",
        ha="right",
        va="bottom",
        fontsize=_FS_CELL,
        color=role_color("crippled"),
        zorder=4,
    )
    ax_full.set_ylim(0, 100)
    ax_full.set_xlim(-0.6, len(bars) - 0.4)
    ax_full.set_xticks(positions)
    ax_full.set_xticklabels(tick_labels, fontsize=_FS_CELL)
    ax_full.tick_params(axis="y", labelsize=_FS_CELL)
    ax_full.set_ylabel("L2B position accuracy, %", fontsize=_FS_AXIS)
    ax_full.set_title("A · full scale", fontsize=_FS_TITLE, pad=10)
    ax_full.grid(axis="y", alpha=0.3, zorder=0)
    ax_full.grid(axis="x", visible=False)

    # Panel B — zoomed, explicitly labeled as truncated, with the paired statistic
    # that carries the claim. A truncated axis alone would overstate the effect; the
    # McNemar box is what makes a margin this small defensible.
    span = max(data["margin_pts"], 0.1)
    lo = min(values) - 3.0 * span
    hi = max(values) + 4.2 * span
    ax_zoom.bar(
        positions[:2],
        values[:2],
        width=0.6,
        color=colors[:2],
        edgecolor=role_color("ink"),
        linewidth=0.8,
        zorder=3,
        yerr=[interval_lows, interval_highs],
        error_kw=error_kw,
    )
    if len(bars) > 2:
        ax_zoom.bar(
            positions[2:],
            values[2:],
            width=0.6,
            color=colors[2:],
            edgecolor=role_color("ink"),
            linewidth=0.8,
            zorder=3,
        )
    for x, (_label, value, _role, display) in zip(positions, bars, strict=True):
        ax_zoom.text(
            x,
            value + (interval_highs[x] if x < 2 else 0.0) + span * 0.22,
            f"{display}%",
            ha="center",
            va="bottom",
            fontsize=_FS_ANNOT,
            fontweight="bold",
            # The margin bracket's leader rises from the top of each of the first
            # two bars, i.e. from exactly the x these labels are centerd on, so it
            # ran straight through the digits. Mask the rule behind the glyphs
            # rather than displace a label that marks a bar height.
            bbox={
                "boxstyle": "square,pad=0.12",
                "facecolor": PALETTE["BACKGROUND"],
                "edgecolor": "none",
            },
            zorder=5,
        )

    bracket_y = max(values) + span * 1.6
    for x, value in ((0, data["baseline_pct"]), (1, data["model_pct"])):
        ax_zoom.plot(
            [x, x],
            [value, bracket_y],
            color=role_color("ink"),
            linewidth=1.0,
            zorder=4,
        )
    ax_zoom.annotate(
        "",
        xy=(0, bracket_y),
        xytext=(1, bracket_y),
        arrowprops={"arrowstyle": "<->", "color": role_color("model"), "linewidth": 1.8},
        zorder=4,
    )
    ax_zoom.text(
        0.5,
        bracket_y + span * 0.18,
        f"margin +{data['displays']['margin_pts']} pts",
        ha="center",
        va="bottom",
        fontsize=_FS_ANNOT,
        fontweight="bold",
        color=role_color("model"),
        zorder=5,
    )

    stats = (
        f"paired McNemar, all {data['displays']['positions']} positions\n"
        f"({data['displays']['test_pairs']} held-out words)\n"
        f"filtering only right  b = {data['displays']['mcnemar_b']}\n"
        f"baseline only right   c = {data['displays']['mcnemar_c']}\n"
        f"discordant total        {data['displays']['discordant']}\n"
        f"p = {data['displays']['mcnemar_p']}"
    )
    # Outside the axes, in the reserved band below panel B. Anchored inside the axes
    # at (0.97, 0.04) this opaque six-line box covered the foot of the filtering and
    # smoothing bars: on a truncated axis every bar runs to the bottom spine, so any
    # bottom-anchored panel occludes data. Neither the canvas check nor the
    # text-versus-text overlap check sees a box drawn over a bar.
    fig.text(
        0.985,
        0.215,
        stats,
        ha="right",
        va="top",
        fontsize=_FS_CELL,
        family="monospace",
        color=role_color("ink"),
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": PALETTE["BACKGROUND"],
            "edgecolor": role_color("model"),
            "alpha": 0.96,
        },
        zorder=5,
    )
    if data["smoothing_ceiling_pct"] is None:
        ax_zoom.text(
            0.97,
            0.97,
            "forward–backward smoothing ceiling:\nnot a recorded claim, so not drawn",
            transform=ax_zoom.transAxes,
            ha="right",
            va="top",
            fontsize=_FS_CELL,
            color=role_color("ceiling"),
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": PALETTE["BACKGROUND"],
                "edgecolor": role_color("ceiling"),
                "linestyle": "--",
                "alpha": 0.96,
            },
            zorder=5,
        )

    ax_zoom.set_ylim(lo, hi)
    ax_zoom.set_ylabel("L2B position accuracy, %")
    ax_zoom.set_xlim(-0.6, len(bars) - 0.4)
    ax_zoom.set_xticks(positions)
    ax_zoom.set_xticklabels(tick_labels, fontsize=_FS_CELL)
    ax_zoom.tick_params(axis="y", labelsize=_FS_CELL)
    ax_zoom.set_title(
        f"B · same bars, y-axis truncated to {lo:.1f}–{hi:.1f}%",
        fontsize=_FS_TITLE,
        pad=10,
    )
    ax_zoom.grid(axis="y", alpha=0.3, zorder=0)
    ax_zoom.grid(axis="x", visible=False)

    fig.suptitle(
        "Grapheme-to-phoneme: a thin but paired-significant margin",
        fontsize=_FS_SUPTITLE,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Filtering is what the online architecture does — position t sees letters 1..t. "
        "Forward–backward smoothing sees the whole word and is headroom AILM does not "
        "claim.",
        ha="center",
        va="bottom",
        fontsize=_FS_CELL,
        color=role_color("ink"),
    )
    # Bottom band widened from 0.055 to hold the McNemar box clear of both axes.
    fig.tight_layout(rect=(0, 0.235, 1, 0.94))
    assert_legible(fig)
    return fig, data


def generate_g2p_accuracy(
    output_dir: Path | None = None,
    variables: dict[str, str] | None = None,
) -> Path:
    """Write ``output/figures/g2p_accuracy.png``.

    Args:
        output_dir: Destination directory; the project's ``output/figures`` by default.
        variables: Token map; loaded from disk when omitted.

    Returns:
        The written PNG path.
    """
    fig, _data = build_g2p_accuracy_figure(variables)
    try:
        return save_figure(fig, "g2p_accuracy", output_dir)
    finally:
        plt.close(fig)
