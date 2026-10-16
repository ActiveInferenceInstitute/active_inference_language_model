"""Methodology figures: the gate taxonomy, and eval-subsample bias.

These two figures carry the project's methodological argument rather than a result.

**fig:gate_taxonomy** — a threshold is only meaningful relative to two other numbers
that are almost never measured: what a trivial no-context model already scores, and what
the best model of this order could possibly score. Place the threshold below the first
and the gate is VACUOUS — a no-op passes. Place it beyond the second and the verdict
depends on what the second measurement *is*: beyond a model-class bound the gate is
UNREACHABLE — nothing passes, however good the implementation — while beyond a
reference comparator it is only ABOVE_REFERENCE, because a reference can be
exceeded. Only between the two measurements does a
gate discriminate. The figure plots where AILM's two original thresholds fell, one of
each kind. ``scripts/audit_gates.py`` makes the same classification re-runnable, and
both it and :func:`_classify` here are thin wrappers over
:func:`ailm.gates.classify_gate`, so the figure and the audit cannot disagree.

**fig:subsample_bias** — a reproducible subsample is not a representative one. A
deterministic *prefix* of the alphabetically sorted G2P pair list reaches only a few of
the word initials the full set covers, and reports a margin several times the truth. How
few is a recorded number the panel reads, not a list of letters typed into the caption,
which a re-run at another truncation size would silently falsify. Shrinking the eval is
not the fix either: a representative stride of the same size is underpowered, and only
the full held-out set reaches significance. Evidence: ``docs/decisions.md``.

Determinism
-----------
No RNG, no timestamps. Every measured field of every comparison row is read from the
claims ledger through the token its row names, so each number the figure plots is the
number the manuscript prints. The module constant carries only labels and color roles.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from ailm.figures._common import assert_legible, save_figure
from ailm.figures.results import (
    SMOOTHING_CEILING_TOKEN,
    declared_gain_nats,
    load_variables,
    role_color,
    token,
)
from ailm.gates import (
    HISTORICAL_STAGE2_G2P_THRESHOLD_PCT,
    ComparatorType,
    Direction,
    classify_gate,
    gate_passes,
)
from ailm.statistics import (
    FAMILYWISE_ALPHA,
    SUBSAMPLE_LARGER_ARM,
    SUBSAMPLE_LARGER_STRIDE,
    SUBSAMPLE_MATCHED_ARM,
    SUBSAMPLE_MATCHED_STRIDE,
)
from ailm.viz_config import FONTS, MIN_FONT_PT, PALETTE, apply_style

__all__ = [
    "ALPHA_LEVEL",
    "SUBSAMPLE_COMPARISON",
    "build_gate_taxonomy_figure",
    "build_subsample_bias_figure",
    "gate_taxonomy_data",
    "generate_gate_taxonomy",
    "generate_subsample_bias",
    "subsample_bias_data",
]

#: Significance level drawn as the reference line in fig:subsample_bias.
#:
#: This MUST match the level the Stage 2 gate actually decides at
#: (``ALPHA`` in ``tests/test_stage2_gate.py``), and
#: ``test_figure_alpha_matches_the_gate_alpha`` enforces that.
#:
#: This is the declared family-wise level before Bonferroni allocation. The
#: executable family and corrected threshold live in :mod:`ailm.statistics`.
#:
#: Deliberately not the conventional 0.05: a figure drawn at a looser level than the
#: gate would tell a reader a subsample is adequate where the project's own gate
#: rejects it, and at 0.05 the representative ``stride[::3]`` reads as significant —
#: precisely the conclusion this panel exists to deny.
ALPHA_LEVEL = FAMILYWISE_ALPHA

#: Schematic marker positions for the taxonomy panel, in arbitrary "strictness" units on
#: [0, 1]. These carry **no** measurement and no units; they exist only to place the
#: three verdict zones and one example threshold inside each. The two rows below the
#: schematic are where the real numbers live.
_SCHEMATIC_BASELINE = 0.28
_SCHEMATIC_CEILING = 0.72
_SCHEMATIC_THRESHOLDS: tuple[tuple[float, str, str], ...] = (
    (0.13, "VACUOUS", "a no-op model\nalready clears it"),
    (0.50, "BRACKETED_BOUND", "baseline fails,\nbound clears"),
    (0.88, "UNREACHABLE", "beyond a model-class bound:\nnothing clears it"),
)

#: Verdict → (color role, hatch). Every zone is distinguished by hatch as well as hue,
#: so the figure survives greyscale printing and color-vision deficiency. UNAUDITED is
#: the exception: both grey verdicts share the "unreachable"/"unaudited" MUTED role, and
#: two grey hatches are near-identical at print size, so UNAUDITED is drawn unhatched on
#: light grey with a dashed border instead (see :func:`_verdict_patch_style`).
_VERDICT_STYLE: dict[str, tuple[str, str]] = {
    "VACUOUS": ("vacuous", "..."),
    "BRACKETED_BOUND": ("ceiling", ""),
    "BRACKETED_REFERENCE": ("ceiling", "OO"),
    "UNREACHABLE": ("unreachable", "///"),
    "ABOVE_REFERENCE": ("unreachable", "\\\\"),
    "UNAUDITED": ("unaudited", ""),
}


def _verdict_patch_style(verdict: str, *, alpha: float = 0.18) -> dict[str, Any]:
    """Style keywords for a verdict's filled patch (zone span or legend swatch).

    UNREACHABLE and UNAUDITED share the MUTED grey role, and their old grey-on-grey
    hatches (``///`` vs ``xx``) were indistinguishable at print size. UNAUDITED is
    therefore unhatched light grey with a dashed border, which no other verdict uses.

    Args:
        verdict: A :data:`_VERDICT_STYLE` key.
        alpha: Fill opacity for the hue-carrying verdicts (legend swatches use a
            stronger fill than the wide zone spans). UNAUDITED's light-grey fill is
            drawn opaque either way so its dashed border stays crisp.
    """
    role, hatch = _VERDICT_STYLE[verdict]
    if verdict == "UNAUDITED":
        return {
            "facecolor": PALETTE["PANEL"],
            "edgecolor": role_color(role),
            "linestyle": "--",
            "linewidth": 1.2,
        }
    return {
        "facecolor": role_color(role),
        "alpha": alpha,
        "hatch": hatch or None,
        "edgecolor": role_color(role),
        "linewidth": 0.0,
    }


#: Measured G2P eval-subsample comparison — the same decoder pair scored on four
#: different subsets of the held-out split.
#:
#: Provenance: every row is recorded in ``data/claims.json`` by
#: ``scripts/record_subsample_claims.py``, and :func:`subsample_bias_data` reads
#: all four measured fields out of the ledger through the ``*_token`` names below.
#: Only the row's metadata — its key, its drawn labels, its prose detail, and its
#: color role — lives here. The rows are re-runnable through
#: ``scripts/audit_gates.py --limit`` and are described in ``docs/decisions.md``.
#:
#: The labels name the *selection rule*, not the sample size: the prefix and the
#: matched stride are both truncated to ``record_subsample_claims.py``'s
#: ``--prefix-size``, which no ledger token records, so a size typed here would
#: not follow a re-run at a different one.
SUBSAMPLE_COMPARISON: tuple[dict[str, Any], ...] = (
    {
        "key": "full",
        "label": "full held-out set",
        "axis_label": "full held-out set",
        "detail": "every pair",
        "margin_token": "SUBSAMPLE_FULL_MARGIN_PTS",
        "mcnemar_p_token": "SUBSAMPLE_FULL_MCNEMAR_P",
        "positions_token": "SUBSAMPLE_FULL_POSITIONS",
        "initials_token": "SUBSAMPLE_FULL_INITIALS",
        "role": "model",
        "hatch": "",
    },
    {
        "key": "prefix",
        "label": "alphabetical prefix of the sorted list",
        "axis_label": "alphabetical prefix",
        "detail": "reproducible,\nnot representative",
        "margin_token": "SUBSAMPLE_PREFIX_MARGIN_PTS",
        "mcnemar_p_token": "SUBSAMPLE_PREFIX_MCNEMAR_P",
        "positions_token": "SUBSAMPLE_PREFIX_POSITIONS",
        "initials_token": "SUBSAMPLE_PREFIX_INITIALS",
        "role": "vacuous",
        "hatch": "\\\\\\",
    },
    {
        "key": SUBSAMPLE_MATCHED_ARM,
        "label": f"stride[::{SUBSAMPLE_MATCHED_STRIDE}], truncated to the prefix size",
        "axis_label": f"stride[::{SUBSAMPLE_MATCHED_STRIDE}], truncated",
        "detail": "representative,\nsame size",
        "margin_token": "SUBSAMPLE_STRIDE5_MARGIN_PTS",
        "mcnemar_p_token": "SUBSAMPLE_STRIDE5_MCNEMAR_P",
        "positions_token": "SUBSAMPLE_STRIDE5_POSITIONS",
        "initials_token": "SUBSAMPLE_STRIDE5_INITIALS",
        "role": "crippled",
        "hatch": "...",
    },
    {
        "key": SUBSAMPLE_LARGER_ARM,
        "label": f"stride[::{SUBSAMPLE_LARGER_STRIDE}]",
        "axis_label": f"stride[::{SUBSAMPLE_LARGER_STRIDE}]",
        "detail": "representative,\nlarger",
        "margin_token": "SUBSAMPLE_STRIDE3_MARGIN_PTS",
        "mcnemar_p_token": "SUBSAMPLE_STRIDE3_MCNEMAR_P",
        "positions_token": "SUBSAMPLE_STRIDE3_POSITIONS",
        "initials_token": "SUBSAMPLE_STRIDE3_INITIALS",
        "role": "honest",
        "hatch": "",
    },
)

_FS_CELL = FONTS["cell"]
_FS_ANNOT = FONTS["annotation"]
_FS_AXIS = FONTS["axis_label"]
_FS_TITLE = FONTS["title"]
_FS_SUPTITLE = FONTS["suptitle"]
if min(_FS_CELL, _FS_ANNOT, _FS_AXIS, _FS_TITLE, _FS_SUPTITLE) < MIN_FONT_PT:
    raise AssertionError("ailm.viz_config.FONTS contains a size below MIN_FONT_PT.")


# ── FIGURE 4: GATE TAXONOMY ──────────────────────────────────────────────────


#: The figure's two ceiling kinds mapped onto the package's comparator types.
#: ``"ceiling"`` marks a bound (Stage 1's empirical order-1 optimum on the scored
#: passage); ``"reference"`` marks an exceedable measured comparator (Stage 2B's
#: forward-backward smoothing).
_CEILING_KIND_TO_TYPE = {
    "ceiling": ComparatorType.MODEL_CLASS_BOUND,
    "reference": ComparatorType.REFERENCE_MODEL,
}


def _classify(
    baseline: float,
    ceiling: float | None,
    threshold: float,
    lower_is_better: bool,
    ceiling_kind: str | None = None,
) -> str:
    """Compatibility wrapper around :func:`ailm.gates.classify_gate`.

    Args:
        baseline: What a trivial no-context model scores.
        ceiling: The best achievable at this model order, or None if unmeasured.
        threshold: The value the gate demands.
        lower_is_better: True for perplexity-like metrics.
        ceiling_kind: ``"ceiling"`` or ``"reference"``; required when a ceiling
            is supplied, so the classifier knows whether an impossibility
            verdict is licensed.

    Returns:
        ``"VACUOUS"``, ``"UNREACHABLE"``, ``"ABOVE_REFERENCE"``, ``"UNAUDITED"``,
        ``"BRACKETED_BOUND"``, or ``"BRACKETED_REFERENCE"``.
    """
    direction: Direction = "lower_is_better" if lower_is_better else "higher_is_better"
    if ceiling is None:
        ceiling_type = None
    else:
        if ceiling_kind is None:
            raise ValueError("ceiling_kind is required when a ceiling is supplied")
        ceiling_type = _CEILING_KIND_TO_TYPE[ceiling_kind]
    return str(classify_gate(baseline, ceiling, threshold, direction, ceiling_type=ceiling_type))


def gate_taxonomy_data(variables: dict[str, str] | None = None) -> dict[str, Any]:
    """Assemble the three taxonomy rows: schematic, Stage 1, Stage 2B.

    Each measured row carries its trivial baseline, the shipped model, its achievable
    ceiling where one is recorded, and the threshold the gate originally demanded —
    together with the verdict that placement earns.

    Stage 2B's ceiling (forward–backward smoothing) is a recorded claim —
    ``stage2_g2p_smoothing_ceiling``, written by ``scripts/record_stage2_claims.py`` —
    so the row carries it and the third bar is drawn. The ``ceiling=None`` branch below
    survives for a stage whose ceiling has not been measured: the verdict does not
    depend on it, because VACUOUS follows from the threshold sitting *below the
    baseline*, which needs no ceiling at all.

    Args:
        variables: Token map; loaded from disk when omitted.

    Returns:
        ``{"schematic": {...}, "stage1": {...}, "stage2b": {...}}``

    Raises:
        MissingVariableError: If a required token is absent.
        ValueError: If a measured row's verdict is no longer the one the project
            reports, which would mean the ledger stopped supporting the paper's
            argument.
    """
    v = variables if variables is not None else load_variables()

    # The declared gain is the ledger's token, cross-checked against the code-owned
    # constant; a hand-typed value here would be exactly the drift the ledger exists
    # to make impossible.
    declared_gain = declared_gain_nats(v)
    stage1: dict[str, Any] = {
        "title": "Stage 1 — letter perplexity",
        "unit": "perplexity (lower is better; axis inverted so stricter is rightward)",
        "lower_is_better": True,
        "baseline": token(v, "STAGE1_UNIGRAM_BASELINE"),
        "baseline_name": "context-free unigram",
        "model": token(v, "STAGE1_LETTER_PERPLEXITY"),
        "ceiling": token(v, "STAGE1_BIGRAM_FLOOR_PERPLEXITY"),
        "ceiling_name": "empirical order-1 optimum",
        "ceiling_kind": "ceiling",
        # "at least the declared gain below uniform", expressed in perplexity:
        # uniform / e^gain.
        "original_threshold": token(v, "STAGE1_UNIFORM_BASELINE") / math.exp(declared_gain),
        "original_text": f"original gate: ≥{declared_gain:.0f} nats below uniform",
        "expected_verdict": "UNREACHABLE",
        "expected_outcome": "PASS",
        # What the row was measured over. A verdict about a threshold is a
        # verdict about a measurement, and a measurement with no population
        # behind it cannot be judged precise or imprecise — only asserted.
        "population": int(token(v, "STAGE1_HELDOUT_POSITIONS")),
        "population_unit": "held-out letter positions",
        "fitted_on": (f"{int(token(v, 'STAGE1_BROWN_BIGRAM_TOKENS')):,} Brown letter bigrams"),
        # Perplexity is not a paired per-item verdict, so no exact conditional
        # test attaches to this row. Saying so is better than leaving the reader
        # to assume one was run.
        "test": None,
    }
    stage2b: dict[str, Any] = {
        "title": "Stage 2B — position-aligned letter-to-phone accuracy",
        "unit": "accuracy, % (higher is better)",
        "lower_is_better": False,
        "baseline": token(v, "STAGE2_G2P_BASELINE_ACCURACY"),
        "baseline_name": "context-free per-letter argmax",
        "model": token(v, "STAGE2_G2P_ACCURACY"),
        "ceiling": (token(v, SMOOTHING_CEILING_TOKEN) if SMOOTHING_CEILING_TOKEN in v else None),
        "ceiling_name": "forward–backward smoothing",
        "ceiling_kind": "reference",
        "original_threshold": HISTORICAL_STAGE2_G2P_THRESHOLD_PCT,
        "original_text": (
            f"original gate: ≥{HISTORICAL_STAGE2_G2P_THRESHOLD_PCT:.0f}% on UD EWT,\n"
            "a treebank with no phonemic field"
        ),
        "expected_verdict": "VACUOUS",
        "expected_outcome": "PASS",
        "population": int(token(v, "STAGE2_G2P_EVAL_POSITIONS")),
        "population_unit": "letter positions in "
        f"{int(token(v, 'STAGE2_G2P_TEST_PAIRS')):,} held-out words",
        "fitted_on": f"{int(token(v, 'STAGE2_G2P_TRAIN_PAIRS')):,} training word pairs",
        # The model-versus-baseline verdict on this row IS a paired per-position
        # comparison, so the exact test and its p belong on the figure beside the
        # markers whose separation it judges.
        "test": {
            "name": "exact McNemar, model vs baseline",
            "p": token(v, "STAGE2_G2P_MCNEMAR_P"),
            "discordant": int(token(v, "STAT_STAGE2_DISCORDANT_N")),
        },
    }

    # The retained case-only clause ablation (decision 114). It belongs on this
    # figure rather than on fig:hierarchy: that table is one row per level, and
    # an ablation is a second gate on a level rather than a second level. Its
    # verdict lives in ``output/reports/gate_audit_ablation.json`` under
    # ``stage5_clause_case_only`` and the row here reproduces it from the same
    # tokens the audit reads.
    #
    # This row is the figure's first non-passing one, and it is the first whose
    # comparator is WORSE than its baseline: the type oracle scores 4.38 against
    # the type-free bar's 4.37 on a scale where lower is better. That is what
    # ABOVE_REFERENCE means here — the threshold sits beyond an exceedable
    # comparator — and it is why the view below is derived from every drawn mark
    # rather than from the baseline-to-ceiling span alone.
    stage5_ablation: dict[str, Any] = {
        "title": "Stage 5 ablation — retained case-only clause model",
        "unit": "per-emission perplexity (lower is better; axis inverted so stricter is rightward)",
        "lower_is_better": True,
        "baseline": token(v, "STAGE5_TYPEFREE_PERPLEXITY"),
        "baseline_name": "type-free case bigram (the bar)",
        "model": token(v, "STAGE5_MIXTURE_PERPLEXITY"),
        "ceiling": token(v, "STAGE5_TYPE_ORACLE_PERPLEXITY"),
        "ceiling_name": "type oracle (true type given)",
        "ceiling_kind": "reference",
        # Baseline-relative by construction, like every audited row here.
        "original_threshold": token(v, "STAGE5_TYPEFREE_PERPLEXITY"),
        "original_text": "gate: beat the type-free bar",
        "expected_verdict": "ABOVE_REFERENCE",
        "expected_outcome": "FAIL",
        "population": int(token(v, "STAGE5_TEST_POSITIONS")),
        "population_unit": "held-out emissions in "
        f"{int(token(v, 'STAGE5_TEST_DOCUMENTS')):,} documents",
        "fitted_on": f"{int(token(v, 'STAGE5_TRAIN_SENTENCES')):,} training sentences",
        "test": {
            "name": "exact McNemar on the classification contrast",
            "p": token(v, "STAGE5_MCNEMAR_P"),
            "discordant": int(token(v, "STAGE5_MCNEMAR_B") + token(v, "STAGE5_MCNEMAR_C")),
        },
    }

    for row in (stage1, stage2b, stage5_ablation):
        verdict = _classify(
            row["baseline"],
            row["ceiling"],
            row["original_threshold"],
            row["lower_is_better"],
            row["ceiling_kind"],
        )
        if verdict != row["expected_verdict"]:
            raise ValueError(
                f"{row['title']}: the ledger now classifies the original gate as "
                f"{verdict}, but the manuscript reports {row['expected_verdict']}. "
                "Reconcile the claim and the prose before rendering."
            )
        row["verdict"] = verdict

        # ── The discriminating band, as a number rather than a shaded region ──
        # The band between the trivial baseline and the achievable ceiling is the
        # only interval in which a threshold discriminates, so its width is the
        # figure's whole subject. Drawn from the two markers it spans, never
        # typed, and refused if the picture's premise stopped holding.
        if row["ceiling"] is None or not _comparator_beats_baseline(row):
            row["band_width"] = None
        else:
            row["band_width"] = abs(row["ceiling"] - row["baseline"])
            if row["band_width"] <= 0.0:
                raise ValueError(
                    f"{row['title']}: the baseline and the ceiling are the same "
                    "measurement, so there is no band in which a threshold could "
                    "discriminate. Refusing to draw a bracket of zero width."
                )
            low, high = sorted((row["baseline"], row["ceiling"]))
            inside = low <= row["model"] <= high
            # Only a PASSING row asserts that its model sits inside the band. A
            # non-passing one asserts the opposite by definition: the ablation
            # loses to its own bar, which is what FAIL means and why its marker
            # is outside. Requiring "inside" of every row would have made this
            # figure unable to draw the one row that fails.
            if row["expected_outcome"] == "PASS" and not inside:
                raise ValueError(
                    f"{row['title']}: the shipped model scores {row['model']}, which is "
                    f"outside the band [{low}, {high}] the figure draws it inside. The "
                    "model now beats its own achievable ceiling or loses to the trivial "
                    "baseline; either is a finding, not a redraw."
                )
        # The outcome half of the badge, checked against the gate arithmetic the
        # audit uses rather than against the word typed beside it.
        #
        # The bar is the BASELINE, not ``original_threshold``. Every audited row
        # here is baseline-relative — the audit records threshold == baseline on
        # all of them — while ``original_threshold`` is the retired declaration
        # this figure exists to classify. Stage 1's model does not clear its
        # original 2-nats-over-uniform target, and that is the finding, not a
        # failing outcome.
        passes = gate_passes(
            row["model"],
            row["baseline"],
            "lower_is_better" if row["lower_is_better"] else "higher_is_better",
        )
        if passes != (row["expected_outcome"] == "PASS"):
            raise ValueError(
                f"{row['title']}: the shipped model {row['model']} against its baseline "
                f"bar {row['baseline']} {'clears' if passes else 'does not clear'} it, "
                f"but the row reports {row['expected_outcome']}. Reconcile the claim and "
                "the prose before rendering."
            )
        row["outcome"] = row["expected_outcome"]
        if row["test"] is not None and row["test"]["discordant"] > row["population"]:
            raise ValueError(
                f"{row['title']}: {row['test']['discordant']} discordant pairs over "
                f"{row['population']} evaluated {row['population_unit']} is impossible; "
                "the paired test and the population claim came from different runs."
            )

    return {
        "schematic": {
            "baseline": _SCHEMATIC_BASELINE,
            "ceiling": _SCHEMATIC_CEILING,
            "thresholds": _SCHEMATIC_THRESHOLDS,
        },
        "stage1": stage1,
        "stage2b": stage2b,
        "stage5_ablation": stage5_ablation,
    }


def _comparator_beats_baseline(row: dict[str, Any]) -> bool:
    """Whether the row's typed comparator actually beats its trivial baseline.

    Usually it does, and the band between them is the region in which a
    threshold discriminates. The Stage 5 ablation is the row where it does not:
    its type oracle scores worse than the type-free bar, so the band is empty and
    every claim that depends on one — the shaded bracketed zone, its stated
    width, the assertion that the shipped model lies inside it — is inapplicable
    rather than merely narrow.
    """
    if row["ceiling"] is None:
        return False
    return gate_passes(
        row["ceiling"],
        row["baseline"],
        "lower_is_better" if row["lower_is_better"] else "higher_is_better",
    )


def _zone_spans(
    row: dict[str, Any], x_left: float, x_right: float
) -> list[tuple[float, float, str]]:
    """Verdict zones as ``(x_start, x_end, verdict)`` in data coordinates.

    ``x_left`` and ``x_right`` are the *visual* edges of the row: for a lower-is-better
    row the axis is inverted, so ``x_left`` holds the numerically larger value. An
    An unrecorded ceiling leaves an UNAUDITED zone running to the strict edge:
    reachability cannot be inferred from a missing measurement.

    Args:
        row: A measured row from :func:`gate_taxonomy_data`.
        x_left: Value at the row's left edge.
        x_right: Value at the row's right (stricter) edge.

    Returns:
        Zone spans in drawing order.
    """
    baseline, ceiling = row["baseline"], row["ceiling"]
    if ceiling is None:
        return [(x_left, baseline, "VACUOUS"), (baseline, x_right, "UNAUDITED")]
    if row["ceiling_kind"] == "ceiling":
        bracketed, beyond = "BRACKETED_BOUND", "UNREACHABLE"
    else:
        bracketed, beyond = "BRACKETED_REFERENCE", "ABOVE_REFERENCE"
    if not _comparator_beats_baseline(row):
        # The comparator loses to the trivial baseline, so no threshold can sit
        # between them and the bracketed band does not exist. The boundary is the
        # baseline: ``classify_gate`` returns VACUOUS on its loose side and the
        # beyond-verdict on its strict side, with nothing in between.
        #
        # Emitting the band anyway produced two overlapping spans — the drawing
        # showed a BRACKETED_REFERENCE region a threshold could never land in,
        # with both zone captions printed on top of each other in the overlap.
        return [(x_left, baseline, "VACUOUS"), (baseline, x_right, beyond)]
    return [
        (x_left, baseline, "VACUOUS"),
        (baseline, ceiling, bracketed),
        (ceiling, x_right, beyond),
    ]


#: Legend wording per verdict, in taxonomy order. The figure draws only the entries
#: whose zone appears on at least one row, so the UNAUDITED entry survives for the
#: ``ceiling=None`` branch and is absent when every ceiling is recorded.
_LEGEND_ROWS: tuple[tuple[str, str], ...] = (
    ("VACUOUS", "VACUOUS — the trivial baseline already passes"),
    ("BRACKETED_BOUND", "BRACKETED_BOUND — baseline fails, a bound clears"),
    ("BRACKETED_REFERENCE", "BRACKETED_REFERENCE — baseline fails, a reference clears"),
    ("UNREACHABLE", "UNREACHABLE — a model-class bound itself fails"),
    ("ABOVE_REFERENCE", "ABOVE_REFERENCE — a reference fails; it could be exceeded"),
    ("UNAUDITED", "UNAUDITED — the ceiling is unmeasured"),
)

#: The measured rows, in the order they are drawn beneath the schematic. Stage 5's
#: retained ablation is here rather than on fig:hierarchy because that table is
#: one row per level and an ablation is a second gate on a level.
_MEASURED_ROWS: tuple[str, ...] = ("stage1", "stage2b", "stage5_ablation")

#: Vertical offset, in data units, of the first tier of marker labels.
_MARKER_BASE_Y = 0.36

#: Extra offset per additional tier when labels would collide.
_MARKER_LEVEL_STEP = 0.42


def _stagger_marker_labels(ax: Axes, marker_texts: list[tuple[Text, str]]) -> dict[str, int]:
    """Push colliding marker labels onto additional tiers, and report tiers used.

    Marker positions are *measured* values, so how close two labels sit is a
    property of the data, not of the layout — two markers can land a few percent of
    the axis apart with labels wide enough to overlap. A hand-tuned offset would fix
    today's numbers and silently re-collide the next time a measurement moves, so
    the tier is assigned from the rendered geometry.

    Labels are processed left to right within each side. A label that overlaps any
    already-placed label on its side moves up a tier until it is clear.

    Args:
        ax: The axes owning the labels; drawn once to obtain a renderer.
        marker_texts: ``(text artist, "above" | "below")`` pairs.

    Returns:
        The highest tier index used on each side, keyed ``"above"`` / ``"below"``.
    """
    figure = ax.get_figure()
    if figure is None:
        raise ValueError("marker labels must belong to axes that are attached to a figure")
    # Drawing once caches a renderer on the figure; the extents below resolve it
    # from the figure rather than reaching into the canvas.
    figure.canvas.draw()

    highest = {"above": 0, "below": 0}
    for side in ("above", "below"):
        on_side = [text for text, text_side in marker_texts if text_side == side]
        # Left to right in display space, so an inverted axis is handled too.
        on_side.sort(key=lambda text: text.get_window_extent().x0)

        placed: list[tuple[int, Bbox]] = []
        for text in on_side:
            level = 0
            while True:
                offset = _MARKER_BASE_Y + level * _MARKER_LEVEL_STEP
                text.set_position((text.get_position()[0], offset if side == "above" else -offset))
                box = text.get_window_extent()
                clash = any(
                    other_level == level and box.x1 > other.x0 and other.x1 > box.x0
                    for other_level, other in placed
                )
                if not clash:
                    placed.append((level, box))
                    break
                level += 1
            highest[side] = max(highest[side], level)
    return highest


def _draw_number_line(
    ax: Axes,
    *,
    zones: list[tuple[float, float, str]],
    markers: list[tuple[float, str, str, str, bool]],
    xlim: tuple[float, float],
    title: str,
    xlabel: str,
    zone_notes: dict[str, str] | None = None,
) -> None:
    """Render one taxonomy row as an annotated 1-D number line.

    Args:
        ax: Target axes.
        zones: ``(x_start, x_end, verdict)`` spans, drawn in the verdict's color and
            hatch and labeled in place.
        markers: ``(x, color role, label, "above"|"below", bold)`` items.
        xlim: ``(left, right)`` — pass a descending pair to invert the axis.
        title: Row title.
        xlabel: Row axis label, including the direction convention.
        zone_notes: Extra caption lines keyed by verdict, printed under the zone's
            name. This is how the bracket states its own width: a shaded region
            tells a reader *that* a band exists, and only a number tells them how
            much room a threshold had inside it.
    """
    ax.set_xlim(*xlim)

    # Zones intersected with the view. A zone whose visible width is zero or
    # negative is not on the chart, so it must not be shaded and — more
    # importantly — must not be captioned: ``(x0 + x1) / 2`` of a reversed span
    # lands *inside the neighboring zone*, printing "UNREACHABLE" over the BRACKETED
    # band. The bottom legend names all three verdicts, so dropping a caption for
    # an absent zone loses no information; printing it asserts something false.
    view_lo, view_hi = sorted(xlim)
    visible_zones: list[tuple[float, float, str]] = []
    for x0, x1, verdict in zones:
        lo = max(min(x0, x1), view_lo)
        hi = min(max(x0, x1), view_hi)
        if hi - lo > 0:
            visible_zones.append((lo, hi, verdict))

    for x0, x1, verdict in visible_zones:
        ax.axvspan(x0, x1, zorder=1, **_verdict_patch_style(verdict))

    ax.axhline(0.0, color=role_color("ink"), linewidth=1.2, zorder=2)

    marker_texts: list[tuple[Text, str]] = []
    for x, role, label, side, bold in markers:
        ax.vlines(x, -0.26, 0.26, color=role_color(role), linewidth=2.8, zorder=5)
        marker_texts.append(
            (
                ax.text(
                    x,
                    _MARKER_BASE_Y if side == "above" else -_MARKER_BASE_Y,
                    label,
                    ha="center",
                    va="bottom" if side == "above" else "top",
                    fontsize=_FS_CELL,
                    fontweight="bold" if bold else "normal",
                    color=role_color(role),
                    zorder=6,
                ),
                side,
            )
        )

    levels = _stagger_marker_labels(ax, marker_texts)

    # Headroom is derived from the staggering, not assumed: close-together markers
    # need a second tier, and the verdict captions must clear whatever it reaches.
    # A zone note is a second caption line, so it buys its own band of headroom
    # rather than being pushed down into the topmost marker label.
    note_headroom = 0.40 if zone_notes else 0.0
    top = _MARKER_BASE_Y + (levels["above"] + 1) * _MARKER_LEVEL_STEP + 0.30 + note_headroom
    bottom = -(_MARKER_BASE_Y + (levels["below"] + 1) * _MARKER_LEVEL_STEP + 0.30)
    ax.set_ylim(bottom, max(top, 1.15))

    verdict_y = max(top - 0.16, 0.92)
    notes = zone_notes or {}
    for x0, x1, verdict in visible_zones:
        role, _hatch = _VERDICT_STYLE[verdict]
        ax.text(
            (x0 + x1) / 2,
            verdict_y,
            verdict,
            ha="center",
            va="center",
            fontsize=_FS_ANNOT,
            fontweight="bold",
            color=role_color(role),
            zorder=4,
        )
        note = notes.get(verdict)
        if note:
            # Under its own zone name, in the zone's own color, so the number and
            # the region it measures cannot be read apart.
            ax.text(
                (x0 + x1) / 2,
                verdict_y - 0.30,
                note,
                ha="center",
                va="center",
                fontsize=_FS_CELL,
                color=role_color(role),
                zorder=4,
            )

    ax.set_title(title, fontsize=_FS_TITLE, loc="left", pad=8)
    ax.set_xlabel(xlabel, fontsize=_FS_CELL)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=_FS_CELL)
    ax.grid(visible=False)
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)


def build_gate_taxonomy_figure(
    variables: dict[str, str] | None = None,
) -> tuple[Figure, dict[str, Any]]:
    """Build fig:gate_taxonomy — the methodological argument in one image.

    Args:
        variables: Token map; loaded from disk when omitted.

    Returns:
        ``(figure, data)``.
    """
    apply_style()
    data = gate_taxonomy_data(variables)

    # One schematic row plus every measured row. The height follows the row
    # count so adding the Stage 5 ablation did not compress the three that
    # were already here.
    fig, axes = plt.subplots(1 + len(_MEASURED_ROWS), 1, figsize=(13.0, 13.6))

    # Row 1 — the schematic. Unitless by construction.
    schematic = data["schematic"]
    schematic_zones = [
        (0.0, schematic["baseline"], "VACUOUS"),
        (schematic["baseline"], schematic["ceiling"], "BRACKETED_BOUND"),
        (schematic["ceiling"], 1.0, "UNREACHABLE"),
    ]
    drawn_verdicts = {verdict for _x0, _x1, verdict in schematic_zones}
    _draw_number_line(
        axes[0],
        zones=schematic_zones,
        markers=[
            (x, "threshold", f"threshold here → {verdict}\n{note}", "below", True)
            for x, verdict, note in schematic["thresholds"]
        ],
        xlim=(0.0, 1.0),
        title="The taxonomy — a threshold means nothing without these two measurements",
        xlabel="increasing strictness →   (schematic: no units, no measurement)",
    )
    axes[0].set_xticks([schematic["baseline"], schematic["ceiling"]])
    axes[0].set_xticklabels(
        ["trivial baseline\n(a no-context model)", "achievable ceiling\n(at this order)"],
        fontsize=_FS_CELL,
    )

    # Rows 2 and 3 — the two measured stages, in their own units.
    for ax, key in zip(axes[1:], _MEASURED_ROWS, strict=True):
        row = data[key]
        if row["lower_is_better"]:
            # Inverted axis: stricter (lower perplexity) is rightward, matching row 1.
            #
            # The extremes are taken over every drawn mark, not over the
            # baseline-to-ceiling span alone. The Stage 5 ablation's comparator is
            # WORSE than its baseline and its model is worse than both, so a pad
            # derived from baseline minus ceiling comes out negative there and
            # inverts the view, while a left edge derived from the baseline alone
            # leaves the model clipped outside it. Both failures draw a marker the
            # reader cannot see while its label prints anyway, because text is not
            # clipped to the axes. The values below reproduce the previous view
            # exactly for the two rows that were already here.
            pad = abs(row["baseline"] - row["ceiling"]) * 0.30
            x_left = max(row["baseline"], row["ceiling"], row["model"]) + pad
            x_right = min(row["original_threshold"], row["ceiling"], row["model"]) - pad
        else:
            pad = max(row["baseline"] - row["original_threshold"], 1.0) * 0.25
            x_left = row["original_threshold"] - pad
            # The strict edge is derived from the ceiling as well as the baseline so
            # the ceiling is always inside the view. Outside it, the vlines marker is
            # clipped while its label is not (text is not clipped to the axes), and
            # the UNREACHABLE span collapses to negative width so its caption lands
            # on the BRACKETED band. Both read as a mislabeled chart and neither is
            # visible to a canvas-containment check.
            strict_edge = row["baseline"]
            if row["ceiling"] is not None:
                strict_edge = max(strict_edge, row["ceiling"])
            x_right = strict_edge + 2.0 * pad

        # A row whose gate IS its baseline draws one marker there, not two on the
        # same coordinate. Stage 1 and Stage 2B each retired a threshold that
        # differs from their bar, so the two markers separate naturally; the
        # Stage 5 ablation's gate is "beat the type-free bar", which is the
        # baseline exactly, and two labels anchored on one x overprinted each
        # other however the staggerer tiered them.
        threshold_is_baseline = row["original_threshold"] == row["baseline"]
        if threshold_is_baseline:
            baseline_label = (
                f"baseline {row['baseline']:.4g}  —  {row['baseline_name']}\n"
                f"{row['original_text']}, so the bar is the baseline  →  {row['verdict']}"
            )
        else:
            baseline_label = f"baseline {row['baseline']:.4g}\n{row['baseline_name']}"
        markers = [
            (row["baseline"], "honest", baseline_label, "below", threshold_is_baseline),
            (row["model"], "model", f"shipped model {row['model']:.4g}", "above", True),
        ]
        if not threshold_is_baseline:
            markers.append(
                (
                    row["original_threshold"],
                    "threshold",
                    f"{row['original_text']}\n"
                    f"= {row['original_threshold']:.4g}  →  {row['verdict']}",
                    "below",
                    True,
                )
            )
        if row["ceiling"] is not None:
            markers.append(
                (
                    row["ceiling"],
                    "ceiling",
                    f"{row.get('ceiling_kind', 'ceiling')} {row['ceiling']:.4g}\n"
                    f"{row['ceiling_name']}",
                    "above",
                    False,
                )
            )

        row_zones = _zone_spans(row, x_left, x_right)
        drawn_verdicts.update(verdict for _x0, _x1, verdict in row_zones)

        # The bracket, stated. Which bracketed verdict this row draws depends on
        # whether its upper marker is a bound or a reference, so the note is keyed
        # off the same branch `_zone_spans` uses rather than a second guess.
        zone_notes: dict[str, str] = {}
        if row["band_width"] is not None:
            bracketed = (
                "BRACKETED_BOUND" if row["ceiling_kind"] == "ceiling" else "BRACKETED_REFERENCE"
            )
            zone_notes[bracketed] = (
                f"the only band a threshold discriminates in:\n"
                f"{row['baseline']:.4g} → {row['ceiling']:.4g}, "
                f"{row['band_width']:.4g} wide"
            )

        # The population line: what the row was measured over, what it was fitted
        # on, and the exact test where the row's verdict rests on one.
        evidence = (
            f"scored on {row['population']:,} {row['population_unit']}; "
            f"fitted on {row['fitted_on']}"
        )
        if row["test"] is None:
            evidence += "; no paired per-item test attaches to this measure"
        else:
            evidence += (
                f"; {row['test']['name']}: p = {row['test']['p']:.4g} "
                f"over {row['test']['discordant']:,} discordant pairs"
            )
        _draw_number_line(
            ax,
            zones=row_zones,
            markers=markers,
            xlim=(x_left, x_right),
            # "the original gate was" is right for a row whose threshold is a
            # retired declaration, and wrong for one whose gate is current. The
            # ablation's bar is its live baseline, and its outcome is the fact a
            # reader most needs from that row, so both halves are stated.
            title=(
                f"{row['title']}  —  the gate is {row['verdict']}, outcome {row['outcome']}"
                if threshold_is_baseline
                else f"{row['title']}  —  the original gate was {row['verdict']}"
            ),
            xlabel=f"{row['unit']}\n{evidence}",
            zone_notes=zone_notes,
        )
        if row["ceiling"] is None:
            ax.text(
                x_right,
                0.55,
                f"the {row['ceiling_name']} ceiling is not a\n"
                "recorded claim, so this edge is unpinned →",
                ha="right",
                va="center",
                fontsize=_FS_CELL,
                color=role_color("unaudited"),
                zorder=6,
            )

    # The legend names exactly the zones on the canvas. An entry for a zone no
    # row draws advertises a region the reader cannot find.
    legend_handles = []
    for verdict, text in _LEGEND_ROWS:
        if verdict not in drawn_verdicts:
            continue
        style = _verdict_patch_style(verdict, alpha=0.35)
        if verdict != "UNAUDITED":
            style["linewidth"] = 1.0  # swatches keep a visible edge at legend size
        legend_handles.append(Patch(label=text, **style))
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=_FS_CELL,
        frameon=True,
        bbox_to_anchor=(0.5, 0.004),
    )
    fig.suptitle(
        "AILM shipped one gate of each defect, in opposite directions",
        fontsize=_FS_SUPTITLE,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.955))
    assert_legible(fig)
    return fig, data


def generate_gate_taxonomy(
    output_dir: Path | None = None,
    variables: dict[str, str] | None = None,
) -> Path:
    """Write ``output/figures/gate_taxonomy.png``.

    Args:
        output_dir: Destination directory; the project's ``output/figures`` by default.
        variables: Token map; loaded from disk when omitted.

    Returns:
        The written PNG path.
    """
    fig, _data = build_gate_taxonomy_figure(variables)
    try:
        return save_figure(fig, "gate_taxonomy", output_dir)
    finally:
        plt.close(fig)


# ── FIGURE 5: SUBSAMPLE BIAS ─────────────────────────────────────────────────


def subsample_bias_data(variables: dict[str, str] | None = None) -> dict[str, Any]:
    """Assemble the four-way subsample comparison.

    Every measured field is read from the ledger token named by its row metadata, so
    the figure has no second hardcoded copy of the recorded numbers.

    Args:
        variables: Token map; loaded from disk when omitted.

    Returns:
        ``{"rows", "overstatement_ratio", "full_margin_pts", "alpha"}`` where each row is
        the constant's metadata with all four measured fields — margin, McNemar p,
        positions, and distinct word initials — read from the ledger token the row names.

    Raises:
        MissingVariableError: If a required token is absent.
    """
    v = variables if variables is not None else load_variables()

    rows: list[dict[str, Any]] = []
    for entry in SUBSAMPLE_COMPARISON:
        row = dict(entry)
        row["margin_pts"] = token(v, row.pop("margin_token"))
        row["mcnemar_p"] = token(v, row.pop("mcnemar_p_token"))
        row["positions"] = int(token(v, row.pop("positions_token")))
        row["initials"] = int(token(v, row.pop("initials_token")))
        rows.append(row)

    full = next(row for row in rows if row["key"] == "full")
    prefix = next(row for row in rows if row["key"] == "prefix")

    return {
        "rows": rows,
        "overstatement_ratio": prefix["margin_pts"] / full["margin_pts"],
        "full_margin_pts": full["margin_pts"],
        "alpha": ALPHA_LEVEL,
    }


def build_subsample_bias_figure(
    variables: dict[str, str] | None = None,
) -> tuple[Figure, dict[str, Any]]:
    """Build fig:subsample_bias — reproducible is not representative.

    Args:
        variables: Token map; loaded from disk when omitted.

    Returns:
        ``(figure, data)``.
    """
    apply_style()
    v = variables if variables is not None else load_variables()
    data = subsample_bias_data(v)
    rows = data["rows"]
    positions = list(range(len(rows)))
    colors = [role_color(row["role"]) for row in rows]
    # ``axis_label`` not ``label``: four prose forms on this axis overlap their
    # neighbors, and a one-line clash inside a multi-line tick block stays under the
    # layout suite's area tolerance because the non-clashing lines dilute it. The full
    # forms remain in ``SUBSAMPLE_COMPARISON`` and in the caption.
    tick_labels = [f"{row['axis_label']}\n{row['detail']}" for row in rows]

    fig, (ax_margin, ax_p) = plt.subplots(1, 2, figsize=(15.0, 7.2))

    # Panel A — the effect each subsample reports.
    margin_bars = ax_margin.bar(
        positions,
        [row["margin_pts"] for row in rows],
        width=0.62,
        color=colors,
        edgecolor=role_color("ink"),
        linewidth=0.9,
        zorder=3,
    )
    for bar, row in zip(margin_bars, rows, strict=True):
        if row["hatch"]:
            bar.set_hatch(row["hatch"])

    # The rule names itself in a legend rather than in a right-hand lane. The lane
    # cost this panel 1.25 x-units of width that panel B does not spend, and since
    # both panels are the same physical width and carry the same four category
    # labels, panel A drew them at 77% of panel B's pitch — enough that
    # "alphabetical prefix" and "stride[::5], truncated" abutted and read as one
    # string. Nothing caught it: the two blocks touch without overlapping, so the
    # text-versus-text sweep stays under tolerance. Both panels now share one x
    # range, so one set of labels cannot be legible in one panel and not the other.
    ax_margin.axhline(
        data["full_margin_pts"],
        color=role_color("model"),
        linestyle=(0, (6, 3)),
        linewidth=1.6,
        zorder=2,
        label=f"full-set margin +{data['full_margin_pts']:.3f} pts",
    )
    margin_top = max(row["margin_pts"] for row in rows) * 1.55
    ax_margin.legend(loc="upper right", frameon=False, fontsize=_FS_CELL)
    for x, row in zip(positions, rows, strict=True):
        ax_margin.text(
            x,
            row["margin_pts"] + 0.04,
            f"+{row['margin_pts']:.3f}",
            ha="center",
            va="bottom",
            fontsize=_FS_ANNOT,
            fontweight="bold",
            # Same masking as the panel-B p-labels: a bar within a hair of the
            # full-set rule necessarily puts its value label on the dashed reference,
            # and a line struck through digits is invisible to a text-vs-text
            # overlap detector.
            bbox={
                "boxstyle": "square,pad=0.12",
                "facecolor": PALETTE["BACKGROUND"],
                "edgecolor": "none",
            },
            zorder=4,
        )
        # Above the bar in ink, not inside it in white: this label is several times
        # wider than the bar, so white text would run off both edges onto the white
        # background and render a truncated position count — a wrong number, not
        # merely an ugly one, and nothing catches it (the glyphs are inside the
        # canvas, they clear the font floor, and white-on-white is invisible rather
        # than overlapping). Short two-line form keeps the block narrower than the
        # category pitch.
        ax_margin.annotate(
            f"n = {row['positions']:,}\n{row['initials']} initials",
            xy=(x, row["margin_pts"]),
            xytext=(0, 26),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=_FS_CELL,
            color=role_color("ink"),
            zorder=5,
            # White backing so the dashed full-set rule cannot strike through the
            # label where a bar sits close to the rule's height.
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
        )

    prefix_row = next(row for row in rows if row["key"] == "prefix")
    full_row = next(row for row in rows if row["key"] == "full")
    ax_margin.annotate(
        f"{data['overstatement_ratio']:.1f}× the full-set margin,\n"
        f"on only {prefix_row['initials']} distinct initials\n"
        f"(the full set spans {full_row['initials']})",
        xy=(1, prefix_row["margin_pts"]),
        # Offset well clear of the tallest bar, whose own value label reaches into
        # the text box otherwise. The arrow keeps the reference.
        xytext=(1.90, prefix_row["margin_pts"] * 0.97),
        fontsize=_FS_CELL,
        fontweight="bold",
        color=role_color("vacuous"),
        arrowprops={"arrowstyle": "->", "color": role_color("vacuous"), "linewidth": 1.6},
        zorder=6,
    )
    # Headroom for the value label plus the two-line n-label stacked above it. The
    # x range matches panel B's exactly, so both panels lay the same four category
    # labels out at the same pitch.
    ax_margin.set_ylim(0, margin_top)
    ax_margin.set_xlim(-0.6, len(rows) - 0.4)
    ax_margin.set_xticks(positions)
    ax_margin.set_xticklabels(tick_labels, fontsize=_FS_CELL)
    ax_margin.tick_params(axis="y", labelsize=_FS_CELL)
    ax_margin.set_ylabel("L2B margin over the context-free bar, points", fontsize=_FS_AXIS)
    ax_margin.set_title("A · the effect each subsample reports", fontsize=_FS_TITLE, pad=10)
    ax_margin.grid(axis="y", alpha=0.3, zorder=0)
    ax_margin.grid(axis="x", visible=False)

    # Panel B — the evidence each subsample supplies.
    p_bars = ax_p.bar(
        positions,
        [row["mcnemar_p"] for row in rows],
        width=0.62,
        color=colors,
        edgecolor=role_color("ink"),
        linewidth=0.9,
        zorder=3,
    )
    for bar, row in zip(p_bars, rows, strict=True):
        if row["hatch"]:
            bar.set_hatch(row["hatch"])

    ax_p.set_yscale("log")
    ax_p.axhline(
        data["alpha"],
        color=role_color("threshold"),
        linestyle=(0, (7, 2, 1, 2)),
        linewidth=1.8,
        zorder=4,
    )
    ax_p.text(
        -0.5,
        data["alpha"] * 1.6,
        f"α = {data['alpha']}",
        ha="left",
        va="bottom",
        fontsize=_FS_CELL,
        fontweight="bold",
        color=role_color("threshold"),
        zorder=5,
    )
    # Family-wise context for the α rule: the repository-wide Bonferroni-corrected
    # level, read from the ledger rather than typed here. The sentence's relational
    # claim — no verdict on this panel flips under the correction — is re-checked
    # before it is drawn, so it cannot rot when the ledger moves.
    alpha_bonferroni = token(v, "STAT_BONFERRONI_ALPHA")
    for row in rows:
        if (row["mcnemar_p"] < data["alpha"]) != (row["mcnemar_p"] < alpha_bonferroni):
            raise ValueError(
                f"the {row['key']!r} subsample's verdict flips between α = "
                f"{data['alpha']} and the family-corrected α_B = "
                f"{v['STAT_BONFERRONI_ALPHA']}; the caption's 'verdicts unchanged "
                "at either level' claim no longer holds."
            )
    ax_p.text(
        len(rows) / 2 - 0.5,
        6.0,
        f"family-corrected α_B = {v['STAT_BONFERRONI_ALPHA']}; verdicts unchanged at either level",
        ha="center",
        va="top",
        fontsize=_FS_CELL,
        color=role_color("threshold"),
        zorder=5,
    )
    for x, row in zip(positions, rows, strict=True):
        verdict = "significant" if row["mcnemar_p"] < data["alpha"] else "NOT significant"
        ax_p.text(
            x,
            row["mcnemar_p"] * 2.2,
            f"p = {row['mcnemar_p']:.3g}\n{verdict}",
            ha="center",
            va="bottom",
            fontsize=_FS_CELL,
            fontweight="bold",
            color=role_color("ink"),
            # A label whose p-value is within a small factor of α necessarily sits
            # on the reference line, and the dash-dot rule would strike through the
            # word "significant". The overlap detector compares text with text, so a
            # line drawn over a word is invisible to it; a background patch masks the
            # rule behind the glyphs instead of moving a label whose *height* carries
            # the p-value.
            bbox={
                "boxstyle": "square,pad=0.15",
                "facecolor": PALETTE["BACKGROUND"],
                "edgecolor": "none",
            },
            zorder=5,
        )
    ax_p.set_ylim(1e-9, 8.0)
    ax_p.set_xlim(-0.6, len(rows) - 0.4)
    ax_p.set_xticks(positions)
    ax_p.set_xticklabels(tick_labels, fontsize=_FS_CELL)
    ax_p.tick_params(axis="y", labelsize=_FS_CELL)
    ax_p.set_ylabel("paired McNemar p (log scale)", fontsize=_FS_AXIS)
    ax_p.set_title("B · the evidence each subsample supplies", fontsize=_FS_TITLE, pad=10)
    ax_p.grid(axis="y", alpha=0.3, zorder=0)
    ax_p.grid(axis="x", visible=False)

    fig.suptitle(
        "A reproducible subsample is not a representative one — and a representative "
        "one can still be underpowered",
        fontsize=_FS_TITLE + 2,
        fontweight="bold",
    )
    # The initial counts are read from the ledger rather than spelled as letters:
    # the prefix's span depends on the truncation size, so "a/b/c words" is a
    # sentence that goes stale the first time anyone re-runs at another size.
    prefix_row = next(row for row in rows if row["key"] == "prefix")
    full_row = next(row for row in rows if row["key"] == "full")
    fig.text(
        0.5,
        0.012,
        "Sorted alphabetically, a deterministic prefix reaches only "
        f"{prefix_row['initials']} of the full set's {full_row['initials']} word "
        "initials. The resolution was not to shrink the eval but to verify the pymdp "
        "decoder against the oracle and then score the full set.",
        ha="center",
        va="bottom",
        fontsize=_FS_CELL,
        color=role_color("ink"),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    assert_legible(fig)
    return fig, data


def generate_subsample_bias(
    output_dir: Path | None = None,
    variables: dict[str, str] | None = None,
) -> Path:
    """Write ``output/figures/subsample_bias.png``.

    Args:
        output_dir: Destination directory; the project's ``output/figures`` by default.
        variables: Token map; loaded from disk when omitted.

    Returns:
        The written PNG path.
    """
    fig, _data = build_subsample_bias_figure(variables)
    try:
        return save_figure(fig, "subsample_bias", output_dir)
    finally:
        plt.close(fig)
