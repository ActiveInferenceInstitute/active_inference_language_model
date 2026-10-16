"""Evidence-status of the L4/L5/L6 scores: what runs real pymdp inference.

``inference_evidence``
    One panel per audited upper-level score path, drawn so a reader can see at a
    glance - and a future edit cannot silently blur - the boundary between
    scores produced by live pymdp inference and scores produced by a NumPy or
    closed-form path that a separate test merely *compares* against pymdp:

    - **L4 case (numpy forward filter)**: the recorded Stage 4 agreement on the
      full test set, annotated with the recorded exhaustive live comparison:
      every held-out position (``STAGE4_PYMDP_POSITIONS``) is walked with live
      ``infer_states`` + ``update_empirical_prior`` beside the scorer's NumPy
      filter, with ``STAGE4_PYMDP_ARGMAX_DISAGREEMENTS`` argmax disagreements
      and a largest posterior deviation of
      ``STAGE4_PYMDP_POSTERIOR_MAX_ABS_DEV`` (decision 94). The headline score
      is still the NumPy path, and the annotation says so.
    - **L5 clause (closed-form mixture)**: the recorded type-free perplexity
      (the headline score is still closed-form), annotated with the recorded
      full-set equivalence evidence: over ``STAGE5_PYMDP_PREFIX_SENTENCES``
      held-out sentences, live ``infer_states`` + ``update_empirical_prior``
      agrees with the NumPy joint forward filter on the type-posterior argmax
      with ``STAGE5_PYMDP_TYPE_ARGMAX_DISAGREEMENTS`` disagreements and on the
      full joint posterior within ``STAGE5_PYMDP_POSTERIOR_MAX_ABS_DEV``
      (decisions 92 and 94). Posterior agreement is not score equivalence for
      the headline estimand, and the annotation says so.
    - **L6 narrative (observed-position conditional)**: the registered observed-
      position score, annotated as not-inference, beside the separately recorded
      descriptive live-filter score that *does* route through pymdp
      (``infer_states`` + ``update_empirical_prior`` through ``B`` and ``D``).
      The two sit at different recorded values on purpose; the figure refuses to
      draw if either value moves outside its recorded relation to the other.

Every number is read from the manuscript token map (``token_float``), which
raises :class:`MissingVariableError` - never ``ValueError`` with a plausible
default - when a token is absent. The evidence-kind labels are structural (they
name a code path, audited by source inspection in the equivalence test module),
not measured claims, so no new recorder label is required.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ailm.figures._common import (
    resolve_variables as _v,
)
from ailm.figures._common import (
    save_figure,
    token_float,
)
from ailm.viz_config import FONTS, PALETTE, apply_style

__all__ = [
    "EVIDENCE_KINDS",
    "build_inference_evidence_figure",
    "generate_inference_evidence",
    "inference_evidence_data",
]

COLOR_MODEL = PALETTE["MODEL"]
COLOR_BASELINE = PALETTE["UNIGRAM"]
COLOR_ACCENT = PALETTE["ACCENT"]
COLOR_MUTED = PALETTE["MUTED"]
COLOR_INK = PALETTE["INK"]

#: The audited evidence kinds. A kind the ledger cannot support must be removed
#: here in the same commit that changes its test - the data function
#: cross-checks the drawn claims against recorded values.
EVIDENCE_KINDS: tuple[str, ...] = (
    "live pymdp inference (recorded)",
    "numpy path; live pymdp agrees at every held-out position",
    "closed form (headline); live pymdp posterior agreement on the full held-out set",
    "conditional table, not inference (live-filter score separate)",
)


def inference_evidence_data(variables: dict[str, str] | None = None) -> dict[str, Any]:
    """Load and cross-check every token the evidence panel draws.

    Args:
        variables: Explicit token map; loaded from
            ``output/data/manuscript_variables.json`` when omitted.

    Returns:
        The recorded scores and the structural labels describing how each was
        produced.

    Raises:
        MissingVariableError: If any required token is absent from the map.
        ValueError: If the recorded relations between the scores change - each
            relation is a caption claim, and a violated one means the ledger
            changed and the panel must be re-authored, not redrawn.
    """
    v = _v(variables)
    data: dict[str, Any] = {
        # L4: the numpy forward filter's recorded full-set agreement, plus the
        # recorded exhaustive live-pymdp comparison over the same population
        # (decision 94).
        "l4_agreement_pct": token_float(v, "STAGE4_DERIVED_AGREEMENT"),
        "l4_test_tokens": token_float(v, "STAGE4_DERIVED_TEST_TOKENS"),
        "l4_pymdp_positions": token_float(v, "STAGE4_PYMDP_POSITIONS"),
        "l4_pymdp_argmax_disagreements": token_float(v, "STAGE4_PYMDP_ARGMAX_DISAGREEMENTS"),
        "l4_pymdp_posterior_max_abs_dev": token_float(v, "STAGE4_PYMDP_POSTERIOR_MAX_ABS_DEV"),
        # L5: the closed-form mixture's type-free bar, plus the recorded
        # full-set pymdp equivalence evidence (decisions 92 and 94).
        "l5_typefree_perplexity": token_float(v, "STAGE5_TYPEFREE_PERPLEXITY"),
        "l5_pymdp_prefix_sentences": token_float(v, "STAGE5_PYMDP_PREFIX_SENTENCES"),
        "l5_pymdp_positions": token_float(v, "STAGE5_PYMDP_POSITIONS"),
        "l5_pymdp_type_argmax_disagreements": token_float(
            v, "STAGE5_PYMDP_TYPE_ARGMAX_DISAGREEMENTS"
        ),
        "l5_pymdp_posterior_max_abs_dev": token_float(v, "STAGE5_PYMDP_POSTERIOR_MAX_ABS_DEV"),
        # L6: the registered observed-position score (not inference) and the
        # separately recorded descriptive live-filter score (real pymdp).
        "l6_unigram_logloss": token_float(v, "STAGE6_UNIGRAM_LOGLOSS"),
        "l6_phase_logloss": token_float(v, "STAGE6_PHASE_LOGLOSS"),
        "l6_live_filter_logloss": token_float(v, "STAGE6_LIVE_FILTER_LOGLOSS"),
        "l6_live_filter_improvement_nats": token_float(v, "STAGE6_LIVE_FILTER_IMPROVEMENT_NATS"),
        "l6_registered_improvement_nats": token_float(v, "STAGE6_LOGLOSS_IMPROVEMENT_NATS"),
        "l6_test_sentences": token_float(v, "STAGE6_TEST_SENTENCES"),
    }

    # The live-filter score must be a strict improvement over the unigram bar
    # by exactly its recorded delta: two claims, one run, or the panel lies.
    if not data["l6_live_filter_improvement_nats"] > 0.0:
        raise ValueError(
            "the recorded live-filter improvement is no longer positive; the "
            "'live pymdp inference' annotation no longer describes a score "
            "that beats its bar"
        )
    if not np.isclose(
        data["l6_unigram_logloss"] - data["l6_live_filter_logloss"],
        data["l6_live_filter_improvement_nats"],
        atol=1e-4,
    ):
        raise ValueError(
            "the live-filter score and its recorded improvement disagree - two "
            "claims from different runs cannot share one panel"
        )
    # The registered observed-position score must remain a *different*
    # estimand from the live filter (decision 68's discipline): if the two
    # recorded deltas ever coincide, the figure's separate-estimand annotation
    # is false and must be re-authored.
    if (
        abs(data["l6_registered_improvement_nats"] - data["l6_live_filter_improvement_nats"])
        <= 1e-6
    ):
        raise ValueError(
            "the registered observed-position delta and the live-filter delta "
            "coincide; the separate-estimand annotation is no longer true and "
            "the panel must be re-authored, not redrawn"
        )
    # The recorded L5 equivalence must remain an equivalence: a non-zero
    # disagreement count means the live agent stopped matching the filter and
    # the panel's annotation is false.
    if data["l5_pymdp_type_argmax_disagreements"] != 0.0:
        raise ValueError(
            "the recorded L5 pymdp type-argmax disagreement count is non-zero; "
            "the full-set equivalence annotation no longer holds and the "
            "panel must be re-authored, not redrawn"
        )
    # The recorded L4 comparison must remain exhaustive and agreeing: the
    # panel says every held-out position was walked, so the walked count must
    # equal the scored population and the disagreement count must be zero.
    if data["l4_pymdp_argmax_disagreements"] != 0.0:
        raise ValueError(
            "the recorded L4 pymdp argmax disagreement count is non-zero; the "
            "every-position agreement annotation no longer holds and the panel "
            "must be re-authored, not redrawn"
        )
    if data["l4_pymdp_positions"] != data["l4_test_tokens"]:
        raise ValueError(
            "the recorded L4 pymdp comparison does not cover the scored population "
            f"({data['l4_pymdp_positions']:g} walked vs {data['l4_test_tokens']:g} "
            "scored tokens); the exhaustive annotation is false"
        )
    # L4's recorded agreement must stay a majority share for the annotation
    # describing it as the model's headline score to hold.
    if not 50.0 <= data["l4_agreement_pct"] <= 100.0:
        raise ValueError(
            f"the recorded Stage 4 agreement {data['l4_agreement_pct']:g} is "
            "outside [50, 100]; the numpy-decoder headline claim changed"
        )
    return data


def build_inference_evidence_figure(variables: dict[str, str] | None = None) -> Figure:
    """Build ``inference_evidence`` - three score paths, honestly labeled.

    Args:
        variables: Explicit token map; loaded from disk when omitted.

    Returns:
        A matplotlib Figure. The caller owns it and should pass it to
        :func:`ailm.figures._common.save_figure` and then close it.
    """
    apply_style()
    data = inference_evidence_data(variables)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.6))

    # L4: numpy filter, pymdp-checked at every held-out position
    ax = axes[0]
    ax.bar(
        ["numpy filter\n(recorded, full set)"],
        [data["l4_agreement_pct"]],
        color=COLOR_BASELINE,
        width=0.45,
    )
    ax.set_ylim(0.0, 100.0)
    ax.axhline(50.0, color=COLOR_INK, lw=1.2, linestyle="--")
    ax.text(
        0.0,
        44.0,
        "majority share required (data guard)",
        fontsize=FONTS["footnote"],
        ha="center",
        color=COLOR_INK,
    )
    ax.text(
        0.0,
        data["l4_agreement_pct"] + 3.0,
        f"{data['l4_agreement_pct']:g}%\nof {data['l4_test_tokens']:,.0f} tokens",
        ha="center",
        fontsize=FONTS["cell"],
    )
    ax.set_title(
        "L4 case - numpy forward filter\n"
        f"live pymdp agrees at all {data['l4_pymdp_positions']:,.0f} held-out positions",
        fontsize=FONTS["subtitle"],
    )
    ax.set_ylabel("case-role agreement, % of test tokens")
    ax.set_xlabel(EVIDENCE_KINDS[1], fontsize=FONTS["cell"], color=COLOR_MUTED)
    # Notes sit in the white headroom above each bar, never on the fill: muted
    # grey on the bar colors falls far below the contrast a reader needs, while
    # on the white background it is legible.
    ax.text(
        0.0,
        72.0,
        f"{data['l4_pymdp_argmax_disagreements']:,.0f} argmax disagreements,\n"
        f"max posterior deviation {data['l4_pymdp_posterior_max_abs_dev']:.2g}\n"
        "(decision 94; headline score stays NumPy)",
        ha="center",
        va="bottom",
        fontsize=FONTS["cell"],
        color=COLOR_MUTED,
    )
    ax.spines[["top", "right"]].set_visible(False)

    # L5 (the retained case-only ablation): closed-form mixture path, arrays
    # walked by hand; the bar is its recorded type-free perplexity, not the
    # mixture's own score.
    ax = axes[1]
    ax.bar(
        ["type-free bar\n(recorded, full set)"],
        [data["l5_typefree_perplexity"]],
        color=COLOR_MODEL,
        width=0.45,
    )
    ax.text(
        0.0,
        data["l5_typefree_perplexity"] + 0.12,
        f"{data['l5_typefree_perplexity']:g}",
        ha="center",
        fontsize=FONTS["cell"],
    )
    ax.set_ylim(0.0, data["l5_typefree_perplexity"] * 1.45)
    ax.set_title(
        "L5 case-only ablation - closed-form mixture path\n"
        f"live pymdp agrees on all {data['l5_pymdp_positions']:,.0f} held-out positions",
        fontsize=FONTS["subtitle"],
    )
    ax.set_ylabel("type-free per-position perplexity")
    ax.set_xlabel(EVIDENCE_KINDS[2], fontsize=FONTS["cell"], color=COLOR_MUTED)
    ax.text(
        0.0,
        data["l5_typefree_perplexity"] * 1.12,
        f"{data['l5_pymdp_type_argmax_disagreements']:,.0f} argmax disagreements "
        f"in {data['l5_pymdp_prefix_sentences']:,.0f} held-out sentences,\n"
        f"max joint-posterior deviation {data['l5_pymdp_posterior_max_abs_dev']:.2g}\n"
        "(decision 94; posterior agreement, not score equivalence)",
        ha="center",
        va="bottom",
        fontsize=FONTS["cell"],
        color=COLOR_MUTED,
    )
    ax.spines[["top", "right"]].set_visible(False)

    # L6: registered conditional (not inference) vs the live pymdp filter
    ax = axes[2]
    losses = [
        data["l6_unigram_logloss"],
        data["l6_phase_logloss"],
        data["l6_live_filter_logloss"],
    ]
    ax.bar(
        [
            "type unigram\n(the bar)",
            "observed position\n(registered)",
            "live Agent\n(pymdp filter)",
        ],
        losses,
        color=[COLOR_BASELINE, COLOR_ACCENT, COLOR_MODEL],
        width=0.45,
    )
    span = max(losses) - min(losses[1:])
    ax.set_ylim(min(losses[1:]) - 2.5 * span, max(losses) + 2.5 * span)
    for x, value in enumerate(losses):
        ax.text(x, value + 0.25 * span, f"{value:g}", ha="center", fontsize=FONTS["footnote"])
    ax.set_title(
        "L6 narrative - two distinct scores\nthe registered one is not inference",
        fontsize=FONTS["subtitle"],
    )
    ax.set_ylabel("held-out type log-loss, nats per sentence")
    ax.set_xlabel(EVIDENCE_KINDS[3], fontsize=FONTS["cell"], color=COLOR_MUTED)
    ax.text(
        0.5,
        0.97,
        f"live-filter delta +{data['l6_live_filter_improvement_nats']:g} nats,"
        f" {data['l6_test_sentences']:,.0f} sentences\n"
        "descriptive; document-clustered primary non-confirmatory",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=FONTS["cell"],
        color=COLOR_MUTED,
    )
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "What each upper-level score is: inference-level evidence vs array-level checks",
        fontsize=FONTS["title"],
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return fig


def generate_inference_evidence(
    output_dir: Any = None,
    variables: dict[str, str] | None = None,
) -> Any:
    """Write ``inference_evidence.png`` deterministically; return the path."""
    fig = build_inference_evidence_figure(variables)
    try:
        return save_figure(fig, "inference_evidence", output_dir)
    finally:
        plt.close(fig)
