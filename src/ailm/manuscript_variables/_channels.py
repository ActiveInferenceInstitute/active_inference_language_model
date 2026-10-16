"""Cross-level channel classification and qualification helpers."""

from __future__ import annotations

from typing import (
    Any,
    Mapping,
)

from ailm.channel_registry import (
    ChannelEvidenceRule,
    ChannelOutcome,
    channel_by_key,
)
from ailm.value_formatting import ManuscriptVariableError

from ._claims import _claim_number
from ._exceptions import MissingClaimError


def classify_cross_level_channels(claims: Mapping[str, Any]) -> dict[str, str]:
    """Classify the registered receiver interventions from their own estimands.

    The five measured seams read their recorded claims; the two case-word
    structural seams (the model-sender bridge and its privileged oracle
    diagnostic) have no scored estimand — their evaluation is structural and
    records nothing — so their class is the prior-only insertion the registry
    itself declares, checked against that declaration rather than a ledger
    number that does not exist.
    """

    def n(label: str) -> float:
        return _claim_number(claims, label)

    outcomes: dict[str, str] = {}
    for channel in channel_by_key().values():
        if channel.evidence_rule is ChannelEvidenceRule.GENERATION:
            outcome = (
                ChannelOutcome.HELPS
                if n("generation_tv_improvement") > 0
                and n("generation_best_membership_pct") > n("generation_baseline_membership_pct")
                else "changed"
            )
        elif channel.evidence_rule is ChannelEvidenceRule.LEXICAL:
            outcome = ChannelOutcome.HELPS if n("lexical_best_margin_pts") > 0 else "changed"
        elif channel.evidence_rule is ChannelEvidenceRule.LETTER_COUPLING:
            outcome = ChannelOutcome.HARMS if n("coupling_margin_pts") < 0 else "changed"
        elif channel.evidence_rule is ChannelEvidenceRule.NARRATIVE_PRIOR:
            static = n("stage6_static_interrogative_mass")
            real_shift = n("stage6_opening_interrogative_shift")
            scrambled_shift = n("stage6_scrambled_opening_shift")
            shifted_masses = (
                static + real_shift,
                static + n("stage6_middle_interrogative_shift"),
                static + n("stage6_closing_interrogative_shift"),
            )
            outcome = (
                ChannelOutcome.PRIOR_ONLY
                if real_shift != 0
                and real_shift != scrambled_shift
                and all(0 <= mass <= 1 for mass in shifted_masses)
                else "changed"
            )
        elif channel.evidence_rule is ChannelEvidenceRule.CLAUSE_WORD:
            aggregate_helps = n("clause_word_best_delta_ppl") < 0
            improved = n("clause_word_best_improved_tokens")
            degraded = n("clause_word_best_degraded_tokens")
            if aggregate_helps and improved > degraded and n("clause_word_confirms_harm") == 0:
                outcome = ChannelOutcome.HELPS
            elif aggregate_helps and improved < degraded and n("clause_word_confirms_harm") == 0:
                outcome = ChannelOutcome.MIXED_DIRECTION
            else:
                outcome = "changed"
        elif channel.evidence_rule in (
            ChannelEvidenceRule.CASE_WORD_BRIDGE,
            ChannelEvidenceRule.CASE_WORD_ORACLE_DIAGNOSTIC,
        ):
            # Structural seam, not a scored estimand: the inlet mixes a
            # message into the word prior and touches no observation
            # likelihood. Read the declared insertion site so a registry
            # relabel away from the prior fails closed through the
            # registry cross-check instead of echoing a stale class.
            outcome = (
                ChannelOutcome.PRIOR_ONLY if channel.insertion == "word prior D" else "changed"
            )
        else:  # pragma: no cover - exhaustive StrEnum guard
            raise ManuscriptVariableError(
                f"channel {channel.key!r} has unknown evidence rule {channel.evidence_rule!r}"
            )
        outcomes[channel.key] = str(outcome)
    return outcomes


LEVEL_QUALIFYING_VERDICT_FLAGS: tuple[str, ...] = ("stage6_primary_nonconfirmatory",)


def qualified_verdict_flags(claims: Mapping[str, Any]) -> dict[str, bool]:
    """The recorded qualification guards, using the actual headline analyses.

    Two guards, and since decision 114 they are not both about a level's
    verdict. ``stage6_primary_nonconfirmatory`` qualifies Level 6.
    ``stage5_nonpassing_unresolved`` describes the retained case-only clause
    ablation, which is no longer Level 5;
    :data:`LEVEL_QUALIFYING_VERDICT_FLAGS` is the subset that still counts
    against a level.
    """

    def n(label: str) -> float:
        return _claim_number(claims, label)

    from ailm.gates import (
        GateVerdict,
        stage5_nonpassing_unresolved_is_consistent,
        stage6_positive_nonconfirmatory_is_consistent,
    )

    return {
        "stage5_nonpassing_unresolved": stage5_nonpassing_unresolved_is_consistent(
            verdict=GateVerdict.ABOVE_REFERENCE,
            point_passes=n("stage5_mixture_perplexity") < n("stage5_typefree_perplexity"),
            oracle_sentence_ci_low=n("stage5_oracle_gain_sentence_ci_low"),
            oracle_sentence_ci_high=n("stage5_oracle_gain_sentence_ci_high"),
            oracle_position_ci_low=n("stage5_oracle_gain_position_ci_low"),
            oracle_position_ci_high=n("stage5_oracle_gain_position_ci_high"),
            control_accuracy=n("stage5_control_classification_pct") / 100.0,
            control_mixture_perplexity=n("stage5_control_mixture_perplexity"),
            control_typefree_perplexity=n("stage5_control_typefree_perplexity"),
        ),
        "stage6_primary_nonconfirmatory": stage6_positive_nonconfirmatory_is_consistent(
            verdict=GateVerdict.BRACKETED_REFERENCE,
            point_passes=n("stage6_logloss_improvement_nats") > 0.0,
            point_improvement=n("stage6_logloss_improvement_nats"),
            cluster_ci_low=n("cluster_stage6_logloss_ci_low"),
            cluster_ci_high=n("cluster_stage6_logloss_ci_high"),
            cluster_p=n("cluster_stage6_logloss_p"),
        ),
    }


def level_six_qualification_limits(claims: Mapping[str, Any]) -> dict[str, bool]:
    """The recorded limits that keep Level 6 a qualified verdict after decision 115.

    Decision 115 promoted Level 6's finding status: the primary estimand excludes
    zero on the held-back population. The badge reads `passing` accordingly. The
    level is nonetheless still QUALIFIED, and by three separate recorded facts
    rather than one:

    * the development-population clustered interval still spans zero, so the two
      populations disagree and the earlier result is not retracted;
    * the same held-back bootstrap read at ``CORRECTED_CONFIDENCE`` spans zero,
      so the resolution is a claim at the development analysis's coverage and
      not at the stricter one;
    * the held-back population cannot resolve an effect the size of the
      development one, so its null would have been uninterpretable and the
      detection is of a larger effect rather than of this effect generalizing.

    ``LEVEL_QUALIFYING_VERDICT_FLAGS`` counts Level 6 through the first of these
    alone. That was sufficient when it was the only one. It is not sufficient
    now: were the development interval ever to move, the count would silently
    report no qualified levels while the other two limits still stood, and the
    abstract would say the stack is unqualified when it is not. This function is
    what the render checks that against.

    A missing label reads as a limit that no longer holds, not as one that does,
    so a ledger that lost the arm cannot manufacture a qualification.
    """

    def flag(label: str) -> bool:
        entry = claims.get(label)
        value = None if entry is None else entry.get("value")
        return value if isinstance(value, bool) else False

    development_spans_zero = False
    try:
        low = _claim_number(claims, "cluster_stage6_logloss_ci_low")
        high = _claim_number(claims, "cluster_stage6_logloss_ci_high")
        development_spans_zero = low <= 0.0 <= high
    except (MissingClaimError, ManuscriptVariableError):
        development_spans_zero = False

    return {
        "development_interval_spans_zero": development_spans_zero,
        "corrected_interval_spans_zero": not flag(
            "confirmatory_stage6_clustered_corrected_excludes_zero"
        )
        and claims.get("confirmatory_stage6_clustered_corrected_excludes_zero") is not None,
        "development_sized_effect_unresolvable": not flag(
            "confirmatory_stage6_clustered_development_effect_resolved"
        )
        and claims.get("confirmatory_stage6_clustered_development_effect_resolved") is not None,
    }


def canonical_stage5_passes(claims: Mapping[str, Any]) -> bool:
    """Whether the canonical Level 5 still clears the bar decision 114 promoted.

    Decision 114 made the widened-observation clause model of decision 112 the
    canonical Level 5, on the strength of a gate that was registered before its
    artifact existed and could have failed. Its falsifier is what this reads:
    the mixture beating the refitted type-free bar, the privileged-label gain
    resolved away from zero on both document-resampled scales, and the
    detectability control still firing. Losing any of them withdraws the
    promotion, so a render that would count Level 5 among the passing levels
    checks them first rather than assuming them.

    The labels are the ``stage5w_`` prefix throughout, which decision 114 kept
    pointing at the canonical arm rather than renaming; ``stage5_`` labels
    belong to the retained case-only ablation and are read by
    :func:`qualified_verdict_flags`.
    """

    def n(label: str) -> float:
        return _claim_number(claims, label)

    from ailm.gates import STAGE5_DETECTABILITY_ACCURACY_FLOOR

    return bool(
        n("stage5w_mixture_perplexity") < n("stage5w_typefree_perplexity")
        and n("stage5w_type_oracle_perplexity") < n("stage5w_typefree_perplexity")
        and n("stage5w_oracle_gain_sentence_ci_low") > 0.0
        and n("stage5w_oracle_gain_position_ci_low") > 0.0
        and n("stage5w_control_classification_pct") / 100.0 > STAGE5_DETECTABILITY_ACCURACY_FLOOR
        and n("stage5w_control_mixture_perplexity") < n("stage5w_control_typefree_perplexity")
    )
