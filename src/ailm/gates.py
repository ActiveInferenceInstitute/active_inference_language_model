"""Shared, fail-closed semantics for every quantitative AILM gate.

Gate *placement* and model *outcome* are different claims, and a placement
verdict is only as strong as the comparator that produced it.  Every ceiling
therefore carries a declared :class:`ComparatorType`, and the verdict names are
chosen so that an impossibility claim can only be produced by a comparator that
licenses one:

``VACUOUS``
    The trivial baseline already passes the strict threshold.
``UNREACHABLE``
    A *bound-type* ceiling does not pass the strict threshold.  Licensed only
    by :data:`ComparatorType.MODEL_CLASS_BOUND` or
    :data:`ComparatorType.ANALYTIC_BOUND`, because ``c < t`` implies
    "no model of the class passes" only when ``c`` bounds the class.
``ABOVE_REFERENCE``
    A measured, exceedable ceiling (a reference model or a privileged oracle)
    does not pass the strict threshold.  The threshold sits above that
    comparator; no impossibility claim follows, because another model in the
    admissible class may still exceed it.
``UNAUDITED``
    The baseline fails, but no ceiling was measured.
``BRACKETED_BOUND``
    The baseline fails and a bound-type ceiling passes.
``BRACKETED_REFERENCE``
    The baseline fails and a reference-type or oracle-type ceiling passes.

Only the two bracketed verdicts license a threshold.  Neither says that the
shipped model passed; :func:`evaluate_gate` reports that separately.
Centralising this logic prevents the audit script and methodology figures from
silently acquiring different boundary semantics.

The comparator typing is fail-closed: :func:`classify_gate` refuses a measured
ceiling whose type is undeclared, so an untyped comparator cannot silently
acquire bound semantics.  The four types mirror the comparator-type column of
the manuscript's gate-audit table and are formalized in
``formal/AILM/GateVerdict.lean``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from typing import Literal, TypeAlias

Direction: TypeAlias = Literal["higher_is_better", "lower_is_better"]


class ComparatorType(StrEnum):
    """What kind of quantity a gate's ceiling is.

    The distinction that matters is whether the number *bounds the model
    class* on the scored population.  Only a bound turns "the ceiling fails
    the threshold" into "no implementation could pass".
    """

    #: A theorem-backed bound on every model of the class, measured on the
    #: same population the gate scores (e.g. the evaluation passage's own
    #: conditional entropy, which lower-bounds any order-1 model's log loss
    #: there by the decomposition in the manuscript's held-out-bound equation).
    MODEL_CLASS_BOUND = "model_class_bound"
    #: A closed-form bound that needs no measurement (e.g. a total-variation
    #: distance of zero: no generator can score below it).
    ANALYTIC_BOUND = "analytic_bound"
    #: A measured comparator given information the model forgoes (a
    #: fitted-on-test distribution, a true-label oracle).  Strong, but not a
    #: bound: the Bayes-optimal use of less information can in principle beat
    #: a naive use of more.
    PRIVILEGED_ORACLE = "privileged_oracle"
    #: A measured system with no privileged access (a smoothing decode of the
    #: same model, a stronger lookup).  Explicitly exceedable — the Stage 4
    #: derived row's shipped agent scores above its smoothing reference.
    REFERENCE_MODEL = "reference_model"


#: The comparator types that license an impossibility verdict.
BOUND_COMPARATOR_TYPES: frozenset[ComparatorType] = frozenset(
    {ComparatorType.MODEL_CLASS_BOUND, ComparatorType.ANALYTIC_BOUND}
)


class GateVerdict(StrEnum):
    """Epistemic status of a threshold's placement."""

    VACUOUS = "VACUOUS"
    UNREACHABLE = "UNREACHABLE"
    ABOVE_REFERENCE = "ABOVE_REFERENCE"
    UNAUDITED = "UNAUDITED"
    BRACKETED_BOUND = "BRACKETED_BOUND"
    BRACKETED_REFERENCE = "BRACKETED_REFERENCE"


#: The verdicts that place the threshold strictly between the baseline and the
#: measured ceiling.  Historical prose calls both simply "bracketed".
BRACKETED_VERDICTS: frozenset[GateVerdict] = frozenset(
    {GateVerdict.BRACKETED_BOUND, GateVerdict.BRACKETED_REFERENCE}
)


def verdict_is_bracketed(verdict: GateVerdict) -> bool:
    """Whether a verdict certifies threshold placement (of either strength)."""
    return verdict in BRACKETED_VERDICTS


class GateId(StrEnum):
    """Stable identities for the gates audited by ``scripts/audit_gates.py``."""

    STAGE1_LETTER = "stage1_letter"
    STAGE2_PHONEME = "stage2_phoneme"
    STAGE2_G2P = "stage2_g2p"
    STAGE3_WORD = "stage3_word"
    STAGE4_CASE = "stage4_case"
    GENERATION = "generation"
    STAGE5_CLAUSE = "stage5_clause"
    STAGE6_NARRATIVE = "stage6_narrative"


AUDITED_GATE_IDS: tuple[GateId, ...] = tuple(GateId)

#: Stage 2B's original fixed accuracy threshold. This is historical method
#: configuration, not a measurement: the same constant drives the gate-taxonomy
#: figure and the manuscript token that quotes the retired declaration.
HISTORICAL_STAGE2_G2P_THRESHOLD_PCT = 60.0

#: Stage 1's original gain-over-uniform target, in nats. Historical method
#: configuration, not a measurement, and this constant is its single owner: the
#: recorder writes it to the ledger under ``stage1_declared_gate_nats``, and the
#: figures read that ledger token and refuse to draw if the two disagree.
STAGE1_DECLARED_GATE_NATS = 2.0

#: The planted Stage 5 corpus must be classified above this accuracy before an
#: English non-passing result is interpretable. The recorder and audit share this value so a
#: control cannot be live under one path and dead under the other.
STAGE5_DETECTABILITY_ACCURACY_FLOOR = 0.9

#: Conventional level used only to label the Stage 6 clustered primary as
#: non-confirmatory.  It is not the repository's multiplicity threshold.
PRIMARY_NONSIGNIFICANCE_ALPHA = 0.05

#: The widened Stage 5 arm's registered prediction-1 band, exactly as
#: registered. Decision 112 (prediction 1 and falsifier 1) requires the arm's
#: type-oracle gain to be positive and to fall "inside the interval the probe
#: registered"; the registration quotes that interval — the probe's own 99%
#: bootstrap CI — at the widened item of ``docs/research_queue.md`` as
#: ``[+0.039, +0.321]`` nats/sentence. Those committed endpoints are the
#: registration's own three-decimal quoting of the probe's bootstrap interval:
#: the same entry's original evidence line (commit 3162b61) recorded the
#: endpoints to four decimals as ``[+0.0394, +0.3209]``, computed by
#: uncommitted scratchpad scope scripts on the retired sentence split. The
#: committed rounding therefore widens the band outward on both sides, and
#: this constant encodes the registration exactly as written: endpoints
#: inclusive, in nats per sentence, a contamination band across populations —
#: not a confidence interval of the recorded arm, and not the
#: document-resampled sentence/position intervals the recorder computes.
STAGE5W_REGISTERED_GAIN_INTERVAL_NATS: tuple[float, float] = (0.039, 0.321)

#: ── Declared smallest effect sizes of interest (decision 123) ──────────────
#: Every audited gate now declares, in its own reporting scale, how large a
#: margin has to be before it can be read as more than fitting noise. The
#: declared relation in :func:`evaluate_gate` keeps ``min_effect = 0.0`` on
#: these eight retrospective rows on purpose — their thresholds were restated
#: after measurement, so raising the outcome bar retrospectively would be as
#: retrospective as the threshold and would rewrite recorded outcomes. What a
#: declared SES does instead is make the margin readable beside the verdict:
#: :func:`smallest_effect_size` supplies the floor, ``scripts/audit_gates.py``
#: reports the margin in this unit next to each verdict and flags a margin
#: below the floor, and each stage recorder writes its constant to the ledger
#: beside the gate's claims. A hairline positive margin can therefore no
#: longer certify a gate silently: it is still a pass (nothing here flips any
#: recorded verdict), but the audit and the ledger now say so in the same
#: breath.
#
#: The values are chosen from each gate's own scale and its recorded
#: baselines; the rationale travels with each constant and is restated in
#: decision 123.

#: Stage 1 (letter perplexity, lower is better). The margin is read on the
#: log scale — ln(threshold/model) — because perplexities multiply. The
#: recorded margin is ln(19.9558/11.9215) ≈ 0.514 nats, and the level's whole
#: captured signal over uniform is 0.9229 nats, so a quarter of a nat is
#: roughly a quarter of everything this level claims: below that an
#: improvement is not distinguishable from the surprise of a single symbol.
STAGE1_GATE_SESOI_NATS = 0.25

#: Stage 2A (phoneme perplexity, lower is better, in-sample full-dictionary
#: protocol). The recorded margin is ln(27.5944/16.7445) ≈ 0.501 nats; the
#: same quarter-nat floor applies, chosen from the scale of the level's own
#: gain over uniform (0.9744 nats).
STAGE2_PHONEME_GATE_SESOI_NATS = 0.25

#: Stage 2B (G2P accuracy, higher is better, percentage points on the full
#: 42,820-position held-out set). The recorded margin is 0.546 pts — the
#: hairline that motivated this contract: significant only at the full
#: population (McNemar p=3.5e-6), while its bounded-subsample margins (e.g.
#: 0.327 pts on the stride-5 subset) are recorded as NOT significant. Half a
#: point is 214 of 42,820 positions; the recorded margin clears it, barely,
#: and the audit now says so explicitly instead of letting it read as a
#: finding.
STAGE2_G2P_GATE_SESOI_PTS = 0.5

#: Stage 3 (word per-emission perplexity, lower is better). The recorded gain
#: over the letter-bigram bar is 1.0181 nats (``stage3_gain_nats``); the same
#: quarter-nat floor as Stages 1 and 2A keeps the three log-scale gates on
#: one declared scale.
STAGE3_WORD_GATE_SESOI_NATS = 0.25

#: Stage 4 (derived-population case agreement, percentage points on the
#: 1,319-token held-out set). The recorded margin is 7.81 pts but the results
#: section records that one stratum carries it, so the floor is deliberately
#: conservative: two points is about 26 held-out tokens, below which the
#: aggregate margin can move with a single stratum. The gold population's
#: margin is 0.0 and non-passing; this floor changes nothing there.
STAGE4_CASE_GATE_SESOI_PTS = 2.0

#: Generation (exact TV against ancestral sampling, on the 0-1 TV scale). The
#: recorded improvement is 0.5643 and the degenerate one-word control sits at
#: 0.9071; a TV improvement below a tenth does not separate the policy
#: generator from sampling from B, whatever the diversity conjunct says.
GENERATION_GATE_SESOI_TV = 0.1

#: Stage 5, the canonical widened-observation clause gate. The point
#: perplexity margin is hairline by construction (the type-free bar is
#: refitted on the identical stream), so the declared effect size is placed
#: on the scale the registration itself uses: the privileged-label oracle
#: gain in nats per sentence. The recorded gain is 0.198647 with a 99%
#: document-bootstrap interval whose low end is 0.0246; below 0.05 the gain
#: would sit inside a single document's contribution and inside the low
#: quarter of the registered band [0.039, 0.321].
STAGE5_CLAUSE_GATE_SESOI_NATS_PER_SENTENCE = 0.05

#: Stage 6 (document-position association, nats per sentence of type
#: log-loss). The clustered design's own resolvable unit sets this floor: the
#: planted detectability control shifts 0.0051 nats and the recorded min
#: detectable effect is 0.005726 nats, so below half a percent of a nat the
#: development point improvement is within document-sampling noise — which is
#: exactly why the primary is recorded non-confirmatory (decision 115). The
#: recorded point improvement of 0.0051 nats clears this floor by a hair, and
#: the audit now displays that hairline instead of letting the positive sign
#: carry it.
STAGE6_NARRATIVE_GATE_SESOI_NATS = 0.005


@dataclass(frozen=True, slots=True)
class SmallestEffectSize:
    """A gate's declared smallest effect size of interest, on its own scale."""

    #: The floor the margin must meet or exceed, in ``unit``.
    value: float
    #: The reporting scale the margin is read on ("nats" for perplexity and
    #: log-loss rows, "percentage points" for accuracy/agreement rows,
    #: "total variation" for the generation row, "nats per sentence" for the
    #: registered Stage 5 and Stage 6 estimands).
    unit: str


#: The declared SESOI per audited gate id. A gate id absent here fails closed
#: in :func:`smallest_effect_size`; the retained case-only ablation is not a
#: :class:`GateId` member and carries its recorded non-passing finding rather
#: than a declared floor.
SMALLEST_EFFECT_SIZES: dict[GateId, SmallestEffectSize] = {
    GateId.STAGE1_LETTER: SmallestEffectSize(STAGE1_GATE_SESOI_NATS, "nats"),
    GateId.STAGE2_PHONEME: SmallestEffectSize(STAGE2_PHONEME_GATE_SESOI_NATS, "nats"),
    GateId.STAGE2_G2P: SmallestEffectSize(STAGE2_G2P_GATE_SESOI_PTS, "percentage points"),
    GateId.STAGE3_WORD: SmallestEffectSize(STAGE3_WORD_GATE_SESOI_NATS, "nats"),
    GateId.STAGE4_CASE: SmallestEffectSize(STAGE4_CASE_GATE_SESOI_PTS, "percentage points"),
    GateId.GENERATION: SmallestEffectSize(GENERATION_GATE_SESOI_TV, "total variation"),
    GateId.STAGE5_CLAUSE: SmallestEffectSize(
        STAGE5_CLAUSE_GATE_SESOI_NATS_PER_SENTENCE, "nats per sentence"
    ),
    GateId.STAGE6_NARRATIVE: SmallestEffectSize(
        STAGE6_NARRATIVE_GATE_SESOI_NATS, "nats per sentence"
    ),
}


def smallest_effect_size(gate_id: GateId | str) -> SmallestEffectSize:
    """Return a gate's declared SESOI, refusing an undeclared one.

    Args:
        gate_id: The audited gate's stable identity. A plain string that is
            not a :class:`GateId` member (the retained ablation) raises, by
            design: only the eight canonical rows declare a floor.
    """
    try:
        return SMALLEST_EFFECT_SIZES[GateId(gate_id)]
    except (KeyError, ValueError):
        raise ValueError(f"gate {gate_id!r} declares no smallest effect size") from None


def effect_size_is_of_interest(margin: float, sesoi: SmallestEffectSize) -> bool:
    """Whether a signed margin, in the SESOI's own unit, meets the floor.

    A negative margin (a non-passing gate) is never of interest: the floor is
    a lower bound on a positive effect, not an indifference band.
    """
    value = _finite_metric("margin", margin)
    return bool(value >= sesoi.value)


@dataclass(frozen=True, slots=True)
class GateEvaluation:
    """Placement verdict and shipped-model outcome for one gate."""

    verdict: GateVerdict
    passes: bool
    margin: float


def stage5_nonpassing_unresolved_is_consistent(
    *,
    verdict: GateVerdict,
    point_passes: bool,
    oracle_sentence_ci_low: object,
    oracle_sentence_ci_high: object,
    oracle_position_ci_low: object,
    oracle_position_ci_high: object,
    control_accuracy: object,
    control_mixture_perplexity: object,
    control_typefree_perplexity: object,
) -> bool:
    """Whether Stage 5 remains non-passing and its oracle gain unresolved.

    A failed point gate plus a planted-effect control does not establish a null.
    The document-disjoint remeasurement additionally requires the fitted
    privileged-label reference to remain on the failing side of the bar and
    both whole-document-bootstrap readings of its paired gain to span zero. If
    the comparator clears the bar, either interval resolves a direction, or the
    mixture passes, the public status has changed and must be reviewed.
    """
    sentence_low = _finite_metric("oracle_sentence_ci_low", oracle_sentence_ci_low)
    sentence_high = _finite_metric("oracle_sentence_ci_high", oracle_sentence_ci_high)
    position_low = _finite_metric("oracle_position_ci_low", oracle_position_ci_low)
    position_high = _finite_metric("oracle_position_ci_high", oracle_position_ci_high)
    accuracy = _accuracy_fraction("control_accuracy", control_accuracy)
    mixture = _finite_metric("control_mixture_perplexity", control_mixture_perplexity)
    typefree = _finite_metric("control_typefree_perplexity", control_typefree_perplexity)
    return bool(
        verdict is GateVerdict.ABOVE_REFERENCE
        and not point_passes
        and sentence_low <= 0.0 <= sentence_high
        and position_low <= 0.0 <= position_high
        and accuracy > STAGE5_DETECTABILITY_ACCURACY_FLOOR
        and mixture < typefree
    )


def stage5_widened_registered_gain_in_interval(oracle_gain_nats_per_sentence: object) -> bool:
    """Whether the widened arm's gain sits inside its registered interval.

    Decision 112, prediction 1 and falsifier 1: the type-oracle gain must be
    positive and fall inside the interval the probe registered
    (:data:`STAGE5W_REGISTERED_GAIN_INTERVAL_NATS`). A point outside means the
    pre-computation was contaminated and the proposal is withdrawn rather than
    re-fitted, which is the one registered prediction with the authority to
    refuse a record. Missing, non-numeric, and nonfinite values fail closed.
    """
    gain = _finite_metric("oracle_gain_nats_per_sentence", oracle_gain_nats_per_sentence)
    low, high = STAGE5W_REGISTERED_GAIN_INTERVAL_NATS
    return bool(gain > 0.0 and low <= gain <= high)


def stage5_widened_auc_above_registered_comparators(
    *,
    auc: object,
    auc_length_control: object,
    auc_first_aux_control: object,
) -> bool:
    """Whether the widened arm still holds registered prediction 2.

    Decision 112, prediction 2: classification AUC is **above both**
    comparator (b), length-only, and comparator (c), first-symbol-is-AUX.
    Above one and not the other is a failure, not a partial success, so the
    conjunction is required and each comparison is strict.

    The relation is only meaningful on matching populations and estimands:
    every AUC must be the tie-aware Mann-Whitney rank statistic of
    interrogative-versus-declarative over the identical held-out sentences
    and labels as the mixture's. Each value is validated as a finite real
    (missing and nonfinite fail closed) and as a 0-1 fraction, so a 0-100
    percentage field cannot masquerade as an AUC.
    """
    auc_value = _accuracy_fraction("auc", auc)
    length_control = _accuracy_fraction("auc_length_control", auc_length_control)
    first_aux_control = _accuracy_fraction("auc_first_aux_control", auc_first_aux_control)
    return bool(auc_value > length_control and auc_value > first_aux_control)


def stage5_widened_passing_resolved_is_consistent(
    *,
    verdict: GateVerdict,
    point_passes: bool,
    oracle_sentence_ci_low: object,
    oracle_sentence_ci_high: object,
    oracle_position_ci_low: object,
    oracle_position_ci_high: object,
    control_accuracy: object,
    control_mixture_perplexity: object,
    control_typefree_perplexity: object,
    swapped_blocks_accuracy: object,
    model_classification_accuracy: object,
    leak_control_divergence: object,
    decoder_tolerance: object,
) -> bool:
    """Whether the canonical widened row still satisfies decision 114.

    Decision 114 withdraws the promotion when the canonical arm's mixture
    stops beating the widened type-free bar, when either document-resampled
    interval on its privileged-label gain comes to span zero, or when either
    of its negative controls stops rejecting. All must hold together:

    * the placement stays bracketed and the point gate passes — the ordinary
      passing rule this row shares with Stages 1-4;
    * both whole-document-bootstrap intervals on the paired privileged-label
    * gain remain wholly positive. The arm exists to size a positive effect,
      so an interval resolving on the negative side is also a changed finding,
      and an interval whose low exceeds its high is rejected outright;
    * the detectability control still fires at the widened alphabet — a 0-1
      accuracy fraction above the shared floor (a percentage-scaled value
      fails closed on units), with the control's mixture beating its type-free
      chain;
    * both negative controls still reject: exchanging the two types' fitted
      chains must cost accuracy, and planted cross-type transition mass must
      break the closed-form/oracle agreement by more than decoder tolerance.

    The development-session and no-held-back limitations (decision 112;
    decision 111's spent population) are labeling and prose duties, not
    numeric states: this predicate neither reads nor relaxes them, and it
    never touches the retained case-only ablation's identity or numbers.
    """
    sentence_low = _finite_metric("oracle_sentence_ci_low", oracle_sentence_ci_low)
    sentence_high = _finite_metric("oracle_sentence_ci_high", oracle_sentence_ci_high)
    position_low = _finite_metric("oracle_position_ci_low", oracle_position_ci_low)
    position_high = _finite_metric("oracle_position_ci_high", oracle_position_ci_high)
    accuracy = _accuracy_fraction("control_accuracy", control_accuracy)
    swapped = _accuracy_fraction("swapped_blocks_accuracy", swapped_blocks_accuracy)
    model_accuracy = _accuracy_fraction(
        "model_classification_accuracy", model_classification_accuracy
    )
    mixture = _finite_metric("control_mixture_perplexity", control_mixture_perplexity)
    typefree = _finite_metric("control_typefree_perplexity", control_typefree_perplexity)
    leak = _finite_metric("leak_control_divergence", leak_control_divergence)
    tolerance = _finite_metric("decoder_tolerance", decoder_tolerance)
    return bool(
        verdict_is_bracketed(verdict)
        and point_passes
        and sentence_high >= sentence_low > 0.0
        and position_high >= position_low > 0.0
        and accuracy > STAGE5_DETECTABILITY_ACCURACY_FLOOR
        and mixture < typefree
        and swapped < model_accuracy
        and leak > tolerance
    )


def stage6_positive_nonconfirmatory_is_consistent(
    *,
    verdict: GateVerdict,
    point_passes: bool,
    point_improvement: object,
    cluster_ci_low: object,
    cluster_ci_high: object,
    cluster_p: object,
) -> bool:
    """Whether Stage 6 is positive at the point and non-confirmatory by document."""
    improvement = _finite_metric("point_improvement", point_improvement)
    low = _finite_metric("cluster_ci_low", cluster_ci_low)
    high = _finite_metric("cluster_ci_high", cluster_ci_high)
    p_value = _finite_metric("cluster_p", cluster_p)
    return bool(
        verdict_is_bracketed(verdict)
        and point_passes
        and improvement > 0.0
        and low <= 0.0 <= high
        and p_value >= PRIMARY_NONSIGNIFICANCE_ALPHA
    )


def stage6_positive_replicated_is_consistent(
    *,
    verdict: GateVerdict,
    point_passes: bool,
    point_improvement: object,
    cluster_ci_low: object,
    cluster_ci_high: object,
    held_back_ci_low: object,
    held_back_ci_high: object,
    held_back_excludes_zero: object,
    held_back_sign_crosses_threshold: object,
    held_back_sign_direction_agrees: object,
) -> bool:
    """Whether Stage 6's primary estimand replicated on the held-back population.

    Decision 115. The promotion is a conjunction of two independent held-back
    results and one unchanged development fact, and every part is required:

    * the development-population primary interval still spans zero, because that
      sentence stays in the manuscript and the promotion is not a claim that it
      became false;
    * the held-back sign test crossed its frozen threshold in the direction the
      development result pointed (decision 111);
    * the held-back primary interval excludes zero **on the positive side**.

    The direction check is not redundant with exclusion. An interval lying wholly
    below zero also excludes it, and would mean the phase model was reliably
    *worse* on the held-back population -- a result that must never be readable
    as a replicated improvement.
    """
    improvement = _finite_metric("point_improvement", point_improvement)
    development_low = _finite_metric("cluster_ci_low", cluster_ci_low)
    development_high = _finite_metric("cluster_ci_high", cluster_ci_high)
    held_back_low = _finite_metric("held_back_ci_low", held_back_ci_low)
    held_back_high = _finite_metric("held_back_ci_high", held_back_ci_high)
    return bool(
        verdict_is_bracketed(verdict)
        and point_passes
        and improvement > 0.0
        and development_low <= 0.0 <= development_high
        and _declared_flag("held_back_excludes_zero", held_back_excludes_zero)
        and _declared_flag("held_back_sign_crosses_threshold", held_back_sign_crosses_threshold)
        and _agreeing_direction(held_back_sign_direction_agrees)
        and held_back_low > 0.0
        and held_back_high > 0.0
    )


def _declared_flag(name: str, value: object) -> bool:
    """Return a recorded boolean, refusing anything that merely looks like one.

    The Stage 6 status is a function of ledger booleans, so a truthy string or a
    ``1`` arriving where a flag is expected would silently promote the finding.
    """
    if not isinstance(value, bool):
        raise ValueError(
            f"{name} must be a recorded boolean, got {value!r} of type {type(value).__name__}"
        )
    return value


def _agreeing_direction(value: object) -> bool:
    """Whether a recorded three-valued direction flag says the directions agree.

    Decision 111 records direction agreement as ``"true"`` / ``"false"`` /
    ``"undetermined"`` rather than a boolean, because a comparison with no
    discordant pairs has no direction at all. Only the first promotes.
    """
    if value in (True, False):
        return bool(value)
    if not isinstance(value, str):
        raise ValueError(
            f"held_back_sign_direction_agrees must be a recorded direction flag, got {value!r}"
        )
    if value not in ("true", "false", "undetermined"):
        raise ValueError(
            f"held_back_sign_direction_agrees must be one of true/false/undetermined, got {value!r}"
        )
    return value == "true"


def _finite_metric(name: str, value: object) -> float:
    """Return ``value`` as a finite float, rejecting bools and opaque coercions."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"{name} must be a finite real number, got {value!r} of type {type(value).__name__}"
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return numeric


def _accuracy_fraction(name: str, value: object) -> float:
    """Return ``value`` as a finite accuracy fraction in the unit interval.

    The Stage 5 detectability floor is a fraction (0.90), while the ledger's
    ``*_pct`` fields are percentages on a 0-100 scale. A percentage reaching
    one of these predicates would clear the floor trivially and in the wrong
    units, so anything outside [0, 1] is a schema violation, not a finding.
    """
    numeric = _finite_metric(name, value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(
            f"{name} must be an accuracy or AUC fraction on the 0-1 scale, got "
            f"{value!r}; a 0-100 percentage would silently clear the "
            f"{STAGE5_DETECTABILITY_ACCURACY_FLOOR} floor"
        )
    return numeric


def _direction(direction: object) -> Direction:
    """Validate and narrow a direction string."""
    if direction not in ("higher_is_better", "lower_is_better"):
        raise ValueError(
            f"direction must be 'higher_is_better' or 'lower_is_better', got {direction!r}"
        )
    return direction


def _comparator_type(ceiling_type: object) -> ComparatorType:
    """Validate and narrow a comparator type, accepting the enum or its value."""
    if isinstance(ceiling_type, ComparatorType):
        return ceiling_type
    try:
        return ComparatorType(ceiling_type)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        raise ValueError(
            "ceiling_type must be a ComparatorType, got "
            f"{ceiling_type!r}; a measured ceiling with an undeclared type "
            "would silently decide between an impossibility claim and a "
            "reference comparison"
        ) from None


def gate_passes(value: object, threshold: object, direction: Direction) -> bool:
    """Return whether a metric strictly passes a threshold.

    Equality is a failure in both directions.  This is load-bearing: a baseline
    equal to its bar must fail, while a ceiling equal to its bar cannot establish
    reachability.
    """
    metric = _finite_metric("value", value)
    bar = _finite_metric("threshold", threshold)
    resolved = _direction(direction)
    return metric > bar if resolved == "higher_is_better" else metric < bar


def gate_margin(value: object, threshold: object, direction: Direction) -> float:
    """Signed margin, positive exactly when :func:`gate_passes` is true."""
    metric = _finite_metric("value", value)
    bar = _finite_metric("threshold", threshold)
    resolved = _direction(direction)
    return metric - bar if resolved == "higher_is_better" else bar - metric


def gate_is_baseline_relative(baseline: object, threshold: object) -> bool:
    """Whether the threshold *is* the baseline, making the verdict predetermined.

    A gate restated as "beat the trivial baseline" sets ``threshold ==
    baseline``. Under the strict comparison :func:`gate_passes` uses, such a
    baseline can never clear its own bar, so :data:`GateVerdict.VACUOUS` becomes
    logically unreachable and the verdict is decided entirely by the ceiling.
    The audit then reports a bracketed verdict for every such gate whose ceiling
    beats the baseline, which is every one of them: a ceiling that did not beat
    the baseline would not be a ceiling.

    That is not a bug in the taxonomy. It is a fact about applying the taxonomy
    to a threshold defined as the baseline: the classification was determined
    before any measurement, so it carries no information about where the
    threshold sits. This project's doctrine is that a metric which cannot fail
    is not a gate; the same discipline turned on itself says a verdict which
    cannot come out otherwise is not an audit.

    The predicate is deliberately separate from :func:`classify_gate` rather
    than an extra verdict. The verdicts are formalized in
    ``formal/AILM/GateVerdict.lean`` with theorems fixing exactly when each
    fires, and a new case would invalidate those proofs in order to record
    something that is a property of how the gate was *constructed* rather than
    of how it classifies.

    What carries the evidential weight for a baseline-relative gate is not the
    verdict but the two things beside it: whether the shipped model clears the
    baseline, and whether that margin is supported. That is why the Stage 2B
    restatement names a paired McNemar test rather than a number.

    Args:
        baseline: The trivial baseline's metric.
        threshold: The declared or restated bar.

    Returns:
        True when the two are equal, so the verdict is predetermined.

    Raises:
        ValueError: If either value is not a finite real number.
    """
    return _finite_metric("baseline", baseline) == _finite_metric("threshold", threshold)


def classify_gate(
    baseline: object,
    ceiling: object | None,
    threshold: object,
    direction: Direction,
    *,
    ceiling_type: ComparatorType | str | None = None,
) -> GateVerdict:
    """Classify threshold placement from a baseline and a typed ceiling.

    Args:
        baseline: The trivial baseline's metric.
        ceiling: The measured or analytic ceiling, or ``None`` if unmeasured.
        threshold: The declared or restated bar.
        direction: Which way the metric improves.
        ceiling_type: What kind of comparator the ceiling is.  Required
            whenever ``ceiling`` is not ``None``; forbidden when it is.  Only
            a bound-type comparator (:data:`BOUND_COMPARATOR_TYPES`) can
            produce :data:`GateVerdict.UNREACHABLE` or
            :data:`GateVerdict.BRACKETED_BOUND`.

    Raises:
        ValueError: On a measured ceiling with an undeclared type, or a type
            declared for a missing ceiling — both fail closed rather than
            guessing the comparator's strength.
    """
    resolved = _direction(direction)
    if gate_passes(baseline, threshold, resolved):
        return GateVerdict.VACUOUS
    if ceiling is None:
        if ceiling_type is not None:
            raise ValueError(
                f"ceiling_type {ceiling_type!r} was declared for a missing "
                "ceiling; a type without a measurement classifies nothing"
            )
        return GateVerdict.UNAUDITED
    resolved_type = _comparator_type(ceiling_type)
    is_bound = resolved_type in BOUND_COMPARATOR_TYPES
    if not gate_passes(ceiling, threshold, resolved):
        return GateVerdict.UNREACHABLE if is_bound else GateVerdict.ABOVE_REFERENCE
    return GateVerdict.BRACKETED_BOUND if is_bound else GateVerdict.BRACKETED_REFERENCE


def evaluate_gate(
    *,
    baseline: object,
    model: object,
    ceiling: object | None,
    threshold: object,
    direction: Direction,
    ceiling_type: ComparatorType | str | None = None,
    min_effect: object = 0.0,
) -> GateEvaluation:
    """Evaluate placement and model outcome without conflating the two.

    ``min_effect`` is a prospective smallest effect of interest on the metric's
    own scale: the outcome is a pass only when the margin strictly exceeds it.
    The default of ``0.0`` reproduces the bare strict comparison and is what the
    eight retrospective audited rows use — each of their thresholds was restated
    after measurement, so a nonzero ``min_effect`` there would be as
    retrospective as the threshold. A *new* gate should declare a nonzero
    ``min_effect`` before its population is scored (pre-build checklist, and the
    external audit's "beats baseline by enough to matter" requirement); where
    the margin is a statistic, the recorder's paired test at the
    family-corrected level carries the uncertainty half of that requirement.
    """
    resolved = _direction(direction)
    floor = _finite_metric("min_effect", min_effect)
    if floor < 0.0:
        raise ValueError(f"min_effect must be non-negative, got {min_effect!r}")
    margin = gate_margin(model, threshold, resolved)
    return GateEvaluation(
        verdict=classify_gate(baseline, ceiling, threshold, resolved, ceiling_type=ceiling_type),
        passes=margin > floor,
        margin=margin,
    )


__all__ = [
    "AUDITED_GATE_IDS",
    "BOUND_COMPARATOR_TYPES",
    "BRACKETED_VERDICTS",
    "SMALLEST_EFFECT_SIZES",
    "ComparatorType",
    "Direction",
    "GateEvaluation",
    "GateId",
    "GateVerdict",
    "SmallestEffectSize",
    "classify_gate",
    "effect_size_is_of_interest",
    "evaluate_gate",
    "gate_is_baseline_relative",
    "gate_margin",
    "gate_passes",
    "smallest_effect_size",
    "verdict_is_bracketed",
]
