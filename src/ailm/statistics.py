"""Executable multiplicity, exact-power, and confidence-interval contracts.

The repository-wide significance family is explicit here rather than described
approximately in prose.  In particular, the full-set Stage 2 McNemar result is
listed once: ``subsample_full_mcnemar_p`` is a second label for the same paired
comparison and must not consume a second family slot.

Interval convention
-------------------

Several results in this project are null or near-null.  "We found no effect" is
weaker than the data support: what the data support is a *bound*.  The
convention below is stated once here, executable, rather than restated
approximately at each null.

Every paired comparison here is exact and conditional.  Two scales are reported
and they answer different questions:

- The **discordant share** ``b / (b + c)`` — the probability that a discordant
  pair falls the probe's way.  Its interval is a Clopper-Pearson interval at
  ``SHARE_CONFIDENCE`` (the ``stat_*_share_ci_*`` labels), or at
  ``CORRECTED_CONFIDENCE`` when the interval must pair with the
  family-corrected decision threshold.  ``0.5`` is the null on this scale.
- The **paired difference** ``(b - c) / n`` — the probe's accuracy minus the
  comparator's on the same positions.  Its interval is the share interval
  mapped back through ``delta = (2 * share - 1) * (b + c) / n``, at
  ``DIFFERENCE_CONFIDENCE``.  ``0`` is the null on this scale.
  ``ailm.benchmarks.paired_difference_interval`` is the single implementation.

What an interval licenses.  It bounds the effect: values outside it are
inconsistent with the data at the stated coverage.  An interval that contains
the null value bounds the effect; it does **not** establish that the effect is
absent.  "The difference is between -0.03 and +0.04" is a claim about how large
the effect can be.  "There is no difference" is a different claim, and no
interval that contains the null ever supports it.

What an interval does not license.  Coverage is *marginal*, not simultaneous:
these intervals carry no multiplicity correction unless taken at
``CORRECTED_CONFIDENCE``.  A ``DIFFERENCE_CONFIDENCE`` interval that excludes
the null is therefore **not** equivalent to clearing ``BONFERRONI_ALPHA`` — the
Stage 3 high-mixing-weight arm is exactly that case in this project, so reading
"excludes zero" as "significant" would contradict the multiplicity discipline
the same module declares.  Conditioning on the discordant set is what makes the
construction exact, and the price is that it is conservative: with few
discordant pairs the interval is wide, and a wide interval is a weak bound, not
a strong null.

Equivalence.  Turning a bound into an equivalence claim needs a
smallest-effect-of-interest, and the project already declares one:
``REFERENCE_SHARE``.  See ``equivalence_region`` for the argument that it is the
right anchor on the share scale and the wrong one on the difference scale.

Independence across pairs.  Every construction in this module is exact
*conditional on independent matched pairs*.  That condition fails wherever the
scored observations nest inside a larger sampling unit — G2P positions inside
held-out words, Stage 4 tokens inside sentences, Stage 6 sentences inside
documents — and under positive intracluster correlation the exact test is
anti-conservative rather than conservative: the p-value is too small and the
interval too narrow.  ``ailm.clustering`` recomputes the same effects with the
cluster as the resampling unit.  It leaves the effect untouched by construction
and moves only the uncertainty, so a cluster reanalysis can retire a small
p-value without retiring the measurement it came from.  The labels produced here
remain the record of the position-level analysis; they are not replaced by the
cluster-aware ones, because the two answer questions about different units and a
reader needs both to see what the correction did.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from scipy.stats import beta, binom, binomtest

FAMILYWISE_ALPHA = 0.01

#: Frozen selection parameters for the Stage 2 subsample-bias reanalysis.
#:
#: These are method configuration, not measured results.  They live beside the
#: executable multiplicity family because that family names the three resulting
#: reanalyses.  The recorder, figure labels, and manuscript tokens all import
#: these constants so a changed selection rule cannot leave an old stride in a
#: caption or plot label.
SUBSAMPLE_PREFIX_SIZE = 1200
SUBSAMPLE_MATCHED_ARM = "stride5"
SUBSAMPLE_LARGER_ARM = "stride3"
SUBSAMPLE_STRIDE_BY_ARM: dict[str, int] = {
    SUBSAMPLE_MATCHED_ARM: 5,
    SUBSAMPLE_LARGER_ARM: 3,
}
for _subsample_arm, _subsample_stride in SUBSAMPLE_STRIDE_BY_ARM.items():
    if _subsample_arm != f"stride{_subsample_stride}":
        raise RuntimeError(
            "subsample arm identity must encode its frozen stride; changing a "
            "selection rule requires new claim labels rather than overwriting an old arm"
        )
SUBSAMPLE_MATCHED_STRIDE = SUBSAMPLE_STRIDE_BY_ARM[SUBSAMPLE_MATCHED_ARM]
SUBSAMPLE_LARGER_STRIDE = SUBSAMPLE_STRIDE_BY_ARM[SUBSAMPLE_LARGER_ARM]

SIGNIFICANCE_COUNT_LABELS: tuple[tuple[str, str, str], ...] = (
    # The channel factorial (decision 46) re-measures the shipped sweeps as its
    # diagonal cells: letter-at-evidence restates the hypothesis behind
    # ``coupling_mcnemar_p``, and lexicon-at-predictive restates the one behind
    # ``lexical_best_mcnemar_p``. Each searches the factorial's own weight grid
    # over the same evaluated positions against the same uncoupled anchor, so a
    # diagonal cell is one of those hypotheses asked again under a different
    # selection rule rather than a new question. The family counts hypotheses,
    # not weight choices, so listing a diagonal cell would count one hypothesis
    # twice and tighten every other threshold on the strength of a weight choice.
    # Only the off-diagonal cells, which ask something neither sweep asked, join.
    #
    # Decision 117 is the basis, and it is checked rather than asserted.
    # :data:`SHARED_HYPOTHESIS_EXCLUSIONS` below carries the invariants each
    # exclusion rests on; ``EXCLUDED_FROM_FAMILY`` in tests/test_statistics.py
    # carries the label set, and
    # ``test_shared_hypothesis_exclusions_hold_their_ledger_invariants`` checks
    # the two against the ledger. An earlier form of this comment justified the
    # exclusion by a numerical identity between each cell and its member.
    # Decision 110's re-recording made that false for the lexical cell and
    # nothing failed, because no consumer reads a reason string.
    (
        "channel_letter_predictive_paired_p",
        "channel_letter_predictive_paired_b",
        "channel_letter_predictive_paired_c",
    ),
    (
        "channel_lexicon_evidence_paired_p",
        "channel_lexicon_evidence_paired_b",
        "channel_lexicon_evidence_paired_c",
    ),
    # Decision 47's two substantive arms. These are hypotheses the paper rests
    # on, not diagnostics: "the lexical message helps under a rule that cannot
    # dilute" and "the letter message still harms under it" are the sentences
    # that vindicate decision 18, and a claim the manuscript makes is a claim
    # that gets corrected. Their two control arms are excluded instead, with
    # reasons, in tests/test_statistics.py.
    ("fitted_paired_p", "fitted_paired_b", "fitted_paired_c"),
    ("geometric_letter_paired_p", "geometric_letter_paired_b", "geometric_letter_paired_c"),
    (
        "geometric_lexicon_paired_p",
        "geometric_lexicon_paired_b",
        "geometric_lexicon_paired_c",
    ),
    ("coupling_mcnemar_p", "coupling_mcnemar_b", "coupling_mcnemar_c"),
    ("lexical_best_mcnemar_p", "lexical_best_mcnemar_b", "lexical_best_mcnemar_c"),
    ("stage2_g2p_mcnemar_p", "stage2_g2p_mcnemar_b", "stage2_g2p_mcnemar_c"),
    (
        "stage2_g2p_subsample_mcnemar_p",
        "stage2_g2p_subsample_mcnemar_b",
        "stage2_g2p_subsample_mcnemar_c",
    ),
    ("stage4_derived_mcnemar_p", "stage4_derived_mcnemar_b", "stage4_derived_mcnemar_c"),
    ("stage4_gold_mcnemar_p", "stage4_gold_mcnemar_b", "stage4_gold_mcnemar_c"),
    ("stage5_mcnemar_p", "stage5_mcnemar_b", "stage5_mcnemar_c"),
    (
        "stage6_sign_test_p",
        "stage6_sign_test_positive",
        "stage6_sign_test_negative",
    ),
    ("subsample_prefix_mcnemar_p", "subsample_prefix_mcnemar_b", "subsample_prefix_mcnemar_c"),
    (
        "subsample_stride3_mcnemar_p",
        "subsample_stride3_mcnemar_b",
        "subsample_stride3_mcnemar_c",
    ),
    (
        "subsample_stride5_mcnemar_p",
        "subsample_stride5_mcnemar_b",
        "subsample_stride5_mcnemar_c",
    ),
)

#: The p-value family and its sufficient-count registry are one object, not two
#: lists that can drift. Every family decision can therefore be re-derived from
#: integer counts rather than trusting a rounded scalar in the claims ledger.
SIGNIFICANCE_FAMILY: tuple[str, ...] = tuple(
    p_label for p_label, _favored_label, _opposed_label in SIGNIFICANCE_COUNT_LABELS
)

#: Family members introduced only after the alphabetical-prefix bias was seen.
#: This inventory is method history, not an inference from a name prefix: the
#: prefix, stride-3, and stride-5 re-analyses are the post-inspection members the
#: manuscript discloses.  The manuscript derives its count from this registry so
#: family growth cannot leave the old natural-language tally behind.
RETROSPECTIVE_POSTHOC_MEMBERS: tuple[str, ...] = (
    "subsample_prefix_mcnemar_p",
    "subsample_stride3_mcnemar_p",
    "subsample_stride5_mcnemar_p",
)
if len(RETROSPECTIVE_POSTHOC_MEMBERS) != len(set(RETROSPECTIVE_POSTHOC_MEMBERS)) or not set(
    RETROSPECTIVE_POSTHOC_MEMBERS
) <= set(SIGNIFICANCE_FAMILY):
    raise RuntimeError(
        "retrospective post-inspection members must be unique members of SIGNIFICANCE_FAMILY"
    )

BONFERRONI_ALPHA = FAMILYWISE_ALPHA / len(SIGNIFICANCE_FAMILY)


# ── Why a p-value may sit outside the family (decision 117) ──────────────────


class ExclusionBasis(StrEnum):
    """Why an excluded p-value does not consume a family slot."""

    #: The excluded label reproduces its counterpart to every recorded digit,
    #: because the selected weight lies on both grids.
    EXACT_IDENTITY = "exact-identity"
    #: The excluded label restates its counterpart's hypothesis on the same
    #: population, searched over a different weight grid, so the recorded values
    #: differ while the question does not.
    SECOND_SELECTION_GRID = "second-selection-grid"


@dataclass(frozen=True, slots=True)
class SharedHypothesisExclusion:
    """One excluded p-value that restates a family member's hypothesis.

    The channel factorial re-measures the shipped sweeps as its diagonal cells,
    searching its own weight grid over the same evaluated positions against the
    same uncoupled anchor. That makes each diagonal cell the same *hypothesis* as
    a family member, asked again under a different selection rule, so admitting it
    would count one hypothesis twice and tighten every other threshold on the
    strength of a weight choice.

    This object is the declared basis, not a description of it. It lives here
    rather than beside the test that checks it so the manuscript figures, the
    prose generators, and the test all read one object; decision 117 exists
    because the previous basis was a sentence in three places, one of which
    stopped being true without anything failing.

    Attributes:
        excluded: The p-value label kept out of the family.
        family_member: The declared family member whose hypothesis it restates.
        basis: Which of the two exclusion arguments applies.
        channel_key: Its entry in :mod:`ailm.channel_registry`. Both sides name
            this one channel; the registry is default-deny, so a renamed or
            retired channel fails rather than passing quietly.
        insertion: The receiver site that registry entry declares.
        positions: ``(excluded side, family side)`` evaluated-position labels,
            which the exclusion requires to be equal.
        anchor: ``(excluded side, family side)`` anchor-accuracy labels, which the
            exclusion requires to be equal.
        identical: Label pairs that must additionally match exactly. Non-empty
            exactly when ``basis`` is :data:`ExclusionBasis.EXACT_IDENTITY`.
        differs: Label pairs that must NOT match, recording where the grids are
            known to select apart. Non-empty exactly when ``basis`` is
            :data:`ExclusionBasis.SECOND_SELECTION_GRID`. This is the executable
            form of decision 117's rejected alternative: re-recording a cell over
            the other grid so a caption becomes true again would make these equal.
    """

    excluded: str
    family_member: str
    basis: ExclusionBasis
    channel_key: str
    insertion: str
    positions: tuple[str, str]
    anchor: tuple[str, str]
    identical: tuple[tuple[str, str], ...]
    differs: tuple[tuple[str, str], ...]


#: Decision 117. Until then this pair was justified by prose asserting a
#: numerical identity, which decision 110's re-recording made false for the
#: lexical cell. No consumer evaluates a reason string, so the exclusion stayed
#: correct while its justification decayed and no gate noticed.
#: ``tests/test_statistics.py`` checks every field below against the ledger.
SHARED_HYPOTHESIS_EXCLUSIONS: tuple[SharedHypothesisExclusion, ...] = (
    SharedHypothesisExclusion(
        excluded="channel_letter_evidence_paired_p",
        family_member="coupling_mcnemar_p",
        basis=ExclusionBasis.EXACT_IDENTITY,
        channel_key="letter_coupling",
        insertion="observation likelihood",
        positions=("channel_factorial_positions", "coupling_eval_positions"),
        anchor=("channel_factorial_anchor_accuracy", "coupling_alpha0_accuracy"),
        # The coupling sweep declines monotonically, so its argmax over engaged
        # weights is the smallest positive weight on the grid, and both grids
        # carry the same smallest positive weight. The older identity claim
        # therefore still holds here, and is asserted rather than described.
        identical=(
            ("channel_letter_evidence_best_alpha", "coupling_best_alpha"),
            ("channel_letter_evidence_paired_b", "coupling_mcnemar_b"),
            ("channel_letter_evidence_paired_c", "coupling_mcnemar_c"),
            ("channel_letter_evidence_paired_p", "coupling_mcnemar_p"),
        ),
        differs=(),
    ),
    SharedHypothesisExclusion(
        excluded="channel_lexicon_predictive_paired_p",
        family_member="lexical_best_mcnemar_p",
        basis=ExclusionBasis.SECOND_SELECTION_GRID,
        channel_key="lexical",
        insertion="phone predictive",
        positions=("channel_factorial_positions", "lexical_eval_positions"),
        anchor=("channel_factorial_anchor_accuracy", "lexical_anchor_accuracy"),
        # No identity here. The corrected lexical sweep rises to a plateau, the
        # factorial's grid reaches a weight the sweep's grid does not carry, and
        # the two searches select apart.
        identical=(),
        differs=(("channel_lexicon_predictive_best_alpha", "lexical_best_alpha"),),
    ),
)
if len({record.excluded for record in SHARED_HYPOTHESIS_EXCLUSIONS}) != len(
    SHARED_HYPOTHESIS_EXCLUSIONS
) or not {record.family_member for record in SHARED_HYPOTHESIS_EXCLUSIONS} <= set(
    SIGNIFICANCE_FAMILY
):
    raise RuntimeError(
        "every shared-hypothesis exclusion must name a distinct excluded label and a "
        "counterpart that is actually in SIGNIFICANCE_FAMILY; an exclusion that "
        "restates a non-member argues for nothing"
    )

#: The pre-registered confirmatory replication protocol (decision 79), frozen in
#: code so the family cannot be quietly re-shaped between now and the run. Every
#: member of :data:`SIGNIFICANCE_FAMILY` appears exactly once, mapped either to
#: a population that could re-score it without touching data this project has
#: already consumed or to the explicit ``split_diagnostic`` non-rescoring
#: disposition:
#:
#: - ``ud_ewt_test``: re-vendor ``en_ewt-ud-test.conllu`` through the identical
#:   generator and oracle, score the frozen dev-trained parameters once over the
#:   whole of it. No family member has been scored against that split; decision
#:   91 records the replication's execution and retirement, and the status of
#:   these members is an open principal decision recorded in TODO.md.
#: - ``external_lexicon``: the vendored CMUdict artifact holds no unscored
#:   material — the interleave's train side fitted the model and its test side
#:   is fully scored, and decision 74's family-disjoint re-partition has also
#:   been scored — so replication needs an independently sourced pronunciation
#:   lexicon (the registered candidate: entries added upstream to CMUdict since
#:   the vendored revision, filtered identically).
#: - ``split_diagnostic``: the member is a re-measurement of a specific shipped
#:   subsample and has no analogue on a fresh population; the full-set member it
#:   re-analyzes is what the confirmatory run re-scores.
#:
#: Endpoint, test and sidedness are frozen with each rescored member: the same
#: paired endpoint each label records today, the exact two-sided conditional
#: test (McNemar / sign), judged at :data:`BONFERRONI_ALPHA` for the frozen
#: family size — one run, no re-selection of weights, thresholds or populations
#: after the new data are seen.
CONFIRMATORY_POPULATIONS: tuple[str, ...] = (
    "ud_ewt_test",
    "external_lexicon",
)

#: Explicit protocol dispositions that are not populations and cannot be
#: re-scored. Keeping this separate prevents prose and downstream code from
#: treating an analysis of the old split as held-back data.
CONFIRMATORY_NON_RESCORING_DISPOSITIONS: tuple[str, ...] = ("split_diagnostic",)

#: Every legal right-hand side in :data:`CONFIRMATORY_REPLICATION`.
CONFIRMATORY_DISPOSITIONS: tuple[str, ...] = (
    *CONFIRMATORY_POPULATIONS,
    *CONFIRMATORY_NON_RESCORING_DISPOSITIONS,
)

CONFIRMATORY_REPLICATION: tuple[tuple[str, str], ...] = (
    ("channel_letter_predictive_paired_p", "external_lexicon"),
    ("channel_lexicon_evidence_paired_p", "external_lexicon"),
    ("fitted_paired_p", "external_lexicon"),
    ("geometric_letter_paired_p", "external_lexicon"),
    ("geometric_lexicon_paired_p", "external_lexicon"),
    ("coupling_mcnemar_p", "external_lexicon"),
    ("lexical_best_mcnemar_p", "external_lexicon"),
    ("stage2_g2p_mcnemar_p", "external_lexicon"),
    ("stage2_g2p_subsample_mcnemar_p", "split_diagnostic"),
    ("stage4_derived_mcnemar_p", "ud_ewt_test"),
    ("stage4_gold_mcnemar_p", "ud_ewt_test"),
    ("stage5_mcnemar_p", "ud_ewt_test"),
    ("stage6_sign_test_p", "ud_ewt_test"),
    ("subsample_prefix_mcnemar_p", "split_diagnostic"),
    ("subsample_stride3_mcnemar_p", "split_diagnostic"),
    ("subsample_stride5_mcnemar_p", "split_diagnostic"),
)

_confirmatory_labels = tuple(label for label, _disposition in CONFIRMATORY_REPLICATION)
if len(_confirmatory_labels) != len(set(_confirmatory_labels)) or set(_confirmatory_labels) != set(
    SIGNIFICANCE_FAMILY
):
    raise RuntimeError(
        "CONFIRMATORY_REPLICATION must map every significance-family member exactly once"
    )
if set(CONFIRMATORY_POPULATIONS) & set(CONFIRMATORY_NON_RESCORING_DISPOSITIONS):
    raise RuntimeError("confirmatory populations and non-rescoring dispositions must be disjoint")
if any(
    disposition not in CONFIRMATORY_DISPOSITIONS for _label, disposition in CONFIRMATORY_REPLICATION
):
    raise RuntimeError("CONFIRMATORY_REPLICATION contains an undeclared disposition")

#: Reference favored share for retrospective power: the smallest paired effect
#: the project would consider scientifically interesting. Power at the
#: *observed* share is a monotone transform of the p-value (Hoenig & Heisey
#: 2001) and adds no independent evidence, so the reported detectability
#: diagnostic is computed at this fixed share instead.
REFERENCE_SHARE = 0.55

#: Two-sided coverage of every *paired difference* interval, matching the
#: default of ``ailm.benchmarks.paired_difference_interval`` that produced the
#: recorded ``*_delta_low_*`` / ``*_delta_high_*`` claims. A derived bound at any
#: other level would silently be a different quantity from the recorded ones it
#: sits beside in the same table.
DIFFERENCE_CONFIDENCE = 0.95

#: Two-sided coverage of every *discordant share* interval. Harmonized onto
#: :data:`DIFFERENCE_CONFIDENCE` per decision 37, so the two families of interval
#: now read at one level.
#:
#: They previously differed at 0.99 against 0.95, for no reason beyond having been
#: introduced by different recorders, and they sat backwards on their roles: the
#: stricter level went to the primary effects, which are already judged against
#: ``BONFERRONI_ALPHA`` and do not need their interval to carry extra coverage
#: too, while the looser went to the nulls, where the interval *is* the whole
#: claim. They are also one interval and not two: ``paired_difference_interval``
#: maps this Clopper-Pearson share interval through an affine rescaling, which
#: preserves coverage exactly.
#:
#: Kept as a distinct name rather than deleted, because the *scales* remain
#: distinct even when the level does not. A share lives in [0, 1] and a paired
#: difference in [-1, 1]; collapsing the names would invite reading one as the
#: other. This alias is what makes the harmonization checkable: any future
#: divergence has to be written here, deliberately, instead of accumulating.
SHARE_CONFIDENCE = DIFFERENCE_CONFIDENCE

#: Coverage of the interval that pairs with the family-corrected decision
#: threshold. An interval must be read at the level of the decision it supports,
#: so a result judged against ``BONFERRONI_ALPHA`` gets its bound here.
CORRECTED_CONFIDENCE = 1.0 - BONFERRONI_ALPHA

#: How far past a candidate ``n`` the equivalence-pair search must keep holding.
#: The predicate is not monotone in ``n``: an odd pair count cannot split evenly,
#: so its most balanced split is off-center and its interval is wider than the
#: even count below it. Even counts satisfy the region from 402 while odd counts
#: do not until 421, and returning 402 would report a parity artifact as a
#: sample-size requirement. The window must exceed 1 to see the neighboring
#: parity at all; 32 is comfortably past the last gap.
EQUIVALENCE_STABILITY_WINDOW = 32


@dataclass(frozen=True, slots=True)
class ExactBinomialSummary:
    """Exact-binomial precision summary for one paired effect."""

    favored: int
    opposed: int
    ratio: float
    ci_low: float
    ci_high: float
    power: float
    power_at_reference: float

    @property
    def n(self) -> int:
        """Number of discordant pairs."""
        return self.favored + self.opposed


def _validate_count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _validate_probability(name: str, value: object, *, open_interval: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite probability, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {value!r}")
    valid = 0.0 < numeric < 1.0 if open_interval else 0.0 <= numeric <= 1.0
    if not valid:
        interval = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"{name} is outside {interval}: got {value!r}")
    return numeric


@lru_cache(maxsize=None)
def exact_two_sided_rejection_region(n: int, alpha: float) -> tuple[int, int]:
    """Return inclusive lower/upper critical counts under ``Bin(n, 0.5)``.

    The null is symmetric, so scipy's probability-ordered two-sided exact test
    reduces to twice the lower-tail mass. The returned region is ``x <= lower``
    or ``x >= upper``. This project declares threshold crossing strictly as
    ``p < alpha`` in its Bonferroni and Holm decisions, so an exact p-value equal
    to ``alpha`` is outside the rejection region here too.
    """
    count = _validate_count("n", n)
    level = _validate_probability("alpha", alpha, open_interval=True)
    if count == 0:
        raise ValueError("n must be positive")
    candidates = [k for k in range(count // 2 + 1) if 2.0 * float(binom.cdf(k, count, 0.5)) < level]
    if not candidates:
        return -1, count + 1
    lower = max(candidates)
    return lower, count - lower


def exact_binomial_power(n: int, alternative_p: float, alpha: float) -> float:
    """Power of the exact two-sided sign/McNemar test at ``alternative_p``."""
    count = _validate_count("n", n)
    probability = _validate_probability("alternative_p", alternative_p, open_interval=False)
    lower, upper = exact_two_sided_rejection_region(count, alpha)
    if lower < 0:
        return 0.0
    return float(binom.cdf(lower, count, probability) + binom.sf(upper - 1, count, probability))


def exact_two_sided_binomial_p(favored: int, opposed: int) -> float:
    """Exact two-sided sign/McNemar p-value from sufficient counts.

    Zero discordance has no direction to condition on and therefore returns
    ``1.0``. Keeping this count-level operation beside the power calculation
    gives recorders one implementation for both the stored p-value and the
    decision reconstructed from its counts.
    """
    wins = _validate_count("favored", favored)
    losses = _validate_count("opposed", opposed)
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    return float(binomtest(wins, discordant, 0.5).pvalue)


def clopper_pearson_interval(
    successes: int, trials: int, *, confidence: float = SHARE_CONFIDENCE
) -> tuple[float, float]:
    """Conservative exact binomial confidence interval at the project level."""
    success = _validate_count("successes", successes)
    count = _validate_count("trials", trials)
    level = _validate_probability("confidence", confidence, open_interval=True)
    if count == 0:
        raise ValueError("trials must be positive")
    if success > count:
        raise ValueError(f"successes ({success}) cannot exceed trials ({count})")
    tail = (1.0 - level) / 2.0
    low = 0.0 if success == 0 else float(beta.ppf(tail, success, count - success + 1))
    high = 1.0 if success == count else float(beta.ppf(1.0 - tail, success + 1, count - success))
    return low, high


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Originally lived in ``ailm.benchmarks.scoring``; moved here so the generic
    interval primitives are reachable without importing ``ailm.benchmarks``.
    ``ailm.benchmarks.scoring.wilson_interval`` re-exports this exact function.
    """
    success_count = _validate_count("successes", successes)
    total_count = _validate_count("total", total)
    if isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(z) or z <= 0.0:
        raise ValueError(f"z must be a positive finite number, got {z!r}")
    critical_value = float(z)
    if success_count > total_count:
        raise ValueError(f"successes ({success_count}) cannot exceed total ({total_count})")
    if total_count == 0:
        return (0.0, 0.0)
    phat = success_count / total_count
    denom = 1 + critical_value * critical_value / total_count
    center = (phat + critical_value * critical_value / (2 * total_count)) / denom
    margin = (
        critical_value
        * math.sqrt(
            phat * (1 - phat) / total_count
            + critical_value * critical_value / (4 * total_count * total_count)
        )
        / denom
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def paired_difference_interval(
    b: int, c: int, n: int, *, confidence: float = DIFFERENCE_CONFIDENCE
) -> tuple[float, float]:
    """Exact conditional interval for the paired accuracy difference.

    The two per-group Wilson intervals answer a different question than the
    paired comparison does: they can overlap widely while every discordant pair
    runs one way. This is the interval on the quantity the paired test is about,
    ``(b - c) / n`` — the probe's accuracy minus the comparator's on the same
    positions.

    Construction is the standard exact conditional one: condition on the
    ``b + c`` discordant pairs, take a Clopper-Pearson interval on the share
    ``b / (b + c)``, and map it back with ``delta = (2 * share - 1) * (b + c) / n``.
    Conditioning is what makes it exact; the price is that it is conservative.

    Originally lived in ``ailm.benchmarks.scoring``; moved here so the generic
    interval primitives are reachable without importing ``ailm.benchmarks``.
    ``ailm.benchmarks.scoring.paired_difference_interval`` re-exports this exact
    function, and this is the single implementation referenced throughout this
    module's docstrings as ``paired_difference_interval``.

    Args:
        b: Positions the probe got right and the comparator got wrong.
        c: Positions the comparator got right and the probe got wrong.
        n: Total paired positions (the denominator of the difference).
        confidence: Two-sided coverage of the interval.

    Returns:
        ``(low, high)`` on the paired difference, probe minus comparator.
    """
    wins = _validate_count("b", b)
    losses = _validate_count("c", c)
    total = _validate_count("n", n)
    if total <= 0:
        raise ValueError("n must be positive")
    discordant = wins + losses
    if discordant > total:
        raise ValueError(f"discordant pairs ({discordant}) cannot exceed n ({total})")
    if discordant == 0:
        return (0.0, 0.0)
    low, high = clopper_pearson_interval(wins, discordant, confidence=confidence)
    scale = discordant / total
    return ((2.0 * low - 1.0) * scale, (2.0 * high - 1.0) * scale)


#: Two-sided significance level for :func:`paired_power`'s retrospective power
#: calculation. This is a duplicate literal, not an import, of
#: ``ailm.benchmarks.constants.POWER_ALPHA`` (also ``0.05``): that constant is
#: benchmark configuration read by recorders and manuscript variables, and
#: importing it here would give this module the ``ailm.benchmarks`` dependency
#: it exists to avoid. If the benchmark family's alpha ever moves, update both.
_PAIRED_POWER_ALPHA = 0.05


def paired_power(
    b: int, c: int, *, alpha: float = _PAIRED_POWER_ALPHA, reference_share: float = REFERENCE_SHARE
) -> float:
    """Retrospective power of the paired exact test at the recorded pair count.

    Mirrors ``stat_*_power_at_reference``: power is evaluated at a *fixed*
    reference share rather than at the observed one, because power at the
    observed effect is a monotone transform of the p-value (Hoenig & Heisey
    2001) and carries no evidence the p-value does not already carry. The
    honest question this answers is "if the true discordant split were
    ``reference_share``, would this many pairs have found it?".

    Originally lived in ``ailm.benchmarks.scoring``; moved here so the generic
    power primitive is reachable without importing ``ailm.benchmarks``.
    ``ailm.benchmarks.scoring.paired_power`` re-exports this exact function.
    """
    discordant = _validate_count("b", b) + _validate_count("c", c)
    share = _validate_probability("reference_share", reference_share, open_interval=False)
    return 0.0 if discordant == 0 else exact_binomial_power(discordant, share, alpha)


def summarize_paired_effect(
    favored: int,
    opposed: int,
    *,
    alpha: float = BONFERRONI_ALPHA,
    confidence: float = SHARE_CONFIDENCE,
) -> ExactBinomialSummary:
    """Summarize discordant pairs in the direction named by the caller."""
    wins = _validate_count("favored", favored)
    losses = _validate_count("opposed", opposed)
    total = wins + losses
    if total == 0:
        raise ValueError("at least one discordant pair is required")
    ratio = wins / total
    low, high = clopper_pearson_interval(wins, total, confidence=confidence)
    return ExactBinomialSummary(
        favored=wins,
        opposed=losses,
        ratio=ratio,
        ci_low=low,
        ci_high=high,
        power=exact_binomial_power(total, ratio, alpha),
        power_at_reference=exact_binomial_power(total, REFERENCE_SHARE, alpha),
    )


def equivalence_region(reference_share: float = REFERENCE_SHARE) -> tuple[float, float]:
    """Region of discordant shares too small to be scientifically interesting.

    An equivalence claim needs a smallest-effect-of-interest declared before the
    data are seen, and this project already declares one: ``REFERENCE_SHARE``,
    "the smallest paired effect the project would consider scientifically
    interesting". The region is that share and its mirror, so it is symmetric
    about the null at ``0.5`` and covers effects in either direction.

    Why the share scale and not the difference scale. ``REFERENCE_SHARE`` was
    declared as a share, and the paired test is conditional on the discordant
    set, so the share is the scale the test is actually about. Converting it to
    a difference requires ``delta = (2 * s - 1) * (b + c) / n``, whose second
    factor is the *observed* discordance rate. A smallest-effect-of-interest
    that depends on the data is no longer pre-declared, and the dependence runs
    the wrong way: the fewer discordant pairs an arm produced, the smaller the
    converted threshold becomes, so equivalence would be easiest to claim
    exactly where the evidence is thinnest. That is the defect this function
    exists to avoid, and it is why no difference-scale equivalence threshold is
    offered here. A difference-scale equivalence claim would need its own
    pre-declared margin, which this project has not declared.

    Args:
        reference_share: Smallest interesting favored share, in ``(0.5, 1)``.

    Returns:
        ``(low, high)`` bounds of the equivalence region on the share scale.

    Raises:
        ValueError: If the share is not strictly above the null. A reference at
            or below ``0.5`` would make the region empty or inverted, and every
            arm would silently fail to be equivalent for a reason having nothing
            to do with its data.
    """
    share = _validate_probability("reference_share", reference_share, open_interval=True)
    if share <= 0.5:
        raise ValueError(
            f"reference_share must exceed 0.5 to bound an effect in both directions, got {share!r}"
        )
    return 1.0 - share, share


def share_interval_supports_equivalence(
    favored: int,
    opposed: int,
    *,
    confidence: float = DIFFERENCE_CONFIDENCE,
    reference_share: float = REFERENCE_SHARE,
) -> bool:
    """Whether the share interval lies wholly inside the equivalence region.

    This is the two-one-sided-tests decision, read off the interval: the effect
    is bounded away from every share the project called interesting, in both
    directions. It is the one condition under which a null in this project may
    be stated as an equivalence rather than as a bound.

    Containment must be checked at both ends. Checking only that the interval
    contains ``0.5`` would pass every arm with too few pairs to say anything —
    the defect that makes an underpowered null look like a positive finding.

    Args:
        favored: Discordant pairs in the probe's direction.
        opposed: Discordant pairs the other way.
        confidence: Two-sided coverage of the share interval.
        reference_share: Smallest interesting favored share.

    Returns:
        ``True`` only if the whole interval sits inside the equivalence region.
    """
    total = _validate_count("favored", favored) + _validate_count("opposed", opposed)
    if total == 0:
        raise ValueError("at least one discordant pair is required")
    low, high = clopper_pearson_interval(favored, total, confidence=confidence)
    region_low, region_high = equivalence_region(reference_share)
    return low >= region_low and high <= region_high


@lru_cache(maxsize=None)
def minimum_discordant_pairs_for_equivalence(
    *,
    confidence: float = DIFFERENCE_CONFIDENCE,
    reference_share: float = REFERENCE_SHARE,
    limit: int = 20000,
) -> int:
    """Smallest pair count from which an equivalence claim is ever reachable.

    Analytic in the sense that matters: it uses no measurement. It asks, of the
    *most favorable* possible outcome at each pair count — the split closest to
    even, which is where the interval is narrowest — how many pairs are needed
    before that interval fits inside the equivalence region. Any arm below this
    count cannot support an equivalence claim no matter how its pairs fell, so
    the number is what separates "this null is only a bound" from "this null
    could have been an equivalence and was not".

    The search requires the predicate to keep holding for
    ``EQUIVALENCE_STABILITY_WINDOW`` further counts, because it is not monotone:
    see that constant for the parity artifact this rules out.

    Args:
        confidence: Two-sided coverage of the share interval.
        reference_share: Smallest interesting favored share.
        limit: Highest pair count to search before giving up.

    Returns:
        The smallest stable pair count.

    Raises:
        ValueError: If no count under ``limit`` qualifies. Silently returning
            ``limit`` would report the search bound as a scientific requirement.
    """

    maximum = _validate_count("limit", limit)
    if maximum < 2 + EQUIVALENCE_STABILITY_WINDOW:
        raise ValueError(
            "limit must leave room for the complete stability window: "
            f"need at least {2 + EQUIVALENCE_STABILITY_WINDOW}, got {maximum}"
        )

    def reachable(pairs: int) -> bool:
        return share_interval_supports_equivalence(
            pairs // 2, pairs - pairs // 2, confidence=confidence, reference_share=reference_share
        )

    for pairs in range(2, maximum - EQUIVALENCE_STABILITY_WINDOW + 1):
        if reachable(pairs) and all(
            reachable(pairs + step) for step in range(1, EQUIVALENCE_STABILITY_WINDOW + 1)
        ):
            return pairs
    raise ValueError(
        f"no complete stable window at or below {maximum} admits an equivalence claim at "
        f"confidence={confidence!r} and reference_share={reference_share!r}"
    )


@dataclass(frozen=True, slots=True)
class HolmDecision:
    """One family member's Holm outcome: its rank, its threshold, its verdict."""

    label: str
    p_value: float
    rank: int
    threshold: float
    rejected: bool


def holm_step_down(
    p_values: dict[str, float],
    alpha: float = FAMILYWISE_ALPHA,
) -> tuple[HolmDecision, ...]:
    """Apply Holm's step-down correction to a family of p-values.

    Holm controls the same familywise error rate as Bonferroni under the same
    arbitrary dependence, and it rejects whenever Bonferroni does and sometimes
    more, so it dominates the fixed threshold rather than trading against it.
    Sort ascending; compare the ``i``-th smallest against ``alpha / (m - i)``;
    stop at the first failure and retain every hypothesis from there on.

    The step-down structure is why the fixed threshold is kept beside it. Holm's
    threshold for a given hypothesis depends on where every *other* p-value in
    the family fell, so it is not a constant a regression test can assert
    against, and a member can change verdict because a different measurement
    moved. ``BONFERRONI_ALPHA`` is a constant and stays the automated gate; Holm
    is the more informative scientific reading of the same family.

    Ties are handled by enforcing monotonicity of the rejection decision: once
    the step-down halts, no later hypothesis is rejected, so equal p-values
    always receive the same verdict regardless of sort order.
    """
    if not p_values:
        raise ValueError("Holm correction needs a non-empty family")
    level = _validate_probability("alpha", alpha, open_interval=True)
    validated: dict[str, float] = {}
    for label, value in p_values.items():
        if not isinstance(label, str) or not label:
            raise ValueError(f"p-value labels must be non-empty strings, got {label!r}")
        validated[label] = _validate_probability(
            f"p-value for {label!r}", value, open_interval=False
        )

    size = len(validated)
    ordered = sorted(validated.items(), key=lambda item: (item[1], item[0]))
    decisions: list[HolmDecision] = []
    halted = False
    for index, (label, p_value) in enumerate(ordered):
        threshold = level / (size - index)
        if not halted and p_value >= threshold:
            halted = True
        decisions.append(
            HolmDecision(
                label=label,
                p_value=p_value,
                rank=index + 1,
                threshold=threshold,
                rejected=not halted,
            )
        )
    return tuple(decisions)


__all__ = [
    "BONFERRONI_ALPHA",
    "CONFIRMATORY_DISPOSITIONS",
    "CONFIRMATORY_NON_RESCORING_DISPOSITIONS",
    "CONFIRMATORY_POPULATIONS",
    "CONFIRMATORY_REPLICATION",
    "CORRECTED_CONFIDENCE",
    "DIFFERENCE_CONFIDENCE",
    "EQUIVALENCE_STABILITY_WINDOW",
    "FAMILYWISE_ALPHA",
    "REFERENCE_SHARE",
    "RETROSPECTIVE_POSTHOC_MEMBERS",
    "SHARED_HYPOTHESIS_EXCLUSIONS",
    "SHARE_CONFIDENCE",
    "SIGNIFICANCE_COUNT_LABELS",
    "SIGNIFICANCE_FAMILY",
    "ExactBinomialSummary",
    "ExclusionBasis",
    "HolmDecision",
    "SharedHypothesisExclusion",
    "clopper_pearson_interval",
    "equivalence_region",
    "exact_binomial_power",
    "exact_two_sided_binomial_p",
    "exact_two_sided_rejection_region",
    "holm_step_down",
    "minimum_discordant_pairs_for_equivalence",
    "paired_difference_interval",
    "paired_power",
    "share_interval_supports_equivalence",
    "summarize_paired_effect",
    "wilson_interval",
]
