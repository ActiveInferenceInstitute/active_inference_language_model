"""Exact statistical contracts and the repository-wide multiplicity family."""

from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path

import pytest
from scipy.stats import binomtest

from ailm.benchmarks.scoring import exact_mcnemar_p
from ailm.benchmarks.synthetic_protocol import SYNTHETIC_INCONTEXT_PROTOCOL
from ailm.claims import load_ledger
from ailm.manuscript_variables import (
    generate_manuscript_variables,
    paired_difference_bound_from_counts,
)
from ailm.statistics import (
    BONFERRONI_ALPHA,
    CONFIRMATORY_DISPOSITIONS,
    CONFIRMATORY_NON_RESCORING_DISPOSITIONS,
    CONFIRMATORY_POPULATIONS,
    CONFIRMATORY_REPLICATION,
    CORRECTED_CONFIDENCE,
    DIFFERENCE_CONFIDENCE,
    EQUIVALENCE_STABILITY_WINDOW,
    FAMILYWISE_ALPHA,
    REFERENCE_SHARE,
    SHARE_CONFIDENCE,
    SHARED_HYPOTHESIS_EXCLUSIONS,
    SIGNIFICANCE_COUNT_LABELS,
    SIGNIFICANCE_FAMILY,
    ExclusionBasis,
    clopper_pearson_interval,
    equivalence_region,
    exact_binomial_power,
    exact_two_sided_binomial_p,
    exact_two_sided_rejection_region,
    holm_step_down,
    minimum_discordant_pairs_for_equivalence,
    share_interval_supports_equivalence,
    summarize_paired_effect,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_statistical_recorder():
    path = PROJECT_ROOT / "scripts" / "record_statistical_claims.py"
    spec = importlib.util.spec_from_file_location("_statistical_recorder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ledger_entries() -> dict[str, dict[str, object]]:
    payload = json.loads((PROJECT_ROOT / "data" / "claims.json").read_text(encoding="utf-8"))
    return payload["claims"]


def test_family_is_explicit_unique_and_includes_stage6():
    assert len(SIGNIFICANCE_FAMILY) == len(set(SIGNIFICANCE_FAMILY)) == 16
    assert "stage6_sign_test_p" in SIGNIFICANCE_FAMILY
    assert "stage2_g2p_mcnemar_p" in SIGNIFICANCE_FAMILY
    assert "subsample_full_mcnemar_p" not in SIGNIFICANCE_FAMILY
    assert BONFERRONI_ALPHA == FAMILYWISE_ALPHA / len(SIGNIFICANCE_FAMILY)


def test_family_labels_are_all_ledger_backed():
    ledger = load_ledger(PROJECT_ROOT)
    assert [label for label in SIGNIFICANCE_FAMILY if ledger.get(label) is None] == []


def test_every_family_member_has_integer_sufficient_counts_and_reconstructs():
    """The scalar p-value is an auditable projection, never the authority."""
    entries = _ledger_entries()
    recorder = _load_statistical_recorder()
    reconstructed = recorder.reconstruct_family_p_values(entries)

    assert tuple(p for p, _b, _c in SIGNIFICANCE_COUNT_LABELS) == SIGNIFICANCE_FAMILY
    assert set(reconstructed) == set(SIGNIFICANCE_FAMILY)
    for p_label, b_label, c_label in SIGNIFICANCE_COUNT_LABELS:
        b = entries[b_label]["value"]
        c = entries[c_label]["value"]
        assert type(b) is int and type(c) is int
        assert reconstructed[p_label] == exact_two_sided_binomial_p(b, c)


def test_family_reconstruction_fails_when_a_stored_p_or_count_drifts():
    """NEGATIVE CONTROL: a plausible-looking scalar cannot override its counts."""
    recorder = _load_statistical_recorder()
    entries = json.loads(json.dumps(_ledger_entries()))
    entries["stage2_g2p_mcnemar_p"]["value"] = 0.5
    with pytest.raises(SystemExit, match="disagrees with exact reconstruction"):
        recorder.reconstruct_family_p_values(entries)

    entries = json.loads(json.dumps(_ledger_entries()))
    entries["stage2_g2p_mcnemar_b"]["value"] = 1379.0
    with pytest.raises(SystemExit, match="non-negative integer"):
        recorder.reconstruct_family_p_values(entries)


def test_the_confirmatory_replication_protocol_covers_the_family_exactly():
    """Decision 79's frozen protocol: every member once, every disposition legal.

    The pre-registration's whole value is that the family cannot be re-shaped
    between registration and the run. A member missing from the protocol could
    be quietly dropped before replication; a member appearing twice could be
    scored on whichever population came out better; an undeclared disposition
    could smuggle in data or silently reclassify a diagnostic as a replication.
    """
    protocol_labels = [label for label, _population in CONFIRMATORY_REPLICATION]
    assert sorted(protocol_labels) == sorted(SIGNIFICANCE_FAMILY)
    assert len(protocol_labels) == len(set(protocol_labels))
    assert set(CONFIRMATORY_POPULATIONS) == {"ud_ewt_test", "external_lexicon"}
    assert set(CONFIRMATORY_NON_RESCORING_DISPOSITIONS) == {"split_diagnostic"}
    assert set(CONFIRMATORY_POPULATIONS).isdisjoint(CONFIRMATORY_NON_RESCORING_DISPOSITIONS)
    assert set(CONFIRMATORY_DISPOSITIONS) == {
        "ud_ewt_test",
        "external_lexicon",
        "split_diagnostic",
    }
    for label, disposition in CONFIRMATORY_REPLICATION:
        assert disposition in CONFIRMATORY_DISPOSITIONS, (label, disposition)

    # The feasibility split is itself a registered fact: the four UD EWT members
    # are the only ones a held-back partition exists for without new sourcing,
    # and the split diagnostics are exactly the four subsample re-analyses.
    by_disposition = {
        disposition: sorted(
            label for label, target in CONFIRMATORY_REPLICATION if target == disposition
        )
        for disposition in CONFIRMATORY_DISPOSITIONS
    }
    assert by_disposition["ud_ewt_test"] == [
        "stage4_derived_mcnemar_p",
        "stage4_gold_mcnemar_p",
        "stage5_mcnemar_p",
        "stage6_sign_test_p",
    ]
    assert by_disposition["split_diagnostic"] == [
        "stage2_g2p_subsample_mcnemar_p",
        "subsample_prefix_mcnemar_p",
        "subsample_stride3_mcnemar_p",
        "subsample_stride5_mcnemar_p",
    ]
    assert len(by_disposition["external_lexicon"]) == 8


def test_the_family_survivors_are_exactly_these_and_stage6_is_not_one():
    """The survivor set by exact equality, never by count.

    A count is satisfied by any set of that size, and the identity of what
    survives a familywise correction is the whole claim. The set grew when the
    channel factorial (decision 46) added two cells: the corrected threshold
    tightened from 0.01/11 to 0.01/13, and the lexicon-at-evidence *harm* cleared
    it. That harm was an artifact of the lexical backoff defect corrected in
    decision 110; under the documented rule the cell is a null and left the
    set. A member leaving a familywise survivor set because its measurement was
    wrong is a result, not a bookkeeping change, which is why it is named here
    rather than absorbed into a number.
    """
    p_values = _load_statistical_recorder().reconstruct_family_p_values(_ledger_entries())
    survivors = {label for label in SIGNIFICANCE_FAMILY if p_values[label] < BONFERRONI_ALPHA}
    assert p_values["channel_lexicon_evidence_paired_p"] > BONFERRONI_ALPHA
    assert survivors == {
        "coupling_mcnemar_p",
        "fitted_paired_p",
        "geometric_lexicon_paired_p",
        "lexical_best_mcnemar_p",
        "stage2_g2p_mcnemar_p",
        "stage4_derived_mcnemar_p",
        "subsample_prefix_mcnemar_p",
    }
    assert p_values["stage6_sign_test_p"] > BONFERRONI_ALPHA


def test_holm_dominates_bonferroni_on_the_real_family():
    """Holm rejects everything Bonferroni does, and on this ledger nothing more.

    Named by identity rather than by count, for the reason the Bonferroni
    survivor test gives: a count is satisfied by any set of that size, and which
    hypothesis a correction admits is the whole claim. Before decision 110 the
    one member Holm gained was the Stage 6 sign test, whose two-sided p sat
    between the fixed threshold and its own step-down bar. Re-ranking the family
    after the lexicon-at-evidence member left the survivor set tightened that
    bar past the p-value, so the step-down now admits exactly the fixed set. The
    dominance assertion is the theorem (formal/AILM/Holm.lean); the identity
    assertion is the recorded ledger, and it is pinned so a future re-record
    that moves the Stage 6 test back across its bar changes this test on
    purpose rather than the prose by accident.
    """
    p_values = _load_statistical_recorder().reconstruct_family_p_values(_ledger_entries())
    decisions = holm_step_down(p_values, FAMILYWISE_ALPHA)
    holm = {decision.label for decision in decisions if decision.rejected}
    bonferroni = {label for label, value in p_values.items() if value < BONFERRONI_ALPHA}
    assert bonferroni <= holm, "Holm must never reject less than Bonferroni"
    assert holm - bonferroni == set()
    stage6 = next(d for d in decisions if d.label == "stage6_sign_test_p")
    assert not stage6.rejected and stage6.p_value > stage6.threshold


def test_holm_thresholds_step_down_from_the_family_size():
    """Rank i of m is compared against alpha / (m - i + 1)."""
    p_values = {"a": 0.001, "b": 0.004, "c": 0.5}
    decisions = holm_step_down(p_values, 0.05)
    assert [decision.rank for decision in decisions] == [1, 2, 3]
    assert [round(decision.threshold, 10) for decision in decisions] == [
        round(0.05 / 3, 10),
        round(0.05 / 2, 10),
        round(0.05 / 1, 10),
    ]
    # The smallest threshold is Bonferroni's, so the two agree on rank 1.
    assert decisions[0].threshold == 0.05 / len(p_values)


def test_holm_halts_and_stays_halted():
    """Negative control: a hypothesis past the first failure is never rejected.

    The halt is what separates Holm from comparing each rank against its own
    threshold independently. Here rank 2 fails at 0.03 against 0.025, and rank 3
    at 0.04 would otherwise clear its threshold of 0.05 and be reported as
    significant despite a larger p-value than the one that stopped the
    procedure.
    """
    decisions = holm_step_down({"a": 0.001, "b": 0.03, "c": 0.04}, 0.05)
    verdicts = {decision.label: decision.rejected for decision in decisions}
    assert verdicts == {"a": True, "b": False, "c": False}
    halted = next(decision for decision in decisions if decision.label == "c")
    assert halted.p_value < halted.threshold and not halted.rejected


def test_holm_gives_tied_p_values_the_same_verdict():
    """Ties cannot be split by sort order."""
    decisions = holm_step_down({"a": 0.02, "b": 0.02, "c": 0.001}, 0.05)
    tied = [decision.rejected for decision in decisions if decision.p_value == 0.02]
    assert len(tied) == 2
    assert len(set(tied)) == 1


def test_holm_rejects_a_degenerate_family_or_p_value():
    with pytest.raises(ValueError, match="non-empty"):
        holm_step_down({}, 0.05)
    with pytest.raises(ValueError, match="outside"):
        holm_step_down({"a": 1.5}, 0.05)
    with pytest.raises(ValueError, match="alpha"):
        holm_step_down({"a": 0.01}, 1.0)
    for invalid in ("0.01", True, float("nan")):
        with pytest.raises(ValueError, match="p-value"):
            holm_step_down({"a": invalid}, 0.05)  # type: ignore[dict-item]


def test_exact_region_is_symmetric_and_rejects_analytic_small_case():
    lower, upper = exact_two_sided_rejection_region(10, 0.05)
    assert (lower, upper) == (1, 9)
    assert lower + upper == 10


def test_power_uses_the_same_strict_boundary_as_every_decision_rule():
    """A p-value equal to alpha is not rejected anywhere in the project."""
    boundary = exact_two_sided_binomial_p(1, 9)
    assert boundary == 22 / 1024
    assert exact_two_sided_rejection_region(10, boundary) == (0, 10)
    just_above = math.nextafter(boundary, math.inf)
    assert exact_two_sided_rejection_region(10, just_above) == (1, 9)
    assert exact_binomial_power(10, 0.7, boundary) < exact_binomial_power(10, 0.7, just_above)
    [decision] = holm_step_down({"boundary": boundary}, boundary)
    assert not decision.rejected


@pytest.mark.parametrize(
    ("favored", "opposed"),
    [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (2, 7),
        (7, 2),
        (12, 12),
        (18, 30),
        (434, 371),
        (1379, 1145),
    ],
)
def test_exact_mcnemar_wrapper_matches_the_central_scipy_contract(favored: int, opposed: int):
    """The benchmark API must remain a transparent count-level wrapper.

    The asymmetric, tied, degenerate, and shipped-count cases make this a live
    parity check rather than a single happy-path example.
    """
    expected = (
        1.0 if favored + opposed == 0 else float(binomtest(favored, favored + opposed, 0.5).pvalue)
    )
    central = exact_two_sided_binomial_p(favored, opposed)
    assert central == expected
    assert exact_mcnemar_p(favored, opposed) == central


def test_power_is_symmetric_about_the_null():
    assert exact_binomial_power(50, 0.7, 0.01) == pytest.approx(
        exact_binomial_power(50, 0.3, 0.01), abs=1e-14
    )
    assert exact_binomial_power(50, 0.5, 0.01) <= 0.01


def test_clopper_pearson_boundary_cases_are_total():
    low0, high0 = clopper_pearson_interval(0, 10)
    low1, high1 = clopper_pearson_interval(10, 10)
    assert low0 == 0.0 < high0 < 1.0
    assert 0.0 < low1 < high1 == 1.0


def test_clopper_pearson_default_is_the_declared_share_confidence():
    assert clopper_pearson_interval(7, 12) == clopper_pearson_interval(
        7, 12, confidence=SHARE_CONFIDENCE
    )


def test_real_stage2_summary_matches_the_documented_anchor():
    """A regression anchor on the summary, pinned so it cannot move with the family.

    The interval and ratio are properties of the counts alone. Power is not: it is
    evaluated at whatever alpha the caller passes, which defaults to
    ``BONFERRONI_ALPHA`` and therefore moves every time the project declares
    another hypothesis. This test hardcoded the default-alpha value and broke
    twice in one session as the family grew 11 to 13 to 15 — each time reporting a
    failure in a file that had not changed, about a function that was correct.

    So alpha is passed explicitly. The anchor now tests ``summarize_paired_effect``
    rather than the current size of the multiplicity family, and a genuine
    regression in the power computation still fails it. The family-dependent value
    is asserted separately, and derived rather than pinned.
    """
    fixed_alpha = 0.01 / 11
    # Confidence is passed explicitly for the same reason alpha is, and it broke
    # the same way: the interval is a property of the counts *at a fixed
    # coverage*, and the coverage is a project-wide constant that decision 37
    # moved from 0.99 to 0.95. Pinning the 0.99 numbers while reading the level
    # from a global made this anchor a test of the global. Passing it makes the
    # anchor test `summarize_paired_effect` again, and a genuine regression in
    # the interval still fails it.
    fixed_confidence = 0.99
    summary = summarize_paired_effect(1379, 1145, alpha=fixed_alpha, confidence=fixed_confidence)
    assert summary.n == 2524
    assert summary.ratio == pytest.approx(0.5463549921)
    assert summary.ci_low == pytest.approx(0.5205631295)
    assert summary.ci_high == pytest.approx(0.5719704221)
    assert summary.power == pytest.approx(0.9097030518)


def test_power_at_the_declared_threshold_falls_as_the_family_grows():
    """The family-dependent reading, derived from the declaration.

    Tightening the corrected threshold buys less power on the same counts. That
    relationship is the thing worth asserting; the particular number it currently
    produces is not, because it is a fact about how many hypotheses the project
    has declared rather than about the data.
    """
    counts = (1379, 1145)
    loose = summarize_paired_effect(*counts, alpha=0.01 / 11)
    current = summarize_paired_effect(*counts, alpha=BONFERRONI_ALPHA)
    # Deliberately NOT compared against a second call with the same arguments:
    # that asserts a pure function is pure, which is true of every possible
    # implementation including a wrong one. The signal is the relationship
    # below, between two DIFFERENT alphas.
    assert current.power <= loose.power, (
        f"power at the current corrected threshold ({current.power}) exceeds power at the "
        f"looser historical one ({loose.power}); a tighter threshold cannot buy more power"
    )


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((-1, 5), "non-negative"),
        ((1, 0), "positive"),
        ((11, 10), "cannot exceed"),
    ],
)
def test_interval_rejects_impossible_counts(args, match):
    with pytest.raises(ValueError, match=match):
        clopper_pearson_interval(*args)


def test_summary_requires_a_discordant_pair():
    with pytest.raises(ValueError, match="discordant"):
        summarize_paired_effect(0, 0)


# ── The equivalence anchor ───────────────────────────────────────────────────


def test_equivalence_region_is_symmetric_about_the_null():
    low, high = equivalence_region()
    assert high == REFERENCE_SHARE
    assert low + high == pytest.approx(1.0, abs=1e-15)


def test_equivalence_region_rejects_an_anchor_at_or_below_the_null():
    """A reference share at 0.5 makes the region empty, not permissive.

    Every arm would then fail to be equivalent for a reason having nothing to do
    with its data, and the failure would read as a scientific finding.
    """
    with pytest.raises(ValueError, match="exceed 0.5"):
        equivalence_region(0.5)
    with pytest.raises(ValueError, match="exceed 0.5"):
        equivalence_region(0.4)


def test_equivalence_verdict_is_not_constant_in_either_direction():
    """POSITIVE CONTROL: a stub returning one verdict would pass a one-sided check.

    A function hard-wired to ``False`` would satisfy every underpowered arm in
    this project, and one hard-wired to ``True`` would satisfy the assertion
    that the real Stage 3 high-weight arm is equivalent. Both directions are
    pinned: many balanced pairs are equivalent, few pairs are not, and many
    lopsided pairs are not.
    """
    assert share_interval_supports_equivalence(2000, 2000)
    assert not share_interval_supports_equivalence(5, 5)
    assert not share_interval_supports_equivalence(3000, 1000)


def test_minimum_pairs_is_the_first_stable_count_not_the_first_hit():
    """POSITIVE CONTROL for the parity guard, which is otherwise invisible.

    The predicate is not monotone: an odd pair count cannot split evenly, so its
    most balanced split is off-center and its interval is wider than the even
    count below it. A search returning the first count that happens to satisfy
    the region would report an even-count artifact — 402 rather than 420 — as a
    sample-size requirement, and every assertion about the returned value would
    still pass. This pins that such a count exists and is not what is returned.
    """
    minimum = minimum_discordant_pairs_for_equivalence()

    def reachable(pairs: int) -> bool:
        return share_interval_supports_equivalence(pairs // 2, pairs - pairs // 2)

    assert reachable(minimum)
    assert not reachable(minimum - 1), "the returned count is not the first stable one"
    assert all(reachable(minimum + step) for step in range(1, EQUIVALENCE_STABILITY_WINDOW + 1))
    earlier_hits = [pairs for pairs in range(2, minimum) if reachable(pairs)]
    assert earlier_hits, (
        "no count below the minimum satisfies the region in isolation, so the "
        "stability window is doing nothing and this guard is vacuous"
    )


def test_equivalence_search_never_evaluates_past_its_declared_limit():
    minimum = minimum_discordant_pairs_for_equivalence()
    incomplete_limit = minimum + EQUIVALENCE_STABILITY_WINDOW - 1
    with pytest.raises(ValueError, match="no complete stable window"):
        minimum_discordant_pairs_for_equivalence(limit=incomplete_limit)
    assert (
        minimum_discordant_pairs_for_equivalence(limit=minimum + EQUIVALENCE_STABILITY_WINDOW)
        == minimum
    )


def test_a_stricter_level_needs_more_pairs():
    """Sanity that the requirement responds to the level it is asked about."""
    assert minimum_discordant_pairs_for_equivalence(
        confidence=CORRECTED_CONFIDENCE
    ) > minimum_discordant_pairs_for_equivalence(confidence=DIFFERENCE_CONFIDENCE)


#: Paired arms whose bound the recorder wrote, as ``(arm prefix, n label)``.
#: These are the only arms where the derived and recorded paths can be compared,
#: and agreement here is what licenses the derived bounds elsewhere.
_RECORDED_BOUND_ARMS: tuple[tuple[str, str], ...] = (
    ("bench_letter", "bench_letter_positions"),
    ("bench_phoneme", "bench_phoneme_positions"),
    ("bench_g2p", "bench_g2p_words"),
    ("bench_letter_prompt", "bench_letter_positions"),
)


def _load_bench_recorder():
    """Import the external-benchmark recorder as a module, without running it."""
    import importlib.util

    path = PROJECT_ROOT / "scripts" / "record_external_benchmark_claims.py"
    spec = importlib.util.spec_from_file_location("_bench_recorder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The served probes, read from the recorder rather than restated. A literal copy
#: here is the drift this file already paid for once: the exclusion table went
#: asymmetric across probes and every check stayed green, because each check was
#: comparing one hand-maintained list against another.
_BENCH_MODELS: tuple[str, ...] = tuple(sorted(_load_bench_recorder().MODELS))

#: Derived arms whose denominator is checked against a separately recorded point
#: estimate: ``(name, b, c, n, control terms, scale, tolerance, why)``. The
#: control is ``minuend - subtrahend``, on the same scale as ``(b - c) / n``.
_DENOMINATOR_CONTROLS: tuple[tuple[str, str, str, str, str, str, float, float], ...] = (
    (
        "stage5 clause classification",
        "stage5_mcnemar_b",
        "stage5_mcnemar_c",
        "stage5_test_sentences",
        "stage5_classification_pct",
        "stage5_majority_baseline_pct",
        100.0,
        # Both accuracies are recorded rounded to two decimals, so their
        # difference carries up to 0.01 of rounding from either side.
        0.02,
    ),
    (
        "stage3 high mixing weight",
        "lexical_sweep_alpha_0_9_mcnemar_b",
        "lexical_sweep_alpha_0_9_mcnemar_c",
        "lexical_eval_positions",
        "lexical_sweep_alpha_0_9_accuracy",
        "lexical_anchor_accuracy",
        100.0,
        0.002,
    ),
    (
        "letter context window, gemma3",
        "bench_letter_ctx_paired_b_gemma3",
        "bench_letter_ctx_paired_c_gemma3",
        "bench_letter_positions",
        "bench_letter_long_top1_gemma3",
        "bench_letter_short_top1_gemma3",
        1.0,
        1e-12,
    ),
    (
        "letter context window, qwen25",
        "bench_letter_ctx_paired_b_qwen25",
        "bench_letter_ctx_paired_c_qwen25",
        "bench_letter_positions",
        "bench_letter_long_top1_qwen25",
        "bench_letter_short_top1_qwen25",
        1.0,
        1e-12,
    ),
)

#: Arms whose point estimate is checked against a single recorded quantity
#: rather than a difference of two accuracies, as
#: ``(name, b, c, n, recorded, scale, tolerance)``.
#:
#: These seven were reported as a margin and a *p*-value and nothing else until
#: the derived bounds were added, which is the ordering a reviewer flagged:
#: whether an effect cleared a threshold was legible, how large it could be was
#: not. They share one denominator, so a wrong choice would rescale all seven
#: intervals at once while each stayed individually plausible. Each is therefore
#: pinned to a margin recorded by a different code path.
_SINGLE_LABEL_DENOMINATORS: tuple[tuple[str, str, str, str, str, float, float], ...] = (
    (
        "fitted channel",
        "fitted_paired_b",
        "fitted_paired_c",
        "fitted_positions",
        "fitted_gain_pts",
        100.0,
        # The recorded margin carries four decimals, so half a unit in the last
        # place from either side.
        1e-4,
    ),
    (
        "geometric pooling, lexical",
        "geometric_lexicon_paired_b",
        "geometric_lexicon_paired_c",
        "channel_factorial_positions",
        "geometric_lexicon_best_margin_pts",
        100.0,
        1e-4,
    ),
    (
        "geometric pooling, letter",
        "geometric_letter_paired_b",
        "geometric_letter_paired_c",
        "channel_factorial_positions",
        "geometric_letter_best_margin_pts",
        100.0,
        1e-4,
    ),
    (
        "factorial, letter into evidence",
        "channel_letter_evidence_paired_b",
        "channel_letter_evidence_paired_c",
        "channel_factorial_positions",
        "channel_letter_evidence_best_margin_pts",
        100.0,
        1e-4,
    ),
    (
        "factorial, letter into predictive",
        "channel_letter_predictive_paired_b",
        "channel_letter_predictive_paired_c",
        "channel_factorial_positions",
        "channel_letter_predictive_best_margin_pts",
        100.0,
        1e-4,
    ),
    (
        "factorial, lexicon into evidence",
        "channel_lexicon_evidence_paired_b",
        "channel_lexicon_evidence_paired_c",
        "channel_factorial_positions",
        "channel_lexicon_evidence_best_margin_pts",
        100.0,
        1e-4,
    ),
    (
        "factorial, lexicon into predictive",
        "channel_lexicon_predictive_paired_b",
        "channel_lexicon_predictive_paired_c",
        "channel_factorial_positions",
        "channel_lexicon_predictive_best_margin_pts",
        100.0,
        1e-4,
    ),
)

#: Derived arms with no tied pairs, where the denominator is checked by
#: exhaustion instead: every paired unit is discordant, so ``b + c`` must equal
#: ``n`` exactly. A denominator off by anything at all fails this.
_EXHAUSTIVE_DENOMINATORS: tuple[tuple[str, str, str, str], ...] = (
    (
        "stage2 G2P conditional direction",
        "stage2_g2p_mcnemar_b",
        "stage2_g2p_mcnemar_c",
        "stat_stage2_discordant_n",
    ),
    (
        "stage6 narrative sign test",
        "stage6_sign_test_positive",
        "stage6_sign_test_negative",
        "stage6_test_sentences",
    ),
)

#: Token names this module's derivation block writes. Listed here rather than
#: scraped so the collision check reads a declaration, not an inference.
_DERIVED_BOUND_TOKENS: tuple[str, ...] = (
    "BOUND_CONFIDENCE_PCT",
    "BOUND_CORRECTED_CONFIDENCE_PCT",
    "EQUIVALENCE_REGION_LOW",
    "EQUIVALENCE_REGION_HIGH",
    "EQUIVALENCE_REFERENCE_SHARE",
    "EQUIVALENCE_MIN_DISCORDANT_PAIRS",
    "EQUIVALENCE_MIN_DISCORDANT_PAIRS_CORRECTED",
    "STAGE5_CLASSIFICATION_DELTA_PTS",
    "STAGE5_CLASSIFICATION_DELTA_LOW_PTS",
    "STAGE5_CLASSIFICATION_DELTA_HIGH_PTS",
    "STAGE5_CLASSIFICATION_DISCORDANT",
    "STAGE6_SIGN_NET_SHARE",
    "STAGE6_SIGN_NET_SHARE_LOW",
    "STAGE6_SIGN_NET_SHARE_HIGH",
    "STAGE6_SIGN_DISCORDANT",
    "STAGE2_G2P_DISCORDANT_NET_SHARE",
    "STAGE2_G2P_DISCORDANT_NET_SHARE_LOW",
    "STAGE2_G2P_DISCORDANT_NET_SHARE_HIGH",
    "LEXICAL_HIGH_ALPHA_DELTA_LOW_PTS",
    "LEXICAL_HIGH_ALPHA_DELTA_HIGH_PTS",
    "LEXICAL_HIGH_ALPHA_DISCORDANT",
    "LEXICAL_HIGH_ALPHA_EQUIVALENT",
    "BENCH_LETTER_CTX_DELTA_GEMMA3",
    "BENCH_LETTER_CTX_DELTA_QWEN25",
    "BENCH_LETTER_CTX_DELTA_LOW_GEMMA3",
    "BENCH_LETTER_CTX_DELTA_HIGH_GEMMA3",
    "BENCH_LETTER_CTX_DELTA_LOW_QWEN25",
    "BENCH_LETTER_CTX_DELTA_HIGH_QWEN25",
    "BENCH_LETTER_CTX_DISCORDANT_GEMMA3",
    "BENCH_LETTER_CTX_DISCORDANT_QWEN25",
)


def _claim_value(label: str) -> float:
    entry = load_ledger(PROJECT_ROOT).get(label)
    assert entry is not None, f"the ledger has no claim {label!r}"
    return float(entry["value"])


class TestDerivedPairedBound:
    """A bound derived at render time must be the recorder's, not a lookalike.

    Several results here are nulls, and a null stated as "we found no effect" is
    weaker than the data support: what the data support is an interval bounding
    how large the effect can be. The recorder wrote ``*_delta_low_*`` /
    ``*_delta_high_*`` for the four external-benchmark arms and for no other
    paired comparison, so every null in the project could state its contrast but
    not its bound. The inputs are recorded, so the bounds are derived rather
    than remeasured.

    The defect that makes this worth guarding is a derived interval that merely
    *looks* like a recorded one. A different confidence level, a Wald interval
    instead of the exact conditional one, or a wrong denominator would each
    produce a plausible pair of numbers, render without error, and put two
    different kinds of interval in one table under one name. The equivalence
    test below rules that out by running the derivation on the arms that *do*
    have a recorded bound and demanding it reproduce them.
    """

    @pytest.mark.parametrize("arm_and_n", _RECORDED_BOUND_ARMS, ids=lambda pair: pair[0])
    @pytest.mark.parametrize("model", _BENCH_MODELS)
    def test_derivation_reproduces_every_recorded_bound(self, arm_and_n, model):
        """EQUIVALENCE PROOF: derived path == recorder path wherever both exist.

        Every (arm, probe) pair is the only place the two paths can be compared; the count follows from the recorder's own MODELS, so registering a probe extends it rather than dating this sentence. A
        failure here does not mean the tolerance is too tight; it means the
        recorder builds its interval from something other than ``b``, ``c`` and
        ``n``, and deriving a bound from those three alone is the wrong
        derivation for this project.
        """
        arm, n_label = arm_and_n
        low, high = paired_difference_bound_from_counts(
            _claim_value(f"{arm}_paired_b_{model}"),
            _claim_value(f"{arm}_paired_c_{model}"),
            _claim_value(n_label),
        )
        assert low == pytest.approx(_claim_value(f"{arm}_delta_low_{model}"), rel=0, abs=1e-12)
        assert high == pytest.approx(_claim_value(f"{arm}_delta_high_{model}"), rel=0, abs=1e-12)

    @pytest.mark.parametrize("control", _DENOMINATOR_CONTROLS, ids=lambda row: row[0])
    def test_denominator_reproduces_a_separately_recorded_point_estimate(self, control):
        """The one input the counts cannot check about themselves.

        ``b`` and ``c`` are read from labels that name the arm, so reading the
        wrong one is loud. ``n`` is read from a different label entirely, and a
        plausible wrong choice — test positions instead of test sentences, the
        full pool instead of the subsample — rescales every bound in the block
        while leaving each number looking reasonable. Each arm's ``(b - c) / n``
        is therefore checked against an accuracy difference recorded by a
        different code path.
        """
        _name, b_label, c_label, n_label, minuend, subtrahend, scale, tolerance = control
        b, c, n = (_claim_value(b_label), _claim_value(c_label), _claim_value(n_label))
        # The production wrapper enforces this during rendering; repeat it here
        # to diagnose the exact labels if the shipped ledger is inconsistent.
        # Discordant pairs exceeding the denominator means the two labels
        # describe different position sets and every bound built from them is wrong.
        assert b + c <= n, f"{b_label} + {c_label} exceeds {n_label}"
        derived = scale * (b - c) / n
        recorded = _claim_value(minuend) - _claim_value(subtrahend)
        assert derived == pytest.approx(recorded, rel=0, abs=tolerance)

    @pytest.mark.parametrize("control", _SINGLE_LABEL_DENOMINATORS, ids=lambda row: row[0])
    def test_single_label_denominators_reproduce_their_recorded_margin(self, control):
        """Same check, for arms pinned to one recorded margin instead of two."""
        _name, b_label, c_label, n_label, recorded_label, scale, tolerance = control
        b, c, n = (_claim_value(b_label), _claim_value(c_label), _claim_value(n_label))
        assert b + c <= n, f"{b_label} + {c_label} exceeds {n_label}"
        derived = scale * (b - c) / n
        assert derived == pytest.approx(_claim_value(recorded_label), rel=0, abs=tolerance)

    @pytest.mark.parametrize("control", _SINGLE_LABEL_DENOMINATORS, ids=lambda row: row[0])
    def test_single_label_bounds_contain_their_point_estimate(self, control):
        _name, b_label, c_label, n_label, _recorded, _scale, _tolerance = control
        b, c, n = (_claim_value(b_label), _claim_value(c_label), _claim_value(n_label))
        low, high = paired_difference_bound_from_counts(b, c, n)
        assert low <= (b - c) / n <= high

    def test_the_one_factorial_cell_spanning_zero_is_the_one_that_missed_its_threshold(
        self,
    ):
        """The channel-factorial caption's claim, checked rather than asserted.

        The caption says the cells whose intervals span zero are exactly the
        cells whose *p* does not reach significance, and names them: the
        letter-at-predictive cell and, since the lexical backoff correction of
        decision 110 turned its recorded harm into a null, the
        lexicon-at-evidence cell. That is a statement about recorded data and it
        can go stale; if a re-measurement moved either the counts or the
        p-values, the sentence would quietly become false while every number in
        the table stayed correct.
        """
        cells = (
            "channel_letter_evidence",
            "channel_letter_predictive",
            "channel_lexicon_evidence",
            "channel_lexicon_predictive",
        )
        n = _claim_value("channel_factorial_positions")
        spans_zero, misses_threshold = set(), set()
        for cell in cells:
            low, high = paired_difference_bound_from_counts(
                _claim_value(f"{cell}_paired_b"), _claim_value(f"{cell}_paired_c"), n
            )
            if low <= 0.0 <= high:
                spans_zero.add(cell)
            if _claim_value(f"{cell}_paired_p") >= 0.05:
                misses_threshold.add(cell)
        assert spans_zero == misses_threshold, (
            f"cells spanning zero {sorted(spans_zero)} no longer coincide with cells "
            f"missing the threshold {sorted(misses_threshold)}; the caption of "
            "[@tbl:channel_factorial] says they do"
        )
        assert spans_zero == {"channel_letter_predictive", "channel_lexicon_evidence"}, (
            f"the caption names the letter-at-predictive and lexicon-at-evidence cells "
            f"as the ones spanning zero, found {sorted(spans_zero)}"
        )

    @pytest.mark.parametrize("control", _EXHAUSTIVE_DENOMINATORS, ids=lambda row: row[0])
    def test_exhaustive_denominators_are_accounted_for_exactly(self, control):
        """These arms have no ties, so the counts must exhaust the denominator."""
        _name, b_label, c_label, n_label = control
        assert _claim_value(b_label) + _claim_value(c_label) == _claim_value(n_label)

    def test_the_denominator_control_could_fail(self):
        """POSITIVE CONTROL: the denominator is load-bearing in the bound.

        If the interval barely moved with ``n``, the control above would pass
        against any denominator and would be checking nothing. Doubling ``n``
        must halve the bound, because the conditional share is mapped back
        through the discordance rate.
        """
        low, high = paired_difference_bound_from_counts(27, 24, 400)
        wide_low, wide_high = paired_difference_bound_from_counts(27, 24, 800)
        assert wide_low == pytest.approx(low / 2, rel=1e-12)
        assert wide_high == pytest.approx(high / 2, rel=1e-12)

    def test_the_bound_is_not_vacuous(self):
        """POSITIVE CONTROL: a stub returning a fixed wide interval passes little else.

        A derivation hard-wired to ``(-1, 1)`` would contain every point
        estimate and satisfy any assertion that only checked containment. Three
        properties rule it out: the bound is narrower than the whole range, it
        moves with its inputs, and more discordant pairs at the same share buy a
        narrower bound.
        """
        bound = paired_difference_bound_from_counts(27, 24, 400)
        assert -1.0 < bound[0] < bound[1] < 1.0
        assert bound != paired_difference_bound_from_counts(6, 14, 400)
        narrow = paired_difference_bound_from_counts(200, 200, 4000)
        wide = paired_difference_bound_from_counts(20, 20, 400)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    @pytest.mark.parametrize("control", _DENOMINATOR_CONTROLS, ids=lambda row: row[0])
    def test_every_derived_bound_contains_its_point_estimate(self, control):
        """An interval that excludes its own estimate is built the wrong way round."""
        _name, b_label, c_label, n_label, _minuend, _subtrahend, _scale, _tolerance = control
        b, c, n = (_claim_value(b_label), _claim_value(c_label), _claim_value(n_label))
        low, high = paired_difference_bound_from_counts(b, c, n)
        assert low <= (b - c) / n <= high

    def test_a_fractional_count_is_refused_rather_than_truncated(self):
        """A rate label read where a count was meant must not become a plausible bound."""
        with pytest.raises(ValueError, match="whole number"):
            paired_difference_bound_from_counts(26.5, 10, 400)

    def test_impossible_discordance_is_refused_during_derivation(self):
        with pytest.raises(ValueError, match="cannot exceed n"):
            paired_difference_bound_from_counts(250, 200, 400)

    def test_no_derived_token_name_collides_with_a_recorded_claim(self):
        """A derived name matching a ledger label overwrites the measured value.

        ``generate_manuscript_variables`` writes every claim as
        ``label.upper()`` first and the derived block second, so a collision
        silently replaces a recorded number with a recomputed one and nothing
        reports it. The repository-wide scan lives in ``tests/test_benchmarks``;
        this one is scoped to the bound block and names the tokens explicitly,
        so a rename here cannot drift out from under the check.
        """
        ledger_labels = {
            label.lower()
            for label in json.loads(
                (PROJECT_ROOT / "data" / "claims.json").read_text(encoding="utf-8")
            )["claims"]
        }
        assert ledger_labels, "no ledger labels loaded; this scan would be vacuous"
        assert "stage5_mcnemar_b" in ledger_labels, (
            "the scan cannot see a label it is meant to catch; it is comparing the "
            "wrong things and would pass regardless of the token names"
        )
        collisions = sorted(
            token for token in _DERIVED_BOUND_TOKENS if token.lower() in ledger_labels
        )
        assert collisions == [], (
            f"these derived tokens share a name with a recorded claim, so the "
            f"derivation overwrites the measurement: {collisions}. Rename the "
            f"derived token or delete it and let the recorder be the only producer."
        )

    def test_every_declared_token_is_actually_produced(self):
        """The collision list must describe the render, not a stale intention.

        A token removed from the derivation block but left in the list above
        keeps being checked for collisions it can no longer cause, and a token
        added without being listed is checked by nothing.
        """
        variables = generate_manuscript_variables(PROJECT_ROOT)
        missing = [token for token in _DERIVED_BOUND_TOKENS if token not in variables]
        assert missing == [], f"declared derived tokens absent from the render: {missing}"

    def test_the_underpowered_nulls_are_bounds_and_not_equivalences(self):
        """The honest verdict, pinned so it cannot be softened by accident.

        Stating an equivalence needs the share interval inside the equivalence
        region, and that needs roughly four hundred discordant pairs. The arms
        below have fewer, so they are bounds and nothing more. If a later run
        makes one of them equivalent this test fails, which is the correct moment
        to rewrite the prose.

        The two context-window arms were here and are gone. Over the census pool
        they carry several hundred discordant pairs each and both contrasts clear
        the corrected threshold, so they are established effects rather than
        nulls; listing them as underpowered nulls would have been the same defect
        in a test that `fig:null_bounds` refused to commit in a figure. The arm
        that replaced them is the prompt-format contrast on Qwen 2.5 3B, which is
        the one contrast the census left unresolved.
        """
        minimum = minimum_discordant_pairs_for_equivalence()
        for b_label, c_label in (
            ("stage5_mcnemar_b", "stage5_mcnemar_c"),
            ("bench_letter_prompt_paired_b_qwen25", "bench_letter_prompt_paired_c_qwen25"),
        ):
            b, c = int(_claim_value(b_label)), int(_claim_value(c_label))
            assert b + c < minimum, (
                f"{b_label}/{c_label} now carries {b + c} discordant pairs against a "
                f"minimum of {minimum}. It is no longer an underpowered null, and the "
                f"prose describing it as one has to be rewritten rather than this list "
                f"quietly extended."
            )
            assert not share_interval_supports_equivalence(b, c)

    def test_the_high_weight_arm_is_a_decided_effect_not_an_equivalence(self):
        """The arm that was the one supported equivalence, after its correction.

        Under the defective backoff of decision 110, Stage 3's high-mixing-weight
        row was the one null with enough discordant pairs for its share interval
        to sit inside the equivalence region while its p-value missed the
        corrected threshold. Under the documented rule the same arm, with even
        more pairs, is a decided effect: its corrected-level interval excludes
        one half, the equivalence indicator is off, and its p-value clears the
        corrected threshold by many orders. The pair count is still asserted so
        the arm remains the demonstration that precision alone does not make an
        equivalence claim; the near-null has to be there too, and it no longer
        is.
        """
        b = int(_claim_value("lexical_sweep_alpha_0_9_mcnemar_b"))
        c = int(_claim_value("lexical_sweep_alpha_0_9_mcnemar_c"))
        assert b + c >= minimum_discordant_pairs_for_equivalence(confidence=CORRECTED_CONFIDENCE)
        assert not share_interval_supports_equivalence(b, c, confidence=CORRECTED_CONFIDENCE)
        assert _claim_value("lexical_sweep_alpha_0_9_mcnemar_p") < BONFERRONI_ALPHA


#: p-value labels deliberately outside the family, each with its reason.
#: Any label carrying ``_p`` as a trailing or interior segment that is neither
#: here nor in SIGNIFICANCE_FAMILY fails the closure test below — a new recorder
#: cannot silently add a p-value without either joining the family (changing the
#: corrected threshold) or documenting its exclusion.
EXCLUDED_FROM_FAMILY: tuple[tuple[str, str], ...] = (
    # ── The decision-111 confirmatory replication. These four are RESCORINGS of
    # four members already in the family, on the held-back ud_ewt_test
    # population, under the frozen endpoint, test, sidedness and threshold. They
    # are the same four hypotheses measured on new data, not four new ones.
    # Admitting them would count each hypothesis twice and tighten the corrected
    # threshold for every established result on the strength of a replication,
    # which inverts what a replication is for. Decision 111 forbids editing the
    # family for this run, and reports Holm beside Bonferroni by SUBSTITUTING
    # these p-values for their development counterparts over the same sixteen,
    # which is the arithmetic that keeps the family size at sixteen.
    (
        "confirmatory_stage4_derived_mcnemar_p",
        "decision-111 rescoring of stage4_derived_mcnemar_p on the held-back "
        "ud_ewt_test population; the same hypothesis, not a new one",
    ),
    (
        "confirmatory_stage4_gold_mcnemar_p",
        "decision-111 rescoring of stage4_gold_mcnemar_p on the held-back "
        "ud_ewt_test population; the same hypothesis, not a new one",
    ),
    (
        "confirmatory_stage5_mcnemar_p",
        "decision-111 rescoring of stage5_mcnemar_p on the held-back "
        "ud_ewt_test population; the same hypothesis, not a new one",
    ),
    (
        "confirmatory_stage6_sign_test_p",
        "decision-111 rescoring of stage6_sign_test_p on the held-back "
        "ud_ewt_test population; the same hypothesis, not a new one",
    ),
    # The two decision-111 controls. A control is evidence that the instrument
    # can detect an effect and does not credit noise; it is not a hypothesis
    # about English, and it is never used to make a threshold-crossing claim.
    # Correcting for it would spend familywise budget on the measurement
    # apparatus.
    (
        "confirmatory_control_positive_p",
        "planted-effect positive control for the decision-111 run; an instrument "
        "check on privileged parameters, not a hypothesis about the corpus",
    ),
    (
        "confirmatory_control_negative_p",
        "label-shuffled negative control for the decision-111 run; an instrument "
        "check that the comparison does not credit noise",
    ),
    # ── The decision-115 held-back document-clustered re-analysis. The family
    # member on this population is the SENTENCE-level sign test above; this arm
    # measures the document-clustered mean, which is a different estimand and
    # not a rescoring of one. It is excluded for a different reason: what it
    # reports is an interval, and the p beside it is an assumption-conditional
    # whole-document sign-flip sensitivity that no threshold-crossing claim
    # rests on. Decision 115 declines confirmatory status for the whole arm
    # because entry 111 had already spent the population, so admitting its p
    # would spend familywise budget on an analysis whose own registration
    # declines to make an inferential claim with it.
    (
        "confirmatory_stage6_clustered_p",
        "descriptive secondary analysis of a family member's population at the "
        "document unit under decision 115; an assumption-conditional sign-flip "
        "sensitivity beside a reported interval, not a hypothesis of its own",
    ),
    (
        "confirmatory_stage6_clustered_p_floor",
        "Monte Carlo resolution constant for the decision-115 document sign-flip "
        "test, a property of the draw count rather than a p-value of any comparison",
    ),
    # The lexical sweep's peak-against-highest-weight comparison (decision 110)
    # asks whether the sweep's argmax is distinguishable from its top weight,
    # an ordering question between two arms of one sweep whose family member is
    # `lexical_best_mcnemar_p`. Both arms score the same positions that member
    # already scores; admitting it would count them twice and tighten the
    # corrected threshold on the strength of a shape diagnostic. Its verdict is
    # reported as what it is: the peak's lead is not a measured cost.
    (
        "lexical_peak_vs_high_mcnemar_p",
        "within-sweep ordering diagnostic between the recorded argmax and the "
        "highest weight; the sweep's family member is lexical_best_mcnemar_p",
    ),
    # The live-Agent Stage 6 score was added after the observed-position
    # estimand was registered.  It verifies that the built pymdp recursion is
    # executable and transition-sensitive, but it is recorded as descriptive
    # and is never used to make a threshold-crossing claim.  Counting it as a
    # seventeenth family member would retroactively alter the frozen family on
    # the strength of an implementation diagnostic.
    (
        "stage6_live_filter_sign_p",
        "post-registered descriptive live-Agent execution diagnostic; distinct "
        "from the registered observed-position estimand",
    ),
    # ── Stage 4's oracle-path partition (decision 63). These three PARTITION a
    # hypothesis already in the family: every position they score is a position
    # `stage4_derived_mcnemar_p` already scored, and the three strata sum back to
    # the derived population exactly. They ask where an admitted effect lives,
    # not whether a new one exists. Admitting them would count the same positions
    # twice and tighten the corrected threshold for every established result on
    # the strength of a diagnostic, which is precisely the cost the cluster-aware
    # exclusions below refuse to pay for the same reason.
    #
    # The consequence is stated rather than hidden: none of the three may be
    # reported as a confirmed effect. The deprel stratum's p-value is quoted in
    # `04_results.md` to show that the aggregate is a mixture, which is a
    # statement about decomposition; the morphology and adposition strata are
    # registered as controls whose discordant counts fall below the resolvability
    # minimum, so their outcome is "not shown" and never "absent".
    (
        "stage4_path_deprel_mcnemar_p",
        "partition of a family member; shows where the admitted margin lives",
    ),
    (
        "stage4_path_morphology_mcnemar_p",
        "pre-registered control stratum of the same partition, under-powered by design",
    ),
    (
        "stage4_path_adposition_mcnemar_p",
        "third stratum of the same partition; sign disagrees, unresolved",
    ),
    # ── Cluster-aware re-analyzes (decision on the review's sampling-unit
    # finding). Each of these re-tests a hypothesis ALREADY in the family at a
    # coarser sampling unit — word, sentence or document — rather than asking a
    # new question. Admitting them would inflate the family and tighten the
    # corrected threshold for re-examining the same effects, which is the
    # opposite of what a robustness check should cost. The naive variants are
    # the same comparison with clustering switched off, kept so the two paths
    # are readable side by side.
    (
        "cluster_stage2_g2p_p",
        "cluster-aware re-analysis of a family member at a coarser unit",
    ),
    (
        "cluster_stage2_g2p_normal_p",
        "self-normalized normal-tail approximation for the same re-analysis",
    ),
    (
        "cluster_stage2_g2p_naive_normal_p",
        "the same re-analysis with clustering off, for comparison",
    ),
    (
        "cluster_stage4_gold_p",
        "cluster-aware re-analysis of a family member at a coarser unit",
    ),
    (
        "cluster_stage4_gold_normal_p",
        "self-normalized normal-tail approximation for the same re-analysis",
    ),
    (
        "cluster_stage4_gold_naive_normal_p",
        "the same re-analysis with clustering off, for comparison",
    ),
    (
        "cluster_stage4_derived_p",
        "cluster-aware re-analysis of a family member at a coarser unit",
    ),
    (
        "cluster_stage4_derived_normal_p",
        "self-normalized normal-tail approximation for the same re-analysis",
    ),
    (
        "cluster_stage4_derived_naive_normal_p",
        "the same re-analysis with clustering off, for comparison",
    ),
    (
        "stage5_classification_document_p",
        "document-cluster sensitivity analysis of the existing Stage 5 classification contrast",
    ),
    (
        "stage5_classification_document_p_floor",
        "Monte Carlo resolution of the Stage 5 document-cluster sensitivity analysis",
    ),
    # ── The widened-observation clause arm (decision 112). Three exclusions,
    # one reason: entry 79 FROZE this family for the replication run, and entry
    # 112 is a development-session specification with no independently
    # timestamped prior artifact. Admitting a post-freeze exploratory arm would
    # tighten the corrected threshold for every already-established result on
    # the strength of a second arm nobody registered against that family — the
    # same cost the Czech and cluster-aware exclusions above refuse to pay.
    #
    # The consequence is stated rather than hidden: none of the three may be
    # reported as a confirmed effect, and the arm's classification contrast is
    # therefore read from its document-clustered interval rather than from a
    # threshold crossing.
    (
        "stage5w_mcnemar_p",
        "post-freeze exploratory arm outside the frozen family (decisions 79 and 112); "
        "the shipped arm's family-adjacent member is stage5_mcnemar_p",
    ),
    (
        "stage5w_classification_document_p",
        "document-cluster sensitivity analysis of the widened arm's classification "
        "contrast, itself outside the frozen family",
    ),
    (
        "stage5w_classification_document_p_floor",
        "Monte Carlo resolution constant of that widened document-cluster analysis, "
        "not a p-value of any comparison",
    ),
    (
        "cluster_stage6_sign_p",
        "cluster-aware re-analysis of a family member at a coarser unit",
    ),
    (
        "cluster_stage6_sign_normal_p",
        "self-normalized normal-tail approximation for the same re-analysis",
    ),
    (
        "cluster_stage6_sign_naive_normal_p",
        "the same re-analysis with clustering off, for comparison",
    ),
    (
        "cluster_stage6_logloss_p",
        "cluster-aware re-analysis of a family member at a coarser unit",
    ),
    (
        "cluster_stage6_logloss_normal_p",
        "self-normalized normal-tail approximation for the same re-analysis",
    ),
    (
        "cluster_stage6_logloss_naive_normal_p",
        "the same re-analysis with clustering off, for comparison",
    ),
    (
        "cluster_p_floor",
        "the Monte Carlo resolution limit of the resampler, a protocol constant "
        "rather than a p-value of any comparison",
    ),
    (
        "subsample_full_mcnemar_p",
        "byte-identical duplicate of stage2_g2p_mcnemar_p (same comparison, second label)",
    ),
    (
        "stat_stage6_sign_test_one_sided_p",
        "sidedness sensitivity diagnostic for a family member, not a new hypothesis",
    ),
    # ── The Czech case-morphology arm (decision 77). Prospectively specified
    # within the development session but not independently timestamped; it is
    # exploratory and outside the already-frozen confirmatory family on both
    # the document-clustered primary and token-conditional secondary scales.
    # Decision 79 freezes the family for the replication run, so admission is
    # doubly barred. The log10 form is the same test stored on a readable scale
    # (the exact p underflows a float), not a second hypothesis.
    (
        "stage4_czech_mcnemar_p",
        "prospectively specified exploratory arm outside the frozen family (decision 77)",
    ),
    (
        "stage4_czech_mcnemar_log10_p",
        "the same Czech test on a log scale, one hypothesis under two labels",
    ),
    (
        "cluster_stage4_czech_p",
        "assumption-conditional document sign-flip sensitivity for Czech decision 77",
    ),
    (
        "cluster_stage4_czech_p_floor",
        "Monte Carlo resolution constant for the Czech document sign-flip test",
    ),
    (
        "cluster_stage4_czech_p_at_resolution_floor",
        "boolean diagnostic that the Czech randomization result reached its declared floor",
    ),
    # Per-mixing-weight McNemar p-values of the L3 → L2B channel. These
    # are the SAME comparison as the family member lexical_best_mcnemar_p
    # (same receiver, same held-out positions), just read off each point of
    # the weight sweep — interior-optimum diagnostics of one hypothesis, not
    # new members of the multiplicity family. See decisions entry 19.
    (
        "lexical_sweep_alpha_0_0_mcnemar_p",
        "alpha=0 anchor of the L3 → L2B sweep; b+c=0 so p=1.0 by definition, not a test",
    ),
    (
        "lexical_sweep_alpha_0_1_mcnemar_p",
        "same comparison as lexical_best_mcnemar_p at alpha=0.1 (sweep diagnostic)",
    ),
    (
        "lexical_sweep_alpha_0_2_mcnemar_p",
        "same comparison as lexical_best_mcnemar_p at alpha=0.2 (sweep diagnostic)",
    ),
    (
        "lexical_sweep_alpha_0_3_mcnemar_p",
        "alpha=0.3 sweep point; equal to the family member lexical_best_mcnemar_p",
    ),
    (
        "lexical_sweep_alpha_0_5_mcnemar_p",
        "same comparison as lexical_best_mcnemar_p at alpha=0.5 (sweep diagnostic)",
    ),
    (
        "lexical_sweep_alpha_0_7_mcnemar_p",
        "same comparison as lexical_best_mcnemar_p at alpha=0.7 (sweep diagnostic)",
    ),
    (
        "lexical_sweep_alpha_0_9_mcnemar_p",
        "alpha=0.9 sweep point where the effect reverses (sweep diagnostic)",
    ),
    (
        "stage2_g2p_famsplit_mcnemar_p",
        "the family-disjoint G2P robustness arm (decision 74): the same Stage 2B "
        "hypothesis re-tested on a split that removes four-character morphological "
        "neighbors, pre-registered as exploratory. Entering it would count one "
        "hypothesis twice; the manuscript cites it as a bound on the leakage story, "
        "never as a familywise survivor",
    ),
    # ── The family-grouped G2P sensitivity arm (decision 102). Each rank row
    # re-scores one prefix family of the population that
    # `stage2_g2p_famsplit_mcnemar_p` already scored, with the same decoder,
    # comparator and paired test; the position-weighted pool over the families
    # reproduces the famsplit counts exactly. They are diagnostics of a
    # diagnostic arm — they ask whether the famsplit margin is level-wide, not
    # whether a new effect exists — and inherit the decision-74 exclusion. None
    # may be reported as a confirmed effect.
    *(
        (
            f"stage2_g2p_famgrp_rank{rank}_mcnemar_p",
            "per-family sensitivity re-analysis (decision 102) of the population "
            "stage2_g2p_famsplit_mcnemar_p already scored; a partition of the "
            "excluded famsplit arm, recorded as exploratory and never a familywise "
            "survivor",
        )
        for rank in range(1, 11)
    ),
    # Punctuation-type-oracle -> word-prior intervention (decisions 27 and 89).
    # Token outcomes are nested in Brown source documents, so the old float-p
    # labels were retired.  The log10 labels preserve an exact binomial
    # diagnostic without pretending the tokens are independent sampling units;
    # the document-resampled labels are the assumption-conditional uncertainty
    # analyses.  Both are outside the already-frozen confirmatory family: this
    # is a correction-driven remeasurement on an inspected population, not a
    # new model-output-channel hypothesis.
    (
        "clause_word_best_pair_log10_p",
        "exact token-binomial diagnostic on a log10 scale; tokens are nested in documents and this is not a familywise p-value",
    ),
    *(
        (
            f"clause_word_sweep_alpha_{alpha}_pair_log10_p",
            "the same nested-token diagnostic at one weight of the oracle-intervention sweep, not a new hypothesis",
        )
        for alpha in ("0_0", "0_1", "0_2", "0_3", "0_5", "0_7", "0_9")
    ),
    (
        "clause_word_cluster_token_p",
        "whole-document resampling sensitivity for the descriptive token-direction estimand outside the frozen family",
    ),
    (
        "clause_word_cluster_loggain_p",
        "whole-document resampling sensitivity for the descriptive per-emission log-gain estimand outside the frozen family",
    ),
    (
        "clause_word_cluster_token_p_floor",
        "Monte Carlo resolution constant for the oracle-intervention token-direction analysis",
    ),
    (
        "clause_word_cluster_loggain_p_floor",
        "Monte Carlo resolution constant for the oracle-intervention log-gain analysis",
    ),
    (
        "clause_word_cluster_token_p_at_resolution_floor",
        "boolean diagnostic that the token-direction resampling result reached its declared floor",
    ),
    (
        "clause_word_cluster_loggain_p_at_resolution_floor",
        "boolean diagnostic that the log-gain resampling result reached its declared floor",
    ),
    # External benchmark paired diagnostics (each probe vs the AILM comparator on
    # identical positions). These are excluded from the REPOSITORY-WIDE family
    # because they test an external reference point rather than any of this
    # project's own hypotheses: nothing in the manuscript's argument turns on
    # whether an Ollama model beats the comparator. They are not thereby
    # uncorrected. The benchmark declares its own family and its own Bonferroni
    # threshold — `bench_family_size` and `bench_family_alpha`, recorded by
    # scripts/record_external_benchmark_claims.py from FAMILY_TEMPLATES — and the
    # supplement reports which contrasts clear it. An exclusion here means "a
    # different family", not "no correction"; adding a bench p-value without a
    # matching FAMILY_TEMPLATES entry would be the actual dissolution of the
    # discipline, and `test_bench_p_values_are_in_the_declared_bench_family`
    # below refuses it.
    (
        "bench_letter_prompt_paired_p_gemma3",
        "paired prompt-format sensitivity diagnostic on the external benchmark positions; separate protocol-sensitivity question, not a model-vs-comparator hypothesis",
    ),
    (
        "bench_letter_prompt_paired_p_qwen25",
        "paired prompt-format sensitivity diagnostic on the external benchmark positions; separate protocol-sensitivity question, not a model-vs-comparator hypothesis",
    ),
    (
        "bench_letter_prompt_paired_p_smollm",
        "paired prompt-format sensitivity diagnostic on the external benchmark positions; separate protocol-sensitivity question, not a model-vs-comparator hypothesis",
    ),
    (
        "bench_letter_paired_p_gemma3",
        "external letter benchmark, gemma3:4b vs the comparator; corrected inside the declared bench family",
    ),
    (
        "bench_phoneme_paired_p_gemma3",
        "external phoneme benchmark, gemma3:4b vs the comparator; corrected inside the declared bench family",
    ),
    (
        "geometric_uniform_paired_p",
        "a control that CANNOT reject: a constant raised to any power cancels in "
        "normalization, so this arm returns the anchor by algebra and its p-value is 1 by "
        "construction. Including a test that cannot reject would inflate the family and "
        "tighten every other threshold on the strength of an arithmetic identity",
    ),
    (
        "geometric_scrambled_paired_p",
        "the entropy-preserving scramble control for decision 47's lexical arm; it tests the "
        "instrument rather than the model, in the same sense as the benchmark's "
        "protocol-sensitivity diagnostic",
    ),
    # ── The channel factorial's diagonal cells (decision 117). Each is the SAME
    # hypothesis as a family member already declared, measured under a different
    # weight grid on the same evaluated positions against the same anchor. The
    # family counts hypotheses, not weight choices, so counting a cell as well
    # would count one hypothesis twice and tighten every other threshold on the
    # strength of a weight choice. The reasons below are pointers: the basis
    # itself is SHARED_HYPOTHESIS_EXCLUSIONS, checked against the ledger by
    # test_shared_hypothesis_exclusions_hold_their_ledger_invariants. Until
    # decision 117 this pair was justified by prose asserting a numerical
    # identity, which decision 110's re-recording made false for the lexical cell
    # while every consumer went on reading the label set and nothing noticed.
    (
        "channel_letter_evidence_paired_p",
        "same hypothesis as coupling_mcnemar_p under a second weight grid; basis and "
        "invariants in SHARED_HYPOTHESIS_EXCLUSIONS (decision 117)",
    ),
    (
        "channel_lexicon_predictive_paired_p",
        "same hypothesis as lexical_best_mcnemar_p under a second weight grid; basis and "
        "invariants in SHARED_HYPOTHESIS_EXCLUSIONS (decision 117)",
    ),
    (
        "bench_g2p_paired_p_gemma3",
        "external G2P exact-match benchmark, gemma3:4b vs the comparator; corrected inside the declared bench family",
    ),
    (
        "bench_letter_paired_p_qwen25",
        "external letter benchmark, qwen2.5:3b vs the comparator; corrected inside the declared bench family",
    ),
    (
        "bench_letter_paired_p_smollm",
        "external letter benchmark, smollm:1.7b-base-v0.2-q4_K_M vs the comparator; corrected inside the declared bench family",
    ),
    (
        "bench_phoneme_paired_p_qwen25",
        "external phoneme benchmark, qwen2.5:3b vs the comparator; corrected inside the declared bench family",
    ),
    (
        "bench_phoneme_paired_p_smollm",
        "external phoneme benchmark, smollm:1.7b-base-v0.2-q4_K_M vs the comparator; corrected inside the declared bench family",
    ),
    (
        "bench_g2p_paired_p_qwen25",
        "external G2P exact-match benchmark, qwen2.5:3b vs the comparator; corrected inside the declared bench family",
    ),
    (
        "bench_g2p_paired_p_smollm",
        "external G2P exact-match benchmark, smollm:1.7b-base-v0.2-q4_K_M vs the comparator; corrected inside the declared bench family",
    ),
    (
        "bench_letter_az_paired_p_gemma3",
        "external letter benchmark restricted to alphabetic gold, gemma3:4b; corrected inside the declared bench family",
    ),
    (
        "bench_letter_az_paired_p_qwen25",
        "external letter benchmark restricted to alphabetic gold, qwen2.5:3b; corrected inside the declared bench family",
    ),
    (
        "bench_letter_az_paired_p_smollm",
        "external letter benchmark restricted to alphabetic gold, smollm:1.7b-base-v0.2-q4_K_M; corrected inside the declared bench family",
    ),
    (
        "bench_letter_ctx_paired_p_gemma3",
        "gemma3:4b long vs short letter context on identical positions; corrected inside the declared bench family",
    ),
    (
        "bench_letter_ctx_paired_p_qwen25",
        "qwen2.5:3b long vs short letter context on identical positions; corrected inside the declared bench family",
    ),
    (
        "bench_letter_ctx_paired_p_smollm",
        "smollm:1.7b-base-v0.2-q4_K_M long vs short letter context on identical positions; corrected inside the declared bench family",
    ),
    (
        "bench_synth_paired_p_gemma3",
        "stipulated-language exact-artifact-overlap probe: gemma3:4b vs the comparator on a generated corpus; one member of the separately declared descriptive synthetic family spanning every registered probe",
    ),
    (
        "bench_synth_paired_p_qwen25",
        "stipulated-language exact-artifact-overlap probe: qwen2.5:3b vs the comparator on a generated corpus; one member of the separately declared descriptive synthetic family spanning every registered probe",
    ),
    (
        "bench_synth_paired_p_smollm",
        "stipulated-language exact-artifact-overlap probe: smollm:1.7b-base-v0.2-q4_K_M vs the comparator on a generated corpus; one member of the separately declared descriptive synthetic family spanning every registered probe",
    ),
    (
        "bench_synth_context_paired_p_gemma3",
        "audit-specified full-prefix versus one-symbol-prefix sensitivity for gemma3:4b on identical stipulated-language positions; descriptive and corrected inside its separate context family",
    ),
    (
        "bench_synth_context_paired_p_qwen25",
        "audit-specified full-prefix versus one-symbol-prefix sensitivity for qwen2.5:3b on identical stipulated-language positions; descriptive and corrected inside its separate context family",
    ),
    (
        "bench_synth_context_paired_p_smollm",
        "audit-specified full-prefix versus one-symbol-prefix sensitivity for smollm:1.7b-base-v0.2-q4_K_M on identical stipulated-language positions; descriptive and corrected inside its separate context family",
    ),
    (
        "bench_synthphone_paired_p_gemma3",
        "stipulated-language exact-artifact-overlap probe, phone arm: gemma3:4b vs the comparator on a generated ARPAbet chain; one member of the separately declared descriptive synthetic-phone family spanning every registered probe",
    ),
    (
        "bench_synthphone_paired_p_qwen25",
        "stipulated-language exact-artifact-overlap probe, phone arm: qwen2.5:3b vs the comparator on a generated ARPAbet chain; one member of the separately declared descriptive synthetic-phone family spanning every registered probe",
    ),
    (
        "bench_synthphone_paired_p_smollm",
        "stipulated-language exact-artifact-overlap probe, phone arm: smollm:1.7b-base-v0.2-q4_K_M vs the comparator on a generated ARPAbet chain; one member of the separately declared descriptive synthetic-phone family spanning every registered probe",
    ),
    (
        "bench_synthphone_context_paired_p_gemma3",
        "audit-specified full-prefix versus one-phone-prefix sensitivity for gemma3:4b on identical stipulated-phone positions; descriptive and corrected inside its separate context family",
    ),
    (
        "bench_synthphone_context_paired_p_qwen25",
        "audit-specified full-prefix versus one-phone-prefix sensitivity for qwen2.5:3b on identical stipulated-phone positions; descriptive and corrected inside its separate context family",
    ),
    (
        "bench_synthphone_context_paired_p_smollm",
        "audit-specified full-prefix versus one-phone-prefix sensitivity for smollm:1.7b-base-v0.2-q4_K_M on identical stipulated-phone positions; descriptive and corrected inside its separate context family",
    ),
    # The in-context arm (scripts/record_incontext_benchmark_claims.py). These
    # test one attempted remediation of the fitted-versus-unfitted asymmetry by
    # giving the probe labeled observations and sweeping how many. They do not
    # prove that the asymmetry is absent when the probes fail to use the supplied
    # mapping. They are excluded from the
    # repository-wide family for the same reason as every other bench p-value:
    # they are about an external reference point, not about AILM's own
    # hypotheses. The model contrasts are corrected inside that recorder's
    # declared template-times-probe family, while the fixed mapping learner is
    # a separate one-member positive-control family. The scan below reads both
    # directly from P_VALUE_FAMILIES.
    *(
        (
            f"bench_ctx_k{k}_vs_zero_paired_p_{probe}",
            f"in-context arm at k={k} against the zero-shot arm on identical positions; "
            f"corrected inside the declared in-context family",
        )
        for k in SYNTHETIC_INCONTEXT_PROTOCOL.k_grid[1:]
        for probe in ("gemma3", "qwen25", "smollm")
    ),
    *(
        (
            f"bench_ctx_k{k}_vs_deranged_paired_p_{probe}",
            f"in-context arm at k={k} against its character-level prefix/label-multiset-preserving derangement control; "
            f"tests whether the examples are used rather than merely displayed, and is "
            f"corrected inside the declared in-context family",
        )
        for k in SYNTHETIC_INCONTEXT_PROTOCOL.k_grid[1:]
        for probe in ("gemma3", "qwen25", "smollm")
    ),
    *(
        (
            f"bench_ctx_k{SYNTHETIC_INCONTEXT_PROTOCOL.headline_k}_vs_ailm_paired_p_{probe}",
            "the in-context arm at the frozen protocol-owned headline k against the AILM comparator "
            "on identical positions; corrected inside the declared in-context family",
        )
        for probe in ("gemma3", "qwen25", "smollm")
    ),
    (
        "bench_ctx_mapping_control_paired_p",
        "fixed terminal-mapping learner on treatment versus deranged labeled windows; separately corrected one-member positive-control family",
    ),
)


# ── An exclusion's basis is a checked invariant, not a sentence (decision 117) ─


def test_shared_hypothesis_exclusions_hold_their_ledger_invariants():
    """Decision 117: the reason an exclusion stands must be checked, not stated.

    The prose that justified these exclusions asserted a numerical identity. When
    decision 110 re-recorded the lexical arm, half of it stopped being true and
    nothing failed, because every consumer reads ``EXCLUDED_FROM_FAMILY`` as a set
    of labels. This test is the replacement: it asserts the shared-population
    invariants the exclusion actually rests on, straight from the ledger.

    NON-VACUITY, both directions: an identity leg that asserted nothing and a
    difference leg that asserted nothing would each let the roster go quiet, so
    each must be exercised by at least one record.

    CLOSURE: every excluded factorial cell must appear here. Excluding a future
    diagonal cell without a checked basis is the defect this entry rules on, and
    it would otherwise be indistinguishable from the two that are argued.
    """
    from ailm.channel_registry import channel_by_key

    registry = channel_by_key()
    excluded_labels = {label for label, _reason in EXCLUDED_FROM_FAMILY}

    covered = {record.excluded for record in SHARED_HYPOTHESIS_EXCLUSIONS}
    factorial_exclusions = {
        label
        for label in excluded_labels
        if label.startswith("channel_") and label.endswith("_paired_p")
    }
    assert factorial_exclusions == covered, (
        f"channel-factorial exclusions without a checked basis: "
        f"{sorted(factorial_exclusions - covered)}; records naming a label that is "
        f"no longer excluded: {sorted(covered - factorial_exclusions)}"
    )
    assert covered, "the roster is empty, so this test checks nothing"

    for record in SHARED_HYPOTHESIS_EXCLUSIONS:
        assert record.excluded in excluded_labels, record.excluded
        assert record.family_member in SIGNIFICANCE_FAMILY, (
            f"{record.excluded} is excluded as a restatement of "
            f"{record.family_member}, which is not in the family at all"
        )

        channel = registry.get(record.channel_key)
        assert channel is not None, (
            f"{record.excluded} names channel {record.channel_key!r}, which the "
            f"cross-level registry does not carry"
        )
        assert channel.insertion == record.insertion, (
            f"{record.excluded} declares the {record.insertion!r} site but the "
            f"registry says {channel.insertion!r}; the cell and its family member "
            f"are no longer the same intervention"
        )

        cell_positions, member_positions = record.positions
        assert _claim_value(cell_positions) == _claim_value(member_positions), (
            f"{record.excluded} is scored on {_claim_value(cell_positions)} positions "
            f"against {_claim_value(member_positions)} for {record.family_member}. "
            f"Different populations are different hypotheses, so the exclusion is "
            f"void and the family must be re-declared with this cell in it."
        )

        cell_anchor, member_anchor = record.anchor
        assert _claim_value(cell_anchor) == _claim_value(member_anchor), (
            f"{record.excluded} is compared against an anchor of "
            f"{_claim_value(cell_anchor)} while {record.family_member} uses "
            f"{_claim_value(member_anchor)}. A margin against a different anchor "
            f"answers a different question, so the exclusion is void."
        )

        for cell_label, member_label in record.identical:
            assert _claim_value(cell_label) == _claim_value(member_label), (
                f"{cell_label} and {member_label} were recorded as identical and no "
                f"longer are. The shared-population basis may still hold; narrow "
                f"this record to it rather than leaving the identity asserted."
            )

        if record.basis is ExclusionBasis.EXACT_IDENTITY:
            assert record.identical and not record.differs, (
                f"{record.excluded} declares an exact identity but its record names "
                f"no pair to check, or names a pair that must differ"
            )
        else:
            assert record.differs and not record.identical, (
                f"{record.excluded} declares a second-grid basis but its record names "
                f"no pair that differs, or still asserts an identity"
            )

        for cell_label, member_label in record.differs:
            assert _claim_value(cell_label) != _claim_value(member_label), (
                f"{cell_label} now equals {member_label}. If a recorder was changed "
                f"to make the two grids agree, decision 117 rejects that: a "
                f"measurement is not adjusted to make a sentence true. If they "
                f"converged on their own, move the pair into `identical`."
            )

    assert any(
        record.basis is ExclusionBasis.EXACT_IDENTITY for record in SHARED_HYPOTHESIS_EXCLUSIONS
    ), "no record asserts an identity, so the identity leg of this test is dead"
    assert any(
        record.basis is ExclusionBasis.SECOND_SELECTION_GRID
        for record in SHARED_HYPOTHESIS_EXCLUSIONS
    ), (
        "no record asserts a difference, so nothing pins decision 117's finding "
        "that the grids select apart"
    )


# ── Excluded p-values printed in manuscript tables must say so ───────────────

_MANUSCRIPT_DIR = Path(__file__).resolve().parent.parent / "docs" / "manuscript"
_TABLE_CAPTION = re.compile(r"^:\s.*\{#tbl:([a-z0-9_]+)\}\s*$")
_TOKEN = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
#: A caption that prints a p-value outside the family must name that fact. Either
#: word is a reader-facing statement of it; the guard asks for the disclosure,
#: not for a fixed sentence.
_EXCLUSION_DISCLOSURE = re.compile(r"\bfamily\b|\buncorrected\b", re.I)


def _table_block(lines: list[str], start: int, step: int) -> list[str]:
    """The pipe-table rows adjacent to ``start``, walking in ``step`` direction.

    The manuscript places a caption either directly after its table (blank line
    between) or directly before it, so both neighbors are read.
    """
    index = start + step
    while 0 <= index < len(lines) and not lines[index].strip():
        index += step
    rows: list[str] = []
    while 0 <= index < len(lines) and lines[index].lstrip().startswith("|"):
        rows.append(lines[index])
        index += step
    return rows


def _tables_printing_excluded_labels(
    text: str, excluded: set[str]
) -> list[tuple[str, tuple[str, ...], str]]:
    """``(table label, excluded labels printed, caption)`` for every offending table."""
    lines = text.splitlines()
    found: list[tuple[str, tuple[str, ...], str]] = []
    for index, line in enumerate(lines):
        match = _TABLE_CAPTION.match(line)
        if match is None:
            continue
        body = _table_block(lines, index, -1) + _table_block(lines, index, +1)
        printed = tuple(
            sorted(
                {
                    token.lower()
                    for row in body
                    for token in _TOKEN.findall(row)
                    if token.lower() in excluded
                }
            )
        )
        if printed:
            found.append((match.group(1), printed, line))
    return found


def _undisclosed_excluded_tables(
    text: str, excluded: set[str]
) -> list[tuple[str, tuple[str, ...]]]:
    return [
        (label, printed)
        for label, printed, caption in _tables_printing_excluded_labels(text, excluded)
        if _EXCLUSION_DISCLOSURE.search(caption) is None
    ]


def test_every_table_printing_an_excluded_p_value_says_so_in_its_caption():
    """A p-value outside the family may be printed, but the caption must say so.

    DEFECT: ``tbl:case_oracle_paths`` printed three oracle-path McNemar p-values
    beside a bolded margin with a caption that spoke only of the strata summing
    to the population. The exclusion — uncorrected, partition of a family
    member, never a confirmed effect — lived only in ``EXCLUDED_FROM_FAMILY``
    above, which no reader of the PDF sees. The guard is stated over the
    general class: any table whose body prints a token for an excluded label.
    """
    excluded = {label for label, _reason in EXCLUDED_FROM_FAMILY}
    scanned = 0
    offenders: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    for source in sorted(_MANUSCRIPT_DIR.glob("[0-9]*.md")):
        text = source.read_text(encoding="utf-8")
        scanned += len(_tables_printing_excluded_labels(text, excluded))
        bad = _undisclosed_excluded_tables(text, excluded)
        if bad:
            offenders[source.name] = bad
    assert scanned > 0, "no manuscript table prints an excluded p-value; this guard is vacuous"
    assert offenders == {}, (
        f"tables printing p-values outside the declared family without saying so in "
        f"the caption (file -> [(table, labels)]): {offenders}. Add 'uncorrected' or "
        f"'outside the ... family' to the caption, or stop printing the token."
    )


def test_the_excluded_table_guard_detects_a_planted_silent_caption():
    """POSITIVE CONTROL: the scan must see both caption placements and both words."""
    excluded = {"stage4_path_deprel_mcnemar_p"}
    after = (
        "| stratum | *p* |\n|---|---:|\n| deprel | {{STAGE4_PATH_DEPREL_MCNEMAR_P}} |\n\n"
        ": The strata sum to the population. {#tbl:planted_after}\n"
    )
    before = (
        ": Rows as recorded. {#tbl:planted_before}\n\n"
        "| stratum | *p* |\n|---|---:|\n| deprel | {{STAGE4_PATH_DEPREL_MCNEMAR_P}} |\n"
    )
    assert [label for label, _ in _undisclosed_excluded_tables(after, excluded)] == [
        "planted_after"
    ]
    assert [label for label, _ in _undisclosed_excluded_tables(before, excluded)] == [
        "planted_before"
    ]
    disclosed = after.replace(
        "The strata sum to the population.",
        "The strata sum to the population; the *p*-values are uncorrected.",
    )
    assert _undisclosed_excluded_tables(disclosed, excluded) == []
    named = before.replace("Rows as recorded.", "Rows outside the declared family.")
    assert _undisclosed_excluded_tables(named, excluded) == []
    # A table printing only family members is not the guard's business.
    assert _tables_printing_excluded_labels(after, {"some_other_p"}) == []


def _p_value_labels(labels) -> list[str]:
    """Every ledger label that holds a p-value.

    A trailing ``_p`` was the whole rule, which meant the scan never saw a
    p-value with anything after it — ``bench_letter_paired_p_gemma3`` and its
    siblings sat outside the closure check while their exclusion entries read as
    though they were load-bearing. Any label with ``_p`` as an interior segment
    is a p-value too. ``bench_g2p_*`` is not a false positive: the segment there
    is ``g2p``, not ``p``.
    """
    return [label for label in labels if label.endswith("_p") or "_p_" in label]


def _bench_recorder_paths() -> list[Path]:
    """Every recorder that declares a benchmark multiplicity family.

    Discovered by scanning, not listed. As a hand-written tuple of two paths
    this was a guard that named its instances: a third recorder writing a
    ``bench_*`` p-value would simply not be looked at, and the closure check
    below would then report the new p-value as *missing from* a family it had
    never been asked to read. A guard against uncorrected tests that silently
    stops covering new tests is the failure mode it exists to prevent.

    The rule is the general one: a module under ``scripts/`` that defines
    ``MODELS`` and either ``P_VALUE_FAMILIES`` or the legacy single
    ``FAMILY_TEMPLATES`` tuple is declaring one or more families.
    """
    found: list[Path] = []
    for path in sorted((PROJECT_ROOT / "scripts").glob("record_*.py")):
        text = path.read_text(encoding="utf-8")
        if "\nMODELS" in text and ("\nP_VALUE_FAMILIES" in text or "\nFAMILY_TEMPLATES" in text):
            found.append(path)
    assert found, "no benchmark recorder declares FAMILY_TEMPLATES; the scan is broken"
    return found


def _declared_family_templates(recorder) -> tuple[str, ...]:
    """Flatten every separately corrected family a recorder declares."""

    families = getattr(recorder, "P_VALUE_FAMILIES", None)
    if families is None:
        families = {"default": recorder.FAMILY_TEMPLATES}
    assert isinstance(families, dict) and families, "benchmark family registry is empty"
    templates = tuple(template for family in families.values() for template in family)
    assert templates and len(templates) == len(set(templates)), (
        "benchmark family templates must be nonempty and unique across families"
    )
    return templates


def test_the_bench_recorder_scan_finds_every_declaring_recorder():
    """NON-VACUITY: the scan is what makes the closure check load-bearing.

    Pinned as a lower bound rather than an exact set, so adding a recorder does
    not fail this, but deleting the scan's ability to see one does.
    """
    names = {path.name for path in _bench_recorder_paths()}
    required = {
        "record_external_benchmark_claims.py",
        "record_synthetic_benchmark_claims.py",
        "record_incontext_benchmark_claims.py",
    }
    assert required <= names, f"scan missed declaring recorders: {sorted(required - names)}"


def test_the_p_value_scan_sees_labels_with_a_trailing_suffix():
    """NON-VACUITY: the widened scan must actually pick up suffixed p-values."""
    found = _p_value_labels(["a_mcnemar_p", "bench_letter_paired_p_gemma3", "bench_g2p_exact_ailm"])
    assert found == ["a_mcnemar_p", "bench_letter_paired_p_gemma3"]


def test_bench_p_values_are_in_the_declared_bench_family():
    """Every excluded bench p-value is corrected inside the benchmark's own family.

    The exclusions above claim the bench p-values are corrected elsewhere rather
    than uncorrected. This is that "elsewhere": the recorder's FAMILY_TEMPLATES,
    expanded over the registered models, must name exactly the bench p-values in
    the ledger. Without this, "excluded from the repo family" would be a way to
    report an uncorrected test with a footnote.
    """
    declared: set[str] = set()
    recorders = _bench_recorder_paths()
    for recorder_path in recorders:
        spec = importlib.util.spec_from_file_location("_bench_recorder", recorder_path)
        assert spec is not None and spec.loader is not None
        recorder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(recorder)
        declared |= {
            template.format(model=model)
            for model in recorder.MODELS
            for template in _declared_family_templates(recorder)
        }
    ledger = load_ledger(PROJECT_ROOT)
    labels = list(ledger.labels()) if hasattr(ledger, "labels") else None
    if labels is None:
        import json

        labels = list(json.loads((PROJECT_ROOT / "data" / "claims.json").read_text())["claims"])
    recorded = {label for label in _p_value_labels(labels) if label.startswith("bench_")}
    assert recorded == declared, (
        f"bench p-values in the ledger and the recorder's declared family disagree. "
        f"Only in the ledger: {sorted(recorded - declared)}. Only declared: "
        f"{sorted(declared - recorded)}. A bench p-value outside P_VALUE_FAMILIES is "
        f"corrected by nothing at all."
    )
    excluded = {label for label, _reason in EXCLUDED_FROM_FAMILY}
    assert declared <= excluded, (
        f"declared bench-family members missing a documented repo-family exclusion: "
        f"{sorted(declared - excluded)}"
    )


def test_family_is_closed_over_the_ledger():
    """Every p-value in the ledger is in the family or explicitly excluded.

    The manuscript claims every reported p-value belongs to one declared
    multiplicity family. Without this scan, a new recorder could add a p-value
    label and silently falsify that claim while every existing test stayed
    green.
    """
    ledger = load_ledger(PROJECT_ROOT)
    labels = list(ledger.labels()) if hasattr(ledger, "labels") else None
    if labels is None:
        import json

        labels = list(json.loads((PROJECT_ROOT / "data" / "claims.json").read_text())["claims"])
    p_labels = sorted(_p_value_labels(labels))
    assert len(p_labels) >= len(SIGNIFICANCE_FAMILY), (
        "scan found fewer p-value labels than the family declares — the scan "
        "itself is broken, not the ledger"
    )
    excluded = {label for label, _reason in EXCLUDED_FROM_FAMILY}
    orphans = [
        label for label in p_labels if label not in SIGNIFICANCE_FAMILY and label not in excluded
    ]
    assert orphans == [], (
        f"p-value labels outside the declared family and the documented "
        f"exclusions: {orphans}. Either add them to SIGNIFICANCE_FAMILY (the "
        f"corrected threshold changes) or document the exclusion here."
    )


def test_every_bench_p_value_a_recorder_can_produce_is_family_or_excluded():
    """Closure stated over the recorders, not over the ledger.

    ``test_family_is_closed_over_the_ledger`` scans the labels that exist. That
    is one direction, and it cannot see a row the ledger is also missing: the
    alphabetic and prompt-format arms carried an exclusion for one probe and none
    for the other, and every check stayed green because each was comparing one
    hand-maintained list against another that had drifted the same way. The
    missing rows would have surfaced only when those p-values landed — that is,
    when it was too late for the threshold to be right.

    So the expected set is generated: each benchmark recorder declares the probes
    it serves and the family templates it writes, and their product is every
    bench p-value label the repository can produce. Each must be in the corrected
    family or carry a documented exclusion, *before* it is measured.

    The rule is stated per recorder rather than inferred from the ledger. Both
    current recorders serve the shared registry, but producer-owned expansion is
    still what catches a recorder that silently drops a probe before a missing
    label can reach the ledger.
    """
    import importlib.util

    expected: set[str] = set()
    serves: dict[str, tuple[str, ...]] = {}
    for path in _bench_recorder_paths():
        name = path.stem
        spec = importlib.util.spec_from_file_location(f"_{name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        probes = tuple(sorted(module.MODELS))
        serves[name] = probes
        expected |= {
            template.format(model=probe)
            for probe in probes
            for template in _declared_family_templates(module)
        }

    assert expected, "no recorder declared a family template — the scan is broken"
    excluded = {label for label, _reason in EXCLUDED_FROM_FAMILY}
    missing = sorted(
        label for label in expected if label not in SIGNIFICANCE_FAMILY and label not in excluded
    )
    assert missing == [], (
        f"these bench p-values can be written by a recorder but are in neither the "
        f"corrected family nor the documented exclusions: {missing}. Probe sets in "
        f"play: {serves}. A p-value a recorder can produce and nothing corrects is "
        f"a threshold that is wrong the moment the run lands, not when someone "
        f"notices."
    )


def test_no_exclusion_pre_authorises_a_label_the_ledger_does_not_have():
    """A stale exclusion silently widens the allowance for a future p-value.

    ``EXCLUDED_FROM_FAMILY`` is checked one way — every ledger p-value must be in
    the family or here. Nothing checked the mirror: an entry left behind after
    its measurement was retired sits waiting, and a later recorder that happens
    to reuse the name inherits a documented-looking exemption it never argued
    for. This is the same closure gap as the one the scan itself had, in the
    other direction.
    """
    ledger = load_ledger(PROJECT_ROOT)
    labels = list(ledger.labels()) if hasattr(ledger, "labels") else None
    if labels is None:
        import json

        labels = list(json.loads((PROJECT_ROOT / "data" / "claims.json").read_text())["claims"])
    recorded = set(labels)
    assert recorded, "no ledger labels loaded; this scan would be vacuous"

    stale = sorted(label for label, _reason in EXCLUDED_FROM_FAMILY if label not in recorded)
    assert stale == [], (
        f"these exclusions name labels absent from the ledger: {stale}. Either the "
        f"measurement was retired and the exclusion should go with it, or the label "
        f"was renamed and the exclusion now pre-authorises a name nobody has argued "
        f"for. Remove the entry rather than leaving it to catch a future p-value."
    )


def test_the_prose_survivor_list_matches_the_recorded_one():
    """The enumeration drifted to five while the count token rendered eight.

    The survivor *count* is a token and could not go stale. The survivor *names*
    were typed, and three of them were dropped as the family grew, so the
    sentence read "8 of 16 cross it" and then listed five. A peer reviewer
    caught it; a token cannot catch it, because prose naming labels in backticks
    is exactly the case the token mechanism does not cover.

    Since decision 110 the names are no longer typed either: the passage
    renders `{{STAT_FAMILY_SURVIVOR_LIST_PROSE}}`, derived by the
    manuscript-variables generator from the recorded list. So the check has two
    halves: the source passage must carry the token rather than typed names,
    and the token's rendering must enumerate exactly the recorded set.

    NEGATIVE CONTROL: the recorded set must be non-empty and the rendering must
    actually enumerate, or this passes on a passage that lists nothing.
    """
    import json
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    claims = json.loads((root / "data" / "claims.json").read_text(encoding="utf-8"))["claims"]
    recorded = set(claims["stat_family_survivors"]["value"])
    assert recorded, "no survivors recorded; this check would be vacuous"

    # Scanned across sections rather than in one named file, for the reason
    # `test_observed_power_is_not_reported.py` gives: the enumeration moved into
    # `98_supplement_statistics.md`, and a filename-bound check would read that
    # as the passage having been deleted.
    prose = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((root / "docs" / "manuscript").glob("[0-9]*.md"))
    )
    sentence = re.search(
        r"declared tests cross it, the recorded survivor set\s+being (.*?)\. The prefix",
        prose,
        re.S,
    )
    assert sentence, "the survivor enumeration is gone; update this test with the new wording"
    assert sentence.group(1).strip() == "{{STAT_FAMILY_SURVIVOR_LIST_PROSE}}", (
        "the survivor set must be rendered from the token, not typed"
    )
    assert not re.findall(r"`([a-z0-9_]+_p)`", sentence.group(1)), "typed survivor names"

    from ailm.manuscript_variables import generate_manuscript_variables

    rendered = generate_manuscript_variables(root)["STAT_FAMILY_SURVIVOR_LIST_PROSE"]
    enumerated = set(re.findall(r"`([a-z0-9_]+)`", rendered))
    assert enumerated, "the token enumerates no labels"
    assert enumerated == recorded, (
        f"token enumerates {sorted(enumerated)} but the ledger records "
        f"{sorted(recorded)}; missing {sorted(recorded - enumerated)}, "
        f"extra {sorted(enumerated - recorded)}"
    )
