"""Interpretive sentences in the external-benchmark supplement, re-derived from the ledger.

What this file protects
-----------------------
Every *number* in ``docs/manuscript/98_supplement_external_benchmarks.md`` is a
``{{TOKEN}}`` resolved from ``data/claims.json``, so a re-run of
``scripts/record_external_benchmark_claims.py`` against a live Ollama endpoint
updates the manuscript automatically. The *readings* of those numbers are
English: "Two of the probes score on almost none of them", "the instruct
probes' context-window contrasts clear it", "the headline comparison is the one
less favorable to AILM". Prose does not re-render. A recorded value that moves
therefore leaves the digits correct and the sentence around them false, with
nothing in the suite noticing — the exact failure this file exists to catch.

Each test below re-derives one such sentence from the ledger and names, on
failure, the sentence that has stopped being true and the file it lives in.
The tests do not check that a number is right; ``tests/test_benchmarks.py``,
``tests/test_calibration_floor.py`` and ``tests/test_manuscript_variables.py``
own the measurements, their derivations and their provenance. What is owned
here is the English wrapped around them.

Two failure modes are guarded in both directions:

* **The number moved.** Every assertion is a comparison re-derived from the
  ledger, never a pinned digit, so it tracks a re-recorded run and fires only
  when the *relation* the sentence asserts stops holding.
* **The sentence moved.** ``test_every_guarded_sentence_is_still_in_the_supplement``
  is the positive control for the whole file: if a guarded sentence is reworded
  or deleted, its guard silently becomes an assertion about nothing. That test
  fails first and says which fragment to re-check.

Non-vacuity is enforced by ``_values``, which fails on a missing label rather
than skipping. A renamed claim must break loudly here: a guard that quietly
stops running is indistinguishable from a guard that passes.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import pytest

from ailm.benchmarks.constants import PROBE_DISPLAY_NAMES
from ailm.manuscript_contract import resolve_manuscript_text
from ailm.manuscript_variables import generate_manuscript_variables, paired_power_from_counts
from ailm.statistics import REFERENCE_SHARE, exact_binomial_power

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEDGER = PROJECT_ROOT / "data" / "claims.json"
SUPPLEMENT = PROJECT_ROOT / "docs" / "manuscript" / "98_supplement_external_benchmarks.md"
DECISIONS = PROJECT_ROOT / "docs" / "decisions.md"

#: How the supplement is named in every failure message. The point of a failure
#: here is to send a reader to one sentence in one file, so the path is spelled
#: out rather than left to the test name.
WHERE = "docs/manuscript/98_supplement_external_benchmarks.md"

#: The externally served probes, in the order the supplement tabulates them.
#: AILM is the comparator and is deliberately not a member: several sentences
#: below say "of the three probes", and counting the comparator in would make them
#: true or false for the wrong reason.
#:
#: Retiring a probe is not a bookkeeping change: the declared correction family
#: scales with the probe count, so removing one loosens the Bonferroni threshold
#: every guard here reads p-values against, and it can empty a gate's reject side
#: entirely, which is what sentences about withheld calibration describe.
#: ``test_the_guarded_probe_list_is_the_registered_one`` holds this tuple to the
#: recorder's registry so the two cannot drift.
PROBES = ("gemma3", "qwen25", "smollm")

#: The six paired contrast types the supplement declares as its correction
#: family, one per arm, each measured on every probe.
FAMILY_ARMS = (
    "bench_letter",
    "bench_letter_az",
    "bench_letter_prompt",
    "bench_phoneme",
    "bench_g2p",
    "bench_letter_ctx",
)

_NEEDS_LEDGER = pytest.mark.skipif(
    not LEDGER.exists(), reason="data/claims.json absent; run the recorders"
)

#: The recorder whose registry decides which probes exist at all.
RECORDER = PROJECT_ROOT / "scripts" / "record_external_benchmark_claims.py"


def test_the_guarded_probe_list_is_the_registered_one() -> None:
    """POSITIVE CONTROL for ``PROBES``: this file must guard every measured probe.

    DEFECT: every count in this file — "one of the two", "the six derived
    values", the family inventory — is derived from ``PROBES``. If the recorder
    gained or lost a model and this tuple did not follow, each of those counts
    would keep passing while describing a different set of systems than the one
    the supplement tabulates, and a probe could be measured, published, and
    guarded by nothing at all.

    The registry is read from the recorder rather than restated, so the two
    cannot agree by coincidence.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_bench_recorder_probes", RECORDER)
    assert spec is not None and spec.loader is not None
    recorder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recorder)

    assert set(PROBES) == set(recorder.MODELS), (
        f"this file guards {sorted(PROBES)} while the recorder measures "
        f"{sorted(recorder.MODELS)}. Every 'of the two probes' count below is now "
        f"about a different set of systems than the supplement reports."
    )
    assert len(PROBES) == len(set(PROBES)), "a repeated probe double-counts every guard"
    assert len(FAMILY_ARMS) == len(recorder.FAMILY_TEMPLATES), (
        f"the supplement's membership rule names {len(FAMILY_ARMS)} contrast types; the "
        f"recorder declares {len(recorder.FAMILY_TEMPLATES)}"
    )


@lru_cache(maxsize=1)
def _claims() -> dict[str, dict]:
    """The shipped ledger as a label -> entry mapping."""
    return json.loads(LEDGER.read_text(encoding="utf-8"))["claims"]


def _values(*labels: str) -> tuple[float, ...]:
    """Read recorded values, failing loudly when a label is gone.

    NON-VACUITY GUARD. Every test in this file compares recorded quantities. A
    lookup that returned ``None``, or a test that skipped on a missing label,
    would turn a renamed claim into a green run: the sentence in the supplement
    would still be there, the guard would no longer be checking it, and nothing
    would say so. Absence is therefore a failure naming the labels, so a rename
    is fixed here in the same change that renames.
    """
    claims = _claims()
    assert claims, f"data/claims.json holds no claims; every guard for {WHERE} is vacuous"
    missing = [label for label in labels if label not in claims]
    assert not missing, (
        f"data/claims.json no longer carries {missing}. The sentences in {WHERE} that "
        f"these labels back are now guarded by nothing — repoint this test at the new "
        f"label names rather than deleting the assertion."
    )
    return tuple(float(claims[label]["value"]) for label in labels)


def _labels_present(*labels: str) -> list[str]:
    """Which of these labels the ledger currently carries."""
    claims = _claims()
    return [label for label in labels if label in claims]


@lru_cache(maxsize=1)
def _supplement_text() -> str:
    """The resolved supplement with runs of whitespace collapsed.

    The source is hard-wrapped at the paragraph level, so a sentence a reader
    sees as one line is two or three lines in the file. Searching the raw text
    for a sentence fragment would fail on the wrap rather than on the content.
    """
    variables = generate_manuscript_variables(PROJECT_ROOT)
    resolved = resolve_manuscript_text(SUPPLEMENT.read_text(encoding="utf-8"), variables)
    return " ".join(resolved.split())


# ── The letter task: compliance versus prediction ────────────────────────────


@_NEEDS_LEDGER
def test_two_probes_score_almost_none_of_the_non_alphabetic_targets() -> None:
    """Guards: "Two of the probes score on almost none of them and the remaining
    one on most", and the follow-on "That two probes sit at the floor — one
    instruct, one base — says the floor is about emission behavior rather than
    instruction tuning".

    This sentence previously read "no non-alphabetic target right at all", and
    the census falsified it: at 400 sampled positions the weaker probe scored
    exactly zero, and over the full pool it gets one. It next read "One probe is
    right on almost none of them", and registering the third probe falsified
    that too: the base model joins the instruct one at the compliance floor,
    which is itself informative — a base model emits a space when the stream
    does, so the floor is about emission behavior rather than instruction
    tuning, and the sentence says so.

    DEFECT: a re-recorded run in which either floored probe rises above a
    percent, or the remaining one drops below half, leaves that sentence
    counting probes that no longer behave that way. The counts "two" and "the
    remaining one" are spelled in words, so no token updates them.

    The comparator is excluded on purpose: AILM also scores on these positions,
    and including it would make the split a statement about four systems.
    """
    scores = dict(
        zip(
            PROBES,
            _values(*(f"bench_letter_nonalpha_acc_{probe}" for probe in PROBES)),
            strict=True,
        )
    )
    # NON-VACUITY: the sentence is about accuracy "on exactly those positions",
    # so the arm must be non-empty and must be the complement of the alphabetic
    # one — "Of the {{BENCH_LETTER_POSITIONS}} sampled positions,
    # {{BENCH_LETTER_NONALPHA_POSITIONS}} have a non-alphabetic target".
    (positions, alphabetic, sampled) = _values(
        "bench_letter_nonalpha_positions",
        "bench_letter_az_positions",
        "bench_letter_positions",
    )
    assert positions > 0, (
        f"no non-alphabetic positions were scored, so the sentence in {WHERE} about "
        f"accuracy on exactly those positions describes an empty arm"
    )
    assert positions + alphabetic == sampled, (
        f"{WHERE} splits the {sampled} sampled letter positions into {alphabetic} "
        f"alphabetic and {positions} non-alphabetic targets; those no longer partition "
        f"the sample, so the two arms are not complements and the reading that one "
        f"removes the behavior the other measures does not follow"
    )

    # "Almost none" is derived from the pool, not from an arbitrary epsilon: the
    # smallest non-zero rate possible is 1/positions, so a probe at or below twice
    # that is right on at most a couple of targets in the whole arm. Written this
    # way the threshold follows the sample size instead of going stale with it.
    near_zero = 2.0 / positions
    zeros = sorted(probe for probe, value in scores.items() if value <= near_zero)
    majority = sorted(probe for probe, value in scores.items() if value > 0.5)
    assert len(zeros) == 2, (
        f"{WHERE} says 'Two of the probes score on almost none of them', but "
        f"{len(zeros)} score at or below {near_zero}: {scores}"
    )
    assert len(majority) == 1, (
        f"{WHERE} says 'the remaining one on most' on non-alphabetic "
        f"targets, but {len(majority)} score above half: {scores}"
    )
    assert set(zeros).isdisjoint(majority), (
        "a probe cannot be both the near-zero one and the majority one"
    )
    assert len(zeros) + len(majority) == len(PROBES), (
        f"{WHERE} splits the probes into two right on almost none and one right most "
        f"of the time, which accounts for all {len(PROBES)} of them. It now accounts for "
        f"{len(zeros) + len(majority)}: {scores}"
    )


@_NEEDS_LEDGER
def test_removing_non_alphabetic_targets_reverses_which_probe_looks_best() -> None:
    """Guards: "The full-inventory letter arm therefore measures inventory compliance
    more than next-symbol prediction", and its sharpest statement, "Qwen 2.5 3B
    remains the extreme case in both directions, scoring highest of the probes on
    non-alphabetic targets and lowest once they are excluded."

    DEFECT: the reading rests on one probe topping the full-inventory ranking by
    emitting spaces and falling to last once spaces cannot score. If a re-run
    puts a different probe at the top of either arm, or puts the same probe at
    the top of both, the supplement still says the ranking is a compliance
    artifact while the ledger no longer shows the reversal that established it.
    The named model is checked too: the sentence names Qwen 2.5 3B, and a
    re-run in which some other probe is the extreme case makes it false by
    attribution even if a reversal survives.
    """
    full = dict(
        zip(PROBES, _values(*(f"bench_letter_top1_{probe}" for probe in PROBES)), strict=True)
    )
    alphabetic = dict(
        zip(PROBES, _values(*(f"bench_letter_az_top1_{probe}" for probe in PROBES)), strict=True)
    )
    non_alphabetic = dict(
        zip(
            PROBES,
            _values(*(f"bench_letter_nonalpha_acc_{probe}" for probe in PROBES)),
            strict=True,
        )
    )

    assert len(set(full.values())) == len(full), (
        f"two probes tie on the full-inventory letter arm, so 'which probe looks best' "
        f"is not defined and the reversal claim in {WHERE} cannot be evaluated: {full}"
    )
    assert len(set(alphabetic.values())) == len(alphabetic), (
        f"two probes tie on the alphabetic letter arm: {alphabetic}"
    )

    best_full = max(full, key=lambda probe: full[probe])
    worst_alphabetic = min(alphabetic, key=lambda probe: alphabetic[probe])
    best_non_alphabetic = max(non_alphabetic, key=lambda probe: non_alphabetic[probe])

    assert best_full == worst_alphabetic, (
        f"{WHERE} reads the full-inventory letter ranking as a compliance artifact "
        f"because the best probe there is the worst once non-alphabetic targets are "
        f"removed. The best full-inventory probe is now {best_full} and the worst "
        f"alphabetic one is {worst_alphabetic}: full={full}, alphabetic={alphabetic}"
    )
    assert best_full == best_non_alphabetic == "qwen25", (
        f"{WHERE} names Qwen 2.5 3B as 'the extreme case in both directions'. The "
        f"top scorer is now {best_full} on the full inventory and "
        f"{best_non_alphabetic} on non-alphabetic targets: "
        f"full={full}, non_alphabetic={non_alphabetic}"
    )


@_NEEDS_LEDGER
def test_both_alphabetic_contrasts_clear_the_threshold_in_the_comparators_favor() -> None:
    """Guards: "every alphabetic-arm contrast clears it, and the discordant pairs
    run in the comparator's favor in every one", and the conclusion's "The
    separation does survive on alphabetic targets alone ... on effects larger
    than the reference share."

    DEFECT: this is the supplement's one surviving letter finding. If a re-run
    moved any alphabetic p-value above the corrected threshold, or flipped a
    discordant majority to the probe, the section would still claim the
    separation survives where compliance cannot act.

    The b/c orientation is not assumed. The recorder writes the paired
    difference as ``(b - c) / positions`` on the full-inventory arm, so that
    identity is re-derived first; only then does ``c > b`` mean the comparator's
    favor rather than the probe's.
    """
    (alpha, positions, reference_share) = _values(
        "bench_family_alpha", "bench_letter_positions", "bench_power_reference_share"
    )
    assert 0.0 < alpha < 1.0, "the corrected threshold is not a probability"

    for probe in PROBES:
        b, c, delta = _values(
            f"bench_letter_paired_b_{probe}",
            f"bench_letter_paired_c_{probe}",
            f"bench_letter_delta_{probe}",
        )
        assert delta == pytest.approx((b - c) / positions, abs=1e-12), (
            f"{probe}: the recorded probe-minus-AILM difference is no longer "
            f"(b - c) / positions, so 'the discordant pairs run in the comparator's "
            f"favor' in {WHERE} can no longer be read off c > b"
        )

    for probe in PROBES:
        b, c, p = _values(
            f"bench_letter_az_paired_b_{probe}",
            f"bench_letter_az_paired_c_{probe}",
            f"bench_letter_az_paired_p_{probe}",
        )
        assert b + c > 0, f"{probe}: the alphabetic arm recorded no discordant pairs at all"
        assert p < alpha, (
            f"{WHERE} says every alphabetic-arm contrast clears the corrected "
            f"threshold {alpha}. {probe} is at p = {p}, so the section's only "
            f"surviving letter finding no longer holds."
        )
        assert c > b, (
            f"{WHERE} says the discordant pairs run in the comparator's favor in every "
            f"alphabetic contrast. {probe} now has b = {b} against c = {c}."
        )
        assert c / (b + c) > reference_share, (
            f"{WHERE} says the alphabetic contrasts clear the threshold 'on effects "
            f"larger than the reference share' of {reference_share}. {probe}'s "
            f"discordant share is {c / (b + c)}."
        )


@_NEEDS_LEDGER
def test_the_two_sensitivity_arms_resolve_as_the_supplement_now_says() -> None:
    """Guards: "The Gemma 3 4B contrast clears the corrected family threshold ...
    the Qwen 2.5 3B and SmolLM contrasts do not clear it and stay unresolved"
    (prompt format), and "The Gemma 3 4B and Qwen 2.5 3B contrasts clear the
    corrected family threshold; the SmolLM contrast does not and is unresolved"
    (context window).

    This guard used to assert the opposite, and the census falsified it. At 400
    sampled positions all six contrasts were unresolved and the supplement said
    so; over the full pool three of the six resolve. The sentences were rewritten
    and this test with them, in that order — the prose is what the ledger has to
    agree with, not the reverse.

    DEFECT, and it now points both ways. A later run that pushed the Qwen
    prompt-format contrast below the threshold would make "stays unresolved"
    false; one that pushed any of the other three above it would make "clears"
    false. Neither direction announces itself, and a newly-significant result is
    the one nobody re-reads, so each is asserted explicitly rather than as a
    blanket rule over the arm.

    The retirement of a probe pushes on these thresholds by itself: a smaller
    family is a looser correction, so a contrast can change verdict with no
    measurement changing. That is why the threshold is read from the ledger here
    rather than written into the assertion.
    """
    (alpha,) = _values("bench_family_alpha")

    #: (arm, probe, must_clear) — the verdict each sentence in the supplement now
    #: states, one row per contrast so a flip in either direction is named.
    expected = (
        ("bench_letter_prompt", "gemma3", True),
        ("bench_letter_prompt", "qwen25", False),
        ("bench_letter_prompt", "smollm", False),
        ("bench_letter_ctx", "gemma3", True),
        ("bench_letter_ctx", "qwen25", True),
        ("bench_letter_ctx", "smollm", False),
    )
    for arm, probe, must_clear in expected:
        b, c, p_value = _values(
            f"{arm}_paired_b_{probe}",
            f"{arm}_paired_c_{probe}",
            f"{arm}_paired_p_{probe}",
        )
        assert b + c > 0, f"{arm}/{probe}: no discordant pairs, so the arm measured nothing"
        clears = p_value < alpha
        assert clears == must_clear, (
            f"{WHERE} states that {arm} on {probe} "
            f"{'clears' if must_clear else 'does not clear'} the corrected family "
            f"threshold of {alpha}, but p = {p_value}. Rewrite the sentence rather "
            f"than the threshold."
        )


@_NEEDS_LEDGER
def test_the_phoneme_and_g2p_contrasts_clear_the_corrected_threshold() -> None:
    """Guards: "The phoneme and G2P contrasts do not share the compliance weakness",
    which the section relies on to hold two decisive arms while the letter
    headline is discounted.

    DEFECT: the letter arm is discounted throughout on the grounds that phoneme
    and G2P are not. If a re-run left one of those four contrasts short of the
    corrected threshold, the section would be discounting its weak arm against
    arms that had stopped being decisive.
    """
    (alpha,) = _values("bench_family_alpha")
    for arm in ("bench_phoneme", "bench_g2p"):
        for probe in PROBES:
            b, c, p = _values(
                f"{arm}_paired_b_{probe}",
                f"{arm}_paired_c_{probe}",
                f"{arm}_paired_p_{probe}",
            )
            assert b + c > 0, f"{arm}/{probe}: no discordant pairs, so the arm measured nothing"
            assert p < alpha, (
                f"{WHERE} treats the phoneme and G2P arms as the section's decisive "
                f"contrasts. {arm} on {probe} is at p = {p} against a corrected threshold "
                f"of {alpha}, so it no longer clears."
            )
            assert c > b, (
                f"{arm}/{probe}: the discordant majority no longer favors the comparator "
                f"(b = {b}, c = {c}), so the sign of the decisive contrast has changed"
            )


# ── The declared correction family ───────────────────────────────────────────


@_NEEDS_LEDGER
def test_the_declared_family_is_exactly_six_contrasts_on_each_probe() -> None:
    """Guards: "They are corrected within their own declared family of
    {{BENCH_FAMILY_SIZE}} tests at a Bonferroni threshold of
    {{BENCH_FAMILY_ALPHA}} ... The membership rule is six contrast types on each
    of the two probes", and the synthetic arm's "corrected inside its own
    declared family of {{BENCH_SYNTH_FAMILY_SIZE}} test ... not inside the
    {{BENCH_FAMILY_SIZE}}-test external-reference family used above."

    DEFECT: the family size is a token, so adding a seventh paired arm updates
    the digit in the manuscript but not the membership *rule* spelled out in
    words — and, worse, silently loosens every threshold in the section if the
    recorder's declared size is not updated with it. Retiring a probe does the
    same thing from the other side: it removes six tests, loosens the threshold
    by half again, and leaves every "clears nothing" sentence in the section
    reading against a correction that no longer applies. The check is therefore
    on the recorded p-value inventory rather than on the number: the ledger must
    hold one paired p-value per named arm per probe, and no others.
    """
    (size, alpha, power_alpha) = _values(
        "bench_family_size", "bench_family_alpha", "bench_power_alpha"
    )
    (synth_size, synth_alpha, context_size, context_alpha) = _values(
        "bench_synth_family_size",
        "bench_synth_family_alpha",
        "bench_synth_context_family_size",
        "bench_synth_context_family_alpha",
    )

    # Membership is by RECORDED SOURCE, not by prefix. As a prefix rule with a
    # `bench_synth_` exclusion, this collected the p-values of every future
    # benchmark family into the external one: the in-context arm's then-declared
    # contrasts (decision 51) landed here and read as unexpected members
    # of a family they were never in, while the real defect the test exists to
    # catch — an arm added to or dropped from the EXTERNAL recorder — would
    # have been buried among them. A family is the set of tests one recorder
    # declared, and the ledger records which recorder wrote each label.
    recorded = {
        label
        for label, entry in _claims().items()
        if "_paired_p_" in label
        and Path(str(entry.get("source", ""))).name == "record_external_benchmark_claims.py"
    }
    assert recorded, "no bench_*_paired_p_* labels found; this inventory check is vacuous"
    expected = {f"{arm}_paired_p_{probe}" for arm in FAMILY_ARMS for probe in PROBES}
    assert recorded == expected, (
        f"{WHERE} states the family membership rule as {len(FAMILY_ARMS)} contrast types "
        f"on each of {len(PROBES)} probes. The ledger's paired p-values do not match "
        f"that rule: "
        f"unexpected {sorted(recorded - expected)}, missing {sorted(expected - recorded)}. "
        f"An arm added or dropped here changes every threshold in the section."
    )
    assert size == len(expected), (
        f"{WHERE} declares a family of {size} tests, but {len(expected)} paired contrasts "
        f"are recorded under the stated membership rule"
    )
    assert alpha == pytest.approx(power_alpha / size, rel=1e-12), (
        f"the declared threshold {alpha} is not the Bonferroni correction of "
        f"{power_alpha} over {size} tests, which is what {WHERE} says it is"
    )

    synth_recorded = {label for label in _claims() if label.startswith("bench_synth_paired_p_")}
    expected_synth = {f"bench_synth_paired_p_{probe}" for probe in PROBES}
    assert synth_recorded == expected_synth, (
        "the synthetic producer must record one paired contrast for every registered "
        f"probe; unexpected {sorted(synth_recorded - expected_synth)}, "
        f"missing {sorted(expected_synth - synth_recorded)}"
    )
    assert len(synth_recorded) == synth_size == len(PROBES), (
        f"{WHERE} says the synthetic contrasts form their own registered-probe family; "
        f"the ledger declares {synth_size} and records {sorted(synth_recorded)}"
    )
    assert synth_alpha == pytest.approx(power_alpha / synth_size, rel=1e-12), (
        f"the synthetic threshold {synth_alpha} is not the Bonferroni correction of "
        f"{power_alpha} over {synth_size} tests, which is what {WHERE} says it is"
    )
    assert synth_alpha != alpha, (
        f"{WHERE} insists the two families are corrected separately, but both thresholds "
        f"are now {alpha}"
    )

    context_recorded = {
        label for label in _claims() if label.startswith("bench_synth_context_paired_p_")
    }
    expected_context = {f"bench_synth_context_paired_p_{probe}" for probe in PROBES}
    assert context_recorded == expected_context, (
        "the audit-specified query-window sensitivity family drifted: "
        f"unexpected {sorted(context_recorded - expected_context)}, "
        f"missing {sorted(expected_context - context_recorded)}"
    )
    assert len(context_recorded) == context_size == len(PROBES)
    assert context_alpha == pytest.approx(power_alpha / context_size, rel=1e-12)


# ── Detectability ────────────────────────────────────────────────────────────


@_NEEDS_LEDGER
def test_the_largest_derived_power_is_the_alphabetic_smollm_arm() -> None:
    """Guards: "The largest of the nine derived values is
    {{BENCH_LETTER_AZ_POWER_SMOLLM}} ... every remaining derived value is no
    larger", repeated in the conclusion as "The largest of those figures is
    {{BENCH_LETTER_AZ_POWER_SMOLLM}}."

    DEFECT: the supplement quotes one of nine derived powers and asserts it is
    the maximum. Exact binomial power is not monotone in the pair count — the
    recorded letter arm already carries a contrast with more discordant pairs
    and less power than one of its neighbors — so a re-run can move the maximum
    to another arm while every token still renders a plausible number. The
    sentence would then quote a figure that is not the largest and draw a bound
    from it.

    At the sampled size these arms were underpowered and the supplement said so;
    at the census pool they are not, and the sentence that survives is the
    narrower bound-from-the-largest one guarded here. The adequacy of the
    alphabetic arms is a separate reading with its own guard below.
    """
    arms = ("bench_letter_az", "bench_letter_prompt", "bench_letter_ctx")
    derived: dict[tuple[str, str], float] = {}
    for arm in arms:
        for probe in PROBES:
            b, c = _values(f"{arm}_paired_b_{probe}", f"{arm}_paired_c_{probe}")
            derived[(arm, probe)] = paired_power_from_counts(b, c)

    assert len(derived) == len(arms) * len(PROBES) == 9, (
        f"the supplement counts nine derived values, one per arm per probe; found "
        f"{len(derived)} across {len(arms)} arms and {len(PROBES)} probes"
    )
    assert len(set(derived.values())) > 1, (
        "every derived power is identical, so 'the largest of the six' is not a "
        "statement about the data"
    )

    largest = max(derived, key=lambda key: derived[key])
    assert largest == ("bench_letter_az", "smollm"), (
        f"{WHERE} quotes BENCH_LETTER_AZ_POWER_SMOLLM as the largest of the nine derived "
        f"power figures. The largest is now {largest[0]}/{largest[1]} at "
        f"{derived[largest]}; the sentence cites a figure that no longer bounds the rest."
    )
    # The upper-bound assertion, no longer a low-power one. At the sampled size
    # every derived figure sat far below adequacy and the supplement said so; the
    # census resolved these arms, so the sentence that survives is the narrower
    # one — this figure bounds the other five. Asserting the old ceiling here
    # would now fail on a run that got *better*, which is the wrong direction for
    # a guard to point.
    assert all(value <= derived[largest] for value in derived.values()), (
        f"{WHERE} quotes this figure as bounding the nine derived powers, but "
        f"{derived} contains a larger one."
    )


#: The conventional adequacy level for power. A reading of "well powered" that
#: named a different level would be a different sentence.
_ADEQUATE_POWER = 0.8

#: Wording retired when the alphabetic arms stopped being underpowered. Kept
#: here so the guard can say which sentence must not come back, and nowhere in
#: the supplement.
_RETIRED_LOW_POWER_WORDING = ("low even so", "could not reliably have found")


@_NEEDS_LEDGER
def test_the_alphabetic_arms_are_well_powered_at_the_reporting_alpha() -> None:
    """Guards: "The alphabetic arms are well powered at the reporting level
    {{BENCH_POWER_ALPHA}} ... so each arm could reliably have found a
    reference-sized effect at that level".

    DEFECT: the sentence this replaced said the opposite — "Detectability here
    is low even so ... the arm could not reliably have found an effect the size
    of the reference share" — and stayed in the supplement after the census
    pool moved every alphabetic arm's derived power far above adequacy. Power is
    a derived token, so the digits updated and the reading around them did
    not. The claim is re-derived at both thresholds from the recorded pair
    counts: adequate at the reporting alpha for every probe, and still above
    one half at the corrected family threshold, so the sentence would be wrong
    at neither.
    """
    (power_alpha, family_alpha) = _values("bench_power_alpha", "bench_family_alpha")
    for probe in PROBES:
        b, c = _values(f"bench_letter_az_paired_b_{probe}", f"bench_letter_az_paired_c_{probe}")
        pairs = int(b + c)
        at_reporting = paired_power_from_counts(b, c)
        assert at_reporting == pytest.approx(
            exact_binomial_power(pairs, REFERENCE_SHARE, power_alpha), rel=1e-12, abs=1e-12
        ), f"{probe}: the derived power is not the power at the reporting alpha {power_alpha}"
        assert at_reporting >= _ADEQUATE_POWER, (
            f"{WHERE} says the alphabetic arms are well powered at {power_alpha}; the "
            f"{probe} arm's derived power is {at_reporting} on {pairs} discordant pairs, "
            f"below {_ADEQUATE_POWER}. The retired 'could not reliably have found' reading "
            f"is the true one again and the sentence must change back."
        )
        at_corrected = exact_binomial_power(pairs, REFERENCE_SHARE, family_alpha)
        assert at_corrected > 0.5, (
            f"{WHERE} says only that power at the corrected threshold {family_alpha} is "
            f"lower; for {probe} it is {at_corrected}, below one half, which the section "
            f"would have to call underpowered rather than merely lower."
        )

    text = _supplement_text()
    revived = [fragment for fragment in _RETIRED_LOW_POWER_WORDING if fragment in text]
    assert revived == [], (
        f"{WHERE} again describes an alphabetic arm as underpowered ({revived}); the "
        f"derived powers contradict it."
    )


def test_the_well_powered_guard_rejects_an_underpowered_arm() -> None:
    """NEGATIVE CONTROL: the adequacy check must be able to fail.

    A pair count small enough that a reference-sized effect is undetectable at
    the reporting alpha must fall below the level the guard asserts; otherwise
    the guard above would pass on any ledger.
    """
    (power_alpha,) = _values("bench_power_alpha")
    tiny = exact_binomial_power(8, REFERENCE_SHARE, power_alpha)
    assert tiny < _ADEQUATE_POWER
    assert paired_power_from_counts(4, 4) == pytest.approx(tiny, rel=1e-12, abs=1e-12)


@_NEEDS_LEDGER
def test_every_power_figure_overstates_the_detectability_of_the_rule_in_force() -> None:
    """Guards: "Power falls as the threshold tightens, so every power figure in this
    section, recorded or derived, is an upper bound on the detectability of the
    rule actually in force", and "at the letter arm's discordance rate, a pair
    count that would be well powered at {{BENCH_POWER_ALPHA}} loses a substantial
    part of that power at {{BENCH_FAMILY_ALPHA}}."

    DEFECT: both sentences are relations between two recorded alphas. If the
    family size changed so that the corrected threshold approached the reporting
    alpha, "upper bound" would become an equality and "the gap between the two is
    wide rather than incidental" would be false — while every power token in the
    section still rendered the same digits, because they are all computed at the
    reporting alpha.

    The recorded powers are re-derived at both thresholds from the same
    discordant counts, which also confirms the recorded value is the one at the
    reporting alpha rather than at the corrected one.
    """
    (power_alpha, family_alpha) = _values("bench_power_alpha", "bench_family_alpha")
    assert family_alpha < power_alpha, (
        f"the corrected threshold {family_alpha} is not tighter than the reporting alpha "
        f"{power_alpha}, so nothing in {WHERE} about power falling as the threshold "
        f"tightens applies"
    )

    for arm in ("bench_letter", "bench_phoneme", "bench_g2p"):
        for probe in PROBES:
            b, c, recorded = _values(
                f"{arm}_paired_b_{probe}",
                f"{arm}_paired_c_{probe}",
                f"{arm}_power_{probe}",
            )
            pairs = int(b + c)
            at_reporting = exact_binomial_power(pairs, REFERENCE_SHARE, power_alpha)
            at_corrected = exact_binomial_power(pairs, REFERENCE_SHARE, family_alpha)
            assert at_reporting == pytest.approx(recorded, rel=1e-12, abs=1e-12), (
                f"{arm}/{probe}: the recorded power {recorded} is not the power at the "
                f"reporting alpha {power_alpha}, so {WHERE} calling it an upper bound on "
                f"detectability at {family_alpha} describes a different quantity"
            )
            assert at_corrected < recorded, (
                f"{arm}/{probe}: power at the corrected threshold ({at_corrected}) is not "
                f"below the reported figure ({recorded}); {WHERE} says every power figure "
                f"is an upper bound on the rule actually in force"
            )

    # The wide-gap sentence, re-derived rather than pinned: take the smallest
    # pair count that would be well powered at the reporting alpha and show it
    # loses a substantial part of that power at the corrected one.
    well_powered = next(
        pairs
        for pairs in range(2, 4000)
        if exact_binomial_power(pairs, REFERENCE_SHARE, power_alpha) >= 0.8
    )
    tightened = exact_binomial_power(well_powered, REFERENCE_SHARE, family_alpha)
    assert tightened < 0.5, (
        f"{WHERE} says the gap between the two thresholds is 'wide rather than "
        f"incidental': a pair count well powered at {power_alpha} ({well_powered} pairs, "
        f"power >= 0.8) retains {tightened} at {family_alpha}, which is no longer a "
        f"substantial loss"
    )


@_NEEDS_LEDGER
def test_the_letter_pool_is_larger_than_the_reference_share_needs() -> None:
    """Guards: "The letter contrast is therefore no longer unresolved by construction",
    which rests on the fixture "holding {{BENCH_LETTER_POOL_UNIQUE}} distinct
    positions, above the number the reference share needs at
    {{BENCH_POWER_ALPHA}}."

    DEFECT: the earlier revision's letter arm could not reach a reference-sized
    effect at any sampling, because the pool was too small. That defect is
    structural and returns silently if the fixture is ever shortened: the pool
    token would update, the power tokens would update, and the sentence saying
    the bound is no longer structural would stay.

    The needed pool is derived from the ledger's own discordance rates rather
    than assumed: at the lowest rate any probe records, the pool must still hold
    enough positions to yield a well-powered pair count at the reporting alpha.
    """
    (pool_total, pool_unique, positions, power_alpha) = _values(
        "bench_letter_pool_total",
        "bench_letter_pool_unique",
        "bench_letter_positions",
        "bench_power_alpha",
    )
    assert pool_unique <= pool_total, (
        "more distinct positions than scored positions is impossible; the deduplication "
        f"described in {WHERE} cannot produce this"
    )
    assert positions <= pool_unique, "more positions were sampled than the pool holds"

    rates = []
    for probe in PROBES:
        b, c = _values(f"bench_letter_paired_b_{probe}", f"bench_letter_paired_c_{probe}")
        rates.append((b + c) / positions)
    assert min(rates) > 0, "no probe records any discordant pair; the rate is undefined"

    well_powered = next(
        pairs
        for pairs in range(2, 4000)
        if exact_binomial_power(pairs, REFERENCE_SHARE, power_alpha) >= 0.8
    )
    needed = well_powered / min(rates)
    assert pool_unique > needed, (
        f"{WHERE} says the letter fixture holds more distinct positions than the "
        f"reference share needs at {power_alpha}. The pool is {pool_unique} distinct "
        f"positions and the lowest recorded discordance rate needs {needed:.0f}, so the "
        f"letter arm is unresolved by construction again."
    )


# ── Calibration: what the captured-mass gate admits ──────────────────────────


@_NEEDS_LEDGER
def test_the_direct_rule_gate_decides_measurability() -> None:
    """Guards: the supplement's statement that calibration is reported only for
    systems whose own candidates survive the direct dual-rule check (decision
    93), and that a withheld system's figures remain recorded so the gate's
    verdict is recomputable.

    History: this guard once asserted the captured-mass mass floor admitted
    every probe, and before that it asserted a row was absent below the floor.
    Decision 93 replaced the floor with the direct check the floor was a proxy
    for, so the guard now asserts the adopted rule: ``measurable`` equals the
    rule check's verdict, computed here from the recorded divergences and the
    declared epsilon, and the metrics exist for every probe either way.

    DEFECT if this were merely deleted: a future gate that diverges from the
    recorded rule divergences — in either direction — would move which systems
    appear in the calibration table while the prose went on describing the
    adopted rule.
    """
    (epsilon,) = _values("bench_calib_rule_epsilon")
    assert 0.0 < epsilon < 1.0, "the rule tolerance is not a metric-scale quantity"
    ece_div = dict(
        zip(
            PROBES,
            _values(*(f"bench_calib_rule_ece_divergence_{probe}" for probe in PROBES)),
            strict=True,
        )
    )
    brier_div = dict(
        zip(
            PROBES,
            _values(*(f"bench_calib_rule_brier_divergence_{probe}" for probe in PROBES)),
            strict=True,
        )
    )
    measurable = dict(
        zip(PROBES, _values(*(f"bench_calib_measurable_{probe}" for probe in PROBES)), strict=True)
    )

    for probe in PROBES:
        verdict = float(max(ece_div[probe], brier_div[probe]) <= epsilon)
        assert measurable[probe] == verdict, (
            f"{probe}: the recorded measurable flag {measurable[probe]} disagrees with the "
            f"direct dual-rule gate (ECE divergence {ece_div[probe]}, Brier divergence "
            f"{brier_div[probe]}, epsilon {epsilon})"
        )

    # The five statistics the calibration table carries per row are recorded for
    # every probe whether the gate admits it or not: a withheld calibration is
    # an audited one.
    statistics = ("top1", "confidence", "ece", "brier", "nll")
    _values(
        *(f"bench_calib_{name}_{system}" for name in statistics for system in ("ailm", *PROBES))
    )

    # The five statistics the calibration table carries per row.
    statistics = ("top1", "confidence", "ece", "brier", "nll")
    # POSITIVE CONTROL: every row the table reports is recorded, so a wholesale
    # rename of the calibration namespace fails here rather than reading as a
    # clean set of absences.
    _values(
        *(f"bench_calib_{name}_{system}" for name in statistics for system in ("ailm", *PROBES))
    )

    stray = sorted(
        label
        for label in _claims()
        if label.startswith("bench_calib_")
        and label.rsplit("_", 1)[-1] not in {"ailm", *PROBES}
        and any(label.startswith(f"bench_calib_{name}_") for name in statistics)
    )
    assert stray == [], (
        f"the calibration table reports AILM and {sorted(PROBES)}; the ledger also carries "
        f"{stray}. A retired probe's numbers are still being written, so the table in "
        f"{WHERE} and the ledger disagree about which systems were scored."
    )

    agreement_present = _labels_present(
        *(f"bench_calib_argmax_agreement_{probe}" for probe in PROBES)
    )
    expected_agreement = [f"bench_calib_argmax_agreement_{probe}" for probe in PROBES]
    assert agreement_present == expected_agreement, (
        f"{WHERE} reports the argmax agreement for every reportable system. The ledger "
        f"carries {agreement_present} against the expected {expected_agreement}."
    )


@_NEEDS_LEDGER
def test_the_reconstructed_argmax_agrees_with_the_scored_answer_almost_always() -> None:
    """Guards: "On this run they agree on {{...}} and {{...}} of positions for the two
    reportable systems. That is near-total, and it is not guaranteed."

    DEFECT: the supplement scores calibration on the returned answer rather than
    on the reconstructed argmax, and justifies reporting the two side by side by
    saying they nearly always coincide. If a re-run dropped agreement to, say,
    four fifths, "near-total" would be false and the choice between the two
    scoring rules would start to matter to the published numbers.

    "Near-total" is read as at least 0.95. The sentence's own hedge — agreement
    "is a property of the decode rather than of the method" — is exactly why the
    reading needs a number a re-run can fail.
    """
    for probe in PROBES:
        (agreement,) = _values(f"bench_calib_argmax_agreement_{probe}")
        assert 0.0 <= agreement <= 1.0, f"{probe}: agreement {agreement} is not a fraction"
        assert agreement >= 0.95, (
            f"{WHERE} calls the agreement between the reconstructed argmax and the scored "
            f"answer 'near-total'. {probe} is now at {agreement}, so the section has to "
            f"say which of the two the numbers came from rather than that it does not "
            f"much matter."
        )


@_NEEDS_LEDGER
def test_the_calibration_table_contains_exactly_the_gate_admitted_systems() -> None:
    """Guards, under the adopted decision-93 gate: "the calibration table above
    already reports under the adopted gate: Gemma 3 4B and SmolLM are withheld
    and Qwen 2.5 3B reports. `CALIBRATION_MASS_FLOOR` stands in the code as
    provenance only; it no longer decides anything."

    History, kept because it is why this guard looks the way it does: under the
    pre-93 gate the captured-mass floor decided measurability, and this test
    asserted that every measured probe cleared it. Registering the third probe
    broke that premise — the floor rejected smollm, the first probe a floor ever
    rejected — and decision 93 then retired the premise rather than restoring
    it: the gate is now the direct dual-rule divergence check, and the floor no
    longer decides measurability.

    DEFECT, restated for the adopted gate: if the calibration table ever carries
    a row for a withheld system, or drops a row for an admitted one, the section
    reports a calibration number the gate did not license — or hides one it did.
    And if a withheld system's figures ever leave the ledger, its verdict stops
    being recomputable and the gate becomes frozen into the record instead of
    auditable against it.

    The floor keeps a provenance role, so it is still checked where it still
    applies: every system the gate admits sits above the floor, and the
    starved-fraction diagnostic that explains each captured mass is recorded for
    every probe, admitted or not.
    """
    (floor,) = _values("bench_calib_mass_floor")
    admitted, withheld = [], []
    for probe in PROBES:
        mean, median = _values(
            f"bench_calib_captured_mass_{probe}", f"bench_calib_captured_median_{probe}"
        )
        (measurable,) = _values(f"bench_calib_measurable_{probe}")
        (admitted if measurable == 1 else withheld).append(probe)
        if measurable == 1:
            assert mean >= floor, (
                f"decision 93 keeps the floor as provenance, so every gate-admitted "
                f"system should still sit above it. {probe} is admitted with a mean "
                f"captured mass of {mean} against a floor of {floor}."
            )
            assert median >= floor, (
                f"{WHERE} reports the median beside the mean so a reader can see how "
                f"much of each figure rests on mostly-reconstructed positions. {probe} "
                f"is admitted on a mean of {mean} while its median is {median}, below "
                f"the floor of {floor}."
            )
        else:
            # Withheld, not deleted: the figures stay in the ledger so the
            # verdict is recomputable from the record.
            _values(f"bench_calib_rule_ece_divergence_{probe}")
            _values(f"bench_calib_rule_brier_divergence_{probe}")
            _values(f"bench_calib_starved_fraction_{probe}")

    assert admitted == ["qwen25"] and withheld == ["gemma3", "smollm"], (
        f"decision 93's calibration table reports exactly the gate-admitted systems: "
        f"admitted={admitted}, withheld={withheld}. A changed split means the table in "
        f"{WHERE} and the gate disagree about which calibrations are published."
    )
    # The floor no longer decides anything, so it must not reproduce the gate's
    # split: gemma3 clears the floor and is withheld anyway, by the divergence
    # check. If the floor's verdict ever coincided with measurability for every
    # probe again, this distinction has collapsed and the prose above it is
    # describing a gate that no longer exists.
    floor_verdicts = {}
    for probe in PROBES:
        (mean,) = _values(f"bench_calib_captured_mass_{probe}")
        floor_verdicts[probe] = mean >= floor
    assert floor_verdicts["gemma3"] and not floor_verdicts["smollm"], (
        f"decision 93's provenance claim rests on the floor and the gate disagreeing "
        f"on gemma3; the floor now says {floor_verdicts}, so the two rules no longer "
        f"come apart and the 'provenance only' sentence is describing nothing."
    )


@_NEEDS_LEDGER
def test_the_spread_case_and_the_concentrated_case_are_the_ones_named() -> None:
    """Guards: "Gemma 3 4B is the spread case, Qwen 2.5 3B the concentrated one,
    and SmolLM the starved one — nearly every position scored from an assumed
    distribution rather than a reported one."

    DEFECT: three named attributions with nothing but prose holding them to the
    ledger. A re-run that concentrated Gemma 3 4B's captured mass, starved part
    of Qwen 2.5 3B's, or spread SmolLM's, would swap the roles while the
    sentence kept the original assignment — and the surrounding paragraph, which
    exists to explain why a mean alone cannot describe any of the three cases,
    would be illustrating the wrong system with the right numbers.
    """
    gemma_starved, qwen_starved, smollm_starved = _values(
        "bench_calib_starved_fraction_gemma3",
        "bench_calib_starved_fraction_qwen25",
        "bench_calib_starved_fraction_smollm",
    )
    gemma_mean, gemma_median = _values(
        "bench_calib_captured_mass_gemma3", "bench_calib_captured_median_gemma3"
    )
    qwen_mean, qwen_median = _values(
        "bench_calib_captured_mass_qwen25", "bench_calib_captured_median_qwen25"
    )

    assert gemma_starved > 0.0, (
        f"{WHERE} calls Gemma 3 4B the spread case: a minority of near-empty positions "
        f"pulling the mean away from the median. Its starved fraction is now "
        f"{gemma_starved}, so there is no minority to spread."
    )
    assert qwen_starved == 0.0, (
        f"{WHERE} calls Qwen 2.5 3B the concentrated case, where every position is well "
        f"observed and the mean is simply the answer. Its starved fraction is now "
        f"{qwen_starved}."
    )
    assert smollm_starved >= 0.9, (
        f"{WHERE} calls SmolLM the starved one — 'nearly every position scored from "
        f"an assumed distribution rather than a reported one'. Its starved fraction "
        f"is now {smollm_starved}."
    )
    assert (gemma_median - gemma_mean) > (qwen_median - qwen_mean), (
        f"{WHERE} contrasts a mean that 'sits between two clusters and matches neither' "
        f"with one that is 'simply the answer'. The gap between median and mean is now "
        f"{gemma_median - gemma_mean} for Gemma 3 4B and {qwen_median - qwen_mean} for "
        f"Qwen 2.5 3B, so the two cases no longer differ in the stated direction."
    )


@_NEEDS_LEDGER
def test_calibration_is_not_a_restatement_of_accuracy() -> None:
    """Guards: "Calibration is a distinct axis from accuracy rather than a restatement
    of it: a system whose distribution were one-hot at every position would post
    an expected calibration error of exactly one minus its accuracy, and the
    measured values above do not coincide with that."

    DEFECT: the sentence licenses reporting calibration at all, by ruling out the
    degenerate case where the calibration column carries no information the
    accuracy column does not. If a re-run produced a one-hot-like system, the
    section would present a second axis that is arithmetic on the first.
    """
    for system in ("ailm", "gemma3", "qwen25", "smollm"):
        ece, top1 = _values(f"bench_calib_ece_{system}", f"bench_calib_top1_{system}")
        assert ece != pytest.approx(1.0 - top1, abs=1e-6), (
            f"{WHERE} says the measured calibration errors do not coincide with one minus "
            f"accuracy. {system} now posts an expected calibration error of {ece} against "
            f"an accuracy of {top1}, which is the one-hot degenerate case: calibration has "
            f"become a restatement of accuracy for that system."
        )


# ── The captured-mass floor sweep, as the section reads it ───────────────────


@_NEEDS_LEDGER
def test_gemma_sits_inside_the_flagged_band_and_qwen_above_it() -> None:
    """Guards: "Gemma 3 4B's {{...}} sits above the declared floor and below the located
    one, so one of the two reported systems is inside the band this sweep flags;
    Qwen 2.5 3B's {{...}} sits above both."

    DEFECT: two placements on one axis, both stated in words. Qwen 2.5 3B sits
    only slightly above the located floor, so a re-recorded captured mass can put
    a second system inside the flagged band while the sentence still says one is.
    The reverse also matters: if Gemma 3 4B rose above the located floor, the
    subsection's conclusion — that the gate admits a system whose published
    number is partly rule-decided — would no longer have an instance, and this is
    now the only instance the section has. The third probe, whose captured mass
    sat far below every floor and made the placement paragraph's third clause,
    has been retired along with that clause.
    """
    (declared, located) = _values(
        "bench_floor_declared_floor",
        "bench_floor_located_captured",
    )
    gemma, qwen = _values(
        "bench_calib_captured_mass_gemma3",
        "bench_calib_captured_mass_qwen25",
    )
    assert declared < located, (
        "the located floor is not above the declared one, so there is no band to sit "
        f"inside and the whole placement paragraph in {WHERE} is void"
    )

    assert declared < gemma < located, (
        f"{WHERE} places Gemma 3 4B inside the band between the declared floor "
        f"({declared}) and the located one ({located}). Its captured mass is now {gemma}, "
        f"so the section's example of a system admitted at a rule-dependent mass is gone."
    )
    assert qwen > located, (
        f"{WHERE} places Qwen 2.5 3B above both floors. Its captured mass is now {qwen} "
        f"against a located floor of {located}, so both reported systems are inside the "
        f"flagged band and 'one of the two' is wrong."
    )


@_NEEDS_LEDGER
def test_both_first_crossings_sit_just_below_the_declared_floor() -> None:
    """Guards: "Each metric's divergence first drops below {{BENCH_FLOOR_EPSILON}} far
    below the located floor: at {{...}} captured mass for expected calibration
    error and {{...}} for Brier, both just below the declared
    {{BENCH_FLOOR_DECLARED_FLOOR}}."

    DEFECT: "just below the declared" is what makes the first-crossing trap
    interesting — a naive locator would have certified almost exactly the floor
    the project declared, which is why the coincidence is worth a paragraph. If a
    re-swept crossing moved well away from the declared floor, the paragraph
    would still present it as the near-miss that makes the declared value look
    conservative.

    "Just below" is read as within a tenth of the declared floor; "far below the
    located floor" as at least a tenth away from it. Both readings are stated
    here so a re-run can fail them rather than quietly stretch the words.
    """
    (declared, located, epsilon) = _values(
        "bench_floor_declared_floor", "bench_floor_located_captured", "bench_floor_epsilon"
    )
    assert epsilon > 0, "a non-positive divergence threshold locates nothing"
    for metric, label in (
        ("expected calibration error", "bench_floor_naive_ece_captured"),
        ("Brier", "bench_floor_naive_brier_captured"),
    ):
        (crossing,) = _values(label)
        assert crossing < declared, (
            f"{WHERE} says the first crossing in {metric} sits just below the declared "
            f"floor {declared}; it is now at {crossing}"
        )
        assert declared - crossing < 0.1, (
            f"{WHERE} says the first crossing in {metric} is 'just below' the declared "
            f"floor {declared}. It is now {declared - crossing} below it, which is not a "
            f"near miss and does not make the declared floor look conservative."
        )
        assert located - crossing > 0.1, (
            f"{WHERE} says the first crossing in {metric} is far below the located floor "
            f"{located}; it is now at {crossing}, and the trap the locator is built to "
            f"avoid has closed"
        )


@_NEEDS_LEDGER
def test_the_rule_gap_closes_monotonically_but_its_metric_image_does_not() -> None:
    """Guards: "the metric divergences are not monotone in captured mass, rising on
    {{...}} grid steps for expected calibration error and {{...}} for Brier, even
    though the total variation between the two reconstructions rises on
    {{BENCH_FLOOR_TV_INCREASES}}. The distributional gap closes monotonically;
    its image in either metric does not."

    DEFECT: this asymmetry is the whole argument for a sustained crossing rather
    than a first one. All three counts are tokens, so a re-sweep in which the
    metric divergences became monotone would render "rising on 0 grid steps" into
    a sentence that says they are not monotone. The recorded counts must
    therefore be read as the relation the sentence asserts, not as digits.
    """
    (tv_increases, ece_increases, brier_increases, ece_signs, brier_signs) = _values(
        "bench_floor_tv_increases",
        "bench_floor_ece_increases",
        "bench_floor_brier_increases",
        "bench_floor_ece_sign_changes",
        "bench_floor_brier_sign_changes",
    )
    assert tv_increases == 0, (
        f"{WHERE} says the distributional gap closes monotonically; the recorded "
        f"total-variation series now rises on {tv_increases} grid steps"
    )
    for metric, increases in (
        ("expected calibration error", ece_increases),
        ("Brier", brier_increases),
    ):
        assert increases > 0, (
            f"{WHERE} says the {metric} divergence is not monotone in captured mass and "
            f"that this non-monotonicity is what makes a first crossing unsound. It now "
            f"rises on {increases} grid steps, so the sustained-crossing requirement is "
            f"no longer load-bearing and the paragraph arguing for it is wrong."
        )
    for metric, changes in (("expected calibration error", ece_signs), ("Brier", brier_signs)):
        assert changes == 1, (
            f"{WHERE} says each signed difference 'passes through zero exactly once'; the "
            f"{metric} series now records {changes} sign changes"
        )


@_NEEDS_LEDGER
def test_the_located_floor_is_unreachable_through_this_endpoint() -> None:
    """Guards: "Raising the gate to the located floor is not available through this
    endpoint ... The declared floor by contrast needs {{BENCH_FLOOR_ATDECLARED_K}}
    candidates, comfortably inside the cap ... The located floor is consequently
    unreachable in principle here rather than merely unmet by the registered probes."

    DEFECT: "unreachable in principle" is a claim about the server's candidate
    cap, not about the models, and it is the reason the section defers the gate
    change instead of making it. If a re-sweep put the located floor's candidate
    cost inside the cap, or the declared floor's cost outside it, the paragraph
    would keep asserting a constraint that had moved — and the deferral it
    justifies would have lost its justification.
    """
    (located_k, declared_k, cap, inventory, over_cap) = _values(
        "bench_floor_located_k",
        "bench_floor_atdeclared_k",
        "bench_floor_cap",
        "bench_floor_inventory_size",
        "bench_floor_located_over_cap_fraction",
    )
    assert 0 < cap < inventory, (
        f"the server cap {cap} does not truncate a {inventory}-symbol inventory, so "
        f"nothing in {WHERE} about candidates the endpoint cannot return applies"
    )
    assert over_cap > 0.0, (
        f"{WHERE} says reaching the located floor needs more candidates than the cap "
        f"allows on a share of positions; that share is now {over_cap}, so the located "
        f"floor is reachable and 'unreachable in principle' is false"
    )
    assert declared_k < cap, (
        f"{WHERE} says the declared floor's candidate cost sits comfortably inside the cap "
        f"of {cap}; it is now {declared_k}"
    )
    assert located_k > declared_k, (
        f"{WHERE} contrasts the two costs; the located floor now needs {located_k} "
        f"candidates against the declared floor's {declared_k}"
    )


@_NEEDS_LEDGER
def test_the_rule_decided_share_is_a_large_fraction_of_the_recorded_calibration_error() -> None:
    """Guards: "AILM's whole recorded expected calibration error is
    {{BENCH_CALIB_ECE_AILM}}, so a rule-induced disagreement of
    {{BENCH_FLOOR_ATDECLARED_ECE_DIVERGENCE}} is a large fraction of the effect
    this axis reports."

    DEFECT: two tokens and one word. The word carries the subsection's
    conclusion — that a system admitted at the gate could have a substantial part
    of its published calibration number decided by a reconstruction choice — and
    the word does not re-render. Both quantities come from different recorders,
    one live and one offline, so their ratio can move without either looking
    wrong.

    "A large fraction" is read as at least a quarter. Below that the sentence
    would need to be rewritten rather than reinterpreted.
    """
    (divergence, ailm_ece) = _values(
        "bench_floor_atdeclared_ece_divergence", "bench_calib_ece_ailm"
    )
    assert ailm_ece > 0, "AILM records no calibration error, so there is no fraction to take"
    assert divergence / ailm_ece >= 0.25, (
        f"{WHERE} calls the rule-induced disagreement at the declared floor ({divergence}) "
        f"a large fraction of the effect this axis reports (AILM's recorded expected "
        f"calibration error, {ailm_ece}). It is now {divergence / ailm_ece:.3f} of it."
    )


@_NEEDS_LEDGER
def test_the_floor_location_is_not_an_artifact_of_the_sample_size() -> None:
    """Guards: "Repeating the sweep at {{BENCH_FLOOR_CHECK_POSITIONS}} positions puts the
    floor at {{BENCH_FLOOR_CHECK_LOCATED_CAPTURED}}, so the location is not an
    artifact of the sample size a binned statistic is computed at."

    DEFECT: the conclusion is the word "so". Two floors are printed and the
    reader is told they agree. If a re-sweep separated them, both tokens would
    still render and the sentence would be drawing a stability conclusion from
    two numbers that no longer support it — the failure mode is a conclusion
    detaching from its evidence, not a wrong digit.

    "Not an artifact" is read as: the check sweep runs at a different sample
    size, and both floors land above the declared one and within a twentieth of
    each other. A larger gap than that is a sample-size effect being reported as
    its absence.
    """
    (declared, located, check_located, positions, check_positions) = _values(
        "bench_floor_declared_floor",
        "bench_floor_located_captured",
        "bench_floor_check_located_captured",
        "bench_floor_positions",
        "bench_floor_check_positions",
    )
    assert check_positions != positions, (
        f"{WHERE} says the floor was relocated at a different sample size; both sweeps "
        f"ran at {positions} positions, so the check is not a check"
    )
    assert check_located > declared, (
        f"{WHERE} reports the check sweep as confirming the located floor sits above the "
        f"declared one; the check now puts it at {check_located} against a declared "
        f"{declared}"
    )
    assert abs(check_located - located) < 0.05, (
        f"{WHERE} says the location is not an artifact of the sample size. The floor is "
        f"{located} at {positions} positions and {check_located} at {check_positions}, a "
        f"gap of {abs(check_located - located)}."
    )


@_NEEDS_LEDGER
def test_the_floor_sweep_scores_the_inventory_the_benchmark_scores() -> None:
    """Guards: "it uses AILM's own letter transition columns: real next-symbol
    distributions over the same {{BENCH_FLOOR_INVENTORY_SIZE}}-symbol inventory
    the benchmark scores."

    DEFECT: the sweep's authority over the live arm's gate comes entirely from
    the two being the same measurement problem. The inventory size is written by
    the floor recorder and the scored alphabet by the Stage 0 recorder, so the
    two can drift apart without either looking wrong on its own — and a sweep
    over a different inventory would locate a floor for a benchmark nobody ran.
    """
    (floor_inventory, scored_alphabet) = _values(
        "bench_floor_inventory_size", "stage0_letter_alphabet_size"
    )
    assert floor_inventory == scored_alphabet, (
        f"{WHERE} says the floor sweep runs over the same inventory the benchmark scores. "
        f"The sweep records {floor_inventory} symbols against the scored alphabet's "
        f"{scored_alphabet}."
    )


@_NEEDS_LEDGER
def test_the_probes_report_more_peaked_distributions_than_the_comparator() -> None:
    """Guards: "AILM's mean confidence is {{...}}, against {{...}} for Gemma 3 4B and
    {{...}} for Qwen 2.5 3B: the probes' reported distributions are far more
    peaked, so a truncation reaching a given captured mass leaves them far less
    structure to disagree about", and the conclusion's "the swept distributions
    are AILM-shaped and the probes' are far more peaked, so the sweep's
    divergence at a given captured mass is not a prediction for them."

    DEFECT: this is the limit the section places on its own floor experiment,
    and it is the only reason the subsection can say "Nothing here shows that the
    calibration figures reported above are wrong." If a re-run made a probe's
    reported distribution flatter than the comparator's, the sweep's divergence
    would start transferring to it and the disclaimer would understate what the
    experiment implies.
    """
    (comparator,) = _values("bench_calib_confidence_ailm")
    for probe in PROBES:
        (confidence,) = _values(f"bench_calib_confidence_{probe}")
        assert confidence > comparator, (
            f"{WHERE} rests its 'does not transfer' argument on the probes reporting far "
            f"more peaked distributions than AILM. {probe} reports a mean confidence of "
            f"{confidence} against AILM's {comparator}."
        )


@_NEEDS_LEDGER
def test_the_per_model_rule_divergence_says_what_the_section_says() -> None:
    """Guards: "Qwen 2.5 3B is inside that tolerance on both metrics and Gemma 3
    4B and SmolLM are outside it on both", "the two verdicts disagree on exactly
    one system", and "Both signed divergences were negative for every probe, so
    the shipped min-bound rule reports a *lower* error than the observed-only
    rule in every case rather than in some."

    The disagreement claim is re-derived the way decision 93 states it: the
    captured-mass proxy and the direct divergence check disagree on exactly one
    system — gemma3, which the floor admits and the divergence check withholds.
    SmolLM is withheld by both rules, so its two divergences exceeding the
    tolerance do not add a disagreement; they are the flattening trap the
    section demonstrates, a recorded-but-unpublished calibration error that is
    the lowest among the probes' recorded figures.

    This guard replaces one that asserted the opposite. The section used to say
    no per-model divergence was written to the ledger, and used that absence to
    justify deferring the gate change; the census recorded it, and both the
    sentence and this test had to be rewritten. Worth naming how the old guard
    would have failed to notice: it scanned the ``bench_floor_`` namespace for a
    probe suffix, and the new labels landed under ``bench_calib_rule_``, so it
    would have gone on passing against a namespace that no longer held the
    quantity it was about.

    DEFECT: the section makes two claims that could each flip without warning.
    A re-run that pulled Gemma or SmolLM inside the tolerance, or pushed Qwen
    outside it, makes the inside/outside sentence false; one that moved the
    floor's verdict or the rule's verdict for any probe other than gemma3 into
    conflict makes the disagreement-on-exactly-one-system sentence false. A signed
    divergence that changed sign on any probe or metric makes the one-directional
    bias claim false, and that one matters most — a reconstruction biased in one
    direction and one merely noisy around the observed value support entirely
    different conclusions about the shipped rule.
    """
    (epsilon,) = _values("bench_calib_rule_epsilon")
    outside = []
    for probe in PROBES:
        ece, brier = _values(
            f"bench_calib_rule_ece_divergence_{probe}",
            f"bench_calib_rule_brier_divergence_{probe}",
        )
        assert ece >= 0 and brier >= 0, (
            f"{probe}: a divergence is a magnitude and cannot be negative; "
            f"got ece={ece}, brier={brier}"
        )
        beyond = {"ece": ece > epsilon, "brier": brier > epsilon}
        assert len(set(beyond.values())) == 1, (
            f"{WHERE} says each probe is inside the tolerance on *both* metrics or "
            f"outside on both, but {probe} is split: {beyond} against {epsilon}"
        )
        if all(beyond.values()):
            outside.append(probe)

    assert outside == ["gemma3", "smollm"], (
        f"{WHERE} says Gemma 3 4B and SmolLM are outside the tolerance of {epsilon} "
        f"on both metrics and Qwen 2.5 3B is inside it. Probes now outside: {outside}."
    )

    # Decision-93 semantics: the gate is the direct rule check, and the
    # captured-mass floor is provenance. The prose claims the two verdicts
    # disagree on exactly one system; re-derive both verdicts from the ledger.
    (floor,) = _values("bench_calib_mass_floor")
    proxy_admits, rule_admits = {}, {}
    for probe in PROBES:
        (mean,) = _values(f"bench_calib_captured_mass_{probe}")
        (measurable,) = _values(f"bench_calib_measurable_{probe}")
        proxy_admits[probe] = mean >= floor
        rule_admits[probe] = measurable == 1 and probe not in outside
    disagreement = sorted(p for p in PROBES if proxy_admits[p] != rule_admits[p])
    assert disagreement == ["gemma3"], (
        f"{WHERE} says the two verdicts disagree on exactly one system — the floor "
        f"admitting what the divergence check withholds. They now disagree on "
        f"{disagreement} (proxy: {proxy_admits}, rule: {rule_admits})."
    )

    signed = {
        f"{metric}_{probe}": value
        for probe in PROBES
        for metric in ("ece", "brier")
        for value in _values(f"bench_calib_rule_{metric}_signed_{probe}")
    }
    positive = sorted(name for name, value in signed.items() if value >= 0)
    assert positive == [], (
        f"{WHERE} says every signed divergence is negative, so the min-bound rule "
        f"reports a lower error than the observed-only rule in every case. These "
        f"are not: {positive}. A one-directional bias and a two-directional "
        f"scatter license different conclusions about the shipped reconstruction."
    )


# ── The other arms ───────────────────────────────────────────────────────────


@_NEEDS_LEDGER
def test_the_invalid_answer_denominator_covers_all_three_tasks() -> None:
    """Guards: "{{...}} unparsable answers for Gemma 3 4B and {{...}} for Qwen 2.5 3B
    out of {{BENCH_UNPARSABLE_DENOMINATOR}} calls each
    (invalid rates {{...}}) across the three tasks."

    DEFECT: "each" and "across the three tasks" are the English holding one
    denominator to three task sizes. Change any task's sample and the per-task
    tokens update while this one is written by its own expression; the section
    would then report an invalid rate against a call count nobody made. The rate
    tokens are checked against their own numerator and denominator for the same
    reason: three quantities that are only consistent by construction.
    """
    (denominator, letters, phonemes, words) = _values(
        "bench_unparsable_denominator",
        "bench_letter_positions",
        "bench_phoneme_positions",
        "bench_g2p_words",
    )
    assert denominator == letters + phonemes + words, (
        f"{WHERE} reports the invalid-answer diagnostic out of {denominator} calls each "
        f"'across the three tasks', but the three tasks scored "
        f"{letters} + {phonemes} + {words} = {letters + phonemes + words} positions"
    )
    for probe in PROBES:
        count, rate = _values(f"bench_unparsable_{probe}", f"bench_unparsable_rate_{probe}")
        assert 0 <= count <= denominator, f"{probe}: {count} invalid answers out of {denominator}"
        assert rate == pytest.approx(count / denominator, abs=1e-12), (
            f"{probe}: the recorded invalid rate {rate} is not {count} / {denominator}, so "
            f"the rate in {WHERE} is not the rate its own numerator implies"
        )


@_NEEDS_LEDGER
def test_the_comparator_scores_identically_at_both_prefix_windows() -> None:
    """Guards: "The AILM row prints its accuracy twice so a reader can check the
    invariance rather than take it on assurance. That equality is a property of
    an order-one model, not a finding about context."

    DEFECT: the context-window table invites the reader to verify an invariance
    by comparing two printed cells. Both cells are tokens from separate labels,
    so a recorder that scored the comparator differently at the long window would
    print two different numbers under a sentence telling the reader they must
    match — and the arm's whole premise, that extra context is available to a
    transformer and unavailable to AILM by construction, would be in question.

    ``tests/test_benchmarks.py`` proves the invariance of the comparator itself;
    what is checked here is that the two recorded cells the reader compares are
    the ones that invariance predicts, and that both equal the headline letter
    accuracy the table above reports.
    """
    short, long, headline = _values(
        "bench_letter_short_top1_ailm",
        "bench_letter_long_top1_ailm",
        "bench_letter_top1_ailm",
    )
    (window_short, window_long) = _values("bench_letter_window_short", "bench_letter_window_long")
    assert window_long > window_short > 0, (
        f"{WHERE} widens the prefix from {window_short} to {window_long} characters; that "
        f"is no longer a widening, so the arm compares nothing"
    )
    assert short == long, (
        f"{WHERE} prints the comparator's accuracy twice so a reader can check that an "
        f"order-one model is invariant to prefix width. The two cells are now {short} at "
        f"the {window_short}-character window and {long} at {window_long}."
    )
    assert short == headline, (
        f"the comparator scores {short} in the context-window table and {headline} in the "
        f"headline table of {WHERE}; the same system on the same positions is reported as "
        f"two different numbers"
    )


@_NEEDS_LEDGER
def test_the_headline_g2p_comparison_is_the_one_less_favorable_to_ailm() -> None:
    """Guards: "under the probes' own alignment rule the comparator scores
    {{BENCH_G2P_POS_ACC_LEV_AILM}}, so the headline comparison is the one less
    favorable to AILM."

    DEFECT: the section reports the comparator under a stricter alignment rule
    than the probes and says so, which is a disclosure only while the stricter
    rule really is the lower number. If the two scores ever crossed, the
    supplement would be claiming a conservative choice while reporting the
    flattering one.
    """
    strict, lenient = _values("bench_g2p_pos_acc_ailm", "bench_g2p_pos_acc_lev_ailm")
    assert strict < lenient, (
        f"{WHERE} says the headline G2P comparison is the one less favorable to AILM. "
        f"The headline strict-alignment score is now {strict} against {lenient} under the "
        f"probes' own Levenshtein rule, so the headline is the more favorable number."
    )


@_NEEDS_LEDGER
def test_the_long_g2p_bucket_is_the_one_that_leaves_two_probes_unseparated() -> None:
    """Guards: "the short and the mid-length columns separate all three [probes]
    pairwise. The long bucket holds far fewer words than either of the other two,
    and its column is where that shows: the [gemma3] and [qwen25] intervals
    overlap there, so the long column orders those two without separating them,
    while [smollm] stays clear of both."

    DEFECT: this reading is entirely about which intervals touch, and every part
    of it is a live arithmetic fact rather than a stable property of the design.
    A rerun that tightened the long bucket's intervals would make "orders those
    two without separating them" false; one that loosened the short or mid
    columns would make "separate all three pairwise" false; and either could
    move which pair is the overlapping one. The stratification must also still
    partition the scored words, so a bucket rule that dropped or double-counted
    a word fails here rather than in the reading of the table.

    The supplement's earlier caveat — that a comparator cell and a probe cell in
    one column are on two alignment rules — is why only probe-probe pairs are
    checked for separation. AILM's own cells are required to carry intervals
    (the table prints them) but are deliberately not compared across rows.

    "Far fewer" is read as less than half of either other bucket. Two intervals
    "separate" when they are disjoint.
    """
    (short, mid, long, words) = _values(
        "bench_g2p_bucket_short_words",
        "bench_g2p_bucket_mid_words",
        "bench_g2p_bucket_long_words",
        "bench_g2p_words",
    )
    assert short > 0 and mid > 0 and long > 0, (
        f"a G2P length bucket is empty, so the stratified table in {WHERE} has a column "
        f"describing no words"
    )
    assert short + mid + long == words, (
        f"the three buckets hold {short + mid + long} words against {words} scored; the "
        f"stratification in {WHERE} is not a partition of the task"
    )
    assert long * 2 < short and long * 2 < mid, (
        f"{WHERE} says the long bucket holds far fewer words than either of the other two. "
        f"It now holds {long} against {short} short and {mid} mid-length."
    )

    _values(  # POSITIVE CONTROL: the per-bucket cells themselves are recorded.
        *(
            f"bench_g2p_pos_{bucket}_{system}"
            for bucket in ("short", "mid", "long")
            for system in ("ailm", *PROBES)
        )
    )

    def _interval(bucket: str, system: str) -> tuple[float, float]:
        low, high = _values(
            f"bench_g2p_pos_{bucket}_wilson_low_{system}",
            f"bench_g2p_pos_{bucket}_wilson_high_{system}",
        )
        assert low <= high, (
            f"the {bucket}-bucket Wilson interval for {system} is inverted ({low} > {high}); "
            f"the stratified table in {WHERE} would print a bound pair backwards"
        )
        return (low, high)

    # NON-VACUITY: every cell the table prints an interval for must have one,
    # the comparator row included. A missing bound here is a table cell the
    # renderer cannot fill, not a reading that quietly weakens.
    for bucket in ("short", "mid", "long"):
        for system in ("ailm", *PROBES):
            _interval(bucket, system)

    def _unseparated(bucket: str) -> list[tuple[str, str]]:
        """Probe pairs whose intervals overlap in this bucket's column."""
        pairs = []
        for index, first in enumerate(PROBES):
            for second in PROBES[index + 1 :]:
                low_a, high_a = _interval(bucket, first)
                low_b, high_b = _interval(bucket, second)
                if low_a <= high_b and low_b <= high_a:
                    pairs.append((first, second))
        return pairs

    for bucket in ("short", "mid"):
        assert _unseparated(bucket) == [], (
            f"{WHERE} says the {bucket} column separates all three probes pairwise. "
            f"These pairs now overlap: {_unseparated(bucket)}."
        )
    assert _unseparated("long") == [("gemma3", "qwen25")], (
        f"{WHERE} says the long column leaves exactly the {PROBES[0]}/{PROBES[1]} pair "
        f"unseparated and holds {PROBES[2]} clear of both. The overlapping pairs are now "
        f"{_unseparated('long')}."
    )


@_NEEDS_LEDGER
def test_synthetic_arm_covers_every_probe_and_generated_scope_tracks_results() -> None:
    """The producer inventory, paired family, table, and interpretation move together.

    The prior one-probe recorder made the two-probe in-context consumer
    impossible to run. The manuscript now receives a generated result table and
    scope sentence, so a changed p-value or direction updates the reading rather
    than leaving a hard-coded separation claim behind.
    """
    ailm, uniform, alpha, family_size = _values(
        "bench_synth_letter_top1_ailm",
        "bench_synth_uniform_baseline",
        "bench_synth_family_alpha",
        "bench_synth_family_size",
    )
    assert family_size == len(PROBES)
    assert ailm > uniform, (
        f"the synthetic comparator scores {ailm} against a uniform floor of {uniform}; "
        "the stipulated-language arm no longer demonstrates that AILM estimated its chain"
    )

    all_ailm_separates = True
    for probe in PROBES:
        accuracy, p_value, b, c = _values(
            f"bench_synth_letter_top1_{probe}",
            f"bench_synth_paired_p_{probe}",
            f"bench_synth_paired_b_{probe}",
            f"bench_synth_paired_c_{probe}",
        )
        _values(
            f"bench_synth_letter_top1_wilson_low_{probe}",
            f"bench_synth_letter_top1_wilson_high_{probe}",
            f"bench_synth_unparsable_{probe}",
        )
        assert b + c > 0, f"the synthetic contrast for {probe} has no discordant pairs"
        all_ailm_separates &= p_value < alpha and c > b and ailm > accuracy

    variables = generate_manuscript_variables(PROJECT_ROOT)
    scope = variables["BENCH_SYNTH_SCOPE_RESULT"]
    # The scope sentence names the arm as the recorder does ("no-demonstration");
    # earlier drafts said "zero-shot" before the two-probe protocol rename.
    assert ("separates from every registered no-demonstration probe" in scope) is all_ailm_separates
    table = variables["BENCH_SYNTH_RESULTS_TABLE_ROWS"]
    assert table.count("\n") == len(PROBES)
    _assert_hosted_cohort_table(table)
    for namespace in ("synth", "ctx"):
        for field in ("model", "provider"):
            label = f"bench_{namespace}_openrouter_{field}"
            assert variables[label.upper()] == _claims()[label]["value"]
    assert variables["BENCH_SYNTH_OPENROUTER_MODEL"] == variables["BENCH_CTX_OPENROUTER_MODEL"]


def _assert_hosted_cohort_table(table: str) -> None:
    row_labels = [line.split("|")[1].strip() for line in table.splitlines()]
    assert row_labels == ["AILM", *(f"{probe} run" for probe in PROBES)], (
        "hosted result rows must identify cohorts, not local model weights"
    )


@_NEEDS_LEDGER
def test_hosted_table_refuses_local_weight_names(tmp_path) -> None:
    table = generate_manuscript_variables(PROJECT_ROOT)["BENCH_SYNTH_RESULTS_TABLE_ROWS"]
    path = tmp_path / "hosted-table.md"
    path.write_text(table, encoding="utf-8")
    _assert_hosted_cohort_table(path.read_text(encoding="utf-8"))
    path.write_text(table.replace("gemma3 run", "Gemma 3 4B"), encoding="utf-8")
    with pytest.raises(AssertionError, match="cohorts, not local model weights"):
        _assert_hosted_cohort_table(path.read_text(encoding="utf-8"))


@_NEEDS_LEDGER
def test_the_greedy_protocol_reproduces_itself_exactly() -> None:
    """Guards: "the sampling seed cannot reach the output and the expected run-to-run
    spread is exactly zero. That is a prediction about the protocol, and it was
    measured rather than assumed", and "These are determinism replications and
    not estimates of sampling variance; there is no sampling variance to
    estimate under greedy decoding."

    DEFECT: the section refuses to treat the two-seed table as a variance
    estimate, on the grounds that the measured spread is exactly zero. A re-run
    with a non-zero spread would make the two-seed table an estimate of
    something, and the paragraph forbidding that reading would be the only thing
    standing between a reader and a variance claim the data now supports.

    The recorded spread is re-derived from the two seed accuracies rather than
    trusted, so a spread column that stopped tracking its own columns fails here.
    """
    (ensemble_positions, determinism_positions) = _values(
        "bench_ensemble_positions", "bench_determinism_positions"
    )
    assert ensemble_positions > 0 and determinism_positions > 0, (
        f"{WHERE} says the prediction was measured rather than assumed; one of the two "
        f"replication arms scored no positions"
    )
    for probe in PROBES:
        first, second, spread = _values(
            f"bench_run_letter_top1_{probe}_seed1",
            f"bench_run_letter_top1_{probe}_seed2",
            f"bench_run_letter_spread_{probe}",
        )
        assert spread == pytest.approx(abs(first - second), abs=1e-12), (
            f"{probe}: the recorded spread {spread} is not the gap between the two seed "
            f"accuracies ({first}, {second})"
        )
        assert spread == 0.0, (
            f"{WHERE} says the expected run-to-run spread under greedy decoding is exactly "
            f"zero and that the two-seed table is therefore not a variance estimate. "
            f"{probe} now spreads {spread} across neighboring seeds."
        )
        (agreement,) = _values(f"bench_determinism_agreement_{probe}")
        assert agreement == 1.0, (
            f"{WHERE} says an identical request at an identical seed reproduces the reply "
            f"string. {probe} now agrees on a fraction {agreement} of positions, so "
            f"'reproducibility holds under matched server state' is no longer what was "
            f"measured."
        )


# ── The positive control for this whole file ─────────────────────────────────


#: Distinctive fragments of the sentences the tests above guard, in the order
#: they appear in the supplement. Whitespace is collapsed before matching, so a
#: fragment may span the file's hard wrap.
GUARDED_SENTENCES = (
    "calls each (invalid rates",
    "the headline comparison is the one less favorable to AILM",
    "The membership rule is six contrast types on each of the three probes",
    "The largest of the nine derived values is",
    "every power figure in this section, recorded or derived, is an upper bound",
    "The letter contrast is therefore no longer unresolved by construction",
    "have a non-alphabetic target, almost all of them a space",
    "almost none of them and the remaining one on most",
    "The full-inventory letter arm therefore measures inventory compliance more than "
    "next-symbol prediction",
    "every alphabetic-arm contrast clears it, and the discordant pairs run in the "
    "comparator's favor in every one",
    "scoring highest of the probes on non-alphabetic targets and lowest once they are excluded",
    "The Gemma 3 4B contrast clears the",
    "contrasts clear the corrected family threshold of",
    "The AILM row prints its accuracy twice so a reader can check the invariance",
    "Every cell carries",
    "columns separate all three pairwise",
    "holds far fewer words than either of the other two",
    "orders those two without separating them",
    "stays clear of both",
    "does not and is unresolved",
    "is the spread case, Qwen 2.5 3B the concentrated one, and",
    "applied to the mean, which is the stricter summary whenever the two differ",
    "That is near-total, and it is not guaranteed",
    "Calibration is a distinct axis from accuracy rather than a restatement of it",
    "is a large fraction of the effect this axis reports",
    "The located floor is consequently unreachable in principle here rather than merely "
    "unmet by the registered probes",
    "both just below the declared",
    "-symbol inventory the benchmark scores",
    "so the location is not an artifact of the sample size",
    "The distributional gap closes monotonically; its image in either metric does not",
    "the probes' reported distributions are far more peaked",
    "sits above the declared floor and below the located one",
    "That quantity is now recorded",
    "the two verdicts disagree on exactly one system",
    "the expected run-to-run spread is exactly zero",
    "The phoneme and G2P contrasts do not share the compliance weakness",
    "The alphabetic arms are well powered at the reporting level",
    "on effects larger than the reference share",
    "with one declared exception on G2P (position-aligned letter-to-phone) alignment",
)


def test_every_guarded_sentence_is_still_in_the_supplement() -> None:
    """POSITIVE CONTROL for this file: the guards must still have something to guard.

    DEFECT: every test above is an assertion about a sentence that exists. Delete
    or reword the sentence and its guard keeps passing, having become an
    assertion about the ledger with no manuscript claim attached — and the next
    reader has no way to tell which of these tests are still load-bearing. The
    file's premise is that prose does not re-render; this is what makes the
    reverse, prose moving out from under a guard, fail loudly too.

    A failure here is not a license to delete the fragment. Re-read the rewritten
    sentence, decide whether the guard still derives it, then update both.
    """
    assert SUPPLEMENT.exists(), f"{WHERE} is missing; every guard in this file is orphaned"
    text = _supplement_text()
    assert len(text) > 10_000, f"{WHERE} is far shorter than the section it should hold"
    missing = [fragment for fragment in GUARDED_SENTENCES if fragment not in text]
    assert missing == [], (
        f"these sentences are guarded by tests in this file but are no longer in {WHERE}: "
        f"{missing}. A reworded sentence needs its guard re-derived, not its fragment "
        f"deleted — the guard is what keeps the reading true after the next recorder run."
    )


#: A model name as this section writes them: a family word, then a version, then
#: a parameter count. Used to find any probe the supplement names that the
#: registry does not serve, without naming a retired one — a guard that has to
#: spell out the model it excludes reintroduces the text it exists to remove, and
#: only ever catches that one model.
_MODEL_MENTION = re.compile(r"\b[A-Z][A-Za-z]+\s?\d[\d.]*\s?[MB]\b")

#: Count wordings that stop being true when the probe inventory changes size.
#: Derived from the registry rather than listed, so this needs no edit the next
#: time a probe is added or retired.
_COUNT_WORDS = {2: "two", 3: "three", 4: "four", 5: "five"}


def test_the_supplement_no_longer_describes_a_retired_probe() -> None:
    """The section must not still be written for three probes.

    DEFECT: the third probe's removal changes counts spelled in words all over
    this section — "two of the three", "the nine derived values", "one of the
    three probes fails it" — and shrinks the correction family, which loosens
    every threshold those sentences are read against. Tokens do not update
    English. ``test_every_guarded_sentence_is_still_in_the_supplement`` catches a
    guarded sentence that was reworded; it cannot catch one that was left alone,
    because a fragment that is still present still matches.

    This is the other direction: any model the section names that the registry
    does not serve is a row, a caption, or a paragraph describing a system the
    benchmark no longer measures and the ledger no longer carries. The rule is
    stated over the registry rather than over a named model, so it catches a
    probe swapped in as readily as one left behind.

    POSITIVE CONTROL: every registered probe must be named in the section, so a
    supplement that lost its model names entirely fails here rather than passing
    a scan that found nothing.
    """
    assert SUPPLEMENT.exists(), f"{WHERE} is missing; this guard has nothing to scan"
    text = _supplement_text()
    served = set(PROBE_DISPLAY_NAMES.values())
    for name in sorted(served):
        assert name in text, (
            f"{WHERE} does not name {name}, so the scan below would pass on a section "
            f"that had stopped describing the benchmark at all"
        )
    # Exact display names only: a mention is foreign when it is not a served
    # display name and not a prefix of one. 'SmolLM 1.7B' inside the registered
    # 'SmolLM 1.7B Base v0.2' is a substring collision, not a retired probe.
    foreign = sorted(
        {
            m
            for m in _MODEL_MENTION.findall(text)
            if m not in served and not any(s.startswith(m) for s in served)
        }
    )
    assert foreign == [], (
        f"{WHERE} names {foreign}, which the registry does not serve. The benchmark "
        f"measures {len(PROBES)} probes ({sorted(PROBES)}); a model named here that is "
        f"not measured is a row or caption describing a system the ledger no longer "
        f"carries."
    )

    wrong_counts = [
        phrase
        for count, word in _COUNT_WORDS.items()
        if count != len(PROBES)
        for phrase in (f"the {word} probes", f"of the {word} probes", f"{word} probe contrasts")
        if phrase in text
    ]
    assert wrong_counts == [], (
        f"{WHERE} still contains {wrong_counts}, which count a probe inventory of a "
        f"different size. The correction family scales with the probe count, so the "
        f"thresholds those sentences are read against have changed with it."
    )


# ── The fairness table's status column ───────────────────────────────────────
#
# The table added to the supplement states each fairness condition's status in
# English about the ledger, which is exactly the sentence class this file exists
# to guard, and it arrived unguarded. Two of its rows are derivable; the rest are
# statements about the protocol's design rather than about recorded values, and
# are left to the reader of the protocol section rather than guarded here.


def test_stipulated_arm_claims_only_reduced_exact_artifact_overlap() -> None:
    """The generated corpus narrows one overlap concern; it proves no non-exposure."""

    claims = _claims()
    synthetic = {label for label in claims if label.startswith("bench_synth_")}
    assert synthetic, (
        "no bench_synth_* labels are recorded, so the supplement's claim that a "
        "stipulated-language arm exists at all is unsupported"
    )
    assert "bench_synth_protocol_version" in claims, (
        "the synthetic arm no longer declares its own protocol version, which is "
        "how the table binds its bounded exact-artifact-overlap statement"
    )

    text = SUPPLEMENT.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    required = (
        "the stipulated arm reduces exact-artifact overlap but cannot establish their training contents",
        "does not establish the contents of any external model's pretraining data",
        "does not prove absence from or independence of pretraining",
    )
    missing = [fragment for fragment in required if fragment not in normalized]
    assert missing == [], f"{WHERE} lost the bounded exposure language: {missing}"

    decisions = DECISIONS.read_text(encoding="utf-8")
    decision_90 = decisions.split("## 90 —", 1)[1]
    decision_90 = decision_90.split("\n## ", 1)[0]
    normalized_decision = " ".join(decision_90.split()).lower()
    for fragment in (
        "reduces exact-artifact overlap as an explanation",
        "cannot establish absence of coincident or structurally similar material",
        "makes no general contamination or non-exposure claim",
    ):
        assert fragment in normalized_decision, (
            f"docs/decisions.md decision 90 lost its bounded claim: {fragment!r}"
        )

    forbidden = (
        "contamination-free",
        "holds only in the synthetic arm",
        "cannot have been trained on",
        "no prior exposure to the synthetic",
    )
    present = [fragment for fragment in forbidden if fragment in text.lower()]
    assert present == [], f"{WHERE} overstates the stipulated arm: {present}"
    decision_present = [fragment for fragment in forbidden if fragment in normalized_decision]
    assert decision_present == [], f"Decision 90 overstates the stipulated arm: {decision_present}"

    # POSITIVE CONTROL: the forbidden-language scan must detect a planted
    # non-exposure claim rather than pass because its vocabulary is ineffective.
    planted = text + "\nThe stipulated arm is contamination-free.\n"
    assert any(fragment in planted.lower() for fragment in forbidden), (
        "the planted contamination-free claim escaped the scope guard"
    )


def test_the_g2p_alignment_exception_is_real_and_disfavors_the_comparator() -> None:
    """The one declared exception to "same scoring rule", and its direction.

    DEFECT: the table records G2P alignment as the single exception to scoring
    both systems alike — the comparator by strict positional alignment, the
    probes by Levenshtein. The supplement states that the headline comparison is
    therefore the reading *less* favorable to the comparator. If the two scores
    ever converged, the exception would be immaterial and the caveat misleading;
    if the direction reversed, the caveat would be backwards, which is worse.
    """
    claims = _claims()
    strict = claims.get("bench_g2p_pos_acc_ailm", {}).get("value")
    lenient = claims.get("bench_g2p_pos_acc_lev_ailm", {}).get("value")
    assert strict is not None and lenient is not None, (
        "the strict and Levenshtein comparator scores are what make the declared "
        "G2P alignment exception checkable; one of them is no longer recorded"
    )
    assert lenient != strict, (
        "the two alignment rules now score the comparator identically, so the "
        "table's 'one declared exception' is no longer an exception at all"
    )
    assert lenient > strict, (
        f"the Levenshtein reading ({lenient}) no longer exceeds the strict one "
        f"({strict}), so the supplement's claim that the headline comparison is "
        f"the reading less favorable to the comparator has reversed direction"
    )
