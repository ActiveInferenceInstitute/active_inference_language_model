#!/usr/bin/env python3
"""Record the document-clustered interval of the Level 6 improvement on the held-back split.

What this measures, and why it is not already recorded
------------------------------------------------------
Level 6 scores sentences that nest inside documents. The project's primary
inferential procedure for that shape is the whole-document bootstrap in
``ailm.clustering``, not the sentence-level sign test: sentences inside one
document share a topic, an author and one narrative arc, so treating each as an
independent matched pair is the assumption the sign test needs and the nesting
denies. On the *development* population that primary interval spans zero, which
is what makes Stage 6's finding status a positive point estimate with a
non-confirmatory primary analysis (decision 26).

Decision 111 replicated the *sign test* on the held-back ``ud_ewt_test`` split
and it crossed the frozen threshold in the registered direction. That is a real
result about the sign, and it is not the paper's primary estimand. The primary
estimand on the held-back population -- the document-clustered interval of the
mean per-sentence improvement in nats -- had never been computed. Decision 115
registers it, states its prediction before the computation, and this recorder is
what computes it.

The distinction matters because the development population is the case that
proves the two answers can differ: there, the clustered *sign* arm excluded zero
while the clustered *log-loss* arm spanned it. A held-back sign test therefore
licenses nothing about the held-back interval, and the interval has to be run.

Why this recorder may re-run, when the replication recorder may not
-------------------------------------------------------------------
``scripts/record_confirmatory_replication.py`` refuses a second run and is
deliberately named outside the ``record_*_claims.py`` glob so the orchestrator
cannot call it twice. The rule it enforces is that a held-back population may be
*scored* once: a second scoring pass, with parameters chosen after seeing the
first, un-holds the population.

That rule has already done its work here. The population is spent -- the
sentinel ``confirmatory_run_completed_ud_ewt_test`` records it -- and decision
115's estimand is registered *on an already-spent population*, which is exactly
why decision 115 assigns its own outcome the ``descriptive`` status rather than
``confirmatory``, despite the registration commit preceding the outcome commit.
Nothing here is fitted, tuned or selected on the held-back data: the models are
the frozen development fits, asserted equal array for array, and the estimand,
the resamples, the seed and the coverage were all fixed in writing beforehand.
Re-running a deterministic function of frozen inputs adds no information and
leaks none, so this recorder is idempotent and belongs in the ``core`` phase.

The controls, as amended
------------------------
An interval that spans zero is only informative if the instrument could have
found an effect on this population, and only trustworthy if it does not find one
where none exists. Every control runs on the held-back differences themselves,
in the same scoring frame, so none is a different experiment wearing the name.

The controls below are decision 115's *amended* specification. The entry records
what the first specification got wrong and why the amendment was written before
any ledger write: it planted one arbitrary effect size, smaller than this arm's
own resolution, and made non-detection refuse a *detection* as well as a null. A
positive control establishes that a null is interpretable. It says nothing about
a detection, and a detection is its own evidence that the instrument detects.

*Minimum detectable effect.* Shifting every difference by a constant shifts every
bootstrap replicate by the same constant, so the lower bound is exactly
equivariant under a plant and the smallest detectable constant shift has a closed
form: the negated lower bound of the centered differences. That value is
computed, recorded, and then **verified against the machinery rather than
trusted**, which is what the next control is for.

*Positive control, two-sided and gating.* The bootstrap is re-run at the minimum
detectable effect plus one tolerance, which must be detected, and at minus one
tolerance, which must not be. Because the closed form above is derived from
equivariance rather than measured, these two probes are exactly the test of
equivariance: any shift-dependent defect in the resampling breaks one of them.
The pair also cannot pass on a dead instrument, which fails the upper probe, nor
on a hair-trigger one, which fails the lower.

*Negative control, gating.* Whole documents are sign-flipped under a fixed
recorded seed, which destroys the aggregate direction while preserving every
document's size and internal structure. The resulting interval must span zero.

*Development-sized-effect probe, recorded and NOT gating.* Whether an effect the
size of the development population's own clustered effect is resolved here. This
was the original positive control and it is kept as a measurement, because it is
the fact that decides whether a null on this arm would be interpretable. It does
not refuse: if the primary interval spans zero while this probe is unresolved,
the null is under-powered rather than evidence of absence, and the manuscript
must say so in those words.

Every control is deterministic given its seed, so a failure is a reproducible
fact about the instrument and not a flaky draw. The recorder refuses on the
negative control firing or on either side of the two-sided probe failing, and
says which.

Coverage
--------
The primary interval is computed at ``ailm.statistics.DIFFERENCE_CONFIDENCE``,
the coverage the *development* Stage 6 clustered arm in
``scripts/record_cluster_claims.py`` uses. Decision 115 records why: the finding
status compares "the development interval spans zero" against "the held-back
interval excludes zero", and two intervals at different coverages are two
different quantities, so a comparison across them would not be the comparison
the status rule claims to make. The stricter family-corrected coverage is
recorded alongside as a declared sensitivity, so a reader who wants the
conservative reading has it without either bound being presented as the primary.

Usage:
    .venv/bin/python scripts/record_confirmatory_stage6_clustered_claims.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ailm.clustering import ClusteredPairedSummary

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
from record_confirmatory_replication import (  # noqa: E402
    ProtocolRefusal,
    merged_narrative_artifact,
)

from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402

RUN_ID = "confirmatory_stage6_clustered_v1"
SOURCE = "scripts/record_confirmatory_stage6_clustered_claims.py"
DECISION = 115
SENTINEL = "confirmatory_run_completed_ud_ewt_test"
PREFIX = "confirmatory_stage6_clustered_"

#: Seed for the whole-document sign flips of the negative control. Distinct from
#: ``ailm.clustering.DEFAULT_SEED`` so the control's draw cannot coincide with a
#: resampling stream it is supposed to be independent of.
CONTROL_FLIP_SEED = 20260906

#: Decimal places every recorded nats-scale bound keeps. Six is what the
#: development cluster recorder keeps for the same quantity, so the two arms'
#: bounds are comparable at the precision they are printed at.
NATS_PLACES = 6

#: Half-width of the bracket the two-sided positive control probes the minimum
#: detectable effect with. Two orders of magnitude above the recorded rounding
#: granularity, so a probe result cannot be an artifact of the rounding, and two
#: orders below the effect scale, so the bracket is tight enough that passing it
#: means the recorded value really is the boundary.
MDE_TOLERANCE_NATS = 1e-4


def _refuse(message: str) -> ProtocolRefusal:
    """A refusal that writes nothing, in the replication recorder's register."""
    return ProtocolRefusal(f"decision {DECISION}: {message} No claims written.")


def held_back_differences() -> tuple[list[float], list[str], int]:
    """Per-sentence held-back log-loss deltas in nats, keyed by document id.

    The models are the development fits. ``merged_narrative_artifact`` builds a
    frame whose train arm is the development train split and whose test arm is
    the held-back population, and the equality assertion below is what proves
    the merge did not re-index or re-fit anything: the narrative emission and
    unigram built from the merged frame must equal the development ones array
    for array, exactly as ``score_stage6`` in the replication recorder demands.

    Returns:
        The per-sentence deltas, the document id each delta belongs to, and the
        number of held-back documents.
    """
    from ailm.levels.narrative import (
        build_phase_models,
        heldout_type_logloss,
        load_narrative_artifact,
    )

    merged = merged_narrative_artifact()
    development = load_narrative_artifact()
    models = build_phase_models(development.train)
    for key, array in build_phase_models(merged.train).items():
        if not np.array_equal(array, models[key]):
            raise _refuse(
                f"the narrative {key} built from the merged frame differs from the "
                f"development one, so a parameter was re-fitted on the held-back split."
            )

    recognition = heldout_type_logloss(merged.test, models)
    deltas = [float(delta) for delta in recognition["deltas"]]
    documents = [
        document_id
        for document_id, types in zip(merged.test_document_ids, merged.test, strict=True)
        for _ in types
    ]
    if len(documents) != len(deltas):
        raise _refuse(
            f"{len(deltas)} scored held-back sentences but the split accounts for "
            f"{len(documents)}; the cluster key is not the split."
        )
    return deltas, documents, len(merged.test)


def _summary_values(prefix: str, summary: ClusteredPairedSummary) -> dict[str, Any]:
    """The bounds and shape of one clustered summary, under one label prefix."""
    return {
        f"{prefix}mean_improvement_nats": round(summary.effect, NATS_PLACES),
        f"{prefix}ci_low": round(summary.ci_low, NATS_PLACES),
        f"{prefix}ci_high": round(summary.ci_high, NATS_PLACES),
        f"{prefix}excludes_zero": bool(summary.interval_excludes_zero),
        f"{prefix}confidence": summary.confidence,
        f"{prefix}confidence_pct": round(100.0 * summary.confidence, 4),
    }


def _flip_whole_documents(
    deltas: list[float], documents: list[str], seed: int
) -> tuple[list[float], int]:
    """Reverse the sign of a deterministic random subset of whole documents.

    Flipping whole documents rather than sentences is what makes this a control
    on the clustered procedure: every document keeps its size and its internal
    correlation structure, and only the direction it contributes changes.
    """
    order = sorted(set(documents))
    generator = np.random.default_rng(seed)
    flipped = {
        document_id: bool(draw)
        for document_id, draw in zip(order, generator.integers(0, 2, size=len(order)), strict=True)
    }
    values = [
        -delta if flipped[document_id] else delta
        for delta, document_id in zip(deltas, documents, strict=True)
    ]
    return values, sum(flipped.values())


def main() -> int:
    from ailm.claims import ClaimBatch, load_ledger
    from ailm.clustering import DEFAULT_RESAMPLES, DEFAULT_SEED, summarize_clustered_difference
    from ailm.statistics import CORRECTED_CONFIDENCE, DIFFERENCE_CONFIDENCE

    ledger = load_ledger(PROJECT_ROOT)

    def recorded(label: str) -> float:
        entry = ledger.get(label)
        if entry is None:
            raise _refuse(
                f"this recorder reanalyzes a measurement it does not own, so the "
                f"owning recorder must run first. Missing: {label!r}."
            )
        return float(entry["value"])

    # The population must already be spent. This recorder does not score a
    # held-back population; it re-analyzes one that decision 111 already scored,
    # and that ordering is the whole basis for it being allowed to re-run.
    if ledger.get(SENTINEL) is None:
        raise _refuse(
            f"the held-back population has not been scored: {SENTINEL!r} is absent. "
            f"This recorder re-analyzes a spent population and must not be the "
            f"first pass over a held-back one; run the decision 111 replication first."
        )

    deltas, documents, n_documents = held_back_differences()

    # Reproduction guards against the labels decision 111 already recorded. A
    # silent drift in the loaders would otherwise let this arm report an
    # interval around a different mean from the one the manuscript states.
    if len(deltas) != int(recorded("confirmatory_stage6_sentences")):
        raise _refuse(
            f"the held-back recomputation scored {len(deltas)} sentences but the "
            f"ledger records {int(recorded('confirmatory_stage6_sentences'))}."
        )
    if n_documents != int(recorded("confirmatory_stage6_documents")):
        raise _refuse(
            f"the held-back recomputation covers {n_documents} documents but the "
            f"ledger records {int(recorded('confirmatory_stage6_documents'))}."
        )
    mean_delta = float(np.mean(deltas))
    if round(mean_delta, 4) != round(recorded("confirmatory_stage6_improvement_nats"), 4):
        raise _refuse(
            f"the held-back recomputation gives a mean improvement of "
            f"{mean_delta:.4f} nats but the ledger records "
            f"{recorded('confirmatory_stage6_improvement_nats')}."
        )
    positive_sentences = sum(1 for delta in deltas if delta > 0)
    negative_sentences = sum(1 for delta in deltas if delta < 0)
    if (positive_sentences, negative_sentences) != (
        int(recorded("confirmatory_stage6_sign_test_favored")),
        int(recorded("confirmatory_stage6_sign_test_opposed")),
    ):
        raise _refuse(
            f"the held-back recomputation gives {positive_sentences} favored and "
            f"{negative_sentences} opposed sentences but the ledger records "
            f"{int(recorded('confirmatory_stage6_sign_test_favored'))} and "
            f"{int(recorded('confirmatory_stage6_sign_test_opposed'))}."
        )

    # The coverage is passed explicitly rather than left to the default: decision
    # 115's status rule compares this bound against the development one, and a
    # default that moved would silently make the two sides different quantities.
    primary = summarize_clustered_difference(deltas, documents, confidence=DIFFERENCE_CONFIDENCE)
    corrected = summarize_clustered_difference(deltas, documents, confidence=CORRECTED_CONFIDENCE)
    if primary.n_clusters >= primary.n_observations:
        raise _refuse(
            "the held-back arm has as many clusters as observations, so its "
            "cluster analysis is the naive analysis under another name."
        )
    if round(primary.effect, 4) != round(mean_delta, 4):
        raise _refuse(
            f"the clustered effect {primary.effect:.4f} is not the mean improvement "
            f"{mean_delta:.4f}; the two are the same estimand and must agree."
        )

    # ── The centered baseline every plant below is measured against.
    centered = [delta - mean_delta for delta in deltas]
    centered_summary = summarize_clustered_difference(
        centered, documents, confidence=DIFFERENCE_CONFIDENCE
    )
    if centered_summary.interval_excludes_zero:
        raise _refuse(
            f"the centered differences, whose mean is exactly zero, still produce the "
            f"interval [{centered_summary.ci_low:+.6f}, {centered_summary.ci_high:+.6f}], "
            f"which excludes zero. Every plant below is measured against this baseline, "
            f"so a baseline that already detects makes the whole control family "
            f"meaningless."
        )

    def _plant(shift: float) -> ClusteredPairedSummary:
        return summarize_clustered_difference(
            [value + shift for value in centered], documents, confidence=DIFFERENCE_CONFIDENCE
        )

    # ── Minimum detectable effect, in closed form, then verified two-sided.
    # A constant plant shifts every bootstrap replicate by the same constant, so
    # the lower bound is exactly equivariant and the smallest detectable shift is
    # the negated lower bound of the centered differences. The two probes below
    # are what turn that derivation into a measurement: any shift-dependent
    # defect in the resampling breaks one of them.
    minimum_detectable = -centered_summary.ci_low
    probe_above = _plant(minimum_detectable + MDE_TOLERANCE_NATS)
    probe_below = _plant(minimum_detectable - MDE_TOLERANCE_NATS)
    if not probe_above.interval_excludes_zero:
        raise _refuse(
            f"POSITIVE CONTROL FAILED (upper probe). A plant of "
            f"{minimum_detectable + MDE_TOLERANCE_NATS:.6f} nats, one tolerance above "
            f"the closed-form minimum detectable effect, produced the interval "
            f"[{probe_above.ci_low:+.6f}, {probe_above.ci_high:+.6f}], which spans zero. "
            f"The bootstrap is not equivariant under a constant plant, so the recorded "
            f"minimum detectable effect is not a boundary and no interval this recorder "
            f"writes can be interpreted."
        )
    if probe_below.interval_excludes_zero:
        raise _refuse(
            f"POSITIVE CONTROL FAILED (lower probe). A plant of "
            f"{minimum_detectable - MDE_TOLERANCE_NATS:.6f} nats, one tolerance below "
            f"the closed-form minimum detectable effect, produced the interval "
            f"[{probe_below.ci_low:+.6f}, {probe_below.ci_high:+.6f}], which excludes "
            f"zero. The procedure detects an effect smaller than its own resolution, so "
            f"it is a hair trigger and its detections cannot be trusted."
        )

    # ── Development-sized-effect probe: recorded, and deliberately not gating.
    # Decision 115's amendment: this decides whether a NULL here would be
    # interpretable. It says nothing about a detection, so it does not refuse.
    planted_shift = recorded("cluster_stage6_logloss_effect")
    positive_control = _plant(planted_shift)

    # ── Negative control: no effect where the aggregate has been destroyed.
    flipped_values, flipped_documents = _flip_whole_documents(deltas, documents, CONTROL_FLIP_SEED)
    negative_control = summarize_clustered_difference(
        flipped_values, documents, confidence=DIFFERENCE_CONFIDENCE
    )
    if negative_control.interval_excludes_zero:
        raise _refuse(
            f"NEGATIVE CONTROL FIRED. Reversing the sign of {flipped_documents} whole "
            f"documents under seed {CONTROL_FLIP_SEED} still produced the interval "
            f"[{negative_control.ci_low:+.6f}, {negative_control.ci_high:+.6f}], which "
            f"excludes zero. The procedure finds an effect in differences whose "
            f"aggregate direction has been destroyed, so it cannot be trusted to "
            f"have found a real one in the differences that have not."
        )

    values: dict[str, Any] = {
        f"{PREFIX}decision": DECISION,
        f"{PREFIX}documents": n_documents,
        f"{PREFIX}sentences": len(deltas),
        f"{PREFIX}cluster_unit": "document",
        f"{PREFIX}effect_unit": "nats",
        f"{PREFIX}largest_document": primary.largest_cluster,
        f"{PREFIX}positive_documents": primary.positive_clusters,
        f"{PREFIX}negative_documents": primary.negative_clusters,
        f"{PREFIX}resamples": DEFAULT_RESAMPLES,
        f"{PREFIX}seed": DEFAULT_SEED,
        f"{PREFIX}p": round(primary.p_value, NATS_PLACES),
        f"{PREFIX}p_floor": float(f"{primary.p_floor:.4g}"),
        f"{PREFIX}control_flip_seed": CONTROL_FLIP_SEED,
        f"{PREFIX}control_flipped_documents": flipped_documents,
        f"{PREFIX}control_planted_shift_nats": round(planted_shift, NATS_PLACES),
        f"{PREFIX}min_detectable_effect_nats": round(minimum_detectable, NATS_PLACES),
        f"{PREFIX}min_detectable_tolerance_nats": MDE_TOLERANCE_NATS,
        f"{PREFIX}probe_above_detects": bool(probe_above.interval_excludes_zero),
        f"{PREFIX}probe_below_detects": bool(probe_below.interval_excludes_zero),
        f"{PREFIX}development_effect_resolved": bool(positive_control.interval_excludes_zero),
    }
    values.update(_summary_values(PREFIX, primary))
    values.update(_summary_values(f"{PREFIX}corrected_", corrected))
    values.update(_summary_values(f"{PREFIX}centered_baseline_", centered_summary))
    values.update(_summary_values(f"{PREFIX}control_positive_", positive_control))
    values.update(_summary_values(f"{PREFIX}control_negative_", negative_control))

    stale = [
        label
        for label in ledger.labels_owned_by(SOURCE)
        if label.startswith(PREFIX) and label not in values
    ]
    ledger.replace_claims(
        remove=stale,
        batches=[ClaimBatch(values, run_id=RUN_ID, source=SOURCE, status="descriptive")],
    )
    verdict = "EXCLUDES zero" if primary.interval_excludes_zero else "SPANS zero"
    print(
        f"Recorded {len(values)} held-back Stage 6 clustered claims "
        f"(decision {DECISION}); the primary interval {verdict}."
    )
    for label, value in sorted(values.items()):
        print(f"  {label} = {value}")
    return 0


if __name__ == "__main__":
    # The ledger is single-writer: ``ClaimsLedger`` reads the whole file at
    # construction and writes the whole file at the end, so two overlapping
    # recorders lose one of the two results with no error on either side.
    with single_writer_lock(ledger_lock_path(PROJECT_ROOT)):
        raise SystemExit(main())
