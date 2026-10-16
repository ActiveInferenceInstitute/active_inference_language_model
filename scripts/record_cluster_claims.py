#!/usr/bin/env python3
"""Re-run every paired comparison at its natural independent sampling unit.

Thin orchestrator. The estimator lives in ``ailm.clustering``; the models and
splits live in ``ailm.levels``. This script is the single writer for every
``cluster_*`` label.

## What this recorder is for

The project's paired tests are exact conditional on independent matched pairs.
Where the scored observations nest inside a larger unit that assumption does not
hold, and the recorded p-values and intervals describe more independent evidence
than the corpus contains. This recorder measures the same effect at the same
scale over the unit that actually varies:

| Arm             | Scored observation | Independent unit |
|-----------------|--------------------|------------------|
| Stage 2 G2P     | phone position     | held-out word    |
| Stage 4 gold    | token              | source document  |
| Stage 4 derived | token              | source document  |
| Stage 4 Czech   | token              | source document  |
| Stage 6         | sentence           | document         |

The inherited arms are recorded twice: once resampling the cluster and once
resampling the observation. The second is not a decoration — it is the
independence-assuming analysis computed by the same code, so the pair isolates
the clustering and nothing else. The design effect is the ratio of the empirical
variances of the exact bootstrap-replicate vectors used for those two percentile
intervals; it is not inferred from the interval widths. The Czech arm is the
exception: its existing token-level McNemar counts and finite log-tail stay as
the explicitly secondary conditional analysis, while this recorder computes
only the outer document-clustered primary. Bootstrapping its claim:stage4_czech_test_tokens
singleton tokens would add no evidence and would spend orders of magnitude more
memory than the primary calculation.

The wild cluster bootstrap (Cameron, Gelbach and Miller 2008, Section 4;
Rademacher weights at the cluster level) is recorded beside the cluster
bootstrap as a pre-registered sensitivity arm, never in place of it: for every
arm this recorder covers, ``wild_{arm}_*`` labels carry the wild interval
endpoints over the same demeaned-cluster perturbation machinery, the same
resample count, seed discipline, and coverage as the primary arm, and whether
its spans-zero verdict agrees with the recorded cluster-bootstrap verdict.
The count of agreeing arms is ``wild_verdict_agreements``; a disagreement is
recorded as a finding, not refused, because the prediction is about the data
and the ledger must be able to say the prediction failed.

The cluster-bootstrap interval carries primary uncertainty. Whole-cluster
sign-flip p-values are assumption-conditional sensitivity analyses: exactness
requires independent cluster sums with a joint distribution invariant under
whole-cluster sign reversal. That is stronger than marginally fair discordance
directions or zero expected cluster sum and is not pair-level McNemar lifted
automatically to clusters. The normal-tail values are approximations, not exact
weak-null tests.

Stage 6 is recorded on two scales because its registered test and its effect
size are different quantities. ``stage6_sign`` is the sign-transformed analysis,
which is the recorded sentence sign test with the unit moved to the document;
``stage6_logloss`` is the mean held-out log-loss improvement in nats, which is
the effect size that has to survive into the paper whatever the test does.

## Arms this recorder deliberately does NOT cover

A cluster key must be *recoverable* from the vendored artifacts, not guessed.
Where it is not, nothing is recorded here and the reason is stated:

- **Stage 5 (clause) sentence-type classification.** Its v2 artifact now carries
  authenticated document membership. The owning Stage 5 recorder computes the
  document-clustered classification, privileged-label gain, and AUC intervals
  together so their shared resampling contract cannot drift across recorders.
- **The L1→L2B coupling sweep and the L3→L2B lexical sweep.** Their cluster key
  *is* recoverable — both score positions word by word — but both report the
  comparison at an ``alpha`` chosen on the same evaluation set. A cluster-aware
  p-value there would correct the sampling unit while leaving the selection
  uncorrected, which invites reading a still-invalid number as fixed.
- **The Stage 2 subsample arms.** They exist to demonstrate that a subsample
  cannot carry the significance claim. Re-testing them at the word unit answers
  a question nobody asked of them.

## These p-values do not join the significance family

``SIGNIFICANCE_FAMILY`` in ``ailm.statistics`` enumerates the project's declared
tests. Each non-Czech ``cluster_*_normal_p`` recorded here is a *reanalysis of a
test already in it* rather than an additional hypothesis. Adding them would
enlarge the Bonferroni denominator for looking at the same comparisons a second
time, which penalizes the correction instead of the multiplicity. The Czech arm
was exploratory from its development-session specification onward and remains
outside that family on both sampling scales.

## Ordering

This recorder cross-checks its recomputation against labels owned by the stage
recorders and refuses to write if they are absent, so on a from-scratch ledger
the stage recorders must run first. ``orchestrate_claim_recorders.py`` discovers
the inventory dynamically but applies explicit producer→consumer edges, rather
than trusting filename order or the stale values in a populated ledger.

Run:  .venv/bin/python scripts/record_cluster_claims.py
"""

from __future__ import annotations

import math
import sys
from collections.abc import Hashable, Sequence
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ailm.clustering import ClusteredPairedSummary  # noqa: E402
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402

RUN_ID = "cluster_aware_reanalysis_v3"
SOURCE = "scripts/record_cluster_claims.py"

#: How far the closed-form naive p-value may sit from the recorded exact p-value
#: before this recorder refuses to write, as a factor. The two are different
#: approximations of the same null — the recorded one is exact-binomial, this
#: one is a normal tail with no continuity correction — so they do not agree to
#: many digits, and requiring that would be a false precision. They must agree to
#: well within an order of magnitude, which is enough to catch a mis-derived
#: difference vector, a mis-signed comparator, or a wrong position set: every one
#: of those moves the statistic far more than the approximation does.
NAIVE_AGREEMENT_FACTOR = 2.0


def _refuse(message: str) -> int:
    print(f"REFUSING TO RECORD: {message}", file=sys.stderr)
    return 1


def _g2p_differences() -> tuple[list[float], list[str]]:
    """Per-position G2P hit differences in percentage points, keyed by word.

    Reproduces the position vector ``scripts/record_stage2_claims.py`` pairs its
    McNemar test over — same split, same oracle matrices, same length-matching
    rule — and carries the word alongside each position. The word is the cluster:
    every position in it shares one lexical item, one filtering trajectory, and
    one orthographic context.

    The recomputation is verified against the recorded discordant counts by the
    caller. That check is what makes this a reanalysis of the manuscript's
    measurement rather than a second, similar-looking measurement.
    """
    from ailm.levels.letter import ALPHA_TO_IDX
    from ailm.levels.phoneme import (
        PHONES,
        build_g2p_emission,
        build_g2p_start,
        build_g2p_transition,
        g2p_split,
        load_phoneme_counts,
        per_letter_argmax_baseline,
    )
    from ailm.verification.oracle import forward_filter

    counts = load_phoneme_counts(_PROJECT_ROOT / "vendor")
    train, test = g2p_split(counts)
    emission = np.array(build_g2p_emission(train))[0]
    transition = np.array(build_g2p_transition(train))[0, :, :, 0]
    start = np.array(build_g2p_start(train)).ravel()
    mapping = per_letter_argmax_baseline(train)

    differences: list[float] = []
    words: list[str] = []
    for word, phones in test:
        letters = [character for character in word if character in ALPHA_TO_IDX]
        if not letters:
            continue
        posteriors, _ = forward_filter(
            emission, transition, start, [ALPHA_TO_IDX[c] for c in letters]
        )
        decoded = [PHONES[int(np.argmax(row))] for row in posteriors]
        if len(decoded) != len(phones):
            continue
        for predicted, letter, phone in zip(decoded, letters, phones, strict=True):
            differences.append(
                100.0 * (int(predicted == phone) - int(mapping.get(letter) == phone))
            )
            words.append(word)
    return differences, words


def _case_differences(population: str) -> tuple[list[float], list[int], int, int]:
    """Per-token Stage 4 hit differences, keyed by authenticated source document.

    ``decode_case_positions`` and ``baseline_case_positions`` both emit one entry
    per held-out token in sentence-then-position order, which is what makes them
    pairable at all. The artifact retains one document index per sequence; the
    same iteration order expands that outer key to every scored token without
    reopening or re-parsing the raw corpus.
    """
    from ailm.levels.case import (
        baseline_case_positions,
        build_case_emission,
        build_case_start,
        build_case_transition,
        decode_case_positions,
        load_case_artifact,
    )

    artifact = load_case_artifact(population)
    emission = build_case_emission(artifact)
    transition = build_case_transition(artifact)
    start = build_case_start(artifact)

    model_hits = decode_case_positions(artifact, emission, transition, start)
    baseline_hits = baseline_case_positions(artifact)
    if len(model_hits) != len(baseline_hits):
        raise ValueError(
            f"unpaired Stage 4 {population} vectors: "
            f"{len(model_hits)} model vs {len(baseline_hits)} baseline"
        )
    differences = [
        100.0 * (int(model) - int(base))
        for model, base in zip(model_hits, baseline_hits, strict=True)
    ]
    if len(artifact.test_sequence_document_indices) != len(artifact.test):
        raise ValueError(
            f"Stage 4 {population} artifact has no one-to-one sequence/document-index mapping"
        )
    documents = [
        artifact.test_sequence_document_indices[sequence_index]
        for sequence_index, sequence in enumerate(artifact.test)
        for _ in sequence
    ]
    if len(documents) != len(differences):
        raise ValueError(
            f"Stage 4 {population}: {len(differences)} scored tokens but the split "
            f"accounts for {len(documents)} document-keyed tokens; the cluster "
            "key is not the split"
        )
    return differences, documents, len(artifact.test), artifact.n_test_documents


def _czech_case_differences() -> tuple[list[float], list[int], int, int]:
    """Per-token Czech hit differences, keyed by authenticated source document.

    Each retained sequence carries an index into the ``# newdoc id`` alphabet
    derived from the authenticated dev shard.  The decoders flatten in
    sequence-then-token order, so repeating that outer document index by the
    sequence length recovers the primary unit without consulting raw data here.
    """
    from typing import cast

    from ailm.levels.case import CaseArtifact
    from ailm.levels.case_czech import (
        baseline_case_positions,
        build_case_emission,
        build_case_start,
        build_case_transition,
        decode_case_positions,
        load_czech_case_artifact,
    )

    artifact = load_czech_case_artifact()
    shared_artifact = cast(CaseArtifact, artifact)
    model_hits = decode_case_positions(
        shared_artifact,
        build_case_emission(shared_artifact),
        build_case_transition(shared_artifact),
        build_case_start(shared_artifact),
    )
    baseline_hits = baseline_case_positions(shared_artifact)
    if len(model_hits) != len(baseline_hits):
        raise ValueError(
            "unpaired Stage 4 Czech vectors: "
            f"{len(model_hits)} model vs {len(baseline_hits)} baseline"
        )
    differences = [
        100.0 * (int(model) - int(base))
        for model, base in zip(model_hits, baseline_hits, strict=True)
    ]
    if len(artifact.test_sequence_document_indices) != len(artifact.test):
        raise ValueError("Stage 4 Czech artifact has no one-to-one sequence/document-index mapping")
    documents = [
        artifact.test_sequence_document_indices[sequence_index]
        for sequence_index, sequence in enumerate(artifact.test)
        for _ in sequence
    ]
    if len(documents) != len(differences):
        raise ValueError(
            f"Stage 4 Czech: {len(differences)} scored tokens but the split "
            f"accounts for {len(documents)} document-keyed tokens; the cluster "
            "key is not the split"
        )
    return differences, documents, len(artifact.test), artifact.n_test_documents


def _narrative_differences() -> tuple[list[float], list[str], int]:
    """Per-sentence Stage 6 log-loss deltas in nats, keyed by document."""
    from ailm.levels.narrative import (
        build_phase_models,
        heldout_type_logloss,
        load_narrative_artifact,
    )

    artifact = load_narrative_artifact()
    recognition = heldout_type_logloss(artifact.test, build_phase_models(artifact.train))
    deltas = [float(delta) for delta in recognition["deltas"]]
    documents = [
        document_id
        for document_id, document in zip(artifact.test_document_ids, artifact.test, strict=True)
        for _ in document
    ]
    if len(documents) != len(deltas):
        raise ValueError(
            f"Stage 6: {len(deltas)} scored sentences but the split accounts for "
            f"{len(documents)}; the cluster key is not the split"
        )
    return deltas, documents, len(artifact.test)


def _arm_values(
    arm: str,
    differences: Sequence[float],
    clusters: Sequence[Hashable],
    *,
    cluster_unit: str,
    effect_unit: str,
    include_naive: bool = True,
) -> tuple[dict[str, object], ClusteredPairedSummary, ClusteredPairedSummary | None]:
    """Both analyzes of one arm, plus the two summaries for the caller's guards."""
    from ailm.clustering import (
        intracluster_correlation,
        summarize_clustered_difference,
        summarize_wild_cluster_difference,
        variance_inflation,
    )

    clustered = summarize_clustered_difference(differences, clusters)
    wild = summarize_wild_cluster_difference(differences, clusters)

    values: dict[str, object] = {
        f"cluster_{arm}_cluster_unit": cluster_unit,
        f"cluster_{arm}_effect_unit": effect_unit,
        f"cluster_{arm}_observations": clustered.n_observations,
        f"cluster_{arm}_clusters": clustered.n_clusters,
        f"cluster_{arm}_largest_cluster": clustered.largest_cluster,
        f"cluster_{arm}_positive_clusters": clustered.positive_clusters,
        f"cluster_{arm}_negative_clusters": clustered.negative_clusters,
        f"cluster_{arm}_effect": round(clustered.effect, 6),
        f"cluster_{arm}_ci_low": round(clustered.ci_low, 6),
        f"cluster_{arm}_ci_high": round(clustered.ci_high, 6),
        f"cluster_{arm}_p": float(f"{clustered.p_value:.4g}"),
        f"cluster_{arm}_normal_p": float(f"{clustered.normal_p_value:.4g}"),
        f"cluster_{arm}_z": round(clustered.z_statistic, 4),
    }

    naive = summarize_clustered_difference(differences, None) if include_naive else None
    if naive is not None:
        values.update(
            {
                f"cluster_{arm}_naive_ci_low": round(naive.ci_low, 6),
                f"cluster_{arm}_naive_ci_high": round(naive.ci_high, 6),
                f"cluster_{arm}_naive_normal_p": float(f"{naive.normal_p_value:.4g}"),
                f"cluster_{arm}_naive_z": round(naive.z_statistic, 4),
            }
        )

    # Both diagnostics are undefined on a difference vector with no variation —
    # Stage 4's gold population is exactly that, since its decoder and its
    # comparator never once disagree. Recording a zero there would assert that
    # the clustering was measured and found irrelevant, when it was not
    # measurable at all.
    values.update(
        {
            f"wild_{arm}_ci_low": round(wild.ci_low, 6),
            f"wild_{arm}_ci_high": round(wild.ci_high, 6),
            f"wild_{arm}_verdict_agrees": (
                wild.interval_excludes_zero == clustered.interval_excludes_zero
            ),
        }
    )

    if naive is not None and naive.interval_width > 0.0:
        values[f"cluster_{arm}_variance_inflation"] = round(variance_inflation(clustered, naive), 4)
    correlation = intracluster_correlation(differences, clusters)
    if correlation is not None:
        values[f"cluster_{arm}_icc"] = round(correlation, 6)

    return values, clustered, naive


def _agrees_with_recorded_exact_p(naive_p: float, exact_p: float) -> bool:
    """Whether the closed-form naive p reproduces the recorded exact p.

    Compared as a ratio rather than a difference: these p-values span seven
    orders of magnitude across the arms, so an absolute tolerance would be
    vacuous at one end and impossible at the other.
    """
    if exact_p <= 0.0 or naive_p <= 0.0:
        return False
    return abs(math.log(naive_p / exact_p)) <= math.log(NAIVE_AGREEMENT_FACTOR)


def main() -> int:
    from ailm.claims import load_ledger
    from ailm.clustering import DEFAULT_RESAMPLES, DEFAULT_SEED
    from ailm.statistics import DIFFERENCE_CONFIDENCE

    ledger = load_ledger(_PROJECT_ROOT)

    def recorded(label: str) -> float:
        entry = ledger.get(label)
        if entry is None:
            raise KeyError(f"required recorded claim is absent: {label}")
        return float(entry["value"])

    values: dict[str, object] = {}
    arms: list[str] = []
    # Every arm whose recorded test is a sign test over a paired difference, with
    # the exact p-value the naive path must reproduce. Stage 6's magnitude arm is
    # not in this map: its recorded test is on the sign, not the mean in nats.
    exact_p_labels: dict[str, str] = {}

    # ── Stage 2: phone positions nested in held-out words ────────────────────
    prerequisites = (
        "stage2_g2p_mcnemar_b",
        "stage2_g2p_mcnemar_c",
        "stage2_g2p_mcnemar_p",
        "stage2_g2p_eval_positions",
        "stage2_g2p_margin_pts",
        "stage2_g2p_test_pairs",
        "stage4_gold_mcnemar_b",
        "stage4_gold_mcnemar_c",
        "stage4_gold_mcnemar_p",
        "stage4_gold_test_tokens",
        "stage4_gold_test_sentences",
        "stage4_gold_test_documents",
        "stage4_derived_mcnemar_b",
        "stage4_derived_mcnemar_c",
        "stage4_derived_mcnemar_p",
        "stage4_derived_test_tokens",
        "stage4_derived_test_sentences",
        "stage4_derived_test_documents",
        "stage4_czech_mcnemar_b",
        "stage4_czech_mcnemar_c",
        "stage4_czech_mcnemar_log10_p",
        "stage4_czech_test_tokens",
        "stage4_czech_test_sequences",
        "stage4_czech_test_documents",
        "stage4_czech_margin_pts",
        "stage6_sign_test_positive",
        "stage6_sign_test_negative",
        "stage6_sign_test_p",
        "stage6_test_sentences",
        "stage6_test_documents",
        "stage6_logloss_improvement_nats",
    )
    missing = [label for label in prerequisites if ledger.get(label) is None]
    if missing:
        return _refuse(
            "this recorder reanalyzes measurements it does not own and verifies "
            "its recomputation against them, so the owning recorders must run "
            f"first. Missing: {missing}"
        )

    g2p_differences, g2p_words = _g2p_differences()
    g2p_b = sum(1 for value in g2p_differences if value > 0)
    g2p_c = sum(1 for value in g2p_differences if value < 0)
    if (g2p_b, g2p_c) != (
        int(recorded("stage2_g2p_mcnemar_b")),
        int(recorded("stage2_g2p_mcnemar_c")),
    ):
        return _refuse(
            f"the G2P recomputation found b={g2p_b}, c={g2p_c} but the ledger "
            f"records b={int(recorded('stage2_g2p_mcnemar_b'))}, "
            f"c={int(recorded('stage2_g2p_mcnemar_c'))}. This is not the same "
            f"comparison, so a cluster analysis of it would not correct the "
            f"recorded result."
        )
    if len(g2p_differences) != int(recorded("stage2_g2p_eval_positions")):
        return _refuse(
            f"the G2P recomputation scored {len(g2p_differences)} positions but the "
            f"ledger records {int(recorded('stage2_g2p_eval_positions'))}."
        )

    arm_values, clustered, naive = _arm_values(
        "stage2_g2p",
        g2p_differences,
        g2p_words,
        cluster_unit="word",
        effect_unit="percentage points",
    )
    if naive is None:
        return _refuse("the G2P control unexpectedly omitted its observation-level analysis")
    values.update(arm_values)
    arms.append("stage2_g2p")
    clustered_summaries: dict[str, ClusteredPairedSummary] = {"stage2_g2p": clustered}
    naive_summaries: dict[str, ClusteredPairedSummary] = {"stage2_g2p": naive}
    exact_p_labels["stage2_g2p"] = "stage2_g2p_mcnemar_p"
    if round(clustered.effect, 3) != round(recorded("stage2_g2p_margin_pts"), 3):
        return _refuse(
            f"the reanalyzed G2P effect is {clustered.effect:.3f} points but the "
            f"ledger records {recorded('stage2_g2p_margin_pts')}. The cluster "
            f"reanalysis must leave the effect untouched and change only its "
            f"uncertainty."
        )
    if clustered.n_clusters != int(recorded("stage2_g2p_test_pairs")):
        return _refuse(
            f"the G2P word clusters number {clustered.n_clusters} but the held-out "
            f"split has {int(recorded('stage2_g2p_test_pairs'))} pairs."
        )

    # ── Stage 4: tokens and sentences nested in source documents ────────────
    for population in ("gold", "derived"):
        differences, documents, n_sentences, n_documents = _case_differences(population)
        favored = sum(1 for value in differences if value > 0)
        opposed = sum(1 for value in differences if value < 0)
        expected = (
            int(recorded(f"stage4_{population}_mcnemar_b")),
            int(recorded(f"stage4_{population}_mcnemar_c")),
        )
        if (favored, opposed) != expected:
            return _refuse(
                f"the Stage 4 {population} recomputation found b={favored}, "
                f"c={opposed} but the ledger records b={expected[0]}, c={expected[1]}."
            )
        if len(differences) != int(recorded(f"stage4_{population}_test_tokens")):
            return _refuse(
                f"the Stage 4 {population} recomputation scored {len(differences)} "
                f"tokens but the ledger records "
                f"{int(recorded(f'stage4_{population}_test_tokens'))}."
            )
        if n_sentences != int(recorded(f"stage4_{population}_test_sentences")):
            return _refuse(
                f"the Stage 4 {population} split has {n_sentences} held-out "
                f"sentences but the ledger records "
                f"{int(recorded(f'stage4_{population}_test_sentences'))}."
            )
        if n_documents != int(recorded(f"stage4_{population}_test_documents")):
            return _refuse(
                f"the Stage 4 {population} split has {n_documents} contributing "
                f"held-out documents but the ledger records "
                f"{int(recorded(f'stage4_{population}_test_documents'))}."
            )
        arm = f"stage4_{population}"
        arm_values, case_clustered, case_naive = _arm_values(
            arm,
            differences,
            documents,
            cluster_unit="document",
            effect_unit="percentage points",
        )
        if case_naive is None:
            return _refuse(
                f"the Stage 4 {population} control unexpectedly omitted its "
                "observation-level analysis"
            )
        values.update(arm_values)
        arms.append(arm)
        clustered_summaries[arm] = case_clustered
        naive_summaries[arm] = case_naive
        exact_p_labels[arm] = f"stage4_{population}_mcnemar_p"
        if case_clustered.n_clusters != n_documents:
            return _refuse(
                f"the Stage 4 {population} analysis recovered "
                f"{case_clustered.n_clusters} document clusters from "
                f"{n_documents} contributing source documents."
            )

    # ── Stage 4 Czech: retained sequences nested in source documents ───────
    czech_differences, czech_documents, n_czech_sequences, n_czech_documents = (
        _czech_case_differences()
    )
    czech_b = sum(1 for value in czech_differences if value > 0)
    czech_c = sum(1 for value in czech_differences if value < 0)
    expected_czech = (
        int(recorded("stage4_czech_mcnemar_b")),
        int(recorded("stage4_czech_mcnemar_c")),
    )
    if (czech_b, czech_c) != expected_czech:
        return _refuse(
            f"the Stage 4 Czech recomputation found b={czech_b}, c={czech_c} "
            f"but the ledger records b={expected_czech[0]}, c={expected_czech[1]}."
        )
    if len(czech_differences) != int(recorded("stage4_czech_test_tokens")):
        return _refuse(
            f"the Stage 4 Czech recomputation scored {len(czech_differences)} "
            f"tokens but the ledger records {int(recorded('stage4_czech_test_tokens'))}."
        )
    if n_czech_sequences != int(recorded("stage4_czech_test_sequences")):
        return _refuse(
            f"the Stage 4 Czech split has {n_czech_sequences} held-out sequences "
            "but the ledger records "
            f"{int(recorded('stage4_czech_test_sequences'))}."
        )
    if n_czech_documents != int(recorded("stage4_czech_test_documents")):
        return _refuse(
            f"the Stage 4 Czech split has {n_czech_documents} contributing source "
            "documents but the ledger records "
            f"{int(recorded('stage4_czech_test_documents'))}."
        )
    arm_values, czech_clustered, czech_naive = _arm_values(
        "stage4_czech",
        czech_differences,
        czech_documents,
        cluster_unit="document",
        effect_unit="percentage points",
        include_naive=False,
    )
    # The closed-form normal tail underflows at this effect size.  Zero is not
    # a probability worth recording; the finite Monte Carlo sign-flip sensitivity
    # result and its resolution floor are recorded explicitly beside the primary
    # cluster-bootstrap interval.
    arm_values.pop("cluster_stage4_czech_normal_p")
    if czech_naive is not None:
        return _refuse("the Czech primary unexpectedly ran an observation-level bootstrap")
    if round(czech_clustered.effect, 2) != round(recorded("stage4_czech_margin_pts"), 2):
        return _refuse(
            f"the reanalyzed Czech effect is {czech_clustered.effect:.2f} points "
            f"but the ledger records {recorded('stage4_czech_margin_pts')}. The "
            "cluster reanalysis must leave the effect untouched and change only "
            "its uncertainty."
        )
    if czech_clustered.n_clusters != n_czech_documents:
        return _refuse(
            f"the Czech analysis recovered {czech_clustered.n_clusters} document "
            f"clusters from {n_czech_documents} contributing source documents."
        )
    arm_values.update(
        {
            "cluster_stage4_czech_resamples": czech_clustered.resamples,
            "cluster_stage4_czech_seed": czech_clustered.seed,
            "cluster_stage4_czech_confidence": czech_clustered.confidence,
            "cluster_stage4_czech_confidence_pct": 100.0 * czech_clustered.confidence,
            "cluster_stage4_czech_p_floor": float(f"{czech_clustered.p_floor:.4g}"),
            "cluster_stage4_czech_p_at_resolution_floor": (czech_clustered.p_at_resolution_floor),
        }
    )
    values.update(arm_values)
    arms.append("stage4_czech")
    clustered_summaries["stage4_czech"] = czech_clustered

    # ── Stage 6: sentences nested in documents, on both scales ───────────────
    deltas, documents, n_documents = _narrative_differences()
    if len(deltas) != int(recorded("stage6_test_sentences")):
        return _refuse(
            f"the Stage 6 recomputation scored {len(deltas)} sentences but the "
            f"ledger records {int(recorded('stage6_test_sentences'))}."
        )
    if n_documents != int(recorded("stage6_test_documents")):
        return _refuse(
            f"the Stage 6 split has {n_documents} held-out documents but the "
            f"ledger records {int(recorded('stage6_test_documents'))}."
        )
    positive = sum(1 for delta in deltas if delta > 0)
    negative = sum(1 for delta in deltas if delta < 0)
    if (positive, negative) != (
        int(recorded("stage6_sign_test_positive")),
        int(recorded("stage6_sign_test_negative")),
    ):
        return _refuse(
            f"the Stage 6 recomputation found {positive} positive and {negative} "
            f"negative sentences but the ledger records "
            f"{int(recorded('stage6_sign_test_positive'))} and "
            f"{int(recorded('stage6_sign_test_negative'))}."
        )

    # The sign transform is what makes this the recorded test with a different
    # unit: the recorded Stage 6 p-value is a sign test, which uses only the
    # direction of each sentence's delta. Clustering the magnitudes would answer
    # a different question under the same label.
    signs = [float(np.sign(delta)) for delta in deltas]
    arm_values, sign_clustered, sign_naive = _arm_values(
        "stage6_sign",
        signs,
        documents,
        cluster_unit="document",
        effect_unit="signed sentence share",
    )
    if sign_naive is None:
        return _refuse("the Stage 6 sign control omitted its observation-level analysis")
    values.update(arm_values)
    arms.append("stage6_sign")
    clustered_summaries["stage6_sign"] = sign_clustered
    naive_summaries["stage6_sign"] = sign_naive
    exact_p_labels["stage6_sign"] = "stage6_sign_test_p"

    arm_values, logloss_clustered, logloss_naive = _arm_values(
        "stage6_logloss",
        deltas,
        documents,
        cluster_unit="document",
        effect_unit="nats",
    )
    if logloss_naive is None:
        return _refuse("the Stage 6 log-loss control omitted its observation-level analysis")
    values.update(arm_values)
    arms.append("stage6_logloss")
    clustered_summaries["stage6_logloss"] = logloss_clustered
    naive_summaries["stage6_logloss"] = logloss_naive
    if round(logloss_clustered.effect, 4) != round(recorded("stage6_logloss_improvement_nats"), 4):
        return _refuse(
            f"the reanalyzed Stage 6 log-loss improvement is "
            f"{logloss_clustered.effect:.4f} nats but the ledger records "
            f"{recorded('stage6_logloss_improvement_nats')}."
        )

    # ── Guards over the whole set ────────────────────────────────────────────
    # Vacuity: if a cluster key were mis-derived to one cluster per observation,
    # every "cluster-aware" number above would silently be the naive number and
    # nothing else in this recorder would notice.
    degenerate = [
        arm
        for arm in arms
        if clustered_summaries[arm].n_clusters >= clustered_summaries[arm].n_observations
    ]
    if degenerate:
        return _refuse(
            f"these arms have as many clusters as observations, so their cluster "
            f"analysis is the naive analysis under another name: {degenerate}"
        )

    # Positive control on the machinery itself: with one observation per cluster
    # the sign-flip statistic reduces to McNemar's, so the naive closed-form
    # p-value must reproduce each arm's recorded exact p-value.
    disagreements = {
        arm: (naive_summaries[arm].normal_p_value, recorded(label))
        for arm, label in exact_p_labels.items()
        if not _agrees_with_recorded_exact_p(naive_summaries[arm].normal_p_value, recorded(label))
    }
    if disagreements:
        return _refuse(
            f"the independence-assuming path does not reproduce the recorded exact "
            f"p-values within a factor of {NAIVE_AGREEMENT_FACTOR}: {disagreements}. "
            f"The cluster analysis would then be correcting a different comparison "
            f"from the one the manuscript reports."
        )

    values.update(
        {
            "cluster_arms": sorted(arms),
            "cluster_arm_count": len(arms),
            "cluster_resamples": DEFAULT_RESAMPLES,
            "cluster_seed": DEFAULT_SEED,
            "cluster_p_floor": float(f"{1.0 / (DEFAULT_RESAMPLES + 1.0):.4g}"),
            "cluster_confidence": DIFFERENCE_CONFIDENCE,
            "wild_resamples": DEFAULT_RESAMPLES,
            "wild_seed": DEFAULT_SEED,
            "wild_confidence": DIFFERENCE_CONFIDENCE,
            "wild_arm_count": len(arms),
            "wild_verdict_agreements": sum(
                int(values[f"wild_{arm}_verdict_agrees"]) for arm in arms
            ),
        }
    )

    # The Czech arm is exploratory by design; its effect and uncertainty must
    # retain that status even though the other cluster reanalyses historically
    # carry ``descriptive``.  One atomic multi-batch write prevents a crash from
    # leaving the protocol claims and the inferential claims out of sync.
    from ailm.claims import ClaimBatch

    czech_prefix = "cluster_stage4_czech_"
    wild_czech_prefix = "wild_stage4_czech_"
    wild_czech_exploratory_suffixes = {"ci_low", "ci_high"}
    czech_exploratory_suffixes = {
        "effect",
        "ci_low",
        "ci_high",
        "p",
        "normal_p",
        "z",
        "p_at_resolution_floor",
    }
    czech_exploratory = {
        label: value
        for label, value in values.items()
        if (
            label.startswith(czech_prefix)
            and label.removeprefix(czech_prefix) in czech_exploratory_suffixes
        )
        or (
            label.startswith(wild_czech_prefix)
            and label.removeprefix(wild_czech_prefix) in wild_czech_exploratory_suffixes
        )
    }
    czech_descriptive = {
        label: value
        for label, value in values.items()
        if (label.startswith(czech_prefix) and label not in czech_exploratory)
        or (
            label.startswith(wild_czech_prefix)
            and label.removeprefix(wild_czech_prefix) not in wild_czech_exploratory_suffixes
        )
    }
    inherited = {
        label: value
        for label, value in values.items()
        if not label.startswith(czech_prefix) and not label.startswith(wild_czech_prefix)
    }
    ledger.replace_claims(
        remove=["cluster_stage4_czech_normal_p"],
        batches=[
            ClaimBatch(inherited, RUN_ID, SOURCE),
            ClaimBatch(czech_exploratory, RUN_ID, SOURCE, "exploratory"),
            ClaimBatch(czech_descriptive, RUN_ID, SOURCE, "descriptive"),
        ],
    )

    print(f"Recorded {len(values)} cluster-aware claims over {len(arms)} arms.")
    for arm in sorted(arms):
        print(
            f"  {arm}: effect {values[f'cluster_{arm}_effect']} "
            f"{values[f'cluster_{arm}_effect_unit']} over "
            f"{values[f'cluster_{arm}_observations']} observations in "
            f"{values[f'cluster_{arm}_clusters']} "
            f"{values[f'cluster_{arm}_cluster_unit']}s; "
            f"cluster CI [{values[f'cluster_{arm}_ci_low']}, "
            f"{values[f'cluster_{arm}_ci_high']}]; cluster randomization p "
            f"{values[f'cluster_{arm}_p']}"
            + (
                f", cluster normal-tail p {values[f'cluster_{arm}_normal_p']}"
                if f"cluster_{arm}_normal_p" in values
                else ""
            )
            + (
                f" vs naive CI [{values[f'cluster_{arm}_naive_ci_low']}, "
                f"{values[f'cluster_{arm}_naive_ci_high']}] and naive normal-tail "
                f"p {values[f'cluster_{arm}_naive_normal_p']}"
                if f"cluster_{arm}_naive_ci_low" in values
                else ""
            )
            + (
                f"; wild CI [{values[f'wild_{arm}_ci_low']}, "
                f"{values[f'wild_{arm}_ci_high']}] verdict agrees with cluster "
                f"bootstrap: {values[f'wild_{arm}_verdict_agrees']}"
            )
        )
    print(
        f"  Wild cluster bootstrap (Rademacher weights) agrees with the cluster "
        f"bootstrap verdict on {values['wild_verdict_agreements']} of {len(arms)} arms."
    )
    print(
        "  Monte Carlo p-values are floored at "
        f"{values['cluster_p_floor']} by {DEFAULT_RESAMPLES} draws; the *_normal_p "
        "labels carry the self-normalized normal-tail approximation, which is not floored."
    )
    return 0


if __name__ == "__main__":
    # The ledger is single-writer: ``ClaimsLedger`` reads the whole file at
    # construction and writes the whole file at the end, so two overlapping
    # recorders lose one of the two results, with no error on either side.
    with single_writer_lock(ledger_lock_path(_PROJECT_ROOT)):
        raise SystemExit(main())
