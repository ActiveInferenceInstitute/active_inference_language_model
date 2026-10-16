#!/usr/bin/env python3
"""Record every Stage 5 clause-level number, uncertainty, and control.

Thin orchestrator; computation lives in ``ailm.levels.clause``. Single writer
for every ``stage5_*`` label.

What is being recorded
----------------------
The Stage 5 finding (decisions 24 and 83): on this UD EWT split at bigram order,
the mixture does not beat its type-free bar, and the paired privileged-label
gain is unresolved. Recorded with the same rigor as a positive result:

* the three perplexities (mixture, type-free bar, and privileged true-type
  reference) over the identical held-out positions; the reference is exceedable
  and is not an information bound;
* paired document-cluster bootstrap intervals for the privileged-label reference gain,
  both in nats per sentence and on the position-weighted scale underlying the
  published perplexity comparison;
* MAP classification against the always-majority baseline, McNemar-paired,
  with per-type recall and the prior-free AUC;
* the length-only comparator (a clause-blind classifier that also fails);
* the **detectability control**: the identical harness on a synthetic corpus
  whose types genuinely differ, where classification and perplexity must both
  fire decisively — without this the null would be indistinguishable from a
  broken instrument (the failure-to-reject trap).

Guards — the recorder refuses to write when
-------------------------------------------
1. The closed-form mixture evidence disagrees with the oracle
   ``forward_filter`` on the joint block-diagonal chain (decoder equivalence).
2. The positive control fails to fire (classification ≤ 0.9 against a 0.5
   majority, or mixture ≥ type-free) — the instrument cannot detect a planted
  effect, so the non-passing EWT result would remain instrument-ambiguous.
3. The artifact's observation alphabet strays outside the 8 case roles — the
   structural disjointness guard inherited from decision 16.

Usage
-----
    .venv/bin/python scripts/record_stage5_claims.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402
from ailm.statistics import exact_two_sided_binomial_p  # noqa: E402

RUN_ID = "stage5_clause_gate"
SOURCE = "scripts/record_stage5_claims.py"


def main() -> int:
    from ailm.claims import ClaimBatch, load_ledger
    from ailm.levels.clause import (
        CLAUSE_TYPES,
        STAGE5_BOOTSTRAP_CONFIDENCE,
        STAGE5_BOOTSTRAP_RESAMPLES,
        STAGE5_BOOTSTRAP_SEED,
        build_joint_emission,
        build_joint_start,
        build_joint_transition,
        build_type_free_chain,
        build_type_models,
        classify_sentences,
        clustered_binary_auc_interval,
        load_clause_artifact,
        proper_scores,
        sequence_log_evidence,
        summarize_oracle_gain,
        synthetic_control_corpus,
        type_conditional_perplexities,
    )
    from ailm.verification.oracle import forward_filter

    artifact = load_clause_artifact()
    models = build_type_models(artifact.train)
    comparator = build_type_free_chain(artifact.train)

    # ── Guard 1: decoder equivalence, closed form vs oracle on the joint ─────
    emission = build_joint_emission()
    joint_b = build_joint_transition(models)
    joint_d = build_joint_start(models)
    worst = 0.0
    for _sentence_type, sequence in artifact.test[:50]:
        closed, _ = sequence_log_evidence(sequence, models)
        _, oracle = forward_filter(emission, joint_b, joint_d, sequence)
        worst = max(worst, abs(closed - oracle))
    if worst > 1e-9:
        raise SystemExit(
            f"decoder equivalence FAILED: closed-form mixture and oracle "
            f"filtering disagree by {worst:.3e}; neither may be recorded."
        )

    # ── Guard 2: the detectability control must fire ─────────────────────────
    control_train, control_test = synthetic_control_corpus()
    control_models = build_type_models(control_train)
    control_ppl = type_conditional_perplexities(
        control_test, control_models, build_type_free_chain(control_train)
    )
    control_cls = classify_sentences(control_test, control_models)
    from ailm.gates import STAGE5_DETECTABILITY_ACCURACY_FLOOR

    if not (
        control_cls["accuracy"] > STAGE5_DETECTABILITY_ACCURACY_FLOOR
        and control_ppl["mixture"] < control_ppl["type_free"]
    ):
        raise SystemExit(
            f"detectability control FAILED (acc {control_cls['accuracy']:.3f}, "
            f"mixture {control_ppl['mixture']:.3f} vs type-free "
            f"{control_ppl['type_free']:.3f}) — the instrument cannot detect a "
            f"planted effect, so the English non-passing result remains instrument-ambiguous."
        )

    # ── The measurement ──────────────────────────────────────────────────────
    perplexities = type_conditional_perplexities(artifact.test, models, comparator)
    classification = classify_sentences(artifact.test, models)
    oracle_gain = summarize_oracle_gain(
        artifact.test,
        models,
        comparator,
        clusters=artifact.test_sequence_document_indices,
    )

    # Honest uncertainty for the rank statistic: whole authenticated source
    # documents move together. Degenerate single-class draws are rejected by
    # the production helper, which also requires each class in multiple docs.
    auc_summary = clustered_binary_auc_interval(
        classification["scores"],
        classification["labels"],
        artifact.test_sequence_document_indices,
    )
    b = sum(
        1
        for m, j in zip(classification["model_hits"], classification["majority_hits"], strict=True)
        if m and not j
    )
    c = sum(
        1
        for m, j in zip(classification["model_hits"], classification["majority_hits"], strict=True)
        if j and not m
    )
    p = exact_two_sided_binomial_p(b, c)

    # Classification accuracy is a paired sentence outcome but documents are
    # the outer sampling unit. The interval is primary; pair-level McNemar is
    # retained as the historical conditional analysis.
    from ailm.clustering import summarize_clustered_difference

    classification_differences = [
        100.0 * (int(model) - int(base))
        for model, base in zip(
            classification["model_hits"], classification["majority_hits"], strict=True
        )
    ]
    classification_clustered = summarize_clustered_difference(
        classification_differences,
        artifact.test_sequence_document_indices,
        confidence=STAGE5_BOOTSTRAP_CONFIDENCE,
        resamples=STAGE5_BOOTSTRAP_RESAMPLES,
        seed=STAGE5_BOOTSTRAP_SEED,
    )

    # Length-only comparator: threshold fitted on train, applied to test.
    best = (0.0, 1)
    for threshold in range(1, 21):
        accuracy = float(
            np.mean(
                [
                    (("interrogative" if len(seq) <= threshold else "declarative") == sentence_type)
                    for sentence_type, seq in artifact.train
                ]
            )
        )
        if accuracy > best[0]:
            best = (accuracy, threshold)
    length_accuracy = float(
        np.mean(
            [
                ("interrogative" if len(seq) <= best[1] else "declarative") == sentence_type
                for sentence_type, seq in artifact.test
            ]
        )
    )

    by_type_train = {name: sum(1 for t, _ in artifact.train if t == name) for name in CLAUSE_TYPES}
    by_type_test = {name: sum(1 for t, _ in artifact.test if t == name) for name in CLAUSE_TYPES}

    # Threshold-free scores that AUC alone cannot supply. The Brier reference is
    # the constant training prevalence — the probabilistic counterpart of the
    # majority-class accuracy baseline — so it never sees a test label.
    train_prevalence = by_type_train["interrogative"] / len(artifact.train)
    proper = proper_scores(classification["scores"], classification["labels"], train_prevalence)

    values: dict[str, float | int] = {
        "stage5_train_sentences": len(artifact.train),
        "stage5_test_sentences": len(artifact.test),
        "stage5_source_documents": len(artifact.document_ids),
        "stage5_document_split_train": len(artifact.train_split_document_indices),
        "stage5_document_split_test": len(artifact.test_split_document_indices),
        "stage5_train_documents": len(artifact.train_document_ids),
        "stage5_test_documents": len(artifact.test_document_ids),
        "stage5_train_interrogatives": by_type_train["interrogative"],
        "stage5_test_interrogatives": by_type_test["interrogative"],
        "stage5_test_positions": perplexities["positions"],
        "stage5_excluded_exclamative": artifact.provenance["excluded_exclamative"],
        "stage5_excluded_unpunctuated": artifact.provenance["excluded_unpunctuated"],
        "stage5_mixture_perplexity": round(perplexities["mixture"], 4),
        "stage5_typefree_perplexity": round(perplexities["type_free"], 4),
        "stage5_type_oracle_perplexity": round(perplexities["type_oracle"], 4),
        "stage5_privileged_label_gain_pct": round(
            100.0
            * (perplexities["type_free"] - perplexities["type_oracle"])
            / perplexities["type_free"],
            3,
        ),
        "stage5_oracle_gain_nats_per_sentence": round(oracle_gain.mean_nats_per_sentence, 6),
        "stage5_oracle_gain_sentence_ci_low": round(oracle_gain.sentence_ci_low, 6),
        "stage5_oracle_gain_sentence_ci_high": round(oracle_gain.sentence_ci_high, 6),
        "stage5_oracle_gain_nats_per_position": round(oracle_gain.pooled_nats_per_position, 6),
        "stage5_oracle_gain_position_ci_low": round(oracle_gain.position_ci_low, 6),
        "stage5_oracle_gain_position_ci_high": round(oracle_gain.position_ci_high, 6),
        "stage5_oracle_gain_ci_confidence_pct": 100.0 * oracle_gain.confidence,
        "stage5_oracle_gain_bootstrap_resamples": oracle_gain.resamples,
        "stage5_oracle_gain_bootstrap_seed": oracle_gain.seed,
        "stage5_oracle_gain_bootstrap_documents": oracle_gain.n_documents,
        "stage5_classification_pct": round(100.0 * classification["accuracy"], 2),
        "stage5_majority_baseline_pct": round(100.0 * classification["majority_accuracy"], 2),
        "stage5_interrogative_recall_pct": round(
            100.0 * classification["recalls"]["interrogative"], 2
        ),
        "stage5_auc": round(classification["auc"], 4),
        "stage5_auc_ci99_low": round(auc_summary.ci_low, 4),
        "stage5_auc_ci99_high": round(auc_summary.ci_high, 4),
        "stage5_auc_ci_confidence_pct": 100.0 * auc_summary.confidence,
        "stage5_auc_bootstrap_documents": auc_summary.n_documents,
        "stage5_classification_margin_pts": round(classification_clustered.effect, 4),
        "stage5_classification_document_ci_low": round(classification_clustered.ci_low, 4),
        "stage5_classification_document_ci_high": round(classification_clustered.ci_high, 4),
        "stage5_classification_document_clusters": classification_clustered.n_clusters,
        "stage5_classification_document_largest_cluster": (
            classification_clustered.largest_cluster
        ),
        "stage5_classification_document_p": float(f"{classification_clustered.p_value:.4g}"),
        "stage5_classification_document_p_floor": float(f"{classification_clustered.p_floor:.4g}"),
        "stage5_classification_document_ci_confidence_pct": (
            100.0 * classification_clustered.confidence
        ),
        "stage5_classification_document_bootstrap_resamples": classification_clustered.resamples,
        "stage5_classification_document_bootstrap_seed": classification_clustered.seed,
        "stage5_average_precision": round(proper["average_precision"], 4),
        "stage5_positive_prevalence_pct": round(100.0 * proper["prevalence"], 2),
        "stage5_train_prevalence_pct": round(100.0 * train_prevalence, 2),
        "stage5_brier": round(proper["brier"], 4),
        "stage5_brier_reference": round(proper["brier_reference"], 4),
        "stage5_brier_skill": round(proper["brier_skill"], 4),
        "stage5_mcnemar_b": b,
        "stage5_mcnemar_c": c,
        "stage5_mcnemar_p": float(p),
        "stage5_length_classifier_pct": round(100.0 * length_accuracy, 2),
        "stage5_decoder_max_divergence": worst,
        "stage5_control_classification_pct": round(100.0 * control_cls["accuracy"], 2),
        # The control corpus is balanced by construction, so its majority
        # baseline is 50 — but "by construction" is a property of the
        # generator's signature, and the paper typed the 50 beside a measured
        # accuracy as though both came from the same run. Measured from the
        # corpus actually built, it moves if the generator ever does.
        "stage5_control_majority_baseline_pct": round(
            100.0
            * max(
                sum(1 for stype, _ in control_test if stype == candidate)
                for candidate in {stype for stype, _ in control_test}
            )
            / len(control_test),
            2,
        ),
        "stage5_control_mixture_perplexity": round(control_ppl["mixture"], 4),
        "stage5_control_typefree_perplexity": round(control_ppl["type_free"], 4),
    }

    ledger = load_ledger(_PROJECT_ROOT)
    stale = [label for label in ledger.labels_owned_by(SOURCE) if label not in values]
    ledger.replace_claims(
        remove=stale,
        batches=[
            ClaimBatch(
                values=values,
                run_id=RUN_ID,
                source=SOURCE,
                status="descriptive",
            )
        ],
    )

    print(f"Recorded {len(values)} Stage 5 claims (non-passing result, bounds, and controls).")
    for label, value in sorted(values.items()):
        print(f"  {label} = {value}")
    return 0


if __name__ == "__main__":
    # The ledger is single-writer: ``ClaimsLedger`` reads the whole file at
    # construction and writes the whole file at the end, so two overlapping
    # recorders lose one of the two results, with no error on either side.
    with single_writer_lock(ledger_lock_path(_PROJECT_ROOT)):
        raise SystemExit(main())
