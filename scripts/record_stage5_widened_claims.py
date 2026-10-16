#!/usr/bin/env python3
"""Record every widened-alphabet Stage 5 number, uncertainty, and control.

Thin orchestrator; computation lives in ``ailm.levels.clause`` (the shipped
estimators, called at the widened alphabet size) and ``ailm.levels.widened``.
Single writer for every ``stage5w_*`` label.

What is being recorded
----------------------
The widened-observation arm registered in ``docs/decisions.md`` entry 112: the
same clause level over an alphabet widened by one distinguished symbol per
``VERB`` and per ``AUX`` token in surface order, on decision 85's authenticated
document partition. Decision 114 made this arm the canonical Level 5 and the
case-only arm a retained ablation; that arm is untouched — it has its own
recorder, and ``scripts/audit_gates.py::audit_stage5_case_only`` still enforces
its recorded finding.

Recorded with the same rigor as the shipped arm:

* the three perplexities (mixture, the type-free bar REFITTED on the identical
  widened stream, and the privileged true-type reference) over the identical
  held-out positions; the reference is exceedable and is not an information
  bound;
* document-clustered paired bootstrap intervals for the privileged-label gain,
  in nats per sentence and on the position-weighted scale;
* MAP classification against the always-majority baseline, McNemar-paired, with
  per-type recall and the prior-free AUC with a document-clustered interval;
* the entropy of the type variable and the share of it the gain realizes, which
  is the quantity the widening was proposed to move;
* the two clause-blind comparators entry 112 registers — sentence length only,
  and the single first-symbol-is-``AUX`` indicator;
* the case-only arm as the does-not-use-the-widening baseline, scored on the
  sentences where BOTH arms are defined so the two AUCs are paired. Per-position
  perplexity is deliberately NOT compared across the two arms: their position
  counts and their alphabets differ, so that ratio would not be one comparison;
* the **detectability control**: the identical harness at the widened alphabet
  on a synthetic corpus whose types genuinely differ.

Guards — the recorder refuses to write when
-------------------------------------------
1. The closed-form mixture evidence disagrees with the oracle ``forward_filter``
   on the joint block-diagonal chain (decoder equivalence).
2. That equivalence check is vacuous — a **cross-type-leak** transition must
   make the same comparison disagree. A guard that cannot fail is a comment.
3. The **swapped-blocks** control classifies at least as well as the correct
   wiring. If exchanging the two types' chains costs nothing, the type
   assignment carries nothing and this arm is not measuring what it claims.
4. The positive detectability control fails to fire.
5. The artifact's observations stray outside the widened alphabet — the
   structural guard of decision 16 at the new alphabet.

Status
------
Every label is recorded ``descriptive``. Entry 112 is a development-session
specification with no independently timestamped prior artifact, so nothing here
is confirmatory, and none of its p-values joins the frozen multiplicity family
(entry 79); each is documented in ``EXCLUDED_FROM_FAMILY`` with its reason.

Usage
-----
    .venv/bin/python scripts/record_stage5_widened_claims.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402
from ailm.statistics import exact_two_sided_binomial_p  # noqa: E402

RUN_ID = "stage5_widened_clause_gate"
SOURCE = "scripts/record_stage5_widened_claims.py"


def _length_only_accuracy(train, test) -> float:
    """Best length threshold fitted on train, applied to test.

    Interrogatives are shorter than declaratives on this corpus, so a
    length-only rule is a live competitor rather than a strawman. The threshold
    is fitted on the training split; fitting it on the evaluation would be the
    comparator defect decision 54 records.
    """
    best = (0.0, 1)
    for threshold in range(1, 31):
        accuracy = float(
            np.mean(
                [
                    (("interrogative" if len(seq) <= threshold else "declarative") == stype)
                    for stype, seq in train
                ]
            )
        )
        if accuracy > best[0]:
            best = (accuracy, threshold)
    return float(
        np.mean(
            [
                ("interrogative" if len(seq) <= best[1] else "declarative") == stype
                for stype, seq in test
            ]
        )
    )


def main() -> int:  # noqa: C901 - one linear recorder, guards then measurement
    from ailm.claims import ClaimBatch, load_ledger
    from ailm.clustering import summarize_clustered_difference
    from ailm.gates import (
        STAGE5_CLAUSE_GATE_SESOI_NATS_PER_SENTENCE,
        STAGE5_DETECTABILITY_ACCURACY_FLOOR,
        STAGE5W_REGISTERED_GAIN_INTERVAL_NATS,
        stage5_widened_registered_gain_in_interval,
    )
    from ailm.levels.clause import (
        CLAUSE_TYPES,
        N_CASES,
        STAGE5_BOOTSTRAP_CONFIDENCE,
        STAGE5_BOOTSTRAP_RESAMPLES,
        STAGE5_BOOTSTRAP_SEED,
        binary_auc,
        build_joint_emission,
        build_joint_start,
        build_joint_transition,
        build_type_free_chain,
        build_type_models,
        classify_sentences,
        clustered_binary_auc_interval,
        proper_scores,
        sequence_log_evidence,
        summarize_oracle_gain,
        synthetic_control_corpus,
        type_conditional_perplexities,
    )
    from ailm.levels.widened import (
        AUX_SYMBOL,
        N_WIDENED_SYMBOLS,
        VERB_SYMBOL,
        WIDENED_DECODER_TOLERANCE,
        WIDENED_LEAK_MASS,
        load_widened_clause_artifact,
    )
    from ailm.verification.oracle import forward_filter

    artifact = load_widened_clause_artifact()
    models = build_type_models(artifact.train, n_symbols=N_WIDENED_SYMBOLS)
    comparator = build_type_free_chain(artifact.train, n_symbols=N_WIDENED_SYMBOLS)

    # ── Guard 5: the structural alphabet guard, restated over the general class
    observed = {symbol for _t, sequence in artifact.train + artifact.test for symbol in sequence}
    if not observed <= set(range(N_WIDENED_SYMBOLS)):
        raise SystemExit(
            f"widened artifact carries observations {sorted(observed - set(range(N_WIDENED_SYMBOLS)))} "
            f"outside the {N_WIDENED_SYMBOLS}-symbol alphabet; nothing may be recorded from it."
        )

    # ── Guard 1: decoder equivalence, closed form vs oracle on the joint ─────
    emission = build_joint_emission(N_WIDENED_SYMBOLS)
    joint_b = build_joint_transition(models)
    joint_d = build_joint_start(models)
    worst = 0.0
    for _sentence_type, sequence in artifact.test[:50]:
        closed, _ = sequence_log_evidence(sequence, models)
        _, oracle = forward_filter(emission, joint_b, joint_d, sequence)
        worst = max(worst, abs(closed - oracle))
    if worst > WIDENED_DECODER_TOLERANCE:
        raise SystemExit(
            f"decoder equivalence FAILED: closed-form mixture and oracle "
            f"filtering disagree by {worst:.3e}; neither may be recorded."
        )

    # ── Guard 2: NEGATIVE CONTROL — cross-type leak must break that equivalence
    #
    # Guard 1 compares two implementations of the same block-diagonal model. If
    # the comparison cannot fail, it certifies nothing. Planting mass across the
    # type blocks makes the joint chain express something the closed-form
    # mixture cannot represent — a sentence changing type mid-way — so the two
    # must now disagree.
    leaked = joint_b.copy()
    for source_type in range(len(CLAUSE_TYPES)):
        target_type = 1 - source_type
        source = slice(source_type * N_WIDENED_SYMBOLS, (source_type + 1) * N_WIDENED_SYMBOLS)
        target = slice(target_type * N_WIDENED_SYMBOLS, (target_type + 1) * N_WIDENED_SYMBOLS)
        leaked[target, source] += WIDENED_LEAK_MASS
    leaked /= leaked.sum(axis=0, keepdims=True)
    leak_divergence = 0.0
    for _sentence_type, sequence in artifact.test[:50]:
        closed, _ = sequence_log_evidence(sequence, models)
        _, leaked_evidence = forward_filter(emission, leaked, joint_d, sequence)
        leak_divergence = max(leak_divergence, abs(closed - leaked_evidence))
    if leak_divergence <= WIDENED_DECODER_TOLERANCE:
        raise SystemExit(
            f"cross-type-leak NEGATIVE CONTROL did not fire: a transition with "
            f"{WIDENED_LEAK_MASS} mass across the type blocks still agrees with the "
            f"closed-form mixture to {leak_divergence:.3e}, so the decoder-"
            f"equivalence guard above is vacuous and nothing may be recorded."
        )

    # ── Guard 3: NEGATIVE CONTROL — swapping the two types' chains must cost ──
    classification = classify_sentences(artifact.test, models)
    swapped_models = {
        "transitions": models["transitions"][::-1].copy(),
        "starts": models["starts"][::-1].copy(),
        "type_prior": models["type_prior"],
    }
    swapped = classify_sentences(artifact.test, swapped_models)
    if swapped["accuracy"] >= classification["accuracy"]:
        raise SystemExit(
            f"swapped-blocks NEGATIVE CONTROL did not fire: exchanging the two "
            f"types' fitted chains classifies at {swapped['accuracy']:.4f} "
            f"against the correct wiring's {classification['accuracy']:.4f}. If "
            f"the type assignment costs nothing, this arm is not measuring the "
            f"quantity it claims and nothing may be recorded."
        )

    # ── Guard 4: the detectability control must fire, at the WIDENED alphabet ─
    control_train, control_test = synthetic_control_corpus(n_symbols=N_WIDENED_SYMBOLS)
    control_models = build_type_models(control_train, n_symbols=N_WIDENED_SYMBOLS)
    control_ppl = type_conditional_perplexities(
        control_test,
        control_models,
        build_type_free_chain(control_train, n_symbols=N_WIDENED_SYMBOLS),
    )
    control_cls = classify_sentences(control_test, control_models)
    if not (
        control_cls["accuracy"] > STAGE5_DETECTABILITY_ACCURACY_FLOOR
        and control_ppl["mixture"] < control_ppl["type_free"]
    ):
        raise SystemExit(
            f"detectability control FAILED (acc {control_cls['accuracy']:.3f}, "
            f"mixture {control_ppl['mixture']:.3f} vs type-free "
            f"{control_ppl['type_free']:.3f}) — the instrument cannot detect a "
            f"planted effect at the widened alphabet, so any English result "
            f"here would remain instrument-ambiguous."
        )

    # ── The measurement ──────────────────────────────────────────────────────
    perplexities = type_conditional_perplexities(artifact.test, models, comparator)
    oracle_gain = summarize_oracle_gain(
        artifact.test,
        models,
        comparator,
        clusters=artifact.test_sequence_document_indices,
        n_symbols=N_WIDENED_SYMBOLS,
        symbol_kind="widened-alphabet",
    )

    # ── Guard 6: the registered prediction-1 interval (decision 112) ─────────
    #
    # Prediction 1 requires the type-oracle gain to be positive and inside the
    # interval the probe registered. Decision 112's falsifier: a point outside
    # that interval means the pre-computation was contaminated and the proposal
    # is withdrawn rather than re-fitted — the one registered prediction, with
    # the negative controls above, that carries the authority to refuse this
    # write. Predictions 2-4 are registered as directions and relations whose
    # failure is a recorded outcome, not a refusal; the standing audit guard
    # and the tests assert them instead.
    if not stage5_widened_registered_gain_in_interval(
        oracle_gain_nats_per_sentence=oracle_gain.mean_nats_per_sentence
    ):
        raise SystemExit(
            f"registered prediction 1 violated (decision 112): the type-oracle "
            f"gain {oracle_gain.mean_nats_per_sentence:.6f} nats/sentence is not "
            f"positive and inside the probe-registered interval "
            f"{STAGE5W_REGISTERED_GAIN_INTERVAL_NATS}; the pre-computation is "
            f"contaminated and the proposal is withdrawn — nothing recorded."
        )
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

    classification_clustered = summarize_clustered_difference(
        [
            100.0 * (int(model) - int(base))
            for model, base in zip(
                classification["model_hits"], classification["majority_hits"], strict=True
            )
        ],
        artifact.test_sequence_document_indices,
        confidence=STAGE5_BOOTSTRAP_CONFIDENCE,
        resamples=STAGE5_BOOTSTRAP_RESAMPLES,
        seed=STAGE5_BOOTSTRAP_SEED,
    )

    # ── Entropy of the type variable, and the share the gain realizes ────────
    #
    # The widening cannot raise this cap: it is a property of the binary type
    # variable's prior, not of the receiver. What it can move is the share.
    train_interrogatives = sum(1 for t, _ in artifact.train if t == "interrogative")
    p_int = train_interrogatives / len(artifact.train)
    type_entropy = float(-(p_int * np.log(p_int) + (1.0 - p_int) * np.log(1.0 - p_int)))

    # ── Comparator (b): sentence length only ─────────────────────────────────
    labels = np.asarray(classification["labels"])
    lengths = np.asarray([len(seq) for _t, seq in artifact.test], dtype=float)
    auc_length = binary_auc(-lengths, labels)  # shorter scores as more interrogative

    # ── Comparator (c): the single first-symbol-is-AUX indicator ─────────────
    first_is_aux = np.asarray(
        [float(seq[0] == AUX_SYMBOL) for _t, seq in artifact.test], dtype=float
    )
    auc_first_aux = binary_auc(first_is_aux, labels)

    # ── Comparator (d): the case-only arm on the SAME sentences ──────────────
    #
    # The two arms' populations are nested rather than equal, and their
    # per-position perplexities are over different position counts and different
    # alphabets, so those are not compared. What is comparable is a per-sentence
    # statistic about the same binary variable on the same sentences: both arms'
    # AUC, restricted to the held-out sentences where the case-only projection
    # is still defined. The count of that subset is recorded beside the result.
    def project_to_cases(sequence: list[int]) -> list[int]:
        return [s for s in sequence if s not in (VERB_SYMBOL, AUX_SYMBOL)]

    case_train = [
        (stype, projected)
        for stype, seq in artifact.train
        if len(projected := project_to_cases(seq)) >= 1
    ]
    paired = [
        (index, stype, projected)
        for index, (stype, seq) in enumerate(artifact.test)
        if len(projected := project_to_cases(seq)) >= 1
    ]
    case_models = build_type_models(case_train, n_symbols=N_CASES)
    case_scores: list[float] = []
    paired_labels: list[int] = []
    widened_paired_scores: list[float] = []
    for index, stype, projected in paired:
        _, posterior = sequence_log_evidence(projected, case_models)
        case_scores.append(float(posterior[1]))
        paired_labels.append(int(stype == CLAUSE_TYPES[1]))
        widened_paired_scores.append(classification["scores"][index])
    case_only_auc = binary_auc(case_scores, paired_labels)
    widened_paired_auc = binary_auc(widened_paired_scores, paired_labels)

    proper = proper_scores(classification["scores"], classification["labels"], p_int)
    by_type_test = {name: sum(1 for t, _ in artifact.test if t == name) for name in CLAUSE_TYPES}

    values: dict[str, float | int] = {
        # ── Census ───────────────────────────────────────────────────────────
        "stage5w_train_sentences": len(artifact.train),
        "stage5w_test_sentences": len(artifact.test),
        "stage5w_source_documents": len(artifact.document_ids),
        "stage5w_train_documents": len(artifact.train_split_document_indices),
        "stage5w_test_documents": len(artifact.test_split_document_indices),
        "stage5w_train_interrogatives": train_interrogatives,
        "stage5w_test_interrogatives": by_type_test["interrogative"],
        "stage5w_test_positions": perplexities["positions"],
        "stage5w_alphabet_symbols": N_WIDENED_SYMBOLS,
        "stage5w_joint_states": len(joint_d),
        "stage5w_excluded_exclamative": artifact.provenance["excluded_exclamative"],
        "stage5w_excluded_unpunctuated": artifact.provenance["excluded_unpunctuated"],
        "stage5w_excluded_too_short": artifact.provenance["excluded_too_short"],
        # ── The gate: mixture against the bar refitted on this stream ────────
        "stage5w_mixture_perplexity": round(perplexities["mixture"], 4),
        "stage5w_typefree_perplexity": round(perplexities["type_free"], 4),
        "stage5w_type_oracle_perplexity": round(perplexities["type_oracle"], 4),
        "stage5w_gate_sesoi_nats_per_sentence": STAGE5_CLAUSE_GATE_SESOI_NATS_PER_SENTENCE,
        "stage5w_privileged_label_gain_pct": round(
            100.0
            * (perplexities["type_free"] - perplexities["type_oracle"])
            / perplexities["type_free"],
            3,
        ),
        # ── The paired gain, resampled by authenticated source document ──────
        "stage5w_oracle_gain_nats_per_sentence": round(oracle_gain.mean_nats_per_sentence, 6),
        "stage5w_oracle_gain_sentence_ci_low": round(oracle_gain.sentence_ci_low, 6),
        "stage5w_oracle_gain_sentence_ci_high": round(oracle_gain.sentence_ci_high, 6),
        "stage5w_oracle_gain_nats_per_position": round(oracle_gain.pooled_nats_per_position, 6),
        "stage5w_oracle_gain_position_ci_low": round(oracle_gain.position_ci_low, 6),
        "stage5w_oracle_gain_position_ci_high": round(oracle_gain.position_ci_high, 6),
        "stage5w_oracle_gain_ci_confidence_pct": 100.0 * oracle_gain.confidence,
        "stage5w_oracle_gain_bootstrap_resamples": oracle_gain.resamples,
        "stage5w_oracle_gain_bootstrap_seed": oracle_gain.seed,
        "stage5w_oracle_gain_bootstrap_documents": oracle_gain.n_documents,
        # ── The cap the widening cannot raise, and the share it realizes ─────
        "stage5w_type_entropy_nats": round(type_entropy, 6),
        "stage5w_type_entropy_captured_pct": round(
            100.0 * oracle_gain.mean_nats_per_sentence / type_entropy, 2
        ),
        # ── Classification, paired against always-majority ───────────────────
        "stage5w_classification_pct": round(100.0 * classification["accuracy"], 2),
        "stage5w_majority_baseline_pct": round(100.0 * classification["majority_accuracy"], 2),
        "stage5w_interrogative_recall_pct": round(
            100.0 * classification["recalls"]["interrogative"], 2
        ),
        "stage5w_mcnemar_b": b,
        "stage5w_mcnemar_c": c,
        "stage5w_mcnemar_p": float(p),
        "stage5w_classification_margin_pts": round(classification_clustered.effect, 4),
        "stage5w_classification_document_ci_low": round(classification_clustered.ci_low, 4),
        "stage5w_classification_document_ci_high": round(classification_clustered.ci_high, 4),
        "stage5w_classification_document_clusters": classification_clustered.n_clusters,
        "stage5w_classification_document_p": float(f"{classification_clustered.p_value:.4g}"),
        "stage5w_classification_document_p_floor": float(f"{classification_clustered.p_floor:.4g}"),
        "stage5w_classification_document_ci_confidence_pct": (
            100.0 * classification_clustered.confidence
        ),
        "stage5w_classification_document_bootstrap_resamples": classification_clustered.resamples,
        "stage5w_classification_document_bootstrap_seed": classification_clustered.seed,
        # ── The rank statistic and the three registered comparators ──────────
        "stage5w_auc": round(classification["auc"], 4),
        "stage5w_auc_ci99_low": round(auc_summary.ci_low, 4),
        "stage5w_auc_ci99_high": round(auc_summary.ci_high, 4),
        "stage5w_auc_ci_confidence_pct": 100.0 * auc_summary.confidence,
        "stage5w_auc_bootstrap_documents": auc_summary.n_documents,
        "stage5w_auc_length_control": round(float(auc_length), 4),
        "stage5w_auc_first_aux_control": round(float(auc_first_aux), 4),
        "stage5w_length_classifier_pct": round(
            100.0 * _length_only_accuracy(artifact.train, artifact.test), 2
        ),
        # ── Comparator (d): the case-only arm, paired on the same sentences ──
        "stage5w_paired_subset_sentences": len(paired),
        "stage5w_paired_widened_auc": round(float(widened_paired_auc), 4),
        "stage5w_paired_case_only_auc": round(float(case_only_auc), 4),
        # ── Threshold-free scores ────────────────────────────────────────────
        "stage5w_average_precision": round(proper["average_precision"], 4),
        "stage5w_positive_prevalence_pct": round(100.0 * proper["prevalence"], 2),
        "stage5w_brier": round(proper["brier"], 4),
        "stage5w_brier_reference": round(proper["brier_reference"], 4),
        "stage5w_brier_skill": round(proper["brier_skill"], 4),
        # ── Controls ─────────────────────────────────────────────────────────
        "stage5w_decoder_max_divergence": worst,
        "stage5w_leak_control_divergence": round(float(leak_divergence), 6),
        "stage5w_swapped_blocks_classification_pct": round(100.0 * swapped["accuracy"], 2),
        "stage5w_control_classification_pct": round(100.0 * control_cls["accuracy"], 2),
        "stage5w_control_majority_baseline_pct": round(
            100.0
            * max(
                sum(1 for stype, _ in control_test if stype == candidate)
                for candidate in {stype for stype, _ in control_test}
            )
            / len(control_test),
            2,
        ),
        "stage5w_control_mixture_perplexity": round(control_ppl["mixture"], 4),
        "stage5w_control_typefree_perplexity": round(control_ppl["type_free"], 4),
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

    print(f"Recorded {len(values)} widened Stage 5 claims (decision 112, descriptive).")
    for label, value in sorted(values.items()):
        print(f"  {label} = {value}")
    return 0


if __name__ == "__main__":
    # The ledger is single-writer: ``ClaimsLedger`` reads the whole file at
    # construction and writes the whole file at the end, so two overlapping
    # recorders lose one of the two results, with no error on either side.
    with single_writer_lock(ledger_lock_path(_PROJECT_ROOT)):
        raise SystemExit(main())
