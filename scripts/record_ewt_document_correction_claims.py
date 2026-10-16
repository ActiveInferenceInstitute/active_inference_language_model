#!/usr/bin/env python3
"""Record the reproducible before/after English EWT boundary correction.

The retired Stage 4/5 generators assigned individual sentences to a stride
before recognizing that multiple EWT sentences share one ``# newdoc id``. This
recorder reconstructs that retired population from the pinned raw CoNLL-U file,
re-measures it with the current scorers, and records it beside the canonical
document-disjoint artifacts. It is the single writer for ``ewt_boundary_*``
claims; no historical result is copied as a numeric literal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ailm.ledger_lock import ledger_lock_path, single_writer_lock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_ID = "ewt_document_boundary_correction"
SOURCE = "scripts/record_ewt_document_correction_claims.py"
TEST_EVERY = 5
MIN_SEQUENCE_LEN = 2


@dataclass(slots=True)
class _RetiredCaseArtifact:
    """Minimal scorer input for the retired derived sentence-stride split."""

    train: list[list[tuple[str, str]]]
    test: list[list[tuple[str, str]]]
    upos_tags: list[str] = field(init=False)
    upos_to_idx: dict[str, int] = field(init=False)
    n_obs: int = field(init=False)

    def __post_init__(self) -> None:
        self.upos_tags = sorted({upos for seq in self.train + self.test for upos, _ in seq})
        self.upos_to_idx = {upos: index for index, upos in enumerate(self.upos_tags)}
        self.n_obs = len(self.upos_tags)


def _sentence_arm(index: int) -> str:
    """The retired split, retained only to measure the correction."""
    return "test" if index % TEST_EVERY == 0 else "train"


def _retired_populations(sentences) -> tuple[_RetiredCaseArtifact, list, dict[str, set[str]]]:
    """Reconstruct Stage 4 derived and Stage 5 typed retired populations."""
    from ailm.levels.case import CASE_TO_IDX

    case_splits: dict[str, list[list[tuple[str, str]]]] = {"train": [], "test": []}
    clause_splits: dict[str, list[tuple[str, list[int]]]] = {"train": [], "test": []}
    documents: dict[str, set[str]] = {
        "stage4_train": set(),
        "stage4_test": set(),
        "stage5_train": set(),
        "stage5_test": set(),
    }
    for index, sentence in enumerate(sentences):
        arm = _sentence_arm(index)
        case_tokens = sentence.tokens_with_case()
        case_sequence = [(token.upos, token.case_derived.name) for token in case_tokens]
        if len(case_sequence) >= MIN_SEQUENCE_LEN:
            case_splits[arm].append(case_sequence)
            documents[f"stage4_{arm}"].add(sentence.document_id)

            forms = [token.form for token in sentence.tokens]
            final = next((form for form in reversed(forms) if form in {".", "?", "!"}), None)
            sentence_type = {".": "declarative", "?": "interrogative"}.get(final)
            if sentence_type is not None:
                clause_splits[arm].append(
                    (
                        sentence_type,
                        [CASE_TO_IDX[token.case_derived] for token in case_tokens],
                    )
                )
                documents[f"stage5_{arm}"].add(sentence.document_id)
    return (
        _RetiredCaseArtifact(case_splits["train"], case_splits["test"]),
        [clause_splits["train"], clause_splits["test"]],
        documents,
    )


def _retired_case_measurement(artifact: _RetiredCaseArtifact) -> dict[str, float | int]:
    from ailm.clustering import summarize_clustered_difference
    from ailm.levels.case import (
        baseline_case_positions,
        build_case_emission,
        build_case_start,
        build_case_transition,
        decode_case_positions,
    )

    emission = build_case_emission(artifact)
    transition = build_case_transition(artifact)
    start = build_case_start(artifact)
    model = decode_case_positions(artifact, emission, transition, start)
    baseline = baseline_case_positions(artifact)
    differences = [100.0 * (int(m) - int(b)) for m, b in zip(model, baseline, strict=True)]
    sentence_clusters = [
        sentence_index for sentence_index, sequence in enumerate(artifact.test) for _ in sequence
    ]
    clustered = summarize_clustered_difference(differences, sentence_clusters)
    return {
        "test_sentences": len(artifact.test),
        "test_tokens": len(model),
        "agreement": round(100.0 * sum(model) / len(model), 2),
        "baseline": round(100.0 * sum(baseline) / len(baseline), 2),
        "margin_pts": round(sum(differences) / len(differences), 6),
        "sentence_cluster_ci_low": round(clustered.ci_low, 6),
        "sentence_cluster_ci_high": round(clustered.ci_high, 6),
    }


def _retired_clause_measurement(train, test) -> dict[str, float | int]:
    from ailm.levels.clause import (
        build_type_free_chain,
        build_type_models,
        classify_sentences,
        clustered_binary_auc_interval,
        summarize_oracle_gain,
        type_conditional_perplexities,
    )

    models = build_type_models(train)
    comparator = build_type_free_chain(train)
    perplexities = type_conditional_perplexities(test, models, comparator)
    gain = summarize_oracle_gain(test, models, comparator)
    classification = classify_sentences(test, models)
    auc = clustered_binary_auc_interval(
        classification["scores"], classification["labels"], list(range(len(test)))
    )
    return {
        "train_sentences": len(train),
        "test_sentences": len(test),
        "test_positions": perplexities["positions"],
        "mixture_perplexity": round(perplexities["mixture"], 4),
        "typefree_perplexity": round(perplexities["type_free"], 4),
        "type_reference_perplexity": round(perplexities["type_oracle"], 4),
        "gain_nats_per_sentence": round(gain.mean_nats_per_sentence, 6),
        "gain_sentence_ci_low": round(gain.sentence_ci_low, 6),
        "gain_sentence_ci_high": round(gain.sentence_ci_high, 6),
        "gain_nats_per_position": round(gain.pooled_nats_per_position, 6),
        "gain_position_ci_low": round(gain.position_ci_low, 6),
        "gain_position_ci_high": round(gain.position_ci_high, 6),
        "auc": round(classification["auc"], 4),
        "auc_ci_low": round(auc.ci_low, 4),
        "auc_ci_high": round(auc.ci_high, 4),
    }


def main() -> int:
    from ailm.claims import load_ledger
    from ailm.corpus.ud import load_ud_ewt_dev
    from ailm.levels.case import (
        baseline_case_positions,
        build_case_emission,
        build_case_start,
        build_case_transition,
        decode_case_positions,
        load_case_artifact,
    )
    from ailm.levels.clause import (
        build_type_free_chain,
        build_type_models,
        classify_sentences,
        clustered_binary_auc_interval,
        load_clause_artifact,
        summarize_oracle_gain,
        type_conditional_perplexities,
    )

    sentences = load_ud_ewt_dev()
    retired_case, retired_clause, retired_documents = _retired_populations(sentences)
    before_case = _retired_case_measurement(retired_case)
    before_clause = _retired_clause_measurement(*retired_clause)

    current_case = load_case_artifact("derived")
    case_emission = build_case_emission(current_case)
    case_transition = build_case_transition(current_case)
    case_start = build_case_start(current_case)
    case_model = decode_case_positions(current_case, case_emission, case_transition, case_start)
    case_baseline = baseline_case_positions(current_case)

    current_clause = load_clause_artifact()
    clause_models = build_type_models(current_clause.train)
    clause_comparator = build_type_free_chain(current_clause.train)
    clause_perplexities = type_conditional_perplexities(
        current_clause.test, clause_models, clause_comparator
    )
    clause_gain = summarize_oracle_gain(
        current_clause.test,
        clause_models,
        clause_comparator,
        clusters=current_clause.test_sequence_document_indices,
    )
    clause_classification = classify_sentences(current_clause.test, clause_models)
    clause_auc = clustered_binary_auc_interval(
        clause_classification["scores"],
        clause_classification["labels"],
        current_clause.test_sequence_document_indices,
    )

    old_stage4_overlap = retired_documents["stage4_train"] & retired_documents["stage4_test"]
    old_stage5_overlap = retired_documents["stage5_train"] & retired_documents["stage5_test"]
    current_stage4_overlap = set(current_case.train_document_ids) & set(
        current_case.test_document_ids
    )
    current_stage5_overlap = set(current_clause.train_document_ids) & set(
        current_clause.test_document_ids
    )
    if not old_stage4_overlap or not old_stage5_overlap:
        raise SystemExit("retired sentence split unexpectedly has no document leakage")
    if current_stage4_overlap or current_stage5_overlap:
        raise SystemExit("canonical document split leaks a source document")

    values: dict[str, float | int] = {
        "ewt_boundary_source_documents": len(current_case.document_ids),
        "ewt_boundary_stage4_retired_overlap_documents": len(old_stage4_overlap),
        "ewt_boundary_stage4_current_overlap_documents": len(current_stage4_overlap),
        "ewt_boundary_stage5_retired_overlap_documents": len(old_stage5_overlap),
        "ewt_boundary_stage5_current_overlap_documents": len(current_stage5_overlap),
    }
    values.update(
        {f"ewt_boundary_stage4_before_{key}": value for key, value in before_case.items()}
    )
    current_case_differences = [
        100.0 * (int(model) - int(base))
        for model, base in zip(case_model, case_baseline, strict=True)
    ]
    values.update(
        {
            "ewt_boundary_stage4_after_test_sentences": len(current_case.test),
            "ewt_boundary_stage4_after_test_tokens": len(case_model),
            "ewt_boundary_stage4_after_agreement": round(
                100.0 * sum(case_model) / len(case_model), 2
            ),
            "ewt_boundary_stage4_after_baseline": round(
                100.0 * sum(case_baseline) / len(case_baseline), 2
            ),
            "ewt_boundary_stage4_after_margin_pts": round(
                sum(current_case_differences) / len(current_case_differences), 6
            ),
        }
    )
    values.update(
        {f"ewt_boundary_stage5_before_{key}": value for key, value in before_clause.items()}
    )
    values.update(
        {
            "ewt_boundary_stage5_after_train_sentences": len(current_clause.train),
            "ewt_boundary_stage5_after_test_sentences": len(current_clause.test),
            "ewt_boundary_stage5_after_test_positions": clause_perplexities["positions"],
            "ewt_boundary_stage5_after_mixture_perplexity": round(
                clause_perplexities["mixture"], 4
            ),
            "ewt_boundary_stage5_after_typefree_perplexity": round(
                clause_perplexities["type_free"], 4
            ),
            "ewt_boundary_stage5_after_type_reference_perplexity": round(
                clause_perplexities["type_oracle"], 4
            ),
            "ewt_boundary_stage5_after_gain_nats_per_sentence": round(
                clause_gain.mean_nats_per_sentence, 6
            ),
            "ewt_boundary_stage5_after_gain_sentence_ci_low": round(clause_gain.sentence_ci_low, 6),
            "ewt_boundary_stage5_after_gain_sentence_ci_high": round(
                clause_gain.sentence_ci_high, 6
            ),
            "ewt_boundary_stage5_after_gain_nats_per_position": round(
                clause_gain.pooled_nats_per_position, 6
            ),
            "ewt_boundary_stage5_after_gain_position_ci_low": round(clause_gain.position_ci_low, 6),
            "ewt_boundary_stage5_after_gain_position_ci_high": round(
                clause_gain.position_ci_high, 6
            ),
            "ewt_boundary_stage5_after_auc": round(clause_classification["auc"], 4),
            "ewt_boundary_stage5_after_auc_ci_low": round(clause_auc.ci_low, 4),
            "ewt_boundary_stage5_after_auc_ci_high": round(clause_auc.ci_high, 4),
        }
    )

    ledger = load_ledger(PROJECT_ROOT)
    ledger.record_many(values, run_id=RUN_ID, source=SOURCE, status="descriptive")
    print(f"Recorded {len(values)} EWT document-boundary before/after claims.")
    for label, value in sorted(values.items()):
        print(f"  {label} = {value}")
    return 0


if __name__ == "__main__":
    with single_writer_lock(ledger_lock_path(PROJECT_ROOT)):
        raise SystemExit(main())
