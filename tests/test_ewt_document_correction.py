"""End-to-end controls for the English EWT document-boundary correction."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from ailm.corpus.ud import load_ud_ewt_dev
from ailm.levels.case import load_case_artifact
from ailm.levels.clause import load_clause_artifact

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _module():
    return importlib.import_module("scripts.record_ewt_document_correction_claims")


@pytest.fixture(scope="module")
def recorded_claims() -> dict[str, dict]:
    return json.loads((PROJECT_ROOT / "data" / "claims.json").read_text(encoding="utf-8"))["claims"]


def test_retired_measurements_are_recomputed_from_the_pinned_raw_corpus(recorded_claims):
    """The before table is executable evidence, not copied historical prose."""
    module = _module()
    retired_case, retired_clause, documents = module._retired_populations(load_ud_ewt_dev())
    stage4 = module._retired_case_measurement(retired_case)
    stage5 = module._retired_clause_measurement(*retired_clause)

    for key, value in stage4.items():
        assert recorded_claims[f"ewt_boundary_stage4_before_{key}"]["value"] == pytest.approx(
            value, abs=1e-12
        )
    for key, value in stage5.items():
        assert recorded_claims[f"ewt_boundary_stage5_before_{key}"]["value"] == pytest.approx(
            value, abs=1e-12
        )

    stage4_overlap = documents["stage4_train"] & documents["stage4_test"]
    stage5_overlap = documents["stage5_train"] & documents["stage5_test"]
    assert (
        len(stage4_overlap)
        == recorded_claims["ewt_boundary_stage4_retired_overlap_documents"]["value"]
    )
    assert (
        len(stage5_overlap)
        == recorded_claims["ewt_boundary_stage5_retired_overlap_documents"]["value"]
    )
    assert stage4_overlap and stage5_overlap, (
        "negative control died: the retired sentence split no longer leaks a real document"
    )


def test_current_artifacts_are_document_disjoint_and_bind_the_after_claims(recorded_claims):
    case = load_case_artifact("derived")
    clause = load_clause_artifact()

    assert set(case.train_document_ids).isdisjoint(case.test_document_ids)
    assert set(clause.train_document_ids).isdisjoint(clause.test_document_ids)
    assert recorded_claims["ewt_boundary_stage4_current_overlap_documents"]["value"] == 0
    assert recorded_claims["ewt_boundary_stage5_current_overlap_documents"]["value"] == 0

    paired_labels = {
        "ewt_boundary_stage4_after_test_sentences": "stage4_derived_test_sentences",
        "ewt_boundary_stage4_after_test_tokens": "stage4_derived_test_tokens",
        "ewt_boundary_stage4_after_agreement": "stage4_derived_agreement",
        "ewt_boundary_stage4_after_baseline": "stage4_derived_baseline",
        "ewt_boundary_stage5_after_train_sentences": "stage5_train_sentences",
        "ewt_boundary_stage5_after_test_sentences": "stage5_test_sentences",
        "ewt_boundary_stage5_after_test_positions": "stage5_test_positions",
        "ewt_boundary_stage5_after_mixture_perplexity": "stage5_mixture_perplexity",
        "ewt_boundary_stage5_after_typefree_perplexity": "stage5_typefree_perplexity",
        "ewt_boundary_stage5_after_type_reference_perplexity": ("stage5_type_oracle_perplexity"),
        "ewt_boundary_stage5_after_gain_nats_per_sentence": (
            "stage5_oracle_gain_nats_per_sentence"
        ),
        "ewt_boundary_stage5_after_auc": "stage5_auc",
    }
    for correction, canonical in paired_labels.items():
        assert recorded_claims[correction]["value"] == recorded_claims[canonical]["value"]


def test_correction_changes_stage5_point_direction_without_resolving_it(recorded_claims):
    def value(label: str):
        return recorded_claims[label]["value"]

    assert value("ewt_boundary_stage5_before_gain_nats_per_sentence") > 0
    assert value("ewt_boundary_stage5_after_gain_nats_per_sentence") < 0
    assert (
        value("ewt_boundary_stage5_before_gain_sentence_ci_low")
        < 0
        < value("ewt_boundary_stage5_before_gain_sentence_ci_high")
    )
    assert (
        value("ewt_boundary_stage5_after_gain_sentence_ci_low")
        < 0
        < value("ewt_boundary_stage5_after_gain_sentence_ci_high")
    )
