"""Rejection contracts of the Stage 4 case artifact.

The case artifact carries the oracle-scope sufficient counts that delimit what
the heuristic-versus-gold agreement can validate, plus the authenticated
document partition every clustered interval depends on. Each test plants one
defect in a minimal valid payload and matches the exact message the loader
raises, so a rejection for the wrong reason cannot read as a pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from ailm.corpus.ud import document_identity_sha256
from ailm.levels.case import (
    CASE_ARTIFACT_SCHEMA,
    N_CASES,
    CaseArtifact,
    build_case_agent,
    load_case_artifact,
)

_TRAIN = [("NOUN", "NOM"), ("NOUN", "ACC")] * (N_CASES * N_CASES)
_TEST = [("NOUN", "NOM"), ("NOUN", "ACC"), ("NOUN", "NOM"), ("NOUN", "ACC")]


def _fidelity(both: int) -> dict:
    return {
        "tokens_with_both": both,
        "agreements": both,
        "source_path_tokens": {"deprel": both, "adposition_token": 0},
    }


def _minimal_case_payload() -> dict:
    """A payload the loader accepts, with one document per split arm."""
    document_ids = ["fixture-train", "fixture-test"]
    total = len(_TRAIN) + len(_TEST)
    return {
        "schema": CASE_ARTIFACT_SCHEMA,
        "provenance": {
            "documents": 2,
            "train_documents": 1,
            "test_documents": 1,
            "document_identity_sha256": document_identity_sha256(document_ids),
        },
        "heuristic_vs_gold": _fidelity(len(_TRAIN)),
        "oracle_scope_audit": {
            "full_derived_tokens": total,
            "derived_path_tokens": {"morphology": total, "deprel": 0, "adposition_token": 0},
            "heuristic_vs_gold": _fidelity(len(_TRAIN)),
            "adposition_token": {
                "tokens": 0,
                "morphological_case_tokens": 0,
                "head_resolved_tokens": 0,
                "head_unresolved_tokens": 0,
                "head_also_labeled_tokens": 0,
                "head_role_agreements": 0,
                "head_role_disagreements": 0,
            },
        },
        "populations": {"gold": {"train": [list(map(list, _TRAIN))], "test": [_TEST]}},
        "documents": {
            "ids": document_ids,
            "split_indices": {"train": [0], "test": [1]},
            "sequence_document_indices": {"gold": {"train": [0], "test": [1]}},
        },
    }


def _rejection(mutate, message: str) -> None:
    payload = _minimal_case_payload()
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        CaseArtifact(payload, "gold")


def test_the_minimal_payload_is_accepted_and_counts_its_test_tokens() -> None:
    artifact = CaseArtifact(_minimal_case_payload(), "gold")
    assert artifact.n_test_tokens == len(_TEST)
    assert artifact.n_test_documents == 1


# -- oracle-scope sufficient counts ------------------------------------------


def test_rejects_a_missing_scope_audit() -> None:
    _rejection(lambda p: p.__setitem__("oracle_scope_audit", None), "lacks an oracle_scope_audit")


@pytest.mark.parametrize("value", [-1, True, 2.0, "3"])
def test_rejects_a_non_count_in_the_scope_audit(value) -> None:
    _rejection(
        lambda p: p["oracle_scope_audit"].__setitem__("full_derived_tokens", value),
        "oracle_scope_audit.full_derived_tokens must be a non-negative integer",
    )


def test_rejects_path_counts_that_do_not_name_every_path() -> None:
    _rejection(
        lambda p: p["oracle_scope_audit"]["derived_path_tokens"].pop("deprel"),
        "derived_path_tokens must name exactly",
    )


def test_rejects_path_counts_that_do_not_sum_to_the_total() -> None:
    _rejection(
        lambda p: p["oracle_scope_audit"]["derived_path_tokens"].__setitem__("deprel", 1),
        "do not sum to full_derived_tokens",
    )


def test_rejects_a_scope_fidelity_record_that_drifts_from_the_top_level() -> None:
    _rejection(
        lambda p: p["oracle_scope_audit"]["heuristic_vs_gold"].__setitem__("agreements", 1),
        "must equal the artifact's",
    )


def test_rejects_more_agreements_than_measured_tokens() -> None:
    def inflate(payload: dict) -> None:
        for record in (
            payload["heuristic_vs_gold"],
            payload["oracle_scope_audit"]["heuristic_vs_gold"],
        ):
            record["agreements"] = record["tokens_with_both"] + 1

    _rejection(inflate, "agreements exceed tokens_with_both")


def test_rejects_fidelity_paths_that_do_not_name_both_routes() -> None:
    def drop(payload: dict) -> None:
        for record in (
            payload["heuristic_vs_gold"],
            payload["oracle_scope_audit"]["heuristic_vs_gold"],
        ):
            record["source_path_tokens"] = {"deprel": record["tokens_with_both"]}

    _rejection(drop, "must name deprel and adposition_token")


def test_rejects_fidelity_paths_that_do_not_sum_to_the_overlap() -> None:
    def shift(payload: dict) -> None:
        for record in (
            payload["heuristic_vs_gold"],
            payload["oracle_scope_audit"]["heuristic_vs_gold"],
        ):
            record["source_path_tokens"]["adposition_token"] = 1

    _rejection(shift, "do not sum to tokens_with_both")


def test_rejects_a_missing_adposition_record() -> None:
    _rejection(
        lambda p: p["oracle_scope_audit"].__setitem__("adposition_token", []),
        "lacks adposition_token sufficient counts",
    )


def test_rejects_an_adposition_extent_that_disagrees_with_the_path_composition() -> None:
    _rejection(
        lambda p: p["oracle_scope_audit"]["adposition_token"].__setitem__("tokens", 1),
        "audit extent disagrees with the path composition",
    )


def test_rejects_adposition_heads_that_do_not_sum_to_the_extent() -> None:
    def move_two_tokens_onto_the_adposition_route(payload: dict) -> None:
        audit = payload["oracle_scope_audit"]
        audit["derived_path_tokens"]["morphology"] -= 2
        audit["derived_path_tokens"]["adposition_token"] = 2
        audit["adposition_token"]["tokens"] = 2
        audit["adposition_token"]["head_resolved_tokens"] = 1

    _rejection(move_two_tokens_onto_the_adposition_route, "do not sum to the path extent")


# -- populations and provenance ----------------------------------------------


def test_rejects_a_foreign_schema() -> None:
    _rejection(
        lambda p: p.__setitem__("schema", "ailm-case-counts-v1"), "Unexpected artifact schema"
    )


def test_rejects_an_unknown_population() -> None:
    with pytest.raises(ValueError, match="Unknown population 'derived'"):
        CaseArtifact(_minimal_case_payload(), "derived")


def test_rejects_an_empty_split() -> None:
    _rejection(lambda p: p["populations"]["gold"].__setitem__("test", []), "has an empty split")


def test_rejects_a_provenance_digest_that_disagrees_with_the_ids() -> None:
    _rejection(
        lambda p: p["provenance"].__setitem__("document_identity_sha256", "0" * 64),
        "identity digest disagrees",
    )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("documents", "provenance document count is inconsistent"),
        ("train_documents", "train-document count is inconsistent"),
        ("test_documents", "test-document count is inconsistent"),
    ],
)
def test_rejects_provenance_document_counts_that_disagree(field, message) -> None:
    _rejection(lambda p: p["provenance"].__setitem__(field, 5), message)


# -- documents block ---------------------------------------------------------


def test_rejects_a_documents_block_with_the_wrong_keys() -> None:
    _rejection(lambda p: p["documents"].pop("split_indices"), "documents must contain ids")


@pytest.mark.parametrize("ids", [[], ["same", "same"], ["ok", ""], ["ok", 3], ["a", "b\0c"]])
def test_rejects_malformed_document_identities(ids) -> None:
    _rejection(
        lambda p: p["documents"].__setitem__("ids", ids), "document ids must be non-empty unique"
    )


def test_rejects_split_indices_with_the_wrong_arms() -> None:
    _rejection(
        lambda p: p["documents"].__setitem__("split_indices", {"train": [0], "heldout": [1]}),
        "split_indices must name train and test",
    )


@pytest.mark.parametrize("indices", [[2], [True], [0, 0], "0"])
def test_rejects_invalid_split_indices(indices) -> None:
    _rejection(
        lambda p: p["documents"]["split_indices"].__setitem__("test", indices),
        "test document indices are invalid",
    )


def test_rejects_overlapping_split_arms() -> None:
    _rejection(
        lambda p: p["documents"]["split_indices"].__setitem__("test", [0]),
        "train and test documents overlap",
    )


def test_rejects_a_split_that_does_not_partition_the_identities() -> None:
    def add_an_unassigned_document(payload: dict) -> None:
        ids = ["fixture-train", "fixture-test", "fixture-orphan"]
        payload["documents"]["ids"] = ids
        payload["provenance"]["document_identity_sha256"] = document_identity_sha256(ids)
        payload["provenance"]["documents"] = 3

    _rejection(add_an_unassigned_document, "does not partition its identities")


def test_rejects_a_sequence_mapping_that_does_not_name_every_population() -> None:
    _rejection(
        lambda p: p["documents"].__setitem__("sequence_document_indices", {}),
        "document mapping does not match its populations",
    )


def test_rejects_a_malformed_population_mapping() -> None:
    _rejection(
        lambda p: p["documents"]["sequence_document_indices"].__setitem__("gold", {"train": [0]}),
        "gold document mapping is malformed",
    )


# -- loader and agent overrides ----------------------------------------------


def test_loader_reports_an_absent_artifact(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="generate_case_counts"):
        load_case_artifact("gold", tmp_path / "missing.json")


def test_agent_builder_rejects_a_mis_shaped_transition_override() -> None:
    artifact = CaseArtifact(_minimal_case_payload(), "gold")
    with pytest.raises(ValueError, match=f"transition must be \\({N_CASES}, {N_CASES}\\)"):
        build_case_agent(artifact, transition=np.ones((N_CASES, N_CASES + 1)))


def test_agent_builder_rejects_a_mis_shaped_start_override() -> None:
    artifact = CaseArtifact(_minimal_case_payload(), "gold")
    with pytest.raises(ValueError, match=f"start must be \\({N_CASES},\\)"):
        build_case_agent(artifact, start=np.ones((N_CASES, 1)))
