"""Rejection contracts of the Stage 6 narrative artifact and the live-agent scorer.

The narrative artifact is a per-document sequence of sentence types over an
authenticated document partition. Each test plants one defect in a minimal
valid payload and matches the loader's exact message. The scorer tests build
real pymdp agents with deliberately wrong shapes or a zero predictive mass, so
that the scorer's refusals are demonstrated rather than assumed.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from pymdp.agent import Agent

from ailm.corpus.ud import document_identity_sha256
from ailm.levels.clause import CLAUSE_TYPES, N_TYPES
from ailm.levels.narrative import (
    N_PHASES,
    NARRATIVE_ARTIFACT_SCHEMA,
    NarrativeArtifact,
    build_narrative_agent,
    live_agent_type_logloss,
    load_narrative_artifact,
)

_DECLARATIVE, _INTERROGATIVE = CLAUSE_TYPES


def _minimal_narrative_payload() -> dict:
    """A payload the loader accepts, with one eligible document per arm."""
    ids = ["doc-train", "doc-test"]
    arc = [_DECLARATIVE, _INTERROGATIVE, _DECLARATIVE]
    return {
        "schema": NARRATIVE_ARTIFACT_SCHEMA,
        "provenance": {
            "document_identity_sha256": document_identity_sha256(ids),
            "documents_total": 2,
            "train_documents": 1,
            "test_documents": 1,
            "train_eligible_documents": 1,
            "test_eligible_documents": 1,
            "documents_excluded_short": 0,
            "documents_with_typed_sentences": 2,
        },
        "splits": {
            "train": [{"doc": "doc-train", "types": list(arc)}],
            "test": [{"doc": "doc-test", "types": list(arc)}],
        },
        "documents": {
            "ids": ids,
            "split_indices": {"train": [0], "test": [1]},
            "sequence_document_indices": {"train": [0], "test": [1]},
        },
    }


def _rejection(mutate, message: str) -> None:
    payload = _minimal_narrative_payload()
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        NarrativeArtifact(payload)


def test_the_minimal_payload_is_accepted() -> None:
    artifact = NarrativeArtifact(_minimal_narrative_payload())
    assert artifact.train_document_ids == ["doc-train"]
    assert artifact.test == [[0, 1, 0]]


@pytest.mark.parametrize("payload", [[], {"schema": NARRATIVE_ARTIFACT_SCHEMA}])
def test_rejects_a_payload_without_the_four_sections(payload) -> None:
    with pytest.raises(ValueError, match="must contain schema, provenance, splits, documents"):
        NarrativeArtifact(payload)


def test_rejects_a_foreign_schema() -> None:
    _rejection(lambda p: p.__setitem__("schema", "ailm-narrative-counts-v1"), "Unexpected")


def test_rejects_a_non_object_provenance() -> None:
    _rejection(lambda p: p.__setitem__("provenance", []), "provenance must be an object")


def test_rejects_splits_with_the_wrong_arms() -> None:
    _rejection(lambda p: p["splits"].pop("test"), "splits must name train and test")


def test_rejects_a_typed_document_count_outside_its_bounds() -> None:
    _rejection(
        lambda p: p["provenance"].__setitem__("documents_with_typed_sentences", 1),
        "documents_with_typed_sentences must bound",
    )


# -- split entries -----------------------------------------------------------


@pytest.mark.parametrize("entries", [[], "not-a-list"])
def test_rejects_an_empty_split(entries) -> None:
    _rejection(lambda p: p["splits"].__setitem__("test", entries), "test split is empty")


def test_rejects_a_malformed_document_entry() -> None:
    _rejection(
        lambda p: p["splits"]["test"][0].__setitem__("stray", 1),
        "test split has a malformed document entry",
    )


@pytest.mark.parametrize("doc", ["", 3, "doc\x00test"])
def test_rejects_an_invalid_entry_document_id(doc) -> None:
    _rejection(
        lambda p: p["splits"]["test"][0].__setitem__("doc", doc),
        "carries an invalid document id",
    )


def test_rejects_an_entry_without_a_type_sequence() -> None:
    _rejection(
        lambda p: p["splits"]["test"][0].__setitem__("types", "declarative"),
        "has no type sequence",
    )


def test_rejects_an_unknown_sentence_type() -> None:
    _rejection(
        lambda p: p["splits"]["test"][0]["types"].__setitem__(1, "exclamative"),
        "unknown sentence type 'exclamative'",
    )


def test_rejects_a_repeated_document_within_a_split() -> None:
    def repeat(payload: dict) -> None:
        entry = payload["splits"]["train"][0]
        payload["splits"]["train"] = [entry, dict(entry)]

    _rejection(repeat, "train split repeats a document id")


# -- documents block ---------------------------------------------------------


def test_rejects_a_documents_block_with_the_wrong_keys() -> None:
    _rejection(lambda p: p["documents"].pop("split_indices"), "documents must contain ids")


@pytest.mark.parametrize("ids", [[], ["same", "same"], ["ok", ""], ["ok", 3]])
def test_rejects_malformed_document_identities(ids) -> None:
    _rejection(
        lambda p: p["documents"].__setitem__("ids", ids), "document ids must be non-empty unique"
    )


def test_documents_validator_rejects_splits_with_the_wrong_arms() -> None:
    payload = _minimal_narrative_payload()
    with pytest.raises(ValueError, match="splits must name train and test"):
        NarrativeArtifact._validate_documents(payload["documents"], {"train": []})


def test_rejects_document_mappings_with_the_wrong_arms() -> None:
    _rejection(
        lambda p: p["documents"]["split_indices"].__setitem__("heldout", []),
        "document mappings must name train and test",
    )


@pytest.mark.parametrize("indices", [[2], [True], [0, 0], "0"])
def test_rejects_invalid_index_lists(indices) -> None:
    _rejection(
        lambda p: p["documents"]["split_indices"].__setitem__("test", indices),
        "test split indices must contain unique valid indices",
    )


def test_rejects_a_partition_that_is_not_disjoint_and_exhaustive() -> None:
    _rejection(
        lambda p: p["documents"]["split_indices"].__setitem__("test", [0]),
        "partition must be disjoint and exhaustive",
    )


def test_rejects_a_sequence_mapping_of_the_wrong_length() -> None:
    _rejection(
        lambda p: p["documents"]["sequence_document_indices"].__setitem__("test", []),
        "test sequence/document mapping length is inconsistent",
    )


def test_loader_reports_an_absent_artifact(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="generate_narrative_counts"):
        load_narrative_artifact(tmp_path / "missing.json")


# -- the live-agent scorer ---------------------------------------------------


@pytest.fixture(scope="module")
def shipped_agent():
    return build_narrative_agent(load_narrative_artifact())


def _agent(emission: np.ndarray, start: np.ndarray) -> Agent:
    """A real pymdp agent with caller-chosen emission and start shapes."""
    n_types, n_phases = emission.shape
    transition = np.eye(n_phases, dtype=np.float32)
    return Agent(
        A=[jnp.array(emission, dtype=jnp.float32)[jnp.newaxis]],
        B=[jnp.array(transition, dtype=jnp.float32)[jnp.newaxis, :, :, jnp.newaxis]],
        C=[jnp.zeros((1, n_types))],
        D=[jnp.array(start, dtype=jnp.float32)[jnp.newaxis]],
        num_controls=[1],
        batch_size=1,
    )


def _uniform_unigram() -> np.ndarray:
    return np.full(N_TYPES, 1.0 / N_TYPES)


def test_scorer_rejects_a_mis_shaped_comparator(shipped_agent) -> None:
    with pytest.raises(ValueError, match=f"unigram must have shape \\({N_TYPES},\\)"):
        live_agent_type_logloss([[0, 1]], shipped_agent, np.ones(N_TYPES + 1))


def test_scorer_rejects_a_comparator_with_zero_mass(shipped_agent) -> None:
    comparator = _uniform_unigram()
    comparator[0] = 0.0
    with pytest.raises(ValueError, match="finite positive probabilities"):
        live_agent_type_logloss([[0, 1]], shipped_agent, comparator)


def test_scorer_rejects_a_comparator_that_is_not_a_distribution(shipped_agent) -> None:
    with pytest.raises(ValueError, match="not 1"):
        live_agent_type_logloss([[0, 1]], shipped_agent, np.full(N_TYPES, 0.4))


def test_scorer_rejects_an_agent_with_an_extra_phase() -> None:
    emission = np.full((N_TYPES, N_PHASES + 1), 1.0 / N_TYPES)
    start = np.zeros(N_PHASES + 1)
    start[0] = 1.0
    with pytest.raises(ValueError, match="agent emission must have shape"):
        live_agent_type_logloss([[0, 1]], _agent(emission, start), _uniform_unigram())


def test_scorer_rejects_a_zero_predictive_mass() -> None:
    """A type the agent declares impossible must fail loudly, not score infinity."""
    emission = np.zeros((N_TYPES, N_PHASES))
    emission[1, :] = 1.0
    start = np.zeros(N_PHASES)
    start[0] = 1.0
    with pytest.raises(ValueError, match="invalid predictive mass"):
        live_agent_type_logloss([[0]], _agent(emission, start), _uniform_unigram())
