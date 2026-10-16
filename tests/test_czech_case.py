"""The Czech case arm — Level 4 against real morphology (decision 77).

What is under test
------------------
The one thing English EWT could never provide: the sequence mechanism scored
against morphological ground truth. The vendored artifact carries UPOS and
``feats[Case]`` only; the model, comparator and decode are the English arm's
builders reused unmodified through a duck-typed artifact. The prospectively specified
question — does the margin survive when the labels are real case — resolves on
these numbers, so the tests re-derive them rather than trusting the ledger.

Claims discipline
-----------------
This file **verifies**; ``scripts/record_czech_case_claims.py`` writes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binomtest

from ailm.cases import CaseRole
from ailm.claims import load_ledger
from ailm.levels.case_czech import (
    CZECH_ARTIFACT_SCHEMA,
    CZECH_MIN_SEQUENCE_LEN,
    CZECH_NOMINAL_UPOS,
    CZECH_STUDY_DESIGN_PROVENANCE,
    CzechCaseArtifact,
    baseline_case_positions,
    build_case_emission,
    build_case_start,
    build_case_transition,
    czech_case_accuracy,
    decode_case_positions,
    load_czech_case_artifact,
    validate_study_design_provenance,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = PROJECT_ROOT / "vendor" / "ud_czech_case_counts.json"

#: The UPOS categories the artifact may carry: nominal, case-inflecting ones.
#: Stated as the admissible class rather than as "not ADP", so any future
#: non-nominal leak — ADP, VERB, PART — fails this guard, not only the one
#: instance decision 77 argued about.
NOMINAL_UPOS = CZECH_NOMINAL_UPOS

_SKIP_NO_ARTIFACT = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason="vendor/ud_czech_case_counts.json absent; run scripts/generate_czech_case_counts.py",
)

pytestmark = _SKIP_NO_ARTIFACT


@pytest.fixture(scope="module")
def artifact() -> CzechCaseArtifact:
    return load_czech_case_artifact()


@pytest.fixture(scope="module")
def measured(artifact) -> dict:
    emission = build_case_emission(artifact)
    transition = build_case_transition(artifact)
    start = build_case_start(artifact)
    model = decode_case_positions(artifact, emission, transition, start)
    baseline = baseline_case_positions(artifact)
    b = sum(1 for m, s in zip(model, baseline, strict=True) if m and not s)
    c = sum(1 for m, s in zip(model, baseline, strict=True) if s and not m)
    return {
        "emission": emission,
        "transition": transition,
        "start": start,
        "model_hits": model,
        "baseline_hits": baseline,
        "model": 100.0 * czech_case_accuracy(model),
        "baseline": 100.0 * czech_case_accuracy(baseline),
        "b": b,
        "c": c,
        "p": binomtest(b, b + c, 0.5).pvalue if b + c else 1.0,
    }


@pytest.fixture(scope="module")
def clustered(measured, artifact):
    """The publication-primary outer document-clustered Czech comparison."""
    from ailm.clustering import summarize_clustered_difference

    differences = [
        100.0 * (int(model) - int(baseline))
        for model, baseline in zip(measured["model_hits"], measured["baseline_hits"], strict=True)
    ]
    document_ids = [
        artifact.test_sequence_document_indices[sequence_index]
        for sequence_index, sequence in enumerate(artifact.test)
        for _ in sequence
    ]
    return summarize_clustered_difference(differences, document_ids)


def _artifact_payload() -> dict:
    """A detached real payload for constructor and load-path controls."""
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _write_payload(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "planted-czech-artifact.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ══ THE ARTIFACT — what it carries, and what it must not ═════════════════════


class TestArtifact:
    def test_schema_is_pinned(self, artifact):
        assert artifact.provenance["label_source_morphology_pct"] == 100.0
        assert artifact.provenance["study_design"] == dict(CZECH_STUDY_DESIGN_PROVENANCE)

    def test_wrong_schema_is_refused(self):
        payload = _artifact_payload()
        payload["schema"] = "something-else"
        with pytest.raises(ValueError, match="schema"):
            CzechCaseArtifact(payload)

    def test_non_morphological_label_source_is_refused(self):
        """The load-time face of the recorder's coverage guard (decision 77)."""
        payload = _artifact_payload()
        payload["provenance"]["label_source_morphology_pct"] = 96.84  # the English rate
        with pytest.raises(ValueError, match="morphology"):
            CzechCaseArtifact(payload)

    def test_missing_raw_source_identity_is_refused(self):
        """The loader, not only the generator, enforces source provenance."""
        payload = _artifact_payload()
        payload["provenance"].pop("raw_source")
        with pytest.raises(ValueError, match="raw_source"):
            CzechCaseArtifact(payload)

    def test_moving_raw_source_url_is_refused(self):
        """NEGATIVE CONTROL: valid populations cannot launder branch provenance."""
        payload = _artifact_payload()
        payload["provenance"]["raw_source"]["shards"]["test"]["source_url"] = (
            "https://raw.githubusercontent.com/UniversalDependencies/"
            "UD_Czech-PDTC/master/cs_pdtc-ud-dev.conllu"
        )
        with pytest.raises(ValueError, match="disagrees with the pin"):
            CzechCaseArtifact(payload)

    def test_permuted_case_order_is_refused(self):
        """A reordered case alphabet would silently rename every role."""
        payload = _artifact_payload()
        values = payload["ud_case_values"]
        payload["ud_case_values"] = values[1:] + values[:1]
        with pytest.raises(ValueError, match="case order"):
            CzechCaseArtifact(payload)

    def test_observation_alphabet_is_nominal_only(self, artifact):
        """The decision 77 exclusion, enforced as a class rather than a name.

        ADP carries ``Case=`` in Czech PDT, but there the feature records
        government, not inflection — the analogue of the English adposition
        path whose margin reversed (decision 63). Any non-nominal tag in the
        alphabet re-imports that label source.
        """
        assert set(artifact.upos_tags) <= NOMINAL_UPOS
        assert artifact.n_obs > 1, "the English gold degeneracy this arm exists to escape"

    def test_seven_of_eight_cases_present(self, artifact):
        """Czech carries the CEREBRUM inventory minus the ablative."""
        assert artifact.cases_present == {c.name for c in CaseRole} - {"ABL"}

    def test_case_names_decode_to_roles(self, artifact):
        for split in (artifact.train, artifact.test):
            for sequence in split[:200]:
                for _upos, case in sequence:
                    assert case in CaseRole.__members__

    def test_the_artifact_carries_only_encoded_upos_case_pairs(self, artifact):
        """The artifact cannot leak fields it does not carry — check it doesn't.

        The English trap needed a guard on what the observation may contain;
        here the stronger property holds structurally: every token is a single
        integer decoding to (UPOS index, case index), so deprel, lemma and form
        never entered the file at all. A regeneration that widened the token
        representation would be the first step of the same circularity, so the
        payload's shape is the guard.
        """
        payload = _artifact_payload()
        assert set(payload) == {
            "schema",
            "provenance",
            "upos_tags",
            "ud_case_values",
            "populations",
            "documents",
        }
        assert set(payload["populations"]) == {"train", "test"}
        limit = artifact.n_obs * 8
        for sequences in payload["populations"].values():
            for sequence in sequences:
                assert all(isinstance(code, int) and 0 <= code < limit for code in sequence)

    def test_authenticated_document_hierarchy_is_complete(self, artifact):
        """Every retained sequence maps to one contributing source document."""
        assert len(artifact.test_sequence_document_indices) == len(artifact.test) == 20_349
        assert artifact.n_test_documents == len(artifact.test_document_ids) == 1_005
        assert set(artifact.test_sequence_document_indices) == set(range(1_005))
        assert artifact.provenance["splits"]["test"]["documents"] == 1_006
        assert artifact.provenance["splits"]["test"]["retained_documents"] == 1_005
        document_token_counts = [0] * artifact.n_test_documents
        for sequence, document_index in zip(
            artifact.test, artifact.test_sequence_document_indices, strict=True
        ):
            document_token_counts[document_index] += len(sequence)
        assert max(document_token_counts) == 2_123

    def test_test_shard_is_the_specified_size(self, artifact):
        """169,331 held-out tokens — 82x the 420-pair arithmetic's requirement."""
        from ailm.statistics import minimum_discordant_pairs_for_equivalence

        assert artifact.n_test_tokens >= minimum_discordant_pairs_for_equivalence() / 0.204

    @pytest.mark.parametrize("code", (-1, True, 48, 1.5, "1"))
    def test_invalid_encoded_tokens_are_refused_through_the_load_path(self, tmp_path, code):
        """NEGATIVE CONTROL: Python indexing must not launder malformed codes.

        ``bool`` is an ``int`` subclass and negative indices are legal, so both
        would silently decode without an exact-type/range check.  The load path,
        not only a direct constructor call, has to enforce the contract.
        """
        payload = _artifact_payload()
        payload["populations"]["test"][0][0] = code
        with pytest.raises(ValueError, match="invalid encoded value"):
            load_czech_case_artifact(_write_payload(tmp_path, payload))

    @pytest.mark.parametrize("mutation", ("duplicate", "unsorted", "non_nominal"))
    def test_upos_alphabet_must_be_sorted_unique_and_nominal(self, mutation):
        payload = _artifact_payload()
        if mutation == "duplicate":
            payload["upos_tags"][1] = payload["upos_tags"][0]
        elif mutation == "unsorted":
            payload["upos_tags"][0], payload["upos_tags"][1] = (
                payload["upos_tags"][1],
                payload["upos_tags"][0],
            )
        else:
            payload["upos_tags"][-1] = "VERB"
        with pytest.raises(ValueError, match="upos_tags"):
            CzechCaseArtifact(payload)

    @pytest.mark.parametrize("mutation", ("missing", "extra", "not_mapping", "short"))
    def test_population_split_surface_and_minimum_length_fail_closed(self, mutation):
        payload = _artifact_payload()
        if mutation == "missing":
            payload["populations"].pop("test")
        elif mutation == "extra":
            payload["populations"]["validation"] = []
        elif mutation == "not_mapping":
            payload["populations"] = []
        else:
            payload["populations"]["test"][0] = payload["populations"]["test"][0][
                : CZECH_MIN_SEQUENCE_LEN - 1
            ]
        with pytest.raises(ValueError, match="populations|sequence"):
            CzechCaseArtifact(payload)

    def test_retained_split_counts_must_match_the_encoded_population(self):
        payload = _artifact_payload()
        payload["provenance"]["splits"]["test"]["retained_tokens"] += 1
        with pytest.raises(ValueError, match="retained counts"):
            CzechCaseArtifact(payload)

    def test_payload_and_provenance_surfaces_are_exact(self):
        top = _artifact_payload()
        top["surprise"] = "accepted fields become unreviewed model inputs"
        with pytest.raises(ValueError, match="contain exactly"):
            CzechCaseArtifact(top)

        provenance = _artifact_payload()
        provenance["provenance"]["surprise"] = "unreviewed"
        with pytest.raises(ValueError, match="stale or widened"):
            CzechCaseArtifact(provenance)

    @pytest.mark.parametrize(
        "field",
        (
            "corpus",
            "license",
            "split_scope",
            "observation",
            "label_source",
            "adp_excluded_reason",
            "encoding",
            "document_identity_source",
            "cluster_hierarchy",
        ),
    )
    def test_static_provenance_values_are_exact(self, field):
        """A plausible alternate description is still a different artifact."""
        payload = _artifact_payload()
        payload["provenance"][field] = f"{payload['provenance'][field]} (drifted)"
        with pytest.raises(ValueError, match="static provenance|encoding contract"):
            CzechCaseArtifact(payload)

    @pytest.mark.parametrize("bad_value", (True, 2.0, "2", None))
    def test_minimum_sequence_length_requires_the_exact_integer_contract(self, bad_value):
        payload = _artifact_payload()
        payload["provenance"]["min_sequence_len"] = bad_value
        with pytest.raises(ValueError, match="min_sequence_len"):
            CzechCaseArtifact(payload)

    @pytest.mark.parametrize("mutation", ("missing", "extra", "not_mapping"))
    def test_split_metadata_surface_is_exact(self, mutation):
        payload = _artifact_payload()
        metadata = payload["provenance"]["splits"]["test"]
        if mutation == "missing":
            metadata.pop("tokens")
        elif mutation == "extra":
            metadata["unreviewed"] = 1
        else:
            payload["provenance"]["splits"]["test"] = []
        with pytest.raises(ValueError, match=r"provenance\.splits\.test"):
            CzechCaseArtifact(payload)

    @pytest.mark.parametrize("bad_count", (True, -1, 1.5, "1"))
    def test_split_metadata_counts_require_nonnegative_exact_integers(self, bad_count):
        payload = _artifact_payload()
        payload["provenance"]["splits"]["test"]["tokens"] = bad_count
        with pytest.raises(ValueError, match="non-negative integers"):
            CzechCaseArtifact(payload)

    def test_split_metadata_source_file_is_exact(self):
        payload = _artifact_payload()
        payload["provenance"]["splits"]["test"]["file"] = "cs_pdtc-ud-test.conllu"
        with pytest.raises(ValueError, match="wrong source file"):
            CzechCaseArtifact(payload)

    def test_logically_impossible_split_counts_are_refused(self):
        payload = _artifact_payload()
        payload["provenance"]["splits"]["test"]["sentences"] = 0
        with pytest.raises(ValueError, match="internally impossible"):
            CzechCaseArtifact(payload)

    @pytest.mark.parametrize("mutation", ("missing", "extra", "not_mapping"))
    def test_document_mapping_surface_is_exact(self, mutation):
        payload = _artifact_payload()
        mapping = payload["documents"]["test"]
        if mutation == "missing":
            mapping.pop("ids")
        elif mutation == "extra":
            mapping["sentence_ids"] = []
        else:
            payload["documents"]["test"] = []
        with pytest.raises(ValueError, match=r"documents\.test"):
            CzechCaseArtifact(payload)

    @pytest.mark.parametrize("bad_index", (-1, True, 1.5, "1", 1_005))
    def test_document_indices_require_exact_in_range_integers(self, bad_index):
        payload = _artifact_payload()
        payload["documents"]["test"]["sequence_document_indices"][0] = bad_index
        with pytest.raises(ValueError, match="one in-range exact integer"):
            CzechCaseArtifact(payload)

    def test_duplicate_document_identity_is_refused(self):
        payload = _artifact_payload()
        payload["documents"]["test"]["ids"][1] = payload["documents"]["test"]["ids"][0]
        with pytest.raises(ValueError, match="unique"):
            CzechCaseArtifact(payload)

    def test_per_sequence_ids_cannot_be_promoted_to_outer_primary(self):
        """LIVE NEGATIVE CONTROL: the former one-level-too-low unit is refused."""
        payload = _artifact_payload()
        n_sequences = len(payload["populations"]["test"])
        payload["documents"]["test"] = {
            "ids": [f"planted-sentence-{index}" for index in range(n_sequences)],
            "sequence_document_indices": list(range(n_sequences)),
        }
        split = payload["provenance"]["splits"]["test"]
        split["documents"] = n_sequences
        split["retained_documents"] = n_sequences
        with pytest.raises(ValueError, match="no outer document cluster"):
            CzechCaseArtifact(payload)


class TestStudyDesignProvenance:
    def test_current_protocol_is_not_independently_preregistered(self, artifact):
        assert artifact.provenance["study_design"] == {
            "protocol_status": "prospective-development-session-specification",
            "independently_timestamped": False,
            "immutable_prior_artifact": None,
        }

    def test_preregistered_terminology_without_a_prior_artifact_is_refused(self):
        """POSITIVE CONTROL for the terminology guard requested by audit."""
        planted = {
            "protocol_status": "preregistered",
            "independently_timestamped": True,
            "immutable_prior_artifact": None,
        }
        with pytest.raises(ValueError, match="preregistered terminology requires"):
            validate_study_design_provenance(planted)

    def test_content_addressed_prior_artifact_earns_preregistered_terminology(self):
        """NEGATIVE CONTROL: the guard accepts the evidence class it names."""
        provenance = {
            "protocol_status": "preregistered",
            "independently_timestamped": True,
            "immutable_prior_artifact": {
                "path": "docs/protocols/czech-case.md",
                "git_commit": "a" * 40,
                "sha256": "b" * 64,
                "committed_at": "2026-08-20T12:00:00+00:00",
            },
        }
        assert validate_study_design_provenance(provenance) == provenance

    def test_czech_artifact_cannot_claim_a_different_design_status(self):
        payload = _artifact_payload()
        payload["provenance"]["study_design"] = {
            "protocol_status": "preregistered",
            "independently_timestamped": True,
            "immutable_prior_artifact": {
                "path": "docs/protocols/czech-case.md",
                "git_commit": "a" * 40,
                "sha256": "b" * 64,
                "committed_at": "2026-08-20T12:00:00+00:00",
            },
        }
        with pytest.raises(ValueError, match="does not match decision 77"):
            CzechCaseArtifact(payload)


# ══ THE MODEL — same builders, real labels ═══════════════════════════════════


class TestModel:
    def test_emission_is_column_stochastic(self, measured, artifact):
        emission = measured["emission"]
        assert emission.shape == (artifact.n_obs, 8)
        np.testing.assert_allclose(emission.sum(axis=0), 1.0)

    def test_transition_is_column_stochastic(self, measured):
        transition = measured["transition"]
        assert transition.shape == (8, 8)
        np.testing.assert_allclose(transition.sum(axis=0), 1.0)

    def test_start_is_a_distribution(self, measured):
        start = measured["start"]
        assert start.shape == (8,)
        assert abs(start.sum() - 1.0) < 1e-12

    def test_positions_align_with_the_held_out_tokens(self, measured, artifact):
        assert (
            len(measured["model_hits"]) == len(measured["baseline_hits"]) == artifact.n_test_tokens
        )

    def test_accuracy_over_no_positions_is_refused(self):
        with pytest.raises(ValueError, match="undefined"):
            czech_case_accuracy([])


# ══ THE REGISTERED QUESTION — and its paired defense ═════════════════════════


class TestSpecifiedQuestion:
    def test_the_margin_survives_on_real_morphology(self, measured):
        """Decision 77's specified question, answered in the surviving direction.

        POSITIVE CONTROL for the level's mechanism: on labels that are real
        case morphology — not the output of a heuristic the model could be
        agreeing with — the sequence model beats the context-free comparator.
        """
        assert measured["model"] > measured["baseline"]
        assert measured["b"] > measured["c"]
        assert measured["p"] < 1e-6

    def test_document_clustered_primary_bounds_the_effect_away_from_zero(
        self, measured, clustered, artifact
    ):
        """Real-data primary: effect identity plus cluster-aware uncertainty."""
        assert clustered.n_observations == artifact.n_test_tokens
        assert clustered.n_clusters == artifact.n_test_documents
        assert clustered.largest_cluster == 2_123
        assert clustered.effect == pytest.approx(
            measured["model"] - measured["baseline"], abs=1e-12
        )
        assert clustered.ci_low > 0.0
        assert clustered.p_at_resolution_floor

    def test_collapsing_all_documents_to_one_cluster_is_refused(self, measured):
        """NEGATIVE CONTROL: a manufactured one-cluster primary has no variance."""
        from ailm.clustering import summarize_clustered_difference

        differences = [
            float(int(model) - int(baseline))
            for model, baseline in zip(
                measured["model_hits"], measured["baseline_hits"], strict=True
            )
        ]
        with pytest.raises(ValueError, match="at least two independent clusters"):
            summarize_clustered_difference(differences, ["one"] * len(differences))

    def test_the_margin_beats_the_whole_context_free_class(self, measured, artifact):
        """Stronger than the fitted bar: above the fit-on-test ceiling.

        A context-free predictor fitted on the test labels themselves is the
        best any member of the comparator class can do. The model sitting above
        it means the sequence mechanism is not reproducing a better lookup —
        no lookup on this observation reaches it.
        """
        from collections import Counter, defaultdict

        per_upos: dict[str, Counter] = defaultdict(Counter)
        total = 0
        for sequence in artifact.test:
            for upos, case in sequence:
                per_upos[upos][case] += 1
                total += 1
        ceiling = 100.0 * sum(cnt.most_common(1)[0][1] for cnt in per_upos.values()) / total
        assert measured["model"] > ceiling

    def test_margin_reproduces_from_the_discordant_counts(self, measured, artifact):
        """(b - c) / n IS the margin; a mismatch means misaligned pairing."""
        margin_pts = measured["model"] - measured["baseline"]
        assert margin_pts == pytest.approx(
            100.0 * (measured["b"] - measured["c"]) / artifact.n_test_tokens, abs=1e-9
        )

    def test_self_comparison_shows_no_effect(self, measured):
        """NEGATIVE CONTROL: the paired test must be silent on identical vectors."""
        hits = measured["model_hits"]
        b = sum(1 for m, s in zip(hits, hits, strict=True) if m and not s)
        c = sum(1 for m, s in zip(hits, hits, strict=True) if s and not m)
        assert (b, c) == (0, 0)

    def test_a_deranged_emission_fails(self, measured, artifact):
        """NEGATIVE CONTROL: cyclically mis-wiring P(UPOS | case) must hurt.

        The permutation keeps every column a distribution, so nothing about the
        deranged model looks broken — it is simply wrong about which case emits
        which tags. It must fall below the intact model AND below the
        context-free baseline; measured at 25.49% against 39.21%/32.36% when
        this control was built, so the assertion has a wide moat.
        """
        deranged = measured["emission"][:, [1, 2, 3, 4, 5, 6, 7, 0]]
        hits = decode_case_positions(artifact, deranged, measured["transition"], measured["start"])
        accuracy = 100.0 * czech_case_accuracy(hits)
        assert accuracy < measured["model"] - 5.0
        assert accuracy < measured["baseline"]

    def test_a_deranged_transition_fails(self, measured, artifact):
        """NEGATIVE CONTROL: the margin must come from the real case dynamics."""
        perm = [1, 2, 3, 4, 5, 6, 7, 0]
        deranged = measured["transition"][:, perm][perm, :]
        hits = decode_case_positions(artifact, measured["emission"], deranged, measured["start"])
        accuracy = 100.0 * czech_case_accuracy(hits)
        assert accuracy < measured["model"] - 5.0
        assert accuracy < measured["baseline"]


# ══ CLAIMS LEDGER — verify, never write ══════════════════════════════════════


def test_czech_claims_verified(measured, artifact):
    ledger = load_ledger(PROJECT_ROOT)
    expected = {
        "stage4_czech_train_documents": len(artifact.train_document_ids),
        "stage4_czech_test_tokens": artifact.n_test_tokens,
        "stage4_czech_test_documents": artifact.n_test_documents,
        "stage4_czech_upos_alphabet_size": artifact.n_obs,
        "stage4_czech_cases_present": len(artifact.cases_present),
        "stage4_czech_label_source_morphology_pct": 100.0,
        "stage4_czech_accuracy": round(measured["model"], 2),
        "stage4_czech_baseline": round(measured["baseline"], 2),
        "stage4_czech_margin_pts": round(measured["model"] - measured["baseline"], 2),
        "stage4_czech_mcnemar_b": measured["b"],
        "stage4_czech_mcnemar_c": measured["c"],
        "stage4_czech_discordant_n": measured["b"] + measured["c"],
    }
    for label, value in expected.items():
        entry = ledger.get(label)
        assert entry is not None, (
            f"{label} is absent. Run scripts/record_czech_case_claims.py — "
            f"this test may not write it."
        )
        assert entry["source"].startswith("scripts/"), (
            f"{label} was last written by {entry['source']!r}. Recorders write; tests verify."
        )
        assert entry["value"] == pytest.approx(value, rel=1e-4), (
            f"Ledger {label}={entry['value']!r} != fresh measurement {value!r}."
        )


def test_czech_cluster_primary_claims_verified(clustered, artifact):
    """Freshly re-derive every headline claim from the real held-out sequences."""
    from ailm.clustering import DEFAULT_RESAMPLES, DEFAULT_SEED

    ledger = load_ledger(PROJECT_ROOT)
    expected = {
        "cluster_stage4_czech_observations": artifact.n_test_tokens,
        "cluster_stage4_czech_clusters": artifact.n_test_documents,
        "cluster_stage4_czech_largest_cluster": 2_123,
        "cluster_stage4_czech_effect": round(clustered.effect, 6),
        "cluster_stage4_czech_ci_low": round(clustered.ci_low, 6),
        "cluster_stage4_czech_ci_high": round(clustered.ci_high, 6),
        "cluster_stage4_czech_p": float(f"{clustered.p_value:.4g}"),
        "cluster_stage4_czech_p_floor": float(f"{clustered.p_floor:.4g}"),
        "cluster_stage4_czech_resamples": DEFAULT_RESAMPLES,
        "cluster_stage4_czech_seed": DEFAULT_SEED,
    }
    for label, value in expected.items():
        entry = ledger.get(label)
        assert entry is not None, f"{label} is absent; run scripts/record_cluster_claims.py"
        assert entry["source"] == "scripts/record_cluster_claims.py"
        assert entry["value"] == pytest.approx(value, rel=1e-9)
    floor_entry = ledger.get("cluster_stage4_czech_p_at_resolution_floor")
    assert floor_entry is not None
    assert floor_entry["value"] is True
    unit_entry = ledger.get("cluster_stage4_czech_cluster_unit")
    assert unit_entry is not None
    assert unit_entry["value"] == "document"


def test_the_exploratory_p_stays_out_of_the_frozen_family():
    """Decision 77 specifies this arm as exploratory; the family is closed."""
    from ailm.statistics import SIGNIFICANCE_FAMILY

    assert "stage4_czech_mcnemar_p" not in SIGNIFICANCE_FAMILY


def test_copy_of_payload_is_not_shared():
    """Guard the guard: the refusal tests above mutate copies, not the file."""
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(payload)
    mutated["schema"] = "something-else"
    assert payload["schema"] == CZECH_ARTIFACT_SCHEMA
