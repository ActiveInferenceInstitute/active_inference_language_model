"""Stage 6: an observed-position association and a distinct live agent.

Decision 25 predicted, before the held-out measurement: document position
informs sentence type out of sample (P1); the top-down channel into the
clause level shifts type priors with an exact α = 0 anchor and a
distinguishable scramble (P2); and proposed a sensitivity interpretation (P3).
The registered P1 estimand is the observation-weighted mean sentence log-loss
difference, with the source document as the resampling and sign-flip unit. Its
observed-position point estimate is positive, but the document-clustered interval spans
zero, so it is imprecise and non-confirmatory; power was not separately evaluated.
The sentence sign test is a
secondary estimand and cannot license the mean claim: a worse scrambled model
passes it. This file re-measures both the observed-position association and the
separate live-pymdp filter, enforces P2's three guards, asserts the
structural upper-triangle zero the first draft mis-stated (entry 26's caught
defect), and binds the recorded ``stage6_*`` labels to fresh executions.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binomtest

from ailm.claims import load_ledger
from ailm.clustering import summarize_clustered_difference
from ailm.control import ControlHarness
from ailm.levels.clause import (
    build_joint_start,
    build_type_models,
    load_clause_artifact,
)
from ailm.levels.narrative import (
    N_PHASES,
    build_narrative_agent,
    build_phase_models,
    build_phase_transition,
    clause_prior_with_narrative,
    heldout_type_logloss,
    live_agent_type_logloss,
    load_narrative_artifact,
    tercile,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _stage6_power_wording_defects(surfaces: dict[Path, str]) -> list[str]:
    """Reject claims that infer an unmeasured power result from interval width."""

    retired = (
        "under" + "powered rather than confirmed",
        "under" + "powered at document level",
    )
    return [str(path) for path, text in surfaces.items() if any(term in text for term in retired)]


def test_stage6_current_surfaces_do_not_claim_an_unmeasured_power_result() -> None:
    paths = (
        Path(__file__),
        PROJECT_ROOT / "docs" / "manuscript" / "03_methods.md",
        PROJECT_ROOT / "docs" / "manuscript" / "04_results.md",
        PROJECT_ROOT / "docs" / "decisions.md",
    )
    surfaces = {path: path.read_text(encoding="utf-8") for path in paths}
    assert _stage6_power_wording_defects(surfaces) == []

    # NEGATIVE CONTROL: restore the retired inference on an isolated surface.
    planted = {Path("planted.md"): "the interval is under" + "powered rather than confirmed"}
    assert _stage6_power_wording_defects(planted) == ["planted.md"]


@pytest.fixture(scope="module")
def artifact():
    return load_narrative_artifact()


@pytest.fixture(scope="module")
def models(artifact):
    return build_phase_models(artifact.train)


@pytest.fixture(scope="module")
def clause_models():
    return build_type_models(load_clause_artifact().train)


@pytest.fixture(scope="module")
def ledger():
    return load_ledger(PROJECT_ROOT)


# ── P1: observed-position association; imprecise, non-confirmatory primary ───


def test_observed_position_point_estimate_beats_the_unigram_bar(artifact, models):
    recognition = heldout_type_logloss(artifact.test, models)
    assert recognition["mean_delta"] > 0.0, (
        "the observed-position point estimate changed sign; re-run the recorder "
        "and revisit the documented non-confirmatory result"
    )


def test_an_empty_population_cannot_produce_a_recognition_measurement(models):
    """NEGATIVE CONTROL: zero scored sentences raises, never returns NaN.

    ``heldout_type_logloss`` shares ``case_agreement``'s fail-closed contract:
    a mean over zero sentences is undefined, not small, so an evaluation
    stream silently emptied upstream must surface as an error rather than as
    an ``nan`` mean delta that every comparison below would silently fail.
    Planted states: no documents, and a document list whose documents are all
    empty — each scores nothing and each must raise the same ValueError. A
    document list mixing empty and non-empty documents scores the non-empty
    ones and is a legitimate population, not a refusal case.
    """
    with pytest.raises(ValueError, match="no sentences scored"):
        heldout_type_logloss([], models)
    with pytest.raises(ValueError, match="no sentences scored"):
        heldout_type_logloss([[], []], models)
    with pytest.raises(ValueError, match="no sentences scored"):
        heldout_type_logloss([[]], models)


def test_recorded_recognition_matches_a_fresh_measurement(artifact, models, ledger):
    recognition = heldout_type_logloss(artifact.test, models)
    assert ledger.get("stage6_phase_logloss")["value"] == round(recognition["phase_mean"], 4)
    assert ledger.get("stage6_unigram_logloss")["value"] == round(recognition["unigram_mean"], 4)
    assert ledger.get("stage6_logloss_improvement_nats")["value"] == round(
        recognition["mean_delta"], 4
    )
    assert ledger.get("stage6_test_sentences")["value"] == recognition["n_sentences"]
    positive = sum(delta > 0 for delta in recognition["deltas"])
    negative = sum(delta < 0 for delta in recognition["deltas"])
    assert ledger.get("stage6_sign_test_positive")["value"] == positive
    assert ledger.get("stage6_sign_test_negative")["value"] == negative
    assert ledger.get("stage6_sign_test_p")["value"] == pytest.approx(
        binomtest(positive, positive + negative, 0.5).pvalue
    )
    assert ledger.get("stage6_logloss_improvement_nats")["status"] == "descriptive"


def test_document_clustered_primary_interval_spans_zero(artifact, models, ledger):
    """The primary randomization and normal-tail labels cannot be interchanged."""
    recognition = heldout_type_logloss(artifact.test, models)
    documents = [
        document_id
        for document_id, document in zip(artifact.test_document_ids, artifact.test, strict=True)
        for _ in document
    ]
    summary = summarize_clustered_difference(recognition["deltas"], documents)
    effect = float(ledger.get("cluster_stage6_logloss_effect")["value"])
    low = float(ledger.get("cluster_stage6_logloss_ci_low")["value"])
    high = float(ledger.get("cluster_stage6_logloss_ci_high")["value"])
    p_value = float(ledger.get("cluster_stage6_logloss_p")["value"])
    normal_p = float(ledger.get("cluster_stage6_logloss_normal_p")["value"])
    assert effect == pytest.approx(
        float(ledger.get("stage6_logloss_improvement_nats")["value"]), abs=5e-5
    )
    assert p_value == pytest.approx(float(f"{summary.p_value:.4g}"))
    assert normal_p == pytest.approx(float(f"{summary.normal_p_value:.4g}"))
    assert p_value != normal_p
    assert low <= 0.0 <= high
    assert p_value >= 0.05


def test_worse_scrambled_model_can_pass_the_sentence_sign_test(artifact, models):
    """NEGATIVE CONTROL: sign significance cannot certify mean log loss.

    Reversing the phase columns preserves their entropy and makes the mean score
    worse than unigram, yet the secondary sentence-win sign test still rejects
    0.5. A gate using that p-value for P1 would therefore certify its own
    falsifier.
    """
    scrambled = {
        "unigram": models["unigram"],
        "emission": models["emission"][:, ::-1],
    }
    result = heldout_type_logloss(artifact.test, scrambled)
    positive = sum(delta > 0 for delta in result["deltas"])
    negative = sum(delta < 0 for delta in result["deltas"])
    sign_p = binomtest(positive, positive + negative, 0.5).pvalue
    assert result["mean_delta"] < 0.0
    assert sign_p < 0.05


def test_phase_blind_negative_control_cannot_clear_the_recognition_gate(artifact, models):
    """The recognition metric must reject a model whose phase columns are equal."""
    phase_blind = {
        "unigram": models["unigram"],
        "emission": np.repeat(models["unigram"][:, None], N_PHASES, axis=1),
    }
    real_delta = heldout_type_logloss(artifact.test, models)["mean_delta"]
    blind_delta = heldout_type_logloss(artifact.test, phase_blind)["mean_delta"]
    assert real_delta > 0.0
    assert blind_delta == 0.0
    assert ControlHarness("stage6_recognition").verify_negative_control_fails(
        lambda candidate: heldout_type_logloss(artifact.test, candidate)["mean_delta"],
        phase_blind,
        threshold=0.0,
        direction="higher_is_better",
    )


def test_live_agent_filter_is_recorded_as_a_separate_estimand(artifact, models, ledger):
    live = live_agent_type_logloss(
        artifact.test,
        build_narrative_agent(artifact),
        models["unigram"],
    )
    positive = sum(delta > 0 for delta in live["deltas"])
    negative = sum(delta < 0 for delta in live["deltas"])
    assert ledger.get("stage6_live_filter_logloss")["value"] == round(live["live_mean"], 4)
    assert ledger.get("stage6_live_filter_improvement_nats")["value"] == round(
        live["mean_delta"], 4
    )
    assert ledger.get("stage6_live_filter_sign_positive")["value"] == positive
    assert ledger.get("stage6_live_filter_sign_negative")["value"] == negative
    assert ledger.get("stage6_live_filter_sign_p")["value"] == pytest.approx(
        binomtest(positive, positive + negative, 0.5).pvalue
    )
    assert ledger.get("stage6_live_filter_sign_p")["status"] == "descriptive"


def test_live_agent_score_depends_on_its_transition(artifact, models):
    """NEGATIVE CONTROL: replacing B must move the sequential score."""
    import jax.numpy as jnp
    from pymdp.agent import Agent

    shipped = build_narrative_agent(artifact)
    identity_transition = jnp.eye(N_PHASES)[jnp.newaxis, :, :, jnp.newaxis]
    no_progression = Agent(
        A=list(shipped.A),
        B=[identity_transition],
        C=list(shipped.C),
        D=list(shipped.D),
        num_controls=[1],
        batch_size=1,
    )
    shipped_score = live_agent_type_logloss(artifact.test, shipped, models["unigram"])
    broken_score = live_agent_type_logloss(artifact.test, no_progression, models["unigram"])
    assert broken_score["live_mean"] != pytest.approx(shipped_score["live_mean"], abs=1e-6)


def test_the_opening_phase_carries_the_interrogative_excess(models):
    """The train-side pattern that motivated P1, present in the estimates."""
    interrogative = 1
    p_by_phase = models["emission"][interrogative, :]
    assert p_by_phase[0] > p_by_phase[1]
    assert p_by_phase[0] > p_by_phase[2]


# ── P2: the channel, its anchor, and its scramble ────────────────────────────


def test_the_alpha_zero_anchor_is_exact(clause_models, models):
    static = build_joint_start(clause_models)
    for phase in range(N_PHASES):
        anchored = clause_prior_with_narrative(clause_models, phase, models, alpha=0.0)
        np.testing.assert_array_equal(anchored, static)


def test_the_channel_moves_the_type_prior_and_the_scramble_differs(clause_models, models):
    static = build_joint_start(clause_models)
    real = clause_prior_with_narrative(clause_models, 0, models, alpha=1.0)
    scrambled = clause_prior_with_narrative(clause_models, 0, models, alpha=1.0, scrambled=True)
    assert float(np.max(np.abs(real - static))) > 1e-4
    assert not np.allclose(real, scrambled, atol=1e-9)


def test_every_channel_output_remains_a_distribution(clause_models, models):
    for phase in range(N_PHASES):
        for alpha in (0.0, 0.3, 1.0):
            for scrambled in (False, True):
                prior = clause_prior_with_narrative(
                    clause_models, phase, models, alpha=alpha, scrambled=scrambled
                )
                assert abs(prior.sum() - 1.0) < 1e-9
                assert prior.min() >= 0.0


def test_channel_inputs_are_validated(clause_models, models):
    with pytest.raises(ValueError, match="alpha"):
        clause_prior_with_narrative(clause_models, 0, models, alpha=1.5)
    with pytest.raises(ValueError, match="phase"):
        clause_prior_with_narrative(clause_models, 9, models, alpha=0.5)


# ── The structural invariant the first draft mis-stated (entry 26) ───────────


def test_the_phase_transition_upper_triangle_is_exactly_zero(artifact, ledger):
    transition = build_phase_transition(artifact.train)
    above = float(np.triu(transition, k=1).sum())
    assert above == 0.0, (
        f"a tercile decreased ({above:.3e} mass above the diagonal), which is "
        f"arithmetically impossible; the counting is mis-oriented"
    )
    assert ledger.get("stage6_transition_above_diagonal")["value"] == 0.0


def test_tercile_is_total_and_monotone():
    for length in (3, 4, 7, 39):
        phases = [tercile(i, length) for i in range(length)]
        assert phases[0] == 0 and phases[-1] == N_PHASES - 1
        assert all(b >= a for a, b in pairwise(phases))
    with pytest.raises(ValueError):
        tercile(5, 5)


# ── GNN CONFORMANCE ──────────────────────────────────────────────────────────


def test_live_narrative_agent_conforms_to_its_spec(artifact):
    from ailm.gnn import assert_conforms, describe_shapes, load_spec

    assert_conforms(load_spec("L6_narrative"), describe_shapes(build_narrative_agent(artifact)))


def test_narrative_spec_mutation_is_rejected(artifact):
    from ailm.gnn import check_conformance, describe_shapes, load_spec

    violations = check_conformance(
        load_spec("L6_narrative@uniform_phase_prior"),
        describe_shapes(build_narrative_agent(artifact)),
    )
    assert violations


# ── LEDGER COMPLETENESS ──────────────────────────────────────────────────────

STAGE6_LABELS = (
    "stage6_train_documents",
    "stage6_test_documents",
    "stage6_test_sentences",
    "stage6_documents_excluded_short",
    "stage6_phase_logloss",
    "stage6_unigram_logloss",
    "stage6_logloss_improvement_nats",
    "stage6_sign_test_positive",
    "stage6_sign_test_negative",
    "stage6_sign_test_p",
    "stage6_live_filter_logloss",
    "stage6_live_filter_improvement_nats",
    "stage6_live_filter_sign_positive",
    "stage6_live_filter_sign_negative",
    "stage6_live_filter_sign_p",
    "stage6_opening_interrogative_shift",
    "stage6_middle_interrogative_shift",
    "stage6_closing_interrogative_shift",
    "stage6_scrambled_opening_shift",
    "stage6_static_interrogative_mass",
    "stage6_transition_above_diagonal",
)


def test_every_stage6_label_is_recorded(ledger):
    missing = [label for label in STAGE6_LABELS if not ledger.verify_label(label)]
    assert missing == [], f"unrecorded stage6 labels {missing}; run scripts/record_stage6_claims.py"


def test_degenerate_artifacts_are_refused(tmp_path):
    import json

    from ailm.levels.narrative import load_narrative_artifact as load

    bad = {
        "schema": "ailm-narrative-counts-v2",
        "provenance": {},
        "splits": {
            "train": [{"doc": "d", "types": ["declarative", "declarative"]}],
            "test": [{"doc": "e", "types": ["declarative"] * 3}],
        },
        "documents": {
            "ids": ["d", "e"],
            "split_indices": {"train": [0], "test": [1]},
            "sequence_document_indices": {"train": [0], "test": [1]},
        },
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="no arc"):
        load(path)


def test_document_identity_partition_is_authenticated_and_disjoint(artifact):
    assert len(artifact.document_ids) == len(set(artifact.document_ids))
    assert not (set(artifact.train_document_ids) & set(artifact.test_document_ids))
    assert not (
        set(artifact.train_split_document_indices) & set(artifact.test_split_document_indices)
    )
    assert set(artifact.train_split_document_indices) | set(
        artifact.test_split_document_indices
    ) == set(range(len(artifact.document_ids)))
    assert len(artifact.train_document_ids) == len(artifact.train)
    assert len(artifact.test_document_ids) == len(artifact.test)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("cross_split", "outside its document split"),
        ("duplicate_sequence", "unique valid indices"),
        ("entry_alias", "document ids disagree"),
        ("identity_digest", "identity digest"),
        ("provenance_count", "train_eligible_documents"),
    ),
)
def test_narrative_document_identity_planted_controls_fail_closed(tmp_path, mutation, message):
    import json

    from ailm.levels.narrative import DEFAULT_ARTIFACT, load_narrative_artifact

    payload = json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))
    documents = payload["documents"]
    if mutation == "cross_split":
        documents["sequence_document_indices"]["test"][0] = documents["split_indices"]["train"][0]
    elif mutation == "duplicate_sequence":
        documents["sequence_document_indices"]["test"][1] = documents["sequence_document_indices"][
            "test"
        ][0]
    elif mutation == "entry_alias":
        payload["splits"]["test"][0]["doc"] = payload["splits"]["train"][0]["doc"]
    elif mutation == "identity_digest":
        payload["provenance"]["document_identity_sha256"] = "0" * 64
    else:
        payload["provenance"]["train_eligible_documents"] += 1
    path = tmp_path / f"planted-{mutation}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_narrative_artifact(path)


def test_channel_shift_summary_matches_the_ledger(clause_models, models, ledger):
    """The recorder's shift table, re-derived through the public helper."""
    from ailm.levels.narrative import channel_shift_summary

    summary = channel_shift_summary(clause_models, models, alpha=1.0)
    assert ledger.get("stage6_opening_interrogative_shift")["value"] == round(
        summary["shifts"]["opening"], 4
    )
    assert ledger.get("stage6_static_interrogative_mass")["value"] == round(
        summary["static_interrogative_mass"], 4
    )
    assert ledger.get("stage6_scrambled_opening_shift")["value"] == round(
        summary["scrambled_shifts"]["opening"], 4
    )
    # The middle phase's scramble receives the opening message and vice versa;
    # identical real and scrambled shift tables would mean phase is decorative.
    assert summary["shifts"] != summary["scrambled_shifts"]


# ── P4: the finding status is computed, and its promotion is falsifiable ─────
#
# Decision 115 made the Stage 6 finding status a function of two ledger booleans
# rather than a declared string. What that buys is only as good as the predicate
# behind it, so these tests attack the predicate rather than reading the status.


def _replicated(**overrides) -> bool:
    """The promotion predicate at the recorded values, with named substitutions."""
    from ailm.gates import GateVerdict, stage6_positive_replicated_is_consistent

    arguments = {
        "verdict": GateVerdict.BRACKETED_REFERENCE,
        "point_passes": True,
        "point_improvement": 0.0051,
        "cluster_ci_low": -0.009563,
        "cluster_ci_high": 0.024959,
        "held_back_ci_low": 0.002562,
        "held_back_ci_high": 0.014556,
        "held_back_excludes_zero": True,
        "held_back_sign_crosses_threshold": True,
        "held_back_sign_direction_agrees": "true",
    }
    arguments.update(overrides)
    return stage6_positive_replicated_is_consistent(**arguments)


def test_the_promotion_holds_at_the_recorded_values() -> None:
    """POSITIVE CONTROL: the predicate accepts the configuration actually recorded."""
    assert _replicated() is True


@pytest.mark.parametrize(
    ("field", "value", "why"),
    [
        ("held_back_excludes_zero", False, "the held-back interval spans zero"),
        ("held_back_sign_crosses_threshold", False, "the sign test misses its threshold"),
        ("held_back_sign_direction_agrees", "false", "the sign points the other way"),
        ("held_back_sign_direction_agrees", "undetermined", "the sign has no direction"),
        ("point_passes", False, "the point gate does not pass"),
        ("point_improvement", -0.0051, "the development point estimate is negative"),
        ("cluster_ci_low", 0.001, "the development interval no longer spans zero"),
    ],
)
def test_removing_any_single_support_withdraws_the_promotion(field, value, why) -> None:
    """NEGATIVE CONTROLS: every conjunct is load-bearing, one at a time.

    A predicate that promoted on a subset of its stated conditions would let the
    cover print `passing` on evidence the manuscript does not have.
    """
    assert _replicated(**{field: value}) is False, (
        f"the promotion survived a configuration in which {why}"
    )


def test_an_interval_wholly_below_zero_is_not_a_replicated_improvement() -> None:
    """NEGATIVE CONTROL: exclusion is not enough; the direction must be the right one.

    An interval lying entirely below zero also excludes zero, and would mean the
    phase model was reliably WORSE on the held-back population. Reading that as a
    replicated improvement is the single worst failure this predicate could have.
    """
    assert _replicated(held_back_ci_low=-0.014556, held_back_ci_high=-0.002562) is False


def test_a_truthy_non_boolean_cannot_promote_the_finding() -> None:
    """The flags are recorded booleans, and a lookalike is refused, not coerced."""
    from ailm.gates import stage6_positive_replicated_is_consistent  # noqa: F401

    for lookalike in (1, "true", "yes"):
        with pytest.raises(ValueError):
            _replicated(held_back_excludes_zero=lookalike)


def test_the_audited_status_follows_the_recorded_booleans(ledger, gate_audit_rows) -> None:
    """The status in the exported audit is the one the ledger's flags imply.

    Reads the generated report rather than recomputing the gate, so a status that
    was edited into the export instead of computed fails here.
    """
    row = gate_audit_rows["stage6_narrative"]

    excludes = ledger.get("confirmatory_stage6_clustered_excludes_zero")
    crosses = ledger.get("confirmatory_stage6_sign_test_crosses_frozen_threshold")
    assert excludes is not None and crosses is not None, (
        "required held-back clustered evidence is absent from the ledger"
    )
    expected = (
        "positive_point_replicated"
        if bool(excludes["value"]) and bool(crosses["value"])
        else "positive_point_nonconfirmatory"
    )
    assert row["finding_status"] == expected, (
        f"the audit reports {row['finding_status']!r} while the ledger's booleans "
        f"imply {expected!r}; the status was not computed from the record"
    )
    if expected == "positive_point_replicated":
        held_back = row["held_back_cluster"]
        assert held_back["ci_low"] == ledger.get("confirmatory_stage6_clustered_ci_low")["value"]
        assert held_back["ci_high"] == ledger.get("confirmatory_stage6_clustered_ci_high")["value"]


def test_the_development_interval_still_spans_zero_in_the_export(gate_audit_rows) -> None:
    """The promotion does not quietly rewrite the development-population result.

    That sentence stays in the manuscript, and the exported row is where a reader
    checks it. A promotion that moved this bound would be describing a different
    measurement under the same name.
    """
    row = gate_audit_rows["stage6_narrative"]
    primary = row["cluster_primary"]
    assert primary["ci_low"] <= 0.0 <= primary["ci_high"]


# ── P5: the promotion did not make Level 6 unqualified ──────────────────────
#
# Decision 115 moved the badge to `passing`. It did not remove the level's
# qualification, and the abstract's headline count depends on that distinction
# surviving. These guard the seam between the two.


def _limits(claims):
    from ailm.manuscript_variables import level_six_qualification_limits

    return level_six_qualification_limits(claims)


def test_level_six_still_carries_every_recorded_limit(ledger) -> None:
    """All three limits hold on the current record, and each is read separately.

    Asserted individually rather than through ``any``, because the count reaches
    Level 6's qualification through the first one alone. If the other two were
    silently absent the count would still be right today and wrong the moment
    the first one moved.
    """
    claims = {label: ledger.get(label) for label in ledger.all_labels()}
    limits = _limits(claims)
    assert set(limits) == {
        "development_interval_spans_zero",
        "corrected_interval_spans_zero",
        "development_sized_effect_unresolvable",
    }
    assert limits["development_interval_spans_zero"], (
        "the development-population interval no longer spans zero; the manuscript "
        "sentence saying it does is now false and decision 115 must be reviewed"
    )
    assert limits["corrected_interval_spans_zero"], (
        "the family-corrected held-back interval no longer spans zero; the "
        "sensitivity the results section reports has changed"
    )
    assert limits["development_sized_effect_unresolvable"], (
        "a development-sized effect is now resolvable on the held-back population; "
        "the under-power caveat in the discussion is no longer true"
    )


def test_a_missing_arm_cannot_manufacture_a_limit(ledger) -> None:
    """NEGATIVE CONTROL: absence reads as a limit that does not hold.

    A limit derived from a missing label would let a ledger that lost the
    decision-115 arm keep claiming the qualification it no longer has evidence
    for. The two limits owned by that arm must read False when it is gone.
    """
    claims = {label: ledger.get(label) for label in ledger.all_labels()}
    stripped = {
        label: entry
        for label, entry in claims.items()
        if not label.startswith("confirmatory_stage6_clustered_")
    }
    limits = _limits(stripped)
    assert limits["corrected_interval_spans_zero"] is False
    assert limits["development_sized_effect_unresolvable"] is False


def test_the_count_guard_fires_when_the_flag_stops_describing_the_level(ledger) -> None:
    """POSITIVE CONTROL: the render refuses rather than reporting no qualification.

    The qualifying flag reads the development interval alone. Move that interval
    off zero while the other two limits stand, and Level 6 would be counted as an
    unqualified verdict on a record that still qualifies it. That is precisely
    the drift the guard exists to catch, so it is planted here rather than
    trusted.
    """
    from ailm.manuscript_variables import qualified_verdict_flags

    claims = {label: ledger.get(label) for label in ledger.all_labels()}
    planted = dict(claims)
    planted["cluster_stage6_logloss_ci_low"] = {
        **claims["cluster_stage6_logloss_ci_low"],
        "value": 1e-6,
    }
    limits = _limits(planted)
    assert limits["development_interval_spans_zero"] is False
    assert any(limits.values()), "the plant removed every limit; the control is vacuous"
    assert qualified_verdict_flags(planted)["stage6_primary_nonconfirmatory"] is False, (
        "the qualifying flag survived an interval that no longer spans zero, so "
        "the guard below would never be reachable"
    )
