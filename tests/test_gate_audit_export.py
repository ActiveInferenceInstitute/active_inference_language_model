"""Schema tests for the ``--json`` export of ``scripts/audit_gates.py``.

The full audit is slow (it runs pymdp over every stage), so these tests never
call an ``audit_*`` function. They exercise the serialization boundary alone:
two synthetic audit dicts shaped exactly like the real ones, pushed through
``gate_audit_row`` and ``write_gate_audit_json``, then read back as JSON. That
is the contract a machine consumer of ``output/reports/gate_audit.json``
depends on — enums flattened to their stable string values, numbers as plain
floats, ``eval_positions`` present only where the audit measured one.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from ailm.claims import load_ledger
from ailm.gates import (
    STAGE5W_REGISTERED_GAIN_INTERVAL_NATS,
    ComparatorType,
    GateId,
    evaluate_gate,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _PROJECT_ROOT / "scripts" / "audit_gates.py"


def _load_audit_gates():
    """Import ``scripts/audit_gates.py`` by path — scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("ailm_audit_gates", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "arguments,bounded", [([], False), (["--full"], False), (["--limit", "3"], True)]
)
def test_population_note_matches_the_cli_selected_scope(arguments, bounded, capsys) -> None:
    module = _load_audit_gates()
    selected = module.build_argument_parser().parse_args(arguments)
    module.print_g2p_population_note(selected.limit)
    output = capsys.readouterr().out
    assert ("bounded subsample" in output) is bounded
    assert ("full held-out CMUdict population" in output) is not bounded


def _synthetic_audits() -> list[dict]:
    """Two audit dicts covering both directions, both comparator strengths,
    both outcomes, and the presence/absence of ``eval_positions``."""
    bracketed_pass = {
        "gate_id": GateId.STAGE1_LETTER,
        "stage": "synthetic lower-is-better",
        "baseline": 20.0,
        "baseline_name": "synthetic unigram",
        "model": 10.0,
        "ceiling": 8.0,
        "ceiling_name": "synthetic entropy floor",
        "ceiling_type": ComparatorType.MODEL_CLASS_BOUND,
        "threshold": 20.0,
        "lower_is_better": True,
        # ln(20/10) nats: clears Stage 1's declared 0.25-nat SESOI.
        "sesoi_margin": math.log(20.0 / 10.0),
    }
    above_reference_fail = {
        "gate_id": GateId.STAGE2_G2P,
        "stage": "synthetic higher-is-better",
        "baseline": 0.50,
        "baseline_name": "synthetic argmax",
        "model": 0.55,
        "ceiling": 0.60,
        "ceiling_name": "synthetic smoothing reference",
        "ceiling_type": ComparatorType.REFERENCE_MODEL,
        "threshold": 0.70,
        "lower_is_better": False,
        "eval_positions": 1234,
        # 0.05 pts: below Stage 2B's declared 0.5-pt SESOI, so the export
        # must say so rather than leave the hairline invisible.
        "sesoi_margin": 0.05,
    }
    return [bracketed_pass, above_reference_fail]


def _rows(module) -> list[dict]:
    rows = []
    for audit in _synthetic_audits():
        evaluation = evaluate_gate(
            baseline=audit["baseline"],
            model=audit["model"],
            ceiling=audit["ceiling"],
            threshold=audit["threshold"],
            direction="lower_is_better" if audit["lower_is_better"] else "higher_is_better",
            ceiling_type=audit["ceiling_type"],
        )
        rows.append(module.gate_audit_row(audit, evaluation))
    return rows


def _evaluation(audit: dict):
    """Evaluate a real or synthetic audit through the shared gate contract."""
    return evaluate_gate(
        baseline=audit["baseline"],
        model=audit["model"],
        ceiling=audit["ceiling"],
        threshold=audit["threshold"],
        direction="lower_is_better" if audit["lower_is_better"] else "higher_is_better",
        ceiling_type=audit["ceiling_type"],
    )


def test_export_round_trips_through_json_with_the_declared_schema(tmp_path):
    module = _load_audit_gates()
    out = tmp_path / "gate_audit.json"
    module.write_gate_audit_json(_rows(module), out)

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(loaded, list) and len(loaded) == 2

    base_keys = {
        "gate_id",
        "stage",
        "baseline",
        "baseline_name",
        "model",
        "ceiling",
        "ceiling_name",
        "ceiling_type",
        "threshold",
        "verdict",
        "outcome",
        "baseline_relative",
        "margin_of_interest",
    }
    first, second = loaded
    assert set(first) == base_keys, "eval_positions must be absent when unmeasured"
    assert set(second) == base_keys | {"eval_positions"}
    assert second["eval_positions"] == 1234


def test_enums_serialize_as_their_stable_string_values(tmp_path):
    module = _load_audit_gates()
    out = tmp_path / "gate_audit.json"
    module.write_gate_audit_json(_rows(module), out)
    first, second = json.loads(out.read_text(encoding="utf-8"))

    # str() on the StrEnums, never repr(): a consumer greps for these values.
    assert first["gate_id"] == "stage1_letter"
    assert first["ceiling_type"] == "model_class_bound"
    assert second["gate_id"] == "stage2_g2p"
    assert second["ceiling_type"] == "reference_model"
    for row in (first, second):
        assert "GateId" not in row["gate_id"] and "ComparatorType" not in row["ceiling_type"]


def test_verdict_outcome_and_baseline_relativity_are_derived_not_copied(tmp_path):
    module = _load_audit_gates()
    out = tmp_path / "gate_audit.json"
    module.write_gate_audit_json(_rows(module), out)
    first, second = json.loads(out.read_text(encoding="utf-8"))

    # Row 1: threshold == baseline (baseline-relative), bound-type ceiling
    # clears it, model beats it: BRACKETED_BOUND / PASS.
    assert first["verdict"] == "BRACKETED_BOUND"
    assert first["outcome"] == "PASS"
    assert first["baseline_relative"] is True

    # Row 2: threshold sits above the reference ceiling and the model:
    # ABOVE_REFERENCE / FAIL, and the threshold is not the baseline.
    assert second["verdict"] == "ABOVE_REFERENCE"
    assert second["outcome"] == "FAIL"
    assert second["baseline_relative"] is False

    # Numbers must land as plain JSON floats, not strings or nulls.
    for row in (first, second):
        for key in ("baseline", "model", "ceiling", "threshold"):
            assert isinstance(row[key], float)


def test_export_path_is_the_tracked_reports_surface():
    module = _load_audit_gates()
    assert module.GATE_AUDIT_JSON == _PROJECT_ROOT / "output" / "reports" / "gate_audit.json"


@pytest.mark.parametrize(
    ("argv", "expected_limit"),
    (
        ((), None),
        (("--full",), None),
        (("--limit", "400"), 400),
    ),
)
def test_gate_audit_cli_population_contract(argv: tuple[str, ...], expected_limit: int | None):
    """Default and explicit full both mean the complete held-out population."""
    module = _load_audit_gates()
    assert module.build_argument_parser().parse_args(argv).limit == expected_limit


def test_gate_audit_cli_refuses_two_population_selectors():
    module = _load_audit_gates()
    with pytest.raises(SystemExit):
        module.build_argument_parser().parse_args(("--full", "--limit", "400"))


def test_stage2b_full_population_selector_exhausts_the_recorded_heldout_split():
    """Prove ``limit=None`` means full, without paying for a live pymdp audit."""
    from ailm.levels.letter import ALPHA_TO_IDX
    from ailm.levels.phoneme import _iter_pairs, g2p_split, load_phoneme_counts

    counts = load_phoneme_counts(_PROJECT_ROOT / "vendor")
    _train, heldout = g2p_split(counts)
    selected = _iter_pairs(heldout, None)
    positions = sum(
        len(phones)
        for word, phones in selected
        if len([char for char in word if char in ALPHA_TO_IDX]) == len(phones)
    )
    ledger = load_ledger(_PROJECT_ROOT)
    pair_entry = ledger.get("stage2_g2p_test_pairs")
    position_entry = ledger.get("stage2_g2p_eval_positions")
    assert pair_entry is not None and position_entry is not None
    assert len(selected) == len(heldout) == int(pair_entry["value"])
    assert positions == int(position_entry["value"])

    bounded = _iter_pairs(heldout, min(len(heldout), 400))
    assert len(bounded) <= len(selected)
    if len(heldout) > len(bounded):
        assert bounded != selected


def test_stage2a_audits_the_full_dictionary_sufficient_statistics():
    module = _load_audit_gates()
    from ailm.levels.phoneme import (
        full_corpus_order1_floor_perplexity,
        full_corpus_perplexity,
        full_corpus_unigram_perplexity,
        load_phoneme_counts,
    )

    counts = load_phoneme_counts(_PROJECT_ROOT / "vendor")
    audit = module.audit_stage2a()
    model, positions = full_corpus_perplexity(counts)
    floor, floor_positions = full_corpus_order1_floor_perplexity(counts)
    assert audit["model"] == pytest.approx(model)
    assert audit["baseline"] == pytest.approx(full_corpus_unigram_perplexity(counts))
    assert audit["ceiling"] == pytest.approx(floor)
    assert audit["eval_positions"] == positions == floor_positions
    assert audit["evaluation_population"] == "in-sample full CMUdict dictionary"


def test_stage2a_entropy_floor_scores_starts_and_transitions_on_one_population():
    from ailm.levels.phoneme import full_corpus_order1_floor_perplexity

    counts = {
        "phone_initials": {"AA": 2, "AE": 1},
        "phone_bigrams": {"AA,AE": 2, "AE,AA": 1},
    }
    floor, positions = full_corpus_order1_floor_perplexity(counts)

    # All three observed transitions are deterministic under their preceding
    # symbol. Only the three start events carry entropy, across six positions.
    start_loss = -(2 * math.log(2 / 3) + math.log(1 / 3))
    assert floor == pytest.approx(math.exp(start_loss / 6))
    assert positions == 6
    assert floor > 1.0, (
        "a transitions-only floor would be exactly one here and would silently "
        "omit every independently reset start position"
    )


def test_stage2a_entropy_floor_rejects_an_empty_or_foreign_population():
    from ailm.levels.phoneme import full_corpus_order1_floor_perplexity

    with pytest.raises(ValueError, match="at least one word start"):
        full_corpus_order1_floor_perplexity({"phone_initials": {}, "phone_bigrams": {}})
    with pytest.raises(ValueError, match="non-canonical phone"):
        full_corpus_order1_floor_perplexity({"phone_initials": {"FOREIGN": 1}, "phone_bigrams": {}})


def test_generation_audit_uses_exact_tv_and_the_composite_gate():
    """The scalar-TV false positive at gamma 16 must fail the exported outcome."""
    module = _load_audit_gates()
    from ailm.generation import (
        exact_ancestral_lexicon_total_variation,
        exact_lexicon_total_variation,
    )
    from ailm.levels.letter import build_letter_transition, load_brown_counts
    from ailm.levels.word import build_word_prior, load_word_artifact

    artifact = load_word_artifact()
    prior = build_word_prior(artifact)
    habit = np.asarray(build_letter_transition(load_brown_counts()))[0]

    shipped = module.audit_generation()
    shipped_evaluation = _evaluation(shipped)
    assert shipped["baseline"] == pytest.approx(
        exact_ancestral_lexicon_total_variation(artifact, prior, habit)
    )
    gamma_entry = load_ledger(_PROJECT_ROOT).get("generation_best_gamma")
    assert gamma_entry is not None
    assert shipped["model"] == pytest.approx(
        exact_lexicon_total_variation(
            artifact,
            prior,
            habit,
            float(gamma_entry["value"]),
        )
    )
    assert shipped["generation_gate"]["tv_passes"]
    assert shipped["generation_gate"]["diversity_passes"]
    assert module.gate_outcome_passes(shipped, shipped_evaluation)

    degenerate = module.audit_generation(gamma=16.0)
    degenerate_evaluation = _evaluation(degenerate)
    assert degenerate_evaluation.passes, "gamma 16 must preserve the scalar-TV false positive"
    assert degenerate["generation_gate"]["model_distinct"] <= 2.0
    assert not degenerate["generation_gate"]["diversity_passes"]
    assert not module.gate_outcome_passes(degenerate, degenerate_evaluation)

    row = module.gate_audit_row(degenerate, degenerate_evaluation)
    assert row["outcome"] == "FAIL"
    assert row["generation_gate"]["tv_estimator"] == "exact finite distribution"
    assert row["generation_gate"]["tv_passes"] is True
    assert row["generation_gate"]["diversity_passes"] is False
    assert row["generation_gate"]["composite_passes"] is False


def test_stage6_export_separates_point_pass_from_clustered_inference():
    """A PASS must serialize the inference it actually has, and no more.

    Decision 115 changed which inference that is. The point gate has always
    passed and the development-population interval has always spanned zero;
    what the level now also has is the same estimand on the held-back split.
    The exported ``finding_status`` is computed from two ledger booleans, so
    this test derives the expected word from those booleans rather than naming
    one, and would have caught the promotion as a failure had it been typed.
    """
    module = _load_audit_gates()
    audit = module.audit_stage6()
    evaluation = _evaluation(audit)
    ledger = load_ledger(_PROJECT_ROOT)

    # Unchanged by the promotion, and the manuscript still says all three.
    assert evaluation.passes
    assert audit["cluster_ci_low"] <= 0.0 <= audit["cluster_ci_high"]
    assert audit["cluster_p"] >= 0.05

    excludes = ledger.get("confirmatory_stage6_clustered_excludes_zero")
    crosses = ledger.get("confirmatory_stage6_sign_test_crosses_frozen_threshold")
    replicated = (
        excludes is not None
        and crosses is not None
        and bool(excludes["value"])
        and bool(crosses["value"])
    )
    expected = "positive_point_replicated" if replicated else "positive_point_nonconfirmatory"

    assert module.expected_finding_is_consistent(audit, evaluation)
    row = module.gate_audit_row(audit, evaluation)
    assert row["outcome"] == "PASS"
    assert row["finding_status"] == expected, (
        f"the export reports {row['finding_status']!r} while the ledger's booleans "
        f"imply {expected!r}; the status was not computed from the record"
    )
    assert row["cluster_primary"]["unit"] == "document"
    assert row["cluster_primary"]["p_method"] == (
        "cluster sign-flip sensitivity under joint central symmetry"
    )

    if not replicated:
        # The pre-decision-115 shape. Its own controls are exercised below via
        # the planted held-back arm, so nothing is skipped silently.
        assert "held_back_cluster" not in row or not row["held_back_cluster"]["excludes_zero"]
        return

    # Both intervals travel together: the status is a statement about the pair.
    held_back = row["held_back_cluster"]
    assert held_back["unit"] == "document"
    assert held_back["population"] == "ud_ewt_test"
    assert held_back["excludes_zero"] is True
    assert held_back["ci_low"] > 0.0, (
        "the held-back interval must exclude zero on the POSITIVE side; an "
        "interval wholly below zero also excludes it and would mean the model "
        "was reliably worse"
    )
    assert held_back["ci_low"] == ledger.get("confirmatory_stage6_clustered_ci_low")["value"]
    assert held_back["ci_high"] == ledger.get("confirmatory_stage6_clustered_ci_high")["value"]

    # POSITIVE CONTROLS on the promoted status: each support removed alone must
    # withdraw it. Planted on the audit dict, so the real recorder is untouched.
    for field, value, why in (
        ("excludes_zero", False, "the held-back interval spans zero"),
        ("sign_crosses_frozen_threshold", False, "the sign test misses its threshold"),
        ("sign_direction_agrees", "false", "the sign points the other way"),
    ):
        planted = {**audit, "held_back_cluster": {**audit["held_back_cluster"], field: value}}
        assert not module.expected_finding_is_consistent(planted, evaluation), (
            f"the promoted status survived a configuration in which {why}"
        )

    wholly_negative = {
        **audit,
        "held_back_cluster": {
            **audit["held_back_cluster"],
            "ci_low": -audit["held_back_cluster"]["ci_high"],
            "ci_high": -audit["held_back_cluster"]["ci_low"],
        },
    }
    assert not module.expected_finding_is_consistent(wholly_negative, evaluation)

    # A held-back arm that spans zero must fall back to the earlier status, and
    # that status must still be consistent on the development numbers. Without
    # this the fallback could be unreachable and nobody would notice.
    spanning = {
        **audit,
        "expected_outcome": "positive_point_nonconfirmatory",
        "held_back_cluster": {**audit["held_back_cluster"], "excludes_zero": False},
    }
    assert module.expected_finding_is_consistent(spanning, evaluation)
    assert module.gate_audit_row(spanning, evaluation)["finding_status"] == (
        "positive_point_nonconfirmatory"
    )

    # A missing held-back arm cannot promote: the status is evidence-backed.
    absent = {k: v for k, v in audit.items() if k != "held_back_cluster"}
    assert not module.expected_finding_is_consistent(absent, evaluation)

    # POSITIVE CONTROLS retained from before the promotion: the development-side
    # supports are still load-bearing.
    one_sided = {**audit, "cluster_ci_low": 1e-6}
    assert not module.expected_finding_is_consistent(one_sided, evaluation)
    negative_point = {**audit, "point_improvement": -1e-3}
    assert not module.expected_finding_is_consistent(negative_point, evaluation)
    failing_point = {**audit, "model": audit["baseline"] + 1.0}
    assert not module.expected_finding_is_consistent(failing_point, _evaluation(failing_point))


def test_stage5_case_only_ablation_requires_unresolved_gain_and_live_control():
    """Every component of decision 83 must remain true together.

    Decision 114 promoted the widened arm and made this one the retained
    ablation, so this covers ``audit_stage5_case_only`` rather than
    ``audit_stage5``. Nothing it asserts changed with the promotion, which is
    what makes the ablation evidence rather than decoration: if its record
    could drift once it stopped being the headline, keeping it would buy
    nothing.
    """
    module = _load_audit_gates()
    audit = module.audit_stage5_case_only()
    evaluation = _evaluation(audit)
    assert not evaluation.passes
    assert module.expected_finding_is_consistent(audit, evaluation)

    row = module.gate_audit_row(audit, evaluation)
    assert row["finding_status"] == "nonpassing_oracle_gain_unresolved"
    assert row["verdict"] == "ABOVE_REFERENCE"
    assert row["oracle_gain"]["resampling_unit"] == "document"
    assert row["oracle_gain"]["n_sentences"] == audit["oracle_n_sentences"]
    assert row["oracle_gain"]["n_documents"] == audit["oracle_n_documents"]
    assert row["oracle_gain"]["n_documents"] < row["oracle_gain"]["n_sentences"]
    assert row["oracle_gain"]["n_positions"] == audit["oracle_n_positions"]
    assert row["oracle_gain"]["mean_nats_per_sentence"] < 0
    assert row["oracle_gain"]["pooled_nats_per_position"] < 0
    assert row["oracle_gain"]["sentence_ci_low"] < 0 < row["oracle_gain"]["sentence_ci_high"]
    assert row["oracle_gain"]["position_ci_low"] < 0 < row["oracle_gain"]["position_ci_high"]
    assert row["oracle_gain"]["confidence"] == 0.99
    assert row["oracle_gain"]["resamples"] == 10_000
    assert row["oracle_gain"]["seed"] == 20260804

    unexpected_pass = {**audit, "model": audit["baseline"] - 1e-3}
    assert _evaluation(unexpected_pass).passes
    assert not module.expected_finding_is_consistent(unexpected_pass, _evaluation(unexpected_pass))
    dead_accuracy_control = {**audit, "control_accuracy": 0.9}
    assert not module.expected_finding_is_consistent(dead_accuracy_control, evaluation)
    dead_perplexity_control = {
        **audit,
        "control_mixture_perplexity": audit["control_typefree_perplexity"],
    }
    assert not module.expected_finding_is_consistent(dead_perplexity_control, evaluation)
    resolved_sentence_gain = {**audit, "oracle_sentence_ci_low": 1e-6}
    assert not module.expected_finding_is_consistent(resolved_sentence_gain, evaluation)
    resolved_position_gain = {**audit, "oracle_position_ci_high": -1e-6}
    assert not module.expected_finding_is_consistent(resolved_position_gain, evaluation)


def test_the_canonical_stage5_row_is_an_ordinary_passing_gate():
    """The promoted arm claims its gate the same way Stages 1 through 4 do.

    Decision 114 gave the canonical Level 5 no expected-outcome exception on
    purpose. A gate that could have failed and did not is held to the ordinary
    rule, a bracketed placement plus a passing outcome, and gets no special
    predicate to hide behind. The two arms also have to stay apart in the
    export: the canonical row carries the registered gate id and the ablation a
    plain string, so a consumer reading gate_audit.json never sees the
    ablation's numbers under Level 5's name.
    """
    module = _load_audit_gates()
    audit = module.audit_stage5()
    evaluation = _evaluation(audit)
    assert evaluation.passes
    assert "expected_outcome" not in audit
    assert module.expected_finding_is_consistent(audit, evaluation)

    row = module.gate_audit_row(audit, evaluation)
    assert row["gate_id"] == "stage5_clause"
    assert row["outcome"] == "PASS"
    assert row["verdict"] == "BRACKETED_REFERENCE"
    assert row.get("finding_status") is None
    # The promotion rests on a gain that resolved. Both document-resampled
    # intervals must stay off zero, in the direction recorded.
    assert row["oracle_gain"]["sentence_ci_low"] > 0
    assert row["oracle_gain"]["position_ci_low"] > 0
    assert row["oracle_gain"]["resampling_unit"] == "document"

    # POSITIVE CONTROL: with no exception to fall back on, a mixture that stops
    # beating its own bar has to fail the ordinary rule.
    failing_point = {**audit, "model": audit["baseline"] + 1.0}
    assert not module.expected_finding_is_consistent(failing_point, _evaluation(failing_point))

    # Decision 114's withdrawal conditions and the registered predictions are
    # the guard's second half. The audit RECOMPUTES every quantity it asserts,
    # so each planted dict stands in for a run that produced the dead value,
    # and each must fail the guard for its own reason.
    assert audit["swapped_blocks_accuracy"] < audit["model_classification_accuracy"]
    assert audit["leak_control_divergence"] > audit["decoder_tolerance"]
    low, high = STAGE5W_REGISTERED_GAIN_INTERVAL_NATS
    assert low <= audit["oracle_mean_nats_per_sentence"] <= high
    dead_accuracy_control = {**audit, "control_accuracy": 0.5}
    assert not module.expected_finding_is_consistent(dead_accuracy_control, evaluation)
    percentage_scale_control = {**audit, "control_accuracy": 99.5}
    with pytest.raises(ValueError):
        module.expected_finding_is_consistent(percentage_scale_control, evaluation)
    dead_perplexity_control = {
        **audit,
        "control_mixture_perplexity": audit["control_typefree_perplexity"],
    }
    assert not module.expected_finding_is_consistent(dead_perplexity_control, evaluation)
    resolved_sentence_gain = {**audit, "oracle_sentence_ci_low": 0.0}
    assert not module.expected_finding_is_consistent(resolved_sentence_gain, evaluation)
    resolved_position_gain = {**audit, "oracle_position_ci_low": 0.0}
    assert not module.expected_finding_is_consistent(resolved_position_gain, evaluation)
    dead_swapped_control = {
        **audit,
        "swapped_blocks_accuracy": audit["model_classification_accuracy"],
    }
    assert not module.expected_finding_is_consistent(dead_swapped_control, evaluation)
    dead_leak_control = {**audit, "leak_control_divergence": audit["decoder_tolerance"]}
    assert not module.expected_finding_is_consistent(dead_leak_control, evaluation)
    contaminated_gain = {**audit, "oracle_mean_nats_per_sentence": 0.01}
    assert not module.expected_finding_is_consistent(contaminated_gain, evaluation)
    one_sided_auc = {**audit, "auc_first_aux_control": audit["auc"] + 1e-3}
    # Registered prediction 2 is NOT a withdrawal condition: decision 112
    # records its failure as an outcome and decision 114's withdrawal list is
    # exhaustive, so a finite measured AUC that loses to one comparator must
    # keep the exit status intact while the diagnostic names the violation.
    assert module.expected_finding_is_consistent(one_sided_auc, evaluation)
    diagnostic = module.canonical_stage5_prediction2_diagnostic(one_sided_auc)
    assert diagnostic is not None and "prediction 2" in diagnostic
    # A corrupted (percentage-scale) AUC field still fails closed — schema
    # violation, not a reportable finding.
    with pytest.raises(ValueError):
        module.canonical_stage5_prediction2_diagnostic({**audit, "auc": 100 * audit["auc"]})
    assert module.canonical_stage5_prediction2_diagnostic(audit) is None
    missing_control = {k: v for k, v in audit.items() if k != "control_accuracy"}
    with pytest.raises(KeyError):
        module.expected_finding_is_consistent(missing_control, evaluation)
    ablation = module.audit_stage5_case_only()
    assert str(ablation["gate_id"]) == module.STAGE5_CASE_ONLY_GATE_ID
    assert str(row["gate_id"]) != module.STAGE5_CASE_ONLY_GATE_ID


def test_margin_of_interest_reports_the_declared_sesoi(tmp_path):
    """The decision-123 surface: value, unit, floor, and the clears verdict.

    A clears_sesoi of False never flips the outcome — the second synthetic
    row FAILs its gate for placement reasons and the first PASSes — it makes
    a hairline margin legible beside the verdict instead of after it.
    """
    module = _load_audit_gates()
    out = tmp_path / "gate_audit.json"
    module.write_gate_audit_json(_rows(module), out)
    first, second = json.loads(out.read_text(encoding="utf-8"))

    assert first["margin_of_interest"] == {
        "value": math.log(20.0 / 10.0),
        "unit": "nats",
        "smallest_effect_size": 0.25,
        "clears_sesoi": True,
    }
    assert second["margin_of_interest"] == {
        "value": 0.05,
        "unit": "percentage points",
        "smallest_effect_size": 0.5,
        "clears_sesoi": False,
    }
