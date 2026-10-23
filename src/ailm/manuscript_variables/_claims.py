"""Claim reading, numeric-domain validation, and claim-shape helpers."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from enum import Enum
from typing import (
    Any,
    NamedTuple,
)

from ailm.benchmarks import wilson_interval
from ailm.value_formatting import (
    ManuscriptVariableError,
)
from ailm.value_formatting import (
    format_ledger_value as _format_value,
)

from ._exceptions import MissingClaimError
from ._stats import _integer_success_count


def _claim_number(claims: Mapping[str, Any], label: str) -> float:
    """Read one numeric value from a label-to-entry mapping."""
    try:
        value = claims[label]["value"]
    except (KeyError, TypeError) as exc:
        raise MissingClaimError(
            f"categorical manuscript derivation needs numeric {label!r}"
        ) from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManuscriptVariableError(
            f"categorical manuscript derivation needs numeric {label!r}, got {value!r}"
        )
    if not math.isfinite(float(value)):
        raise ManuscriptVariableError(
            f"categorical manuscript derivation needs finite {label!r}, got {value!r}"
        )
    return float(value)


def require_close(
    label: str,
    observed: float,
    expected: float,
    *,
    absolute_tolerance: float = 1e-12,
) -> None:
    """Fail when two independently recorded sides of an identity diverge."""
    if not math.isclose(
        observed,
        expected,
        rel_tol=1e-12,
        abs_tol=absolute_tolerance,
    ):
        raise ManuscriptVariableError(
            f"claim {label!r} is inconsistent: recorded {observed!r}, recomputed {expected!r}"
        )


#: Tolerances for float protocol fields bound to the executable registry
#: (decision 131): a few rounding units of an O(1) probability, summed.
#: Integers, booleans and strings remain bit-exact.
REGISTRY_FLOAT_REL_TOL = 1e-12
REGISTRY_FLOAT_ABS_TOL = 1e-15


class _ClaimReaders:
    """Readers over one validated claims ledger.

    These are the closures that lived nested in
    :func:`generate_manuscript_variables` and captured ``claims`` — ``num``,
    ``require_exact_claim``, ``require_namespace_provenance``, ``claim_text`` —
    plus the formats derived from them (``probability``, ``percentage``,
    ``percentage_cell``, ``p_value_cell``). The ledger is bound once here, and
    each producer aliases the methods its body consumes, so the moved code keeps
    the exact text of the closure it replaced.
    """

    def __init__(self, claims: Mapping[str, Any]) -> None:
        self._claims = claims

    def num(self, label: str) -> float:
        """Read a numeric claim, failing diagnosably rather than with a KeyError.

        Any label reached here must also appear in REQUIRED_CLAIMS so the failure
        surfaces at the guard with the full missing-label list, rather than one at
        a time from deep in the generator.
        """
        if label not in self._claims:
            raise MissingClaimError(
                f"Derived token computation needs claim {label!r}, which is absent "
                f"from the ledger and missing from REQUIRED_CLAIMS. Add it to "
                f"REQUIRED_CLAIMS so the guard catches this up front."
            )
        return _claim_number(self._claims, label)

    def raw(self, label: str) -> Any:
        """Read a claim's recorded value without numeric coercion.

        For list- and boolean-valued labels (survivor sets, verdict flags) that
        prose renders directly. Absence fails the same diagnosable way ``num``
        fails, for the same reason.
        """
        if label not in self._claims:
            raise MissingClaimError(
                f"Derived token computation needs claim {label!r}, which is absent "
                f"from the ledger and missing from REQUIRED_CLAIMS. Add it to "
                f"REQUIRED_CLAIMS so the guard catches this up front."
            )
        entry = self._claims[label]
        if not isinstance(entry, Mapping) or "value" not in entry:
            raise ManuscriptVariableError(f"claim {label!r} has no recorded value")
        return entry["value"]

    def probability(self, label: str) -> float:
        """Return one ledger probability after enforcing its closed domain."""
        value = self.num(label)
        if not 0.0 <= value <= 1.0:
            raise ManuscriptVariableError(
                f"probability claim {label!r} must lie in [0, 1], got {value!r}"
            )
        return value

    def percentage(self, label: str) -> float:
        value = self.num(label)
        if not 0.0 <= value <= 100.0:
            raise ManuscriptVariableError(
                f"percentage claim {label!r} must lie in [0, 100], got {value!r}"
            )
        return value

    def stage5_interval_status(self, low_label: str, high_label: str) -> str:
        """Classify a recorded interval as spanning zero or lying on one side."""
        low = self.num(low_label)
        high = self.num(high_label)
        if low > high:
            raise ManuscriptVariableError(
                f"Stage 5 interval endpoints are reversed: {low_label}={low}, {high_label}={high}"
            )
        if low <= 0.0 <= high:
            return "spans zero"
        return "is entirely positive" if low > 0.0 else "is entirely negative"

    def require_exact_claim(self, label: str, expected: object) -> None:
        """Bind a recorded protocol field to the executable code registry."""

        if label not in self._claims:
            raise MissingClaimError(
                f"protocol reconciliation needs claim {label!r}; add it to "
                "REQUIRED_CLAIMS and re-record the owning family"
            )
        observed = self._claims[label].get("value")
        type_matches = (
            type(observed) is type(expected)
            if isinstance(expected, (bool, int, str))
            else isinstance(observed, (int, float)) and not isinstance(observed, bool)
        )
        if type_matches and isinstance(expected, float):
            # A float protocol field is computed through the platform's LAPACK
            # (``np.linalg.lstsq`` in the target-start TVDs), whose last bits are
            # not stable across library builds: the macOS 27.0.1 Accelerate update
            # moved two such values by about 1e-17 (decision 131). A TVD is half a
            # sum of differences of O(1) probabilities, so its rounding error is
            # absolute on the probability scale. Integer, boolean and string
            # fields stay bit-exact.
            matches = math.isclose(
                float(observed),
                expected,
                rel_tol=REGISTRY_FLOAT_REL_TOL,
                abs_tol=REGISTRY_FLOAT_ABS_TOL,
            )
        else:
            matches = observed == expected
        if not type_matches or not matches:
            raise ManuscriptVariableError(
                f"claim {label!r} disagrees with the executable protocol registry: "
                f"recorded {observed!r}, expected {expected!r}"
            )

    def require_namespace_provenance(
        self,
        prefix: str,
        *,
        source: str,
        run_id: str,
    ) -> None:
        """Reject mixed-run protocol namespaces before generating prose from them."""

        mismatched = sorted(
            label
            for label, entry in self._claims.items()
            if label.startswith(prefix)
            and (entry.get("source") != source or entry.get("run_id") != run_id)
        )
        if mismatched:
            raise ManuscriptVariableError(
                f"claims in {prefix!r} are not all bound to {source} / {run_id}: {mismatched}"
            )

    def claim_text(self, label: str) -> str:
        """Format one numeric claim used by a generated Markdown block."""
        if label not in self._claims:
            raise MissingClaimError(
                f"generated manuscript block needs claim {label!r}; add it to "
                "REQUIRED_CLAIMS and re-record the complete family"
            )
        return _format_value(_claim_number(self._claims, label))

    def percentage_cell(self, label: str) -> str:
        """Format one validated probability as a compact table percentage."""
        return f"{self.probability(label) * 100:.2f}%"

    def p_value_cell(self, label: str) -> str:
        """Format one validated p-value to three significant digits for display."""
        rendered = f"{self.probability(label):.3g}"
        # Python pads single-digit exponents (``e-08``).  The shorter spelling
        # is mathematically identical and keeps narrow manuscript columns from
        # overflowing without replacing the exact value in the claims ledger.
        return re.sub(r"e([+-])0+(\d+)$", r"e\1\2", rendered)


class _SyntheticSweep(NamedTuple):
    """Cross-section state the synthetic-context producer consumes: the scored
    synthetic population and the declared short-window size it displays.
    """

    positions: int
    short_window: int


class _ContextReconciliation(NamedTuple):
    """Cross-section state the context-table producer consumes: the parsed
    recorded grid and the scored context population.
    """

    grid: tuple[int, ...]
    positions: int


class _LexicalSweep(NamedTuple):
    """Cross-section state the null-bounds producer consumes for the
    high-alpha arm: the sweep coordinate prefixes and the per-coordinate
    accuracy/delta/scrambled/discordance metrics.
    """

    prefix_by_alpha: dict[float, str]
    metrics: dict[float, dict[str, float | int]]


def _validate_synthetic_interval(
    readers: _ClaimReaders,
    positions: int,
    system: str,
    accuracy: float,
    *,
    namespace: str = "bench_synth",
    estimate_stem: str = "letter_top1",
    interval_stem: str = "letter_top1_wilson",
) -> tuple[str, str]:
    """Rebuild one Wilson interval from its integer sufficient statistic.

    Was a closure capturing the synthetic sweep's scored population; the
    synthetic, synthetic-context, and synthetic-phone producers share it, so
    the population is an explicit parameter, the readers arrive with the
    ledger binding, and the namespace names which stipulated arm owns the
    labels.
    """
    successes = _integer_success_count(
        f"synthetic {estimate_stem} accuracy for {system}",
        accuracy,
        positions,
    )
    low_label = f"{namespace}_{interval_stem}_low_{system}"
    high_label = f"{namespace}_{interval_stem}_high_{system}"
    low = readers.probability(low_label)
    high = readers.probability(high_label)
    expected_low, expected_high = wilson_interval(successes, positions)
    require_close(low_label, low, expected_low)
    require_close(high_label, high, expected_high)
    return low_label, high_label


_REQUIRED_STRING_CLAIMS = frozenset(
    {
        "connected_gum_corpus_sha256",
        "connected_gum_execution_sha256",
        "connected_gum_outcome",
        "connected_gum_protocol_sha256",
        "connected_gum_report_sha256",
        "connected_gum_runtime_sha256",
        # The decision-111 confirmatory replication's two protocol fields. The
        # population name and the registering decision number are what make a
        # recorded verdict traceable to the rule it was scored under; both are
        # text by nature, and neither is a measurement.
        "confirmatory_population",
        "confirmatory_registration_decision",
        # Direction agreement is three-valued, not boolean: "true", "false", or
        # "undefined" when either comparison has no direction to agree with,
        # which is the b = c = 0 case decision 79 names explicitly. Recording a
        # tie as False would read as a reversal, which it is not.
        "confirmatory_stage4_derived_mcnemar_direction_agrees",
        "confirmatory_stage4_gold_mcnemar_direction_agrees",
        "confirmatory_stage5_mcnemar_direction_agrees",
        "confirmatory_stage6_sign_test_direction_agrees",
        "bench_protocol_version",
        "bench_synth_chain_version",
        "bench_synth_cache_protocol",
        "bench_synth_context_arm_order",
        "bench_synth_position_generation",
        "bench_synth_openrouter_model",
        "bench_synth_openrouter_provider",
        "bench_synth_protocol_version",
        "bench_synth_warmup_policy",
        # The stipulated phone arm's protocol fields (lane ailm-synth-phoneme):
        # same string semantics as the letter arm's.
        "bench_floor_protocol_version",
        "bench_synthphone_chain_version",
        "bench_synthphone_cache_protocol",
        "bench_synthphone_arm_order",
        "bench_synthphone_position_generation",
        "bench_synthphone_protocol_version",
        "bench_synthphone_warmup_policy",
        "bench_logprob_drift_endpoint",
        "bench_ctx_chain_version",
        "bench_ctx_cache_protocol",
        "bench_ctx_arm_order",
        "bench_ctx_position_generation",
        "bench_ctx_example_draw_generation",
        "bench_ctx_producer_protocol_version",
        "bench_ctx_openrouter_model",
        "bench_ctx_openrouter_provider",
        "bench_ctx_protocol_version",
        "bench_ctx_warmup_policy",
        "bench_ctx_k_grid",
        "stage3_oov_stream_schema",
        "env_jax_version",
        "env_jaxlib_version",
        "env_pymdp_version",
        "env_platform",
        "env_machine",
        "generation_word_state_prefix_depth1",
        "generation_word_state_prefix_depth2",
        "generation_word_state_prefix_depth3",
        "generation_word_state_prefix_depth4",
        "cluster_stage2_g2p_cluster_unit",
        "cluster_stage2_g2p_effect_unit",
        "cluster_stage4_gold_cluster_unit",
        "cluster_stage4_gold_effect_unit",
        "cluster_stage4_derived_cluster_unit",
        "cluster_stage4_derived_effect_unit",
        "cluster_stage4_czech_cluster_unit",
        "cluster_stage4_czech_effect_unit",
        "cluster_stage6_logloss_cluster_unit",
        "cluster_stage6_logloss_effect_unit",
        "cluster_stage6_sign_cluster_unit",
        "cluster_stage6_sign_effect_unit",
        "confirmatory_stage6_clustered_cluster_unit",
        "confirmatory_stage6_clustered_effect_unit",
        "clause_word_cluster_unit",
        "worked_q_top1_symbol",
        "worked_q_top2_symbol",
        "worked_q_top3_symbol",
        # The family-grouped G2P arm's per-family row keys (decision 102): the
        # family identifier is a prefix string, not a measurement.
        "stage2_g2p_famgrp_rank1_family",
        "stage2_g2p_famgrp_rank2_family",
        "stage2_g2p_famgrp_rank3_family",
        "stage2_g2p_famgrp_rank4_family",
        "stage2_g2p_famgrp_rank5_family",
        "stage2_g2p_famgrp_rank6_family",
        "stage2_g2p_famgrp_rank7_family",
        "stage2_g2p_famgrp_rank8_family",
        "stage2_g2p_famgrp_rank9_family",
        "stage2_g2p_famgrp_rank10_family",
        "worked_t_top1_symbol",
        "worked_t_top2_symbol",
        "worked_t_top3_symbol",
        # The P2 aligned-G2P comparator's digest-bound artifact identity: a
        # sha256 is text by nature, and without this registration the shape
        # gate would read it as a failed numeric claim and block the render.
        "bench_synthg2p_comparator_artifact_sha256",
        # The P2 zero-shot probe arms bind to that same vendored artifact; the
        # probe recorder (`scripts/record_aligned_g2p_probe_claims.py`) writes
        # its own digest claim with the same text-by-nature shape, so it rides
        # the same registration or the shape gate reads it as a failed
        # numeric claim and blocks the render.
        "bench_synthg2p_probe_artifact_sha256",
        # The three per-probe model digests are sha256 strings by nature —
        # the same text-by-nature shape as the artifact digest above. Without
        # this registration the shape gate reads them as failed numeric
        # claims and blocks the render.
        "bench_synthg2p_probe_gemma3_model_digest",
        "bench_synthg2p_probe_qwen25_model_digest",
        "bench_synthg2p_probe_smollm_model_digest",
        # The OpenRouter transport's model name and digest are text by nature.
        "bench_synthg2p_openrouter_model",
        "bench_synthg2p_openrouter_model_digest",
    }
)
_REQUIRED_BOOLEAN_CLAIMS = frozenset(
    {
        "connected_gum_word_loss_is_infinite",
        # The decision-111 confirmatory replication. The sentinel records that
        # the held-back population has been scored and is spent. The four flags
        # per member are the recorded verdicts: whether the p-value crosses the
        # frozen threshold, whether Holm rejects over the substituted family,
        # whether the comparison was computable at all (b = c = 0 means it was
        # not), and whether the discordance points the way the development
        # result did. Each is a deterministic function of counts that are also
        # in the ledger, and tests/test_confirmatory_replication.py re-derives
        # every one of them from those counts.
        "confirmatory_run_completed_ud_ewt_test",
        "confirmatory_stage4_derived_mcnemar_crosses_frozen_threshold",
        "confirmatory_stage4_derived_mcnemar_holm_rejected",
        "confirmatory_stage4_derived_mcnemar_computable",
        "confirmatory_stage4_gold_mcnemar_crosses_frozen_threshold",
        "confirmatory_stage4_gold_mcnemar_holm_rejected",
        "confirmatory_stage4_gold_mcnemar_computable",
        "confirmatory_stage5_mcnemar_crosses_frozen_threshold",
        "confirmatory_stage5_mcnemar_holm_rejected",
        "confirmatory_stage5_mcnemar_computable",
        "confirmatory_stage6_sign_test_crosses_frozen_threshold",
        "confirmatory_stage6_sign_test_holm_rejected",
        "confirmatory_stage6_sign_test_computable",
        "channel_proper_loss_letter_evidence_weight_test_selected",
        "channel_proper_loss_letter_predictive_weight_test_selected",
        "channel_proper_loss_lexicon_evidence_weight_test_selected",
        "channel_proper_loss_lexicon_predictive_weight_test_selected",
        "channel_proper_loss_geometric_lexicon_weight_test_selected",
        "channel_proper_loss_fitted_weight_test_selected",
        "stat_stage6_familywise_confirmed",
        "stat_stage2_corrected_ci_excludes_half",
        "stat_stage4_corrected_ci_excludes_half",
        "stat_lexical_corrected_ci_excludes_half",
        "stat_coupling_harm_corrected_ci_excludes_half",
        "stat_holm_stage6_rejected",
        "formal_sorry_free",
        "env_jax_enable_x64",
        "cluster_stage4_czech_p_at_resolution_floor",
        "clause_word_cluster_loggain_p_at_resolution_floor",
        "clause_word_cluster_token_p_at_resolution_floor",
        # The wild-cluster bootstrap's verdict-agreement flags (decision 104):
        # a recorder-written boolean, not a number — without these the
        # render gate misreads every wild_{arm}_verdict_agrees as a numeric
        # claim and blocks the whole render.
        "wild_stage2_g2p_verdict_agrees",
        "wild_stage4_gold_verdict_agrees",
        "wild_stage4_derived_verdict_agrees",
        "wild_stage6_sign_verdict_agrees",
        "wild_stage6_logloss_verdict_agrees",
        "wild_stage4_czech_verdict_agrees",
        # The decision-115 held-back Stage 6 clustered arm. ``excludes_zero`` is
        # read by the Stage 6 finding status in scripts/audit_gates.py, so an
        # unregistered one would be misread as a numeric claim and block the
        # render; the rest are the two-sided resolution probe, the controls'
        # verdicts, and whether a development-sized effect is resolvable here.
        "confirmatory_stage6_clustered_centered_baseline_excludes_zero",
        "confirmatory_stage6_clustered_control_negative_excludes_zero",
        "confirmatory_stage6_clustered_control_positive_excludes_zero",
        "confirmatory_stage6_clustered_corrected_excludes_zero",
        "confirmatory_stage6_clustered_development_effect_resolved",
        "confirmatory_stage6_clustered_excludes_zero",
        "confirmatory_stage6_clustered_probe_above_detects",
        "confirmatory_stage6_clustered_probe_below_detects",
        # The P2 aligned-G2P comparator's tie verdict: the recorder writes the
        # comparison between the native engine's predictions and the trivial
        # closed-form T lookup as a boolean, and decision 116 lets a verdict
        # word come only from that recorded boolean.
        "bench_synthg2p_comparator_native_ties_t_lookup",
        # The P2 probe arms' per-probe contract-refusal booleans: the recorder
        # writes a single boolean per probe below the response-contract floor
        # (smollm: a base model without instruction following). Without this
        # registration the shape gate misreads the booleans as numeric claims
        # and blocks the render.
        "bench_synthg2p_probe_gemma3_contract_refused",
        "bench_synthg2p_probe_qwen25_contract_refused",
        "bench_synthg2p_probe_smollm_contract_refused",
    }
)
_REQUIRED_STRING_LIST_CLAIMS = frozenset(
    {
        "stat_family_survivors",
        "stat_holm_survivors",
        "stat_holm_only_survivors",
        "formal_axiom_names_beyond_core",
        "cluster_arms",
    }
)


class NumericClaimDomain(str, Enum):
    """The mutually exclusive semantic domains manuscript numbers may occupy."""

    BINARY_INDICATOR = "binary indicator"
    NONNEGATIVE_COUNT = "non-negative whole count"
    UNIT_INTERVAL = "unit interval"
    SIGNED_UNIT_INTERVAL = "signed unit interval"
    PERCENTAGE = "percentage"
    SIGNED_RELATIVE_PERCENTAGE = "signed relative percentage"
    NONNEGATIVE_RELATIVE_PERCENTAGE = "non-negative relative percentage"
    POSITIVE_REAL = "positive real"
    NONNEGATIVE_REAL = "non-negative real"
    FINITE_REAL = "finite real"


_COUNT_LABEL_RE = re.compile(
    r"(?:"
    r"(?:_count$|^phoneme_count_)|"
    r"_(?:b|c)(?:_[a-z0-9]+)?$|"
    r"_(?:positions|tokens|sentences|pairs|words|documents|entries|seeds|runs|"
    r"phonemes|morphemes|rungs|families|clusters|resamples|sequences|states|types|"
    r"categories|nodes|edges|roles|letters|initials|bins|points|files|calls|records|"
    r"parameters|failures|disagreements|agreements|increases|sign_changes|observations|"
    r"emissions|levels|cases_present|alphabet_size|vocab_size|inventory_size|family_size|"
    r"upos|"
    r"discordant_n|nonzero|stored|free|overlap|rank|index|stride|prefix_length|"
    r"max_word_len|largest_cluster|test_families|train_families|test_interrogatives|"
    r"train_interrogatives|words_per_run|hits|misses|seed|transitions|tables|"
    r"ops_total|nonzeros|verdicts|uses|specs|engaged_weights|largest_document|"
    r"top_logprobs|membership_changes|grid_size|identity_diffs|"
    r"null_mcnemar_discordant|lengths_evaluated|max_phase_spread_length|"
    r"headline_k|pool_total|pool_unique|window_long|window_short)$|"
    r"_(?:nodes|edges)(?:_[a-z0-9]+)?$|"
    r"^stage[45]_document_split_(?:train|test)$|"
    r"^stage4_(?:n_cases|cases_without_morphology)$|"
    r"^stage3_max_agent_vocab$|"
    r"^stage[56]_(?:documents_)?excluded_[a-z0-9]+$|"
    r"^stage6_(?:live_filter_)?sign_(?:positive|negative)$|"
    r"^generation_word_state_support_[a-z0-9]+$|"
    r"^case_word_sweep_.*_(?:improved|degraded|tied)$|"
    r"^clause_word_sweep_.*_(?:improved|degraded)$|"
    r"^g2p_learn_A_mcnemar_(?:learned_only|fixed_only)$|"
    r"^bench_(?:ctx_)?(?:synth_)?unparsable(?!_rate(?:_|$))(?:_|$)|"
    r"^bench_calib_unlocated_[a-z0-9]+$|"
    r"^bench_synth_(?:corpus|held_out)_chars$|"
    r"^stage3_oov_length_(?:buckets(?:_excluded_harder)?|min_bucket)$|"
    r"^(?:bench_floor_cap|fitted_max_prefix|formal_axioms_beyond_core)$"
    r")"
)
_P_VALUE_LABEL_RE = re.compile(r"(?:_p(?:_|$)|_p_value(?:_|$))")
_BINARY_LABEL_RE = re.compile(
    r"(?:_(?:passes|confirmed|selected|exists|is_interior|confirms_harm)$|"
    r"^bench_calib_(?:rule_(?:measurable|degenerate|agrees)|measurable)_[a-z0-9]+$|"
    r"^coupling_supported$|"
    r"^clause_word_(?:monotone_decline|real_beats_[a-z0-9_]+)$)"
)
_UNIT_INTERVAL_COMPONENT_RE = re.compile(
    r"(?:^|_)(?:tv|tvd|ece|auc|prob|probability|fraction|share|agreement|power|mass|"
    r"confidence)(?:_|$)"
)
_BENCHMARK_UNIT_INTERVAL_RE = re.compile(
    r"^bench_.*(?:_(?:top1|exact|pos_acc)(?:_|$)|_(?:position|exact)_accuracy(?:_|$)|"
    r"_(?:uniform_)?(?:position|exact)_floor(?:_|$)|_wilson_(?:low|high)(?:_|$)|"
    r"_(?:captured|confidence|target)(?:_|$)|_unparsable_rate(?:_|$)|"
    r"_(?:pos_(?:long|mid|short)|nonalpha_acc|uniform_baseline|spread)(?:_|$)|"
    r"_logprob_drift_max$|_floor_(?:declared_floor|grid_(?:low|high|step))$)"
)
_BENCHMARK_SIGNED_UNIT_INTERVAL_RE = re.compile(r"^bench_.*_delta(?:_|$)")
_PERCENT_SCALE_STAGE4_RE = re.compile(
    r"^(?:stage4_|ewt_boundary_stage4_).*(?:agreement|baseline|majority|smoothing|"
    r"viterbi|cf_oracle)$"
)
_POSITIVE_REAL_RE = re.compile(
    r"(?:_(?:perplexity|ppl|denominator|pseudocount|tolerance)|^(?:stage[123456]_)?"
    r"uniform_baseline$)"
)
_NONNEGATIVE_REAL_RE = re.compile(
    r"(?:_(?:nll|logloss|brier|entropy_nats|abs_err|abs_error|abs_gap|abs_dev|divergence|"
    r"variance_inflation|drift|spread|ratio|factor|memory_bytes|ops_per_position|"
    r"ops_per_token|gb|mb|tb|bytes|epsilon|gamma|exponent|distinct)$)"
)
_SIGNED_RELATIVE_PERCENTAGE_CLAIMS = frozenset({"stage5_privileged_label_gain_pct"})
_NONNEGATIVE_RELATIVE_PERCENTAGE_CLAIMS = frozenset(
    {"stage3_scale_gain_rise_pct", "stage3_oov_gap_change_under_matching_pct"}
)
_SIGNED_UNIT_INTERVAL_CLAIMS = frozenset(
    {
        "generation_tv_improvement",
        "clause_word_cluster_token_net_share",
        "clause_word_cluster_token_ci_low",
        "clause_word_cluster_token_ci_high",
        "cluster_stage6_sign_effect",
        "cluster_stage6_sign_ci_low",
        "cluster_stage6_sign_ci_high",
        "stage6_opening_interrogative_shift",
        "stage6_middle_interrogative_shift",
        "stage6_closing_interrogative_shift",
        "stage6_scrambled_opening_shift",
    }
)
_UNIT_INTERVAL_EXACT_CLAIMS = frozenset(
    {
        "stage5_average_precision",
        "stage6_transition_above_diagonal",
    }
)


def _numeric_claim_domain(label: str) -> NumericClaimDomain:
    """Classify one numeric claim by meaning, with exactly one result.

    This is an exclusive decision tree rather than a collection of independent
    suffix assertions. A model-qualified label such as
    ``bench_calib_ece_gemma3`` therefore cannot evade the ECE domain merely
    because the metric name is not the final suffix, and a count such as
    ``stage5_auc_bootstrap_documents`` cannot be mistaken for an AUC.
    """
    if re.fullmatch(r"bench_ctx_gap_closed_ratio_[a-z0-9]+", label):
        return NumericClaimDomain.FINITE_REAL
    if label in _SIGNED_RELATIVE_PERCENTAGE_CLAIMS:
        return NumericClaimDomain.SIGNED_RELATIVE_PERCENTAGE
    if label in _NONNEGATIVE_RELATIVE_PERCENTAGE_CLAIMS:
        return NumericClaimDomain.NONNEGATIVE_RELATIVE_PERCENTAGE
    if label.endswith("_pct"):
        return NumericClaimDomain.PERCENTAGE
    if (
        (label.endswith("_accuracy") and not label.startswith("bench_"))
        or _PERCENT_SCALE_STAGE4_RE.fullmatch(label) is not None
        or re.fullmatch(r"stage2_g2p_(?:smoothing|viterbi)_ceiling", label) is not None
        or (label.startswith("lexical_sweep_") and label.endswith("_scrambled"))
    ):
        return NumericClaimDomain.PERCENTAGE
    if _COUNT_LABEL_RE.search(label) is not None:
        return NumericClaimDomain.NONNEGATIVE_COUNT
    if _BINARY_LABEL_RE.search(label) is not None:
        return NumericClaimDomain.BINARY_INDICATOR
    if (
        label in _SIGNED_UNIT_INTERVAL_CLAIMS
        or "_ece_signed_" in label
        or label.endswith("_icc")
        or _BENCHMARK_SIGNED_UNIT_INTERVAL_RE.search(label) is not None
    ):
        return NumericClaimDomain.SIGNED_UNIT_INTERVAL
    if (
        label in _UNIT_INTERVAL_EXACT_CLAIMS
        or (_P_VALUE_LABEL_RE.search(label) is not None and not label.endswith("_log10_p"))
        or label.endswith("_alpha")
        or _UNIT_INTERVAL_COMPONENT_RE.search(label) is not None
        or _BENCHMARK_UNIT_INTERVAL_RE.search(label) is not None
        or label.endswith("_favored_ratio")
    ):
        return NumericClaimDomain.UNIT_INTERVAL
    if _POSITIVE_REAL_RE.search(label) is not None and "delta" not in label:
        return NumericClaimDomain.POSITIVE_REAL
    if _NONNEGATIVE_REAL_RE.search(label) is not None:
        return NumericClaimDomain.NONNEGATIVE_REAL
    return NumericClaimDomain.FINITE_REAL


def _validate_numeric_claim_domain(label: str, value: float, domain: NumericClaimDomain) -> None:
    """Enforce the selected semantic domain for one finite numeric value."""
    if domain is NumericClaimDomain.BINARY_INDICATOR:
        valid = value in (0.0, 1.0) and value == int(value)
        requirement = "must be the binary indicator 0 or 1"
    elif domain is NumericClaimDomain.NONNEGATIVE_COUNT:
        valid = value >= 0.0 and value == int(value)
        requirement = "must be a non-negative whole number"
    elif domain is NumericClaimDomain.UNIT_INTERVAL:
        valid = 0.0 <= value <= 1.0
        requirement = "must lie in [0, 1]"
    elif domain is NumericClaimDomain.SIGNED_UNIT_INTERVAL:
        valid = -1.0 <= value <= 1.0
        requirement = "must lie in [-1, 1]"
    elif domain is NumericClaimDomain.PERCENTAGE:
        valid = 0.0 <= value <= 100.0
        requirement = "must lie in [0, 100]"
    elif domain is NumericClaimDomain.SIGNED_RELATIVE_PERCENTAGE:
        valid = value <= 100.0
        requirement = "must be at most 100 (with no finite lower bound)"
    elif domain is NumericClaimDomain.NONNEGATIVE_RELATIVE_PERCENTAGE:
        valid = value >= 0.0
        requirement = "must be at least 0"
    elif domain is NumericClaimDomain.POSITIVE_REAL:
        valid = value > 0.0
        requirement = "must be positive"
    elif domain is NumericClaimDomain.NONNEGATIVE_REAL:
        valid = value >= 0.0
        requirement = "must be non-negative"
    else:
        valid = True  # finiteness is checked before this function is called
        requirement = "must be finite"
    if not valid:
        raise ManuscriptVariableError(
            f"required manuscript {domain.value} claim {label!r} {requirement}, got {value!r}"
        )


def _validate_required_claim_values(claims: Mapping[str, Any]) -> None:
    """Type and bound every claim mechanically exposed to manuscript prose.

    The ledger schema deliberately permits JSON scalars because it serves more
    than the manuscript.  The manuscript contract is narrower: apart from an
    explicit inventory of textual, boolean, and string-list claims, every
    value is quantitative.  This prevents a syntactically valid ledger
    from rendering strings, booleans, impossible percentages, or negative
    sample counts into scientific prose.
    """
    for label, entry in claims.items():
        value = entry["value"]
        if label in _REQUIRED_STRING_CLAIMS:
            if not isinstance(value, str) or not value.strip():
                raise ManuscriptVariableError(
                    f"required manuscript claim {label!r} must be non-empty text, got {value!r}"
                )
            continue
        if label in _REQUIRED_BOOLEAN_CLAIMS:
            if not isinstance(value, bool):
                raise ManuscriptVariableError(
                    f"required manuscript claim {label!r} must be boolean, got {value!r}"
                )
            continue
        if label in _REQUIRED_STRING_LIST_CLAIMS:
            if (
                not isinstance(value, list)
                or any(not isinstance(item, str) or not item.strip() for item in value)
                or len(value) != len(set(value))
            ):
                raise ManuscriptVariableError(
                    f"required manuscript claim {label!r} must be a duplicate-free list "
                    f"of non-empty strings, got {value!r}"
                )
            continue

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ManuscriptVariableError(
                f"required manuscript claim {label!r} must be numeric, got {value!r}"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ManuscriptVariableError(
                f"required manuscript claim {label!r} must be finite, got {value!r}"
            )
        _validate_numeric_claim_domain(label, numeric, _numeric_claim_domain(label))
