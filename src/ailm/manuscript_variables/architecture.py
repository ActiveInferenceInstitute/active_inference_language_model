"""Calibration, lexical-sweep, fitted/stage-5, and architecture-tally producers."""

from __future__ import annotations

import inspect
import re
from collections.abc import Mapping
from typing import (
    Any,
)

from ailm.benchmarks import exact_mcnemar_p
from ailm.benchmarks.constants import SERVED_PROBES
from ailm.channel_registry import (
    ChannelEvidenceRule,
    ChannelOutcome,
    ChannelSourceKind,
    channel_by_key,
)
from ailm.value_formatting import (
    ManuscriptVariableError,
)
from ailm.value_formatting import (
    format_ledger_value as _format_value,
)

from ._channels import (
    LEVEL_QUALIFYING_VERDICT_FLAGS,
    canonical_stage5_passes,
    classify_cross_level_channels,
    level_six_qualification_limits,
    qualified_verdict_flags,
)
from ._claims import (
    _ClaimReaders,
    _LexicalSweep,
    require_close,
)
from ._formatting import (
    _count_word,
    _quantified_clause,
)
from ._io import _StrictVariableMap
from ._stats import (
    PARAMETER_MODEL_DISPLAY_NAMES,
    _whole,
)


def _produce_calibration_rows(variables: _StrictVariableMap, readers: _ClaimReaders) -> None:
    """The calibration table from the registered probe inventory, and the
    Decision 93 removal of proper-score display tokens for withheld probes.
    """
    num = readers.num
    probability = readers.probability

    #: The calibration table is generated from the registered probe inventory.
    #: Percent-valued columns use a consistent display scale and proper scores
    #: use three decimal places; the ledger continues to hold the exact values.
    _calibration_systems = (("AILM", "ailm"),) + tuple(
        (probe.display_name, probe.key) for probe in SERVED_PROBES
    )
    _calibration_rows: list[str] = []
    for _display_name, _key in _calibration_systems:
        # Decision 93: the table reports only systems the direct dual-rule gate
        # admits. A probe the gate withholds has no row here — its figures stay
        # in the ledger, and the prose states the withholding with the recorded
        # divergences rather than a row a reader would rank.
        if _key != "ailm" and not num(f"bench_calib_measurable_{_key}"):
            continue
        _brier_label = f"bench_calib_brier_{_key}"
        _nll_label = f"bench_calib_nll_{_key}"
        _brier = num(_brier_label)
        _nll = num(_nll_label)
        if _brier < 0.0 or _nll < 0.0:
            raise ManuscriptVariableError(
                f"calibration proper scores must be non-negative for {_key}: "
                f"Brier={_brier}, NLL={_nll}"
            )
        _calibration_rows.append(
            "| "
            + " | ".join(
                (
                    _display_name,
                    f"{probability(f'bench_calib_top1_{_key}') * 100:.2f}%",
                    f"{probability(f'bench_calib_confidence_{_key}') * 100:.2f}%",
                    f"{probability(f'bench_calib_ece_{_key}') * 100:.2f}%",
                    f"{_brier:.3f}",
                    f"{_nll:.3f}",
                )
            )
            + " |"
        )
    variables["BENCH_CALIB_TABLE_ROWS"] = "\n".join(_calibration_rows)

    # Decision 93: a probe the direct dual-rule gate withholds has its
    # calibration proper-score display tokens removed, not merely unquoted. The
    # ledger keeps the measured values, but a resolvable BENCH_CALIB_ECE_<MODEL>
    # token would invite exactly the ranking the gate exists to prevent.
    import ailm.benchmarks.constants as _bench_constants

    for _probe_key in _bench_constants.PROBE_MODEL_TAGS:
        if bool(num(f"bench_calib_measurable_{_probe_key}")):
            continue
        for _metric in ("BENCH_CALIB_ECE_", "BENCH_CALIB_BRIER_"):
            _token_name = f"{_metric}{_probe_key.upper()}"
            variables.pop(_token_name, None)
            variables.pop(f"{_token_name}_RUN", None)
            variables.pop(f"{_token_name}_SOURCE", None)


def _produce_lexical_sweep(
    variables: _StrictVariableMap, claims: Mapping[str, Any], readers: _ClaimReaders
) -> _LexicalSweep:
    """The lexical sweep table from the complete recorded per-alpha family,
    with its coordinate, optimum, and summary reconciliations; returns the
    sweep state the null-bounds producer consumes for the high-alpha arm.
    """
    num = readers.num
    probability = readers.probability
    percentage = readers.percentage
    claim_text = readers.claim_text

    # Likewise, the lexical sweep body is generated from the complete recorded
    # per-alpha family.  The row labels are parsed from the claim labels that own
    # their results, so the source manuscript contains no hand-maintained grid.
    _lexical_labels = sorted(
        label
        for label in claims
        if label.startswith("lexical_sweep_alpha_") and label.endswith("_accuracy")
    )
    from ailm.levels.lexical_channel import LEXICAL_SWEEP_ALPHAS

    try:
        _declared_lexical_alphas = tuple(
            sorted(
                float(
                    label.removeprefix("lexical_sweep_alpha_")
                    .removesuffix("_accuracy")
                    .replace("_", ".")
                )
                for label in _lexical_labels
            )
        )
    except ValueError as exc:
        raise ManuscriptVariableError(
            f"cannot parse lexical sweep coordinates from {_lexical_labels!r}"
        ) from exc
    if _declared_lexical_alphas != LEXICAL_SWEEP_ALPHAS:
        raise ManuscriptVariableError(
            "recorded lexical sweep coordinates do not match the canonical protocol; "
            f"expected {LEXICAL_SWEEP_ALPHAS}, got {_declared_lexical_alphas}"
        )
    _lexical_prefix_by_alpha: dict[float, str] = {}
    if _lexical_labels:
        _lexical_positions = _whole("lexical_eval_positions", num("lexical_eval_positions"))
        if _lexical_positions <= 0:
            raise ManuscriptVariableError(
                f"lexical_eval_positions must be positive, got {_lexical_positions}"
            )
        _lexical_anchor = percentage("lexical_anchor_accuracy")
        _lexical_rows: list[tuple[float, str]] = []
        _lexical_metrics: dict[float, dict[str, float | int]] = {}
        for _label in _lexical_labels:
            _encoded = _label.removeprefix("lexical_sweep_alpha_").removesuffix("_accuracy")
            try:
                _alpha = float(_encoded.replace("_", "."))
            except ValueError as exc:
                raise ManuscriptVariableError(
                    f"cannot parse lexical sweep coordinate from claim {_label!r}"
                ) from exc
            _prefix = _label.removesuffix("_accuracy")
            if _alpha in _lexical_prefix_by_alpha:
                raise ManuscriptVariableError(
                    f"duplicate lexical sweep coordinate {_alpha} in "
                    f"{_lexical_prefix_by_alpha[_alpha]!r} and {_prefix!r}"
                )
            _lexical_prefix_by_alpha[_alpha] = _prefix
            _alpha_text = _encoded.replace("_", ".")
            _accuracy_value = percentage(_label)
            _delta_label = f"{_prefix}_delta"
            _delta_value = num(_delta_label)
            _scrambled_label = f"{_prefix}_scrambled"
            _scrambled_value = percentage(_scrambled_label)
            _b_label = f"{_prefix}_mcnemar_b"
            _c_label = f"{_prefix}_mcnemar_c"
            _p_label = f"{_prefix}_mcnemar_p"
            _b = _whole(_b_label, num(_b_label))
            _c = _whole(_c_label, num(_c_label))
            if _b < 0 or _c < 0 or _b + _c > _lexical_positions:
                raise ManuscriptVariableError(
                    f"lexical discordance {_b_label}/{_c_label} is impossible for "
                    f"n={_lexical_positions}: b={_b}, c={_c}"
                )
            _expected_p = float(f"{exact_mcnemar_p(_b, _c):.4g}")
            require_close(_p_label, probability(_p_label), _expected_p)
            require_close(
                _delta_label,
                _delta_value,
                round(_accuracy_value - _lexical_anchor, 3),
            )
            require_close(
                _delta_label,
                _delta_value,
                round(100.0 * (_b - _c) / _lexical_positions, 3),
            )
            _lexical_metrics[_alpha] = {
                "accuracy": _accuracy_value,
                "delta": _delta_value,
                "scrambled": _scrambled_value,
                "b": _b,
                "c": _c,
                "p": probability(_p_label),
            }
            _accuracy_text = claim_text(_label)
            _delta_text = claim_text(_delta_label)
            _scrambled = claim_text(_scrambled_label)
            if _alpha == 0.0:
                _row = f"| {_alpha_text} — the anchor | {_accuracy_text}% | — | identical by construction |"
            elif _alpha == num("lexical_best_alpha"):
                _row = (
                    f"| **{_alpha_text} — best** | **{_accuracy_text}%** | "
                    f"**+{_delta_text}** | {_scrambled}% |"
                )
            else:
                _row = f"| {_alpha_text} | {_accuracy_text}% | {_delta_text} | {_scrambled}% |"
            _lexical_rows.append((_alpha, _row))
        _lexical_rows.sort()
        _lexical_alphas = tuple(alpha for alpha, _row in _lexical_rows)
        if any(
            current <= previous
            for previous, current in zip(_lexical_alphas[:-1], _lexical_alphas[1:], strict=True)
        ):
            raise ManuscriptVariableError(
                f"lexical sweep coordinates must be unique and increasing, got {_lexical_alphas}"
            )
        if _lexical_alphas != LEXICAL_SWEEP_ALPHAS:
            raise ManuscriptVariableError(
                "recorded lexical sweep coordinates do not match the canonical protocol; "
                f"expected {LEXICAL_SWEEP_ALPHAS}, got {_lexical_alphas}"
            )
        _best_alpha = num("lexical_best_alpha")
        if _best_alpha not in _lexical_alphas:
            raise ManuscriptVariableError(
                f"lexical_best_alpha {_best_alpha} is not a recorded sweep coordinate "
                f"{_lexical_alphas}"
            )
        _maximum_accuracy = max(float(metrics["accuracy"]) for metrics in _lexical_metrics.values())
        _maximizers = tuple(
            alpha
            for alpha, metrics in _lexical_metrics.items()
            if float(metrics["accuracy"]) == _maximum_accuracy
        )
        if len(_maximizers) != 1:
            raise ManuscriptVariableError(
                "lexical sweep must have one pre-interpretable optimum; "
                f"accuracy maximizers are {_maximizers}"
            )
        _selected_alpha = _maximizers[0]
        if _best_alpha != _selected_alpha:
            raise ManuscriptVariableError(
                f"lexical_best_alpha {_best_alpha} is not the unique accuracy argmax "
                f"{_selected_alpha}"
            )
        _best_metrics = _lexical_metrics[_selected_alpha]
        for _summary, _field in (
            ("lexical_best_accuracy", "accuracy"),
            ("lexical_best_margin_pts", "delta"),
            ("lexical_best_scrambled_accuracy", "scrambled"),
            ("lexical_best_mcnemar_b", "b"),
            ("lexical_best_mcnemar_c", "c"),
            ("lexical_best_mcnemar_p", "p"),
        ):
            require_close(_summary, num(_summary), float(_best_metrics[_field]))
        require_close(
            "lexical_best_vs_scrambled_pts",
            num("lexical_best_vs_scrambled_pts"),
            round(
                float(_best_metrics["accuracy"]) - float(_best_metrics["scrambled"]),
                3,
            ),
        )
        _expected_interior = int(0.0 < _selected_alpha < max(_lexical_alphas))
        if (
            _whole("lexical_optimum_is_interior", num("lexical_optimum_is_interior"))
            != _expected_interior
        ):
            raise ManuscriptVariableError(
                "lexical_optimum_is_interior disagrees with the selected sweep coordinate"
            )
        variables["LEXICAL_SWEEP_TABLE_ROWS"] = "\n".join(row for _alpha, row in _lexical_rows)
    else:
        raise ManuscriptVariableError("no lexical sweep accuracy family is recorded")

    return _LexicalSweep(prefix_by_alpha=_lexical_prefix_by_alpha, metrics=_lexical_metrics)


def _produce_fitted_and_stage5(
    variables: _StrictVariableMap, claims: Mapping[str, Any], readers: _ClaimReaders
) -> None:
    """The fitted-hyperparameter census, Level 5's joint state space, and the
    recorded-interval reading of Stage 5's oracle gains.
    """
    num = readers.num

    #: How many of the fitted hyperparameters moved off their hand-set default.
    #: The defaults are the signature of `build_lexical_message` plus the
    #: shipped exponent, so this compares the fit against the code rather than
    #: against a remembered number. The paper said two; this counts.
    from ailm.levels.lexical_channel import build_lexical_message

    _defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(build_lexical_message).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    _fitted = {
        "max_prefix": "fitted_max_prefix",
        "min_count": "fitted_min_count",
        "smoothing": "fitted_smoothing",
    }
    _comparable = {
        name: label for name, label in _fitted.items() if name in _defaults and label in claims
    }
    _moved = sum(1 for name, label in _comparable.items() if num(label) != float(_defaults[name]))
    _compared = len(_comparable)
    # The pooling exponent is the fourth value fitted. Its hand-chosen
    # counterpart is the geometric sweep's selected exponent, not the linear
    # channel's alpha — the two are different rules, and the identity that
    # settles it is that fitted_default_accuracy equals
    # geometric_lexicon_best_accuracy exactly, so the "hand-chosen"
    # configuration the paper compares against IS the geometric best.
    if {"fitted_exponent", "geometric_lexicon_best_exponent"} <= set(claims):
        _compared += 1
        _moved += int(num("fitted_exponent") != num("geometric_lexicon_best_exponent"))
    if _compared:
        variables["FITTED_VALUES_COMPARED"] = str(_compared)
        variables["FITTED_VALUES_MOVED"] = str(_moved)

    #: Level 5's joint state space: one state per (type, case-role) pair. Both
    #: factors are code-owned inventories, so this is arithmetic over constants
    #: rather than a measurement — but it was typed as "sixteen", which is a
    #: number that survives a change to either inventory.
    from ailm.cases import CaseRole
    from ailm.complexity import FLOAT_BYTES
    from ailm.config import DEFAULT_LEVEL_CONFIG
    from ailm.learning import CONJUGATE_LEARNING_RATE
    from ailm.levels.clause import CLAUSE_TYPES

    variables["STAGE5_N_TYPES"] = str(len(CLAUSE_TYPES))
    variables["STAGE5_JOINT_STATES"] = str(len(CLAUSE_TYPES) * len(CaseRole))

    # Decision 83's qualitative reading follows the recorded interval endpoints.
    # Keeping "spans zero" out of authored prose means a recorder replay that
    # resolves either scale cannot leave the manuscript calling it unresolved.
    _stage5_sentence_status = readers.stage5_interval_status(
        "stage5_oracle_gain_sentence_ci_low", "stage5_oracle_gain_sentence_ci_high"
    )
    _stage5_position_status = readers.stage5_interval_status(
        "stage5_oracle_gain_position_ci_low", "stage5_oracle_gain_position_ci_high"
    )
    variables["STAGE5_ORACLE_GAIN_SENTENCE_INTERVAL_STATUS"] = _stage5_sentence_status
    variables["STAGE5_ORACLE_GAIN_POSITION_INTERVAL_STATUS"] = _stage5_position_status
    variables["STAGE5_ORACLE_GAIN_RESOLUTION"] = (
        "unresolved on both reported scales"
        if _stage5_sentence_status == "spans zero" and _stage5_position_status == "spans zero"
        else "resolved on at least one reported scale"
    )
    defaults = DEFAULT_LEVEL_CONFIG
    variables["MODEL_DEFAULT_SMOOTHING"] = _format_value(defaults.smoothing)
    variables["MODEL_DEFAULT_EMISSION_SMOOTHING"] = _format_value(
        defaults.resolved_emission_smoothing
    )
    variables["MODEL_FLOAT_BYTES"] = str(FLOAT_BYTES)
    variables["MODEL_FLOAT_BYTES_WORD"] = _count_word(FLOAT_BYTES)
    variables["ONLINE_LEARNING_RATE"] = _format_value(CONJUGATE_LEARNING_RATE)


def _produce_architecture_tallies(
    variables: _StrictVariableMap, claims: Mapping[str, Any], readers: _ClaimReaders
) -> None:
    """The architecture and verdict tallies stated in natural-language prose:
    the GNN registry census, the receiver-intervention classification, the
    qualified-verdict count, and the benchmark's familywise alpha ratios.
    """
    num = readers.num
    claim_text = readers.claim_text

    # The GNN registry has one or more models at a linguistic level (L2 has two
    # and L1 has an auxiliary generator), so the level count is the distinct
    # numeric level identifiers, not the number of agents or files.
    from ailm.levels.narrative import N_PHASES
    from ailm.parameters import GNN_AUXILIARY_SPECS, GNN_SPEC_FILES

    _registry_keys = set(GNN_SPEC_FILES)
    _display_keys = set(PARAMETER_MODEL_DISPLAY_NAMES)
    if _registry_keys != _display_keys:
        raise ManuscriptVariableError(
            "the primary GNN registry must exactly match the manuscript parameter-table "
            f"inventory; missing displays={sorted(_registry_keys - _display_keys)}, "
            f"stale displays={sorted(_display_keys - _registry_keys)}"
        )
    if len(set(GNN_SPEC_FILES.values())) != len(GNN_SPEC_FILES):
        raise ManuscriptVariableError("the primary GNN registry contains duplicate spec paths")
    _level_ids: set[int] = set()
    for _model_key in GNN_SPEC_FILES:
        _match = re.fullmatch(r"L(\d+)(?:[A-Z])?_[a-z][a-z0-9_]*", _model_key)
        if _match is None:
            raise ManuscriptVariableError(
                f"primary GNN registry key {_model_key!r} does not name a complete level model"
            )
        _level_ids.add(int(_match.group(1)))
    if not _level_ids or _level_ids != set(range(1, max(_level_ids) + 1)):
        raise ManuscriptVariableError(
            f"canonical GNN level registry is not contiguous from L1: {sorted(_level_ids)}"
        )
    _built_level_count = len(_level_ids)
    _model_agent_count = len(GNN_SPEC_FILES)
    _auxiliary_model_count = len(GNN_AUXILIARY_SPECS)
    for _claim_label in ("params_levels", "params_gnn_specs"):
        _recorded_model_count = _whole(_claim_label, num(_claim_label))
        if _recorded_model_count != _model_agent_count:
            raise ManuscriptVariableError(
                f"{_claim_label} records {_recorded_model_count} primary models but "
                f"GNN_SPEC_FILES registers {_model_agent_count}"
            )
    _parameter_rows: list[str] = []
    for _model_key, _spec_path in GNN_SPEC_FILES.items():
        _claim_prefix = _model_key.split("_", 1)[0].lower()
        _parameter_rows.append(
            f"| {PARAMETER_MODEL_DISPLAY_NAMES[_model_key]} | `{_spec_path}` | "
            f"{claim_text(f'params_{_claim_prefix}_stored')} | "
            f"{claim_text(f'params_{_claim_prefix}_nonzero')} | "
            f"{claim_text(f'params_{_claim_prefix}_free')} |"
        )
    variables["PARAMETER_TABLE_ROWS"] = "\n".join(_parameter_rows)
    variables["BUILT_LEVEL_COUNT"] = str(_built_level_count)
    variables["BUILT_LEVEL_COUNT_WORD"] = _count_word(_built_level_count)
    variables["MODEL_AGENT_COUNT"] = str(_model_agent_count)
    variables["MODEL_AGENT_COUNT_WORD"] = _count_word(_model_agent_count)
    variables["AUXILIARY_MODEL_COUNT"] = str(_auxiliary_model_count)
    variables["NARRATIVE_PHASE_COUNT"] = str(N_PHASES)
    variables["NARRATIVE_PHASE_COUNT_WORD"] = _count_word(N_PHASES)

    # The reported receiver interventions are different experiments, not the
    # cells of the later source-by-site factorial. Classify each from the
    # measurement that actually supports the sentence in Results, and derive
    # source-kind counts from the same registry so the oracle arm cannot be
    # silently promoted to model output in prose.
    _channel_outcomes = classify_cross_level_channels(claims)
    _channels = channel_by_key()
    _expected_channel_outcomes = {
        channel.key: channel.expected_outcome.value for channel in _channels.values()
    }
    if _channel_outcomes != _expected_channel_outcomes:
        changed = {
            name: outcome
            for name, outcome in _channel_outcomes.items()
            if outcome != _expected_channel_outcomes[name]
        }
        raise ManuscriptVariableError(
            "the recorded receiver-intervention classification changed; rewrite the "
            f"interpretation before rendering (changed: {changed})"
        )
    _channel_count = len(_channel_outcomes)
    _helping_count = sum(outcome == "helps" for outcome in _channel_outcomes.values())
    _harming_count = sum(outcome == "harms" for outcome in _channel_outcomes.values())
    _prior_only_count = sum(
        outcome == ChannelOutcome.PRIOR_ONLY.value for outcome in _channel_outcomes.values()
    )
    _mixed_direction_count = sum(
        outcome == ChannelOutcome.MIXED_DIRECTION.value for outcome in _channel_outcomes.values()
    )
    variables["CHANNEL_COUNT"] = str(_channel_count)
    variables["CHANNEL_COUNT_WORD"] = _count_word(_channel_count)
    variables["CHANNEL_HELPING_COUNT"] = str(_helping_count)
    variables["CHANNEL_HELPING_COUNT_WORD"] = _count_word(_helping_count)
    variables["CHANNEL_HARMING_COUNT"] = str(_harming_count)
    variables["CHANNEL_HARMING_COUNT_WORD"] = _count_word(_harming_count)
    variables["CHANNEL_PRIOR_ONLY_COUNT"] = str(_prior_only_count)
    variables["CHANNEL_PRIOR_ONLY_COUNT_WORD"] = _count_word(_prior_only_count)
    variables["CHANNEL_MIXED_DIRECTION_COUNT"] = str(_mixed_direction_count)
    variables["CHANNEL_MIXED_DIRECTION_COUNT_WORD"] = _count_word(_mixed_direction_count)
    # The separately measured interventions are the registry's scored-estimand
    # members. The two registered case-word structural seams (the L4→L3 bridge
    # and its privileged-oracle diagnostic) record no estimand, so a prose
    # count of "measured" channels must exclude them by derivation rather than
    # by a hand-typed exception beside the total.
    _channel_separately_measured_count = sum(
        1
        for _channel in _channels.values()
        if _channel.evidence_rule
        not in (
            ChannelEvidenceRule.CASE_WORD_BRIDGE,
            ChannelEvidenceRule.CASE_WORD_ORACLE_DIAGNOSTIC,
        )
    )
    variables["CHANNEL_SEPARATELY_MEASURED_COUNT"] = str(_channel_separately_measured_count)
    variables["CHANNEL_SEPARATELY_MEASURED_COUNT_WORD"] = _count_word(
        _channel_separately_measured_count
    )
    _channel_case_word_seam_count = sum(
        1
        for _channel in _channels.values()
        if _channel.evidence_rule
        in (
            ChannelEvidenceRule.CASE_WORD_BRIDGE,
            ChannelEvidenceRule.CASE_WORD_ORACLE_DIAGNOSTIC,
        )
    )
    variables["CHANNEL_CASE_WORD_SEAM_COUNT"] = str(_channel_case_word_seam_count)
    variables["CHANNEL_CASE_WORD_SEAM_COUNT_WORD"] = _count_word(_channel_case_word_seam_count)
    _outcome_clauses = (
        _quantified_clause(
            _helping_count,
            singular_verb="helps",
            plural_verb="help",
            complement="on its declared result axis",
        ),
        _quantified_clause(
            _harming_count,
            singular_verb="harms",
            plural_verb="harm",
            complement="its receiver",
        ),
        _quantified_clause(
            _prior_only_count,
            singular_verb="moves",
            plural_verb="move",
            complement="a prior without a delivered end-to-end effect estimate",
        ),
        _quantified_clause(
            _mixed_direction_count,
            singular_verb="is",
            plural_verb="are",
            complement="mixed-direction",
        ),
    )
    variables["CHANNEL_OUTCOME_SUMMARY"] = (
        ", ".join(_outcome_clauses[:-1]) + ", and " + _outcome_clauses[-1]
    )
    _model_output_count = sum(
        channel.source_kind is ChannelSourceKind.MODEL_OUTPUT for channel in _channels.values()
    )
    _oracle_intervention_count = sum(
        channel.source_kind is ChannelSourceKind.PRIVILEGED_ORACLE for channel in _channels.values()
    )
    if _model_output_count + _oracle_intervention_count != _channel_count:
        raise ManuscriptVariableError(
            "channel source-kind census does not exhaust the receiver interventions"
        )
    variables["MODEL_OUTPUT_INTERVENTION_COUNT"] = str(_model_output_count)
    variables["MODEL_OUTPUT_INTERVENTION_COUNT_WORD"] = _count_word(_model_output_count)
    variables["PRIVILEGED_ORACLE_INTERVENTION_COUNT"] = str(_oracle_intervention_count)
    variables["PRIVILEGED_ORACLE_INTERVENTION_COUNT_WORD"] = _count_word(_oracle_intervention_count)
    variables["CHANNEL_SOURCE_SUMMARY"] = " and ".join(
        (
            _quantified_clause(
                _model_output_count,
                singular_verb="has",
                plural_verb="have",
                complement="a model-output source",
            ),
            _quantified_clause(
                _oracle_intervention_count,
                singular_verb="has",
                plural_verb="have",
                complement="a privileged-oracle source",
            ),
        )
    )
    _comprehension_channel_count = sum(channel.comprehension_only for channel in _channels.values())
    variables["COMPREHENSION_CHANNEL_COUNT"] = str(_comprehension_channel_count)
    variables["COMPREHENSION_CHANNEL_COUNT_WORD"] = _count_word(_comprehension_channel_count)

    # "Qualified verdict" means a built level whose recorded verdict carries an
    # inferential qualifier: since decision 114 that is Level 6 alone, whose
    # document-clustered *primary* interval includes zero. It is not inferred
    # from a control label's existence or from the obsolete repository-family
    # classification of a secondary sign statistic.
    #
    # The clause qualification left this count when decision 114 made the
    # widened-observation arm the canonical Level 5. The guard behind it is
    # still evaluated above, because it now protects the retained case-only
    # ablation's non-passing record; it simply no longer describes a level.
    # Counting the canonical Level 5 as passing without checking it would make
    # the abstract's headline count an assumption, so it is asserted here
    # against the same ledger labels the audit reads, fail-closed.
    _qualified_flags = qualified_verdict_flags(claims)
    if not canonical_stage5_passes(claims):
        raise ManuscriptVariableError(
            "the canonical Level 5 (decision 114, the widened-observation clause "
            "model) no longer clears its own bar in the ledger, so the count of "
            "passing levels would overstate the record. Review the promotion "
            "against decision 114's falsifier before rendering."
        )
    # The complement of the Level 5 assertion above, for Level 6. Decision 115
    # promoted its finding status but did not make it unqualified, and the count
    # reaches that qualification through one of its three recorded limits. If
    # that one limit ever moves while the others hold, the count would report an
    # unqualified stack the record does not support, so the disagreement is
    # raised rather than rendered.
    _level_six_limits = level_six_qualification_limits(claims)
    if any(_level_six_limits.values()) and not _qualified_flags["stage6_primary_nonconfirmatory"]:
        _standing = sorted(name for name, held in _level_six_limits.items() if held)
        raise ManuscriptVariableError(
            "Level 6 would be counted as an unqualified verdict while these "
            f"recorded limits still hold: {_standing}. The qualifying flag reads "
            "the development-population interval alone, so it has stopped "
            "describing the level. Review decision 115 and "
            "LEVEL_QUALIFYING_VERDICT_FLAGS together before rendering."
        )
    _qualified_verdict_count = sum(_qualified_flags[key] for key in LEVEL_QUALIFYING_VERDICT_FLAGS)
    variables["QUALIFIED_VERDICT_COUNT"] = str(_qualified_verdict_count)
    variables["QUALIFIED_VERDICT_COUNT_WORD"] = _count_word(_qualified_verdict_count)
    # The complement: built levels whose recorded verdict is not qualified. The
    # abstract states both halves so a reader learns what passed as well as
    # what did not; deriving the complement here keeps the two from drifting.
    if _qualified_verdict_count > _built_level_count:
        raise ManuscriptVariableError(
            "more qualified verdicts than built levels: "
            f"{_qualified_verdict_count} > {_built_level_count}"
        )
    _passing_verdict_count = _built_level_count - _qualified_verdict_count
    variables["PASSING_VERDICT_COUNT"] = str(_passing_verdict_count)
    variables["PASSING_VERDICT_COUNT_WORD"] = _count_word(_passing_verdict_count)

    # The benchmark's declared familywise level. The ledger records the derived
    # per-test threshold and the family size, so the level they came from is a
    # product rather than a constant to retype.
    # "About five times looser" was typed; the ratio is measured here, and it is
    # not five. Both operands are recorded per-test thresholds, so the multiplier
    # they imply cannot disagree with them.
    variables["BENCH_VS_REPO_ALPHA_RATIO"] = (
        f"{num('bench_family_alpha') / num('stat_bonferroni_alpha'):.1f}"
    )
    # The share of English the length-matched G2P restriction discards. The
    # complement was recorded; the fraction the paper actually states was not.
    variables["STAGE2_G2P_DISCARDED_PCT"] = f"{100.0 - num('stage2_g2p_length_matched_pct'):.1f}"
    variables["BENCH_FAMILYWISE_ALPHA"] = (
        f"{num('bench_family_alpha') * num('bench_family_size'):.2f}"
    )
