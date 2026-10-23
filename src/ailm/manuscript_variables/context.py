"""In-context benchmark protocol and results producers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import (
    Any,
)

from ailm.benchmarks import (
    draw_in_context_example_sets_with_diagnostics,
    exact_mcnemar_p,
    letter_positions,
    paired_difference_interval,
    wilson_interval,
)
from ailm.benchmarks.constants import (
    HOSTED_SYNTHETIC_MODEL,
    POWER_ALPHA,
    SERVED_PROBE_COUNT,
    SERVED_PROBES,
    hosted_cohort_label,
)
from ailm.benchmarks.synthetic_protocol import (
    SYNTHETIC_INCONTEXT_PROTOCOL,
    SYNTHETIC_ZERO_SHOT_PROTOCOL,
)
from ailm.levels.synthetic import (
    DEFAULT_CORPUS_CHARS as SYNTHETIC_CORPUS_CHARS,
)
from ailm.levels.synthetic import (
    DEFAULT_CORPUS_SEED as SYNTHETIC_CORPUS_SEED,
)
from ailm.levels.synthetic import (
    UNIFORM_BASELINE as SYNTHETIC_UNIFORM_BASELINE,
)
from ailm.levels.synthetic import (
    generate_sequence,
    independent_held_out_positions,
    independent_target_start_tvd,
)
from ailm.statistics import DIFFERENCE_CONFIDENCE
from ailm.value_formatting import (
    ManuscriptVariableError,
)
from ailm.value_formatting import (
    format_ledger_value as _format_value,
)

from ._claims import (
    _ClaimReaders,
    _ContextReconciliation,
    require_close,
)
from ._exceptions import MissingClaimError
from ._formatting import _separation_text
from ._io import _StrictVariableMap
from ._stats import (
    _integer_success_count,
    _require_paired_table,
    _whole,
)


def _produce_context_protocol(
    variables: _StrictVariableMap, claims: Mapping[str, Any], readers: _ClaimReaders
) -> _ContextReconciliation:
    """The in-context protocol reconciliation: frozen-protocol bindings, the
    recorded grid, draw-conditioning diagnostics, schedule arithmetic, and the
    terminal-mapping positive control. Also replays the two relative-change
    operands before any categorical prose consumes their summaries. Returns
    the grid and scored positions the table producer shares.
    """
    num = readers.num
    probability = readers.probability
    require_exact_claim = readers.require_exact_claim
    require_namespace_provenance = readers.require_namespace_provenance
    percentage_cell = readers.percentage_cell
    p_value_cell = readers.p_value_cell

    _zero_protocol = SYNTHETIC_ZERO_SHOT_PROTOCOL
    _context_protocol = SYNTHETIC_INCONTEXT_PROTOCOL
    require_namespace_provenance(
        "bench_ctx_",
        source=_context_protocol.source,
        run_id=_context_protocol.run_id,
    )
    _expected_ctx_grid = ",".join(str(k) for k in _context_protocol.k_grid)
    for _label, _expected in {
        "bench_ctx_openrouter_model": HOSTED_SYNTHETIC_MODEL,
        "bench_ctx_protocol_version": _context_protocol.version,
        "bench_ctx_chain_version": _context_protocol.chain_version,
        "bench_ctx_positions": _context_protocol.positions,
        "bench_ctx_position_generation": _context_protocol.position_generation,
        "bench_ctx_position_seed": _context_protocol.position_seed,
        "bench_ctx_trajectory_length": _context_protocol.query_window + 1,
        "bench_ctx_target_start_tvd": independent_target_start_tvd(_context_protocol.query_window),
        "bench_ctx_query_window": _context_protocol.query_window,
        "bench_ctx_query_context_transitions_max": _context_protocol.query_window - 1,
        "bench_ctx_response_inventory_size": len(_context_protocol.response_inventory),
        "bench_ctx_example_seed": _context_protocol.example_seed,
        "bench_ctx_example_window": _context_protocol.example_window,
        "bench_ctx_example_draw_generation": (_context_protocol.example_draw_generation),
        "bench_ctx_example_internal_transitions_per_window": (
            _context_protocol.example_internal_transitions_per_window
        ),
        "bench_ctx_headline_internal_transitions": (
            _context_protocol.headline_internal_transitions
        ),
        "bench_ctx_headline_total_transitions": (_context_protocol.headline_total_transitions),
        "bench_ctx_k_grid": _expected_ctx_grid,
        "bench_ctx_headline_k": _context_protocol.headline_k,
        "bench_ctx_derangeable_draw_attempts": (_context_protocol.derangeable_draw_attempts),
        "bench_ctx_corpus_chars": SYNTHETIC_CORPUS_CHARS,
        "bench_ctx_corpus_seed": SYNTHETIC_CORPUS_SEED,
        "bench_ctx_context_window": _context_protocol.context_window,
        "bench_ctx_response_tokens": _context_protocol.response_tokens,
        "bench_ctx_prompt_token_overhead_margin": (_context_protocol.prompt_token_overhead_margin),
        "bench_ctx_producer_protocol_version": _zero_protocol.version,
        "bench_ctx_uniform_baseline": SYNTHETIC_UNIFORM_BASELINE,
        "bench_ctx_warmup_policy": _context_protocol.warmup_policy,
        "bench_ctx_arm_order": _context_protocol.arm_order,
        "bench_ctx_cache_protocol": _context_protocol.cache_protocol,
    }.items():
        require_exact_claim(_label, _expected)

    _ctx_grid_value = claims.get("bench_ctx_k_grid", {}).get("value")
    if not isinstance(_ctx_grid_value, str):
        raise ManuscriptVariableError(
            f"bench_ctx_k_grid must be a comma-separated integer grid, got {_ctx_grid_value!r}"
        )
    try:
        _ctx_grid = tuple(int(part.strip()) for part in _ctx_grid_value.split(","))
    except ValueError as exc:
        raise ManuscriptVariableError(
            f"bench_ctx_k_grid must be a comma-separated integer grid, got {_ctx_grid_value!r}"
        ) from exc
    if (
        not _ctx_grid
        or _ctx_grid[0] != 0
        or any(
            current <= previous
            for previous, current in zip(_ctx_grid[:-1], _ctx_grid[1:], strict=True)
        )
    ):
        raise ManuscriptVariableError(
            f"bench_ctx_k_grid must start at zero and be strictly increasing, got {_ctx_grid}"
        )

    # Replay relative-change operands before any categorical prose consumes the
    # summaries. Domain bounds alone cannot distinguish a plausible stale value
    # from the value defined by the current operands.
    require_close(
        "generation_tv_improvement",
        num("generation_tv_improvement"),
        round(num("generation_baseline_tv") - num("generation_best_tv"), 4),
    )
    _oov_raw_gap = num("stage3_oov_raw_gap_nats")
    if _oov_raw_gap <= 0.0:
        raise ManuscriptVariableError(
            "stage3_oov_raw_gap_nats must be positive to define a relative gap change"
        )
    require_close(
        "stage3_oov_gap_change_under_matching_pct",
        num("stage3_oov_gap_change_under_matching_pct"),
        100.0 * (num("stage3_oov_matched_gap_nats") / _oov_raw_gap - 1.0),
    )

    _ctx_positions = _whole("bench_ctx_positions", num("bench_ctx_positions"))
    if _ctx_positions <= 0:
        raise ManuscriptVariableError(f"bench_ctx_positions must be positive, got {_ctx_positions}")
    _diagnostic_positions = independent_held_out_positions(
        _ctx_positions,
        window=_context_protocol.query_window,
        seed=_context_protocol.position_seed,
    )
    _diagnostic_scored_prefixes = {prefix for prefix, _gold in _diagnostic_positions}
    _diagnostic_pool = [
        pair
        for pair in letter_positions(
            generate_sequence(SYNTHETIC_CORPUS_CHARS, SYNTHETIC_CORPUS_SEED),
            window=_context_protocol.example_window,
        )
        if pair[0] not in _diagnostic_scored_prefixes
    ]
    _draw_rows = [
        "| k | Retried positions | Total rejected draws | Maximum attempt used |",
        "|---:|---:|---:|---:|",
    ]
    for _k in _ctx_grid[1:]:
        _pairs, _attempts = draw_in_context_example_sets_with_diagnostics(
            _diagnostic_positions,
            _diagnostic_pool,
            k=_k,
            example_seed=_context_protocol.example_seed,
            draw_attempts=_context_protocol.derangeable_draw_attempts,
            arm=f"k{_k}",
        )
        _expected_retries = sum(attempt - 1 for attempt in _attempts)
        _expected_retried_positions = sum(attempt > 1 for attempt in _attempts)
        _expected_max_attempt = max(_attempts)
        _retry_label = f"bench_ctx_draw_retries_k{_k}"
        _retried_label = f"bench_ctx_draw_retried_positions_k{_k}"
        _max_attempt_label = f"bench_ctx_draw_max_attempts_used_k{_k}"
        _retries = _whole(_retry_label, num(_retry_label))
        _retried_positions = _whole(_retried_label, num(_retried_label))
        _max_attempt = _whole(_max_attempt_label, num(_max_attempt_label))
        for _label, _actual, _expected in (
            (_retry_label, _retries, _expected_retries),
            (_retried_label, _retried_positions, _expected_retried_positions),
            (_max_attempt_label, _max_attempt, _expected_max_attempt),
        ):
            if _actual != _expected:
                raise ManuscriptVariableError(
                    f"{_label} does not reproduce from the frozen seeded draw: "
                    f"recorded {_actual}, expected {_expected}"
                )
        if not (
            0 <= _retried_positions <= _retries
            and 1 <= _max_attempt <= _context_protocol.derangeable_draw_attempts
        ):
            raise ManuscriptVariableError(
                f"draw-conditioning diagnostics are impossible at k={_k}: "
                f"retried={_retried_positions}, retries={_retries}, "
                f"max_attempt={_max_attempt}"
            )
        _draw_rows.append(
            f"| {_k} | {_retried_positions}/{_ctx_positions} | {_retries} | {_max_attempt} |"
        )
    variables["BENCH_CTX_DRAW_DIAGNOSTICS_TABLE"] = "\n".join(_draw_rows)
    _ctx_cache_hits = _whole("bench_ctx_scored_cache_hits", num("bench_ctx_scored_cache_hits"))
    _ctx_cache_misses = _whole(
        "bench_ctx_scored_cache_misses", num("bench_ctx_scored_cache_misses")
    )
    _ctx_arm_count = 2 * (len(_ctx_grid) - 1) + 1
    _expected_ctx_calls = _ctx_positions * _ctx_arm_count * SERVED_PROBE_COUNT
    if _ctx_cache_hits != 0 or _ctx_cache_misses != _expected_ctx_calls:
        raise ManuscriptVariableError(
            "canonical in-context schedule must record zero cache hits and one miss "
            f"per scored call; got hits={_ctx_cache_hits}, misses={_ctx_cache_misses}, "
            f"expected={_expected_ctx_calls}"
        )
    _ctx_max_prompt = _whole(
        "bench_ctx_maximum_prompt_characters",
        num("bench_ctx_maximum_prompt_characters"),
    )
    _ctx_max_prompt_tokens = _whole(
        "bench_ctx_maximum_prompt_tokens",
        num("bench_ctx_maximum_prompt_tokens"),
    )
    if _ctx_max_prompt <= 0:
        raise ManuscriptVariableError("in-context maximum prompt character count must be positive")
    if (
        _ctx_max_prompt_tokens <= 0
        or _ctx_max_prompt_tokens
        + _context_protocol.response_tokens
        + _context_protocol.prompt_token_overhead_margin
        > _context_protocol.context_window
    ):
        raise ManuscriptVariableError(
            "in-context maximum prompt does not fit the declared context window"
        )
    _ctx_family_size = _whole("bench_ctx_family_size", num("bench_ctx_family_size"))
    _expected_ctx_family_size = ((len(_ctx_grid) - 1) * 2 + 1) * SERVED_PROBE_COUNT
    if _ctx_family_size != _expected_ctx_family_size:
        raise ManuscriptVariableError(
            "bench_ctx_family_size must equal two contrasts per nonzero rung plus "
            "one headline contrast, across every served probe; "
            f"recorded {_ctx_family_size}, expected {_expected_ctx_family_size}"
        )
    require_close(
        "bench_ctx_family_alpha",
        num("bench_ctx_family_alpha"),
        POWER_ALPHA / _expected_ctx_family_size,
    )
    _mapping_family_size = _whole(
        "bench_ctx_mapping_control_family_size",
        num("bench_ctx_mapping_control_family_size"),
    )
    if _mapping_family_size != 1:
        raise ManuscriptVariableError(
            "the terminal-mapping positive control must declare its one-test family"
        )
    _mapping_alpha = probability("bench_ctx_mapping_control_family_alpha")
    require_close(
        "bench_ctx_mapping_control_family_alpha",
        _mapping_alpha,
        POWER_ALPHA / _mapping_family_size,
    )
    _mapping_treatment = probability("bench_ctx_mapping_control_top1_treatment")
    _mapping_deranged = probability("bench_ctx_mapping_control_top1_deranged")
    _mapping_treatment_successes = _integer_success_count(
        "mapping-control treatment accuracy", _mapping_treatment, _ctx_positions
    )
    _mapping_deranged_successes = _integer_success_count(
        "mapping-control deranged accuracy", _mapping_deranged, _ctx_positions
    )
    _mapping_b = _whole(
        "bench_ctx_mapping_control_paired_b",
        num("bench_ctx_mapping_control_paired_b"),
    )
    _mapping_c = _whole(
        "bench_ctx_mapping_control_paired_c",
        num("bench_ctx_mapping_control_paired_c"),
    )
    _require_paired_table(
        "mapping-control paired discordance",
        first_successes=_mapping_treatment_successes,
        second_successes=_mapping_deranged_successes,
        b=_mapping_b,
        c=_mapping_c,
        total=_ctx_positions,
    )
    require_close(
        "bench_ctx_mapping_control_paired_p",
        probability("bench_ctx_mapping_control_paired_p"),
        exact_mcnemar_p(_mapping_b, _mapping_c),
    )
    require_close(
        "bench_ctx_mapping_control paired accuracy difference",
        _mapping_treatment - _mapping_deranged,
        (_mapping_b - _mapping_c) / _ctx_positions,
    )
    if (
        _mapping_treatment <= _mapping_deranged
        or probability("bench_ctx_mapping_control_paired_p") >= _mapping_alpha
    ):
        raise ManuscriptVariableError(
            "terminal-mapping positive control did not beat its deranged negative control"
        )
    variables["BENCH_CTX_MAPPING_CONTROL_RESULT"] = (
        f"The fixed terminal-mapping learner scored "
        f"{percentage_cell('bench_ctx_mapping_control_top1_treatment')} on the real "
        f"one-transition examples and "
        f"{percentage_cell('bench_ctx_mapping_control_top1_deranged')} on their "
        f"deranged controls (b/c = {_mapping_b}/{_mapping_c}, p = "
        f"{p_value_cell('bench_ctx_mapping_control_paired_p')}; separately corrected "
        f"one-test family, alpha = "
        f"{p_value_cell('bench_ctx_mapping_control_family_alpha')})."
    )

    return _ContextReconciliation(grid=_ctx_grid, positions=_ctx_positions)


def _produce_context_results(
    variables: _StrictVariableMap,
    claims: Mapping[str, Any],
    readers: _ClaimReaders,
    context: _ContextReconciliation,
) -> None:
    """The context family's accuracy cells, separations, per-probe tables,
    figure title, and headline result, on the protocol's recorded grid.
    """
    num = readers.num
    probability = readers.probability
    percentage_cell = readers.percentage_cell
    p_value_cell = readers.p_value_cell
    _ctx_grid = context.grid
    _ctx_positions = context.positions

    # The context-window table is a generated block, not five typed row labels.
    # Its axis comes from the recorder's own ``bench_ctx_k_grid`` claim and each
    # cell comes from the correspondingly named recorded result.  A new rung
    # therefore changes the table or fails on a missing cell; it cannot leave a
    # plausible row under the wrong coordinate.
    _required_ctx_family_labels: set[str] = set()
    for _probe in SERVED_PROBES:
        _required_ctx_family_labels.add(f"bench_ctx_top1_k0_{_probe.key}")
        for _k in _ctx_grid[1:]:
            _required_ctx_family_labels.update(
                {
                    f"bench_ctx_top1_k{_k}_{_probe.key}",
                    f"bench_ctx_top1_k{_k}_deranged_{_probe.key}",
                }
            )
            for _contrast in ("zero", "deranged"):
                _prefix = f"bench_ctx_k{_k}_vs_{_contrast}"
                _required_ctx_family_labels.update(
                    {
                        f"{_prefix}_paired_b_{_probe.key}",
                        f"{_prefix}_paired_c_{_probe.key}",
                        f"{_prefix}_paired_p_{_probe.key}",
                    }
                )
    _missing_ctx_family = sorted(_required_ctx_family_labels - set(claims))
    if _missing_ctx_family:
        raise MissingClaimError(
            "context semantic family needs every accuracy and discordance cell; "
            f"missing {_missing_ctx_family}"
        )

    _ctx_arm_names = tuple(
        f"k{k}" + ("_deranged" if deranged else "")
        for k in _ctx_grid
        for deranged in (False, True)
        if not (k == 0 and deranged)
    )
    _ctx_accuracy_cells: dict[tuple[str, str], str] = {}
    _ctx_success_counts: dict[tuple[str, str], int] = {}
    _ctx_unparsable_counts: dict[tuple[str, str], int] = {}
    for _probe in SERVED_PROBES:
        for _arm in _ctx_arm_names:
            _accuracy_label = f"bench_ctx_top1_{_arm}_{_probe.key}"
            _accuracy = probability(_accuracy_label)
            _successes = _integer_success_count(
                f"context accuracy {_accuracy_label}", _accuracy, _ctx_positions
            )
            _low_label = f"bench_ctx_top1_wilson_low_{_arm}_{_probe.key}"
            _high_label = f"bench_ctx_top1_wilson_high_{_arm}_{_probe.key}"
            _low = probability(_low_label)
            _high = probability(_high_label)
            _expected_low, _expected_high = wilson_interval(
                _successes,
                _ctx_positions,
            )
            require_close(_low_label, _low, _expected_low)
            require_close(_high_label, _high, _expected_high)
            _bad_label = f"bench_ctx_unparsable_{_arm}_{_probe.key}"
            _bad = _whole(_bad_label, num(_bad_label))
            if not 0 <= _bad <= _ctx_positions - _successes:
                raise ManuscriptVariableError(
                    f"context successes plus unparsable responses for {_bad_label} "
                    f"must not exceed {_ctx_positions}; got successes={_successes}, "
                    f"unparsable={_bad}"
                )
            _ctx_success_counts[(_probe.key, _arm)] = _successes
            _ctx_unparsable_counts[(_probe.key, _arm)] = _bad
            _ctx_accuracy_cells[(_probe.key, _arm)] = (
                f"{percentage_cell(_accuracy_label)} "
                f"[{percentage_cell(_low_label)}–{percentage_cell(_high_label)}]"
            )

    _ctx_ailm = probability("bench_ctx_top1_ailm")
    _ctx_alpha = probability("bench_ctx_family_alpha")
    _ctx_zero_separations: dict[str, list[tuple[int, str]]] = {
        probe.key: [] for probe in SERVED_PROBES
    }
    _ctx_deranged_separations: dict[str, list[tuple[int, str]]] = {
        probe.key: [] for probe in SERVED_PROBES
    }
    for _probe in SERVED_PROBES:
        _zero = probability(f"bench_ctx_top1_k0_{_probe.key}")
        for _k in _ctx_grid[1:]:
            _real_label = f"bench_ctx_top1_k{_k}_{_probe.key}"
            _real = probability(_real_label)
            _deranged = probability(f"bench_ctx_top1_k{_k}_deranged_{_probe.key}")
            for _contrast, _comparator in (("zero", _zero), ("deranged", _deranged)):
                _prefix = f"bench_ctx_k{_k}_vs_{_contrast}"
                _b_label = f"{_prefix}_paired_b_{_probe.key}"
                _c_label = f"{_prefix}_paired_c_{_probe.key}"
                _p_label = f"{_prefix}_paired_p_{_probe.key}"
                _b = _whole(_b_label, num(_b_label))
                _c = _whole(_c_label, num(_c_label))
                _comparator_arm = "k0" if _contrast == "zero" else f"k{_k}_deranged"
                _require_paired_table(
                    f"context discordance {_b_label}/{_c_label}",
                    first_successes=_ctx_success_counts[(_probe.key, f"k{_k}")],
                    second_successes=_ctx_success_counts[(_probe.key, _comparator_arm)],
                    b=_b,
                    c=_c,
                    total=_ctx_positions,
                )
                require_close(_p_label, probability(_p_label), exact_mcnemar_p(_b, _c))
                require_close(
                    f"{_real_label} minus {_contrast}",
                    _real - _comparator,
                    (_b - _c) / _ctx_positions,
                )
                if probability(_p_label) < _ctx_alpha:
                    if _real == _comparator:
                        raise ManuscriptVariableError(
                            f"context contrast {_p_label} is significant at equal accuracies"
                        )
                    _direction = "above" if _real > _comparator else "below"
                    _target = (
                        _ctx_zero_separations if _contrast == "zero" else _ctx_deranged_separations
                    )
                    _target[_probe.key].append((_k, _direction))

    #: Context-window cells where the deranged control outscores the real
    #: prompt. The tally and the table share the parsed recorded grid and the
    #: canonical probe registry; an added rung or probe cannot update one alone.
    _ctx_cells = [
        (f"bench_ctx_top1_k{k}_{probe.key}", f"bench_ctx_top1_k{k}_deranged_{probe.key}")
        for k in _ctx_grid[1:]
        for probe in SERVED_PROBES
    ]
    _missing_ctx_cells = [label for pair in _ctx_cells for label in pair if label not in claims]
    if _missing_ctx_cells:
        raise MissingClaimError(
            "context-control tally needs every nonzero grid/probe cell; missing "
            f"{sorted(_missing_ctx_cells)}"
        )
    _control_ahead = sum(1 for real, ctrl in _ctx_cells if num(ctrl) > num(real))
    variables["BENCH_CTX_CONTROL_AHEAD_CELLS"] = str(_control_ahead)
    variables["BENCH_CTX_CONTROL_TOTAL_CELLS"] = str(len(_ctx_cells))

    _ctx_family_parts = [
        f"{hosted_cohort_label(probe.key)}: "
        f"{_separation_text(_ctx_zero_separations[probe.key], 'the no-demonstration arm')}; "
        f"{_separation_text(_ctx_deranged_separations[probe.key], 'its deranged control')}"
        for probe in SERVED_PROBES
    ]
    variables["BENCH_CTX_FAMILY_RESULT"] = (
        "At the recorded family-corrected threshold, "
        + "; ".join(_ctx_family_parts)
        + f". The deranged control was descriptively ahead in {_control_ahead} of "
        f"{len(_ctx_cells)} nonzero-rung probe cells. This bounds only the declared "
        "prompt format and grid; it does not remove the fitted-versus-unfitted "
        "opportunity asymmetry."
    )
    _zero_separation_count = sum(map(len, _ctx_zero_separations.values()))
    _deranged_separation_count = sum(map(len, _ctx_deranged_separations.values()))
    if _deranged_separation_count:
        _figure_reading = (
            f"{_deranged_separation_count} treatment-versus-deranged contrast(s) "
            "separate after family correction"
        )
    elif _zero_separation_count:
        _figure_reading = (
            f"{_zero_separation_count} treatment-versus-no-demonstration contrast(s) "
            "separate; none separates from its deranged control"
        )
    else:
        _figure_reading = "no treatment arm separates from its no-demonstration or deranged control"
    variables["BENCH_CTX_FIGURE_TITLE"] = (
        f"{HOSTED_SYNTHETIC_MODEL}: protocol runs\n"
        f"Stipulated chain, {_ctx_positions} held-out positions: {_figure_reading}"
    )

    _confidence_text = _format_value(round(DIFFERENCE_CONFIDENCE * 100.0, 4))
    _ctx_tables: list[str] = []
    for _probe in SERVED_PROBES:
        _ctx_rows = [
            "```{=latex}\n\\csname @ifclassloaded\\endcsname{beamer}{}{\\Needspace{8\\baselineskip} }\n```\n",
            f"**{hosted_cohort_label(_probe.key)}**",
            "",
            "```{=latex}\n\\csname @ifclassloaded\\endcsname{beamer}{}{\\vspace{0.5\\baselineskip} }\n```\n",
            f"| k | Top-1 [descriptive {_confidence_text}% Wilson] | "
            f"Deranged [descriptive {_confidence_text}% Wilson] | "
            "Invalid real/control | p vs no-demonstration | p vs deranged |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for _k in _ctx_grid:
            _cells = [
                str(_k),
                _ctx_accuracy_cells[(_probe.key, f"k{_k}")],
            ]
            if _k == 0:
                _cells.extend(
                    (
                        "—",
                        f"{_ctx_unparsable_counts[(_probe.key, 'k0')]}/—",
                        "—",
                        "—",
                    )
                )
            else:
                _cells.extend(
                    (
                        _ctx_accuracy_cells[(_probe.key, f"k{_k}_deranged")],
                        f"{_ctx_unparsable_counts[(_probe.key, f'k{_k}')]}/"
                        f"{_ctx_unparsable_counts[(_probe.key, f'k{_k}_deranged')]}",
                        p_value_cell(f"bench_ctx_k{_k}_vs_zero_paired_p_{_probe.key}"),
                        p_value_cell(f"bench_ctx_k{_k}_vs_deranged_paired_p_{_probe.key}"),
                    )
                )
            _ctx_rows.append("| " + " | ".join(_cells) + " |")
        _ctx_tables.append("\n".join(_ctx_rows))
    variables["BENCH_CTX_TABLES"] = "\n\n".join(_ctx_tables)
    variables["BENCH_CTX_UNIFORM_BASELINE_DISPLAY"] = percentage_cell("bench_ctx_uniform_baseline")
    variables["BENCH_CTX_TOP1_AILM_DISPLAY"] = percentage_cell("bench_ctx_top1_ailm")
    variables["BENCH_CTX_FAMILY_ALPHA_DISPLAY"] = p_value_cell("bench_ctx_family_alpha")

    _headline_k = _whole("bench_ctx_headline_k", num("bench_ctx_headline_k"))
    if _headline_k != _ctx_grid[-1]:
        raise ManuscriptVariableError(
            "bench_ctx_headline_k must equal the final recorded context-grid rung; "
            f"got headline {_headline_k} and grid {_ctx_grid}"
        )
    _headline_parts: list[str] = []
    _gap_parts: list[str] = []
    for _probe in SERVED_PROBES:
        _prefix = f"bench_ctx_k{_headline_k}_vs_ailm"
        _head = probability(f"bench_ctx_top1_k{_headline_k}_{_probe.key}")
        _zero = probability(f"bench_ctx_top1_k0_{_probe.key}")
        _b_label = f"{_prefix}_paired_b_{_probe.key}"
        _c_label = f"{_prefix}_paired_c_{_probe.key}"
        _p_label = f"{_prefix}_paired_p_{_probe.key}"
        _b = _whole(_b_label, num(_b_label))
        _c = _whole(_c_label, num(_c_label))
        _ctx_ailm_successes = _integer_success_count(
            "bench_ctx_top1_ailm", _ctx_ailm, _ctx_positions
        )
        _require_paired_table(
            f"headline context discordance for {_probe.key}",
            first_successes=_ctx_success_counts[(_probe.key, f"k{_headline_k}")],
            second_successes=_ctx_ailm_successes,
            b=_b,
            c=_c,
            total=_ctx_positions,
        )
        _delta_label = f"{_prefix}_delta_{_probe.key}"
        _delta = num(_delta_label)
        require_close(_delta_label, _delta, (_b - _c) / _ctx_positions)
        require_close(_delta_label, _delta, _head - _ctx_ailm)
        require_close(_p_label, probability(_p_label), exact_mcnemar_p(_b, _c))
        _expected_low, _expected_high = paired_difference_interval(
            _b, _c, _ctx_positions, confidence=DIFFERENCE_CONFIDENCE
        )
        require_close(
            f"{_prefix}_delta_low_{_probe.key}",
            num(f"{_prefix}_delta_low_{_probe.key}"),
            _expected_low,
        )
        require_close(
            f"{_prefix}_delta_high_{_probe.key}",
            num(f"{_prefix}_delta_high_{_probe.key}"),
            _expected_high,
        )
        _zero_gap = _ctx_ailm - _zero
        if _zero_gap == 0.0:
            raise ManuscriptVariableError(
                f"bench_ctx gap-closed ratio for {_probe.key} has a zero denominator"
            )
        _gap_label = f"bench_ctx_gap_closed_ratio_{_probe.key}"
        require_close(_gap_label, num(_gap_label), (_head - _zero) / _zero_gap)
        _headline_parts.append(
            f"{num(f'{_prefix}_delta_{_probe.key}') * 100:.2f} percentage points "
            f"for {hosted_cohort_label(_probe.key)} ({_confidence_text}% interval "
            f"{num(f'{_prefix}_delta_low_{_probe.key}') * 100:.2f} to "
            f"{num(f'{_prefix}_delta_high_{_probe.key}') * 100:.2f} percentage points, "
            f"p = {p_value_cell(f'{_prefix}_paired_p_{_probe.key}')})"
        )
        _gap_parts.append(
            f"{hosted_cohort_label(_probe.key)}: {num(f'bench_ctx_gap_closed_ratio_{_probe.key}') * 100:.2f}%"
        )
    variables["BENCH_CTX_HEADLINE_RESULT"] = (
        f"At the audit-specified final registered rung *k* = {_headline_k}, "
        f"probe accuracy minus "
        f"the comparator was {'; '.join(_headline_parts)}. The signed ratios of "
        f"headline lift to each respective no-demonstration gap were "
        f"{'; '.join(_gap_parts)}; negative values mean the examples reduced accuracy."
    )
