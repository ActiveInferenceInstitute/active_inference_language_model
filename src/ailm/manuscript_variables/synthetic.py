"""Synthetic and aligned-G2P benchmark producers."""

from __future__ import annotations

from ailm.benchmarks import (
    exact_mcnemar_p,
    paired_difference_interval,
    wilson_interval,
)
from ailm.benchmarks.constants import (
    G2P_BUCKETS,
    HOSTED_SYNTHETIC_MODEL,
    POWER_ALPHA,
    SERVED_PROBE_COUNT,
    SERVED_PROBES,
    hosted_cohort_label,
)
from ailm.benchmarks.synthetic_protocol import (
    SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL,
    SYNTHETIC_ZERO_SHOT_PROTOCOL,
)
from ailm.levels.synthetic import (
    DEFAULT_CORPUS_CHARS as SYNTHETIC_CORPUS_CHARS,
)
from ailm.levels.synthetic import (
    DEFAULT_CORPUS_SEED as SYNTHETIC_CORPUS_SEED,
)
from ailm.levels.synthetic import (
    DEFAULT_PHONE_CORPUS_SEED as SYNTHETIC_PHONE_CORPUS_SEED,
)
from ailm.levels.synthetic import (
    DEFAULT_PHONE_CORPUS_SYMBOLS as SYNTHETIC_PHONE_CORPUS_SYMBOLS,
)
from ailm.levels.synthetic import (
    PHONE_UNIFORM_BASELINE as SYNTHETIC_PHONE_UNIFORM_BASELINE,
)
from ailm.levels.synthetic import (
    SYNTHETIC_LETTERS,
    SYNTHETIC_PHONES,
    independent_target_start_tvd,
    phone_target_start_tvd,
)
from ailm.levels.synthetic import (
    UNIFORM_BASELINE as SYNTHETIC_UNIFORM_BASELINE,
)
from ailm.value_formatting import ManuscriptVariableError

from ._claims import (
    _ClaimReaders,
    _SyntheticSweep,
    _validate_synthetic_interval,
    require_close,
)
from ._formatting import _count_word
from ._io import _StrictVariableMap
from ._stats import (
    _integer_success_count,
    _require_paired_table,
    _whole,
)


def _g2p_bucket_variables(
    buckets: tuple[tuple[str, int, int], ...] = G2P_BUCKETS,
) -> dict[str, str]:
    """Render the code-owned inclusive G2P bucket boundaries for prose.

    The authored benchmark table currently has short, mid, and long columns, so
    a registry rename or expansion is a manuscript change and fails here rather
    than silently leaving a row or a range unreported.
    """
    if tuple(name for name, _low, _high in buckets) != ("short", "mid", "long"):
        raise ManuscriptVariableError(
            "G2P_BUCKETS must match the authored short/mid/long table before rendering"
        )
    previous_high: int | None = None
    variables: dict[str, str] = {}
    for index, (name, low, high) in enumerate(buckets):
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (low, high)):
            raise ManuscriptVariableError(
                f"G2P bucket {name!r} boundaries must be integers, got {low!r}/{high!r}"
            )
        if low < 0 or high < low:
            raise ManuscriptVariableError(
                f"G2P bucket {name!r} has invalid inclusive range {low}..{high}"
            )
        if previous_high is None:
            if low != 0:
                raise ManuscriptVariableError(f"the first G2P bucket must start at zero, got {low}")
        elif low != previous_high + 1:
            raise ManuscriptVariableError(
                "G2P_BUCKETS must be contiguous and non-overlapping; "
                f"{name!r} starts at {low} after {previous_high}"
            )
        prefix = f"BENCH_G2P_BUCKET_{name.upper()}"
        variables[f"{prefix}_LOW"] = str(low)
        variables[f"{prefix}_HIGH"] = str(high)
        if index == 0:
            variables[f"{prefix}_RANGE"] = f"{_count_word(high)} phones or fewer"
        elif index == len(buckets) - 1:
            variables[f"{prefix}_RANGE"] = f"{_count_word(low)} phones or more"
        else:
            variables[f"{prefix}_RANGE"] = f"{_count_word(low)} to {_count_word(high)} phones"
        previous_high = high
    return variables


def _produce_synthetic_results(
    variables: _StrictVariableMap, readers: _ClaimReaders
) -> _SyntheticSweep:
    """The synthetic no-demonstration table, its protocol reconciliation, and
    the scope reading; returns the sweep state the context producer shares.
    """
    num = readers.num
    probability = readers.probability
    require_exact_claim = readers.require_exact_claim
    require_namespace_provenance = readers.require_namespace_provenance
    percentage_cell = readers.percentage_cell
    p_value_cell = readers.p_value_cell

    #: Synthetic no-demonstration results are a registry-derived block. The producer
    #: and dependent in-context consumer must cover the same probes; generating
    #: the rows here makes a missing producer claim fail hydration rather than
    #: leaving a plausible one-probe paragraph in a two-probe manuscript.
    for probe in SERVED_PROBES:
        variables[f"BENCH_HOSTED_COHORT_{probe.key.upper()}_DISPLAY"] = hosted_cohort_label(
            probe.key
        )
    _zero_protocol = SYNTHETIC_ZERO_SHOT_PROTOCOL
    require_namespace_provenance(
        "bench_synth_",
        source=_zero_protocol.source,
        run_id=_zero_protocol.run_id,
    )
    for _label, _expected in {
        "bench_synth_openrouter_model": HOSTED_SYNTHETIC_MODEL,
        "bench_synth_protocol_version": _zero_protocol.version,
        "bench_synth_chain_version": _zero_protocol.chain_version,
        "bench_synth_positions": _zero_protocol.positions,
        "bench_synth_position_generation": _zero_protocol.position_generation,
        "bench_synth_position_seed": _zero_protocol.position_seed,
        "bench_synth_trajectory_length": _zero_protocol.query_window + 1,
        "bench_synth_target_start_tvd": independent_target_start_tvd(_zero_protocol.query_window),
        "bench_synth_query_window": _zero_protocol.query_window,
        "bench_synth_query_context_transitions_max": (_zero_protocol.query_context_transitions_max),
        "bench_synth_short_query_window": _zero_protocol.short_query_window,
        "bench_synth_short_query_context_transitions_max": (
            _zero_protocol.short_query_context_transitions_max
        ),
        "bench_synth_response_inventory_size": len(_zero_protocol.response_inventory),
        "bench_synth_alphabet_size": len(SYNTHETIC_LETTERS),
        "bench_synth_corpus_chars": SYNTHETIC_CORPUS_CHARS,
        "bench_synth_corpus_seed": SYNTHETIC_CORPUS_SEED,
        "bench_synth_context_window": _zero_protocol.context_window,
        "bench_synth_response_tokens": _zero_protocol.response_tokens,
        "bench_synth_prompt_token_overhead_margin": (_zero_protocol.prompt_token_overhead_margin),
        "bench_synth_uniform_baseline": SYNTHETIC_UNIFORM_BASELINE,
        "bench_synth_warmup_policy": _zero_protocol.warmup_policy,
        "bench_synth_context_arm_order": _zero_protocol.arm_order,
        "bench_synth_cache_protocol": _zero_protocol.cache_protocol,
    }.items():
        require_exact_claim(_label, _expected)

    _synth_positions = _whole("bench_synth_positions", num("bench_synth_positions"))
    if _synth_positions <= 0:
        raise ManuscriptVariableError(
            f"bench_synth_positions must be positive, got {_synth_positions}"
        )
    _synth_cache_hits = _whole(
        "bench_synth_scored_cache_hits", num("bench_synth_scored_cache_hits")
    )
    _synth_cache_misses = _whole(
        "bench_synth_scored_cache_misses", num("bench_synth_scored_cache_misses")
    )
    _expected_synth_calls = _synth_positions * 2 * SERVED_PROBE_COUNT
    if _synth_cache_hits != 0 or _synth_cache_misses != _expected_synth_calls:
        raise ManuscriptVariableError(
            "canonical synthetic schedule must record zero cache hits and one miss "
            f"per scored call; got hits={_synth_cache_hits}, "
            f"misses={_synth_cache_misses}, expected={_expected_synth_calls}"
        )
    _synth_max_prompt = _whole(
        "bench_synth_maximum_prompt_characters",
        num("bench_synth_maximum_prompt_characters"),
    )
    _synth_max_prompt_tokens = _whole(
        "bench_synth_maximum_prompt_tokens",
        num("bench_synth_maximum_prompt_tokens"),
    )
    if _synth_max_prompt <= 0:
        raise ManuscriptVariableError("synthetic maximum prompt character count must be positive")
    if (
        _synth_max_prompt_tokens <= 0
        or _synth_max_prompt_tokens
        + _zero_protocol.response_tokens
        + _zero_protocol.prompt_token_overhead_margin
        > _zero_protocol.context_window
    ):
        raise ManuscriptVariableError(
            "synthetic maximum prompt does not fit the declared context window"
        )
    _synth_family_size = _whole("bench_synth_family_size", num("bench_synth_family_size"))
    if _synth_family_size != SERVED_PROBE_COUNT:
        raise ManuscriptVariableError(
            "the synthetic family must contain one paired contrast for every "
            f"registered probe; recorded {_synth_family_size}, expected {SERVED_PROBE_COUNT}"
        )
    require_close(
        "bench_synth_family_alpha",
        probability("bench_synth_family_alpha"),
        POWER_ALPHA / _synth_family_size,
    )
    _synth_alphabet_size = _whole("bench_synth_alphabet_size", num("bench_synth_alphabet_size"))
    if _synth_alphabet_size <= 1:
        raise ManuscriptVariableError(
            "bench_synth_alphabet_size must exceed one to define a prediction task, "
            f"got {_synth_alphabet_size}"
        )
    require_close(
        "bench_synth_uniform_baseline",
        probability("bench_synth_uniform_baseline"),
        1.0 / _synth_alphabet_size,
    )

    _synth_ailm = probability("bench_synth_letter_top1_ailm")
    _synth_ailm_successes = _integer_success_count(
        "bench_synth_letter_top1_ailm", _synth_ailm, _synth_positions
    )
    _synth_ailm_low_label, _synth_ailm_high_label = _validate_synthetic_interval(
        readers, _synth_positions, "ailm", _synth_ailm
    )
    _synth_rows = [
        "| AILM | "
        f"{percentage_cell('bench_synth_letter_top1_ailm')} ["
        f"{percentage_cell(_synth_ailm_low_label)}–"
        f"{percentage_cell(_synth_ailm_high_label)}] | — | — | — | — |"
    ]
    _synth_ailm_separates: list[str] = []
    _synth_probe_separates: list[str] = []
    _synth_unresolved: list[str] = []
    for _probe in SERVED_PROBES:
        _accuracy_label = f"bench_synth_letter_top1_{_probe.key}"
        _accuracy = probability(_accuracy_label)
        _low_label, _high_label = _validate_synthetic_interval(
            readers, _synth_positions, _probe.key, _accuracy
        )
        _b_label = f"bench_synth_paired_b_{_probe.key}"
        _c_label = f"bench_synth_paired_c_{_probe.key}"
        _p_label = f"bench_synth_paired_p_{_probe.key}"
        _b = _whole(_b_label, num(_b_label))
        _c = _whole(_c_label, num(_c_label))
        _accuracy_successes = _integer_success_count(_accuracy_label, _accuracy, _synth_positions)
        _require_paired_table(
            f"synthetic discordance for {_probe.key}",
            first_successes=_accuracy_successes,
            second_successes=_synth_ailm_successes,
            b=_b,
            c=_c,
            total=_synth_positions,
        )
        _p = probability(_p_label)
        require_close(_p_label, _p, exact_mcnemar_p(_b, _c))
        require_close(
            f"{_accuracy_label} minus bench_synth_letter_top1_ailm",
            _accuracy - _synth_ailm,
            (_b - _c) / _synth_positions,
        )
        _delta_low, _delta_high = paired_difference_interval(_b, _c, _synth_positions)
        _unparsable_label = f"bench_synth_unparsable_{_probe.key}"
        _unparsable = _whole(_unparsable_label, num(_unparsable_label))
        if not 0 <= _unparsable <= _synth_positions - _accuracy_successes:
            raise ManuscriptVariableError(
                f"synthetic successes plus unparsable responses for {_probe.key} "
                f"must not exceed {_synth_positions}; got successes="
                f"{_accuracy_successes}, unparsable={_unparsable}"
            )
        _unparsable_rate_label = f"bench_synth_unparsable_rate_{_probe.key}"
        require_close(
            _unparsable_rate_label,
            probability(_unparsable_rate_label),
            _unparsable / _synth_positions,
        )
        if _p < probability("bench_synth_family_alpha"):
            if _synth_ailm > _accuracy:
                _synth_ailm_separates.append(hosted_cohort_label(_probe.key))
            elif _accuracy > _synth_ailm:
                _synth_probe_separates.append(hosted_cohort_label(_probe.key))
            else:
                raise ManuscriptVariableError(
                    f"synthetic {_probe.key} has a significant paired result but equal accuracies"
                )
        else:
            _synth_unresolved.append(hosted_cohort_label(_probe.key))
        _synth_rows.append(
            "| "
            + " | ".join(
                (
                    hosted_cohort_label(_probe.key),
                    f"{percentage_cell(_accuracy_label)} "
                    f"[{percentage_cell(_low_label)}–{percentage_cell(_high_label)}]",
                    f"{100 * (_accuracy - _synth_ailm):+.2f} pp "
                    f"[{100 * _delta_low:+.2f}, {100 * _delta_high:+.2f}]",
                    f"{_b}/{_c}",
                    p_value_cell(_p_label),
                    str(_unparsable),
                )
            )
            + " |"
        )
    variables["BENCH_SYNTH_RESULTS_TABLE_ROWS"] = "\n".join(_synth_rows)
    variables["BENCH_SYNTH_FAMILY_ALPHA_DISPLAY"] = p_value_cell("bench_synth_family_alpha")
    variables["BENCH_SYNTH_UNIFORM_BASELINE_DISPLAY"] = percentage_cell(
        "bench_synth_uniform_baseline"
    )
    if len(_synth_ailm_separates) == SERVED_PROBE_COUNT:
        variables["BENCH_SYNTH_SCOPE_RESULT"] = (
            "On this declared run, the fitted AILM comparator separates from every "
            "registered no-demonstration probe on the held-out stipulated chain at the "
            "synthetic family's corrected threshold."
        )
    else:
        _readings: list[str] = []
        if _synth_ailm_separates:
            _readings.append("AILM separates from " + ", ".join(_synth_ailm_separates))
        if _synth_probe_separates:
            _readings.append(
                "the probe separates from AILM for " + ", ".join(_synth_probe_separates)
            )
        if _synth_unresolved:
            _readings.append("the contrast is unresolved for " + ", ".join(_synth_unresolved))
        variables["BENCH_SYNTH_SCOPE_RESULT"] = (
            "At the synthetic family's corrected threshold, " + "; ".join(_readings) + "."
        )

    _synth_response_inventory = _whole(
        "bench_synth_response_inventory_size",
        num("bench_synth_response_inventory_size"),
    )
    if _synth_response_inventory != _synth_alphabet_size:
        raise ManuscriptVariableError(
            "the synthetic response opportunity must equal the stipulated alphabet: "
            f"response={_synth_response_inventory}, alphabet={_synth_alphabet_size}"
        )
    _synth_query_window = _whole("bench_synth_query_window", num("bench_synth_query_window"))
    _synth_short_window = _whole(
        "bench_synth_short_query_window", num("bench_synth_short_query_window")
    )
    if not 0 < _synth_short_window < _synth_query_window:
        raise ManuscriptVariableError(
            "synthetic query windows must satisfy 0 < short < full, got "
            f"{_synth_short_window}, {_synth_query_window}"
        )
    for _window_label, _window, _transition_label in (
        (
            "bench_synth_query_window",
            _synth_query_window,
            "bench_synth_query_context_transitions_max",
        ),
        (
            "bench_synth_short_query_window",
            _synth_short_window,
            "bench_synth_short_query_context_transitions_max",
        ),
    ):
        _transitions = _whole(_transition_label, num(_transition_label))
        if _transitions != _window - 1:
            raise ManuscriptVariableError(
                f"{_transition_label} must equal {_window_label} - 1; "
                f"recorded {_transitions}, expected {_window - 1}"
            )

    return _SyntheticSweep(positions=_synth_positions, short_window=_synth_short_window)


def _produce_synthetic_context_results(
    variables: _StrictVariableMap, readers: _ClaimReaders, synthetic: _SyntheticSweep
) -> None:
    """The synthetic context-sensitivity family table and its scope reading,
    paired on the same scored positions as the no-demonstration sweep.
    """
    num = readers.num
    probability = readers.probability
    percentage_cell = readers.percentage_cell
    p_value_cell = readers.p_value_cell
    _synth_positions = synthetic.positions
    _synth_short_window = synthetic.short_window

    _synth_context_family_size = _whole(
        "bench_synth_context_family_size",
        num("bench_synth_context_family_size"),
    )
    if _synth_context_family_size != SERVED_PROBE_COUNT:
        raise ManuscriptVariableError(
            "the synthetic context-sensitivity family must contain one paired "
            f"contrast per probe; got {_synth_context_family_size}, expected "
            f"{SERVED_PROBE_COUNT}"
        )
    require_close(
        "bench_synth_context_family_alpha",
        probability("bench_synth_context_family_alpha"),
        POWER_ALPHA / _synth_context_family_size,
    )
    _synth_context_rows: list[str] = []
    _synth_full_better: list[str] = []
    _synth_short_better: list[str] = []
    _synth_context_unresolved: list[str] = []
    for _probe in SERVED_PROBES:
        _full_label = f"bench_synth_letter_top1_{_probe.key}"
        _full_accuracy = probability(_full_label)
        _short_label = f"bench_synth_short_top1_{_probe.key}"
        _short_accuracy = probability(_short_label)
        _short_low_label, _short_high_label = _validate_synthetic_interval(
            readers,
            _synth_positions,
            _probe.key,
            _short_accuracy,
            estimate_stem="short_top1",
            interval_stem="short_wilson",
        )
        _ctx_b_label = f"bench_synth_context_paired_b_{_probe.key}"
        _ctx_c_label = f"bench_synth_context_paired_c_{_probe.key}"
        _ctx_p_label = f"bench_synth_context_paired_p_{_probe.key}"
        _ctx_b = _whole(_ctx_b_label, num(_ctx_b_label))
        _ctx_c = _whole(_ctx_c_label, num(_ctx_c_label))
        _full_successes = _integer_success_count(_full_label, _full_accuracy, _synth_positions)
        _short_successes = _integer_success_count(_short_label, _short_accuracy, _synth_positions)
        _require_paired_table(
            f"synthetic context discordance for {_probe.key}",
            first_successes=_full_successes,
            second_successes=_short_successes,
            b=_ctx_b,
            c=_ctx_c,
            total=_synth_positions,
        )
        _ctx_p = probability(_ctx_p_label)
        require_close(_ctx_p_label, _ctx_p, exact_mcnemar_p(_ctx_b, _ctx_c))
        require_close(
            f"{_full_label} minus {_short_label}",
            _full_accuracy - _short_accuracy,
            (_ctx_b - _ctx_c) / _synth_positions,
        )
        _ctx_low, _ctx_high = paired_difference_interval(_ctx_b, _ctx_c, _synth_positions)
        _full_first_label = f"bench_synth_context_order_full_first_{_probe.key}"
        _short_first_label = f"bench_synth_context_order_short_first_{_probe.key}"
        _full_first = _whole(_full_first_label, num(_full_first_label))
        _short_first = _whole(_short_first_label, num(_short_first_label))
        if (_full_first, _short_first) != (
            (_synth_positions + 1) // 2,
            _synth_positions // 2,
        ):
            raise ManuscriptVariableError(
                "synthetic context arm order is not the registered alternating "
                f"schedule for {_probe.key}: full-first={_full_first}, "
                f"short-first={_short_first}, n={_synth_positions}"
            )
        _short_bad_label = f"bench_synth_short_unparsable_{_probe.key}"
        _short_bad = _whole(_short_bad_label, num(_short_bad_label))
        if not 0 <= _short_bad <= _synth_positions - _short_successes:
            raise ManuscriptVariableError(
                f"synthetic short-prefix successes plus unparsable responses for "
                f"{_probe.key} must not exceed {_synth_positions}; got successes="
                f"{_short_successes}, unparsable={_short_bad}"
            )
        _short_bad_rate = f"bench_synth_short_unparsable_rate_{_probe.key}"
        require_close(
            _short_bad_rate,
            probability(_short_bad_rate),
            _short_bad / _synth_positions,
        )
        _full_bad = _whole(
            f"bench_synth_unparsable_{_probe.key}",
            num(f"bench_synth_unparsable_{_probe.key}"),
        )
        if not 0 <= _full_bad <= _synth_positions - _full_successes:
            raise ManuscriptVariableError(
                f"synthetic full-prefix successes plus unparsable responses for "
                f"{_probe.key} must not exceed {_synth_positions}; got successes="
                f"{_full_successes}, unparsable={_full_bad}"
            )
        if _ctx_p < probability("bench_synth_context_family_alpha"):
            if _full_accuracy > _short_accuracy:
                _synth_full_better.append(hosted_cohort_label(_probe.key))
            elif _short_accuracy > _full_accuracy:
                _synth_short_better.append(hosted_cohort_label(_probe.key))
            else:
                raise ManuscriptVariableError(
                    f"synthetic context result for {_probe.key} is significant at equal accuracies"
                )
        else:
            _synth_context_unresolved.append(hosted_cohort_label(_probe.key))
        _synth_context_rows.append(
            "| "
            + " | ".join(
                (
                    hosted_cohort_label(_probe.key),
                    f"{percentage_cell(_short_label)} "
                    f"[{percentage_cell(_short_low_label)}–"
                    f"{percentage_cell(_short_high_label)}]",
                    percentage_cell(_full_label),
                    f"{100 * (_full_accuracy - _short_accuracy):+.2f} pp "
                    f"[{100 * _ctx_low:+.2f}, {100 * _ctx_high:+.2f}]",
                    f"{_ctx_b}/{_ctx_c}",
                    p_value_cell(_ctx_p_label),
                    f"{_short_bad}/{_full_bad}",
                )
            )
            + " |"
        )
    variables["BENCH_SYNTH_CONTEXT_RESULTS_TABLE_ROWS"] = "\n".join(_synth_context_rows)
    variables["BENCH_SYNTH_CONTEXT_FAMILY_ALPHA_DISPLAY"] = p_value_cell(
        "bench_synth_context_family_alpha"
    )
    _context_readings: list[str] = []
    if _synth_full_better:
        _context_readings.append(
            "the full prefix scored higher for " + ", ".join(_synth_full_better)
        )
    if _synth_short_better:
        _context_readings.append(
            f"the {_synth_short_window}-symbol prefix scored higher for "
            + ", ".join(_synth_short_better)
        )
    if _synth_context_unresolved:
        _context_readings.append(
            "the paired sensitivity is unresolved for " + ", ".join(_synth_context_unresolved)
        )
    variables["BENCH_SYNTH_CONTEXT_SCOPE_RESULT"] = (
        "At the audit-specified context family's corrected threshold, "
        + "; ".join(_context_readings)
        + "."
    )


def _produce_synthetic_phone_results(variables: _StrictVariableMap, readers: _ClaimReaders) -> None:
    """The stipulated phone arm's protocol reconciliation and result
    re-derivation.

    The phone subsection renders its numbers as uppercased ledger labels
    directly, so this producer adds no display tokens; what it owns is the
    same re-derivation discipline the letter arm's producers apply: the frozen
    protocol must reconcile with the ledger, the paired tables must close
    against their own sufficient statistics, and every interval must rebuild
    from its recorded endpoints. A raw token alone could never catch a
    mis-paired record; this block can.
    """
    num = readers.num
    probability = readers.probability
    require_exact_claim = readers.require_exact_claim
    require_namespace_provenance = readers.require_namespace_provenance

    protocol = SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL
    require_namespace_provenance(
        "bench_synthphone_",
        source=protocol.source,
        run_id=protocol.run_id,
    )
    for _label, _expected in {
        "bench_synthphone_protocol_version": protocol.version,
        "bench_synthphone_chain_version": protocol.chain_version,
        "bench_synthphone_positions": protocol.positions,
        "bench_synthphone_position_generation": protocol.position_generation,
        "bench_synthphone_position_seed": protocol.position_seed,
        "bench_synthphone_trajectory_length": protocol.query_window + 1,
        "bench_synthphone_target_start_tvd": phone_target_start_tvd(protocol.query_window),
        "bench_synthphone_query_window": protocol.query_window,
        "bench_synthphone_query_context_transitions_max": (protocol.query_context_transitions_max),
        "bench_synthphone_short_query_window": protocol.short_query_window,
        "bench_synthphone_short_query_context_transitions_max": (
            protocol.short_query_context_transitions_max
        ),
        "bench_synthphone_response_inventory_size": len(protocol.response_inventory),
        "bench_synthphone_inventory_size": len(SYNTHETIC_PHONES),
        "bench_synthphone_corpus_symbols": SYNTHETIC_PHONE_CORPUS_SYMBOLS,
        "bench_synthphone_corpus_seed": SYNTHETIC_PHONE_CORPUS_SEED,
        "bench_synthphone_context_window": protocol.context_window,
        "bench_synthphone_response_tokens": protocol.response_tokens,
        "bench_synthphone_prompt_token_overhead_margin": (protocol.prompt_token_overhead_margin),
        "bench_synthphone_uniform_baseline": SYNTHETIC_PHONE_UNIFORM_BASELINE,
        "bench_synthphone_warmup_policy": protocol.warmup_policy,
        "bench_synthphone_arm_order": protocol.arm_order,
        "bench_synthphone_cache_protocol": protocol.cache_protocol,
    }.items():
        require_exact_claim(_label, _expected)

    _phone_positions = _whole("bench_synthphone_positions", num("bench_synthphone_positions"))
    if _phone_positions <= 0:
        raise ManuscriptVariableError(
            f"bench_synthphone_positions must be positive, got {_phone_positions}"
        )
    _phone_cache_hits = _whole(
        "bench_synthphone_scored_cache_hits", num("bench_synthphone_scored_cache_hits")
    )
    _phone_cache_misses = _whole(
        "bench_synthphone_scored_cache_misses", num("bench_synthphone_scored_cache_misses")
    )
    _expected_phone_calls = _phone_positions * 2 * SERVED_PROBE_COUNT
    if _phone_cache_hits != 0 or _phone_cache_misses != _expected_phone_calls:
        raise ManuscriptVariableError(
            "canonical stipulated-phone schedule must record zero cache hits and one "
            f"miss per scored call; got hits={_phone_cache_hits}, "
            f"misses={_phone_cache_misses}, expected={_expected_phone_calls}"
        )
    _phone_max_prompt = _whole(
        "bench_synthphone_maximum_prompt_characters",
        num("bench_synthphone_maximum_prompt_characters"),
    )
    _phone_max_prompt_tokens = _whole(
        "bench_synthphone_maximum_prompt_tokens",
        num("bench_synthphone_maximum_prompt_tokens"),
    )
    if _phone_max_prompt <= 0:
        raise ManuscriptVariableError(
            "stipulated-phone maximum prompt character count must be positive"
        )
    if (
        _phone_max_prompt_tokens <= 0
        or _phone_max_prompt_tokens
        + protocol.response_tokens
        + protocol.prompt_token_overhead_margin
        > protocol.context_window
    ):
        raise ManuscriptVariableError(
            "stipulated-phone maximum prompt does not fit the declared context window"
        )
    _phone_family_size = _whole("bench_synthphone_family_size", num("bench_synthphone_family_size"))
    if _phone_family_size != SERVED_PROBE_COUNT:
        raise ManuscriptVariableError(
            "the stipulated-phone family must contain one paired contrast for every "
            f"registered probe; recorded {_phone_family_size}, expected "
            f"{SERVED_PROBE_COUNT}"
        )
    require_close(
        "bench_synthphone_family_alpha",
        probability("bench_synthphone_family_alpha"),
        POWER_ALPHA / _phone_family_size,
    )
    _phone_context_family_size = _whole(
        "bench_synthphone_context_family_size",
        num("bench_synthphone_context_family_size"),
    )
    if _phone_context_family_size != SERVED_PROBE_COUNT:
        raise ManuscriptVariableError(
            "the stipulated-phone context-sensitivity family must contain one paired "
            f"contrast per probe; got {_phone_context_family_size}, expected "
            f"{SERVED_PROBE_COUNT}"
        )
    require_close(
        "bench_synthphone_context_family_alpha",
        probability("bench_synthphone_context_family_alpha"),
        POWER_ALPHA / _phone_context_family_size,
    )
    _phone_inventory_size = _whole(
        "bench_synthphone_inventory_size", num("bench_synthphone_inventory_size")
    )
    if _phone_inventory_size <= 1:
        raise ManuscriptVariableError(
            "bench_synthphone_inventory_size must exceed one to define a prediction "
            f"task, got {_phone_inventory_size}"
        )
    _phone_response_inventory = _whole(
        "bench_synthphone_response_inventory_size",
        num("bench_synthphone_response_inventory_size"),
    )
    if _phone_response_inventory != _phone_inventory_size:
        raise ManuscriptVariableError(
            "the stipulated-phone response opportunity must equal the stipulated "
            f"inventory: response={_phone_response_inventory}, "
            f"inventory={_phone_inventory_size}"
        )
    require_close(
        "bench_synthphone_uniform_baseline",
        probability("bench_synthphone_uniform_baseline"),
        1.0 / _phone_inventory_size,
    )
    _phone_query_window = _whole(
        "bench_synthphone_query_window", num("bench_synthphone_query_window")
    )
    _phone_short_window = _whole(
        "bench_synthphone_short_query_window", num("bench_synthphone_short_query_window")
    )
    if not 0 < _phone_short_window < _phone_query_window:
        raise ManuscriptVariableError(
            "stipulated-phone query windows must satisfy 0 < short < full, got "
            f"{_phone_short_window}, {_phone_query_window}"
        )
    for _window_label, _window, _transition_label in (
        (
            "bench_synthphone_query_window",
            _phone_query_window,
            "bench_synthphone_query_context_transitions_max",
        ),
        (
            "bench_synthphone_short_query_window",
            _phone_short_window,
            "bench_synthphone_short_query_context_transitions_max",
        ),
    ):
        _transitions = _whole(_transition_label, num(_transition_label))
        if _transitions != _window - 1:
            raise ManuscriptVariableError(
                f"{_transition_label} must equal {_window_label} - 1; "
                f"recorded {_transitions}, expected {_window - 1}"
            )

    _phone_ailm = probability("bench_synthphone_phone_top1_ailm")
    _phone_ailm_successes = _integer_success_count(
        "bench_synthphone_phone_top1_ailm", _phone_ailm, _phone_positions
    )
    _ailm_low_label, _ailm_high_label = _validate_synthetic_interval(
        readers,
        _phone_positions,
        "ailm",
        _phone_ailm,
        namespace="bench_synthphone",
        estimate_stem="phone_top1",
        interval_stem="phone_top1_wilson",
    )
    _phone_context_results: list[str] = []
    _phone_rows = [
        "| AILM | "
        f"{readers.percentage_cell('bench_synthphone_phone_top1_ailm')} ["
        f"{readers.percentage_cell(_ailm_low_label)}–"
        f"{readers.percentage_cell(_ailm_high_label)}] | — | — | — | — |"
    ]
    for _probe in SERVED_PROBES:
        _accuracy_label = f"bench_synthphone_phone_top1_{_probe.key}"
        _accuracy = probability(_accuracy_label)
        _low_label, _high_label = _validate_synthetic_interval(
            readers,
            _phone_positions,
            _probe.key,
            _accuracy,
            namespace="bench_synthphone",
            estimate_stem="phone_top1",
            interval_stem="phone_top1_wilson",
        )
        _b_label = f"bench_synthphone_paired_b_{_probe.key}"
        _c_label = f"bench_synthphone_paired_c_{_probe.key}"
        _p_label = f"bench_synthphone_paired_p_{_probe.key}"
        _b = _whole(_b_label, num(_b_label))
        _c = _whole(_c_label, num(_c_label))
        _accuracy_successes = _integer_success_count(_accuracy_label, _accuracy, _phone_positions)
        _require_paired_table(
            f"stipulated-phone discordance for {_probe.key}",
            first_successes=_accuracy_successes,
            second_successes=_phone_ailm_successes,
            b=_b,
            c=_c,
            total=_phone_positions,
        )
        _p = probability(_p_label)
        require_close(_p_label, _p, exact_mcnemar_p(_b, _c))
        require_close(
            f"{_accuracy_label} minus bench_synthphone_phone_top1_ailm",
            _accuracy - _phone_ailm,
            (_b - _c) / _phone_positions,
        )
        _unparsable = _whole(
            f"bench_synthphone_unparsable_{_probe.key}",
            num(f"bench_synthphone_unparsable_{_probe.key}"),
        )
        if not 0 <= _unparsable <= _phone_positions - _accuracy_successes:
            raise ManuscriptVariableError(
                f"stipulated-phone successes plus unparsable responses for {_probe.key} "
                f"must not exceed {_phone_positions}; got successes="
                f"{_accuracy_successes}, unparsable={_unparsable}"
            )
        require_close(
            f"bench_synthphone_unparsable_rate_{_probe.key}",
            probability(f"bench_synthphone_unparsable_rate_{_probe.key}"),
            _unparsable / _phone_positions,
        )
        _delta_low, _delta_high = paired_difference_interval(_b, _c, _phone_positions)
        _phone_rows.append(
            "| "
            + " | ".join(
                (
                    hosted_cohort_label(_probe.key),
                    f"{readers.percentage_cell(_accuracy_label)} "
                    f"[{readers.percentage_cell(_low_label)}–"
                    f"{readers.percentage_cell(_high_label)}]",
                    f"{100 * (_accuracy - _phone_ailm):+.2f} pp "
                    f"[{100 * _delta_low:+.2f}, {100 * _delta_high:+.2f}]",
                    f"{_b}/{_c}",
                    readers.p_value_cell(_p_label),
                    str(_unparsable),
                )
            )
            + " |"
        )
        _short_label = f"bench_synthphone_short_top1_{_probe.key}"
        _short_accuracy = probability(_short_label)
        _validate_synthetic_interval(
            readers,
            _phone_positions,
            _probe.key,
            _short_accuracy,
            namespace="bench_synthphone",
            estimate_stem="short_top1",
            interval_stem="short_wilson",
        )
        _short_successes = _integer_success_count(_short_label, _short_accuracy, _phone_positions)
        _ctx_b = _whole(
            f"bench_synthphone_context_paired_b_{_probe.key}",
            num(f"bench_synthphone_context_paired_b_{_probe.key}"),
        )
        _ctx_c = _whole(
            f"bench_synthphone_context_paired_c_{_probe.key}",
            num(f"bench_synthphone_context_paired_c_{_probe.key}"),
        )
        _require_paired_table(
            f"stipulated-phone context discordance for {_probe.key}",
            first_successes=_accuracy_successes,
            second_successes=_short_successes,
            b=_ctx_b,
            c=_ctx_c,
            total=_phone_positions,
        )
        _ctx_p = probability(f"bench_synthphone_context_paired_p_{_probe.key}")
        require_close(
            f"bench_synthphone_context_paired_p_{_probe.key}",
            _ctx_p,
            exact_mcnemar_p(_ctx_b, _ctx_c),
        )
        require_close(
            f"{_accuracy_label} minus {_short_label}",
            _accuracy - _short_accuracy,
            (_ctx_b - _ctx_c) / _phone_positions,
        )
        _phone_context_results.append(
            f"{hosted_cohort_label(_probe.key)}: "
            + (
                "clears the corrected threshold"
                if _ctx_p < probability("bench_synthphone_context_family_alpha")
                else "does not clear the corrected threshold"
            )
        )
        _full_first = _whole(
            f"bench_synthphone_context_order_full_first_{_probe.key}",
            num(f"bench_synthphone_context_order_full_first_{_probe.key}"),
        )
        _short_first = _whole(
            f"bench_synthphone_context_order_short_first_{_probe.key}",
            num(f"bench_synthphone_context_order_short_first_{_probe.key}"),
        )
        if (_full_first, _short_first) != (
            (_phone_positions + 1) // 2,
            _phone_positions // 2,
        ):
            raise ManuscriptVariableError(
                "stipulated-phone context arm order is not the registered alternating "
                f"schedule for {_probe.key}: full-first={_full_first}, "
                f"short-first={_short_first}, n={_phone_positions}"
            )
        _short_bad = _whole(
            f"bench_synthphone_short_unparsable_{_probe.key}",
            num(f"bench_synthphone_short_unparsable_{_probe.key}"),
        )
        if not 0 <= _short_bad <= _phone_positions - _short_successes:
            raise ManuscriptVariableError(
                f"stipulated-phone short-prefix successes plus unparsable responses "
                f"for {_probe.key} must not exceed {_phone_positions}; got successes="
                f"{_short_successes}, unparsable={_short_bad}"
            )
        require_close(
            f"bench_synthphone_short_unparsable_rate_{_probe.key}",
            probability(f"bench_synthphone_short_unparsable_rate_{_probe.key}"),
            _short_bad / _phone_positions,
        )

    variables["BENCH_SYNTHPHONE_CONTEXT_RESULT"] = "; ".join(_phone_context_results) + "."
    variables["BENCH_SYNTHPHONE_RESULTS_TABLE_ROWS"] = "\n".join(_phone_rows)
    variables["BENCH_SYNTHPHONE_FAMILY_ALPHA_DISPLAY"] = readers.p_value_cell(
        "bench_synthphone_family_alpha"
    )
    variables["BENCH_SYNTHPHONE_CONTEXT_FAMILY_ALPHA_DISPLAY"] = readers.p_value_cell(
        "bench_synthphone_context_family_alpha"
    )
    variables["BENCH_SYNTHPHONE_UNIFORM_BASELINE_DISPLAY"] = readers.percentage_cell(
        "bench_synthphone_uniform_baseline"
    )


ALIGNED_G2P_COMPARATOR_ARMS: tuple[str, ...] = (
    "native",
    "t_lookup",
    "context_free",
    "a1_english",
    "a2_broken",
)


def _produce_aligned_g2p_comparator(variables: _StrictVariableMap, readers: _ClaimReaders) -> None:
    """The stipulated aligned-G2P comparator arm's display tokens.

    Decision 101 puts a Wilson interval on every stratified benchmark bar, on
    its own denominator. The recorder records accuracies and the two test
    counts and never intervals, so the bounds here are derived at render time
    from the recorded accuracy and the recorded denominator — the
    channel-factorial derivation pattern, adapted to a plain proportion rather
    than a paired difference. The arm's pre-registered refusals (the native
    engine above the uniform floor and above both attribution controls on both
    readings, and never above the trivial ``T`` lookup it is compared against)
    are re-checked from the same recorded numbers, so the supplement can state
    the gates' outcome only while the geometry that produced it still holds: a
    re-record that fired a control fails the render instead of inheriting the
    old prose.
    """
    readers.require_namespace_provenance(
        "bench_synthg2p_comparator_",
        source="scripts/record_aligned_g2p_comparator_claims.py",
        run_id="aligned-g2p-comparator-v1",
    )
    positions = _whole(
        "bench_synthg2p_comparator_test_position_count",
        readers.num("bench_synthg2p_comparator_test_position_count"),
    )
    words = _whole(
        "bench_synthg2p_comparator_test_word_count",
        readers.num("bench_synthg2p_comparator_test_word_count"),
    )
    if positions <= 0 or words <= 0:
        raise ManuscriptVariableError("aligned-G2P test populations must be positive")
    if positions % words:
        raise ManuscriptVariableError(
            "aligned-G2P position count must be a whole multiple of the word count "
            f"under the arm's per-letter 1:1 alignment: positions={positions}, words={words}"
        )

    accuracies: dict[tuple[str, str], float] = {}
    for arm in ALIGNED_G2P_COMPARATOR_ARMS:
        for reading, total in (("position", positions), ("exact", words)):
            label = f"bench_synthg2p_comparator_{arm}_{reading}_accuracy"
            accuracy = readers.probability(label)
            accuracies[(arm, reading)] = accuracy
            # The recorder rounds each accuracy to eight decimals, so the
            # implied success count can miss an integer by up to half of that
            # rounding scaled by the denominator. The tolerance here is that
            # rounding's own footprint, not a general slack: a genuinely
            # non-integer count still refuses.
            scaled = accuracy * total
            successes = round(scaled)
            rounding_footprint = 0.5 * 10**-8 * total
            if abs(scaled - successes) > max(rounding_footprint, 1e-9):
                raise ManuscriptVariableError(
                    f"{label} cannot come from an integer success count at n={total}: {accuracy!r}"
                )
            low, high = wilson_interval(successes, total)
            stem = f"BENCH_SYNTHG2P_COMPARATOR_{arm.upper()}_{reading.upper()}"
            variables[f"{stem}_PCT"] = f"{accuracy * 100.0:.2f}%"
            variables[f"{stem}_WILSON_LOW_PCT"] = f"{low * 100.0:.2f}%"
            variables[f"{stem}_WILSON_HIGH_PCT"] = f"{high * 100.0:.2f}%"

    position_floor = readers.probability("bench_synthg2p_comparator_uniform_position_floor")
    exact_floor = readers.probability("bench_synthg2p_comparator_uniform_exact_floor")
    variables["BENCH_SYNTHG2P_COMPARATOR_UNIFORM_POSITION_FLOOR_PCT"] = (
        f"{position_floor * 100.0:.2f}%"
    )
    variables["BENCH_SYNTHG2P_COMPARATOR_UNIFORM_EXACT_FLOOR_PCT"] = f"{exact_floor * 100.0:.3g}%"

    ties_t_lookup = readers._claims["bench_synthg2p_comparator_native_ties_t_lookup"]["value"]
    if not isinstance(ties_t_lookup, bool):
        raise ManuscriptVariableError(
            f"bench_synthg2p_comparator_native_ties_t_lookup must be boolean, got {ties_t_lookup!r}"
        )
    variables["BENCH_SYNTHG2P_COMPARATOR_NATIVE_TIE_WORD"] = (
        "ties" if ties_t_lookup else "does not tie"
    )

    # The pre-registered refusals, restated as render-time geometry over the
    # same recorded accuracies (protocol P2 §3.3, A1–A3). The trivial ``T``
    # lookup is compared, never gated: it is exactly right on ``T``-generated
    # data by construction, so native at or above it would be a defect in the
    # measurement, while native above it is the expected descriptive outcome.
    native = accuracies[("native", "position")]
    lookup = accuracies[("t_lookup", "position")]
    if native > lookup:
        raise ManuscriptVariableError(
            "native aligned-G2P accuracy cannot exceed the exact closed-form T "
            f"lookup on T-generated data: native={native!r}, lookup={lookup!r}"
        )
    for arm, control_name in (
        ("a1_english", "the English-transfer control (A1)"),
        ("a2_broken", "the broken-correspondence control (A2)"),
    ):
        for reading, native_reading in (
            ("position", native),
            ("exact", accuracies[("native", "exact")]),
        ):
            control = accuracies[(arm, reading)]
            if native_reading <= control:
                raise ManuscriptVariableError(
                    f"the aligned-G2P native bar failed above {control_name} on the "
                    f"{reading} reading: native={native_reading!r}, control={control!r}; "
                    "this is the pre-registered refusal, not a display state"
                )
    if native <= position_floor:
        raise ManuscriptVariableError(
            f"the aligned-G2P native bar failed above the uniform floor (A3): "
            f"native={native!r}, floor={position_floor!r}"
        )

    # The zero-shot probe arms (protocol P2 design B,
    # ``scripts/record_aligned_g2p_probe_claims.py``), read at the comparator
    # table's percent precision. The arms are read against the analytic
    # contract floor, never ranked, so the floor is rendered beside them.
    def _probe_pct(label: str) -> str:
        return f"{readers.probability(label) * 100.0:.2f}%"

    variables["BENCH_SYNTHG2P_PROBE_CONTRACT_FLOOR_PCT"] = _probe_pct(
        "bench_synthg2p_probe_contract_floor"
    )
    variables["BENCH_SYNTHG2P_PROBE_GEMMA3_POSITION_ACCURACY_PCT"] = _probe_pct(
        "bench_synthg2p_probe_gemma3_position_accuracy"
    )
    variables["BENCH_SYNTHG2P_PROBE_GEMMA3_EXACT_ACCURACY_PCT"] = _probe_pct(
        "bench_synthg2p_probe_gemma3_exact_accuracy"
    )
    variables["BENCH_SYNTHG2P_PROBE_GEMMA3_CONTRACT_COMPLIANCE_PCT"] = _probe_pct(
        "bench_synthg2p_probe_gemma3_contract_compliance"
    )
    variables["BENCH_SYNTHG2P_PROBE_QWEN25_POSITION_ACCURACY_PCT"] = _probe_pct(
        "bench_synthg2p_probe_qwen25_position_accuracy"
    )
    variables["BENCH_SYNTHG2P_PROBE_QWEN25_EXACT_ACCURACY_PCT"] = _probe_pct(
        "bench_synthg2p_probe_qwen25_exact_accuracy"
    )
    variables["BENCH_SYNTHG2P_PROBE_QWEN25_CONTRACT_COMPLIANCE_PCT"] = _probe_pct(
        "bench_synthg2p_probe_qwen25_contract_compliance"
    )
