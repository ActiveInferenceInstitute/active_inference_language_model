"""Mock-free external reference benchmarks for AILM.

Ollama is an external probe only; no neural model is part of AILM's pymdp stack.

Accuracy arms score what each system answered, so probe and comparator are
scored on the same object. The calibration arm is not like that and the naming
here says so: the comparator is scored on its exact predictive column while a
probe is scored on a reconstruction of one, because the endpoint returns a
truncated raw candidate list rather than the constrained output distribution the
answer was drawn from. See :func:`reconstructed_calibration_metrics`.

## Package layout

The public API is re-exported lazily so callers need not follow internal splits.
Reading constants or the probe registry does not load numerical estimators;
accessing a declared function imports its owning module and returns that same
function object. Unknown attributes raise normally; import errors propagate.
``REPEAT_TRIAL_PROTOCOLS`` contains namespace prefixes, not complete cache tags;
use ``ailm.benchmarks.constants.repeat_trial_protocols(run_scope)`` to bind
those prefixes to one recorder run.

* ``ailm.benchmarks.calibration``
* ``ailm.benchmarks.comparators``
* ``ailm.benchmarks.constants``
* ``ailm.benchmarks.corpora``
* ``ailm.benchmarks.harness``
* ``ailm.benchmarks.ollama``
* ``ailm.benchmarks.prompts``
* ``ailm.benchmarks.scoring``
* ``ailm.benchmarks.synthetic_protocol``
"""

from __future__ import annotations

# ruff: noqa: F401 -- TYPE_CHECKING declarations preserve re-export types.
from importlib import import_module
from typing import TYPE_CHECKING, Any

# Preserve static types without loading model/numerical dependencies just
# to read protocol constants or the probe registry.
if TYPE_CHECKING:
    from ailm.benchmarks.calibration import (
        calibration_floor_grid,
        complete_min_bound,
        complete_observed_only,
        completion_rule_sweep,
        divergence_increases,
        divergence_sign_changes,
        dual_rule_calibration,
        fidelity_crossover,
        locate_divergence_floor,
        reconstructed_calibration_metrics,
        total_variation,
        truncate_to_mass,
    )
    from ailm.benchmarks.comparators import (
        _ailm_g2p_decode,
        _ailm_g2p_metrics,
        _letter_comparator,
        restrict_to_alphabetic,
        restrict_to_non_alphabetic,
        score_symbol_task,
    )
    from ailm.benchmarks.constants import (
        CALIBRATION_BINS,
        CALIBRATION_FLOOR_EPSILON,
        CALIBRATION_FLOOR_GRID_HIGH,
        CALIBRATION_FLOOR_GRID_LOW,
        CALIBRATION_FLOOR_GRID_STEP,
        CALIBRATION_FLOOR_IDENTITY_TOLERANCE,
        CALIBRATION_FLOOR_PROTOCOL,
        CALIBRATION_MASS_FLOOR,
        DEFAULT_CALIBRATION_N,
        DEFAULT_ENSEMBLE_N,
        DEFAULT_REPEAT_N,
        DEFAULT_SEED,
        DEFAULT_TASK_N,
        G2P_BUCKETS,
        HELD_OUT_TEXT,
        HELD_OUT_TEXT_LARGE,
        LETTER_CENSUS,
        LETTER_TEXT,
        LETTER_WINDOW_LONG,
        LETTER_WINDOW_SHORT,
        OLLAMA_URL,
        POWER_ALPHA,
        PROBE_DISPLAY_NAMES,
        PROBE_MODEL_TAGS,
        PROTOCOL_VERSION,
        REPEAT_TRIAL_PROTOCOLS,
        SEED_OFFSETS,
        SERVED_PROBE_COUNT,
        SERVED_PROBES,
        TOP_LOGPROBS,
    )
    from ailm.benchmarks.corpora import (
        letter_passage_period,
        letter_positions,
        letter_posterior,
        phone_statistics,
        phoneme_positions,
        unique_letter_positions,
    )
    from ailm.benchmarks.harness import (
        benchmark_model,
        repeat_agreement,
        seed_replicates,
    )
    from ailm.benchmarks.ollama import (
        _MODEL_IDENTITY,
        CALL_CACHE_PATH_ENV,
        _answer_token_index,
        _candidate_symbol,
        _distribution_from_candidates,
        ask_ollama,
        ask_ollama_answer_candidates,
        ask_ollama_distribution,
        call_cache,
        captured_mass,
        enable_call_cache,
        observed_symbol_mass,
        ollama_available,
        post_generate_uncached,
        resolved_model_identity,
        warm_ollama_model,
    )
    from ailm.benchmarks.prompts import (
        ALPHABETIC_SYMBOLS,
        LETTER_INVENTORY_PHRASE,
        LETTER_INVENTORY_SYMBOLS,
        SYNTHETIC_LETTER_INVENTORY_PHRASE,
        SYNTHETIC_LETTER_INVENTORY_SYMBOLS,
        build_in_context_synthetic_letter_prompt,
        build_in_context_synthetic_letter_prompts,
        build_prompt,
        build_undelimited_letter_prompt,
        derange_examples,
        draw_in_context_example_sets,
        draw_in_context_example_sets_with_diagnostics,
        parse_g2p_answer,
        parse_letter_answer,
        parse_phoneme_answer,
        parse_synthetic_letter_answer,
        subsample_positions,
    )
    from ailm.benchmarks.scoring import (
        REFERENCE_SHARE,
        PairedDiscordance,
        brier_score,
        exact_mcnemar_p,
        expected_calibration_error,
        g2p_length_bucket,
        levenshtein_align,
        negative_log_likelihood,
        paired_difference_interval,
        paired_discordance,
        paired_power,
        wilson_interval,
    )


# Explicit destinations preserve every previous re-export, including the
# private helpers used by conformance tests. Unknown attributes never import.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ALPHABETIC_SYMBOLS": ("ailm.benchmarks.prompts", "ALPHABETIC_SYMBOLS"),
    "CALIBRATION_BINS": ("ailm.benchmarks.constants", "CALIBRATION_BINS"),
    "CALIBRATION_FLOOR_EPSILON": ("ailm.benchmarks.constants", "CALIBRATION_FLOOR_EPSILON"),
    "CALIBRATION_FLOOR_GRID_HIGH": ("ailm.benchmarks.constants", "CALIBRATION_FLOOR_GRID_HIGH"),
    "CALIBRATION_FLOOR_GRID_LOW": ("ailm.benchmarks.constants", "CALIBRATION_FLOOR_GRID_LOW"),
    "CALIBRATION_FLOOR_GRID_STEP": ("ailm.benchmarks.constants", "CALIBRATION_FLOOR_GRID_STEP"),
    "CALIBRATION_FLOOR_IDENTITY_TOLERANCE": (
        "ailm.benchmarks.constants",
        "CALIBRATION_FLOOR_IDENTITY_TOLERANCE",
    ),
    "CALIBRATION_FLOOR_PROTOCOL": ("ailm.benchmarks.constants", "CALIBRATION_FLOOR_PROTOCOL"),
    "CALIBRATION_MASS_FLOOR": ("ailm.benchmarks.constants", "CALIBRATION_MASS_FLOOR"),
    "CALL_CACHE_PATH_ENV": ("ailm.benchmarks.ollama", "CALL_CACHE_PATH_ENV"),
    "DEFAULT_CALIBRATION_N": ("ailm.benchmarks.constants", "DEFAULT_CALIBRATION_N"),
    "DEFAULT_ENSEMBLE_N": ("ailm.benchmarks.constants", "DEFAULT_ENSEMBLE_N"),
    "DEFAULT_REPEAT_N": ("ailm.benchmarks.constants", "DEFAULT_REPEAT_N"),
    "DEFAULT_SEED": ("ailm.benchmarks.constants", "DEFAULT_SEED"),
    "DEFAULT_TASK_N": ("ailm.benchmarks.constants", "DEFAULT_TASK_N"),
    "G2P_BUCKETS": ("ailm.benchmarks.constants", "G2P_BUCKETS"),
    "HELD_OUT_TEXT": ("ailm.benchmarks.constants", "HELD_OUT_TEXT"),
    "HELD_OUT_TEXT_LARGE": ("ailm.benchmarks.constants", "HELD_OUT_TEXT_LARGE"),
    "LETTER_CENSUS": ("ailm.benchmarks.constants", "LETTER_CENSUS"),
    "LETTER_INVENTORY_PHRASE": ("ailm.benchmarks.prompts", "LETTER_INVENTORY_PHRASE"),
    "LETTER_INVENTORY_SYMBOLS": ("ailm.benchmarks.prompts", "LETTER_INVENTORY_SYMBOLS"),
    "LETTER_TEXT": ("ailm.benchmarks.constants", "LETTER_TEXT"),
    "LETTER_WINDOW_LONG": ("ailm.benchmarks.constants", "LETTER_WINDOW_LONG"),
    "LETTER_WINDOW_SHORT": ("ailm.benchmarks.constants", "LETTER_WINDOW_SHORT"),
    "OLLAMA_URL": ("ailm.benchmarks.constants", "OLLAMA_URL"),
    "POWER_ALPHA": ("ailm.benchmarks.constants", "POWER_ALPHA"),
    "PROBE_DISPLAY_NAMES": ("ailm.benchmarks.constants", "PROBE_DISPLAY_NAMES"),
    "PROBE_MODEL_TAGS": ("ailm.benchmarks.constants", "PROBE_MODEL_TAGS"),
    "PROTOCOL_VERSION": ("ailm.benchmarks.constants", "PROTOCOL_VERSION"),
    "PairedDiscordance": ("ailm.benchmarks.scoring", "PairedDiscordance"),
    "REFERENCE_SHARE": ("ailm.benchmarks.scoring", "REFERENCE_SHARE"),
    "REPEAT_TRIAL_PROTOCOLS": ("ailm.benchmarks.constants", "REPEAT_TRIAL_PROTOCOLS"),
    "SEED_OFFSETS": ("ailm.benchmarks.constants", "SEED_OFFSETS"),
    "SERVED_PROBES": ("ailm.benchmarks.constants", "SERVED_PROBES"),
    "SERVED_PROBE_COUNT": ("ailm.benchmarks.constants", "SERVED_PROBE_COUNT"),
    "SYNTHETIC_LETTER_INVENTORY_PHRASE": (
        "ailm.benchmarks.prompts",
        "SYNTHETIC_LETTER_INVENTORY_PHRASE",
    ),
    "SYNTHETIC_LETTER_INVENTORY_SYMBOLS": (
        "ailm.benchmarks.prompts",
        "SYNTHETIC_LETTER_INVENTORY_SYMBOLS",
    ),
    "TOP_LOGPROBS": ("ailm.benchmarks.constants", "TOP_LOGPROBS"),
    "_MODEL_IDENTITY": ("ailm.benchmarks.ollama", "_MODEL_IDENTITY"),
    "_ailm_g2p_decode": ("ailm.benchmarks.comparators", "_ailm_g2p_decode"),
    "_ailm_g2p_metrics": ("ailm.benchmarks.comparators", "_ailm_g2p_metrics"),
    "_answer_token_index": ("ailm.benchmarks.ollama", "_answer_token_index"),
    "_candidate_symbol": ("ailm.benchmarks.ollama", "_candidate_symbol"),
    "_distribution_from_candidates": ("ailm.benchmarks.ollama", "_distribution_from_candidates"),
    "_letter_comparator": ("ailm.benchmarks.comparators", "_letter_comparator"),
    "ask_ollama": ("ailm.benchmarks.ollama", "ask_ollama"),
    "ask_ollama_answer_candidates": ("ailm.benchmarks.ollama", "ask_ollama_answer_candidates"),
    "ask_ollama_distribution": ("ailm.benchmarks.ollama", "ask_ollama_distribution"),
    "benchmark_model": ("ailm.benchmarks.harness", "benchmark_model"),
    "brier_score": ("ailm.benchmarks.scoring", "brier_score"),
    "build_in_context_synthetic_letter_prompt": (
        "ailm.benchmarks.prompts",
        "build_in_context_synthetic_letter_prompt",
    ),
    "build_in_context_synthetic_letter_prompts": (
        "ailm.benchmarks.prompts",
        "build_in_context_synthetic_letter_prompts",
    ),
    "build_prompt": ("ailm.benchmarks.prompts", "build_prompt"),
    "build_undelimited_letter_prompt": (
        "ailm.benchmarks.prompts",
        "build_undelimited_letter_prompt",
    ),
    "calibration_floor_grid": ("ailm.benchmarks.calibration", "calibration_floor_grid"),
    "call_cache": ("ailm.benchmarks.ollama", "call_cache"),
    "captured_mass": ("ailm.benchmarks.ollama", "captured_mass"),
    "complete_min_bound": ("ailm.benchmarks.calibration", "complete_min_bound"),
    "complete_observed_only": ("ailm.benchmarks.calibration", "complete_observed_only"),
    "completion_rule_sweep": ("ailm.benchmarks.calibration", "completion_rule_sweep"),
    "derange_examples": ("ailm.benchmarks.prompts", "derange_examples"),
    "divergence_increases": ("ailm.benchmarks.calibration", "divergence_increases"),
    "divergence_sign_changes": ("ailm.benchmarks.calibration", "divergence_sign_changes"),
    "draw_in_context_example_sets": ("ailm.benchmarks.prompts", "draw_in_context_example_sets"),
    "draw_in_context_example_sets_with_diagnostics": (
        "ailm.benchmarks.prompts",
        "draw_in_context_example_sets_with_diagnostics",
    ),
    "dual_rule_calibration": ("ailm.benchmarks.calibration", "dual_rule_calibration"),
    "enable_call_cache": ("ailm.benchmarks.ollama", "enable_call_cache"),
    "exact_mcnemar_p": ("ailm.benchmarks.scoring", "exact_mcnemar_p"),
    "expected_calibration_error": ("ailm.benchmarks.scoring", "expected_calibration_error"),
    "fidelity_crossover": ("ailm.benchmarks.calibration", "fidelity_crossover"),
    "g2p_length_bucket": ("ailm.benchmarks.scoring", "g2p_length_bucket"),
    "letter_passage_period": ("ailm.benchmarks.corpora", "letter_passage_period"),
    "letter_positions": ("ailm.benchmarks.corpora", "letter_positions"),
    "letter_posterior": ("ailm.benchmarks.corpora", "letter_posterior"),
    "levenshtein_align": ("ailm.benchmarks.scoring", "levenshtein_align"),
    "locate_divergence_floor": ("ailm.benchmarks.calibration", "locate_divergence_floor"),
    "negative_log_likelihood": ("ailm.benchmarks.scoring", "negative_log_likelihood"),
    "observed_symbol_mass": ("ailm.benchmarks.ollama", "observed_symbol_mass"),
    "ollama_available": ("ailm.benchmarks.ollama", "ollama_available"),
    "paired_difference_interval": ("ailm.benchmarks.scoring", "paired_difference_interval"),
    "paired_discordance": ("ailm.benchmarks.scoring", "paired_discordance"),
    "paired_power": ("ailm.benchmarks.scoring", "paired_power"),
    "parse_g2p_answer": ("ailm.benchmarks.prompts", "parse_g2p_answer"),
    "parse_letter_answer": ("ailm.benchmarks.prompts", "parse_letter_answer"),
    "parse_phoneme_answer": ("ailm.benchmarks.prompts", "parse_phoneme_answer"),
    "parse_synthetic_letter_answer": ("ailm.benchmarks.prompts", "parse_synthetic_letter_answer"),
    "phone_statistics": ("ailm.benchmarks.corpora", "phone_statistics"),
    "phoneme_positions": ("ailm.benchmarks.corpora", "phoneme_positions"),
    "post_generate_uncached": ("ailm.benchmarks.ollama", "post_generate_uncached"),
    "reconstructed_calibration_metrics": (
        "ailm.benchmarks.calibration",
        "reconstructed_calibration_metrics",
    ),
    "repeat_agreement": ("ailm.benchmarks.harness", "repeat_agreement"),
    "resolved_model_identity": ("ailm.benchmarks.ollama", "resolved_model_identity"),
    "restrict_to_alphabetic": ("ailm.benchmarks.comparators", "restrict_to_alphabetic"),
    "restrict_to_non_alphabetic": ("ailm.benchmarks.comparators", "restrict_to_non_alphabetic"),
    "score_symbol_task": ("ailm.benchmarks.comparators", "score_symbol_task"),
    "seed_replicates": ("ailm.benchmarks.harness", "seed_replicates"),
    "subsample_positions": ("ailm.benchmarks.prompts", "subsample_positions"),
    "total_variation": ("ailm.benchmarks.calibration", "total_variation"),
    "truncate_to_mass": ("ailm.benchmarks.calibration", "truncate_to_mass"),
    "unique_letter_positions": ("ailm.benchmarks.corpora", "unique_letter_positions"),
    "warm_ollama_model": ("ailm.benchmarks.ollama", "warm_ollama_model"),
    "wilson_interval": ("ailm.benchmarks.scoring", "wilson_interval"),
}


def __getattr__(name: str) -> Any:
    """Load a declared export from its owner, preserving object identity."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose the complete import surface without importing its dependencies."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "ALPHABETIC_SYMBOLS",
    "CALIBRATION_BINS",
    "CALIBRATION_FLOOR_EPSILON",
    "CALIBRATION_FLOOR_GRID_HIGH",
    "CALIBRATION_FLOOR_GRID_LOW",
    "CALIBRATION_FLOOR_GRID_STEP",
    "CALIBRATION_FLOOR_IDENTITY_TOLERANCE",
    "CALIBRATION_FLOOR_PROTOCOL",
    "CALIBRATION_MASS_FLOOR",
    "CALL_CACHE_PATH_ENV",
    "DEFAULT_CALIBRATION_N",
    "DEFAULT_ENSEMBLE_N",
    "DEFAULT_REPEAT_N",
    "DEFAULT_SEED",
    "DEFAULT_TASK_N",
    "G2P_BUCKETS",
    "HELD_OUT_TEXT",
    "HELD_OUT_TEXT_LARGE",
    "LETTER_CENSUS",
    "LETTER_INVENTORY_PHRASE",
    "LETTER_INVENTORY_SYMBOLS",
    "LETTER_TEXT",
    "LETTER_WINDOW_LONG",
    "LETTER_WINDOW_SHORT",
    "OLLAMA_URL",
    "POWER_ALPHA",
    "PROTOCOL_VERSION",
    "REPEAT_TRIAL_PROTOCOLS",
    "PairedDiscordance",
    "SEED_OFFSETS",
    "SYNTHETIC_LETTER_INVENTORY_PHRASE",
    "SYNTHETIC_LETTER_INVENTORY_SYMBOLS",
    "TOP_LOGPROBS",
    "ask_ollama",
    "ask_ollama_answer_candidates",
    "ask_ollama_distribution",
    "benchmark_model",
    "brier_score",
    "build_in_context_synthetic_letter_prompt",
    "build_in_context_synthetic_letter_prompts",
    "build_prompt",
    "build_undelimited_letter_prompt",
    "calibration_floor_grid",
    "call_cache",
    "captured_mass",
    "complete_min_bound",
    "complete_observed_only",
    "completion_rule_sweep",
    "divergence_increases",
    "divergence_sign_changes",
    "dual_rule_calibration",
    "derange_examples",
    "draw_in_context_example_sets",
    "draw_in_context_example_sets_with_diagnostics",
    "enable_call_cache",
    "exact_mcnemar_p",
    "expected_calibration_error",
    "fidelity_crossover",
    "g2p_length_bucket",
    "letter_passage_period",
    "letter_positions",
    "letter_posterior",
    "levenshtein_align",
    "locate_divergence_floor",
    "negative_log_likelihood",
    "observed_symbol_mass",
    "ollama_available",
    "paired_difference_interval",
    "paired_discordance",
    "paired_power",
    "parse_g2p_answer",
    "parse_letter_answer",
    "parse_phoneme_answer",
    "parse_synthetic_letter_answer",
    "phone_statistics",
    "phoneme_positions",
    "post_generate_uncached",
    "reconstructed_calibration_metrics",
    "repeat_agreement",
    "resolved_model_identity",
    "restrict_to_alphabetic",
    "restrict_to_non_alphabetic",
    "score_symbol_task",
    "seed_replicates",
    "subsample_positions",
    "total_variation",
    "truncate_to_mass",
    "unique_letter_positions",
    "wilson_interval",
    "warm_ollama_model",
]
