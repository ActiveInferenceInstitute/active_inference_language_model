#!/usr/bin/env python3
"""Measure and record the external Ollama benchmark; refuse partial runs.

Every registered model is measured by default, because the ledger write is a
wholesale replacement of the ``bench_`` namespace: a run that measured only one
model would silently delete the other model's labels. Selecting a subset is
still possible for a spot check, and is then an explicit act.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import urllib.error
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _artifact_io import atomic_write_text

from ailm.benchmarks import (
    DEFAULT_CALIBRATION_N,
    DEFAULT_ENSEMBLE_N,
    DEFAULT_REPEAT_N,
    DEFAULT_SEED,
    DEFAULT_TASK_N,
    G2P_BUCKETS,
    LETTER_CENSUS,
    OLLAMA_URL,
    POWER_ALPHA,
    PROBE_MODEL_TAGS,
    PROTOCOL_VERSION,
    REFERENCE_SHARE,
    benchmark_model,
    call_cache,
    enable_call_cache,
    ollama_available,
    post_generate_uncached,
)

# Imported from the module that defines it rather than through the package
# re-export: the run-scoping builder is newer than the package's public surface,
# and adding a name to that surface is the package owner's call, not this
# recorder's.
from ailm.benchmarks.constants import (
    OLLAMA_CONTEXT_BUDGET_POLICY,
    OLLAMA_NUM_CTX,
    repeat_trial_protocols,
)
from ailm.benchmarks.ollama import current_ollama_cache_entries
from ailm.call_cache import CacheRequest, CallCache, model_digest
from ailm.claims import ClaimBatch, ClaimsLedger, load_ledger
from ailm.ledger_lock import ledger_lock_path, single_writer_lock
from ailm.replay_provenance import (
    NetworkClaimBindingError,
    generate_run_scope,
    maximum_prompt_eval_tokens,
    network_recorder_configuration,
    source_manifest,
    using_publication_ledger,
    validate_network_measurement_inputs,
    validate_run_scope,
    write_network_claim_binding,
)

#: Recorded as each entry's ``source``. A repo-relative literal, not ``__file__``:
#: an absolute path pins the ledger to one machine's checkout, and
#: ``manuscript_variables`` emits a ``*_SOURCE`` token per label, so a machine
#: path in this field is one citation away from being printed into the PDF.
#: ``stale_labels`` compares only ``Path(source).name``, so entries recorded
#: under the old absolute form are still recognized as this recorder's.
SOURCE = "scripts/record_external_benchmark_claims.py"
SIDECAR_SCHEMA = "ailm-external-benchmark-sidecar-v2"
SIDECAR_REUSE_POLICY = "never"
MEASUREMENT_SCHEMA = "ailm-external-benchmark-measurement-v1"
CACHE_AUDIT_SAMPLING = "current-endpoint-loader-context-record-digest-seeded-v3"
_MODEL_IDENTITY = re.compile(r"sha256:[0-9a-f]{64}")

#: Scope used by ``scripts/warm_call_cache.py`` when it reads this recorder's
#: configuration to plan warmable keys. The repeat-trial arm contributes no
#: warmable key — both of its calls are deliberately live — so the planner needs
#: the configuration's shape, not any particular run's tags. It fails
#: :data:`ailm.replay_provenance.RUN_SCOPE_PATTERN` by construction, so a
#: configuration built for planning can never be mistaken for a receipt's.
PLANNING_RUN_SCOPE = "warm-key-planning"


def _cache_protocols(run_scope: str) -> tuple[str, ...]:
    """Every cache-purpose namespace one run of this recorder writes."""

    return tuple(sorted((PROTOCOL_VERSION, *repeat_trial_protocols(run_scope))))


def _replay_configuration(
    *,
    selected: list[str],
    letter_n: int,
    phoneme_n: int,
    g2p_n: int,
    calibration_n: int,
    ensemble_n: int,
    repeat_n: int,
    run_scope: str,
) -> dict[str, object]:
    """Return the exact publication configuration for one benchmark run."""

    return network_recorder_configuration(
        endpoint=OLLAMA_URL,
        call_protocol=PROTOCOL_VERSION,
        model_tags=[MODELS[name] for name in selected],
        parameters={
            "measurement_protocol": PROTOCOL_VERSION,
            "logical_models": sorted(selected),
            "letter_n": letter_n,
            "phoneme_n": phoneme_n,
            "g2p_n": g2p_n,
            "calibration_n": calibration_n,
            "ensemble_n": ensemble_n,
            "repeat_n": repeat_n,
            "seed": DEFAULT_SEED,
            "run_scope": run_scope,
            "cache_protocols": list(_cache_protocols(run_scope)),
            "cache_audit_sampling": CACHE_AUDIT_SAMPLING,
            "local_context_window": OLLAMA_NUM_CTX,
            "context_budget_policy": OLLAMA_CONTEXT_BUDGET_POLICY,
        },
    )


def publication_replay_configuration(*, run_scope: str) -> dict[str, object]:
    """Canonical configuration required by the full release replay contract.

    Everything but ``run_scope`` is fixed by this module's code, which is what
    makes a receipt's configuration checkable against the checkout. The scope is
    the one field a run gets to choose, so the validator reads it from the
    receipt under audit and asks for the configuration that scope implies. That
    is not a weakening: the scope also determines ``cache_protocols``, and the
    receipt's cited call records must use exactly those protocols, so a receipt
    still cannot claim records it did not make.
    """

    return _replay_configuration(
        selected=sorted(MODELS),
        letter_n=LETTER_CENSUS,
        phoneme_n=DEFAULT_TASK_N,
        g2p_n=DEFAULT_TASK_N,
        calibration_n=DEFAULT_CALIBRATION_N,
        ensemble_n=DEFAULT_ENSEMBLE_N,
        repeat_n=DEFAULT_REPEAT_N,
        run_scope=run_scope,
    )


#: Registered reference models. These are two separate model families, not rungs
#: of one scale ladder: they share no training recipe, so the difference between
#: them is a model difference and nothing more, and the parameter counts do not
#: license a trend. The manuscript reports them as a variation panel and fits
#: nothing across them.
#:
#: The registry is the single place the benchmark's multiplicity family is sized
#: from: ``main`` builds the family as one entry per template per selected model,
#: so ``bench_family_size`` and the Bonferroni threshold derived from it follow
#: from this dict rather than from a literal written anywhere. Adding or removing
#: a model therefore retightens or loosens every threshold in the supplement, and
#: the interpretive guards in ``tests/test_benchmark_interpretations.py`` are what
#: hold the English around those thresholds to the new size.
MODELS = dict(PROBE_MODEL_TAGS)

#: The comparator's label suffix. It is not an Ollama model.
COMPARATOR = "ailm"

#: The paired p-values this benchmark reports, one template per contrast type.
#: They are excluded from the repository-wide significance family (they are one
#: external reference comparison, not repository hypotheses), so the benchmark
#: declares its own correction rather than reporting every contrast uncorrected.
#:
#: The family is the cross product of these templates with the selected models,
#: so its size is ``len(FAMILY_TEMPLATES) * len(selected)`` and is never written
#: down: a template added here, or a model added to ``MODELS``, changes
#: ``bench_family_size`` and ``bench_family_alpha`` by the same arithmetic that
#: produced them.
FAMILY_TEMPLATES = (
    "bench_letter_paired_p_{model}",
    "bench_letter_az_paired_p_{model}",
    "bench_letter_prompt_paired_p_{model}",
    "bench_phoneme_paired_p_{model}",
    "bench_g2p_paired_p_{model}",
    "bench_letter_ctx_paired_p_{model}",
)
P_VALUE_FAMILIES = {"external_reference": FAMILY_TEMPLATES}

#: Task facet for each label prefix, matched longest-prefix-first so that
#: ``bench_letter_ctx_*`` is not swallowed by ``bench_letter_*``. An explicit
#: table rather than a positional guess at the label's last segment, which
#: cannot name the facet of a protocol-level label.
_TASK_PREFIXES: tuple[tuple[str, str], ...] = (
    ("bench_letter_ctx", "letter-context"),
    ("bench_letter_short", "letter-context"),
    ("bench_letter_long", "letter-context"),
    ("bench_letter_window", "letter-context"),
    ("bench_letter_pool", "protocol"),
    ("bench_letter_az", "letter-alphabetic"),
    ("bench_letter_nonalpha", "letter-nonalphabetic"),
    ("bench_letter", "letter"),
    ("bench_phoneme", "phoneme"),
    ("bench_g2p", "g2p"),
    ("bench_calib", "calibration"),
    ("bench_run_letter", "seed-ensemble"),
    ("bench_ensemble", "seed-ensemble"),
    ("bench_determinism", "seed-ensemble"),
    ("bench_unparsable", "diagnostic"),
    ("bench_power", "protocol"),
    ("bench_family", "protocol"),
    ("bench_protocol", "protocol"),
    ("bench_seed", "protocol"),
)


def label_facets(label: str) -> tuple[str, str]:
    """Return the ``(model, task)`` facets a label's run id is built from."""
    parts = set(label.split("_"))
    model = next(
        (key for key in (*sorted(MODELS), COMPARATOR) if key in parts),
        "protocol",
    )
    task = next(
        (
            name
            for prefix, name in sorted(_TASK_PREFIXES, key=lambda item: -len(item[0]))
            if label.startswith(prefix)
        ),
        "diagnostic",
    )
    return model, task


def _model_values(name: str, result: dict) -> dict[str, object]:
    """Every label one measured model contributes."""
    values: dict[str, object] = {}
    for task, measured in (("letter", result["letter"]), ("phoneme", result["phoneme"])):
        values[f"bench_{task}_top1_{name}"] = measured["top1_accuracy"]
        values[f"bench_{task}_top1_wilson_low_{name}"] = measured["wilson_low"]
        values[f"bench_{task}_top1_wilson_high_{name}"] = measured["wilson_high"]
        values[f"bench_{task}_paired_b_{name}"] = measured["paired"]["b"]
        values[f"bench_{task}_paired_c_{name}"] = measured["paired"]["c"]
        values[f"bench_{task}_paired_p_{name}"] = measured["paired"]["p_value"]
        values[f"bench_{task}_delta_{name}"] = measured["delta"]
        values[f"bench_{task}_delta_low_{name}"] = measured["delta_low"]
        values[f"bench_{task}_delta_high_{name}"] = measured["delta_high"]
        values[f"bench_{task}_power_{name}"] = measured["power"]

    g2p = result["g2p"]
    values[f"bench_g2p_pos_acc_{name}"] = g2p["position_accuracy"]
    values[f"bench_g2p_pos_acc_wilson_low_{name}"] = g2p["position_wilson_low"]
    values[f"bench_g2p_pos_acc_wilson_high_{name}"] = g2p["position_wilson_high"]
    values[f"bench_g2p_exact_{name}"] = g2p["exact_match_rate"]
    values[f"bench_g2p_exact_wilson_low_{name}"] = g2p["exact_wilson_low"]
    values[f"bench_g2p_exact_wilson_high_{name}"] = g2p["exact_wilson_high"]
    values[f"bench_g2p_paired_b_{name}"] = g2p["paired_exact"]["b"]
    values[f"bench_g2p_paired_c_{name}"] = g2p["paired_exact"]["c"]
    values[f"bench_g2p_paired_p_{name}"] = g2p["paired_exact"]["p_value"]
    values[f"bench_g2p_delta_{name}"] = g2p["delta"]
    values[f"bench_g2p_delta_low_{name}"] = g2p["delta_low"]
    values[f"bench_g2p_delta_high_{name}"] = g2p["delta_high"]
    values[f"bench_g2p_power_{name}"] = g2p["power"]
    for bucket, _low, _high in G2P_BUCKETS:
        stratum = g2p["buckets"][bucket]
        values[f"bench_g2p_pos_{bucket}_{name}"] = stratum["model_position_accuracy"]
        values[f"bench_g2p_pos_{bucket}_wilson_low_{name}"] = stratum["model_position_wilson_low"]
        values[f"bench_g2p_pos_{bucket}_wilson_high_{name}"] = stratum["model_position_wilson_high"]

    alphabetic = result["letter_alphabetic"]
    values[f"bench_letter_az_top1_{name}"] = alphabetic["top1_accuracy"]
    values[f"bench_letter_az_top1_wilson_low_{name}"] = alphabetic["wilson_low"]
    values[f"bench_letter_az_top1_wilson_high_{name}"] = alphabetic["wilson_high"]
    values[f"bench_letter_az_paired_b_{name}"] = alphabetic["paired"]["b"]
    values[f"bench_letter_az_paired_c_{name}"] = alphabetic["paired"]["c"]
    values[f"bench_letter_az_paired_p_{name}"] = alphabetic["paired"]["p_value"]

    # The complement arm. Without it the supplement can say the two letter arms
    # disagree but not why, and "why" is that almost every non-alphabetic target
    # is a space that some probes never emit.
    values[f"bench_letter_nonalpha_acc_{name}"] = result["letter_non_alphabetic"]["top1_accuracy"]

    sensitivity = result["prompt_sensitivity"]
    undelimited = sensitivity["undelimited"]
    values[f"bench_letter_undelimited_top1_{name}"] = undelimited["top1_accuracy"]
    values[f"bench_letter_undelimited_wilson_low_{name}"] = undelimited["wilson_low"]
    values[f"bench_letter_undelimited_wilson_high_{name}"] = undelimited["wilson_high"]
    values[f"bench_letter_prompt_paired_b_{name}"] = sensitivity["b"]
    values[f"bench_letter_prompt_paired_c_{name}"] = sensitivity["c"]
    values[f"bench_letter_prompt_paired_p_{name}"] = sensitivity["p_value"]
    values[f"bench_letter_prompt_delta_{name}"] = sensitivity["delta"]
    values[f"bench_letter_prompt_delta_low_{name}"] = sensitivity["delta_low"]
    values[f"bench_letter_prompt_delta_high_{name}"] = sensitivity["delta_high"]

    context = result["context"]
    values[f"bench_letter_short_top1_{name}"] = context["short_top1_accuracy"]
    values[f"bench_letter_short_top1_wilson_low_{name}"] = context["short_wilson_low"]
    values[f"bench_letter_short_top1_wilson_high_{name}"] = context["short_wilson_high"]
    values[f"bench_letter_long_top1_{name}"] = context["long_top1_accuracy"]
    values[f"bench_letter_long_top1_wilson_low_{name}"] = context["long_wilson_low"]
    values[f"bench_letter_long_top1_wilson_high_{name}"] = context["long_wilson_high"]
    values[f"bench_letter_ctx_paired_b_{name}"] = context["paired"]["b"]
    values[f"bench_letter_ctx_paired_c_{name}"] = context["paired"]["c"]
    values[f"bench_letter_ctx_paired_p_{name}"] = context["paired"]["p_value"]

    # Captured-mass diagnostics are always recorded. Calibration scores are
    # recorded only when the predeclared captured-mass gate passes; otherwise a
    # token would invite interpretation of a distribution whose observed mass is
    # insufficient. The gate and raw distribution diagnostics remain visible.
    calibration = result["calibration"]
    values[f"bench_calib_captured_mass_{name}"] = calibration["captured_mass"]
    values[f"bench_calib_captured_median_{name}"] = calibration["captured_median"]
    values[f"bench_calib_starved_fraction_{name}"] = calibration["starved_fraction"]
    # Decision 93: measurability is the direct dual-rule divergence check, and
    # the calibration metrics are recorded for every probe whether the gate
    # admits it or not. A withheld system is an audited system: its figures sit
    # in the ledger so the gate's verdict can be recomputed, while the
    # supplement's reporting rule decides what appears as a reportable row.
    values[f"bench_calib_measurable_{name}"] = int(calibration["measurable"])
    values[f"bench_calib_ece_{name}"] = calibration["model_ece"]
    values[f"bench_calib_brier_{name}"] = calibration["model_brier"]
    values[f"bench_calib_nll_{name}"] = calibration["model_nll"]
    values[f"bench_calib_confidence_{name}"] = calibration["model_confidence"]
    values[f"bench_calib_top1_{name}"] = calibration["model_top1_accuracy"]
    values[f"bench_calib_unlocated_{name}"] = calibration["unlocated"]
    values[f"bench_calib_argmax_agreement_{name}"] = calibration["argmax_agreement"]

    # The direct per-model rule check, recorded unconditionally — including for
    # any model the captured-mass gate withholds calibration scores for. Captured
    # mass is a proxy for how much of a reported number the completion rule
    # decides, and the offline sweep found the proxy is not conservative at the
    # declared floor. The check below measures the quantity itself on this
    # model's own reported distributions, so gating it on the gate would leave
    # the proxy unfalsifiable on exactly the models that fail it.
    #
    # Under the current registry the gate withholds nothing: both registered
    # probes clear the declared floor. That makes the unconditional recording
    # load-bearing in the other direction — it is now the ledger's only evidence
    # about how much of each published calibration number the completion rule
    # decides, because the gate itself distinguishes no probe from any other.
    rule = calibration["rule_check"]
    values[f"bench_calib_rule_minbound_ece_{name}"] = rule["min_bound_ece"]
    values[f"bench_calib_rule_observed_ece_{name}"] = rule["observed_only_ece"]
    values[f"bench_calib_rule_ece_signed_{name}"] = rule["ece_signed"]
    values[f"bench_calib_rule_ece_divergence_{name}"] = rule["ece_divergence"]
    values[f"bench_calib_rule_minbound_brier_{name}"] = rule["min_bound_brier"]
    values[f"bench_calib_rule_observed_brier_{name}"] = rule["observed_only_brier"]
    values[f"bench_calib_rule_brier_signed_{name}"] = rule["brier_signed"]
    values[f"bench_calib_rule_brier_divergence_{name}"] = rule["brier_divergence"]
    values[f"bench_calib_rule_joint_divergence_{name}"] = rule["joint_divergence"]
    values[f"bench_calib_rule_tv_{name}"] = rule["rule_tv"]
    # Positions where neither rule had anything to reconstruct. They contribute
    # zero divergence by construction, so the divergences above are only as
    # meaningful as this count is small, and a reader who cannot see it cannot
    # tell agreement from absence.
    values[f"bench_calib_rule_degenerate_{name}"] = rule["degenerate"]
    # A recorded verdict of the direct check, not a gate: the calibration arm is
    # still admitted on captured mass. Writing the verdict is what lets the
    # supplement compare the two criteria model by model.
    values[f"bench_calib_rule_agrees_{name}"] = int(rule["agrees"])

    ensemble = result["ensemble"]
    for index, accuracy in enumerate(ensemble["accuracies"], start=1):
        values[f"bench_run_letter_top1_{name}_seed{index}"] = accuracy
    values[f"bench_run_letter_spread_{name}"] = ensemble["spread"]
    values[f"bench_determinism_agreement_{name}"] = result["repeat"]["agreement"]

    values[f"bench_unparsable_{name}"] = (
        result["letter"]["unparsable_count"]
        + result["phoneme"]["unparsable_count"]
        + g2p["unparsable_count"]
    )
    return values


def _comparator_values(result: dict) -> dict[str, object]:
    """Labels for the AILM comparator, which is identical across probe models."""
    letter, phoneme, g2p = result["letter"], result["phoneme"], result["g2p"]
    values: dict[str, object] = {
        "bench_letter_top1_ailm": letter["comparator_top1_accuracy"],
        "bench_letter_top1_wilson_low_ailm": letter["comparator_wilson_low"],
        "bench_letter_top1_wilson_high_ailm": letter["comparator_wilson_high"],
        "bench_phoneme_top1_ailm": phoneme["comparator_top1_accuracy"],
        "bench_phoneme_top1_wilson_low_ailm": phoneme["comparator_wilson_low"],
        "bench_phoneme_top1_wilson_high_ailm": phoneme["comparator_wilson_high"],
        "bench_g2p_pos_acc_ailm": g2p["comparator_position_accuracy"],
        "bench_g2p_pos_acc_wilson_low_ailm": g2p["comparator_position_wilson_low"],
        "bench_g2p_pos_acc_wilson_high_ailm": g2p["comparator_position_wilson_high"],
        "bench_g2p_pos_acc_lev_ailm": g2p["comparator_levenshtein_position_accuracy"],
        "bench_g2p_exact_ailm": g2p["comparator_exact_match_rate"],
        "bench_g2p_exact_wilson_low_ailm": g2p["comparator_exact_wilson_low"],
        "bench_g2p_exact_wilson_high_ailm": g2p["comparator_exact_wilson_high"],
        "bench_letter_az_top1_ailm": result["letter_alphabetic"]["comparator_top1_accuracy"],
        "bench_letter_az_top1_wilson_low_ailm": result["letter_alphabetic"][
            "comparator_wilson_low"
        ],
        "bench_letter_az_top1_wilson_high_ailm": result["letter_alphabetic"][
            "comparator_wilson_high"
        ],
        "bench_letter_nonalpha_acc_ailm": result["letter_non_alphabetic"][
            "comparator_top1_accuracy"
        ],
        "bench_letter_short_top1_ailm": result["context"]["comparator_short_top1_accuracy"],
        "bench_letter_short_top1_wilson_low_ailm": result["context"]["comparator_short_wilson_low"],
        "bench_letter_short_top1_wilson_high_ailm": result["context"][
            "comparator_short_wilson_high"
        ],
        "bench_letter_long_top1_ailm": result["context"]["comparator_long_top1_accuracy"],
        "bench_letter_long_top1_wilson_low_ailm": result["context"]["comparator_long_wilson_low"],
        "bench_letter_long_top1_wilson_high_ailm": result["context"]["comparator_long_wilson_high"],
        "bench_calib_ece_ailm": result["calibration"]["comparator_ece"],
        "bench_calib_brier_ailm": result["calibration"]["comparator_brier"],
        "bench_calib_nll_ailm": result["calibration"]["comparator_nll"],
        "bench_calib_confidence_ailm": result["calibration"]["comparator_confidence"],
        "bench_calib_top1_ailm": result["calibration"]["comparator_top1_accuracy"],
    }
    for bucket, _low, _high in G2P_BUCKETS:
        stratum = g2p["buckets"][bucket]
        values[f"bench_g2p_pos_{bucket}_ailm"] = stratum["comparator_position_accuracy"]
        values[f"bench_g2p_pos_{bucket}_wilson_low_ailm"] = stratum[
            "comparator_position_wilson_low"
        ]
        values[f"bench_g2p_pos_{bucket}_wilson_high_ailm"] = stratum[
            "comparator_position_wilson_high"
        ]
    return values


def stale_labels(ledger: ClaimsLedger, produced: dict[str, object]) -> list[str]:
    """Labels this recorder must clear before writing, and no others.

    A label from a previous run that this run no longer produces has to go, or a
    renamed measurement would leave its old value behind looking current. But
    the ``bench_`` namespace is shared: another recorder writes ``bench_synth_``
    labels for the bounded stipulated-language exact-artifact-overlap probe.
    Clearing the namespace
    wholesale would silently delete a different producer's measurements on every
    run of this one, so ownership is decided by the recorded ``source`` rather
    than by the label prefix.
    """
    mine = Path(__file__).name
    return sorted(
        label
        for label in ledger.all_labels()
        if label not in produced
        and Path(str((ledger.get(label) or {}).get("source", ""))).name == mine
    )


def _canonical_json(value: object) -> str:
    """Strict canonical JSON used by the result and receipt bindings."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Publish one complete JSON object through a unique sibling temporary."""
    atomic_write_text(path, _canonical_json(payload) + "\n")


def _source_binding(project_root: Path) -> dict[str, object]:
    """Digest every Python source and vendored input the benchmark can consume."""
    return source_manifest(project_root, SOURCE)


_RESULT_ROOT_FIELDS = {
    "model",
    "seed",
    "protocol_version",
    "letter_pool_total",
    "letter_pool_unique",
    "letter_positions_scored",
    "letter",
    "letter_alphabetic",
    "prompt_sensitivity",
    "letter_non_alphabetic",
    "phoneme",
    "g2p",
    "context",
    "calibration",
    "ensemble",
    "repeat",
}


def _validate_result_tree(value: object, path: str = "result") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_result_tree(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{path} contains a non-string JSON key")
        for key, item in value.items():
            _validate_result_tree(item, f"{path}.{key}")
            if key == "n" and (isinstance(item, bool) or not isinstance(item, int) or item < 0):
                raise ValueError(f"{path}.{key} must be a non-negative integer")
        return
    raise ValueError(f"{path} contains non-JSON value {type(value).__name__}")


def _validate_live_result(result: dict, *, model_tag: str, settings: dict) -> None:
    """Validate identity, protocol, top-level schema, counts, and finite values."""
    if not isinstance(result, dict) or set(result) != _RESULT_ROOT_FIELDS:
        raise ValueError(
            f"benchmark result root fields must be exactly {sorted(_RESULT_ROOT_FIELDS)}"
        )
    if result["model"] != model_tag:
        raise ValueError("benchmark result model does not match the measured model tag")
    if result["protocol_version"] != settings["protocol"]:
        raise ValueError("benchmark result protocol does not match the run settings")
    if result["seed"] != settings["seed"]:
        raise ValueError("benchmark result seed does not match the run settings")
    for name in ("letter_pool_total", "letter_pool_unique", "letter_positions_scored"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"benchmark result {name} must be a non-negative integer")
    _validate_result_tree(result)
    _canonical_json(result)


def _cache_binding(
    store: CallCache | None,
    keys: set[str] | frozenset[str],
    *,
    model_tag: str,
    model_identity: str,
    cache_protocols: tuple[str, ...],
) -> dict[str, object]:
    """Describe the exact current records consumed while measuring one model."""
    if store is None:
        if keys:
            raise ValueError("a disabled call store cannot report consumed keys")
        return {
            "enabled": False,
            "measurement": "live",
            "keys": [],
            "keys_sha256": _json_digest([]),
        }
    binding = store.provenance_binding(keys)
    if not keys:
        raise ValueError("an enabled call store produced no identity-bound current records")
    if binding["models"] != [model_tag]:
        raise ValueError("sidecar call-store keys do not belong to the measured model")
    if binding["identities"] != [model_identity]:
        raise ValueError("sidecar call-store keys do not use the measured model identity")
    if binding["protocols"] != list(cache_protocols):
        raise ValueError("sidecar call-store keys do not use this run's benchmark protocols")
    if binding["endpoints"] != [OLLAMA_URL.rstrip("/")]:
        raise ValueError("sidecar call-store keys do not use the benchmark endpoint")
    return {"enabled": True, "measurement": "live-or-v2-replay", **binding}


def _write_sidecar(
    path: Path,
    *,
    logical_name: str,
    model_tag: str,
    model_identity: str,
    settings: dict,
    source: dict[str, object],
    cache: dict[str, object],
    result: dict,
) -> dict:
    """Write an audit receipt that is never a claim-recording input.

    Per-model replay was retired because a self-checksum cannot establish where
    a result came from. A sidecar now records direct measurement provenance only;
    resumability lives in the validated per-call store.
    """
    if _MODEL_IDENTITY.fullmatch(model_identity) is None:
        raise ValueError("benchmark sidecar model identity must be a SHA-256 digest")
    _validate_live_result(result, model_tag=model_tag, settings=settings)
    body = {
        "schema": SIDECAR_SCHEMA,
        "reuse_policy": SIDECAR_REUSE_POLICY,
        "model": {
            "logical_name": logical_name,
            "tag": model_tag,
            "identity": model_identity,
        },
        "settings": settings,
        "source": source,
        "cache": cache,
        "result": result,
        "result_sha256": _json_digest(result),
    }
    receipt = {**body, "receipt_sha256": _json_digest(body)}
    _atomic_write_json(path, receipt)
    return receipt


def _read_sidecar_receipt_for_audit(path: Path) -> dict:
    """Validate a receipt for inspection, never for measurement reuse."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema",
        "reuse_policy",
        "model",
        "settings",
        "source",
        "cache",
        "result",
        "result_sha256",
        "receipt_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("invalid benchmark sidecar fields")
    if payload["schema"] != SIDECAR_SCHEMA or payload["reuse_policy"] != "never":
        raise ValueError("benchmark sidecar is not a non-reusable current receipt")
    model = payload["model"]
    if not isinstance(model, dict) or set(model) != {"logical_name", "tag", "identity"}:
        raise ValueError("invalid benchmark sidecar model identity")
    if not all(isinstance(model[name], str) and model[name] for name in model):
        raise ValueError("benchmark sidecar model identity fields must be non-empty strings")
    if _MODEL_IDENTITY.fullmatch(model["identity"]) is None:
        raise ValueError("benchmark sidecar model identity must be a SHA-256 digest")
    if not isinstance(payload["settings"], dict):
        raise ValueError("invalid benchmark sidecar settings")
    _validate_live_result(payload["result"], model_tag=model["tag"], settings=payload["settings"])
    if payload["result_sha256"] != _json_digest(payload["result"]):
        raise ValueError("benchmark sidecar result digest mismatch")
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload["receipt_sha256"] != _json_digest(body):
        raise ValueError("benchmark sidecar receipt digest mismatch")
    return payload


def _write_cache_receipt(path: Path, settings: dict, selected: list[str]) -> None:
    """Record what this process measured versus what it replayed.

    A run that measured every call and a run that replayed every call produce
    byte-identical claims. The difference is real and is not recoverable from the
    ledger, so it is written down here or it is lost. ``fresh_fraction`` near
    zero is not a defect — that is a resume working — but it does mean this
    process is not evidence that the server still answers this way.
    """
    store = call_cache()
    receipt: dict[str, object] = {
        "settings": settings,
        "models": sorted(selected),
        "cache": store.receipt() if store is not None else {"enabled": False},
    }
    atomic_write_text(path, json.dumps(receipt, indent=2, sort_keys=True))
    if store is not None:
        stats = store.stats
        print(
            f"call store: {stats.hits} replayed, {stats.misses} measured, "
            f"{stats.unidentified} unidentified (fresh fraction "
            f"{store.stats.fresh_fraction:.3f})",
            flush=True,
        )


def _claim_values(
    *,
    selected: list[str],
    results: Mapping[str, dict],
    phoneme_n: int,
    g2p_n: int,
) -> dict[str, object]:
    """Derive this recorder's complete scientific namespace from model results."""

    if not selected or set(results) != set(selected):
        raise NetworkClaimBindingError("external benchmark results do not cover selected models")
    first = results[selected[0]]
    family = [template.format(model=name) for name in selected for template in FAMILY_TEMPLATES]
    values: dict[str, object] = {
        "bench_seed": DEFAULT_SEED,
        "bench_protocol_version": PROTOCOL_VERSION,
        "bench_letter_positions": first["letter_positions_scored"],
        "bench_phoneme_positions": phoneme_n,
        "bench_g2p_words": g2p_n,
        "bench_letter_pool_total": first["letter_pool_total"],
        "bench_letter_pool_unique": first["letter_pool_unique"],
        "bench_letter_az_positions": first["letter_alphabetic"]["n"],
        "bench_letter_nonalpha_positions": first["letter_non_alphabetic"]["n"],
        "bench_letter_window_short": first["context"]["window_short"],
        "bench_letter_window_long": first["context"]["window_long"],
        "bench_calib_positions": first["calibration"]["n"],
        "bench_calib_bins": first["calibration"]["bins"],
        "bench_calib_top_logprobs": first["calibration"]["top_logprobs"],
        "bench_calib_mass_floor": first["calibration"]["mass_floor"],
        "bench_calib_rule_epsilon": first["calibration"]["rule_check"]["epsilon"],
        "bench_ensemble_positions": first["ensemble"]["n"],
        "bench_determinism_positions": first["repeat"]["n"],
        "bench_power_alpha": POWER_ALPHA,
        "bench_power_reference_share": REFERENCE_SHARE,
        "bench_family_size": len(family),
        "bench_family_alpha": POWER_ALPHA / len(family),
        "bench_unparsable_denominator": (first["letter_positions_scored"] + phoneme_n + g2p_n),
    }
    for bucket, _low, _high in G2P_BUCKETS:
        values[f"bench_g2p_bucket_{bucket}_words"] = first["g2p"]["buckets"][bucket]["words"]
        # The denominator of every per-bucket position accuracy and its interval.
        values[f"bench_g2p_bucket_{bucket}_phones"] = first["g2p"]["buckets"][bucket]["gold_phones"]
    for name in selected:
        values.update(_model_values(name, results[name]))
        unparsable = values[f"bench_unparsable_{name}"]
        denominator = values["bench_unparsable_denominator"]
        if not isinstance(unparsable, (int, float)) or not isinstance(denominator, (int, float)):
            raise TypeError("benchmark unparsable counts must be numeric")
        values[f"bench_unparsable_rate_{name}"] = unparsable / denominator
    values.update(_comparator_values(first))
    return values


def _configuration_parameters(configuration: Mapping[str, object]) -> dict[str, object]:
    """Validate and return the exact external semantic-replay parameters."""

    parameters = configuration.get("parameters")
    expected = {
        "measurement_protocol",
        "logical_models",
        "letter_n",
        "phoneme_n",
        "g2p_n",
        "calibration_n",
        "ensemble_n",
        "repeat_n",
        "seed",
        "run_scope",
        "cache_protocols",
        "cache_audit_sampling",
        "local_context_window",
        "context_budget_policy",
    }
    if not isinstance(parameters, dict) or set(parameters) != expected:
        raise NetworkClaimBindingError(
            "external benchmark configuration has invalid semantic parameters"
        )
    selected = parameters["logical_models"]
    if (
        not isinstance(selected, list)
        or not selected
        or any(not isinstance(name, str) or name not in MODELS for name in selected)
        or selected != sorted(set(selected))
    ):
        raise NetworkClaimBindingError("external benchmark logical model registry is invalid")
    if parameters["measurement_protocol"] != PROTOCOL_VERSION:
        raise NetworkClaimBindingError("external benchmark measurement protocol drifted")
    if parameters["seed"] != DEFAULT_SEED:
        raise NetworkClaimBindingError("external benchmark seed drifted")
    if parameters["cache_audit_sampling"] != CACHE_AUDIT_SAMPLING:
        raise NetworkClaimBindingError("external benchmark cache audit sampling policy drifted")
    if (
        type(parameters["local_context_window"]) is not int
        or parameters["local_context_window"] != OLLAMA_NUM_CTX
    ):
        raise NetworkClaimBindingError("external benchmark local context window drifted")
    if parameters["context_budget_policy"] != OLLAMA_CONTEXT_BUDGET_POLICY:
        raise NetworkClaimBindingError("external benchmark context budget policy drifted")
    # The run scope is the one configuration field a run chooses, so it is the
    # one that has to be checked for shape rather than for equality with code:
    # only a scope this recorder could have generated is publishable, and the
    # cache-purpose protocols must be exactly the ones that scope implies.
    scope = validate_run_scope(parameters["run_scope"])
    if parameters["cache_protocols"] != list(_cache_protocols(scope)):
        raise NetworkClaimBindingError(
            "external benchmark cache-purpose protocols do not match the declared run scope"
        )
    model_tags = configuration.get("model_tags")
    if model_tags != sorted(MODELS[name] for name in selected):
        raise NetworkClaimBindingError("external benchmark model tags do not match logical models")
    for name in ("phoneme_n", "g2p_n", "calibration_n", "ensemble_n", "repeat_n"):
        value = parameters[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise NetworkClaimBindingError(f"external benchmark {name} must be positive")
    letter_n = parameters["letter_n"]
    if (
        not isinstance(letter_n, int)
        or isinstance(letter_n, bool)
        or (letter_n != LETTER_CENSUS and letter_n <= 0)
    ):
        raise NetworkClaimBindingError("external benchmark letter_n is invalid")
    return parameters


def _rederive_claim_values(
    receipt: Mapping[str, object],
    store: CallCache,
) -> tuple[dict[str, object], int]:
    """Replay exact cached requests and independently rebuild every owned claim."""

    configuration = receipt.get("configuration")
    cache = receipt.get("cache")
    if not isinstance(configuration, dict) or not isinstance(cache, dict):
        raise NetworkClaimBindingError("external benchmark receipt replay envelope is invalid")
    parameters = _configuration_parameters(configuration)
    endpoint = configuration.get("endpoint")
    if endpoint != OLLAMA_URL:
        raise NetworkClaimBindingError(
            "external benchmark endpoint differs from selected configuration"
        )
    protocol = configuration.get("call_protocol")
    cache_keys = cache.get("keys")
    identity_rows = cache.get("model_identities")
    if (
        not isinstance(endpoint, str)
        or protocol != PROTOCOL_VERSION
        or not isinstance(cache_keys, list)
        or any(not isinstance(key, str) for key in cache_keys)
        or not isinstance(identity_rows, list)
    ):
        raise NetworkClaimBindingError("external benchmark receipt cache identity is invalid")
    identities = {
        row["model"]: row["identity"]
        for row in identity_rows
        if isinstance(row, dict)
        and isinstance(row.get("model"), str)
        and isinstance(row.get("identity"), str)
    }
    selected_names = parameters["logical_models"]
    if not isinstance(selected_names, list):
        raise TypeError("parameters['logical_models'] must be a list")
    selected_tags = {MODELS[name] for name in selected_names}
    if set(identities) != selected_tags or len(set(identities.values())) != len(identities):
        raise NetworkClaimBindingError("external benchmark cached model identities are incomplete")

    selected_keys = set(cache_keys)
    cached = {
        key: (payload, reply) for key, payload, reply in store.entries() if key in selected_keys
    }
    if set(cached) != selected_keys:
        raise NetworkClaimBindingError("external semantic replay is missing cached replies")

    expected_keys: set[str] = set()
    logical_calls = 0

    def response_lookup(payload: dict[str, object], cache_protocol: str) -> dict[str, object]:
        nonlocal logical_calls
        logical_calls += 1
        model = payload.get("model")
        if not isinstance(model, str) or model not in identities:
            raise NetworkClaimBindingError(
                "external semantic replay request names an unknown model"
            )
        request = CacheRequest(
            payload=payload,
            endpoint=endpoint,
            protocol=cache_protocol,
            identity=identities[model],
        )
        expected_keys.add(request.key)
        record = cached.get(request.key)
        if record is None or record[0] != request.payload_snapshot():
            raise NetworkClaimBindingError(
                "external semantic replay lacks an exact protocol-derived request"
            )
        return record[1]

    # Replay the repeat trials under the tags the receipt itself declares, not
    # under a constant. A receipt is evidence about the run that wrote it, so an
    # older receipt replays against its own records and a newer one against its
    # own; reading the constant here would have made every receipt but the last
    # one unreplayable the moment run scoping began.
    trial_protocols = repeat_trial_protocols(cast(str, parameters["run_scope"]))
    results: dict[str, dict] = {}
    for name in selected_names:
        results[name] = benchmark_model(
            MODELS[name],
            repeat_trial_protocols=trial_protocols,
            letter_n=parameters["letter_n"],
            phoneme_n=parameters["phoneme_n"],
            g2p_n=parameters["g2p_n"],
            calibration_n=parameters["calibration_n"],
            ensemble_n=parameters["ensemble_n"],
            repeat_n=parameters["repeat_n"],
            seed=DEFAULT_SEED,
            include_prompt_sensitivity=True,
            response_lookup=response_lookup,
        )
    if expected_keys != selected_keys:
        raise NetworkClaimBindingError(
            "external receipt cache key set is not the exact protocol-derived request set"
        )
    return (
        _claim_values(
            selected=selected_names,
            results=results,
            phoneme_n=cast(int, parameters["phoneme_n"]),
            g2p_n=cast(int, parameters["g2p_n"]),
        ),
        logical_calls,
    )


def _measurement_evidence(
    *,
    cache_hits: int,
    cache_misses: int,
    unique_cache_keys: int,
    logical_calls: int,
    maximum_prompt_tokens: int,
    cache_protocols: tuple[str, ...],
) -> dict[str, object]:
    """Bind execution accounting without promoting process state to claims."""

    total = cache_hits + cache_misses
    return {
        "schema": MEASUREMENT_SCHEMA,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_fresh_fraction": cache_misses / total if total else 0.0,
        "logical_calls": logical_calls,
        "unique_cache_keys": unique_cache_keys,
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "cache_protocols": list(cache_protocols),
    }


def validate_publication_measurement(receipt: Mapping[str, object], store: CallCache) -> None:
    """Require cached replies to reproduce the complete external claim namespace."""

    derived, logical_calls = _rederive_claim_values(receipt, store)
    cache = receipt.get("cache")
    measurement = receipt.get("measurement")
    expected_fields = {
        "schema",
        "cache_hits",
        "cache_misses",
        "cache_fresh_fraction",
        "logical_calls",
        "unique_cache_keys",
        "maximum_prompt_tokens",
        "cache_protocols",
    }
    if (
        not isinstance(cache, dict)
        or not isinstance(cache.get("keys"), list)
        or not isinstance(measurement, dict)
        or set(measurement) != expected_fields
    ):
        raise NetworkClaimBindingError("external receipt has invalid measurement evidence")
    if measurement["schema"] != MEASUREMENT_SCHEMA:
        raise NetworkClaimBindingError("unsupported external measurement schema")
    # Re-validated rather than carried out of the replay: the measurement block
    # and the configuration are two independent statements about the same run,
    # and this is where they are required to agree about which run that was.
    configuration = receipt.get("configuration")
    if not isinstance(configuration, Mapping):
        raise NetworkClaimBindingError("external receipt has no configuration to bind against")
    declared_scope = cast(str, _configuration_parameters(configuration)["run_scope"])
    if measurement["cache_protocols"] != list(_cache_protocols(declared_scope)):
        raise NetworkClaimBindingError(
            "external receipt measurement protocols disagree with its declared run scope"
        )
    hits, misses = measurement["cache_hits"], measurement["cache_misses"]
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (hits, misses)
    ):
        raise NetworkClaimBindingError("external receipt cache accounting is invalid")
    if hits + misses != logical_calls or measurement["logical_calls"] != logical_calls:
        raise NetworkClaimBindingError(
            "external receipt cache accounting does not cover every logical call"
        )
    expected_fraction = misses / logical_calls
    if measurement["cache_fresh_fraction"] != expected_fraction:
        raise NetworkClaimBindingError("external receipt fresh fraction is inconsistent")
    cache_keys = cache["keys"]
    if measurement["unique_cache_keys"] != len(cache_keys) or measurement[
        "maximum_prompt_tokens"
    ] != maximum_prompt_eval_tokens(store, cache_keys):
        raise NetworkClaimBindingError(
            "external receipt execution evidence disagrees with cached replies"
        )

    claims = receipt.get("claims")
    entries = claims.get("entries") if isinstance(claims, dict) else None
    if not isinstance(entries, dict):
        raise NetworkClaimBindingError("external receipt claim projection is invalid")
    recorded = {
        label: entry.get("value")
        for label, entry in entries.items()
        if isinstance(label, str) and isinstance(entry, dict)
    }
    mismatches = {
        label: (recorded.get(label), expected)
        for label, expected in derived.items()
        if recorded.get(label) != expected
    }
    missing_or_extra = sorted(set(recorded) ^ set(derived))
    if mismatches or missing_or_extra:
        raise NetworkClaimBindingError(
            "external receipt claims do not rederive from cached replies: "
            f"mismatches={mismatches}, missing_or_extra={missing_or_extra}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", choices=sorted(MODELS))
    parser.add_argument("--letter-n", type=int, default=LETTER_CENSUS)
    parser.add_argument("--phoneme-n", type=int, default=DEFAULT_TASK_N)
    parser.add_argument("--g2p-n", type=int, default=DEFAULT_TASK_N)
    parser.add_argument("--calibration-n", type=int, default=DEFAULT_CALIBRATION_N)
    parser.add_argument("--ensemble-n", type=int, default=DEFAULT_ENSEMBLE_N)
    parser.add_argument("--repeat-n", type=int, default=DEFAULT_REPEAT_N)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "retained for explicit compatibility diagnostics; publication recording "
            "refuses this mode because exact persisted calls are required for replay binding"
        ),
    )
    parser.add_argument(
        "--ignore-sidecars",
        action="store_true",
        help=(
            "compatibility flag: per-model sidecar reuse is permanently disabled; "
            "individual authenticated v2 calls still resume from the call store"
        ),
    )
    parser.add_argument(
        "--verify-cache",
        type=int,
        default=0,
        metavar="N",
        help=(
            "re-measure N calls already in the store and require byte-identical replies; "
            "this is the audit that keeps 'reproducible' meaning the model agreed rather "
            "than that the disk did"
        ),
    )
    args = parser.parse_args()
    selected = args.models or sorted(MODELS)
    if args.models and args.verify_cache:
        parser.error("--verify-cache audits every registered model; omit positional models")
    if args.verify_cache:
        # An audit, not a measurement: it writes no claims, so it needs no lock
        # and can run while a census holds one.
        return _verify_cache(args.verify_cache)
    if args.no_cache:
        parser.error(
            "--no-cache cannot update publication claims: it leaves no exact call records "
            "for the network-claim replay binding. For a wholly fresh run, archive the "
            "existing ignored store and run normally so every new reply is durably bound."
        )
    if set(selected) != set(MODELS):
        parser.error(
            "partial model measurements cannot update this recorder's atomic claim "
            "namespace; run the complete registered model set"
        )
    selected = sorted(MODELS)
    with single_writer_lock(ledger_lock_path(Path(__file__).resolve().parents[1])):
        return _measure_and_record(args, selected)


#: How many stored calls the recorder re-measures before it writes anything. Small
#: on purpose: this is a precondition, not the audit. It costs a few seconds at the
#: start of a multi-hour run and catches a store the server has stopped agreeing
#: with, which is the failure that would otherwise be recorded as a measurement.
#: ``--verify-cache`` is where a real sample is checked.
PRECONDITION_SAMPLE = 12
CACHE_VERIFY_OK = 0
CACHE_VERIFY_MISMATCH = 1
CACHE_VERIFY_UNREACHABLE = 2

#: Model × request-kind strata the sample must cover, identified from each
#: stored payload rather than from a label. The shipped store was dominated by
#: one model's no-logprob letter census; stratifying only on request kind would
#: still allow the other registered probe to go entirely unchecked. Crossing
#: the declared probe registry with the two request kinds makes "verified"
#: mean that every probe and every response surface contributed a real replay.
_REQUEST_KINDS: tuple[tuple[str, bool], ...] = (
    ("with-logprobs", True),
    ("without-logprobs", False),
)


def _stratum_predicate(model_tag: str, wants_logprobs: bool) -> Callable[[dict[str, object]], bool]:
    def matches(payload: dict[str, object]) -> bool:
        return str(payload.get("model", "")) == model_tag and (
            bool(payload.get("logprobs")) is wants_logprobs
        )

    return matches


_VERIFY_STRATA: tuple[tuple[str, Callable[[dict[str, object]], bool]], ...] = tuple(
    (
        f"{model_name}:{kind}",
        _stratum_predicate(model_tag, wants_logprobs),
    )
    for model_name, model_tag in sorted(MODELS.items())
    for kind, wants_logprobs in _REQUEST_KINDS
)


def _current_model_identities(*, endpoint: str) -> dict[str, str]:
    """Resolve one distinct immutable identity for every registered model tag."""

    identities: dict[str, str] = {}
    for tag in MODELS.values():
        identity = model_digest(tag, endpoint)
        if identity is None:
            raise SystemExit(
                f"call-store verification refused: immutable identity unavailable for {tag}"
            )
        identities[tag] = identity
    if len(set(identities.values())) != len(identities):
        raise SystemExit(
            "call-store verification refused: registered model tags resolve to one "
            "immutable identity"
        )
    return identities


def _verify_cache(
    sample_n: int,
    *,
    seed: int = 20260811,
    store: CallCache | None = None,
    precondition: bool = False,
    endpoint: str = OLLAMA_URL,
    allow_missing_strata: bool = False,
    expected_identities: Mapping[str, str] | None = None,
) -> int:
    """Re-measure a random sample of stored calls and require identical replies.

    Without this, a resumed run proves only that a file was read back. The store
    is keyed on everything believed to determine the answer, and this is the
    experiment that can falsify that belief: replay the recorded payload against
    the live server and compare the field the benchmark actually reads.

    Only ``response`` and position-aligned token logprobs are compared. Timing and
    per-call context fields differ between two identical requests by design, and
    asserting over them would make the audit fail for reasons that say nothing
    about whether the model answered the same way.
    """
    store = store if store is not None else enable_call_cache()
    if store is None or len(store) == 0:
        # A cold store is the normal state of a first run and has nothing to
        # disagree with, so refusing to start would make the precondition a
        # bootstrap failure. As a standalone audit an empty store is a user
        # error, which is why the two cases return differently on purpose.
        print("call store is empty; nothing to verify", flush=True)
        return CACHE_VERIFY_OK if precondition else CACHE_VERIFY_MISMATCH
    identities = (
        dict(expected_identities)
        if expected_identities is not None
        else _current_model_identities(endpoint=endpoint)
    )
    if set(identities) != set(MODELS.values()) or len(set(identities.values())) != len(identities):
        raise SystemExit(
            "call-store verification refused: expected model identities are incomplete "
            "or non-distinct"
        )
    entries = current_ollama_cache_entries(store, endpoint=endpoint)
    if not entries:
        if precondition and allow_missing_strata:
            print(
                "current loader configuration has no cached calls; cold precondition only, "
                "not an audit pass; strict complete-strata verification remains required",
                flush=True,
            )
            return CACHE_VERIFY_OK
        raise SystemExit(
            "call-store verification refused: current loader configuration has no records"
        )
    # Stratified, and re-drawn every run. A fixed seed over a fixed store checks
    # the same handful of records forever, so an adversary — or a partial
    # corruption — need only avoid those. Mixing the authenticated current-record
    # digest into the seed means the draw moves whenever trusted content changes,
    # while a run on an unchanged store still reproduces its own check. Content,
    # not merely row count, matters: replacing one row while preserving the store
    # size must not leave the audit on the same eternal sample.
    eligible_keys = frozenset(key for key, _payload, _stored in entries)
    rng = random.Random(seed ^ int(store.records_digest(eligible_keys)[:16], 16))
    per_stratum = max(1, sample_n // max(len(_VERIFY_STRATA), 1))
    sample: list[tuple[str, str, dict, dict]] = []
    empty_strata: list[str] = []
    for name, predicate in _VERIFY_STRATA:
        pool = [entry for entry in entries if predicate(entry[1])]
        if not pool:
            empty_strata.append(name)
            continue
        sample.extend(
            (name, key, payload, stored)
            for key, payload, stored in rng.sample(pool, min(per_stratum, len(pool)))
        )
    if empty_strata and (not allow_missing_strata or len(empty_strata) == len(_VERIFY_STRATA)):
        # A declared stratum contributing zero samples is the failure this
        # stratification exists to prevent, not a condition to note and pass
        # through. The previous behavior printed the emptiness and continued, so
        # a store holding only one arm drew its whole sample from that arm and
        # reported a pass -- the check "could catch drift anywhere" in its own
        # docstring while structurally covering one stratum. In the observed
        # failure, every entry lacked logprobs, so the response-distribution arm
        # had never once been verified.
        #
        # Fail closed instead. Either the store genuinely lacks that arm, in
        # which case the audit must say so rather than pass, or the predicate has
        # drifted from the payload shape, in which case a silent skip hides it.
        raise SystemExit(
            "call-store verification refused: declared stratum(s) "
            f"{empty_strata} contributed zero entries out of {len(entries)}. "
            "A stratified audit that draws from a subset of its declared strata "
            "reports coverage it does not have. Either warm the missing arm or "
            "amend _VERIFY_STRATA to declare only what this store can contain."
        )
    if empty_strata:
        # This is the bootstrap path used only before a measurement that is
        # about to populate the missing arms. It is not an audit pass: the
        # recorder performs the same check strictly after measurement and
        # refuses the ledger write unless every declared stratum then exists.
        print(
            "call-store precondition is bootstrapping missing stratum(s) "
            f"{empty_strata}; verifying the populated strata now and requiring "
            "all strata before claims are written",
            flush=True,
        )
    sample_keys = {key for _stratum, key, _payload, _stored in sample}
    sample_models = sorted({str(payload.get("model", "")) for _, _, payload, _ in sample})
    sample_binding = store.provenance_binding(sample_keys)
    expected_rows = [{"model": model, "identity": identities[model]} for model in sample_models]
    if sample_binding["model_identities"] != expected_rows:
        raise SystemExit(
            "call-store verification refused: sampled records are not bound to the "
            "currently resolved model identities"
        )
    print(
        f"verifying {len(sample)} of {len(entries)} stored call(s) against the server", flush=True
    )

    mismatches: list[str] = []
    unreachable = 0
    verified_strata: set[str] = set()
    for stratum, key, payload, stored in sample:
        try:
            fresh = post_generate_uncached(payload, endpoint=endpoint)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            # A server that cannot answer is not a store that disagrees. Counting
            # these apart is what keeps a flaky endpoint from reading as a
            # falsified key, which would send the next reader hunting a bug in
            # the cache instead of looking at the server.
            unreachable += 1
            print(f"  unreachable during verification: {exc}", flush=True)
            continue
        if _comparable(fresh) != _comparable(stored) or not _distributions_agree(
            stored, fresh, require_distributions=payload.get("logprobs") is True
        ):
            mismatches.append(key)
        else:
            verified_strata.add(stratum)
    if unreachable:
        print(
            f"{unreachable} of {len(sample)} call(s) could not be re-measured because the "
            f"server did not answer; this is not a cache mismatch",
            flush=True,
        )
    if mismatches:
        print(
            f"CACHE VERIFY FAILED: {len(mismatches)} of {len(sample)} replies changed. "
            f"The key does not name everything that determines the answer, or the server "
            f"is not deterministic under these options. First: {mismatches[0]}",
            flush=True,
        )
        return CACHE_VERIFY_MISMATCH
    if unreachable:
        # Partial reachability is still an incomplete audit. Calling it a pass
        # would let the release receipt claim every stratum reproduced when one
        # or more sampled records were never compared. A distinct code preserves
        # the diagnostic difference from a disagreement without weakening the
        # fail-closed contract at either the precondition or release boundary.
        return CACHE_VERIFY_UNREACHABLE
    changed_identities = sorted(
        model for model, identity in identities.items() if model_digest(model, endpoint) != identity
    )
    if changed_identities:
        raise SystemExit(
            "call-store verification refused: model identity changed during replay for "
            f"{changed_identities}"
        )
    checked = len(sample) - unreachable
    if not checked:
        # Nothing was actually compared. That is not evidence of disagreement,
        # but it is also not verification and therefore cannot pass a recorder
        # precondition or a standalone release audit.
        print("nothing could be verified against the server", flush=True)
        return CACHE_VERIFY_UNREACHABLE
    print(
        f"cache verified: {checked} replayed call(s) reproduced the server's answer exactly "
        f"and its aligned token distributions within {LOGPROB_TOLERANCE} in log space "
        f"across {len(verified_strata)} of {len(_VERIFY_STRATA)} declared stratum(s)",
        flush=True,
    )
    return CACHE_VERIFY_OK


#: How far a replayed position-aligned logprob may sit from a freshly measured one and
#: still count as the same reply. Not a fudge factor: the server reproduces the
#: ANSWER exactly at temperature zero and reproduces the DISTRIBUTION only
#: approximately, and those are two different claims that this audit previously
#: collapsed. Measured drift on a re-queried record is up to ~6e-2 in log space,
#: ~1e-3 in reconstructed probability, and the identity of the tail candidates in
#: a top-20 list is not stable at all. The tolerance is set on the log scale, at
#: an order of magnitude above the largest drift observed, so a genuine change of
#: reply still fails while numerical wobble does not.
LOGPROB_TOLERANCE = 0.5


def _comparable(body: dict) -> object:
    """The exactly-reproducible part of a reply: the answer string.

    Deliberately narrower than it was. Comparing the full ``logprobs`` structure
    made the audit fail on numerical drift that does not change the answer.
    Position-aligned distributions are checked separately, with a tolerance,
    by :func:`_distributions_agree`.
    """
    return json.dumps({"response": body.get("response", "")}, sort_keys=True, default=repr)


def _distributions_agree(stored: dict, fresh: dict, *, require_distributions: bool = False) -> bool:
    """Compare shared candidates only within aligned emitted-token positions.

    Candidate strings can recur at different positions with unrelated scores.
    Tail membership may change, but every observed distribution must retain a
    shared candidate at its own position. Reproducible invalid answer strings
    remain scoreable; locating a valid JSON answer slot is not required here.
    """

    def positions(body: dict) -> list[tuple[str | None, dict[str, float]]] | None:
        entries = body.get("logprobs")
        if entries is None:
            return []
        if not isinstance(entries, list):
            return None
        result = []
        for entry in entries:
            if not isinstance(entry, dict):
                return None
            token = entry.get("token")
            if "token" in entry and not isinstance(token, str):
                return None
            candidates = entry.get("top_logprobs")
            if candidates is None:
                candidates = []
            if not isinstance(candidates, list):
                return None
            values: dict[str, float] = {}
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    return None
                name = candidate.get("token")
                value = candidate.get("logprob")
                if (
                    not isinstance(name, str)
                    or name in values
                    or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                ):
                    return None
                try:
                    numeric = float(value)
                except OverflowError:
                    return None
                if not math.isfinite(numeric):
                    return None
                values[name] = numeric
            result.append((token, values))
        return result

    left, right = positions(stored), positions(fresh)
    if left is None or right is None or len(left) != len(right):
        return False
    has_distributions = any(values for _, values in left + right)
    if require_distributions and not has_distributions:
        return False
    for (left_token, left_values), (right_token, right_values) in zip(left, right, strict=True):
        if left_token != right_token or (has_distributions and left_token is None):
            return False
        if not has_distributions:
            continue
        shared = left_values.keys() & right_values.keys()
        if not shared:
            return False
        if (
            max(abs(left_values[token] - right_values[token]) for token in shared)
            > LOGPROB_TOLERANCE
        ):
            return False
    return True


def _measure_and_record(args: argparse.Namespace, selected: list[str]) -> int:
    """Measure every selected model and replace this recorder's namespace."""
    project_root = Path(__file__).resolve().parents[1]
    publication_run = using_publication_ledger(project_root)
    initial_source = source_manifest(project_root, SOURCE)
    # One scope for the whole run, fixed before the first server call, so every
    # repeat-trial record this process writes is addressed to this run and to no
    # other. A previous run's two records keep their own keys and are neither
    # read nor overwritten; the store's refusal to overwrite an existing key is
    # therefore never exercised by a re-run, and stays available for the case it
    # is actually for.
    run_scope = generate_run_scope()
    trial_protocols = repeat_trial_protocols(run_scope)
    run_cache_protocols = _cache_protocols(run_scope)
    print(f"repeat-trial run scope: {run_scope}", flush=True)
    # Select and validate the provenance store before any model-server call.
    # Isolated ledgers may opt into an absolute AILM_CALL_CACHE_PATH; a
    # publication receipt is validated against the repository store and must
    # therefore refuse a different selection rather than measure against it.
    store = enable_call_cache(
        enabled=not args.no_cache,
        require_default_path=publication_run,
    )
    missing = [name for name in selected if not ollama_available(MODELS[name])]
    if missing:
        raise SystemExit(
            f"benchmark refused: Ollama/model unavailable: {missing}; no claims written"
        )
    identities = _current_model_identities(endpoint=OLLAMA_URL)
    # Measure one model at a time and publish an audit receipt as it lands.
    #
    # The ledger is written exactly once, at the end, so that the retire and the
    # record are one atomic transition. That is the right property for the
    # ledger and the wrong one to rely on for a census-sized run: every measured
    # position is held in memory for the whole run, and an interruption at hour
    # thirty throws away thirty hours of work unless each individual server call
    # is durable. Per-model result replay is intentionally not part of this path:
    # an ordinary JSON checksum cannot authenticate a result's origin.
    settings = {
        "protocol": PROTOCOL_VERSION,
        "endpoint": OLLAMA_URL.rstrip("/"),
        "letter_n": args.letter_n,
        "phoneme_n": args.phoneme_n,
        "g2p_n": args.g2p_n,
        "calibration_n": args.calibration_n,
        "ensemble_n": args.ensemble_n,
        "repeat_n": args.repeat_n,
        "seed": DEFAULT_SEED,
    }
    cache_dir = Path(__file__).resolve().parents[1] / "output" / "bench_raw"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Resumability lives one layer down, at the individual identity-bound model
    # call. Sidecars are write-only receipts and cannot enter ``results``.
    if store is not None:
        print(f"call store: {len(store)} record(s) at {store.path}", flush=True)
        cache_status = _verify_cache(
            PRECONDITION_SAMPLE,
            store=store,
            precondition=True,
            allow_missing_strata=True,
            expected_identities=identities,
        )
        if cache_status == CACHE_VERIFY_UNREACHABLE:
            raise SystemExit(
                "benchmark refused: the live server did not answer every sampled "
                "call-store replay; no claims written. Retry when the server is stable."
            )
        if cache_status != CACHE_VERIFY_OK:
            raise SystemExit(
                "benchmark refused: the call store disagrees with the server; no claims "
                "written. Either the key does not name everything that determines an "
                "answer, or the server no longer answers these requests the same way. "
                "Delete the store and re-measure rather than recording against it."
            )
        store.clear_access_log()

    initial_hits = store.stats.hits if store is not None else 0
    initial_misses = store.stats.misses if store is not None else 0
    results: dict[str, dict] = {}
    measurement_keys: set[str] = set()
    for name in selected:
        model_tag = MODELS[name]
        identity = identities[model_tag]
        if store is not None:
            store.clear_access_log()
        print(f"{name}: measuring", flush=True)
        results[name] = benchmark_model(
            model_tag,
            repeat_trial_protocols=trial_protocols,
            letter_n=args.letter_n,
            phoneme_n=args.phoneme_n,
            g2p_n=args.g2p_n,
            calibration_n=args.calibration_n,
            ensemble_n=args.ensemble_n,
            repeat_n=args.repeat_n,
            seed=DEFAULT_SEED,
            include_prompt_sensitivity=True,
        )
        fresh_identity = model_digest(model_tag, OLLAMA_URL)
        if fresh_identity != identity:
            raise SystemExit(
                f"benchmark refused: model identity changed while measuring {model_tag}; "
                "no sidecar or claims written"
            )
        if source_manifest(project_root, SOURCE) != initial_source:
            raise SystemExit(
                "benchmark refused: source or vendored benchmark input changed during "
                f"measurement of {model_tag}; no sidecar or claims written"
            )
        consumed_keys = store.accessed_keys if store is not None else frozenset()
        measurement_keys.update(consumed_keys)
        binding = _cache_binding(
            store,
            consumed_keys,
            model_tag=model_tag,
            model_identity=identity,
            cache_protocols=run_cache_protocols,
        )
        _write_sidecar(
            cache_dir / f"{name}.json",
            logical_name=name,
            model_tag=model_tag,
            model_identity=identity,
            settings=settings,
            source=initial_source,
            cache=binding,
            result=results[name],
        )
        print(f"{name}: measured; non-reusable provenance receipt written", flush=True)
    if store is not None:
        cache_status = _verify_cache(
            PRECONDITION_SAMPLE,
            store=store,
            expected_identities=identities,
        )
        if cache_status == CACHE_VERIFY_UNREACHABLE:
            raise SystemExit(
                "benchmark refused after measurement: the live server did not answer "
                "every sampled call-store replay; no claims written"
            )
        if cache_status != CACHE_VERIFY_OK:
            raise SystemExit(
                "benchmark refused after measurement: the completed call store did not "
                "reproduce every declared model/request-kind stratum; no claims written"
            )
    _write_cache_receipt(cache_dir / "cache_receipt.json", settings, selected)

    values = _claim_values(
        selected=selected,
        results=results,
        phoneme_n=args.phoneme_n,
        g2p_n=args.g2p_n,
    )

    if source_manifest(project_root, SOURCE) != initial_source:
        raise SystemExit(
            "benchmark refused: source or vendored benchmark input changed before "
            "ledger publication; no claims written"
        )
    if store is None:  # parser-level refusal above makes this defensive only
        raise SystemExit("benchmark claims cannot be published with the call store disabled")
    configuration = _replay_configuration(
        selected=selected,
        letter_n=args.letter_n,
        phoneme_n=args.phoneme_n,
        g2p_n=args.g2p_n,
        calibration_n=args.calibration_n,
        ensemble_n=args.ensemble_n,
        repeat_n=args.repeat_n,
        run_scope=run_scope,
    )
    replay_receipt: dict[str, object] = {
        "configuration": configuration,
        "cache": {
            **store.provenance_binding(measurement_keys),
            "record_count": len(measurement_keys),
        },
    }
    replayed_values, logical_calls = _rederive_claim_values(replay_receipt, store)
    if replayed_values != values:
        raise SystemExit(
            "benchmark refused: exact cached replies do not reproduce the measured "
            "claim namespace; no claims written"
        )
    measurement = _measurement_evidence(
        cache_hits=store.stats.hits - initial_hits,
        cache_misses=store.stats.misses - initial_misses,
        unique_cache_keys=len(measurement_keys),
        logical_calls=logical_calls,
        maximum_prompt_tokens=maximum_prompt_eval_tokens(store, measurement_keys),
        cache_protocols=run_cache_protocols,
    )
    validate_network_measurement_inputs(
        configuration=configuration,
        store=store,
        cache_keys=measurement_keys,
        measurement=measurement,
    )
    ledger = load_ledger(project_root)
    # Labels are grouped by run id so each measurement carries the facet it was
    # measured under, and the whole grouping goes to the ledger in one call.
    # Clearing the superseded labels and writing the new ones are a single
    # atomic write: as two calls they were two whole-file replacements, and a
    # crash in the window between them left the retired labels deleted, the new
    # ones absent, and a ledger that looks internally consistent while short.
    stamp = date.today().isoformat()
    batches: dict[str, dict[str, object]] = {}
    for label, value in sorted(values.items()):
        model, task = label_facets(label)
        batches.setdefault(f"ollama-bench-{stamp}-{model}-{task}", {})[label] = value
    ledger.replace_claims(
        remove=stale_labels(ledger, values),
        batches=[
            ClaimBatch(values=batch, run_id=run_id, source=SOURCE)
            for run_id, batch in sorted(batches.items())
        ],
    )
    print(f"recorded {len(values)} benchmark claims for {selected} in {len(batches)} batches")
    if publication_run:
        receipt_path = write_network_claim_binding(
            project_root=project_root,
            recorder=SOURCE,
            expected_source_manifest=initial_source,
            configuration=configuration,
            store=store,
            cache_keys=measurement_keys,
            measurement=measurement,
        )
        print(f"network claim binding: {receipt_path.relative_to(project_root)}")
    else:
        print("isolated ledger run: publication network claim binding left unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
