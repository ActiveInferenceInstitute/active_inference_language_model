#!/usr/bin/env python3
"""Measure and record both stipulated-language exact-artifact-overlap probes.

The external benchmark (``scripts/record_external_benchmark_claims.py``) scores
AILM against a reference LLM on Brown and CMUdict. Both are public and old
enough to plausibly sit inside the reference model's training data, so neither a
win nor a loss there separates inference from recall.

This recorder narrows that confound without asserting the contents of an
external model's training data. Each stipulated language is written down in
:mod:`ailm.levels.synthetic` — a transition in code, sampled into a vendored
count artifact:

* the letter arm (``synthetic-zero-shot-v2``) over 26 lowercase letters,
  backed by ``vendor/synthetic_counts.json``;
* the phone arm (``synthetic-phone-zero-shot-v1``) over the 39 canonical
  ARPAbet phones, backed by ``vendor/synthetic_phone_counts.json``.

AILM's order-1 comparator is estimated from each arm's own corpus; every
registered reference probe meets the same independently seeded evaluation
trajectories without an offline fit or separately labeled demonstrations. The
query itself still exposes a bounded prefix from the chain, so a paired
one-symbol-prefix sensitivity arm measures how much that within-query evidence
changes each probe. Each arm uses its own prompt, response schema and parser,
exposing exactly that arm's stipulated inventory.

The two arms are one publication unit: one ledger write, and one network claim
binding whose configuration names both cache protocols and whose semantic
replay re-derives both claim namespaces from the cached replies.

Refusals — no claim is written unless every one of these holds:

* every registered reference probe is scored through the OpenRouter transport
  (``meta-llama/llama-3.3-70b-instruct``), with no local-model dependency or warm-up;
* every vendored artifact reproduces byte-for-byte from its generator, so a
  measurement cannot describe a stale corpus;
* the phone arm's fitted comparator clears the comparator refusals pinned in
  the lane's pre-registration stub (a stipulated-fit win that does not clearly
  exceed the uniform floor, a transposed-count fit, and an English-trained
  CMUdict fit refuses publication);
* every HTTP call completes; malformed or empty replies remain in the
  denominator as unparsable and wrong (the loop has no partial-credit path).

Namespace ownership
-------------------
This script, ``record_external_benchmark_claims.py``, and
``record_incontext_benchmark_claims.py`` share the ``bench_`` prefix. Ownership
is decided by the recorded ``source``, not by the prefix: each recorder's
``stale_labels`` clears only the labels it recorded itself, so none can delete
another's labels. The recorders may be run in either order, and all take the
ledger's single-writer lock so they cannot be run at the same time.

Usage
-----
    .venv/bin/python \
        scripts/record_synthetic_benchmark_claims.py

    # isolated validation, never the tracked ledger:
    AILM_CLAIMS_PATH=/tmp/synth_claims.json .venv/bin/python \
        scripts/record_synthetic_benchmark_claims.py --positions N
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ailm.benchmarks import (  # noqa: E402  (path setup must precede the import)
    build_prompt,
    call_cache,
    enable_call_cache,
    paired_discordance,
    parse_phoneme_answer,
    parse_synthetic_letter_answer,
    wilson_interval,
)
from ailm.benchmarks.constants import (  # noqa: E402
    DEFAULT_SEED,
    HOSTED_SYNTHETIC_MODEL,
    POWER_ALPHA,
    PROBE_MODEL_TAGS,
)
from ailm.benchmarks.openrouter import (  # noqa: E402  (path setup must precede the import)
    OPENROUTER_URL,
    _chat_payload,
    ask_openrouter,
    set_call_cache,
)
from ailm.benchmarks.openrouter_budget import (  # noqa: E402
    configured_provider_options,
    validate_provider_options,
)
from ailm.benchmarks.synthetic_protocol import (  # noqa: E402
    SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL,
    SYNTHETIC_ZERO_SHOT_PROTOCOL,
    SyntheticZeroShotProtocol,
)
from ailm.call_cache import CacheRequest, CallCache  # noqa: E402
from ailm.claims import ClaimBatch, load_ledger  # noqa: E402
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402
from ailm.levels.phoneme import load_phoneme_counts  # noqa: E402
from ailm.levels.synthetic import (  # noqa: E402
    DEFAULT_CORPUS_CHARS,
    DEFAULT_CORPUS_SEED,
    DEFAULT_PHONE_CORPUS_SEED,
    DEFAULT_PHONE_CORPUS_SYMBOLS,
    PHONE_UNIFORM_BASELINE,
    SYNTHETIC_LETTERS,
    SYNTHETIC_PHONES,
    UNIFORM_BASELINE,
    build_synthetic_counts,
    build_synthetic_phone_counts,
    correctness_flags,
    independent_held_out_positions,
    independent_phone_held_out_positions,
    independent_target_start_tvd,
    load_synthetic_counts,
    load_synthetic_phone_counts,
    phone_target_start_tvd,
    synthetic_letter_predictions,
    synthetic_phone_predictions,
)
from ailm.replay_provenance import (  # noqa: E402
    NetworkClaimBindingError,
    generate_run_scope,
    maximum_prompt_eval_tokens,
    network_recorder_configuration,
    scoped_cache_protocol,
    source_manifest,
    using_publication_ledger,
    validate_network_measurement_inputs,
    validate_run_scope,
    write_network_claim_binding,
)

#: Registered label suffix → model tag. This is deliberately the complete
#: shared registry: the dependent in-context recorder authenticates each
#: probe's zero-shot result, so allowing the two producer inventories to drift
#: makes the network phase impossible to complete.
MODELS = dict(PROBE_MODEL_TAGS)


#: The registered probe whose per-tag namespace scopes the receipt's primary
#: call protocol. Deterministic: the first key of the shared registry.
_PRIMARY_PROBE_KEY = sorted(MODELS)[0]

#: The OpenRouter slug every registered probe is scored through. The tags in
#: ``MODELS`` keep naming the retained ledger and call-protocol suffixes;
#: the network transport resolves one hosted model for all of them.
OPENROUTER_MODEL = HOSTED_SYNTHETIC_MODEL

#: The paired p-values the letter arm reports. The synthetic probe is a
#: separately declared descriptive contrast on a different corpus from the
#: external benchmark family; it declares its own family across every
#: registered probe rather than joining that family. Declaring it keeps the
#: invariant in tests/test_statistics.py: no bench_* p-value in the ledger is
#: corrected by nothing at all.
FAMILY_TEMPLATES = ("bench_synth_paired_p_{model}",)

#: Audit-specified sensitivity to the within-query chain evidence. This family
#: was added after the original protocol's opportunity asymmetry was found, so
#: it remains descriptive and separate from the original comparator contrasts.
CONTEXT_FAMILY_TEMPLATES = ("bench_synth_context_paired_p_{model}",)

#: The phone arm's families, corrected inside themselves exactly as the letter
#: arm's are. A distinct namespace keeps the two stipulated languages'
#: contrasts from pooling into one family that neither protocol declared.
PHONE_FAMILY_TEMPLATES = ("bench_synthphone_paired_p_{model}",)
PHONE_CONTEXT_FAMILY_TEMPLATES = ("bench_synthphone_context_paired_p_{model}",)
P_VALUE_FAMILIES = {
    "comparator": FAMILY_TEMPLATES,
    "audit_context_sensitivity": CONTEXT_FAMILY_TEMPLATES,
    "phone_comparator": PHONE_FAMILY_TEMPLATES,
    "phone_audit_context_sensitivity": PHONE_CONTEXT_FAMILY_TEMPLATES,
}

SOURCE = SYNTHETIC_ZERO_SHOT_PROTOCOL.source
if SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL.source != SOURCE:
    raise RuntimeError("the stipulated arms must be one recorder with one source")
LETTER_RUN_ID = SYNTHETIC_ZERO_SHOT_PROTOCOL.run_id
PHONE_RUN_ID = SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL.run_id
DEFAULT_POSITIONS = SYNTHETIC_ZERO_SHOT_PROTOCOL.positions
if SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL.positions != DEFAULT_POSITIONS:
    raise RuntimeError("the stipulated arms must score one shared position count")
#: One publication unit: the receipt carries a single measurement object whose
#: ``arms`` member holds one evidence block per stipulated arm.
MEASUREMENT_SCHEMA = "ailm-synthetic-execution-trace-v3"


def _letter_held_out_positions(count: int) -> list[tuple[str, str]]:
    """The letter arm's executed population, reconstructed from its protocol."""

    return independent_held_out_positions(
        count,
        window=SYNTHETIC_ZERO_SHOT_PROTOCOL.query_window,
        seed=SYNTHETIC_ZERO_SHOT_PROTOCOL.position_seed,
    )


def _phone_held_out_positions(count: int) -> list[tuple[str, str]]:
    """The phone arm's executed population, reconstructed from its protocol."""

    return independent_phone_held_out_positions(
        count,
        window=SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL.query_window,
        seed=SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL.position_seed,
    )


def _phone_comparator_refusals(positions: list[tuple[str, str]]) -> None:
    """The phone arm's pre-registered comparator refusals, before any call.

    Thresholds were fixed before this lane's live measurement from offline
    arithmetic on the vendored artifact (see the lane's decision stub): the
    stipulated fit must clear four times the uniform floor, twice a
    transposed-count fit, and twice an English-trained CMUdict fit, all on the
    identical executed positions. Offline, the stipulated comparator scores
    58/120 against 4/120 transposed and 10/120 English-trained.
    """

    counts = load_synthetic_phone_counts()
    stipulated_flags = correctness_flags(synthetic_phone_predictions(counts, positions), positions)
    stipulated = sum(stipulated_flags) / len(stipulated_flags)
    if stipulated < 4 * PHONE_UNIFORM_BASELINE:
        raise SystemExit(
            "phone arm refused: the stipulated-trained comparator does not clear "
            f"four times the uniform floor on the executed positions "
            f"({stipulated:.4f} < {4 * PHONE_UNIFORM_BASELINE:.4f}); the win would "
            "not come from this corpus. No claims written."
        )
    transposed: dict[str, int] = {}
    for key, count in counts.items():
        previous, following = key.split(",", 1)
        transposed[f"{following},{previous}"] = count
    transposed_flags = correctness_flags(
        synthetic_phone_predictions(transposed, positions), positions
    )
    transposed_accuracy = sum(transposed_flags) / len(transposed_flags)
    if stipulated < 2 * transposed_accuracy:
        raise SystemExit(
            "phone arm refused: a transposed count table scores as well as the "
            f"stipulated fit ({transposed_accuracy:.4f} vs {stipulated:.4f}); the "
            "win would not come from the transition's direction. No claims written."
        )
    english_flags = correctness_flags(
        synthetic_phone_predictions(load_phoneme_counts()["phone_bigrams"], positions),
        positions,
    )
    english = sum(english_flags) / len(english_flags)
    if stipulated < 2 * english:
        raise SystemExit(
            "phone arm refused: an English-trained comparator scores as well as "
            f"the stipulated fit ({english:.4f} vs {stipulated:.4f}); the win would "
            "come from English statistics rather than this corpus. No claims written."
        )


class _ZeroShotArm(NamedTuple):
    """One stipulated no-demonstration arm: protocol, task contract, comparator.

    The letter and phone arms share every measurement mechanic and differ only
    in the values this record carries, so a change to one mechanic reaches both
    arms or neither. A NamedTuple rather than a dataclass: this module is
    imported by spec-based loaders (the binding verifier, the statistics
    family scan) that do not register the module in ``sys.modules`` before
    executing it, which dataclass processing requires.
    """

    key: str
    protocol: SyntheticZeroShotProtocol
    #: The exact task contract consumed by ``build_prompt`` and the response
    #: schema; no changes to ``prompts.py`` or ``ollama.py``.
    task: str
    parse_answer: Callable[[str], tuple[str | None, bool]]
    held_out_positions: Callable[[int], list[tuple[str, str]]]
    target_start_tvd: Callable[[int], float]
    load_counts: Callable[[], dict[str, int]]
    regenerate_counts: Callable[[], dict[str, int]]
    comparator_predictions: Callable[[dict[str, int], list[tuple[str, str]]], list[str]]
    #: Claim namespace prefix and the two family templates corrected within it.
    label_prefix: str
    family_templates: tuple[str, ...]
    context_family_templates: tuple[str, ...]
    inventory: tuple[str, ...]
    uniform_baseline: float
    inventory_size_label: str
    corpus_size_label: str
    corpus_size_param: str
    corpus_size: int
    corpus_seed_label: str
    corpus_seed_param: str
    corpus_seed: int
    arm_order_label: str
    top1_stem: str
    artifact: str
    generator: str
    #: Phone-arm-only pre-registration refusals; the letter arm's comparator
    #: controls live in tests, where they have always lived.
    comparator_refusals: Callable[[list[tuple[str, str]]], None] | None = None


LETTER_ARM = _ZeroShotArm(
    key="letter",
    protocol=SYNTHETIC_ZERO_SHOT_PROTOCOL,
    task="synthetic_letter",
    parse_answer=parse_synthetic_letter_answer,
    held_out_positions=_letter_held_out_positions,
    target_start_tvd=independent_target_start_tvd,
    load_counts=load_synthetic_counts,
    regenerate_counts=lambda: build_synthetic_counts(
        n_chars=DEFAULT_CORPUS_CHARS, seed=DEFAULT_CORPUS_SEED
    ),
    comparator_predictions=synthetic_letter_predictions,
    label_prefix="bench_synth_",
    family_templates=FAMILY_TEMPLATES,
    context_family_templates=CONTEXT_FAMILY_TEMPLATES,
    inventory=SYNTHETIC_LETTERS,
    uniform_baseline=UNIFORM_BASELINE,
    inventory_size_label="bench_synth_alphabet_size",
    corpus_size_label="bench_synth_corpus_chars",
    corpus_size_param="corpus_chars",
    corpus_size=DEFAULT_CORPUS_CHARS,
    corpus_seed_label="bench_synth_corpus_seed",
    corpus_seed_param="corpus_seed",
    corpus_seed=DEFAULT_CORPUS_SEED,
    arm_order_label="bench_synth_context_arm_order",
    top1_stem="letter_top1",
    artifact="vendor/synthetic_counts.json",
    generator="scripts/generate_synthetic_counts.py",
)

PHONE_ARM = _ZeroShotArm(
    key="phone",
    protocol=SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL,
    task="phoneme",
    parse_answer=parse_phoneme_answer,
    held_out_positions=_phone_held_out_positions,
    target_start_tvd=phone_target_start_tvd,
    load_counts=load_synthetic_phone_counts,
    regenerate_counts=lambda: build_synthetic_phone_counts(
        n_symbols=DEFAULT_PHONE_CORPUS_SYMBOLS, seed=DEFAULT_PHONE_CORPUS_SEED
    ),
    comparator_predictions=synthetic_phone_predictions,
    label_prefix="bench_synthphone_",
    inventory_size_label="bench_synthphone_inventory_size",
    family_templates=PHONE_FAMILY_TEMPLATES,
    context_family_templates=PHONE_CONTEXT_FAMILY_TEMPLATES,
    inventory=SYNTHETIC_PHONES,
    uniform_baseline=PHONE_UNIFORM_BASELINE,
    corpus_size_label="bench_synthphone_corpus_symbols",
    corpus_size_param="corpus_symbols",
    corpus_size=DEFAULT_PHONE_CORPUS_SYMBOLS,
    corpus_seed_label="bench_synthphone_corpus_seed",
    corpus_seed_param="corpus_seed",
    corpus_seed=DEFAULT_PHONE_CORPUS_SEED,
    arm_order_label="bench_synthphone_arm_order",
    top1_stem="phone_top1",
    artifact="vendor/synthetic_phone_counts.json",
    generator="scripts/generate_synthetic_phone_counts.py",
    comparator_refusals=_phone_comparator_refusals,
)

ARMS: tuple[_ZeroShotArm, ...] = (LETTER_ARM, PHONE_ARM)


def _short_positions(arm: _ZeroShotArm, positions: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """The sensitivity arm's one-symbol prefixes, aligned to the full arms."""

    return [(prefix[-arm.protocol.short_query_window :], gold) for prefix, gold in positions]


def _maximum_prompt_characters(arm: _ZeroShotArm, positions: list[tuple[str, str]]) -> int:
    scored = positions
    short = _short_positions(arm, scored)
    prefixes = [
        prefix
        for full, short_prefix in zip(scored, short, strict=True)
        for prefix in (full[0], short_prefix[0])
    ]
    return max(len(build_prompt(arm.task, prefix)) for prefix in prefixes)


def _digest_json(value: object) -> str:
    """Digest finite canonical JSON exactly as the receipt contract does."""

    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _arm_configuration(arm: _ZeroShotArm, *, positions: int) -> dict[str, object]:
    """One arm's replay parameters, all recomputed offline from code."""

    return {
        "arm_key": arm.key,
        "measurement_protocol": arm.protocol.version,
        "chain_protocol": arm.protocol.chain_version,
        "positions": positions,
        "position_seed": arm.protocol.position_seed,
        "position_generation": arm.protocol.position_generation,
        "trajectory_length": arm.protocol.query_window + 1,
        "target_start_tvd": arm.target_start_tvd(arm.protocol.query_window),
        "query_window": arm.protocol.query_window,
        "short_query_window": arm.protocol.short_query_window,
        "response_inventory_size": len(arm.protocol.response_inventory),
        "context_window": arm.protocol.context_window,
        "response_tokens": arm.protocol.response_tokens,
        "prompt_token_overhead_margin": arm.protocol.prompt_token_overhead_margin,
        "maximum_prompt_characters": _maximum_prompt_characters(
            arm, arm.held_out_positions(positions)
        ),
        "warmup_policy": arm.protocol.warmup_policy,
        "arm_order": arm.protocol.arm_order,
        arm.corpus_size_param: arm.corpus_size,
        arm.corpus_seed_param: arm.corpus_seed,
    }


def _call_cache_protocol(
    arm: _ZeroShotArm, tag: str, index: int, arm_name: str, run_scope: str
) -> str:
    """This run's cache namespace for ONE scored call.

    Every scored call of this probe is deliberately live on a publication run —
    the fresh-call guard refuses any replay — so the namespace carries the run
    scope: without it a second run would ask the append-only store to overwrite
    the first run's replies and be refused. The namespace also carries the probe
    tag because the hosted transport resolves one model for every registered
    probe: the wire payload alone cannot distinguish gemma3's scored call from
    smollm's. And it carries the scored call's position slot because the
    hosted transport's replies are not deterministic: the sensitivity arm's
    one-symbol prompts recur across positions, so identical payloads under one
    namespace would collide inside a single run and each position's fresh
    measurement would be impossible.

    The unscoped ``arm.protocol.cache_protocol`` remains the arm's protocol
    identity — that is what the ledger records and what a reader cites. Only
    the cache namespace carries the run, the tag, and the scored slot.
    """

    return scoped_cache_protocol(
        f"{arm.protocol.cache_protocol}:{tag}:{index:04d}:{arm_name}", run_scope
    )


def _openrouter_cache_protocol(scoped: str) -> str:
    """The cache namespace a scored OpenRouter call of this run writes."""

    return f"openrouter-v1:{scoped}"


def _declared_cache_protocols(run_scope: str) -> list[str]:
    """Every cache namespace one run of this recorder writes, sorted.

    The receipt declares exactly this list and the store projection must match
    it, so a receipt cannot cite records made under another namespace. One
    namespace per scored call: the paired schedule visits
    ``len(positions)`` window pairs for every probe tag on every stipulated
    arm, and each pair scores a full and a short call.
    """

    return sorted(
        _openrouter_cache_protocol(_call_cache_protocol(arm, tag, index, arm_name, run_scope))
        for arm in ARMS
        for tag in MODELS.values()
        for index in range(len(arm.held_out_positions(DEFAULT_POSITIONS)))
        for arm_name in ("full", "short")
    )


def _arm_call_protocol_factory(
    arm: _ZeroShotArm, model: str, run_scope: str
) -> Callable[[int, str], str]:
    """A typed per-call namespace resolver bound to one arm's tag and run."""

    def protocol_for(index: int, arm_name: str) -> str:
        return _openrouter_cache_protocol(
            _call_cache_protocol(arm, model, index, arm_name, run_scope)
        )

    return protocol_for


def _replay_configuration(
    *,
    positions: int,
    run_scope: str,
    endpoint: str = OPENROUTER_URL,
    openrouter_provider: dict | None = None,
) -> dict[str, object]:
    """The canonical configuration both stipulated arms bind their receipt to.

    The endpoint and model tags name the hosted transport: those are the
    requests the receipt's cache records actually bind. The registered probe
    tags stay in the parameters as the claim namespaces the per-tag cache
    protocols serve — each probe is scored fresh through the one hosted model.
    """

    configuration = network_recorder_configuration(
        endpoint=endpoint,
        call_protocol=_openrouter_cache_protocol(
            _call_cache_protocol(LETTER_ARM, MODELS[_PRIMARY_PROBE_KEY], 0, "full", run_scope)
        ),
        model_tags=[OPENROUTER_MODEL],
        parameters={
            "arms": {arm.key: _arm_configuration(arm, positions=positions) for arm in ARMS},
            "probe_tags": sorted(MODELS.values()),
            "run_scope": validate_run_scope(run_scope),
            "cache_protocols": _declared_cache_protocols(run_scope),
        },
    )

    provider = validate_provider_options(openrouter_provider)
    if provider is not None:
        configuration["parameters"]["openrouter_provider"] = provider
    return configuration


def publication_replay_configuration(
    *, run_scope: str, openrouter_provider: dict | None = None
) -> dict[str, object]:
    """Canonical configuration required by the full release replay contract.

    The run scope and optional provider constraints identify the measured run;
    other protocol fields remain fixed by this module's code.
    The validator reads the scope from the receipt under audit and asks for the
    configuration that scope implies, and the receipt's cited records must carry
    exactly the namespaces it names, so a receipt cannot claim another run's
    records.
    """

    return _replay_configuration(
        positions=DEFAULT_POSITIONS,
        run_scope=run_scope,
        openrouter_provider=openrouter_provider,
    )


def _positive_positions(value: str) -> int:
    """Argparse type for a nonempty paired evaluation population."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("positions must be a positive integer")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Record the stipulated letter and phone probes.")
    parser.add_argument("--positions", type=_positive_positions, default=DEFAULT_POSITIONS)
    parser.add_argument(
        "--endpoint",
        default=OPENROUTER_URL,
        help="hosted chat endpoint (non-default endpoints are isolated-validation only)",
    )
    args = parser.parse_args()
    # This recorder shares the bench_ namespace and the single-writer ledger with
    # the external benchmark recorder, so it takes the same lock. Protecting only
    # one of two contending writers is not mutual exclusion.
    with single_writer_lock(ledger_lock_path(REPO_ROOT)):
        return _measure_and_record(args)


def _score_counterbalanced_windows(
    arm: _ZeroShotArm,
    full_positions: list[tuple[str, str]],
    short_positions: list[tuple[str, str]],
    *,
    cache_read: bool,
    cache_protocol_for: Callable[[int, str], str],
    endpoint: str = OPENROUTER_URL,
) -> tuple[list[bool], int, list[bool], int, int, int, list[dict[str, object]]]:
    """Score paired windows with arm order balanced over position index.

    ``cache_protocol_for`` maps one scored call's position index and arm name
    to the complete cache namespace that call writes — the caller scopes it by
    arm, probe tag, scored slot and run, because the hosted transport's replies
    are not deterministic and a recurring one-symbol prompt must not collide.
    """

    if len(full_positions) != len(short_positions):
        raise ValueError("full and short synthetic arms must have equal length")
    full_flags: list[bool] = []
    short_flags: list[bool] = []
    full_unparsable = 0
    short_unparsable = 0
    full_first = 0
    short_first = 0
    trace: list[dict[str, object]] = []
    for index, (full, short) in enumerate(zip(full_positions, short_positions, strict=True)):
        if full[1] != short[1]:
            raise ValueError(f"synthetic window pair {index} has different gold symbols")
        arms: tuple[tuple[str, tuple[str, str]], ...] = (
            ("full", full),
            ("short", short),
        )
        if index % 2:
            arms = tuple(reversed(arms))
            short_first += 1
        else:
            full_first += 1
        outcomes: dict[str, tuple[bool, bool]] = {}
        for arm_name, (prefix, gold) in arms:
            prompt = build_prompt(arm.task, prefix)
            answer = ask_openrouter(
                OPENROUTER_MODEL,
                prompt,
                arm.task,
                arm.protocol.response_tokens,
                cache_protocol=cache_protocol_for(index, arm_name),
                cache_read=cache_read,
                endpoint=endpoint,
            )
            trace.append(
                {
                    "position": index,
                    "arm": arm_name,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                }
            )
            parsed, bad = arm.parse_answer(answer)
            outcomes[arm_name] = (parsed == gold, bad)
        full_correct, full_bad = outcomes["full"]
        short_correct, short_bad = outcomes["short"]
        full_flags.append(full_correct)
        short_flags.append(short_correct)
        full_unparsable += int(full_bad)
        short_unparsable += int(short_bad)
    return (
        full_flags,
        full_unparsable,
        short_flags,
        short_unparsable,
        full_first,
        short_first,
        trace,
    )


def _expected_execution_trace(
    arm: _ZeroShotArm, positions: list[tuple[str, str]]
) -> list[dict[str, object]]:
    """Reconstruct one arm's exact model/arm/prompt order without model calls."""

    short_positions = _short_positions(arm, positions)
    trace: list[dict[str, object]] = []
    for _name, model in sorted(MODELS.items()):
        for index, (full, short) in enumerate(zip(positions, short_positions, strict=True)):
            arms: tuple[tuple[str, tuple[str, str]], ...] = (
                ("full", full),
                ("short", short),
            )
            if index % 2:
                arms = tuple(reversed(arms))
            for arm_name, (prefix, _gold) in arms:
                prompt = build_prompt(arm.task, prefix)
                trace.append(
                    {
                        "model": model,
                        "position": index,
                        "arm": arm_name,
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    }
                )
    return trace


def _arm_measurement_evidence(
    *,
    arm: _ZeroShotArm,
    execution_trace: list[dict[str, object]],
    cache_hits: int,
    cache_misses: int,
    unique_cache_keys: int,
    maximum_prompt_tokens: int,
) -> dict[str, object]:
    """One arm's evidence block inside the combined measurement object."""

    return {
        "schedule": arm.protocol.arm_order,
        "warmup_policy": arm.protocol.warmup_policy,
        "warmup_calls": 0,
        "scored_calls": len(execution_trace),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "unique_cache_keys": unique_cache_keys,
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "ordered_trace_sha256": _digest_json(execution_trace),
    }


def _measurement_evidence(
    *,
    arms: Mapping[str, dict[str, object]],
    cache_hits: int,
    cache_misses: int,
    unique_cache_keys: int,
    maximum_prompt_tokens: int,
) -> dict[str, object]:
    """Build the combined evidence object bound into the publication receipt."""

    return {
        "schema": MEASUREMENT_SCHEMA,
        "arms": dict(arms),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "unique_cache_keys": unique_cache_keys,
        "maximum_prompt_tokens": maximum_prompt_tokens,
    }


def _validate_arm_measurement_fields(
    arm: _ZeroShotArm,
    evidence: object,
    *,
    positions: int,
) -> None:
    """Validate one arm's schedule arithmetic and reproduce its prompt trace."""

    expected_fields = {
        "schedule",
        "warmup_policy",
        "warmup_calls",
        "scored_calls",
        "cache_hits",
        "cache_misses",
        "unique_cache_keys",
        "maximum_prompt_tokens",
        "ordered_trace_sha256",
    }
    if not isinstance(evidence, dict) or set(evidence) != expected_fields:
        raise NetworkClaimBindingError(
            f"synthetic {arm.key}-arm receipt has invalid measurement evidence fields"
        )
    expected_calls = positions * 2 * len(MODELS)
    if evidence["schedule"] != arm.protocol.arm_order:
        raise NetworkClaimBindingError(f"synthetic {arm.key}-arm receipt arm-order policy drifted")
    if evidence["warmup_policy"] != arm.protocol.warmup_policy:
        raise NetworkClaimBindingError(f"synthetic {arm.key}-arm receipt warm-up policy drifted")
    if type(evidence["warmup_calls"]) is not int or evidence["warmup_calls"] != 0:
        raise NetworkClaimBindingError(f"synthetic {arm.key}-arm receipt warm-up count is invalid")
    if evidence["scored_calls"] != expected_calls:
        raise NetworkClaimBindingError(
            f"synthetic {arm.key}-arm receipt scored-call count is invalid"
        )
    if evidence["cache_hits"] != 0 or evidence["cache_misses"] != expected_calls:
        raise NetworkClaimBindingError(
            f"synthetic {arm.key}-arm publication receipt must bind one fresh miss per scored call"
        )
    unique_keys = evidence["unique_cache_keys"]
    if (
        not isinstance(unique_keys, int)
        or isinstance(unique_keys, bool)
        or not 0 < unique_keys <= expected_calls
    ):
        raise NetworkClaimBindingError(
            f"synthetic {arm.key}-arm receipt unique-key count is invalid"
        )
    maximum_tokens = evidence["maximum_prompt_tokens"]
    if (
        not isinstance(maximum_tokens, int)
        or isinstance(maximum_tokens, bool)
        or maximum_tokens <= 0
        or maximum_tokens + arm.protocol.response_tokens + arm.protocol.prompt_token_overhead_margin
        > arm.protocol.context_window
    ):
        raise NetworkClaimBindingError(
            f"synthetic {arm.key}-arm receipt prompt-token maximum does not fit "
            "the declared context"
        )
    expected_positions = arm.held_out_positions(positions)
    if evidence["ordered_trace_sha256"] != _digest_json(
        _expected_execution_trace(arm, expected_positions)
    ):
        raise NetworkClaimBindingError(
            f"synthetic {arm.key}-arm receipt ordered prompt trace does not reproduce from code"
        )


def _validate_measurement_fields(
    measurement: object,
    *,
    positions: int,
    cache_record_count: int,
) -> None:
    """Validate the combined schedule and reproduce both arms' prompt traces."""

    expected_fields = {
        "schema",
        "arms",
        "cache_hits",
        "cache_misses",
        "unique_cache_keys",
        "maximum_prompt_tokens",
    }
    if not isinstance(measurement, dict) or set(measurement) != expected_fields:
        raise NetworkClaimBindingError("synthetic receipt has invalid measurement fields")
    if measurement["schema"] != MEASUREMENT_SCHEMA:
        raise NetworkClaimBindingError("unsupported synthetic measurement schema")
    arms = measurement["arms"]
    if not isinstance(arms, dict) or set(arms) != {arm.key for arm in ARMS}:
        raise NetworkClaimBindingError(
            "synthetic receipt must carry exactly the registered stipulated arms"
        )
    expected_calls = positions * 2 * len(MODELS) * len(ARMS)
    if measurement["cache_hits"] != 0 or measurement["cache_misses"] != expected_calls:
        raise NetworkClaimBindingError(
            "synthetic publication receipt must bind one fresh miss per scored call"
        )
    unique_keys = measurement["unique_cache_keys"]
    if (
        not isinstance(unique_keys, int)
        or isinstance(unique_keys, bool)
        or not 0 < unique_keys <= expected_calls
        or unique_keys != cache_record_count
    ):
        raise NetworkClaimBindingError(
            "synthetic receipt unique-key count disagrees with the cache binding"
        )
    maximum_tokens = measurement["maximum_prompt_tokens"]
    if (
        not isinstance(maximum_tokens, int)
        or isinstance(maximum_tokens, bool)
        or maximum_tokens <= 0
        or any(
            maximum_tokens
            + arm.protocol.response_tokens
            + arm.protocol.prompt_token_overhead_margin
            > arm.protocol.context_window
            for arm in ARMS
        )
    ):
        raise NetworkClaimBindingError(
            "synthetic receipt prompt-token maximum does not fit the declared context"
        )
    for arm in ARMS:
        _validate_arm_measurement_fields(
            arm,
            arms[arm.key],
            positions=positions,
        )


def _receipt_claim_value(receipt: Mapping[str, object], label: str) -> object:
    claims = receipt.get("claims")
    if not isinstance(claims, dict) or not isinstance(claims.get("entries"), dict):
        raise NetworkClaimBindingError("synthetic receipt claim projection is invalid")
    entry = claims["entries"].get(label)
    if not isinstance(entry, dict) or "value" not in entry:
        raise NetworkClaimBindingError(f"synthetic receipt is missing claim {label}")
    return entry["value"]


def _base_claim_values(
    arm: _ZeroShotArm,
    *,
    total: int,
    ailm_flags: list[bool],
    maximum_prompt_characters: int,
    maximum_prompt_tokens: int | None = None,
    cache_hits: int | None = None,
    cache_misses: int | None = None,
) -> dict[str, object]:
    """Derive one arm's model-independent claims from its protocol and data."""

    prefix = arm.label_prefix
    ailm_low, ailm_high = wilson_interval(sum(ailm_flags), total)
    values: dict[str, object] = {
        f"{prefix}protocol_version": arm.protocol.version,
        f"{prefix}chain_version": arm.protocol.chain_version,
        arm.inventory_size_label: len(arm.inventory),
        arm.corpus_size_label: arm.corpus_size,
        arm.corpus_seed_label: arm.corpus_seed,
        f"{prefix}position_generation": arm.protocol.position_generation,
        f"{prefix}position_seed": arm.protocol.position_seed,
        f"{prefix}trajectory_length": arm.protocol.query_window + 1,
        f"{prefix}target_start_tvd": arm.target_start_tvd(arm.protocol.query_window),
        f"{prefix}positions": total,
        f"{prefix}query_window": arm.protocol.query_window,
        f"{prefix}query_context_transitions_max": (arm.protocol.query_context_transitions_max),
        f"{prefix}short_query_window": arm.protocol.short_query_window,
        f"{prefix}short_query_context_transitions_max": (
            arm.protocol.short_query_context_transitions_max
        ),
        f"{prefix}response_inventory_size": len(arm.protocol.response_inventory),
        f"{prefix}context_window": arm.protocol.context_window,
        f"{prefix}response_tokens": arm.protocol.response_tokens,
        f"{prefix}prompt_token_overhead_margin": (arm.protocol.prompt_token_overhead_margin),
        f"{prefix}maximum_prompt_characters": maximum_prompt_characters,
        f"{prefix}warmup_policy": arm.protocol.warmup_policy,
        arm.arm_order_label: arm.protocol.arm_order,
        f"{prefix}cache_protocol": arm.protocol.cache_protocol,
        f"{prefix}uniform_baseline": arm.uniform_baseline,
        f"{prefix}{arm.top1_stem}_ailm": sum(ailm_flags) / total,
        f"{prefix}{arm.top1_stem}_wilson_low_ailm": ailm_low,
        f"{prefix}{arm.top1_stem}_wilson_high_ailm": ailm_high,
        f"{prefix}family_size": len(arm.family_templates) * len(MODELS),
        f"{prefix}family_alpha": POWER_ALPHA / (len(arm.family_templates) * len(MODELS)),
        f"{prefix}context_family_size": len(arm.context_family_templates) * len(MODELS),
        f"{prefix}context_family_alpha": POWER_ALPHA
        / (len(arm.context_family_templates) * len(MODELS)),
    }
    optional = {
        f"{prefix}maximum_prompt_tokens": maximum_prompt_tokens,
        f"{prefix}scored_cache_hits": cache_hits,
        f"{prefix}scored_cache_misses": cache_misses,
    }
    values.update({label: value for label, value in optional.items() if value is not None})
    return values


def _probe_claim_values(
    arm: _ZeroShotArm,
    *,
    name: str,
    reference_flags: list[bool],
    short_flags: list[bool],
    ailm_flags: list[bool],
    unparsable: int,
    short_unparsable: int,
    full_first: int,
    short_first: int,
) -> dict[str, object]:
    """Derive one probe's complete result namespace from per-position outcomes."""

    prefix = arm.label_prefix
    total = len(ailm_flags)
    if len(reference_flags) != total or len(short_flags) != total:
        raise NetworkClaimBindingError("synthetic result arms have unequal populations")
    ref_low, ref_high = wilson_interval(sum(reference_flags), total)
    short_low, short_high = wilson_interval(sum(short_flags), total)
    paired = paired_discordance(reference_flags, ailm_flags)
    context_paired = paired_discordance(reference_flags, short_flags)
    return {
        f"{prefix}{arm.top1_stem}_{name}": sum(reference_flags) / total,
        f"{prefix}{arm.top1_stem}_wilson_low_{name}": ref_low,
        f"{prefix}{arm.top1_stem}_wilson_high_{name}": ref_high,
        f"{prefix}paired_b_{name}": paired["b"],
        f"{prefix}paired_c_{name}": paired["c"],
        f"{prefix}paired_p_{name}": paired["p_value"],
        f"{prefix}unparsable_{name}": unparsable,
        f"{prefix}unparsable_rate_{name}": unparsable / total,
        f"{prefix}short_top1_{name}": sum(short_flags) / total,
        f"{prefix}short_wilson_low_{name}": short_low,
        f"{prefix}short_wilson_high_{name}": short_high,
        f"{prefix}short_unparsable_{name}": short_unparsable,
        f"{prefix}short_unparsable_rate_{name}": short_unparsable / total,
        f"{prefix}context_paired_b_{name}": context_paired["b"],
        f"{prefix}context_paired_c_{name}": context_paired["c"],
        f"{prefix}context_paired_p_{name}": context_paired["p_value"],
        f"{prefix}context_order_full_first_{name}": full_first,
        f"{prefix}context_order_short_first_{name}": short_first,
    }


def _replay_arm_requests(
    arm: _ZeroShotArm,
    store: CallCache,
    cached: dict[str, tuple[dict[str, object], dict[str, object]]],
    selected: set[str],
    endpoint: str,
    declared_scope: str,
    transport_identity: str,
    openrouter_provider: dict | None = None,
    positions_count: int = DEFAULT_POSITIONS,
) -> tuple[dict[str, object], set[str]]:
    """Reparse one arm's cached replies and rebuild its complete namespace.

    The arm's prompt-token maximum is recomputed from its own reconstructed
    request keys, so the receipt cannot attribute another arm's request
    tokens to this arm.
    """

    positions = arm.held_out_positions(positions_count)
    short_positions = _short_positions(arm, positions)
    counts = arm.load_counts()
    ailm_flags = correctness_flags(arm.comparator_predictions(counts, positions), positions)
    base = _base_claim_values(
        arm,
        total=len(positions),
        ailm_flags=ailm_flags,
        maximum_prompt_characters=_maximum_prompt_characters(arm, positions),
        cache_hits=0,
        cache_misses=len(MODELS) * len(positions) * 2,
    )
    arm_keys: set[str] = set()
    expected_keys: set[str] = set()
    for name, model in sorted(MODELS.items()):
        full_flags: list[bool] = []
        short_flags: list[bool] = []
        full_unparsable = 0
        short_unparsable = 0
        for index, (full, short) in enumerate(zip(positions, short_positions, strict=True)):
            outcomes: dict[str, tuple[bool, bool]] = {}
            for arm_name, (prefix, gold) in (("full", full), ("short", short)):
                prompt = build_prompt(arm.task, prefix)
                payload = _chat_payload(
                    OPENROUTER_MODEL,
                    prompt,
                    arm.task,
                    arm.protocol.response_tokens,
                    DEFAULT_SEED,
                )
                if openrouter_provider is not None:
                    payload["provider"] = openrouter_provider
                request = CacheRequest(
                    payload=payload,
                    endpoint=endpoint,
                    protocol=_openrouter_cache_protocol(
                        _call_cache_protocol(arm, model, index, arm_name, declared_scope)
                    ),
                    identity=transport_identity,
                )
                expected_keys.add(request.key)
                arm_keys.add(request.key)
                record = cached.get(request.key)
                if record is None or record[0] != payload:
                    raise NetworkClaimBindingError(
                        f"synthetic semantic replay lacks exact {model} {arm_name} "
                        f"request at {index}"
                    )
                parsed, bad = arm.parse_answer(str(record[1].get("response", "")))
                outcomes[arm_name] = (parsed == gold, bad)
            full_correct, full_bad = outcomes["full"]
            short_correct, short_bad = outcomes["short"]
            full_flags.append(full_correct)
            short_flags.append(short_correct)
            full_unparsable += int(full_bad)
            short_unparsable += int(short_bad)
        base.update(
            _probe_claim_values(
                arm,
                name=name,
                reference_flags=full_flags,
                short_flags=short_flags,
                ailm_flags=ailm_flags,
                unparsable=full_unparsable,
                short_unparsable=short_unparsable,
                full_first=(len(positions) + 1) // 2,
                short_first=len(positions) // 2,
            )
        )
    if not selected.issuperset(expected_keys):
        raise NetworkClaimBindingError(
            "synthetic semantic replay produced keys outside the receipt's cache binding"
        )
    values = dict(base)
    values[f"{arm.label_prefix}maximum_prompt_tokens"] = maximum_prompt_eval_tokens(
        store, sorted(arm_keys)
    )
    return values, expected_keys


def _replay_result_claims(
    receipt: Mapping[str, object],
    store: CallCache,
    cache_keys: list[str],
    *,
    positions_count: int = DEFAULT_POSITIONS,
) -> dict[str, object]:
    """Reparse both arms; bounded local tests may supply an explicit population.

    Publication validation always uses the complete registered population.
    """
    if positions_count <= 0:
        raise NetworkClaimBindingError("synthetic replay population must be positive")

    cache = receipt.get("cache")
    configuration = receipt.get("configuration")
    measurement = receipt.get("measurement")
    if (
        not isinstance(cache, dict)
        or not isinstance(configuration, dict)
        or not isinstance(measurement, dict)
        or not isinstance(measurement.get("arms"), dict)
    ):
        raise NetworkClaimBindingError("synthetic receipt replay envelope is invalid")
    endpoint = configuration.get("endpoint")
    protocol = configuration.get("call_protocol")
    identity_rows = cache.get("model_identities")
    if (
        not isinstance(endpoint, str)
        or not isinstance(protocol, str)
        or not isinstance(identity_rows, list)
    ):
        raise NetworkClaimBindingError("synthetic receipt replay identity is invalid")
    # The run the receipt states is the run its records were made under. Reading
    # it here rather than from the arm constants is what lets an older receipt
    # replay against its own records: the constants name the arm, not the run.
    parameters = configuration.get("parameters")
    if not isinstance(parameters, dict):
        raise NetworkClaimBindingError("synthetic receipt configuration parameters are invalid")
    if "openrouter_provider" in parameters and not isinstance(
        parameters["openrouter_provider"], dict
    ):
        raise NetworkClaimBindingError("receipt provider declaration is invalid")
    openrouter_provider = validate_provider_options(parameters.get("openrouter_provider"))
    declared_scope = validate_run_scope(parameters.get("run_scope"))
    if parameters.get("cache_protocols") != _declared_cache_protocols(declared_scope):
        raise NetworkClaimBindingError(
            "synthetic receipt cache namespaces do not match its declared run scope"
        )
    if parameters.get("probe_tags") != sorted(MODELS.values()):
        raise NetworkClaimBindingError(
            "synthetic receipt probe tags do not match the registered probe labels"
        )
    if protocol != _openrouter_cache_protocol(
        _call_cache_protocol(LETTER_ARM, MODELS[_PRIMARY_PROBE_KEY], 0, "full", declared_scope)
    ):
        raise NetworkClaimBindingError(
            "synthetic receipt call protocol does not match its declared run scope"
        )
    identities = {
        row["model"]: row["identity"]
        for row in identity_rows
        if isinstance(row, dict)
        and isinstance(row.get("model"), str)
        and isinstance(row.get("identity"), str)
    }
    if set(identities) != {OPENROUTER_MODEL}:
        raise NetworkClaimBindingError("synthetic receipt model identities are incomplete")
    transport_identity = identities[OPENROUTER_MODEL]

    selected = set(cache_keys)
    cached = {key: (payload, reply) for key, payload, reply in store.entries() if key in selected}
    if set(cached) != selected:
        raise NetworkClaimBindingError("synthetic semantic replay is missing cached replies")

    values: dict[str, object] = {
        "bench_synth_openrouter_model": OPENROUTER_MODEL,
        "bench_synth_openrouter_provider": (
            openrouter_provider["only"][0] if openrouter_provider is not None else "unpinned"
        ),
    }
    expected_keys: set[str] = set()
    for arm in ARMS:
        arm_values, arm_expected = _replay_arm_requests(
            arm,
            store,
            cached,
            selected,
            endpoint,
            declared_scope,
            transport_identity,
            openrouter_provider,
            positions_count,
        )
        values.update(arm_values)
        expected_keys |= arm_expected
        arm_tokens = measurement["arms"].get(arm.key, {}).get("maximum_prompt_tokens")
        if arm_tokens != arm_values[f"{arm.label_prefix}maximum_prompt_tokens"]:
            raise NetworkClaimBindingError(
                f"synthetic {arm.key}-arm receipt prompt-token maximum disagrees "
                "with cached replies"
            )
    if expected_keys != selected:
        raise NetworkClaimBindingError(
            "synthetic receipt cache key set is not the exact scored request set"
        )
    return values


def validate_publication_measurement(receipt: Mapping[str, object], store: CallCache) -> None:
    """Reconstruct schedule evidence and bind it to cache/ledger projections."""

    cache = receipt.get("cache")
    if not isinstance(cache, dict) or not isinstance(cache.get("record_count"), int):
        raise NetworkClaimBindingError("synthetic receipt cache projection is invalid")
    _validate_measurement_fields(
        receipt.get("measurement"),
        positions=DEFAULT_POSITIONS,
        cache_record_count=cache["record_count"],
    )
    measurement = receipt["measurement"]
    if not isinstance(measurement, dict):
        raise TypeError("receipt measurement must be a dict")
    cache_keys = cache.get("keys")
    if not isinstance(cache_keys, list) or measurement["maximum_prompt_tokens"] != (
        maximum_prompt_eval_tokens(store, cache_keys)
    ):
        raise NetworkClaimBindingError(
            "synthetic receipt prompt-token maximum disagrees with cached replies"
        )
    derived = _replay_result_claims(receipt, store, cache_keys)
    claims = receipt.get("claims")
    entries = claims.get("entries") if isinstance(claims, dict) else None
    if not isinstance(entries, dict):
        raise NetworkClaimBindingError("synthetic receipt claim projection is invalid")
    owned_prefixes = tuple(arm.label_prefix for arm in ARMS)
    foreign_labels = sorted(
        label
        for label in entries
        if not isinstance(label, str) or not label.startswith(owned_prefixes)
    )
    if foreign_labels:
        raise NetworkClaimBindingError(
            f"synthetic recorder owns claims outside its declared namespaces: {foreign_labels}"
        )
    recorded = {
        label: entry.get("value")
        for label, entry in entries.items()
        if isinstance(label, str) and label.startswith(owned_prefixes) and isinstance(entry, dict)
    }
    mismatches = {
        label: (recorded.get(label), expected)
        for label, expected in derived.items()
        if recorded.get(label) != expected
    }
    missing_or_extra = sorted(set(recorded) ^ set(derived))
    if mismatches or missing_or_extra:
        raise NetworkClaimBindingError(
            "synthetic receipt claims do not rederive from cached replies: "
            f"mismatches={mismatches}, missing_or_extra={missing_or_extra}"
        )


class _ArmRunState:
    """One arm's per-execution state accumulated during measurement."""

    def __init__(
        self,
        *,
        arm: _ZeroShotArm,
        counts: dict[str, int],
        positions: list[tuple[str, str]],
        short_positions: list[tuple[str, str]],
        total: int,
        maximum_prompt_characters: int,
        ailm_flags: list[bool],
        values: dict[str, object],
    ) -> None:
        self.arm = arm
        self.counts = counts
        self.positions = positions
        self.short_positions = short_positions
        self.total = total
        self.maximum_prompt_characters = maximum_prompt_characters
        self.ailm_flags = ailm_flags
        self.values = values
        self.cache_hits = 0
        self.cache_misses = 0
        self.trace: list[dict[str, object]] = []
        self.keys: set[str] = set()


def _measure_and_record(args: argparse.Namespace) -> int:
    publication_run = using_publication_ledger(REPO_ROOT)
    # One scope for the whole run, fixed before the first server call. Every
    # scored call here is deliberately live, so without it a second run would
    # ask the store to overwrite the first run's replies and be refused.
    run_scope = generate_run_scope()
    print(f"synthetic run scope: {run_scope}", flush=True)
    if publication_run:
        if args.endpoint.rstrip("/") != OPENROUTER_URL.rstrip("/"):
            raise SystemExit(
                "synthetic probe refused: the publication ledger requires the canonical "
                f"endpoint {OPENROUTER_URL!r}, got {args.endpoint!r}; no claims written"
            )
        if args.positions != DEFAULT_POSITIONS:
            raise SystemExit(
                "synthetic probe refused: the publication ledger requires the canonical "
                f"{DEFAULT_POSITIONS}-position population, got {args.positions}; use an "
                "absolute AILM_CLAIMS_PATH for reduced validation. No claims written."
            )
    openrouter_provider = configured_provider_options(OPENROUTER_MODEL)
    initial_source = source_manifest(REPO_ROOT, SOURCE)
    states: dict[str, _ArmRunState] = {}
    for arm in ARMS:
        counts = arm.load_counts()
        regenerated = arm.regenerate_counts()
        if counts != regenerated:
            raise SystemExit(
                f"synthetic probe refused: {arm.artifact} does not reproduce from "
                f"{arm.generator}; the artifact is stale or hand-edited. "
                "No claims written."
            )
        positions = arm.held_out_positions(args.positions)
        short_positions = _short_positions(arm, positions)
        if [gold for _prefix, gold in positions] != [gold for _prefix, gold in short_positions]:
            raise SystemExit(
                f"synthetic probe refused: {arm.key}-arm query windows lost position alignment"
            )
        total = len(positions)
        maximum_prompt_characters = _maximum_prompt_characters(arm, positions)
        if (
            maximum_prompt_characters
            + arm.protocol.response_tokens
            + arm.protocol.prompt_token_overhead_margin
            > arm.protocol.context_window
        ):
            raise SystemExit(
                f"synthetic probe refused: the {arm.key} arm's declared hosted context "
                "window does not cover the longest scored prompt plus response and "
                "safety margin; no claims written"
            )
        if arm.comparator_refusals is not None:
            arm.comparator_refusals(positions)
        ailm_flags = correctness_flags(arm.comparator_predictions(counts, positions), positions)
        states[arm.key] = _ArmRunState(
            arm=arm,
            counts=counts,
            positions=positions,
            short_positions=short_positions,
            total=total,
            maximum_prompt_characters=maximum_prompt_characters,
            ailm_flags=ailm_flags,
            values=_base_claim_values(
                arm,
                total=total,
                ailm_flags=ailm_flags,
                maximum_prompt_characters=maximum_prompt_characters,
            ),
        )

    store = enable_call_cache(require_default_path=publication_run)
    if store is None:  # enabled=True above; defensive against future API drift
        raise SystemExit("synthetic probe refused: call store could not be enabled")
    # The scored calls run through the OpenRouter transport, which keeps its
    # own module-level cache handle; binding it is what lets post_chat count
    # its hits and misses on the same store the schedule audit reads.
    set_call_cache(store)
    store.clear_access_log()

    measurement_keys: set[str] = set()
    measured_cache_hits = 0
    measured_cache_misses = 0
    for name, model in sorted(MODELS.items()):
        for arm in ARMS:
            state = states[arm.key]
            store.clear_access_log()
            hits_before = store.stats.hits
            misses_before = store.stats.misses
            (
                reference_flags,
                unparsable,
                short_flags,
                short_unparsable,
                full_first,
                short_first,
                model_trace,
            ) = _score_counterbalanced_windows(
                arm,
                state.positions,
                state.short_positions,
                cache_read=not publication_run,
                cache_protocol_for=_arm_call_protocol_factory(arm, model, run_scope),
                endpoint=args.endpoint,
            )
            state.trace.extend({"model": model, **entry} for entry in model_trace)
            model_hits = store.stats.hits - hits_before
            model_misses = store.stats.misses - misses_before
            expected_misses = 2 * state.total
            if publication_run and model_hits:
                raise SystemExit(
                    "synthetic probe refused: canonical arm-order provenance requires "
                    f"fresh scored calls, but {model} replayed {model_hits} cached "
                    "request(s); use an isolated ledger for resumable validation. "
                    "No claims written."
                )
            if model_hits + model_misses != expected_misses:
                raise SystemExit(
                    "synthetic probe refused: scored cache accounting does not match "
                    f"the paired schedule for {model} on the {arm.key} arm: "
                    f"hits={model_hits}, misses={model_misses}, "
                    f"expected={expected_misses}. No claims written."
                )
            state.cache_hits += model_hits
            state.cache_misses += model_misses
            measured_cache_hits += model_hits
            measured_cache_misses += model_misses
            measurement_keys.update(store.accessed_keys)
            state.keys.update(store.accessed_keys)
            if source_manifest(REPO_ROOT, SOURCE) != initial_source:
                raise SystemExit(
                    "synthetic probe refused: source or vendored input changed during "
                    f"measurement of {model}; no claims written"
                )
            state.values.update(
                _probe_claim_values(
                    arm,
                    name=name,
                    reference_flags=reference_flags,
                    short_flags=short_flags,
                    ailm_flags=state.ailm_flags,
                    unparsable=unparsable,
                    short_unparsable=short_unparsable,
                    full_first=full_first,
                    short_first=short_first,
                )
            )
            print(
                f"{model} ({name}) [{arm.key} arm]: full-prefix "
                f"{sum(reference_flags)}/{state.total}, one-symbol "
                f"{sum(short_flags)}/{state.total}; unparsable "
                f"{unparsable}/{short_unparsable}",
                flush=True,
            )

    arm_evidence: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        state = states[arm.key]
        state.values[f"{arm.label_prefix}scored_cache_hits"] = state.cache_hits
        state.values[f"{arm.label_prefix}scored_cache_misses"] = state.cache_misses
        arm_maximum_tokens = maximum_prompt_eval_tokens(store, sorted(state.keys))
        state.values[f"{arm.label_prefix}maximum_prompt_tokens"] = arm_maximum_tokens
        arm_evidence[arm.key] = _arm_measurement_evidence(
            arm=arm,
            execution_trace=state.trace,
            cache_hits=state.cache_hits,
            cache_misses=state.cache_misses,
            unique_cache_keys=len(state.keys),
            maximum_prompt_tokens=arm_maximum_tokens,
        )

    if source_manifest(REPO_ROOT, SOURCE) != initial_source:
        raise SystemExit(
            "synthetic probe refused: source or vendored input changed before ledger "
            "publication; no claims written"
        )
    if call_cache() is not store:
        raise SystemExit("synthetic probe refused: active call store changed during measurement")
    if not measurement_keys:
        raise SystemExit(
            "synthetic probe refused: scored measurement produced no trusted cache keys; "
            "no claims written"
        )
    maximum_prompt_tokens = maximum_prompt_eval_tokens(store, sorted(measurement_keys))
    measurement = _measurement_evidence(
        arms=arm_evidence,
        cache_hits=measured_cache_hits,
        cache_misses=measured_cache_misses,
        unique_cache_keys=len(measurement_keys),
        maximum_prompt_tokens=maximum_prompt_tokens,
    )
    if publication_run:
        _validate_measurement_fields(
            measurement,
            positions=args.positions,
            cache_record_count=len(measurement_keys),
        )
        validate_network_measurement_inputs(
            configuration=_replay_configuration(
                positions=args.positions,
                run_scope=run_scope,
                endpoint=args.endpoint,
                openrouter_provider=openrouter_provider,
            ),
            store=store,
            cache_keys=measurement_keys,
            measurement=measurement,
        )
    if configured_provider_options(OPENROUTER_MODEL) != openrouter_provider:
        raise NetworkClaimBindingError("OpenRouter provider changed during measurement")
    ledger = load_ledger(REPO_ROOT)
    # Ownership is by recorded ``source``: the external benchmark shares the
    # ``bench_`` prefix and decides its own retirements the same way, so neither
    # can reach the other's labels. A label this run no longer produces has to
    # go, or a renamed measurement would leave its old value behind looking
    # current.
    states[LETTER_ARM.key].values.update(
        {
            "bench_synth_openrouter_model": OPENROUTER_MODEL,
            "bench_synth_openrouter_provider": (
                openrouter_provider["only"][0] if openrouter_provider is not None else "unpinned"
            ),
        }
    )
    combined_values: dict[str, object] = {}
    for arm in ARMS:
        combined_values.update(states[arm.key].values)
    stale = [label for label in ledger.labels_owned_by(SOURCE) if label not in combined_values]
    # Clearing the superseded labels and writing the new ones are a single
    # atomic write. As ``remove_labels`` then ``record_many`` they were two
    # whole-file replacements, and a crash in the window between them left the
    # retired labels deleted, the new ones absent, and a ledger that looks
    # internally consistent while being short by both. Each stipulated arm is
    # its own run id, so the two batches carry different provenance while
    # sharing one source and one atomic commit.
    ledger.replace_claims(
        remove=stale,
        batches=[
            ClaimBatch(
                values=states[arm.key].values,
                run_id=arm.protocol.run_id,
                source=SOURCE,
            )
            for arm in ARMS
        ],
    )
    print(
        f"recorded {len(combined_values)} synthetic-probe claims for {len(MODELS)} probes "
        f"over {args.positions} positions across {len(ARMS)} stipulated arms"
    )
    if publication_run:
        receipt_path = write_network_claim_binding(
            project_root=REPO_ROOT,
            recorder=SOURCE,
            expected_source_manifest=initial_source,
            configuration=_replay_configuration(
                positions=args.positions,
                run_scope=run_scope,
                endpoint=args.endpoint,
                openrouter_provider=openrouter_provider,
            ),
            store=store,
            cache_keys=measurement_keys,
            measurement=measurement,
        )
        print(f"network claim binding: {receipt_path.relative_to(REPO_ROOT)}")
    else:
        print("isolated ledger run: publication network claim binding left unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
