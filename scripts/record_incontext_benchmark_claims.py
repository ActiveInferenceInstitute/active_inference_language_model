#!/usr/bin/env python3
"""Measure and record the in-context training-data arm of the synthetic probe.

Why this arm exists
-------------------
Every external margin this paper reports compares an offline-fitted system
against a probe with no offline fit or separately labeled demonstrations. The
probe's no-demonstration query does carry a bounded prefix from the chain, so it
is not observation-free; the producer measures that prefix-width sensitivity
directly. This arm asks the distinct question of what happens when explicit
(source-symbol, next-symbol) transition examples are supplied.

This recorder supplies ``k`` labeled order-one transitions drawn from the same
corpus AILM's counts are estimated from and sweeps ``k`` across a declared grid.
The result measures that bounded prompt intervention. It cannot by itself
equalize offline fitting opportunities or decide whether a remaining margin is
caused by fitting, interface, model family, or another difference.

What is deliberately NOT supplied
---------------------------------
AILM's fitted posterior column. Handing the probe the comparator's own answer
would cap the probe at AILM by construction, put the prediction on line one of
its own prompt, and produce a large impressive number that measures nothing —
the same defect as a case oracle that reads its own answer key. Decision 51
records this as rejected rather than pending.

The synthetic chain, not Brown
------------------------------
This runs on :mod:`ailm.levels.synthetic`, whose transition and local artifact
are stipulated and generated in this project. That reduces exact-artifact recall
as an explanation without certifying what an external model encountered during
pretraining.

The derangement control
-----------------------
A k-shot lift alone does not show the probe *used* the examples: a wall of
plausible symbols may simply pull the output distribution toward the alphabet
and away from refusals. Each ``k`` therefore runs twice. The control shows the
same ``k`` examples with their symbols permuted so that no example keeps its own
— a true derangement, so the displayed prefix multiset, label multiset, and
character length are preserved exactly, and only the prefix-to-symbol *mapping*
is destroyed. Tokenizer-level counts are not claimed invariant. A cross-symbol
swap would not preserve the declared prefix and label multisets, and a rotation
with a fixed point would not be a control at all; where no derangement
exists the position is refused rather than silently shown the treatment prompt,
because a control arm that is byte-identical to its treatment reads as a clean
null.

Refusals — no claim is written unless every one of these holds:

* the reference model is scored through the OpenRouter transport
  (``meta-llama/llama-3.3-70b-instruct``), with no local-model dependency or warm-up;
* the vendored counts reproduce from their generator, as in the zero-shot probe;
* the examples come from the fixed training stream while scored positions come
  from independently seeded evaluation trajectories;
* every zero-shot producer claim is source-bound to the current producer and
  names the exact chain, query window, response inventory, and measurement
  protocol this consumer replays;
* the no-demonstration arm reproduces the value that the source-bound producer put
  in the ledger, so this recorder cannot quietly describe a different population;
* every HTTP call completes; malformed or empty replies stay in the denominator
  as unparsable and wrong in every arm.

Usage
-----
    .venv/bin/python \
        scripts/record_incontext_benchmark_claims.py

    # isolated validation, never the tracked ledger:
    AILM_CLAIMS_PATH=/tmp/ctx_claims.json .venv/bin/python \
        scripts/record_incontext_benchmark_claims.py --positions 40
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from ailm.benchmarks import (  # noqa: E402  (path setup must precede the import)
    build_in_context_synthetic_letter_prompts,
    call_cache,
    draw_in_context_example_sets,
    draw_in_context_example_sets_with_diagnostics,
    enable_call_cache,
    letter_positions,
    paired_difference_interval,
    paired_discordance,
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
    SYNTHETIC_INCONTEXT_PROTOCOL,
    SYNTHETIC_ZERO_SHOT_PROTOCOL,
)
from ailm.call_cache import CacheRequest, CallCache  # noqa: E402
from ailm.claims import ClaimBatch, load_ledger  # noqa: E402
from ailm.control import ControlHarness  # noqa: E402
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402
from ailm.levels.synthetic import (  # noqa: E402
    DEFAULT_CORPUS_CHARS,
    DEFAULT_CORPUS_SEED,
    SYNTHETIC_LETTERS,
    UNIFORM_BASELINE,
    build_synthetic_counts,
    correctness_flags,
    generate_sequence,
    independent_held_out_positions,
    independent_target_start_tvd,
    load_synthetic_counts,
    synthetic_letter_predictions,
)
from ailm.replay_provenance import (  # noqa: E402
    NetworkClaimBindingError,
    declared_openrouter_provider,
    declared_run_scope,
    generate_run_scope,
    maximum_prompt_eval_tokens,
    network_recorder_configuration,
    scoped_cache_protocol,
    source_manifest,
    using_publication_ledger,
    validate_network_claim_binding,
    validate_network_measurement_inputs,
    validate_run_scope,
    write_network_claim_binding,
)

#: Reference model tag → label suffix. Mirrors the zero-shot synthetic recorder
#: so ``bench_ctx_top1_k16_gemma3`` and ``bench_synth_letter_top1_gemma3`` name
#: separate call cohorts on the same chain.
MODELS = dict(PROBE_MODEL_TAGS)

#: The OpenRouter slug the reference model is scored through. The tag in
#: ``MODELS`` keeps naming ledger and call-protocol suffixes; the actual
#: hosted identity belongs to the payload and receipt, not these local tag names.
OPENROUTER_MODEL = HOSTED_SYNTHETIC_MODEL

#: The registered probe whose per-tag namespace scopes the receipt's primary
#: call protocol. Deterministic: the first key of the shared registry.
_PRIMARY_PROBE_KEY = sorted(MODELS)[0]

#: The example counts, fixed here before the run. ``0`` is present as the
#: zero-shot anchor and is scored through :func:`build_prompt`, byte-identical to
#: the arm the zero-shot recorder already measured — which is what makes the
#: reproduction precondition below meaningful.
K_GRID = SYNTHETIC_INCONTEXT_PROTOCOL.k_grid

#: The ``k`` whose contrast against AILM is the headline. Declared here, before
#: the run, rather than chosen afterwards as "the best one": picking the
#: strongest arm post hoc and reporting its p-value uncorrected is the selection
#: defect this repository has already corrected once, in the channel weight
#: sweep (decision 46).
HEADLINE_K = SYNTHETIC_INCONTEXT_PROTOCOL.headline_k

#: Seed for drawing examples. Separate from ``DEFAULT_SEED`` (which selects the
#: scored positions) so the two choices cannot be confounded.
EXAMPLE_SEED = SYNTHETIC_INCONTEXT_PROTOCOL.example_seed

#: How many times a position may redraw its examples looking for one that admits
#: a derangement. Generous: at k=4 on this chain a rejection is a few percent of
#: draws, so exhausting this budget means the pool is not what it should be.
_DERANGEABLE_DRAW_ATTEMPTS = SYNTHETIC_INCONTEXT_PROTOCOL.derangeable_draw_attempts

#: Every p-value this recorder writes, and the family they are corrected in.
#: The derangement controls are members rather than footnotes: "the examples are
#: being used" is a hypothesis the manuscript states, so it is corrected like
#: one.
FAMILY_TEMPLATES: tuple[str, ...] = tuple(
    [f"bench_ctx_k{k}_vs_zero_paired_p_{{model}}" for k in K_GRID if k]
    + [f"bench_ctx_k{k}_vs_deranged_paired_p_{{model}}" for k in K_GRID if k]
    + [f"bench_ctx_k{HEADLINE_K}_vs_ailm_paired_p_{{model}}"]
)
P_VALUE_FAMILIES = {"incontext": FAMILY_TEMPLATES}
MAPPING_CONTROL_FAMILY_TEMPLATES = ("bench_ctx_mapping_control_paired_p",)
P_VALUE_FAMILIES["mapping_positive_control"] = MAPPING_CONTROL_FAMILY_TEMPLATES
FAMILY_ALPHA = POWER_ALPHA

RUN_ID = SYNTHETIC_INCONTEXT_PROTOCOL.run_id
SOURCE = SYNTHETIC_INCONTEXT_PROTOCOL.source
MEASUREMENT_PROTOCOL = SYNTHETIC_INCONTEXT_PROTOCOL.version
PRODUCER_RUN_ID = SYNTHETIC_ZERO_SHOT_PROTOCOL.run_id
PRODUCER_SOURCE = SYNTHETIC_ZERO_SHOT_PROTOCOL.source
PRODUCER_MEASUREMENT_PROTOCOL = SYNTHETIC_ZERO_SHOT_PROTOCOL.version
DEFAULT_POSITIONS = SYNTHETIC_INCONTEXT_PROTOCOL.positions
MEASUREMENT_SCHEMA = "ailm-incontext-execution-trace-v2"


def _call_cache_protocol(tag: str, index: int, arm: str, run_scope: str) -> str:
    """This run's cache namespace for ONE scored call.

    Every scored call of this probe is deliberately live on a publication run —
    the fresh-call guard refuses any replay — so the namespace carries the run
    scope: without it a second run would ask the append-only store to overwrite
    the first run's replies and be refused. The namespace also carries the probe
    tag because the hosted transport resolves one model for every registered
    probe: the wire payload alone cannot distinguish gemma3's scored call from
    smollm's. And it carries the scored call's position slot because the hosted
    transport's replies are not deterministic: a recurring prompt under one
    namespace would collide inside a single run and each position's fresh
    measurement would be impossible.

    The unscoped ``SYNTHETIC_INCONTEXT_PROTOCOL.cache_protocol`` remains the
    measurement's protocol identity — that is what the ledger records and what a
    reader cites. Only the cache namespace carries the run, the tag, and the
    scored slot.
    """

    return scoped_cache_protocol(
        f"{SYNTHETIC_INCONTEXT_PROTOCOL.cache_protocol}:{tag}:{index:04d}:{arm}", run_scope
    )


def _call_protocol_factory(tag: str, run_scope: str) -> Callable[[int, str], str]:
    """A typed per-call namespace resolver bound to one probe tag's run."""

    def protocol_for(index: int, arm: str) -> str:
        return _openrouter_cache_protocol(_call_cache_protocol(tag, index, arm, run_scope))

    return protocol_for


def _producer_protocol_factory(model: str, producer_scope: str) -> Callable[[int], str]:
    """A typed per-call namespace resolver for the producer's zero-shot rows."""

    def protocol_for(index: int) -> str:
        return _openrouter_cache_protocol(
            _producer_call_cache_protocol(model, index, "full", producer_scope)
        )

    return protocol_for


def _producer_call_cache_protocol(tag: str, index: int, arm_name: str, producer_scope: str) -> str:
    """The producer run's cache namespace for one scored zero-shot call."""

    return scoped_cache_protocol(
        f"{SYNTHETIC_ZERO_SHOT_PROTOCOL.cache_protocol}:{tag}:{index:04d}:{arm_name}",
        producer_scope,
    )


def _openrouter_cache_protocol(scoped: str) -> str:
    """The cache namespace a scored OpenRouter call of this run writes."""

    return f"openrouter-v1:{scoped}"


def _declared_cache_protocols(run_scope: str) -> list[str]:
    """Every cache namespace one run of this recorder writes, sorted.

    The receipt declares exactly this list and the store projection must match
    it, so a receipt cannot cite records made under another namespace. One
    namespace per scored call: the rotating schedule scores every registered
    arm at every position for every probe tag.
    """

    return sorted(
        _openrouter_cache_protocol(_call_cache_protocol(tag, index, arm, run_scope))
        for tag in MODELS.values()
        for index in range(DEFAULT_POSITIONS)
        for arm, _k, _deranged in _arm_specs()
    )


def _replay_configuration(
    *,
    positions: int,
    run_scope: str,
    endpoint: str = OPENROUTER_URL,
    openrouter_provider: dict | None = None,
) -> dict[str, object]:
    """The canonical configuration the in-context receipt binds this run to.

    The endpoint and model tags name the hosted transport: those are the
    requests the receipt's cache records actually bind. The registered probe
    tags stay in the parameters as the claim namespaces the per-tag cache
    protocols serve — each probe is scored fresh through the one hosted model.
    """

    configuration = network_recorder_configuration(
        endpoint=endpoint,
        call_protocol=_openrouter_cache_protocol(
            _call_cache_protocol(MODELS[_PRIMARY_PROBE_KEY], 0, _arm_specs()[0][0], run_scope)
        ),
        model_tags=[OPENROUTER_MODEL],
        parameters={
            "measurement_protocol": MEASUREMENT_PROTOCOL,
            "chain_protocol": SYNTHETIC_INCONTEXT_PROTOCOL.chain_version,
            "positions": positions,
            "position_seed": SYNTHETIC_INCONTEXT_PROTOCOL.position_seed,
            "position_generation": SYNTHETIC_INCONTEXT_PROTOCOL.position_generation,
            "trajectory_length": SYNTHETIC_INCONTEXT_PROTOCOL.query_window + 1,
            "target_start_tvd": independent_target_start_tvd(
                SYNTHETIC_INCONTEXT_PROTOCOL.query_window
            ),
            "query_window": SYNTHETIC_INCONTEXT_PROTOCOL.query_window,
            "response_inventory_size": len(SYNTHETIC_INCONTEXT_PROTOCOL.response_inventory),
            "context_window": SYNTHETIC_INCONTEXT_PROTOCOL.context_window,
            "response_tokens": SYNTHETIC_INCONTEXT_PROTOCOL.response_tokens,
            "prompt_token_overhead_margin": (
                SYNTHETIC_INCONTEXT_PROTOCOL.prompt_token_overhead_margin
            ),
            "maximum_prompt_characters": _maximum_prompt_characters(positions),
            "example_seed": SYNTHETIC_INCONTEXT_PROTOCOL.example_seed,
            "example_window": SYNTHETIC_INCONTEXT_PROTOCOL.example_window,
            "example_draw_generation": (SYNTHETIC_INCONTEXT_PROTOCOL.example_draw_generation),
            "example_internal_transitions_per_window": (
                SYNTHETIC_INCONTEXT_PROTOCOL.example_internal_transitions_per_window
            ),
            "headline_internal_transitions": (
                SYNTHETIC_INCONTEXT_PROTOCOL.headline_internal_transitions
            ),
            "headline_total_transitions": (SYNTHETIC_INCONTEXT_PROTOCOL.headline_total_transitions),
            "k_grid": list(SYNTHETIC_INCONTEXT_PROTOCOL.k_grid),
            "headline_k": SYNTHETIC_INCONTEXT_PROTOCOL.headline_k,
            "draw_attempts": SYNTHETIC_INCONTEXT_PROTOCOL.derangeable_draw_attempts,
            "warmup_policy": SYNTHETIC_INCONTEXT_PROTOCOL.warmup_policy,
            "arm_order": SYNTHETIC_INCONTEXT_PROTOCOL.arm_order,
            "corpus_chars": DEFAULT_CORPUS_CHARS,
            "corpus_seed": DEFAULT_CORPUS_SEED,
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

    Every scored call of this probe is deliberately live on a publication run,
    so its cache namespace is scoped by run and the run is the receipt's to
    state. The captured optional provider constraints also belong to that run;
    the rest comes from this module's code.
    """

    return _replay_configuration(
        positions=DEFAULT_POSITIONS,
        run_scope=run_scope,
        openrouter_provider=openrouter_provider,
    )


def _positive_positions(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("positions must be a positive integer")
    return parsed


def _digest_json(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scored_positions(count: int) -> list[tuple[str, str]]:
    """Construct the protocol's independent scored sampling units."""

    return independent_held_out_positions(
        count,
        window=SYNTHETIC_INCONTEXT_PROTOCOL.query_window,
        seed=SYNTHETIC_INCONTEXT_PROTOCOL.position_seed,
    )


def _example_pool(
    positions: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Construct the fixed training-stream pool, excluding scored prefixes."""

    scored_prefixes = {prefix for prefix, _gold in positions}
    return [
        pair
        for pair in letter_positions(
            generate_sequence(DEFAULT_CORPUS_CHARS, DEFAULT_CORPUS_SEED),
            window=SYNTHETIC_INCONTEXT_PROTOCOL.example_window,
        )
        if pair[0] not in scored_prefixes
    ]


def _maximum_prompt_characters(count: int) -> int:
    positions = _scored_positions(count)
    pool = _example_pool(positions)
    prompts = _all_arm_prompts(positions, pool)
    return max(len(prompt) for arm_prompts in prompts.values() for prompt in arm_prompts)


def _draw_diagnostic_claim_values(
    positions: list[tuple[str, str]],
    pool: list[tuple[str, str]],
) -> dict[str, int]:
    """Report the realized conditioning imposed by derangeable-set rejection.

    The treatment and deranged prompts share one accepted example draw at each
    position.  These claims make the otherwise-hidden rejection step observable:
    readers can see how often the initial seeded draw was rejected, rather than
    interpreting the accepted examples as unconditional samples from the pool.
    """

    values: dict[str, int] = {}
    for k in K_GRID:
        if not k:
            continue
        _pairs, attempts = draw_in_context_example_sets_with_diagnostics(
            positions,
            pool,
            k=k,
            example_seed=EXAMPLE_SEED,
            draw_attempts=_DERANGEABLE_DRAW_ATTEMPTS,
            arm=f"k{k}",
        )
        values[f"bench_ctx_draw_retries_k{k}"] = sum(attempt - 1 for attempt in attempts)
        values[f"bench_ctx_draw_retried_positions_k{k}"] = sum(attempt > 1 for attempt in attempts)
        values[f"bench_ctx_draw_max_attempts_used_k{k}"] = max(attempts)
    return values


def _mapping_control_flags(
    positions: list[tuple[str, str]],
    pool: list[tuple[str, str]],
) -> tuple[list[bool], list[bool]]:
    """Score a real terminal-mapping learner on treatment and deranged windows.

    The learner is deliberately small: for each prompt it estimates the next
    symbol conditional on the last symbol of every displayed prefix and uses a
    deterministic marginal fallback.  It consumes only the displayed labeled
    windows.  A treatment/control separation therefore proves the prompt pair
    contains detectable mapping evidence before any external probe is judged.
    """

    example_sets = draw_in_context_example_sets(
        positions,
        pool,
        k=HEADLINE_K,
        example_seed=EXAMPLE_SEED,
        draw_attempts=_DERANGEABLE_DRAW_ATTEMPTS,
        arm="mapping_positive_control",
    )

    def predict(prefix: str, examples: list[tuple[str, str]]) -> str:
        conditional = Counter(
            symbol for example_prefix, symbol in examples if example_prefix[-1] == prefix[-1]
        )
        counts = conditional or Counter(symbol for _example_prefix, symbol in examples)
        return max(
            SYNTHETIC_LETTERS,
            key=lambda symbol: (counts[symbol], -SYNTHETIC_LETTERS.index(symbol)),
        )

    treatment: list[bool] = []
    deranged: list[bool] = []
    for (prefix, gold), (real_examples, deranged_examples) in zip(
        positions, example_sets, strict=True
    ):
        treatment.append(predict(prefix, real_examples) == gold)
        deranged.append(predict(prefix, deranged_examples) == gold)
    return treatment, deranged


def _validate_zero_shot_producer_binding(store: CallCache, scope: str) -> None:
    """Authenticate and semantically replay the producer before consumer calls.

    ``scope`` is the producer run the producer's own receipt declares. Asking the
    producer for the configuration that scope implies is what lets this check
    authenticate a receipt whose cache namespaces no constant can predict.
    """

    producer = importlib.import_module("scripts.record_synthetic_benchmark_claims")

    receipt = validate_network_claim_binding(
        project_root=REPO_ROOT,
        recorder=PRODUCER_SOURCE,
        expected_configuration=producer.publication_replay_configuration(
            run_scope=scope,
            openrouter_provider=declared_openrouter_provider(REPO_ROOT, PRODUCER_SOURCE),
        ),
        cache_path=store.path,
    )
    producer.validate_publication_measurement(receipt, store)


def main() -> int:
    parser = argparse.ArgumentParser(description="Record the in-context letter probe.")
    parser.add_argument("--positions", type=_positive_positions, default=DEFAULT_POSITIONS)
    parser.add_argument(
        "--endpoint",
        default=OPENROUTER_URL,
        help="hosted chat endpoint (non-default endpoints are isolated-validation only)",
    )
    args = parser.parse_args()
    # Shares the bench_ namespace and the ledger with other benchmark recorders.
    with single_writer_lock(ledger_lock_path(REPO_ROOT)):
        return _measure_and_record(args)


def _measure_and_record(args: argparse.Namespace) -> int:
    """Measure all registered hosted cohorts, or refuse the run.

    There is deliberately no ``--model`` flag. Namespace replacement removes
    ``bench_ctx_`` labels this run did not produce, so a partial cohort sweep
    would remove the omitted cohorts' arms. Multiplicity correction covers the
    full declared family across all registered cohorts. Every cohort must be
    measured before publication; their labels do not identify distinct hosted
    model weights.
    """
    publication_run = using_publication_ledger(REPO_ROOT)
    # One scope for the whole run, fixed before the first server call. Every
    # publication scored call is deliberately live, so a fresh scope keeps its
    # replies separate from earlier runs. The producer run this consumer replays
    # has a scope of its own, read from its receipt.
    run_scope = generate_run_scope()
    print(f"in-context run scope: {run_scope}", flush=True)
    if publication_run:
        if args.endpoint.rstrip("/") != OPENROUTER_URL.rstrip("/"):
            raise SystemExit(
                "in-context probe refused: the publication ledger requires the canonical "
                f"endpoint {OPENROUTER_URL!r}, got {args.endpoint!r}; no claims written"
            )
        if args.positions != DEFAULT_POSITIONS:
            raise SystemExit(
                "in-context probe refused: the publication ledger requires the canonical "
                f"{DEFAULT_POSITIONS}-position population, got {args.positions}; use an "
                "absolute AILM_CLAIMS_PATH for reduced validation. No claims written."
            )
    openrouter_provider = configured_provider_options(OPENROUTER_MODEL)
    initial_source = source_manifest(REPO_ROOT, SOURCE)
    counts = load_synthetic_counts()
    if counts != build_synthetic_counts(n_chars=DEFAULT_CORPUS_CHARS, seed=DEFAULT_CORPUS_SEED):
        raise SystemExit(
            "in-context probe refused: vendor/synthetic_counts.json does not reproduce "
            "from its generator; the artifact is stale or hand-edited. No claims written."
        )

    # Canonical publication starts a fresh scoped sweep after interruption:
    # scored calls disable cache reads and any hits are refused. Isolated
    # validation permits cache replay. The store binds requested hosted model
    # identity, exact payload and run scope; a hosted slug digest does not
    # authenticate immutable serving weights.
    store = enable_call_cache(require_default_path=publication_run)
    if store is None:  # enabled=True above; defensive against future API drift
        raise SystemExit("in-context probe refused: call store could not be enabled")
    # The scored calls run through the OpenRouter transport, which keeps its
    # own module-level cache handle; binding it is what lets post_chat count
    # its hits and misses on the same store the schedule audit reads.
    set_call_cache(store)
    store.clear_access_log()
    # The producer's zero-shot replies are stored under the producer run's own
    # scope, and this recorder replays them in the cross-recorder agreement
    # check below. That is true whether or not this run is writing the
    # publication ledger, so the scope is read either way; only the receipt
    # authentication is a publication-run step.
    producer_scope = declared_run_scope(REPO_ROOT, PRODUCER_SOURCE)
    if producer_scope is None:
        raise SystemExit(
            "in-context probe refused: the producer receipt declares no run scope, so "
            "its stored zero-shot replies cannot be located; no claims written"
        )
    if publication_run:
        _validate_zero_shot_producer_binding(store, producer_scope)
    if declared_openrouter_provider(REPO_ROOT, PRODUCER_SOURCE) != openrouter_provider:
        raise NetworkClaimBindingError(
            "producer and consumer OpenRouter provider constraints differ; rerun the producer "
            "under the current route before consumer calls"
        )

    positions = _scored_positions(args.positions)
    ailm_flags = correctness_flags(synthetic_letter_predictions(counts, positions), positions)
    total = len(positions)

    pool = _example_pool(positions)
    if len(pool) < max(K_GRID) * 4:
        raise SystemExit(
            f"in-context probe refused: example pool has {len(pool)} prefix-disjoint "
            f"pairs, too few to draw {max(K_GRID)} per position without near-total reuse."
        )
    maximum_prompt_characters = max(
        len(prompt)
        for arm_prompts in _all_arm_prompts(positions, pool).values()
        for prompt in arm_prompts
    )
    if (
        maximum_prompt_characters
        + SYNTHETIC_INCONTEXT_PROTOCOL.response_tokens
        + SYNTHETIC_INCONTEXT_PROTOCOL.prompt_token_overhead_margin
        > SYNTHETIC_INCONTEXT_PROTOCOL.context_window
    ):
        raise SystemExit(
            "in-context probe refused: the declared hosted context window does not "
            "cover the longest prompt plus response and safety margin; no claims written"
        )

    mapping_treatment, mapping_deranged = _mapping_control_flags(positions, pool)
    mapping_treatment_rate = sum(mapping_treatment) / total
    mapping_deranged_rate = sum(mapping_deranged) / total
    mapping_harness = ControlHarness("synthetic_incontext_mapping_control")
    mapping_harness.assert_gate_is_live(
        mapping_treatment_rate,
        mapping_deranged_rate,
        "higher_is_better",
    )
    mapping_harness.verify_negative_control_fails(
        lambda flags: sum(flags) / len(flags),
        mapping_deranged,
        (mapping_treatment_rate + mapping_deranged_rate) / 2,
        "higher_is_better",
    )
    mapping_paired = paired_discordance(mapping_treatment, mapping_deranged)
    if mapping_paired["p_value"] >= POWER_ALPHA:
        raise SystemExit(
            "in-context probe refused: the fixed terminal-mapping positive control "
            "did not distinguish real from deranged demonstrations; no scored "
            "generations were made and no claims were written"
        )

    values: dict[str, object] = {
        "bench_ctx_openrouter_model": OPENROUTER_MODEL,
        "bench_ctx_openrouter_provider": (
            openrouter_provider["only"][0] if openrouter_provider is not None else "unpinned"
        ),
        "bench_ctx_mapping_control_top1_treatment": mapping_treatment_rate,
        "bench_ctx_mapping_control_top1_deranged": mapping_deranged_rate,
        "bench_ctx_mapping_control_paired_b": mapping_paired["b"],
        "bench_ctx_mapping_control_paired_c": mapping_paired["c"],
        "bench_ctx_mapping_control_paired_p": mapping_paired["p_value"],
        "bench_ctx_mapping_control_family_size": len(MAPPING_CONTROL_FAMILY_TEMPLATES),
        "bench_ctx_mapping_control_family_alpha": POWER_ALPHA
        / len(MAPPING_CONTROL_FAMILY_TEMPLATES),
        "bench_ctx_maximum_prompt_characters": maximum_prompt_characters,
    }
    values.update(_draw_diagnostic_claim_values(positions, pool))
    measurement_keys: set[str] = set()
    measured_cache_hits = 0
    measured_cache_misses = 0
    execution_trace: list[dict[str, object]] = []
    for name, tag in sorted(MODELS.items()):
        print(f"{tag} ({name}):", flush=True)
        # Authenticate the producer and its exact zero-shot population before
        # spending any calls on this consumer's arms. The re-score itself is
        # almost entirely a cache replay after the producer has run.
        _check_zero_shot_reproduces(
            tag, name, endpoint=args.endpoint, producer_scope=producer_scope
        )
        store.clear_access_log()
        hits_before = store.stats.hits
        misses_before = store.stats.misses
        arms, unparsable, model_trace = _score_all_arms_counterbalanced(
            positions,
            pool,
            cache_read=not publication_run,
            cache_protocol_for=_call_protocol_factory(tag, run_scope),
            endpoint=args.endpoint,
        )
        model_hits = store.stats.hits - hits_before
        model_misses = store.stats.misses - misses_before
        expected_calls = len(_arm_specs()) * total
        if publication_run and model_hits:
            raise SystemExit(
                "in-context probe refused: canonical arm-order provenance requires "
                f"fresh scored calls, but {tag} replayed {model_hits} cached "
                "request(s); use an isolated ledger for resumable validation. "
                "No claims written."
            )
        if model_hits + model_misses != expected_calls:
            raise SystemExit(
                "in-context probe refused: scored cache accounting does not match "
                f"the rotating schedule for {tag}: hits={model_hits}, "
                f"misses={model_misses}, expected={expected_calls}. No claims written."
            )
        measured_cache_hits += model_hits
        measured_cache_misses += model_misses
        measurement_keys.update(store.accessed_keys)
        execution_trace.extend({"model": tag, **entry} for entry in model_trace)
        for arm, flags in arms.items():
            print(
                f"  {arm:<16} {100 * sum(flags) / total:8.4f}%  unparsable {unparsable[arm]}",
                flush=True,
            )

        values.update(_claim_values(name, arms, unparsable, ailm_flags, total))
        if source_manifest(REPO_ROOT, SOURCE) != initial_source:
            raise SystemExit(
                "in-context probe refused: source or vendored input changed during "
                f"measurement of {tag}; no claims written"
            )

    values["bench_ctx_scored_cache_hits"] = measured_cache_hits
    values["bench_ctx_scored_cache_misses"] = measured_cache_misses
    if source_manifest(REPO_ROOT, SOURCE) != initial_source:
        raise SystemExit(
            "in-context probe refused: source or vendored input changed before ledger "
            "publication; no claims written"
        )
    if call_cache() is not store:
        raise SystemExit("in-context probe refused: active call store changed during measurement")
    if not measurement_keys:
        raise SystemExit(
            "in-context probe refused: scored measurement produced no trusted cache keys; "
            "no claims written"
        )
    maximum_prompt_tokens = maximum_prompt_eval_tokens(store, measurement_keys)
    values["bench_ctx_maximum_prompt_tokens"] = maximum_prompt_tokens
    measurement = _measurement_evidence(
        execution_trace=execution_trace,
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
    stale = [label for label in ledger.labels_owned_by(SOURCE) if label not in values]
    ledger.replace_claims(
        remove=stale,
        batches=[ClaimBatch(values=values, run_id=RUN_ID, source=SOURCE)],
    )
    print(
        f"recorded {len(values)} in-context claims for {len(MODELS)} probes over {total} positions"
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


def _score_arm(
    positions: list[tuple[str, str]],
    pool: list[tuple[str, str]],
    *,
    k: int,
    deranged: bool,
    arm: str,
    cache_protocol_for: Callable[[int], str],
    cache_read: bool = True,
    response_tokens: int = SYNTHETIC_INCONTEXT_PROTOCOL.response_tokens,
    endpoint: str = OPENROUTER_URL,
) -> tuple[list[bool], int]:
    """One arm of the sweep, scored through the shared parser.

    ``cache_protocol_for`` maps one scored call's position index to the
    complete cache namespace that call writes — the caller scopes it by arm,
    probe tag, scored slot and run.
    """
    try:
        prompts = build_in_context_synthetic_letter_prompts(
            positions,
            pool,
            k=k,
            deranged=deranged,
            example_seed=EXAMPLE_SEED,
            draw_attempts=_DERANGEABLE_DRAW_ATTEMPTS,
            arm=arm,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    flags: list[bool] = []
    bad_count = 0
    for index, ((_prefix, gold), prompt) in enumerate(zip(positions, prompts, strict=True)):
        parsed, bad = parse_synthetic_letter_answer(
            ask_openrouter(
                OPENROUTER_MODEL,
                prompt,
                "synthetic_letter",
                response_tokens,
                cache_protocol=cache_protocol_for(index),
                cache_read=cache_read,
                endpoint=endpoint,
            )
        )
        bad_count += int(bad)
        flags.append(parsed == gold)
    return flags, bad_count


def _arm_specs() -> tuple[tuple[str, int, bool], ...]:
    """Return every in-context arm in its declared, deterministic base order."""

    return tuple(
        (f"k{k}" + ("_deranged" if deranged else ""), k, deranged)
        for k in K_GRID
        for deranged in (False, True)
        if not (k == 0 and deranged)
    )


def _all_arm_prompts(
    positions: list[tuple[str, str]],
    pool: list[tuple[str, str]],
) -> dict[str, list[str]]:
    """Materialize the exact prompt registry shared by scoring and validation."""

    prompts: dict[str, list[str]] = {}
    for arm, k, deranged in _arm_specs():
        try:
            prompts[arm] = build_in_context_synthetic_letter_prompts(
                positions,
                pool,
                k=k,
                deranged=deranged,
                example_seed=EXAMPLE_SEED,
                draw_attempts=_DERANGEABLE_DRAW_ATTEMPTS,
                arm=arm,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
    return prompts


def _rotated_arm_names(position_index: int) -> tuple[str, ...]:
    """Counterbalance temporal slots by rotating the complete arm registry."""

    if position_index < 0:
        raise ValueError("position index must be non-negative")
    names = tuple(name for name, _k, _deranged in _arm_specs())
    offset = position_index % len(names)
    return names[offset:] + names[:offset]


def _score_all_arms_counterbalanced(
    positions: list[tuple[str, str]],
    pool: list[tuple[str, str]],
    *,
    cache_read: bool,
    cache_protocol_for: Callable[[int, str], str],
    endpoint: str = OPENROUTER_URL,
) -> tuple[dict[str, list[bool]], dict[str, int], list[dict[str, object]]]:
    """Score every arm per position, rotating which arm occupies each time slot.

    ``cache_protocol_for`` maps one scored call's position index and arm name
    to the complete cache namespace that call writes — the caller scopes it by
    probe tag, scored slot and run.
    """

    prompts = _all_arm_prompts(positions, pool)

    flags: dict[str, list[bool]] = {arm: [] for arm in prompts}
    bad = {arm: 0 for arm in prompts}
    trace: list[dict[str, object]] = []
    for index, (_prefix, gold) in enumerate(positions):
        for arm in _rotated_arm_names(index):
            prompt = prompts[arm][index]
            parsed, malformed = parse_synthetic_letter_answer(
                ask_openrouter(
                    OPENROUTER_MODEL,
                    prompt,
                    "synthetic_letter",
                    SYNTHETIC_INCONTEXT_PROTOCOL.response_tokens,
                    cache_protocol=cache_protocol_for(index, arm),
                    cache_read=cache_read,
                    endpoint=endpoint,
                )
            )
            trace.append(
                {
                    "position": index,
                    "arm": arm,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                }
            )
            flags[arm].append(parsed == gold)
            bad[arm] += int(malformed)
    return flags, bad, trace


def _expected_execution_trace(
    positions: list[tuple[str, str]],
    pool: list[tuple[str, str]],
) -> list[dict[str, object]]:
    """Reconstruct every ordered prompt hash without making model calls."""

    prompts = _all_arm_prompts(positions, pool)
    trace: list[dict[str, object]] = []
    for _name, model in sorted(MODELS.items()):
        for index, _position in enumerate(positions):
            for arm in _rotated_arm_names(index):
                prompt = prompts[arm][index]
                trace.append(
                    {
                        "model": model,
                        "position": index,
                        "arm": arm,
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    }
                )
    return trace


def _measurement_evidence(
    *,
    execution_trace: list[dict[str, object]],
    cache_hits: int,
    cache_misses: int,
    unique_cache_keys: int,
    maximum_prompt_tokens: int,
) -> dict[str, object]:
    return {
        "schema": MEASUREMENT_SCHEMA,
        "schedule": SYNTHETIC_INCONTEXT_PROTOCOL.arm_order,
        "warmup_policy": SYNTHETIC_INCONTEXT_PROTOCOL.warmup_policy,
        "warmup_calls": 0,
        "scored_calls": len(execution_trace),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "unique_cache_keys": unique_cache_keys,
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "ordered_trace_sha256": _digest_json(execution_trace),
    }


def _validate_measurement_fields(
    measurement: object,
    *,
    positions: int,
    cache_record_count: int,
) -> None:
    expected_fields = {
        "schema",
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
    if not isinstance(measurement, dict) or set(measurement) != expected_fields:
        raise NetworkClaimBindingError("in-context receipt has invalid measurement evidence fields")
    expected_calls = positions * len(_arm_specs()) * len(MODELS)
    if measurement["schema"] != MEASUREMENT_SCHEMA:
        raise NetworkClaimBindingError("unsupported in-context measurement schema")
    if measurement["schedule"] != SYNTHETIC_INCONTEXT_PROTOCOL.arm_order:
        raise NetworkClaimBindingError("in-context receipt arm-order policy drifted")
    if measurement["warmup_policy"] != SYNTHETIC_INCONTEXT_PROTOCOL.warmup_policy:
        raise NetworkClaimBindingError("in-context receipt warm-up policy drifted")
    if type(measurement["warmup_calls"]) is not int or measurement["warmup_calls"] != 0:
        raise NetworkClaimBindingError("in-context receipt warm-up count is invalid")
    if measurement["scored_calls"] != expected_calls:
        raise NetworkClaimBindingError("in-context receipt scored-call count is invalid")
    if measurement["cache_hits"] != 0 or measurement["cache_misses"] != expected_calls:
        raise NetworkClaimBindingError(
            "in-context publication receipt must bind one fresh miss per scored call"
        )
    unique_keys = measurement["unique_cache_keys"]
    if (
        not isinstance(unique_keys, int)
        or isinstance(unique_keys, bool)
        or not 0 < unique_keys <= expected_calls
        or unique_keys != cache_record_count
    ):
        raise NetworkClaimBindingError(
            "in-context receipt unique-key count disagrees with the cache binding"
        )
    maximum_tokens = measurement["maximum_prompt_tokens"]
    if (
        not isinstance(maximum_tokens, int)
        or isinstance(maximum_tokens, bool)
        or maximum_tokens <= 0
        or maximum_tokens
        + SYNTHETIC_INCONTEXT_PROTOCOL.response_tokens
        + SYNTHETIC_INCONTEXT_PROTOCOL.prompt_token_overhead_margin
        > SYNTHETIC_INCONTEXT_PROTOCOL.context_window
    ):
        raise NetworkClaimBindingError(
            "in-context receipt prompt-token maximum does not fit the declared context"
        )
    expected_positions = _scored_positions(positions)
    expected_pool = _example_pool(expected_positions)
    if measurement["ordered_trace_sha256"] != _digest_json(
        _expected_execution_trace(expected_positions, expected_pool)
    ):
        raise NetworkClaimBindingError(
            "in-context receipt ordered prompt trace does not reproduce from code"
        )


def _receipt_claim_value(receipt: Mapping[str, object], label: str) -> object:
    claims = receipt.get("claims")
    if not isinstance(claims, dict) or not isinstance(claims.get("entries"), dict):
        raise NetworkClaimBindingError("in-context receipt claim projection is invalid")
    entry = claims["entries"].get(label)
    if not isinstance(entry, dict) or "value" not in entry:
        raise NetworkClaimBindingError(f"in-context receipt is missing claim {label}")
    return entry["value"]


def _replay_result_claims(
    receipt: Mapping[str, object],
    store: CallCache,
    cache_keys: list[str],
    *,
    positions_count: int = DEFAULT_POSITIONS,
) -> dict[str, object]:
    """Reparse every scored reply and rebuild the complete context namespace.

    ``positions_count`` is an internal test seam for exercising the full semantic
    replay contract without materializing the full declared cohort/arm cache.
    The public validator never passes it and therefore continues to require
    ``DEFAULT_POSITIONS``.
    """

    if positions_count <= 0:
        raise NetworkClaimBindingError("in-context semantic replay population must be positive")

    cache = receipt.get("cache")
    configuration = receipt.get("configuration")
    if not isinstance(cache, dict) or not isinstance(configuration, dict):
        raise NetworkClaimBindingError("in-context receipt replay envelope is invalid")
    endpoint = configuration.get("endpoint")
    protocol = configuration.get("call_protocol")
    parameters = configuration.get("parameters")
    identity_rows = cache.get("model_identities")
    if (
        not isinstance(endpoint, str)
        or not isinstance(protocol, str)
        or not isinstance(parameters, dict)
        or parameters.get("positions") != positions_count
        or not isinstance(identity_rows, list)
    ):
        raise NetworkClaimBindingError("in-context receipt replay identity is invalid")
    # The run the receipt states is the run its records were made under. Reading
    # it here rather than from the protocol constant is what lets an older
    # receipt replay against its own records: the constant names the
    # measurement, not the run.
    if "openrouter_provider" in parameters and not isinstance(
        parameters["openrouter_provider"], dict
    ):
        raise NetworkClaimBindingError("receipt provider declaration is invalid")
    openrouter_provider = validate_provider_options(parameters.get("openrouter_provider"))
    declared_scope = validate_run_scope(parameters.get("run_scope"))
    if parameters.get("cache_protocols") != _declared_cache_protocols(declared_scope):
        raise NetworkClaimBindingError(
            "in-context receipt cache namespaces do not match its declared run scope"
        )
    if parameters.get("probe_tags") != sorted(MODELS.values()):
        raise NetworkClaimBindingError(
            "in-context receipt probe tags do not match the registered probe labels"
        )
    if protocol != _openrouter_cache_protocol(
        _call_cache_protocol(MODELS[_PRIMARY_PROBE_KEY], 0, _arm_specs()[0][0], declared_scope)
    ):
        raise NetworkClaimBindingError(
            "in-context receipt call protocol does not match its declared run scope"
        )
    identities = {
        row["model"]: row["identity"]
        for row in identity_rows
        if isinstance(row, dict)
        and isinstance(row.get("model"), str)
        and isinstance(row.get("identity"), str)
    }
    if set(identities) != {OPENROUTER_MODEL}:
        raise NetworkClaimBindingError("in-context receipt model identities are incomplete")
    transport_identity = identities[OPENROUTER_MODEL]

    selected = set(cache_keys)
    cached = {key: (payload, reply) for key, payload, reply in store.entries() if key in selected}
    if set(cached) != selected:
        raise NetworkClaimBindingError("in-context semantic replay is missing cached replies")

    positions = _scored_positions(positions_count)
    pool = _example_pool(positions)
    prompts = _all_arm_prompts(positions, pool)
    counts = load_synthetic_counts()
    ailm_flags = correctness_flags(synthetic_letter_predictions(counts, positions), positions)
    mapping_treatment, mapping_deranged = _mapping_control_flags(positions, pool)
    mapping_paired = paired_discordance(mapping_treatment, mapping_deranged)
    values: dict[str, object] = {
        "bench_ctx_openrouter_model": OPENROUTER_MODEL,
        "bench_ctx_openrouter_provider": (
            openrouter_provider["only"][0] if openrouter_provider is not None else "unpinned"
        ),
        "bench_ctx_mapping_control_top1_treatment": (sum(mapping_treatment) / len(positions)),
        "bench_ctx_mapping_control_top1_deranged": (sum(mapping_deranged) / len(positions)),
        "bench_ctx_mapping_control_paired_b": mapping_paired["b"],
        "bench_ctx_mapping_control_paired_c": mapping_paired["c"],
        "bench_ctx_mapping_control_paired_p": mapping_paired["p_value"],
        "bench_ctx_mapping_control_family_size": len(MAPPING_CONTROL_FAMILY_TEMPLATES),
        "bench_ctx_mapping_control_family_alpha": POWER_ALPHA
        / len(MAPPING_CONTROL_FAMILY_TEMPLATES),
        "bench_ctx_maximum_prompt_characters": _maximum_prompt_characters(len(positions)),
        "bench_ctx_maximum_prompt_tokens": maximum_prompt_eval_tokens(store, cache_keys),
        "bench_ctx_scored_cache_hits": 0,
        "bench_ctx_scored_cache_misses": (len(MODELS) * len(positions) * len(_arm_specs())),
    }
    values.update(_draw_diagnostic_claim_values(positions, pool))
    expected_keys: set[str] = set()
    for name, model in sorted(MODELS.items()):
        arms: dict[str, list[bool]] = {arm: [] for arm, _k, _deranged in _arm_specs()}
        unparsable = {arm: 0 for arm in arms}
        for index, (_prefix, gold) in enumerate(positions):
            for arm in _rotated_arm_names(index):
                prompt = prompts[arm][index]
                payload = _chat_payload(
                    OPENROUTER_MODEL,
                    prompt,
                    "synthetic_letter",
                    SYNTHETIC_INCONTEXT_PROTOCOL.response_tokens,
                    DEFAULT_SEED,
                )
                if openrouter_provider is not None:
                    payload["provider"] = openrouter_provider
                request = CacheRequest(
                    payload=payload,
                    endpoint=endpoint,
                    protocol=_openrouter_cache_protocol(
                        _call_cache_protocol(model, index, arm, declared_scope)
                    ),
                    identity=transport_identity,
                )
                expected_keys.add(request.key)
                record = cached.get(request.key)
                if record is None or record[0] != payload:
                    raise NetworkClaimBindingError(
                        f"in-context semantic replay lacks exact {model} {arm} request at {index}"
                    )
                parsed, bad = parse_synthetic_letter_answer(str(record[1].get("response", "")))
                arms[arm].append(parsed == gold)
                unparsable[arm] += int(bad)
        try:
            values.update(_claim_values(name, arms, unparsable, ailm_flags, len(positions)))
        except SystemExit as error:
            raise NetworkClaimBindingError(
                f"in-context cached outcomes cannot produce the declared claims: {error}"
            ) from error
    if expected_keys != selected:
        raise NetworkClaimBindingError(
            "in-context receipt cache key set is not the exact scored request set"
        )
    return values


def _validate_semantic_replay(
    receipt: Mapping[str, object],
    store: CallCache,
    cache_keys: list[str],
    *,
    positions_count: int = DEFAULT_POSITIONS,
) -> None:
    """Require that cached replies reproduce exactly the receipt-owned claims."""

    derived = _replay_result_claims(
        receipt,
        store,
        cache_keys,
        positions_count=positions_count,
    )
    claims = receipt.get("claims")
    entries = claims.get("entries") if isinstance(claims, dict) else None
    if not isinstance(entries, dict):
        raise NetworkClaimBindingError("in-context receipt claim projection is invalid")
    foreign_labels = sorted(
        label
        for label in entries
        if not isinstance(label, str) or not label.startswith("bench_ctx_")
    )
    if foreign_labels:
        raise NetworkClaimBindingError(
            f"in-context recorder owns claims outside its declared namespace: {foreign_labels}"
        )
    recorded = {
        label: entry.get("value")
        for label, entry in entries.items()
        if isinstance(label, str) and label.startswith("bench_ctx_") and isinstance(entry, dict)
    }
    mismatches = {
        label: (recorded.get(label), expected)
        for label, expected in derived.items()
        if recorded.get(label) != expected
    }
    missing_or_extra = sorted(set(recorded) ^ set(derived))
    if mismatches or missing_or_extra:
        raise NetworkClaimBindingError(
            "in-context receipt claims do not rederive from cached replies: "
            f"mismatches={mismatches}, missing_or_extra={missing_or_extra}"
        )


def validate_publication_measurement(receipt: Mapping[str, object], store: CallCache) -> None:
    """Reconstruct the full rotating schedule and bind it to claims/cache."""

    cache = receipt.get("cache")
    if not isinstance(cache, dict) or not isinstance(cache.get("record_count"), int):
        raise NetworkClaimBindingError("in-context receipt cache projection is invalid")
    _validate_measurement_fields(
        receipt.get("measurement"),
        positions=DEFAULT_POSITIONS,
        cache_record_count=cache["record_count"],
    )
    measurement = receipt["measurement"]
    if not isinstance(measurement, dict):
        raise TypeError("measurement must be a dict")
    cache_keys = cache.get("keys")
    if not isinstance(cache_keys, list) or measurement["maximum_prompt_tokens"] != (
        maximum_prompt_eval_tokens(store, cache_keys)
    ):
        raise NetworkClaimBindingError(
            "in-context receipt prompt-token maximum disagrees with cached replies"
        )
    _validate_semantic_replay(receipt, store, cache_keys)


def _check_zero_shot_reproduces(
    model: str,
    name: str,
    *,
    producer_scope: str,
    endpoint: str = OPENROUTER_URL,
) -> None:
    """Re-score the zero-shot arm on the zero-shot recorder's own population.

    This is a cross-recorder agreement check, and it only means anything on
    matched positions: the two recorders subsample independently, so at
    different position counts they score different sets and a disagreement
    would carry no information. So the population is taken from the ledger —
    ``bench_synth_positions`` — and the arm is re-scored there. Nearly all of
    those calls are already in the store from the zero-shot run, so the check
    costs approximately nothing. Missing producer claims are a hard refusal,
    checked before this consumer spends calls on any primary arm.
    """
    ledger = load_ledger(REPO_ROOT)
    accuracy_label = f"bench_synth_letter_top1_{name}"
    expected: dict[str, object] = {
        "bench_synth_protocol_version": PRODUCER_MEASUREMENT_PROTOCOL,
        "bench_synth_chain_version": SYNTHETIC_ZERO_SHOT_PROTOCOL.chain_version,
        "bench_synth_query_window": SYNTHETIC_ZERO_SHOT_PROTOCOL.query_window,
        "bench_synth_query_context_transitions_max": (
            SYNTHETIC_ZERO_SHOT_PROTOCOL.query_context_transitions_max
        ),
        "bench_synth_response_inventory_size": len(SYNTHETIC_ZERO_SHOT_PROTOCOL.response_inventory),
        "bench_synth_alphabet_size": len(SYNTHETIC_ZERO_SHOT_PROTOCOL.response_inventory),
        "bench_synth_position_generation": (SYNTHETIC_ZERO_SHOT_PROTOCOL.position_generation),
        "bench_synth_position_seed": SYNTHETIC_ZERO_SHOT_PROTOCOL.position_seed,
        "bench_synth_trajectory_length": (SYNTHETIC_ZERO_SHOT_PROTOCOL.query_window + 1),
        "bench_synth_target_start_tvd": independent_target_start_tvd(
            SYNTHETIC_ZERO_SHOT_PROTOCOL.query_window
        ),
        "bench_synth_corpus_chars": DEFAULT_CORPUS_CHARS,
        "bench_synth_corpus_seed": DEFAULT_CORPUS_SEED,
        "bench_synth_context_window": SYNTHETIC_ZERO_SHOT_PROTOCOL.context_window,
        "bench_synth_response_tokens": SYNTHETIC_ZERO_SHOT_PROTOCOL.response_tokens,
        "bench_synth_prompt_token_overhead_margin": (
            SYNTHETIC_ZERO_SHOT_PROTOCOL.prompt_token_overhead_margin
        ),
    }
    required_labels = (*expected, "bench_synth_positions", accuracy_label)
    entries = {label: ledger.get(label) for label in required_labels}
    missing = [label for label, entry in entries.items() if entry is None]
    if missing:
        raise SystemExit(
            "in-context probe refused: its no-demonstration producer has not recorded "
            f"the required protocol, population, and accuracy claims {missing}; run "
            "scripts/record_synthetic_benchmark_claims.py first. No claims written."
        )

    for label, entry in entries.items():
        if entry is None:
            raise ValueError(f"ledger entry {label!r} is missing")
        if entry.get("source") != PRODUCER_SOURCE or entry.get("run_id") != PRODUCER_RUN_ID:
            raise SystemExit(
                "in-context probe refused: producer claim "
                f"{label} is not bound to {PRODUCER_SOURCE} / {PRODUCER_RUN_ID}; "
                "no claims written"
            )
    mismatches = {
        label: (entry["value"], expected_value)
        for label, expected_value in expected.items()
        if (entry := entries[label]) is not None and entry.get("value") != expected_value
    }
    if mismatches:
        raise SystemExit(
            "in-context probe refused: producer protocol contract disagrees with "
            f"this consumer: {mismatches}; no claims written"
        )

    positions_entry = entries["bench_synth_positions"]
    accuracy = entries[accuracy_label]
    if positions_entry is None or accuracy is None:
        raise ValueError("required bench_synth ledger entries are missing")
    recorded_n = int(positions_entry["value"])
    if recorded_n <= 0:
        raise SystemExit(
            "in-context probe refused: producer population must be positive; no claims written"
        )
    matched = independent_held_out_positions(
        recorded_n,
        window=SYNTHETIC_ZERO_SHOT_PROTOCOL.query_window,
        seed=SYNTHETIC_ZERO_SHOT_PROTOCOL.position_seed,
    )
    flags, _bad = _score_arm(
        matched,
        [],
        k=0,
        deranged=False,
        arm="k0_reproduction",
        cache_protocol_for=_producer_protocol_factory(model, producer_scope),
        response_tokens=SYNTHETIC_ZERO_SHOT_PROTOCOL.response_tokens,
        endpoint=endpoint,
    )
    recorded = float(accuracy["value"])
    measured = sum(flags) / len(matched)
    if abs(measured - recorded) > 1e-9:
        raise SystemExit(
            f"in-context probe refused: re-scoring the no-demonstration arm on the "
            f"recorder's own {recorded_n} positions gave {measured:.6f}, but "
            f"{accuracy_label} in the ledger is {recorded:.6f}. One of the two recorders "
            f"has drifted; no claims written."
        )
    print(f"  no-demonstration arm reproduces {accuracy_label} on its own {recorded_n} positions")


def _claim_values(
    name: str,
    arms: dict[str, list[bool]],
    unparsable: dict[str, int],
    ailm_flags: list[bool],
    total: int,
) -> dict[str, object]:
    values: dict[str, object] = {
        "bench_ctx_protocol_version": MEASUREMENT_PROTOCOL,
        "bench_ctx_chain_version": SYNTHETIC_INCONTEXT_PROTOCOL.chain_version,
        "bench_ctx_query_window": SYNTHETIC_INCONTEXT_PROTOCOL.query_window,
        "bench_ctx_query_context_transitions_max": (SYNTHETIC_INCONTEXT_PROTOCOL.query_window - 1),
        "bench_ctx_response_inventory_size": len(SYNTHETIC_INCONTEXT_PROTOCOL.response_inventory),
        "bench_ctx_context_window": SYNTHETIC_INCONTEXT_PROTOCOL.context_window,
        "bench_ctx_response_tokens": SYNTHETIC_INCONTEXT_PROTOCOL.response_tokens,
        "bench_ctx_prompt_token_overhead_margin": (
            SYNTHETIC_INCONTEXT_PROTOCOL.prompt_token_overhead_margin
        ),
        "bench_ctx_positions": total,
        "bench_ctx_position_generation": (SYNTHETIC_INCONTEXT_PROTOCOL.position_generation),
        "bench_ctx_position_seed": SYNTHETIC_INCONTEXT_PROTOCOL.position_seed,
        "bench_ctx_trajectory_length": SYNTHETIC_INCONTEXT_PROTOCOL.query_window + 1,
        "bench_ctx_target_start_tvd": independent_target_start_tvd(
            SYNTHETIC_INCONTEXT_PROTOCOL.query_window
        ),
        "bench_ctx_example_seed": SYNTHETIC_INCONTEXT_PROTOCOL.example_seed,
        "bench_ctx_example_window": SYNTHETIC_INCONTEXT_PROTOCOL.example_window,
        "bench_ctx_example_draw_generation": (SYNTHETIC_INCONTEXT_PROTOCOL.example_draw_generation),
        "bench_ctx_example_internal_transitions_per_window": (
            SYNTHETIC_INCONTEXT_PROTOCOL.example_internal_transitions_per_window
        ),
        "bench_ctx_headline_internal_transitions": (
            SYNTHETIC_INCONTEXT_PROTOCOL.headline_internal_transitions
        ),
        "bench_ctx_headline_total_transitions": (
            SYNTHETIC_INCONTEXT_PROTOCOL.headline_total_transitions
        ),
        "bench_ctx_k_grid": ",".join(str(k) for k in SYNTHETIC_INCONTEXT_PROTOCOL.k_grid),
        "bench_ctx_headline_k": SYNTHETIC_INCONTEXT_PROTOCOL.headline_k,
        "bench_ctx_derangeable_draw_attempts": (
            SYNTHETIC_INCONTEXT_PROTOCOL.derangeable_draw_attempts
        ),
        "bench_ctx_corpus_chars": DEFAULT_CORPUS_CHARS,
        "bench_ctx_corpus_seed": DEFAULT_CORPUS_SEED,
        "bench_ctx_producer_protocol_version": (SYNTHETIC_ZERO_SHOT_PROTOCOL.version),
        "bench_ctx_warmup_policy": SYNTHETIC_INCONTEXT_PROTOCOL.warmup_policy,
        "bench_ctx_arm_order": SYNTHETIC_INCONTEXT_PROTOCOL.arm_order,
        "bench_ctx_cache_protocol": SYNTHETIC_INCONTEXT_PROTOCOL.cache_protocol,
        "bench_ctx_uniform_baseline": UNIFORM_BASELINE,
        "bench_ctx_top1_ailm": sum(ailm_flags) / total,
        # The declared family includes every template for every registered
        # cohort. Both its size and its corrected alpha must use that full
        # product; cohort labels do not denote distinct hosted model weights.
        "bench_ctx_family_size": len(FAMILY_TEMPLATES) * len(MODELS),
        "bench_ctx_family_alpha": FAMILY_ALPHA / (len(FAMILY_TEMPLATES) * len(MODELS)),
    }
    for arm, flags in arms.items():
        low, high = wilson_interval(sum(flags), total)
        values[f"bench_ctx_top1_{arm}_{name}"] = sum(flags) / total
        values[f"bench_ctx_top1_wilson_low_{arm}_{name}"] = low
        values[f"bench_ctx_top1_wilson_high_{arm}_{name}"] = high
        values[f"bench_ctx_unparsable_{arm}_{name}"] = unparsable[arm]

    for k in K_GRID:
        if not k:
            continue
        for label, comparator in (
            (f"bench_ctx_k{k}_vs_zero", arms["k0"]),
            (f"bench_ctx_k{k}_vs_deranged", arms[f"k{k}_deranged"]),
        ):
            paired = paired_discordance(arms[f"k{k}"], comparator)
            values[f"{label}_paired_b_{name}"] = paired["b"]
            values[f"{label}_paired_c_{name}"] = paired["c"]
            values[f"{label}_paired_p_{name}"] = paired["p_value"]

    headline = paired_discordance(arms[f"k{HEADLINE_K}"], ailm_flags)
    delta_low, delta_high = paired_difference_interval(headline["b"], headline["c"], total)
    values[f"bench_ctx_k{HEADLINE_K}_vs_ailm_paired_b_{name}"] = headline["b"]
    values[f"bench_ctx_k{HEADLINE_K}_vs_ailm_paired_c_{name}"] = headline["c"]
    values[f"bench_ctx_k{HEADLINE_K}_vs_ailm_paired_p_{name}"] = headline["p_value"]
    values[f"bench_ctx_k{HEADLINE_K}_vs_ailm_delta_{name}"] = (
        sum(arms[f"k{HEADLINE_K}"]) - sum(ailm_flags)
    ) / total
    values[f"bench_ctx_k{HEADLINE_K}_vs_ailm_delta_low_{name}"] = delta_low
    values[f"bench_ctx_k{HEADLINE_K}_vs_ailm_delta_high_{name}"] = delta_high
    # The quantity a reader actually wants: how much of the zero-shot gap the
    # examples closed. Reported as a signed ratio so it does not have to be
    # recomputed from three other labels in the prose.
    zero_gap = sum(ailm_flags) - sum(arms["k0"])
    if zero_gap == 0:
        raise SystemExit(
            f"in-context probe refused: the {name} no-demonstration arm has zero "
            "gap to AILM, so the declared gap-closed ratio is undefined; no claims written"
        )
    values[f"bench_ctx_gap_closed_ratio_{name}"] = (
        sum(arms[f"k{HEADLINE_K}"]) - sum(arms["k0"])
    ) / zero_gap
    return values


if __name__ == "__main__":
    raise SystemExit(main())
