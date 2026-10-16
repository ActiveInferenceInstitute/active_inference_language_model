"""Contracts for the in-context training-data arm.

This arm exists to answer one objection: that AILM's external margins are a
fitting artifact, because AILM is fitted to a corpus and every probe meets the
positions zero-shot. Answering it requires that the k-shot and zero-shot arms
differ ONLY in the examples, that the examples cannot contain the answers, and
that the control arm cannot silently collapse into its treatment. Each of those
is a way the measurement could look clean while measuring nothing, so each is
pinned here.
"""

from __future__ import annotations

import importlib
import inspect
import itertools
import json
import random
from pathlib import Path

import pytest

from ailm.benchmarks import (
    build_in_context_synthetic_letter_prompt,
    build_in_context_synthetic_letter_prompts,
    build_prompt,
    derange_examples,
    draw_in_context_example_sets,
    draw_in_context_example_sets_with_diagnostics,
    paired_discordance,
)
from ailm.benchmarks.synthetic_protocol import (
    SYNTHETIC_INCONTEXT_PROTOCOL,
    SYNTHETIC_ZERO_SHOT_PROTOCOL,
)
from ailm.call_cache import CacheRequest, CallCache
from ailm.control import ControlHarness, GateVacuousError
from ailm.levels.synthetic import (
    DEFAULT_CORPUS_CHARS,
    DEFAULT_CORPUS_SEED,
    SYNTHETIC_LETTERS,
    independent_target_start_tvd,
)
from ailm.replay_provenance import NetworkClaimBindingError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECORDER = PROJECT_ROOT / "scripts" / "record_incontext_benchmark_claims.py"


#: One fixture run's scope. Shaped like a recorder-generated run id, because
#: only that shape is publishable.
RUN_SCOPE = "20260905T000000000042Z"


def _recorder():
    return importlib.import_module("scripts.record_incontext_benchmark_claims")


def test_zero_shot_producer_and_in_context_consumer_register_identical_probes():
    """The consumer cannot authenticate a zero-shot row its producer never writes."""
    zero_shot = importlib.import_module("scripts.record_synthetic_benchmark_claims")
    module = _recorder()
    assert module.MODELS == zero_shot.MODELS
    assert module.MODELS, "the shared probe inventory must not be empty"
    scope = RUN_SCOPE
    configuration = module.publication_replay_configuration(run_scope=scope)
    parameters = configuration["parameters"]
    # The receipt binds the hosted transport, exactly as the producer's
    # receipt does: the endpoint and model tag name what the cache records
    # actually carry, while the registered probe tags stay in the parameters
    # as the claim namespaces the per-tag protocols serve. The namespace also
    # carries the scored slot, so a recurring prompt under the hosted
    # transport's non-deterministic replies never collides with itself.
    assert configuration["endpoint"] == module.OPENROUTER_URL
    assert configuration["model_tags"] == [module.OPENROUTER_MODEL]
    assert parameters["probe_tags"] == sorted(module.MODELS.values())
    primary_tag = module.MODELS[sorted(module.MODELS)[0]]
    first_arm = module._arm_specs()[0][0]
    assert configuration["call_protocol"] == (
        f"openrouter-v1:{SYNTHETIC_INCONTEXT_PROTOCOL.cache_protocol}"
        f":{primary_tag}:0000:{first_arm}/{scope}"
    )
    assert parameters["cache_protocols"] == sorted(
        f"openrouter-v1:{SYNTHETIC_INCONTEXT_PROTOCOL.cache_protocol}"
        f":{tag}:{index:04d}:{arm}/{scope}"
        for tag in sorted(module.MODELS.values())
        for index in range(SYNTHETIC_INCONTEXT_PROTOCOL.positions)
        for arm, _k, _deranged in module._arm_specs()
    )
    assert parameters["run_scope"] == scope
    assert parameters["measurement_protocol"] == SYNTHETIC_INCONTEXT_PROTOCOL.version
    assert parameters["position_generation"] == (SYNTHETIC_INCONTEXT_PROTOCOL.position_generation)
    assert parameters["example_draw_generation"] == (
        SYNTHETIC_INCONTEXT_PROTOCOL.example_draw_generation
    )
    assert parameters["k_grid"] == list(SYNTHETIC_INCONTEXT_PROTOCOL.k_grid)


def test_publication_consumer_authenticates_the_producer_before_scored_calls():
    module = _recorder()
    body = inspect.getsource(module._measure_and_record)
    binding = body.index("_validate_zero_shot_producer_binding(store, producer_scope)")
    mapping = body.index("_mapping_control_flags(")
    scored = body.index("_score_all_arms_counterbalanced(")
    assert binding < mapping < scored


def test_synthetic_k_shot_prompt_ends_with_the_executed_zero_shot_prompt_verbatim():
    """The measured stipulated-language arms vary labeled windows, not instructions."""
    zero = build_prompt("synthetic_letter", "adad")
    k_shot = build_in_context_synthetic_letter_prompt(
        "adad",
        [("a", "d"), ("x", "u")],
    )
    assert k_shot.endswith(zero)
    assert k_shot != zero


def test_the_prompt_refuses_k_of_zero():
    """k=0 is the zero-shot arm, and it already exists.

    Producing a near-copy here would put two slightly different prompts behind
    one label, which is how a benchmark ends up comparing itself to a variant of
    itself.
    """
    with pytest.raises(ValueError, match="zero-shot"):
        build_in_context_synthetic_letter_prompt("adad", [])


def test_the_prompt_shows_every_example_and_the_count_it_claims():
    examples = [("ab", "c"), ("de", "f"), ("gh", "i")]
    prompt = build_in_context_synthetic_letter_prompt("zz", examples)
    assert "Here are 3 observed continuations" in prompt
    for prefix, symbol in examples:
        assert f"<{prefix}> -> {symbol!r}" in prompt


def test_the_derangement_preserves_everything_except_the_mapping():
    """The control must preserve the declared character-level margins.

    Prompt length, the displayed prefixes and the multiset of displayed symbols
    all survive; only the prefix-to-symbol correspondence is destroyed. A control
    that also changed the symbol distribution would confound "the mapping is
    used" with "the alphabet was displayed differently". No tokenizer-level
    entropy or token-count identity is claimed.
    """
    examples = [("ab", "c"), ("cd", "e"), ("ef", "g"), ("gh", "i")]
    deranged = derange_examples(examples)
    assert deranged is not None
    assert [p for p, _ in deranged] == [p for p, _ in examples]
    assert sorted(s for _, s in deranged) == sorted(s for _, s in examples)
    assert all(new != old for (_, new), (_, old) in zip(deranged, examples, strict=True))


def test_the_derangement_succeeds_whenever_one_exists():
    """Completeness, checked against brute force on the hard cases.

    The first version searched only the k-1 rotations of the input, which fails
    on inputs that are perfectly derangeable — and a search that refuses
    satisfiable inputs makes the refusal rate a property of the algorithm rather
    than of the data. Verified here against exhaustive permutation over a
    three-symbol alphabet, where near-half multiplicities are common.
    """
    rng = random.Random(7)
    for _ in range(600):
        symbols = [rng.choice("abc") for _ in range(rng.choice([2, 3, 4, 5, 6, 7]))]
        examples = [(f"p{i}", symbol) for i, symbol in enumerate(symbols)]
        exists = any(
            all(candidate[i] != symbols[i] for i in range(len(symbols)))
            for candidate in set(itertools.permutations(symbols))
        )
        assert (derange_examples(examples) is not None) is exists


def test_the_derangement_does_not_refuse_at_the_headline_k():
    """The refusal budget must be a rare exception, not the common path.

    At the headline k the recorder redraws until a derangement exists. If
    refusals were routine there, the examples would come from a heavily
    truncated distribution and "k examples" would silently mean something other
    than k draws from the pool.
    """
    rng = random.Random(11)
    refusals = sum(
        derange_examples([(f"p{i}", rng.choice("abcdefghijklmnopqrstuvwxyz")) for i in range(128)])
        is None
        for _ in range(200)
    )
    assert refusals == 0


def test_the_derangement_refuses_rather_than_returning_its_input():
    """NON-VACUITY, and the defect this control is most likely to acquire.

    When no fixed-point-free rotation exists, the tempting fallback is to return
    the examples unchanged. That makes the control prompt byte-identical to the
    treatment prompt, guarantees a p-value of 1, and publishes "the examples are
    not being used" on the strength of having run the same experiment twice.
    """
    assert derange_examples([("a", "x"), ("b", "x"), ("c", "x")]) is None


def test_the_treatment_and_control_arms_see_identical_examples():
    """Lockstep, which is the whole validity of the derangement contrast.

    The control differs from the treatment in the prefix-to-symbol mapping and
    in nothing else — same prefixes, same order, same symbol multiset, same
    prompt length. If the two arms drew independently, a difference between them
    would be partly a difference in which examples were shown, and the contrast
    would stop being a test of whether the mapping is used.

    The draw is conditioned on being derangeable, and this is where that
    condition earns its cost: applying it on the treatment path too is what
    keeps the two random streams aligned.
    """
    module = _recorder()
    positions = [("aaa", "b"), ("bbb", "c"), ("ccc", "d"), ("ddd", "e")]
    pool = [(f"p{i}", "abcdefghijklmnopqrstuvwxyz"[i % 26]) for i in range(500)]
    common = {
        "k": 4,
        "example_seed": module.EXAMPLE_SEED,
        "draw_attempts": module._DERANGEABLE_DRAW_ATTEMPTS,
    }
    plain = build_in_context_synthetic_letter_prompts(
        positions,
        pool,
        deranged=False,
        arm="plain",
        **common,
    )
    control = build_in_context_synthetic_letter_prompts(
        positions,
        pool,
        deranged=True,
        arm="deranged",
        **common,
    )

    assert len(plain) == len(control) == len(positions)
    for left, right in zip(plain, control, strict=True):
        assert len(left) == len(right), "the control changed the prompt length"
        assert _example_prefixes(left) == _example_prefixes(right)
        assert sorted(_example_symbols(left)) == sorted(_example_symbols(right))
        assert _example_symbols(left) != _example_symbols(right), "the control was not deranged"


def test_example_draws_are_stable_per_position_under_reordering_and_insertion():
    """A position owns its RNG stream; unrelated population edits cannot shift it."""
    protocol = SYNTHETIC_INCONTEXT_PROTOCOL
    positions = [("adad", "u"), ("xuxa", "w"), ("iniz", "o")]
    pool = [(prefix, symbol) for prefix in SYNTHETIC_LETTERS for symbol in SYNTHETIC_LETTERS]
    options = {
        "k": 16,
        "example_seed": protocol.example_seed,
        "draw_attempts": protocol.derangeable_draw_attempts,
        "arm": "stability",
    }
    baseline = draw_in_context_example_sets(positions, pool, **options)
    reordered_positions = [positions[2], ("vvvv", "a"), positions[0], positions[1]]
    reordered = draw_in_context_example_sets(reordered_positions, pool, **options)
    baseline_by_prefix = {
        position[0]: examples for position, examples in zip(positions, baseline, strict=True)
    }
    reordered_by_prefix = {
        position[0]: examples
        for position, examples in zip(reordered_positions, reordered, strict=True)
    }
    assert {
        prefix: reordered_by_prefix[prefix] for prefix in baseline_by_prefix
    } == baseline_by_prefix


def test_synthetic_treatment_and_deranged_draws_share_prefixes_and_label_multiset():
    """The executed control destroys only terminal prefix-to-label assignments."""
    module = _recorder()
    positions = module._scored_positions(12)
    pool = module._example_pool(positions)
    pairs = draw_in_context_example_sets(
        positions,
        pool,
        k=16,
        example_seed=module.EXAMPLE_SEED,
        draw_attempts=module._DERANGEABLE_DRAW_ATTEMPTS,
        arm="contract",
    )
    assert pairs
    for treatment, deranged in pairs:
        assert [prefix for prefix, _symbol in treatment] == [prefix for prefix, _symbol in deranged]
        assert sorted(symbol for _prefix, symbol in treatment) == sorted(
            symbol for _prefix, symbol in deranged
        )
        assert all(
            real_symbol != control_symbol
            for (_prefix, real_symbol), (_same_prefix, control_symbol) in zip(
                treatment, deranged, strict=True
            )
        )

    treatment_prompts = build_in_context_synthetic_letter_prompts(
        positions,
        pool,
        k=16,
        deranged=False,
        example_seed=module.EXAMPLE_SEED,
        draw_attempts=module._DERANGEABLE_DRAW_ATTEMPTS,
        arm="treatment",
    )
    control_prompts = build_in_context_synthetic_letter_prompts(
        positions,
        pool,
        k=16,
        deranged=True,
        example_seed=module.EXAMPLE_SEED,
        draw_attempts=module._DERANGEABLE_DRAW_ATTEMPTS,
        arm="control",
    )
    for treatment_prompt, control_prompt in zip(treatment_prompts, control_prompts, strict=True):
        assert len(treatment_prompt) == len(control_prompt)
        assert _example_prefixes(treatment_prompt) == _example_prefixes(control_prompt)
        assert sorted(_example_symbols(treatment_prompt)) == sorted(
            _example_symbols(control_prompt)
        )


def test_canonical_draw_conditioning_diagnostics_are_complete_and_reproducible():
    """Every nonzero rung exposes the seeded rejection conditioning it incurred."""

    module = _recorder()
    positions = module._scored_positions(module.DEFAULT_POSITIONS)
    pool = module._example_pool(positions)
    recorded = module._draw_diagnostic_claim_values(positions, pool)
    expected_labels = {
        f"bench_ctx_draw_{stem}_k{k}"
        for k in module.K_GRID
        if k
        for stem in ("retries", "retried_positions", "max_attempts_used")
    }
    assert set(recorded) == expected_labels
    for k in module.K_GRID:
        if not k:
            continue
        _pairs, attempts = draw_in_context_example_sets_with_diagnostics(
            positions,
            pool,
            k=k,
            example_seed=module.EXAMPLE_SEED,
            draw_attempts=module._DERANGEABLE_DRAW_ATTEMPTS,
            arm=f"k{k}",
        )
        assert recorded[f"bench_ctx_draw_retries_k{k}"] == sum(attempt - 1 for attempt in attempts)
        assert recorded[f"bench_ctx_draw_retried_positions_k{k}"] == sum(
            attempt > 1 for attempt in attempts
        )
        assert recorded[f"bench_ctx_draw_max_attempts_used_k{k}"] == max(attempts)
    assert any(recorded[f"bench_ctx_draw_retries_k{k}"] > 0 for k in module.K_GRID if k), (
        "the canonical draw never exercises the rejection diagnostic"
    )


def test_example_draw_argument_controls_and_zero_arm_are_live():
    positions = [("ad", "a")]
    pool = [
        (letter, SYNTHETIC_LETTERS[(index + 1) % len(SYNTHETIC_LETTERS)])
        for index, letter in enumerate(SYNTHETIC_LETTERS)
    ]
    assert draw_in_context_example_sets(
        positions,
        pool,
        k=0,
        example_seed=7,
        draw_attempts=1,
        arm="k0",
    ) == [([], [])]
    with pytest.raises(ValueError, match="non-negative"):
        draw_in_context_example_sets(
            positions,
            pool,
            k=-1,
            example_seed=7,
            draw_attempts=1,
            arm="negative-k",
        )
    with pytest.raises(ValueError, match="draw_attempts must be positive"):
        draw_in_context_example_sets(
            positions,
            pool,
            k=1,
            example_seed=7,
            draw_attempts=0,
            arm="zero-attempts",
        )
    with pytest.raises(ValueError, match="cannot draw"):
        draw_in_context_example_sets(
            positions,
            pool,
            k=len(pool) + 1,
            example_seed=7,
            draw_attempts=1,
            arm="oversize",
        )


def test_fixed_terminal_mapping_positive_control_is_live_on_the_executed_population():
    """A real learner detects the mapping, while the planted no-op control cannot."""
    module = _recorder()
    positions = module._scored_positions(module.DEFAULT_POSITIONS)
    pool = module._example_pool(positions)
    treatment, deranged = module._mapping_control_flags(positions, pool)
    treatment_rate = sum(treatment) / len(treatment)
    deranged_rate = sum(deranged) / len(deranged)
    paired = paired_discordance(treatment, deranged)

    harness = ControlHarness("synthetic_terminal_mapping_test")
    harness.assert_gate_is_live(treatment_rate, deranged_rate, "higher_is_better")
    assert paired["b"] > paired["c"]
    assert paired["p_value"] < module.POWER_ALPHA
    with pytest.raises(GateVacuousError):
        harness.assert_gate_is_live(treatment_rate, treatment_rate, "higher_is_better")


def test_example_window_exposes_only_the_order_one_sufficient_statistic():
    protocol = SYNTHETIC_INCONTEXT_PROTOCOL
    assert protocol.example_window == 1
    assert protocol.example_internal_transitions_per_window == 0
    assert protocol.headline_internal_transitions == 0
    assert protocol.headline_total_transitions == protocol.headline_k


def _example_lines(prompt: str) -> list[str]:
    return [line for line in prompt.splitlines() if line.startswith("<") and " -> " in line]


def _example_prefixes(prompt: str) -> list[str]:
    return [line.split("> -> ")[0][1:] for line in _example_lines(prompt)]


def _example_symbols(prompt: str) -> list[str]:
    return [line.split("> -> ")[1] for line in _example_lines(prompt)]


def test_every_model_p_value_the_recorder_writes_is_in_its_declared_family():
    """The recorder cannot write a p-value that its own family does not cover.

    Checked against the assembled claim dictionary rather than against the
    template list alone, so a p-value added to the value assembly without a
    matching template fails here rather than at the ledger-wide scan much later.
    """
    module = _recorder()
    n = 40
    arms = {
        arm: [False] * n
        for arm in ["k0"]
        + [f"k{k}" for k in module.K_GRID if k]
        + [f"k{k}_deranged" for k in module.K_GRID if k]
    }
    values = module._claim_values("gemma3", arms, dict.fromkeys(arms, 0), [True] * n, n)
    written = {label for label in values if label.endswith("_paired_p_gemma3")}
    declared = {
        template.format(model="gemma3") for template in module.P_VALUE_FAMILIES["incontext"]
    }
    assert written == declared
    assert module.P_VALUE_FAMILIES["mapping_positive_control"] == (
        "bench_ctx_mapping_control_paired_p",
    )


def test_the_declared_family_alpha_follows_from_the_template_count():
    module = _recorder()
    expected_size = len(module.FAMILY_TEMPLATES) * len(module.MODELS)
    assert expected_size > len(module.FAMILY_TEMPLATES), (
        "the test needs multiple probes to catch a template-only family-size bug"
    )
    assert module.FAMILY_ALPHA / expected_size == pytest.approx(0.05 / expected_size)
    assert len(module.MAPPING_CONTROL_FAMILY_TEMPLATES) == 1


def test_the_headline_k_is_on_the_grid_and_declared_not_chosen():
    """The headline contrast is fixed before the run, not picked afterwards.

    Reporting whichever k came out strongest, with a p-value corrected only for
    the family it happens to sit in, is the selection defect this repository
    already corrected once in the channel weight sweep.
    """
    module = _recorder()
    assert module.HEADLINE_K in module.K_GRID
    assert module.HEADLINE_K == max(module.K_GRID)
    assert 0 in module.K_GRID, "the grid must carry its own zero-shot anchor"


def test_rotating_schedule_places_every_arm_in_every_temporal_slot() -> None:
    """One full cycle is a Latin rotation over the complete arm registry."""
    module = _recorder()
    names = tuple(name for name, _k, _deranged in module._arm_specs())
    assert len(names) > 1
    slot_counts = {name: [0] * len(names) for name in names}
    for position_index in range(len(names)):
        rotated = module._rotated_arm_names(position_index)
        assert set(rotated) == set(names)
        assert len(rotated) == len(set(rotated))
        for slot, name in enumerate(rotated):
            slot_counts[name][slot] += 1
    assert all(counts == [1] * len(names) for counts in slot_counts.values())
    with pytest.raises(ValueError, match="non-negative"):
        module._rotated_arm_names(-1)

    # NEGATIVE CONTROL: a fixed order leaves every arm trapped in one slot.
    planted = {name: [0] * len(names) for name in names}
    for _position_index in range(len(names)):
        for slot, name in enumerate(names):
            planted[name][slot] += 1
    assert not all(counts == [1] * len(names) for counts in planted.values())


def _valid_incontext_measurement(module, positions_count: int = 3) -> tuple[dict[str, object], int]:
    positions = module._scored_positions(positions_count)
    pool = module._example_pool(positions)
    trace = module._expected_execution_trace(positions, pool)
    expected_calls = len(trace)
    measurement = module._measurement_evidence(
        execution_trace=trace,
        cache_hits=0,
        cache_misses=expected_calls,
        unique_cache_keys=expected_calls,
        maximum_prompt_tokens=1,
    )
    return measurement, expected_calls


def _reduced_semantic_replay_fixture(
    module,
    cache_path: Path,
    *,
    positions_count: int,
    mutate_first_reply: bool = False,
) -> tuple[dict[str, object], CallCache, list[str]]:
    """Materialize a real offline CallCache for the exact rotating schedule."""

    endpoint = "http://127.0.0.1:11434"
    positions = module._scored_positions(positions_count)
    pool = module._example_pool(positions)
    prompts = module._all_arm_prompts(positions, pool)
    # One hosted transport identity: every scored call resolves the same
    # OpenRouter model, so the receipt binds exactly one identity row.
    identities = {module.OPENROUTER_MODEL: "sha256:" + "0" * 64}
    store = CallCache(cache_path)
    keys: set[str] = set()
    first_request = True
    for _name, model in sorted(module.MODELS.items()):
        for position_index, (_prefix, gold) in enumerate(positions):
            for arm in module._rotated_arm_names(position_index):
                payload = module._chat_payload(
                    module.OPENROUTER_MODEL,
                    prompts[arm][position_index],
                    "synthetic_letter",
                    SYNTHETIC_INCONTEXT_PROTOCOL.response_tokens,
                    module.DEFAULT_SEED,
                )
                request = CacheRequest(
                    payload=payload,
                    endpoint=endpoint,
                    protocol=module._openrouter_cache_protocol(
                        module._call_cache_protocol(model, position_index, arm, RUN_SCOPE)
                    ),
                    identity=identities[module.OPENROUTER_MODEL],
                )
                letter = (
                    gold
                    if mutate_first_reply and first_request
                    else SYNTHETIC_INCONTEXT_PROTOCOL.response_inventory[0]
                )
                store.put(
                    request,
                    {
                        "response": json.dumps({"answer": letter}, separators=(",", ":")),
                        "prompt_eval_count": 7,
                    },
                )
                keys.add(request.key)
                first_request = False

    ordered_keys = sorted(keys)
    configuration = module._replay_configuration(
        positions=positions_count,
        run_scope=RUN_SCOPE,
        endpoint=endpoint,
    )
    receipt: dict[str, object] = {
        "configuration": configuration,
        "cache": {
            "record_count": len(ordered_keys),
            "keys": ordered_keys,
            "model_identities": [
                {"model": model, "identity": identity}
                for model, identity in sorted(identities.items())
            ],
        },
        "claims": {"entries": {}},
        "measurement": module._measurement_evidence(
            execution_trace=module._expected_execution_trace(positions, pool),
            cache_hits=0,
            cache_misses=len(ordered_keys),
            unique_cache_keys=len(ordered_keys),
            maximum_prompt_tokens=7,
        ),
    }
    derived = module._replay_result_claims(
        receipt,
        store,
        ordered_keys,
        positions_count=positions_count,
    )
    receipt["claims"] = {
        "entries": {
            label: {"value": value, "run_id": module.RUN_ID, "source": module.SOURCE}
            for label, value in derived.items()
        }
    }
    return receipt, store, ordered_keys


def test_v2_incontext_measurement_validator_reconstructs_the_rotating_trace() -> None:
    module = _recorder()
    measurement, expected_calls = _valid_incontext_measurement(module)
    module._validate_measurement_fields(
        measurement,
        positions=3,
        cache_record_count=expected_calls,
    )


def test_v2_incontext_measurement_validator_rejects_load_bearing_mutations(
    tmp_path: Path,
) -> None:
    module = _recorder()
    measurement, expected_calls = _valid_incontext_measurement(module)
    ledger = tmp_path / "claims.json"
    ledger.write_text('{"_schema":"ailm-claims-v2","claims":{}}\n', encoding="utf-8")
    before = ledger.read_bytes()
    mutations: tuple[tuple[str, object], ...] = (
        ("schema", "ailm-incontext-execution-trace-v1"),
        ("schedule", "fixed-arm-order"),
        ("warmup_policy", "none"),
        ("warmup_calls", 1),
        ("warmup_calls", False),
        ("scored_calls", expected_calls - 1),
        ("cache_hits", 1),
        ("cache_misses", expected_calls - 1),
        ("unique_cache_keys", 0),
        ("maximum_prompt_tokens", 0),
        ("ordered_trace_sha256", "f" * 64),
    )
    for field, replacement in mutations:
        changed = dict(measurement)
        changed[field] = replacement
        with pytest.raises(NetworkClaimBindingError):
            module._validate_measurement_fields(
                changed,
                positions=3,
                cache_record_count=expected_calls,
            )
    assert ledger.read_bytes() == before, "pre-publication refusal changed the ledger"


def test_reduced_real_cache_semantic_replay_accepts_and_rejects_all_mutations(
    tmp_path: Path,
) -> None:
    """Full claim replay is live on a bounded real cache, with three controls."""

    module = _recorder()
    positions_count = 2
    receipt, store, keys = _reduced_semantic_replay_fixture(
        module,
        tmp_path / "baseline.jsonl",
        positions_count=positions_count,
    )
    module._validate_semantic_replay(
        receipt,
        store,
        keys,
        positions_count=positions_count,
    )

    changed_claim = json.loads(json.dumps(receipt))
    claim_entries = changed_claim["claims"]["entries"]
    assert isinstance(claim_entries, dict)
    label = f"bench_ctx_top1_k0_{next(iter(sorted(module.MODELS)))}"
    claim_entries[label]["value"] = 1.0
    with pytest.raises(NetworkClaimBindingError, match="do not rederive"):
        module._validate_semantic_replay(
            changed_claim,
            store,
            keys,
            positions_count=positions_count,
        )

    _changed_receipt, changed_store, changed_keys = _reduced_semantic_replay_fixture(
        module,
        tmp_path / "changed-reply.jsonl",
        positions_count=positions_count,
        mutate_first_reply=True,
    )
    assert changed_keys == keys, "reply content must not change the request-key set"
    with pytest.raises(NetworkClaimBindingError, match="do not rederive"):
        module._validate_semantic_replay(
            receipt,
            changed_store,
            keys,
            positions_count=positions_count,
        )

    with pytest.raises(NetworkClaimBindingError, match="lacks exact|exact scored request set"):
        module._validate_semantic_replay(
            receipt,
            store,
            keys[1:],
            positions_count=positions_count,
        )

    # The reduced seam is internal only. The publication validator still pins
    # the canonical population before semantic replay begins.
    with pytest.raises(NetworkClaimBindingError):
        module.validate_publication_measurement(receipt, store)


def _producer_contract_entries(module) -> dict[str, dict[str, object]]:
    protocol = SYNTHETIC_ZERO_SHOT_PROTOCOL
    values: dict[str, object] = {
        "bench_synth_protocol_version": protocol.version,
        "bench_synth_chain_version": protocol.chain_version,
        "bench_synth_query_window": protocol.query_window,
        "bench_synth_query_context_transitions_max": (protocol.query_context_transitions_max),
        "bench_synth_response_inventory_size": len(protocol.response_inventory),
        "bench_synth_alphabet_size": len(protocol.response_inventory),
        "bench_synth_position_generation": protocol.position_generation,
        "bench_synth_position_seed": protocol.position_seed,
        "bench_synth_trajectory_length": protocol.query_window + 1,
        "bench_synth_target_start_tvd": independent_target_start_tvd(protocol.query_window),
        "bench_synth_corpus_chars": DEFAULT_CORPUS_CHARS,
        "bench_synth_corpus_seed": DEFAULT_CORPUS_SEED,
        "bench_synth_context_window": protocol.context_window,
        "bench_synth_response_tokens": protocol.response_tokens,
        "bench_synth_prompt_token_overhead_margin": (protocol.prompt_token_overhead_margin),
        "bench_synth_positions": 7,
        "bench_synth_letter_top1_gemma3": 0.5,
    }
    return {
        label: {
            "value": value,
            "run_id": module.PRODUCER_RUN_ID,
            "source": module.PRODUCER_SOURCE,
        }
        for label, value in values.items()
    }


def test_consumer_refuses_protocol_source_and_run_drift_before_any_network_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_http_server
) -> None:
    """Each producer-authentication axis fails offline and preserves ledger bytes."""
    module = _recorder()
    cases = {
        "protocol": (
            "bench_synth_protocol_version",
            "value",
            "synthetic-zero-shot-retired",
            "protocol contract disagrees",
        ),
        "source": (
            "bench_synth_protocol_version",
            "source",
            "scripts/record_other_claims.py",
            "is not bound",
        ),
        "run": (
            "bench_synth_protocol_version",
            "run_id",
            "synthetic_letter_probe_retired",
            "is not bound",
        ),
    }
    for name, (label, field, replacement, message) in cases.items():
        entries = _producer_contract_entries(module)
        entries[label][field] = replacement
        ledger = tmp_path / f"{name}-claims.json"
        ledger.write_text(
            json.dumps(
                {
                    "_schema": "ailm-claims-v2",
                    "_description": "isolated producer contract",
                    "claims": entries,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        before = ledger.read_bytes()
        monkeypatch.setenv("AILM_CLAIMS_PATH", str(ledger.resolve()))
        with pytest.raises(SystemExit, match=message):
            module._check_zero_shot_reproduces(
                "never-contact-this-model",
                "gemma3",
                producer_scope=RUN_SCOPE,
                endpoint=local_http_server.base_url,
            )
        assert ledger.read_bytes() == before
    assert len(local_http_server.requests_for("/api/generate")) == 0


def test_the_recorder_does_not_write_into_another_recorders_namespace():
    """Namespace ownership, which here is enforced by prefix rather than source.

    ``record_synthetic_benchmark_claims.py`` deletes every ``bench_synth_``
    label it did not itself produce. A label written under that prefix by this
    recorder would be silently deleted on the next zero-shot run, and the
    manuscript token would go missing with no test failing at the moment of
    loss.
    """
    module = _recorder()
    n = 8
    arms = {
        arm: [False] * n
        for arm in ["k0"]
        + [f"k{k}" for k in module.K_GRID if k]
        + [f"k{k}_deranged" for k in module.K_GRID if k]
    }
    values = module._claim_values("gemma3", arms, dict.fromkeys(arms, 0), [True] * n, n)
    assert values, "the recorder produced no claims"
    assert all(label.startswith("bench_ctx_") for label in values)
    assert not any(label.startswith("bench_synth_") for label in values)


def test_the_declared_family_counts_every_probe_not_just_every_template():
    """The family is templates times probes, and the alpha follows.

    Registering a second probe doubles the number of tests this recorder
    declares. If the recorded size stayed at the template count, the supplement
    would read every "clears correction" sentence against a threshold twice as
    loose as the one the family actually requires — a silent halving that no
    other test in the repository would catch, because the digit in the prose is
    a token and would move with the wrong value.
    """
    module = _recorder()
    n = 24
    arms = {
        arm: [False] * n
        for arm in ["k0"]
        + [f"k{k}" for k in module.K_GRID if k]
        + [f"k{k}_deranged" for k in module.K_GRID if k]
    }
    values = module._claim_values("gemma3", arms, dict.fromkeys(arms, 0), [True] * n, n)
    expected = len(module.FAMILY_TEMPLATES) * len(module.MODELS)
    assert values["bench_ctx_family_size"] == expected
    assert values["bench_ctx_family_alpha"] == pytest.approx(module.FAMILY_ALPHA / expected)


def test_the_recorder_offers_no_single_model_flag():
    """A one-probe run would delete the other probe's arms.

    The stale sweep clears every ``bench_ctx_`` label the run did not produce,
    so scoring one model and writing would retire the other's. The guarantee is
    structural: there is no flag that can ask for it.
    """
    source = RECORDER.read_text(encoding="utf-8")
    assert '"--model"' not in source, "a --model flag would allow a partial, self-deleting run"
    assert "for name, tag in sorted(MODELS.items())" in source, (
        "every registered hosted cohort must be scheduled"
    )
