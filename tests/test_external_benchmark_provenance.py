"""Offline semantic replay controls for the external reference benchmark."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from ailm.benchmarks import PROTOCOL_VERSION, benchmark_model
from ailm.benchmarks.constants import repeat_trial_protocols
from ailm.call_cache import CacheRequest, CallCache
from ailm.replay_provenance import NetworkClaimBindingError, maximum_prompt_eval_tokens

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORDER_PATH = PROJECT_ROOT / "scripts" / "record_external_benchmark_claims.py"
IDENTITY = "sha256:" + "a" * 64
#: One fixture run's repeat-trial scope. Shaped like a recorder-generated run id
#: because the receipt validator refuses any other shape.
RUN_SCOPE = "20260905T000000000001Z"
REPEAT_TRIALS = repeat_trial_protocols(RUN_SCOPE)


def _recorder() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "_external_benchmark_semantic_recorder", RECORDER_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load external benchmark recorder")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _reply(answer: str, *, logprobs: bool, prompt_tokens: int) -> dict[str, object]:
    body: dict[str, object] = {
        "response": json.dumps({"answer": answer}, separators=(",", ":")),
        "prompt_eval_count": prompt_tokens,
    }
    if logprobs:
        body["logprobs"] = [
            {"token": '{"answer":"', "top_logprobs": []},
            {
                "token": answer,
                "top_logprobs": [
                    {"token": answer, "logprob": -0.1},
                    {"token": "b" if answer != "b" else "a", "logprob": -2.0},
                ],
            },
        ]
    return body


def _fixture(
    path: Path,
    *,
    second_repeat_answer: str = "a",
    run_scope: str = RUN_SCOPE,
) -> tuple[ModuleType, dict[str, object], CallCache]:
    recorder = _recorder()
    trials = repeat_trial_protocols(run_scope)
    selected = ["gemma3"]
    model = recorder.MODELS[selected[0]]
    store = CallCache(path)
    keys: set[str] = set()
    logical_calls = 0

    def acquire(payload: dict[str, object], cache_protocol: str) -> dict[str, object]:
        nonlocal logical_calls
        logical_calls += 1
        answer = "b" if cache_protocol == trials[1] and second_repeat_answer == "b" else "a"
        body = _reply(
            answer,
            logprobs=bool(payload.get("logprobs")),
            prompt_tokens=max(1, len(str(payload.get("prompt", ""))) // 8),
        )
        request = CacheRequest(
            payload=payload,
            endpoint=recorder.OLLAMA_URL,
            protocol=cache_protocol,
            identity=IDENTITY,
        )
        store.put(request, body)
        keys.add(request.key)
        return body

    counts = {
        "letter_n": 3,
        "phoneme_n": 2,
        "g2p_n": 2,
        "calibration_n": 2,
        "ensemble_n": 2,
        "repeat_n": 2,
    }
    results = {
        selected[0]: benchmark_model(
            model,
            repeat_trial_protocols=trials,
            **counts,
            include_prompt_sensitivity=True,
            response_lookup=acquire,
        )
    }
    configuration = recorder._replay_configuration(selected=selected, run_scope=run_scope, **counts)
    cache = {**store.provenance_binding(keys), "record_count": len(keys)}
    values = recorder._claim_values(
        selected=selected,
        results=results,
        phoneme_n=counts["phoneme_n"],
        g2p_n=counts["g2p_n"],
    )
    measurement = recorder._measurement_evidence(
        cache_hits=0,
        cache_misses=logical_calls,
        unique_cache_keys=len(keys),
        logical_calls=logical_calls,
        maximum_prompt_tokens=maximum_prompt_eval_tokens(store, keys),
        cache_protocols=recorder._cache_protocols(run_scope),
    )
    receipt: dict[str, object] = {
        "configuration": configuration,
        "cache": cache,
        "measurement": measurement,
        "claims": {
            "entries": {
                label: {
                    "value": value,
                    "run_id": "fixture",
                    "source": recorder.SOURCE,
                }
                for label, value in values.items()
            }
        },
    }
    return recorder, receipt, store


@pytest.fixture(scope="module")
def semantic_fixture(tmp_path_factory):
    return _fixture(tmp_path_factory.mktemp("external-semantic") / "calls.jsonl")


def test_external_receipt_rederives_every_claim_from_exact_cached_replies(
    semantic_fixture,
) -> None:
    recorder, receipt, store = semantic_fixture
    recorder.validate_publication_measurement(receipt, store)
    entries = receipt["claims"]["entries"]
    assert set(entries) == set(recorder._rederive_claim_values(receipt, store)[0])
    assert not {"bench_cache_hits", "bench_cache_misses", "bench_cache_fresh_fraction"} & set(
        entries
    )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda receipt: receipt["claims"]["entries"]["bench_letter_top1_gemma3"].update(
            {"value": 0.123456789}
        ),
        lambda receipt: receipt["claims"]["entries"].pop("bench_letter_top1_gemma3"),
        lambda receipt: receipt["claims"]["entries"].update(
            {"bench_arbitrary": {"value": 1, "run_id": "fixture", "source": "fixture"}}
        ),
        lambda receipt: receipt.update({"measurement": None}),
    ),
)
def test_arbitrary_missing_extra_claims_and_absent_measurement_fail_closed(
    semantic_fixture, mutate
) -> None:
    recorder, original, store = semantic_fixture
    receipt = copy.deepcopy(original)
    mutate(receipt)
    with pytest.raises(NetworkClaimBindingError, match="measurement|claims do not rederive"):
        recorder.validate_publication_measurement(receipt, store)


def test_mutated_second_repeat_reply_changes_the_rederived_reliability_claim(
    semantic_fixture, tmp_path: Path
) -> None:
    recorder, original, _store = semantic_fixture
    _changed_recorder, changed, changed_store = _fixture(
        tmp_path / "changed-repeat.jsonl", second_repeat_answer="b"
    )
    changed["claims"] = copy.deepcopy(original["claims"])
    with pytest.raises(NetworkClaimBindingError, match="claims do not rederive"):
        recorder.validate_publication_measurement(changed, changed_store)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda receipt: receipt["configuration"]["parameters"].update({"letter_n": 4}),
        lambda receipt: receipt["configuration"]["parameters"].update(
            {"cache_protocols": [PROTOCOL_VERSION]}
        ),
        lambda receipt: receipt["cache"]["model_identities"][0].update(
            {"identity": "sha256:" + "b" * 64}
        ),
        lambda receipt: receipt["cache"].update({"keys": receipt["cache"]["keys"][:-1]}),
    ),
)
def test_protocol_configuration_identity_and_key_drift_fail_closed(
    semantic_fixture, mutate
) -> None:
    recorder, original, store = semantic_fixture
    receipt = copy.deepcopy(original)
    mutate(receipt)
    with pytest.raises(NetworkClaimBindingError):
        recorder.validate_publication_measurement(receipt, store)


def test_extra_authenticated_cache_record_is_not_silently_ignored(
    semantic_fixture, tmp_path: Path
) -> None:
    recorder, original, original_store = semantic_fixture
    store = CallCache(tmp_path / "extra.jsonl")
    keys: set[str] = set()
    for key, payload, reply in original_store.entries():
        binding = original_store.provenance_binding({key})
        request = CacheRequest(
            payload=payload,
            endpoint=binding["endpoints"][0],
            protocol=binding["protocols"][0],
            identity=binding["identities"][0],
        )
        store.put(request, reply)
        keys.add(request.key)
    extra = CacheRequest(
        payload={"model": recorder.MODELS["gemma3"], "prompt": "not in the protocol"},
        endpoint=recorder.OLLAMA_URL,
        protocol=PROTOCOL_VERSION,
        identity=IDENTITY,
    )
    store.put(extra, {"response": '{"answer":"a"}', "prompt_eval_count": 2})
    keys.add(extra.key)
    receipt = copy.deepcopy(original)
    receipt["cache"] = {**store.provenance_binding(keys), "record_count": len(keys)}
    receipt["measurement"]["unique_cache_keys"] = len(keys)
    receipt["measurement"]["maximum_prompt_tokens"] = maximum_prompt_eval_tokens(store, keys)
    with pytest.raises(NetworkClaimBindingError, match="exact protocol-derived request set"):
        recorder.validate_publication_measurement(receipt, store)


def test_receipt_declaring_another_runs_repeat_tags_cannot_claim_these_records(
    semantic_fixture,
) -> None:
    """The run a receipt declares must be the run its cited records were made under.

    Run scoping is what lets a re-run append instead of colliding, so the tags
    became a field a receipt states rather than a constant the validator knows.
    That field is only load-bearing if a receipt cannot state one run while
    citing another's records: the declared scope determines the keys the semantic
    replay reconstructs, and those keys must be exactly the ones in the receipt.
    """

    recorder, original, store = semantic_fixture
    receipt = copy.deepcopy(original)
    other_scope = "20260905T000000000002Z"
    assert other_scope != RUN_SCOPE
    parameters = receipt["configuration"]["parameters"]
    parameters["run_scope"] = other_scope
    parameters["cache_protocols"] = list(recorder._cache_protocols(other_scope))
    receipt["measurement"]["cache_protocols"] = list(recorder._cache_protocols(other_scope))
    with pytest.raises(NetworkClaimBindingError, match="protocol-derived request"):
        recorder.validate_publication_measurement(receipt, store)


@pytest.mark.parametrize(
    "scope",
    ("", "warm-key-planning", "20260905T0000Z", REPEAT_TRIALS[0], RUN_SCOPE.lower()),
)
def test_only_a_recorder_generated_run_scope_is_publishable(semantic_fixture, scope) -> None:
    """A scope no run could have generated is refused before anything replays.

    The generated shape is a UTC instant to microsecond resolution. Anything
    else — the warm planner's placeholder, a truncated stamp, a whole protocol
    tag pasted in — would let a receipt name a namespace by hand.
    """

    recorder, original, store = semantic_fixture
    receipt = copy.deepcopy(original)
    receipt["configuration"]["parameters"]["run_scope"] = scope
    with pytest.raises(NetworkClaimBindingError, match="run scope"):
        recorder.validate_publication_measurement(receipt, store)


def test_cache_audit_sampling_policy_is_bound_to_the_current_serving_configuration(
    semantic_fixture,
):
    recorder, original, store = semantic_fixture
    recorder.validate_publication_measurement(original, store)
    altered = copy.deepcopy(original)
    altered["configuration"]["parameters"]["cache_audit_sampling"] = "all-historical-records"
    with pytest.raises(NetworkClaimBindingError, match="cache audit sampling policy"):
        recorder.validate_publication_measurement(altered, store)


@pytest.mark.parametrize(
    "field,value",
    [
        ("local_context_window", 131072),
        ("local_context_window", 8192.0),
        ("context_budget_policy", "unbounded"),
    ],
)
def test_context_choice_and_capacity_policy_are_bound_to_publication_receipt(
    semantic_fixture, field, value
):
    recorder, original, store = semantic_fixture
    altered = copy.deepcopy(original)
    altered["configuration"]["parameters"][field] = value
    with pytest.raises(NetworkClaimBindingError, match="context"):
        recorder.validate_publication_measurement(altered, store)
