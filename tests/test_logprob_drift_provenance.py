"""Authenticated selection and paired-response controls for logprob drift."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from ailm.call_cache import CacheRequest, CallCache
from ailm.replay_provenance import NetworkClaimBindingError
from scripts import record_logprob_drift_claims as drift


def _reply(*, answer: str = "a", answer_logprob: float = -0.2) -> dict:
    return {
        "response": answer,
        "logprobs": [
            {"token": '{"answer":"', "top_logprobs": []},
            {
                "token": answer,
                "top_logprobs": [
                    {"token": "a", "logprob": answer_logprob},
                    {"token": "b", "logprob": -1.7},
                ],
            },
        ],
    }


def _store(tmp_path: Path) -> tuple[CallCache, list[tuple[str, dict, dict]]]:
    store = CallCache(tmp_path / "calls.jsonl")
    for model in sorted(drift.PROBE_MODEL_TAGS.values()):
        identity = "sha256:" + hashlib.sha256(model.encode("utf-8")).hexdigest()
        for index in range(drift.SAMPLE // len(drift.PROBE_MODEL_TAGS)):
            request = CacheRequest(
                payload={
                    "model": model,
                    "prompt": f"position {index}",
                    "logprobs": True,
                    "stream": False,
                    "options": {"use_mmap": False, "num_ctx": 8192},
                },
                endpoint=drift.OLLAMA_URL,
                protocol=drift.PROTOCOL_VERSION,
                identity=identity,
            )
            store.put(request, _reply(answer_logprob=-0.2 - index / 100))
    pool = [entry for entry in store.entries() if entry[1].get("logprobs")]
    assert len(pool) == drift.SAMPLE
    return store, pool


def _producer(store: CallCache, keys: list[str] | None = None) -> dict:
    selected = frozenset(keys if keys is not None else [entry[0] for entry in store.entries()])
    body = {"cache": store.provenance_binding(selected)}
    return {**body, "receipt_sha256": drift._digest(body)}


def _receipt(store: CallCache, pool: list[tuple[str, dict, dict]], producer: dict) -> dict:
    sample, universe_keys, universe_digest = drift._sample_entries(store, pool)
    observations = [
        {
            "key": key,
            "stored": stored,
            "fresh": _reply(answer_logprob=-0.25 - index / 100),
        }
        for index, (key, _payload, stored) in enumerate(sample)
    ]
    values = drift._claim_values(observations)
    return {
        "measurement": drift._measurement(
            universe_keys=universe_keys,
            universe_digest=universe_digest,
            observations=observations,
            producer_receipt_sha256=producer["receipt_sha256"],
        ),
        "claims": {"entries": {label: {"value": value} for label, value in values.items()}},
    }


def test_logprob_measurement_reconstructs_selection_and_claims(tmp_path: Path) -> None:
    store, pool = _store(tmp_path)
    producer = _producer(store)
    receipt = _receipt(store, pool, producer)
    drift._validate_measurement_against_producer(receipt, store, producer)


def test_eligible_universe_growth_invalidates_the_bound_sample(tmp_path: Path) -> None:
    """NEGATIVE: an unselected eligible row cannot leave an old receipt valid."""
    store, pool = _store(tmp_path)
    producer = _producer(store)
    receipt = _receipt(store, pool, producer)
    model = sorted(drift.PROBE_MODEL_TAGS.values())[0]
    request = CacheRequest(
        payload={
            "model": model,
            "prompt": "new eligible row",
            "logprobs": True,
            "options": {"use_mmap": False, "num_ctx": 8192},
        },
        endpoint=drift.OLLAMA_URL,
        protocol=drift.PROTOCOL_VERSION,
        identity="sha256:" + hashlib.sha256(model.encode("utf-8")).hexdigest(),
    )
    store.put(request, _reply())

    # An unrelated cohort is not allowed to grow the current producer's population.
    drift._validate_measurement_against_producer(receipt, store, producer)
    with pytest.raises(NetworkClaimBindingError, match="producer receipt is stale"):
        drift._validate_measurement_against_producer(receipt, store, _producer(store))


def test_fresh_response_mutation_cannot_reproduce_bound_claims(tmp_path: Path) -> None:
    """NEGATIVE: even a re-digested fresh-side edit must invalidate the claims."""
    store, pool = _store(tmp_path)
    producer = _producer(store)
    receipt = copy.deepcopy(_receipt(store, pool, producer))
    observations = receipt["measurement"]["observations"]
    observations[0]["fresh"]["logprobs"][1]["top_logprobs"][0]["logprob"] = -9.0
    receipt["measurement"]["observations_sha256"] = drift._digest(observations)

    with pytest.raises(NetworkClaimBindingError, match="do not reproduce"):
        drift._validate_measurement_against_producer(receipt, store, producer)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"use_mmap": True},
        {"use_mmap": 0},
        {"use_mmap": False},
        {"use_mmap": False, "num_ctx": 4096},
    ],
)
def test_producer_referenced_incompatible_loader_request_is_refused(tmp_path, options):
    store, _ = _store(tmp_path)
    model = sorted(drift.PROBE_MODEL_TAGS.values())[0]
    store.put(
        CacheRequest(
            payload={
                "model": model,
                "prompt": "wrong loader",
                "logprobs": True,
                "options": options,
            },
            endpoint=drift.OLLAMA_URL,
            protocol=drift.PROTOCOL_VERSION,
            identity="sha256:" + hashlib.sha256(model.encode()).hexdigest(),
        ),
        _reply(),
    )
    with pytest.raises(NetworkClaimBindingError, match="fixed loader option"):
        drift._producer_pool(store, _producer(store))


def test_missing_or_insufficient_producer_population_fails_closed(tmp_path):
    store, pool = _store(tmp_path)
    with pytest.raises(NetworkClaimBindingError, match="current external-producer receipt"):
        drift._load_producer_receipt(store, project_root=tmp_path)
    with pytest.raises(NetworkClaimBindingError, match="empty or invalid"):
        drift._producer_pool(store, {"cache": {"keys": []}})
    with pytest.raises(NetworkClaimBindingError, match="missing"):
        drift._producer_pool(store, {"cache": {"keys": ["a" * 64]}})
    remaining = [entry[0] for entry in pool[:-1]]
    limited = drift._producer_pool(store, _producer(store, remaining))
    with pytest.raises(SystemExit, match="fewer than the registered"):
        drift._sample_entries(store, limited)


def test_unrelated_older_rows_do_not_enter_the_current_producer_sample(tmp_path):
    store, pool = _store(tmp_path)
    producer = _producer(store)
    expected = drift._sample_entries(store, pool)
    model = sorted(drift.PROBE_MODEL_TAGS.values())[0]
    store.put(
        CacheRequest(
            payload={"model": model, "prompt": "historical logprobs", "logprobs": True},
            endpoint=drift.OLLAMA_URL,
            protocol=drift.PROTOCOL_VERSION,
            identity="sha256:" + hashlib.sha256(model.encode()).hexdigest(),
        ),
        _reply(),
    )
    actual = drift._sample_entries(store, drift._producer_pool(store, producer))
    assert actual == expected


def test_producer_cannot_rebind_old_endpoint_records_to_current_selection(tmp_path):
    store, _pool = _store(tmp_path)
    other = CallCache(tmp_path / "other-endpoint.jsonl")
    for _key, payload, reply in store.entries():
        other.put(
            CacheRequest(
                payload=payload,
                endpoint="http://127.0.0.1:15434"
                if drift.OLLAMA_URL != "http://127.0.0.1:15434"
                else "http://localhost:11434",
                protocol=drift.PROTOCOL_VERSION,
                identity="sha256:" + hashlib.sha256(payload["model"].encode()).hexdigest(),
            ),
            reply,
        )
    producer = _producer(other)
    with pytest.raises(NetworkClaimBindingError, match="endpoint differs"):
        drift._producer_pool(other, producer)
    producer["cache"]["endpoints"] = [drift.OLLAMA_URL]
    with pytest.raises(NetworkClaimBindingError, match="endpoint differs"):
        drift._producer_pool(other, producer)
