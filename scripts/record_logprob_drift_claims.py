"""Measure how far a replayed answer-slot distribution sits from a fresh one.

Greedy decoding at temperature zero reproduces the ANSWER exactly. It does not
reproduce the answer-slot logprobs exactly, and the supplement previously ran
those two claims together. Every accuracy number in the external reference frame
rests on the first; every calibration number rests on the second, so the size of
the drift is a floor on the precision of ECE, Brier and captured mass.

This recorder re-queries stored calls that requested logprobs and reports the
drift between what was recorded and what the server says now. It measures rather
than asserts: the tolerance used by the cache audit is justified by these numbers
and would be a fudge factor without them.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ailm.benchmarks import (  # noqa: E402
    OLLAMA_URL,
    PROTOCOL_VERSION,
    _answer_token_index,
    _distribution_from_candidates,
    captured_mass,
    enable_call_cache,
    post_generate_uncached,
    resolved_model_identity,
)
from ailm.benchmarks.constants import (  # noqa: E402
    OLLAMA_CONTEXT_BUDGET_POLICY,
    OLLAMA_NUM_CTX,
    OLLAMA_USE_MMAP,
    PROBE_MODEL_TAGS,
)
from ailm.benchmarks.ollama import uses_current_ollama_loader  # noqa: E402
from ailm.call_cache import CallCache, model_digest  # noqa: E402
from ailm.claims import ClaimBatch, load_ledger  # noqa: E402
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402
from ailm.replay_provenance import (  # noqa: E402
    NetworkClaimBindingError,
    declared_run_scope,
    network_recorder_configuration,
    source_manifest,
    using_publication_ledger,
    validate_network_claim_binding,
    validate_network_measurement_inputs,
    write_network_claim_binding,
)

SOURCE = "scripts/record_logprob_drift_claims.py"
RUN_ID = "logprob-drift"
SAMPLE = 24
SEED = 20260812
MEASUREMENT_SCHEMA = "ailm-logprob-drift-measurement-v2"
PRODUCER_SOURCE = "scripts/record_external_benchmark_claims.py"


def publication_replay_configuration() -> dict[str, object]:
    """Canonical configuration required by the full release replay contract."""

    return network_recorder_configuration(
        endpoint=OLLAMA_URL,
        call_protocol=PROTOCOL_VERSION,
        model_tags=list(PROBE_MODEL_TAGS.values()),
        parameters={
            "sample": SAMPLE,
            "seed": SEED,
            "sampling": "model-stratified-current-endpoint-producer-universe-digest-seeded-v5",
            "measurement": "authenticated-fresh-minus-stored-answer-slot-pairs-v2",
            "producer_recorder": PRODUCER_SOURCE,
            "loader_use_mmap": OLLAMA_USE_MMAP,
            "local_context_window": OLLAMA_NUM_CTX,
            "context_budget_policy": OLLAMA_CONTEXT_BUDGET_POLICY,
        },
    )


def _load_producer_receipt(store: CallCache, *, project_root: Path = PROJECT_ROOT) -> dict:
    """Authenticate current producer source, configuration, claims and exact scored requests."""
    scope = declared_run_scope(project_root, PRODUCER_SOURCE)
    if scope is None:
        raise NetworkClaimBindingError("logprob drift requires a current external-producer receipt")
    specification = importlib.util.spec_from_file_location(
        "_logprob_drift_external_producer", project_root / PRODUCER_SOURCE
    )
    if specification is None or specification.loader is None:
        raise NetworkClaimBindingError("cannot load the external-producer contract")
    producer = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(producer)
    receipt = validate_network_claim_binding(
        project_root=project_root,
        recorder=PRODUCER_SOURCE,
        expected_configuration=producer.publication_replay_configuration(run_scope=scope),
        cache_path=store.path,
    )
    producer.validate_publication_measurement(receipt, store)
    return receipt


def _producer_pool(store: CallCache, producer: Mapping[str, Any]) -> list[tuple[str, dict, dict]]:
    """Select only authenticated producer keys; incompatible serving choices are errors."""
    cache = producer.get("cache")
    if not isinstance(cache, dict) or not isinstance(cache.get("keys"), list):
        raise NetworkClaimBindingError("external producer has no bound call keys")
    keys = cache["keys"]
    if not keys or any(not isinstance(key, str) for key in keys) or len(set(keys)) != len(keys):
        raise NetworkClaimBindingError("external producer call keys are empty or invalid")
    available = {key: (key, payload, reply) for key, payload, reply in store.entries()}
    if not set(keys) <= set(available):
        raise NetworkClaimBindingError("external producer call records are missing")
    binding = store.provenance_binding(frozenset(keys))
    if cache.get("endpoints") != [OLLAMA_URL] or binding["endpoints"] != [OLLAMA_URL]:
        raise NetworkClaimBindingError(
            "external producer endpoint differs from selected configuration"
        )
    pool = []
    for key in keys:
        entry = available[key]
        payload = entry[1]
        if payload.get("logprobs") is not True:
            continue
        if not uses_current_ollama_loader(payload):
            raise NetworkClaimBindingError(
                "external producer logprob request lacks the fixed loader option"
            )
        if payload.get("model") not in PROBE_MODEL_TAGS.values():
            raise NetworkClaimBindingError(
                "external producer logprob request names an undeclared model"
            )
        pool.append(entry)
    return pool


def _sample_entries(
    store: CallCache,
    pool: list[tuple[str, dict, dict]],
) -> tuple[list[tuple[str, dict, dict]], list[str], str]:
    """Draw a reproducible, cache-content-bound sample from every model."""

    tags = sorted(PROBE_MODEL_TAGS.values())
    quotient, remainder = divmod(SAMPLE, len(tags))
    universe_keys = sorted(key for key, _payload, _stored in pool)
    universe_digest = store.records_digest(frozenset(universe_keys))
    seed = SEED ^ int(universe_digest[:16], 16)
    rng = random.Random(seed)
    sample: list[tuple[str, dict, dict]] = []
    for index, tag in enumerate(tags):
        requested = quotient + int(index < remainder)
        arm = sorted((entry for entry in pool if entry[1].get("model") == tag), key=lambda x: x[0])
        if len(arm) < requested:
            raise SystemExit(
                f"logprob drift refused: model {tag!r} has {len(arm)} trusted logprob "
                f"records, fewer than the registered {requested}; no claims written"
            )
        sample.extend(rng.sample(arm, requested))
    return sample, universe_keys, universe_digest


def _slot(body: dict) -> list[dict]:
    entries = body.get("logprobs") or []
    tokens = [str(entry.get("token", "")) for entry in entries]
    index = _answer_token_index(tokens)
    if index is None or index >= len(entries):
        return []
    return list(entries[index].get("top_logprobs") or [])


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _claim_values(observations: list[dict[str, Any]]) -> dict[str, object]:
    """Derive the complete ledger namespace from authenticated reply pairs."""

    log_drifts: list[float] = []
    probability_drifts: list[float] = []
    mass_drifts: list[float] = []
    answer_matches = 0
    membership_changes = 0
    for row in observations:
        stored = row["stored"]
        fresh = row["fresh"]
        if not isinstance(stored, dict) or not isinstance(fresh, dict):
            raise NetworkClaimBindingError("logprob drift observations must contain reply objects")
        answer_matches += int(fresh.get("response", "") == stored.get("response", ""))
        old_slot, new_slot = _slot(stored), _slot(fresh)
        old = {str(c.get("token", "")): float(c.get("logprob", 0.0)) for c in old_slot}
        new = {str(c.get("token", "")): float(c.get("logprob", 0.0)) for c in new_slot}
        shared = set(old) & set(new)
        membership_changes += len(set(old) ^ set(new))
        if shared:
            log_drifts.append(max(abs(old[token] - new[token]) for token in shared))
        old_dist, new_dist = (
            _distribution_from_candidates(old_slot),
            _distribution_from_candidates(new_slot),
        )
        if old_dist and new_dist:
            probability_drifts.append(
                max(abs(old_dist[symbol] - new_dist[symbol]) for symbol in old_dist)
            )
        mass_drifts.append(abs(captured_mass(old_slot) - captured_mass(new_slot)))

    if not observations:
        raise NetworkClaimBindingError("logprob drift measurement contains no observations")
    return {
        "bench_logprob_drift_positions": len(observations),
        "bench_logprob_drift_answer_agreement": round(answer_matches / len(observations), 6),
        "bench_logprob_drift_max": round(max(log_drifts), 6) if log_drifts else 0.0,
        "bench_logprob_drift_probability": (
            round(max(probability_drifts), 8) if probability_drifts else 0.0
        ),
        "bench_logprob_drift_captured_mass": (round(max(mass_drifts), 8) if mass_drifts else 0.0),
        "bench_logprob_drift_membership_changes": membership_changes,
        "bench_logprob_drift_endpoint": OLLAMA_URL,
    }


def _measurement(
    *,
    universe_keys: list[str],
    universe_digest: str,
    observations: list[dict[str, Any]],
    producer_receipt_sha256: str,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": MEASUREMENT_SCHEMA,
        "selection_universe_keys": universe_keys,
        "selection_universe_records_sha256": universe_digest,
        "sample_keys": [str(row["key"]) for row in observations],
        "observations": observations,
        "producer_receipt_sha256": producer_receipt_sha256,
    }
    return {**body, "observations_sha256": _digest(observations)}


def validate_publication_measurement(receipt: Mapping[str, object], store: CallCache) -> None:
    """Reconstruct selection and claims from a logprob-drift receipt."""

    producer = _load_producer_receipt(store)
    _validate_measurement_against_producer(receipt, store, producer)


def _validate_measurement_against_producer(
    receipt: Mapping[str, object], store: CallCache, producer: Mapping[str, Any]
) -> None:
    """Semantic paired-response check after authenticating the producer receipt."""

    measurement = receipt.get("measurement")
    if not isinstance(measurement, dict) or set(measurement) != {
        "schema",
        "selection_universe_keys",
        "selection_universe_records_sha256",
        "sample_keys",
        "observations",
        "observations_sha256",
        "producer_receipt_sha256",
    }:
        raise NetworkClaimBindingError("logprob drift receipt has invalid measurement fields")
    if measurement["schema"] != MEASUREMENT_SCHEMA:
        raise NetworkClaimBindingError("unsupported logprob drift measurement schema")
    if measurement["producer_receipt_sha256"] != producer.get("receipt_sha256"):
        raise NetworkClaimBindingError("logprob drift producer receipt is stale")
    universe_keys = measurement["selection_universe_keys"]
    observations = measurement["observations"]
    if not isinstance(universe_keys, list) or any(
        not isinstance(key, str) for key in universe_keys
    ):
        raise NetworkClaimBindingError("logprob drift selection universe is invalid")
    if not isinstance(observations, list) or any(not isinstance(row, dict) for row in observations):
        raise NetworkClaimBindingError("logprob drift observations are invalid")

    pool = _producer_pool(store, producer)
    selected, current_universe, current_digest = _sample_entries(store, pool)
    selected_keys = [key for key, _payload, _stored in selected]
    if (
        universe_keys != current_universe
        or measurement["selection_universe_records_sha256"] != current_digest
    ):
        raise NetworkClaimBindingError("logprob drift selection universe is stale")
    if measurement["sample_keys"] != selected_keys:
        raise NetworkClaimBindingError("logprob drift selected keys do not reproduce")
    if measurement["observations_sha256"] != _digest(observations):
        raise NetworkClaimBindingError("logprob drift observation digest mismatch")

    stored_by_key = {key: stored for key, _payload, stored in selected}
    if [row.get("key") for row in observations] != selected_keys or any(
        row.get("stored") != stored_by_key.get(str(row.get("key"))) for row in observations
    ):
        raise NetworkClaimBindingError("logprob drift stored observations are stale")
    derived = _claim_values(observations)
    claims = receipt.get("claims")
    if not isinstance(claims, dict) or not isinstance(claims.get("entries"), dict):
        raise NetworkClaimBindingError("logprob drift receipt claim projection is invalid")
    recorded = {
        label: entry.get("value")
        for label, entry in claims["entries"].items()
        if isinstance(label, str) and isinstance(entry, dict)
    }
    if recorded != derived:
        raise NetworkClaimBindingError(
            "logprob drift fresh observations do not reproduce the bound claims"
        )


def main() -> int:
    publication_run = using_publication_ledger(PROJECT_ROOT)
    initial_source = source_manifest(PROJECT_ROOT, SOURCE)
    store = enable_call_cache(require_default_path=publication_run)
    if store is None:  # enabled=True above; defensive against future API drift
        raise SystemExit("logprob drift refused: call store could not be enabled")
    producer = _load_producer_receipt(store)
    identities: dict[str, str] = {}
    for tag in PROBE_MODEL_TAGS.values():
        identity = resolved_model_identity(tag, endpoint=OLLAMA_URL)
        if identity is None:
            raise SystemExit(
                f"logprob drift refused: immutable identity unavailable for {tag}; "
                "no claims written"
            )
        identities[tag] = identity
    if len(set(identities.values())) != len(identities):
        raise SystemExit(
            "logprob drift refused: registered model tags resolve to one immutable "
            "identity; no claims written"
        )
    pool = _producer_pool(store, producer)
    if not pool:
        raise SystemExit("no stored calls requested logprobs; nothing to measure")

    sample, universe_keys, universe_digest = _sample_entries(store, pool)
    sample_keys = {key for key, _payload, _stored in sample}
    sample_binding = store.provenance_binding(sample_keys)
    expected_model_identities = [
        {"model": tag, "identity": identities[tag]} for tag in sorted(identities)
    ]
    if sample_binding["model_identities"] != expected_model_identities:
        raise SystemExit(
            "logprob drift refused: sampled cache records are not bound to the "
            "currently resolved model identities; no calls made and no claims written"
        )
    observations: list[dict[str, Any]] = []
    for key, payload, stored in sample:
        fresh = post_generate_uncached(payload, endpoint=OLLAMA_URL)
        observations.append({"key": key, "stored": stored, "fresh": fresh})

    values = _claim_values(observations)
    measurement = _measurement(
        universe_keys=universe_keys,
        universe_digest=universe_digest,
        observations=observations,
        producer_receipt_sha256=producer["receipt_sha256"],
    )

    changed_identities = sorted(
        tag for tag, identity in identities.items() if model_digest(tag, OLLAMA_URL) != identity
    )
    if changed_identities:
        raise SystemExit(
            "logprob drift refused: model identity changed while measuring "
            f"{changed_identities}; no claims written"
        )

    if _load_producer_receipt(store)["receipt_sha256"] != producer["receipt_sha256"]:
        raise NetworkClaimBindingError("external producer changed while measuring logprob drift")

    if source_manifest(PROJECT_ROOT, SOURCE) != initial_source:
        raise SystemExit(
            "logprob drift refused: source or vendored input changed before ledger "
            "publication; no claims written"
        )

    configuration = publication_replay_configuration()
    validate_network_measurement_inputs(
        configuration=configuration,
        store=store,
        cache_keys=sample_keys,
        measurement=measurement,
    )
    with single_writer_lock(ledger_lock_path(PROJECT_ROOT)):
        if source_manifest(PROJECT_ROOT, SOURCE) != initial_source:
            raise SystemExit(
                "logprob drift refused: source or vendored input changed while "
                "waiting for the ledger lock; no claims written"
            )
        ledger = load_ledger(PROJECT_ROOT)
        ledger.replace_claims(
            remove=[label for label in ledger.labels_owned_by(SOURCE) if label not in values],
            batches=[ClaimBatch(values=values, run_id=RUN_ID, source=SOURCE)],
        )
        receipt_path = (
            write_network_claim_binding(
                project_root=PROJECT_ROOT,
                recorder=SOURCE,
                expected_source_manifest=initial_source,
                configuration=configuration,
                store=store,
                cache_keys=sample_keys,
                measurement=measurement,
            )
            if publication_run
            else None
        )
    for label, value in sorted(values.items()):
        print(f"  {label:44s} {value}")
    if receipt_path is None:
        print("isolated ledger run: publication network claim binding left unchanged")
    else:
        print(f"network claim binding: {receipt_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
