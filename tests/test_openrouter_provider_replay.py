"""Provider-pinned real HTTP replies replay without a live budget environment."""

from __future__ import annotations

import copy
import importlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from ailm.benchmarks.openrouter import post_chat, resolved_openrouter_identity, set_call_cache
from ailm.benchmarks.openrouter_budget import (
    BUDGET_FILE_ENV,
    BUDGET_LIMIT_ENV,
    SCHEMA,
    configured_provider_options,
)
from ailm.call_cache import CallCache
from ailm.replay_provenance import NetworkClaimBindingError

RUN_SCOPE = "20261005T000000000001Z"


@pytest.mark.parametrize("recorder", ["synthetic", "incontext"])
def test_binding_verifier_reconstructs_provider_from_receipt_only(
    recorder: str, tmp_path: Path, monkeypatch
) -> None:
    from scripts.verify_network_claim_bindings import expected_configurations

    root = Path(__file__).resolve().parents[1]
    relative = f"scripts/record_{recorder}_benchmark_claims.py"
    destination = tmp_path / relative
    destination.parent.mkdir()
    destination.write_bytes((root / relative).read_bytes())
    route = {
        "only": ["recorded-provider"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "max_price": {"prompt": 1.0, "completion": 1.0, "request": 0.0},
    }
    receipt = tmp_path / "cache/network_claim_bindings" / (destination.stem + ".json")
    receipt.parent.mkdir(parents=True)
    payload = {
        "configuration": {
            "parameters": {
                "run_scope": RUN_SCOPE,
                "openrouter_provider": route,
            }
        }
    }
    receipt.write_text(json.dumps(payload))
    monkeypatch.delenv(BUDGET_FILE_ENV, raising=False)
    monkeypatch.delenv(BUDGET_LIMIT_ENV, raising=False)
    assert (
        expected_configurations(tmp_path, [relative])[relative]["parameters"]["openrouter_provider"]
        == route
    )
    payload["configuration"]["parameters"]["openrouter_provider"] = None
    receipt.write_text(json.dumps(payload))
    with pytest.raises(NetworkClaimBindingError, match="provider declaration"):
        expected_configurations(tmp_path, [relative])


@pytest.mark.parametrize("recorder", ["synthetic", "incontext"])
def test_actual_budgeted_reply_replays_exact_provider_after_environment_removed(
    recorder: str, local_http_server, tmp_path: Path, monkeypatch
) -> None:
    module = importlib.import_module(f"scripts.record_{recorder}_benchmark_claims")
    budget_file = tmp_path / "budget.json"
    budget_file.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "limit_usd": "1",
                "blocked": False,
                "models": {
                    module.OPENROUTER_MODEL: {
                        "provider": "local-test-provider",
                        "context_length": 8192,
                        "max_completion_tokens": 128,
                        "prompt_usd_per_token": "0.000001",
                        "completion_usd_per_token": "0.000001",
                        "request_usd": "0",
                        "source_sha256": "a" * 64,
                    }
                },
                "calls": [],
            }
        )
    )
    monkeypatch.setenv(BUDGET_FILE_ENV, str(budget_file))
    monkeypatch.setenv(BUDGET_LIMIT_ENV, "1")
    provider = configured_provider_options(module.OPENROUTER_MODEL)
    assert provider is not None
    endpoint = local_http_server.base_url + "/chat"
    store = CallCache(tmp_path / "calls.jsonl")
    requests = []
    if recorder == "synthetic":
        for arm in module.ARMS:
            positions = arm.held_out_positions(2)
            short = module._short_positions(arm, positions)
            for tag in module.MODELS.values():
                for index, (full, narrow) in enumerate(zip(positions, short, strict=True)):
                    for arm_name, (prefix, _gold) in (("full", full), ("short", narrow)):
                        payload = module._chat_payload(
                            module.OPENROUTER_MODEL,
                            module.build_prompt(arm.task, prefix),
                            arm.task,
                            arm.protocol.response_tokens,
                            module.DEFAULT_SEED,
                        )
                        protocol = module._openrouter_cache_protocol(
                            module._call_cache_protocol(arm, tag, index, arm_name, RUN_SCOPE)
                        )
                        requests.append((payload, protocol, arm.protocol.response_inventory[0]))
    else:
        positions = module._scored_positions(2)
        prompts = module._all_arm_prompts(positions, module._example_pool(positions))
        for tag in module.MODELS.values():
            for index, _position in enumerate(positions):
                for arm in module._rotated_arm_names(index):
                    payload = module._chat_payload(
                        module.OPENROUTER_MODEL,
                        prompts[arm][index],
                        "synthetic_letter",
                        module.SYNTHETIC_INCONTEXT_PROTOCOL.response_tokens,
                        module.DEFAULT_SEED,
                    )
                    protocol = module._openrouter_cache_protocol(
                        module._call_cache_protocol(tag, index, arm, RUN_SCOPE)
                    )
                    requests.append((payload, protocol, "a"))
    set_call_cache(store)
    try:
        for payload, protocol, answer in requests:
            local_http_server.enqueue_json(
                "/chat",
                {
                    "model": module.OPENROUTER_MODEL,
                    "choices": [{"message": {"content": json.dumps({"answer": answer})}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 1, "cost": 0},
                },
            )
            post_chat(payload, api_key="dummy-key", cache_protocol=protocol, endpoint=endpoint)
    finally:
        set_call_cache(None)
    keys = sorted(key for key, _payload, _reply in store.entries())
    assert len(local_http_server.requests_for("/chat")) == len(keys) == len(requests)
    configuration = module._replay_configuration(
        positions=2, run_scope=RUN_SCOPE, endpoint=endpoint, openrouter_provider=provider
    )
    receipt = {
        "configuration": configuration,
        "cache": {
            "keys": keys,
            "model_identities": [
                {
                    "model": module.OPENROUTER_MODEL,
                    "identity": resolved_openrouter_identity(module.OPENROUTER_MODEL),
                }
            ],
        },
        "measurement": {
            "arms": {arm.key: {"maximum_prompt_tokens": 7} for arm in getattr(module, "ARMS", ())}
        },
    }
    monkeypatch.delenv(BUDGET_FILE_ENV)
    monkeypatch.delenv(BUDGET_LIMIT_ENV)
    values = module._replay_result_claims(receipt, store, keys, positions_count=2)
    prefix = "bench_synth" if recorder == "synthetic" else "bench_ctx"
    assert values[f"{prefix}_openrouter_model"] == module.OPENROUTER_MODEL
    assert values[f"{prefix}_openrouter_provider"] == "local-test-provider"
    assert (
        module.publication_replay_configuration(run_scope=RUN_SCOPE, openrouter_provider=provider)[
            "parameters"
        ]["openrouter_provider"]
        == provider
    )
    for mutation in ("missing", "different"):
        changed = copy.deepcopy(receipt)
        parameters = changed["configuration"]["parameters"]
        if mutation == "missing":
            parameters.pop("openrouter_provider")
        else:
            parameters["openrouter_provider"]["only"] = ["other-provider"]
        with pytest.raises(NetworkClaimBindingError, match="lacks exact"):
            module._replay_result_claims(changed, store, keys, positions_count=2)
    assert (
        "openrouter_provider"
        not in module.publication_replay_configuration(run_scope=RUN_SCOPE)["parameters"]
    )


@pytest.mark.timeout(180)
def test_hosted_producer_consumer_cli_never_queries_local_weights(
    tmp_path: Path, local_http_server
) -> None:
    """Real isolated CLI runs score fresh chat cells and replay their real producer cache."""
    root = Path(__file__).resolve().parents[1]
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    shutil.copytree(root / "src", checkout / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(
        root / "scripts", checkout / "scripts", ignore=shutil.ignore_patterns("__pycache__")
    )
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(root / name, checkout / name)
    (checkout / "vendor").mkdir()
    for name in ("synthetic_counts.json", "synthetic_phone_counts.json", "cmudict_counts.json"):
        shutil.copy2(root / "vendor" / name, checkout / "vendor" / name)
    ledger = tmp_path / "isolated-claims.json"
    cache = tmp_path / "isolated-calls.jsonl"
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("AILM_OPENROUTER") and key != "OPENROUTER_API_KEY"
    }
    environment.update(
        AILM_CLAIMS_PATH=str(ledger),
        AILM_CALL_CACHE_PATH=str(cache),
        OPENROUTER_API_KEY="loopback-test-key",
    )
    producer = importlib.import_module("scripts.record_synthetic_benchmark_claims")
    consumer = importlib.import_module("scripts.record_incontext_benchmark_claims")
    positions = 8
    producer_calls = len(producer.MODELS) * positions * len(producer.ARMS) * 2
    consumer_calls = len(consumer.MODELS) * positions * len(consumer._arm_specs())
    local_http_server.enqueue_json(
        "/chat",
        {
            "model": producer.OPENROUTER_MODEL,
            "choices": [{"message": {"content": '{"answer":"a"}'}}],
            "usage": {"prompt_tokens": 7},
        },
        repeat=producer_calls + consumer_calls,
    )
    outputs = {}
    for name, module in (("synthetic", producer), ("incontext", consumer)):
        result = subprocess.run(
            [
                str(root / ".venv/bin/python"),
                str(checkout / f"scripts/record_{name}_benchmark_claims.py"),
                "--positions",
                str(positions),
                "--endpoint",
                local_http_server.base_url + "/chat",
            ],
            cwd=checkout,
            env=environment,
            text=True,
            capture_output=True,
            timeout=120,
        )
        outputs[name] = {"stdout": result.stdout, "stderr": result.stderr}
        assert result.returncode == 0, outputs
        if name == "synthetic":
            match = re.search(r"synthetic run scope: ([0-9TZ]+)", result.stdout)
            assert match is not None
            # Isolated validation records no publication receipt. Supply only its
            # actual scope/route so the consumer locates the observed producer calls.
            receipt = (
                checkout / "cache/network_claim_bindings/record_synthetic_benchmark_claims.json"
            )
            receipt.parent.mkdir(parents=True)
            receipt.write_text(
                json.dumps(
                    {
                        "configuration": module._replay_configuration(
                            positions=positions,
                            run_scope=match.group(1),
                            endpoint=local_http_server.base_url + "/chat",
                        )
                    }
                )
            )
    claims = json.loads(ledger.read_text())["claims"]
    assert claims["bench_ctx_positions"]["value"] == positions
    for cohort in consumer.MODELS:
        for k in consumer.K_GRID:
            assert f"bench_ctx_top1_k{k}_{cohort}" in claims
    assert claims["bench_ctx_scored_cache_misses"]["value"] == consumer_calls
    assert claims["bench_ctx_scored_cache_hits"]["value"] == 0
    assert claims["bench_ctx_warmup_policy"]["value"] == "hosted-no-warmup-fresh-scored-cells-v1"
    assert len(local_http_server.requests) == producer_calls + consumer_calls
    assert all(request.path == "/chat" for request in local_http_server.requests)
    assert all(
        json.loads(request.body)["model"] == producer.OPENROUTER_MODEL
        for request in local_http_server.requests
    )
    assert "no-demonstration arm reproduces" in outputs["incontext"]["stdout"]


@pytest.mark.parametrize("name", ["synthetic", "incontext"])
def test_publication_hosted_endpoint_refusal_precedes_every_call(
    name: str, local_http_server
) -> None:
    root = Path(__file__).resolve().parents[1]
    ledger = root / "data/claims.json"
    before = ledger.read_bytes()
    environment = dict(os.environ)
    environment.pop("AILM_CLAIMS_PATH", None)
    result = subprocess.run(
        [
            str(root / ".venv/bin/python"),
            str(root / f"scripts/record_{name}_benchmark_claims.py"),
            "--endpoint",
            local_http_server.base_url + "/chat",
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "publication ledger requires the canonical endpoint" in result.stderr
    assert not local_http_server.requests
    assert ledger.read_bytes() == before
