"""Boundary contracts of the Ollama client, against a real loopback server.

The client's refusals are demonstrated with the narrowest malformed argument
or server reply. Every server reply comes from the conftest loopback server,
so what is asserted is the client's behavior on real bytes over a real socket,
never a substituted transport.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ailm.benchmarks.constants import DEFAULT_SEED, PROTOCOL_VERSION
from ailm.benchmarks.ollama import (
    _generate_payload,
    ask_ollama,
    ask_ollama_answer_candidates,
    default_context_token_bound,
    enable_call_cache,
    ollama_available,
    post_generate_uncached,
    resolved_model_identity,
    uses_current_ollama_loader,
    warm_ollama_model,
)
from ailm.call_cache import CacheRequest
from ailm.loopback_http import LoopbackTransportError


def test_generation_payload_refuses_a_non_positive_prediction_budget() -> None:
    with pytest.raises(ValueError, match="num_predict must be positive"):
        _generate_payload("gemma3:4b", "prompt", "letter", 0, 1)


def test_generation_payload_refuses_a_context_no_larger_than_the_prediction() -> None:
    with pytest.raises(ValueError, match="num_ctx must be larger than num_predict"):
        _generate_payload("gemma3:4b", "prompt", "letter", 4, 1, num_ctx=4)


@pytest.mark.parametrize("body", [{"models": "gemma3:4b"}, ["gemma3:4b"], {"tags": []}])
def test_availability_is_false_when_the_tags_reply_is_mis_shaped(local_http_server, body) -> None:
    local_http_server.enqueue_json("/api/tags", body)
    assert ollama_available("gemma3:4b", endpoint=local_http_server.base_url) is False


def test_uncached_generation_refuses_a_non_object_reply(local_http_server) -> None:
    local_http_server.enqueue_json("/api/generate", ["not", "an", "object"])
    payload = _generate_payload("gemma3:4b", "prompt", "letter", 4, 1)
    with pytest.raises(LoopbackTransportError, match="must be a JSON object"):
        post_generate_uncached(payload, endpoint=local_http_server.base_url)


def test_warm_up_refuses_a_non_positive_prediction_budget(local_http_server) -> None:
    with pytest.raises(ValueError, match="warm-up num_predict must be positive"):
        warm_ollama_model("gemma3:4b", "prompt", "letter", 0, endpoint=local_http_server.base_url)
    assert local_http_server.requests == ()


def test_warm_up_returns_the_reply_text_from_a_real_generation_call(local_http_server) -> None:
    local_http_server.enqueue_json("/api/generate", {"response": '{"answer":"q"}'})
    reply = warm_ollama_model(
        "gemma3:4b", "the quick brown fo", "letter", 4, endpoint=local_http_server.base_url
    )
    assert reply == '{"answer":"q"}'
    [request] = local_http_server.requests_for("/api/generate")
    sent = json.loads(request.body)
    assert sent["model"] == "gemma3:4b"
    assert sent["options"]["num_predict"] == 4


def test_warm_up_refuses_a_reply_without_response_text(local_http_server) -> None:
    local_http_server.enqueue_json("/api/generate", {"response": 7})
    with pytest.raises(LoopbackTransportError, match="no string 'response' field"):
        warm_ollama_model("gemma3:4b", "prompt", "letter", 4, endpoint=local_http_server.base_url)


def test_resolved_identity_is_the_content_digest_the_server_reports(local_http_server) -> None:
    digest = "c" * 64
    local_http_server.enqueue_json("/api/show", {"digest": f"sha256:{digest}"})
    identity = resolved_model_identity(
        "fixture-identity-model:latest", endpoint=local_http_server.base_url + "/"
    )
    assert identity == f"sha256:{digest}"
    [request] = local_http_server.requests_for("/api/show")
    assert json.loads(request.body) == {"model": "fixture-identity-model:latest"}


@pytest.mark.parametrize("operation", [warm_ollama_model, ask_ollama, ask_ollama_answer_candidates])
def test_every_local_generation_route_declares_the_loader_option(local_http_server, operation):
    """Observe actual request bytes for warmup, top-one and logprob generation."""
    enable_call_cache(enabled=False)
    local_http_server.enqueue_json("/api/generate", {"response": '{"answer":"q"}'})
    operation("fixture-loader:latest", "prompt", "letter", 4, endpoint=local_http_server.base_url)
    [request] = local_http_server.requests_for("/api/generate")
    assert json.loads(request.body)["options"].get("use_mmap") is False
    assert json.loads(request.body)["options"].get("num_ctx") == 8192


@pytest.mark.parametrize("old_choice", [None, True])
def test_old_loader_cache_record_cannot_satisfy_the_new_actual_request(
    tmp_path: Path, local_http_server, old_choice
):
    endpoint = local_http_server.base_url
    model = "fixture-loader-cache:latest"
    identity = "sha256:" + "f" * 64
    local_http_server.enqueue_json("/api/show", {"digest": identity})
    local_http_server.enqueue_json("/api/generate", {"response": "fresh fixed-loader reply"})
    store = enable_call_cache(tmp_path / "calls.jsonl")
    assert store is not None
    old_payload = _generate_payload(model, "same prompt", "letter", 4, DEFAULT_SEED)
    if old_choice is None:
        old_payload["options"].pop("use_mmap")
    else:
        old_payload["options"]["use_mmap"] = old_choice
    old = CacheRequest(
        payload=old_payload, endpoint=endpoint, protocol=PROTOCOL_VERSION, identity=identity
    )
    store.put(old, {"response": "historical incompatible reply"})
    try:
        actual = ask_ollama(model, "same prompt", "letter", 4, endpoint=endpoint)
        repeated = ask_ollama(model, "same prompt", "letter", 4, endpoint=endpoint)
        assert actual == repeated == "fresh fixed-loader reply"
        [request] = local_http_server.requests_for("/api/generate")
        sent = json.loads(request.body)
        assert sent["options"]["use_mmap"] is False
        assert sent["options"]["seed"] == DEFAULT_SEED
        assert store.get(old)["response"] == "historical incompatible reply"
    finally:
        enable_call_cache(enabled=False)


def test_reconstructed_publication_payloads_include_the_same_fixed_serving_choice():
    from scripts import warm_call_cache

    configuration = warm_call_cache._publication_configuration()
    payloads = warm_call_cache._publication_payloads("gemma3:4b", configuration)
    assert payloads and any(payload.get("logprobs") for payload in payloads)
    assert all(payload["options"].get("use_mmap") is False for payload in payloads)
    assert all(payload["options"].get("num_ctx") == 8192 for payload in payloads)
    assert all(
        default_context_token_bound(payload["prompt"], payload["options"]["num_predict"]) <= 8192
        for payload in payloads
    )


def test_explicit_context_is_preserved_on_the_wire(local_http_server):
    local_http_server.enqueue_json("/api/generate", {"response": "ok"})
    ask_ollama(
        "fixture-explicit:latest",
        "prompt",
        "letter",
        4,
        endpoint=local_http_server.base_url,
        num_ctx=4096,
    )
    [request] = local_http_server.requests_for("/api/generate")
    assert json.loads(request.body)["options"]["num_ctx"] == 4096


@pytest.mark.parametrize("context", [True, 8192.0, "8192"])
def test_context_must_be_an_explicit_integer(context):
    with pytest.raises(ValueError, match="integer"):
        _generate_payload("fixture", "prompt", "letter", 4, 1, context)


def test_default_context_must_cover_prediction_and_conservative_prompt_budget(local_http_server):
    with pytest.raises(ValueError, match="larger than num_predict"):
        ask_ollama("fixture", "prompt", "letter", 8192, endpoint=local_http_server.base_url)
    with pytest.raises(ValueError, match="context budget exceeded"):
        warm_ollama_model("fixture", "a" * 3000, "letter", 4, endpoint=local_http_server.base_url)
    assert not local_http_server.requests


@pytest.mark.parametrize("context", [None, 4096, 8192.0, True])
def test_legacy_context_choice_cannot_satisfy_current_cached_request(
    tmp_path, local_http_server, context
):
    endpoint = local_http_server.base_url
    model = "fixture-context-cache:latest"
    identity = "sha256:" + "e" * 64
    old = _generate_payload(model, "prompt", "letter", 4, DEFAULT_SEED)
    if context is None:
        old["options"].pop("num_ctx")
    else:
        old["options"]["num_ctx"] = context
    assert not uses_current_ollama_loader(old)
    store = enable_call_cache(tmp_path / "calls.jsonl")
    assert store is not None
    store.put(
        CacheRequest(payload=old, endpoint=endpoint, protocol=PROTOCOL_VERSION, identity=identity),
        {"response": "old context"},
    )
    local_http_server.enqueue_json("/api/show", {"digest": identity})
    local_http_server.enqueue_json("/api/generate", {"response": "explicit current context"})
    try:
        assert (
            ask_ollama(model, "prompt", "letter", 4, endpoint=endpoint)
            == "explicit current context"
        )
        [request] = local_http_server.requests_for("/api/generate")
        assert json.loads(request.body)["options"]["num_ctx"] == 8192
        assert len(store) == 2
    finally:
        enable_call_cache(enabled=False)


def test_default_context_bound_accepts_capacity_and_rejects_next_byte():
    prompt = "a" * 1364
    assert default_context_token_bound(prompt, 4) == 8192
    assert _generate_payload("fixture", prompt, "letter", 4, 1)["options"]["num_ctx"] == 8192
    with pytest.raises(ValueError, match="context budget exceeded"):
        _generate_payload("fixture", prompt + "a", "letter", 4, 1)
