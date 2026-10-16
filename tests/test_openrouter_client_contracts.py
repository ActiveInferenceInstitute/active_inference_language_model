"""Boundary contracts of the OpenRouter client, against a real loopback server.

Mirrors ``tests/test_ollama_client_contracts.py``'s discipline: every server
reply comes from the conftest loopback server (never a substituted transport),
so what is asserted is the client's behavior on real bytes over a real socket.
No mocks — see ``tests/test_no_dependency_replacement.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from ailm.benchmarks.openrouter import (
    OPENROUTER_API_KEY_ENV,
    ask_openrouter,
    post_chat,
    set_call_cache,
)
from ailm.benchmarks.openrouter_budget import (
    BUDGET_FILE_ENV,
    BUDGET_LIMIT_ENV,
    OpenRouterBudgetError,
    budget_for,
    configured_provider_options,
    validate_provider_options,
)
from ailm.call_cache import CallCache

#: The path segment the loopback server answers under. OpenRouter's real
#: endpoint is a full URL rather than a base-plus-path split like Ollama's, so
#: every test builds ``local_http_server.base_url + _CHAT_PATH`` and queues
#: the matching path on the server.
_CHAT_PATH = "/api/v1/chat/completions"


@pytest.fixture
def _no_cache() -> None:
    """Ensure the module-level call cache is off, exactly as a fresh import."""
    set_call_cache(None)
    yield
    set_call_cache(None)


def _chat_completion_body(content: str, *, prompt_tokens: int, served_model: str) -> dict:
    return {
        "model": served_model,
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 4},
    }


def test_success_path_returns_response_text_and_prompt_tokens_contract(
    local_http_server, tmp_path, monkeypatch
) -> None:
    """A cached call carries the response text and the server's usage count.

    ``post_chat`` only computes and validates ``prompt_eval_count`` on the
    cache-bound path — the uncached branch returns the raw response text
    alone — so this exercises the client through a real ``CallCache`` rather
    than through ``ask_openrouter`` directly, matching how every recorder in
    this repository actually calls it.
    """
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "dummy-key")
    local_http_server.enqueue_json(
        _CHAT_PATH,
        _chat_completion_body(
            '{"answer":"q"}', prompt_tokens=17, served_model="openrouter/test-model"
        ),
    )
    cache = CallCache(path=tmp_path / "calls.jsonl")
    set_call_cache(cache)
    try:
        payload = {
            "model": "openrouter/test-model",
            "messages": [{"role": "user", "content": "the quick brown fo"}],
            "max_tokens": 4,
            "temperature": 0,
            "seed": 1,
        }
        body = post_chat(
            payload,
            api_key="dummy-key",
            cache_protocol="openrouter-test-v1",
            endpoint=local_http_server.base_url + _CHAT_PATH,
        )
        assert body["response"] == '{"answer":"q"}'
        assert body["prompt_eval_count"] == 17
        assert body["_identity"]
        assert cache.stats.misses == 1

        [request] = local_http_server.requests_for(_CHAT_PATH)
        sent = json.loads(request.body)
        assert sent["model"] == "openrouter/test-model"
        assert sent["max_tokens"] == 4
        assert request.headers.get("authorization") == "Bearer dummy-key"

        # A second, identical call is answered from the store without a
        # second request reaching the server.
        replay = post_chat(
            payload,
            api_key="dummy-key",
            cache_protocol="openrouter-test-v1",
            endpoint=local_http_server.base_url + _CHAT_PATH,
        )
        assert replay == body
        assert len(local_http_server.requests_for(_CHAT_PATH)) == 1
    finally:
        set_call_cache(None)


def test_ask_openrouter_returns_response_text_uncached(
    local_http_server, monkeypatch, _no_cache
) -> None:
    """``ask_openrouter`` end to end, uncached, mirroring ``ask_ollama``'s test."""
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "dummy-key")
    local_http_server.enqueue_json(
        _CHAT_PATH,
        _chat_completion_body("hello", prompt_tokens=3, served_model="openrouter/test-model"),
    )
    reply = ask_openrouter(
        "openrouter/test-model",
        "prompt",
        "letter",
        4,
        endpoint=local_http_server.base_url + _CHAT_PATH,
    )
    assert reply == "hello"
    [request] = local_http_server.requests_for(_CHAT_PATH)
    assert json.loads(request.body)["model"] == "openrouter/test-model"


@pytest.mark.parametrize("status", [400, 401, 402, 403])
def test_http_error_branch_reports_status_and_body_without_retrying(
    local_http_server, _no_cache, status
) -> None:
    """A refused HTTP response raises once, carrying the status and body."""
    local_http_server.enqueue_json(_CHAT_PATH, {"error": "refused"}, status=status)
    with pytest.raises(RuntimeError, match=f"OpenRouter API returned {status}"):
        post_chat(
            {"model": "openrouter/test-model", "messages": []},
            api_key="dummy-key",
            cache_protocol="openrouter-test-v1",
            endpoint=local_http_server.base_url + _CHAT_PATH,
        )
    assert len(local_http_server.requests_for(_CHAT_PATH)) == 1


@pytest.mark.timeout(240)  # waits out the real retry backoff; nothing is replaced
def test_retry_then_success_replays_the_same_payload(local_http_server, _no_cache) -> None:
    """A transient disconnect is retried with the identical payload, then succeeds."""
    local_http_server.enqueue_disconnect(_CHAT_PATH)
    local_http_server.enqueue_json(
        _CHAT_PATH,
        _chat_completion_body("recovered", prompt_tokens=5, served_model="openrouter/test-model"),
    )
    body = post_chat(
        {"model": "openrouter/test-model", "messages": [{"role": "user", "content": "x"}]},
        api_key="dummy-key",
        cache_protocol="openrouter-test-v1",
        endpoint=local_http_server.base_url + _CHAT_PATH,
    )
    assert body["response"] == "recovered"
    requests = local_http_server.requests_for(_CHAT_PATH)
    assert len(requests) == 2
    assert json.loads(requests[0].body) == json.loads(requests[1].body)


@pytest.mark.timeout(240)  # waits out the real retry backoff; nothing is replaced
def test_retry_exhaustion_raises_runtime_error(local_http_server, _no_cache) -> None:
    """Every attempt failing transiently ends in one labeled RuntimeError.

    Not the raw transport exception (``ConnectionResetError`` / ``OSError``):
    a caller of this client should not need to know which stdlib exception a
    dropped socket happens to surface as, and the module already labels its
    other terminal failure (the HTTPError branch) as a ``RuntimeError`` the
    same way.
    """
    local_http_server.enqueue_disconnect(_CHAT_PATH)
    local_http_server.enqueue_disconnect(_CHAT_PATH)
    local_http_server.enqueue_disconnect(_CHAT_PATH)
    with pytest.raises(RuntimeError, match="OpenRouter POST exhausted its retry attempts"):
        post_chat(
            {"model": "openrouter/test-model", "messages": [{"role": "user", "content": "x"}]},
            api_key="dummy-key",
            cache_protocol="openrouter-test-v1",
            endpoint=local_http_server.base_url + _CHAT_PATH,
        )
    assert len(local_http_server.requests_for(_CHAT_PATH)) == 3


def test_oversized_response_is_refused_before_json_parsing(local_http_server, _no_cache) -> None:
    """A reply past the byte cap is refused, not truncated and parsed."""
    from ailm.benchmarks import openrouter as openrouter_module

    oversized = json.dumps(
        {"choices": [{"message": {"content": "x" * openrouter_module._MAX_RESPONSE_BYTES}}]}
    ).encode("utf-8")
    assert len(oversized) > openrouter_module._MAX_RESPONSE_BYTES
    local_http_server.enqueue_bytes(_CHAT_PATH, oversized, content_type="application/json")
    with pytest.raises(ValueError, match="OpenRouter response exceeds"):
        post_chat(
            {"model": "openrouter/test-model", "messages": []},
            api_key="dummy-key",
            cache_protocol="openrouter-test-v1",
            endpoint=local_http_server.base_url + _CHAT_PATH,
        )


def test_missing_api_key_raises_value_error(monkeypatch) -> None:
    """``ask_openrouter`` refuses to run with no key configured anywhere."""
    monkeypatch.delenv(OPENROUTER_API_KEY_ENV, raising=False)
    with pytest.raises(ValueError, match=f"{OPENROUTER_API_KEY_ENV} is not set"):
        ask_openrouter("openrouter/test-model", "prompt", "letter")


@pytest.fixture
def budget_file(tmp_path, monkeypatch):
    path = tmp_path / "budget.json"
    path.write_text(
        json.dumps(
            {
                "schema": "ailm-openrouter-budget-v1",
                "limit_usd": "0.20",
                "blocked": False,
                "models": {
                    "openrouter/test-model": {
                        "provider": "test-provider",
                        "context_length": 10,
                        "max_completion_tokens": 4,
                        "prompt_usd_per_token": "0.01",
                        "completion_usd_per_token": "0.01",
                        "request_usd": "0",
                        "source_sha256": "a" * 64,
                    }
                },
                "calls": [],
            }
        )
    )
    monkeypatch.setenv(BUDGET_FILE_ENV, str(path))
    monkeypatch.setenv(BUDGET_LIMIT_ENV, "0.20")
    return path


def _budgeted_call(server):
    return ask_openrouter(
        "openrouter/test-model",
        "prompt",
        "letter",
        4,
        api_key="dummy-key",
        endpoint=server.base_url + _CHAT_PATH,
    )


def test_budget_reconciles_actual_cost_and_caps_provider_routing(
    local_http_server, budget_file, _no_cache
) -> None:
    body = _chat_completion_body("ok", prompt_tokens=3, served_model="openrouter/test-model")
    body["usage"]["cost"] = 0.02
    local_http_server.enqueue_json(_CHAT_PATH, body, repeat=2)
    assert _budgeted_call(local_http_server) == "ok"
    assert _budgeted_call(local_http_server) == "ok"
    state = json.loads(budget_file.read_text())
    assert [row["reserved_usd"] for row in state["calls"]] == ["0.14", "0.14"]
    assert [row["cost_usd"] for row in state["calls"]] == ["0.02", "0.02"]
    request = json.loads(local_http_server.requests_for(_CHAT_PATH)[0].body)
    assert request["provider"] == {
        "only": ["test-provider"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "max_price": {"prompt": 10000.0, "completion": 10000.0, "request": 0.0},
    }


def test_missing_cost_retains_reservation_and_refuses_next_http(
    local_http_server, budget_file, _no_cache
) -> None:
    local_http_server.enqueue_json(
        _CHAT_PATH, _chat_completion_body("unknown charge", prompt_tokens=3, served_model="x")
    )
    with pytest.raises(OpenRouterBudgetError, match="usage.cost missing"):
        _budgeted_call(local_http_server)
    with pytest.raises(OpenRouterBudgetError, match="exhausted before HTTP"):
        _budgeted_call(local_http_server)
    assert len(local_http_server.requests_for(_CHAT_PATH)) == 1
    assert json.loads(budget_file.read_text())["calls"][0]["cost_usd"] is None


def test_duplicate_response_cost_is_refused_without_releasing_funds(
    local_http_server, budget_file, _no_cache
) -> None:
    local_http_server.enqueue_bytes(
        _CHAT_PATH,
        b'{"choices": [], "usage": {"cost": 0.14, "cost": 0}}',
        content_type="application/json",
    )
    with pytest.raises(OpenRouterBudgetError, match="duplicate JSON key"):
        _budgeted_call(local_http_server)
    with pytest.raises(OpenRouterBudgetError, match="exhausted before HTTP"):
        _budgeted_call(local_http_server)
    assert len(local_http_server.requests_for(_CHAT_PATH)) == 1
    assert json.loads(budget_file.read_text())["calls"][0]["cost_usd"] is None


def test_duplicate_ledger_accounting_is_refused_before_http(
    local_http_server, budget_file, _no_cache
) -> None:
    original = budget_file.read_text()
    budget_file.write_text(original[:-1] + ', "calls": []}')
    with pytest.raises(OpenRouterBudgetError, match="duplicate JSON key"):
        _budgeted_call(local_http_server)
    assert local_http_server.requests_for(_CHAT_PATH) == ()


def test_receipt_provider_snapshot_validates_without_environment(budget_file, monkeypatch) -> None:
    provider = configured_provider_options("openrouter/test-model")
    assert provider is not None
    monkeypatch.delenv(BUDGET_FILE_ENV)
    monkeypatch.delenv(BUDGET_LIMIT_ENV)
    assert validate_provider_options(provider) == provider
    assert configured_provider_options("openrouter/test-model") is None
    assert validate_provider_options(None) is None
    assert json.loads(budget_file.read_text())["calls"] == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("only", []),
        ("only", ["one", "two"]),
        ("only", [""]),
        ("allow_fallbacks", True),
        ("require_parameters", False),
        ("max_price", {"prompt": 1, "completion": 1}),
        ("max_price", {"prompt": "1", "completion": 1, "request": 0}),
        ("max_price", {"prompt": float("nan"), "completion": 1, "request": 0}),
    ],
)
def test_invalid_receipt_provider_constraints_are_refused(budget_file, field, value) -> None:
    provider = configured_provider_options("openrouter/test-model")
    provider[field] = value
    with pytest.raises(OpenRouterBudgetError):
        validate_provider_options(provider)


def test_midrequest_pricing_change_refuses_reservation(budget_file) -> None:
    request = budget_for(
        {
            "model": "openrouter/test-model",
            "messages": [{"role": "user", "content": "prompt"}],
            "max_tokens": 4,
        }
    )
    state = json.loads(budget_file.read_text())
    state["models"]["openrouter/test-model"]["context_length"] = 1
    budget_file.write_text(json.dumps(state))
    with pytest.raises(OpenRouterBudgetError, match="pricing changed"):
        request.reserve()
    assert json.loads(budget_file.read_text())["calls"] == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("limit_usd", "NaN"),
        ("limit_usd", "0.21"),
        ("blocked", True),
        ("prompt_usd_per_token", "NaN"),
        ("completion_usd_per_token", "Infinity"),
        ("request_usd", "-1"),
        ("context_length", True),
        ("source_sha256", "unpinned"),
    ],
)
def test_invalid_budget_refuses_before_http(
    local_http_server, budget_file, _no_cache, field, value
) -> None:
    state = json.loads(budget_file.read_text())
    target = (
        state if field in {"limit_usd", "blocked"} else state["models"]["openrouter/test-model"]
    )
    target[field] = value
    budget_file.write_text(json.dumps(state))
    with pytest.raises(OpenRouterBudgetError):
        _budgeted_call(local_http_server)
    assert local_http_server.requests_for(_CHAT_PATH) == ()


@pytest.mark.parametrize("cost", [None, "NaN", -1, True])
def test_invalid_actual_cost_never_releases_reservation(
    local_http_server, budget_file, _no_cache, cost
) -> None:
    body = _chat_completion_body("bad cost", prompt_tokens=3, served_model="x")
    body["usage"]["cost"] = cost
    local_http_server.enqueue_json(_CHAT_PATH, body)
    with pytest.raises(OpenRouterBudgetError, match="usage.cost"):
        _budgeted_call(local_http_server)
    assert json.loads(budget_file.read_text())["calls"][0]["cost_usd"] is None


def test_provider_charge_above_reservation_blocks_budget(
    local_http_server, budget_file, _no_cache
) -> None:
    body = _chat_completion_body("violated provider bound", prompt_tokens=3, served_model="x")
    body["usage"]["cost"] = 0.15
    local_http_server.enqueue_json(_CHAT_PATH, body)
    with pytest.raises(OpenRouterBudgetError, match="exceeded reserved bound"):
        _budgeted_call(local_http_server)
    with pytest.raises(OpenRouterBudgetError, match="budget blocked"):
        _budgeted_call(local_http_server)
    assert len(local_http_server.requests_for(_CHAT_PATH)) == 1
    state = json.loads(budget_file.read_text())
    assert state["blocked"] is True
    assert state["calls"][0]["cost_usd"] == "0.15"


@pytest.mark.timeout(60)
@pytest.mark.parametrize("optimized", [False, True])
def test_shared_budget_serializes_independent_processes(
    local_http_server, budget_file, optimized
) -> None:
    local_http_server.enqueue_json(
        _CHAT_PATH,
        _chat_completion_body("unknown charge", prompt_tokens=3, served_model="x"),
        repeat=4,
    )
    program = (
        "from ailm.benchmarks.openrouter import ask_openrouter; "
        "ask_openrouter('openrouter/test-model','prompt','letter',4,api_key='dummy-key',"
        "endpoint=__import__('sys').argv[1])"
    )
    workers = [
        subprocess.Popen(
            [
                sys.executable,
                *(["-O"] if optimized else []),
                "-c",
                program,
                local_http_server.base_url + _CHAT_PATH,
            ],
            env=dict(os.environ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]
    outputs = [worker.communicate(timeout=30) for worker in workers]
    assert all(worker.returncode != 0 for worker in workers), outputs
    assert sum("usage.cost missing" in stderr for _, stderr in outputs) == 1, outputs
    assert sum("exhausted before HTTP" in stderr for _, stderr in outputs) == 3, outputs
    assert len(local_http_server.requests_for(_CHAT_PATH)) == 1
    assert len(json.loads(budget_file.read_text())["calls"]) == 1


def _budget_subprocess(server, optimized: bool, calls: int = 1):
    program = """
import sys
from ailm.benchmarks.openrouter import ask_openrouter
from ailm.benchmarks.openrouter_budget import OpenRouterBudgetError
print('OPTIMIZED=' + str(sys.flags.optimize), flush=True)
for _ in range(int(sys.argv[2])):
    try:
        answer = ask_openrouter('openrouter/test-model', 'prompt', 'letter', 4,
                                api_key='dummy-key', endpoint=sys.argv[1])
    except OpenRouterBudgetError as error:
        print('REFUSED=' + str(error), flush=True)
    else:
        print('ACCEPTED=' + answer, flush=True)
"""
    return subprocess.run(
        [
            sys.executable,
            *(["-O"] if optimized else []),
            "-c",
            program,
            server.base_url + _CHAT_PATH,
            str(calls),
        ],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )


@pytest.mark.parametrize("optimized", [False, True])
def test_budget_success_survives_optimization(local_http_server, budget_file, optimized):
    body = _chat_completion_body("ok", prompt_tokens=3, served_model="x")
    body["usage"]["cost"] = 0.02
    local_http_server.enqueue_json(_CHAT_PATH, body, repeat=2)
    result = _budget_subprocess(local_http_server, optimized, calls=2)
    assert f"OPTIMIZED={int(optimized)}" in result.stdout
    assert result.stdout.count("ACCEPTED=ok") == 2
    assert "REFUSED=" not in result.stdout
    assert len(local_http_server.requests_for(_CHAT_PATH)) == 2
    state = json.loads(budget_file.read_text())
    assert [row["cost_usd"] for row in state["calls"]] == ["0.02", "0.02"]


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize(
    "defect",
    ["models", "calls", "reservation", "identity", "duplicate", "missing_cost", "overcharge"],
)
def test_malformed_budget_refuses_http_with_or_without_optimization(
    local_http_server, budget_file, optimized, defect
):
    state = json.loads(budget_file.read_text())
    record = {"id": "prior", "reserved_usd": "0.14", "cost_usd": None}
    if defect == "models":
        state["models"] = []
    elif defect == "calls":
        state["calls"] = {}
    elif defect == "reservation":
        state["calls"] = [0]
    elif defect == "identity":
        state["calls"] = [{**record, "id": 0}]
    elif defect == "duplicate":
        state["calls"] = [record, record]
    elif defect == "missing_cost":
        state["calls"] = [{"id": "prior", "reserved_usd": "0.14"}]
    else:
        state["calls"] = [{**record, "cost_usd": "0.15"}]
    original = json.dumps(state)
    budget_file.write_text(original)
    result = _budget_subprocess(local_http_server, optimized)
    assert f"OPTIMIZED={int(optimized)}" in result.stdout
    assert result.stdout.count("REFUSED=") == 1
    assert "ACCEPTED=" not in result.stdout
    assert local_http_server.requests_for(_CHAT_PATH) == ()
    assert budget_file.read_text() == original


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("cost", [None, "NaN", 0.15])
def test_ambiguous_or_excessive_cost_remains_closed_under_optimization(
    local_http_server, budget_file, optimized, cost
):
    body = _chat_completion_body(
        "cannot release unverified funds", prompt_tokens=3, served_model="x"
    )
    body["usage"]["cost"] = cost
    local_http_server.enqueue_json(_CHAT_PATH, body)
    result = _budget_subprocess(local_http_server, optimized, calls=2)
    assert result.stdout.count("REFUSED=") == 2
    assert "ACCEPTED=" not in result.stdout
    assert len(local_http_server.requests_for(_CHAT_PATH)) == 1
    state = json.loads(budget_file.read_text())
    assert len(state["calls"]) == 1
    if cost == 0.15:
        assert state["blocked"] is True
        assert state["calls"][0]["cost_usd"] == "0.15"
    else:
        assert state["calls"][0]["cost_usd"] is None


@pytest.mark.timeout(60)
def test_retry_requires_a_second_reservation_before_sending(
    local_http_server, budget_file, _no_cache
) -> None:
    local_http_server.enqueue_disconnect(_CHAT_PATH)
    with pytest.raises(OpenRouterBudgetError, match="exhausted before HTTP"):
        _budgeted_call(local_http_server)
    assert len(local_http_server.requests_for(_CHAT_PATH)) == 1
    assert len(json.loads(budget_file.read_text())["calls"]) == 1


def _allow_three_budgeted_attempts(budget_file, monkeypatch):
    state = json.loads(budget_file.read_text())
    state["limit_usd"] = "0.50"
    budget_file.write_text(json.dumps(state))
    monkeypatch.setenv(BUDGET_LIMIT_ENV, "0.50")


def _enqueue_rate_limit(server, *, repeat=1):
    server.enqueue_bytes(
        _CHAT_PATH,
        b'{"error": {"message": "engine_overloaded"}}',
        status=429,
        content_type="application/json",
        headers={"Retry-After": "0"},
        repeat=repeat,
    )


@pytest.mark.timeout(10)
def test_http_429_retries_identical_capped_payload_and_reserves_each_attempt(
    local_http_server, budget_file, monkeypatch, _no_cache
) -> None:
    _allow_three_budgeted_attempts(budget_file, monkeypatch)
    _enqueue_rate_limit(local_http_server)
    body = _chat_completion_body("recovered", prompt_tokens=3, served_model="x")
    body["usage"]["cost"] = 0.02
    local_http_server.enqueue_json(_CHAT_PATH, body)
    assert _budgeted_call(local_http_server) == "recovered"
    requests = local_http_server.requests_for(_CHAT_PATH)
    assert len(requests) == 2
    assert requests[0].body == requests[1].body
    assert json.loads(requests[0].body)["provider"]["only"] == ["test-provider"]
    calls = json.loads(budget_file.read_text())["calls"]
    assert [call["reserved_usd"] for call in calls] == ["0.14", "0.14"]
    assert [call["cost_usd"] for call in calls] == [None, "0.02"]


@pytest.mark.timeout(10)
def test_http_429_exhaustion_retains_every_attempt_reservation(
    local_http_server, budget_file, monkeypatch, _no_cache
) -> None:
    _allow_three_budgeted_attempts(budget_file, monkeypatch)
    _enqueue_rate_limit(local_http_server, repeat=3)
    with pytest.raises(RuntimeError, match="OpenRouter API returned 429.*engine_overloaded"):
        _budgeted_call(local_http_server)
    assert len(local_http_server.requests_for(_CHAT_PATH)) == 3
    calls = json.loads(budget_file.read_text())["calls"]
    assert len(calls) == 3
    assert all(call["reserved_usd"] == "0.14" and call["cost_usd"] is None for call in calls)


@pytest.mark.timeout(10)
def test_http_429_retry_cannot_exceed_remaining_budget(local_http_server, budget_file, _no_cache):
    _enqueue_rate_limit(local_http_server)
    with pytest.raises(OpenRouterBudgetError, match="exhausted before HTTP"):
        _budgeted_call(local_http_server)
    assert len(local_http_server.requests_for(_CHAT_PATH)) == 1
    assert json.loads(budget_file.read_text())["calls"][0]["cost_usd"] is None


@pytest.mark.timeout(10)
def test_http_429_followed_by_invalid_cost_fails_without_another_retry(
    local_http_server, budget_file, monkeypatch, _no_cache
) -> None:
    _allow_three_budgeted_attempts(budget_file, monkeypatch)
    _enqueue_rate_limit(local_http_server)
    body = _chat_completion_body("unpriced", prompt_tokens=3, served_model="x")
    body["usage"]["cost"] = "NaN"
    local_http_server.enqueue_json(_CHAT_PATH, body)
    with pytest.raises(OpenRouterBudgetError, match="usage.cost"):
        _budgeted_call(local_http_server)
    assert len(local_http_server.requests_for(_CHAT_PATH)) == 2
    calls = json.loads(budget_file.read_text())["calls"]
    assert len(calls) == 2 and all(call["cost_usd"] is None for call in calls)


@pytest.mark.parametrize(
    "header,expected",
    [
        (None, 20.0),
        ("", 20.0),
        ("0", 0.0),
        (" 2 ", 2.0),
        ("60", 60.0),
        ("999999", 60.0),
        ("-1", 20.0),
        ("NaN", 20.0),
        ("tomorrow", 20.0),
        ("Thu, 01 Jan 1970 00:00:10 GMT", 10.0),
        ("Wed, 31 Dec 1969 23:59:59 GMT", 0.0),
        ("Thu, 01 Jan 1970 00:02:00 GMT", 60.0),
        ("Thu, 01 Jan 1970 00:00:10", 20.0),
    ],
)
def test_retry_after_delay_is_finite_bounded_and_honors_valid_headers(header, expected):
    from ailm.benchmarks.openrouter import _retry_after_seconds

    assert _retry_after_seconds(header, now=0.0) == expected
