"""OpenRouter transport for scored calls to hosted models.

Mirrors :mod:`ailm.benchmarks.ollama`'s cache-aware discipline but speaks the
OpenAI-compatible chat completions API at ``https://openrouter.ai/api/v1``.
The response is ``choices[0].message.content`` — the same text the scorer
parses, just extracted from a different envelope.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime
from typing import Any

from ailm.benchmarks.ollama import DEFAULT_SEED
from ailm.benchmarks.openrouter_budget import BudgetRequest, budget_for, parse_budget_json
from ailm.call_cache import CacheRequest

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

_DEFAULT_TIMEOUT = 300
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 20
_MAX_RETRY_AFTER_SECONDS = 60
_CALL_CACHE = None


def set_call_cache(cache: Any) -> None:
    """Bind the module-level call cache (the same store ollama.py uses)."""
    global _CALL_CACHE
    _CALL_CACHE = cache


def _get_api_key() -> str:
    """Read the OpenRouter API key from the environment."""
    key = os.environ.get(OPENROUTER_API_KEY_ENV, "")
    if not key:
        raise ValueError(
            f"{OPENROUTER_API_KEY_ENV} is not set; the OpenRouter transport "
            "cannot make scored calls without it"
        )
    return key


def _chat_payload(
    model: str,
    prompt: str,
    task: str,
    num_predict: int,
    seed: int,
) -> dict[str, Any]:
    """OpenAI-compatible chat completions payload."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": num_predict,
        "temperature": 0,
        "seed": seed,
    }


def _identity_for(model: str) -> str:
    """The identity the cache key uses for an OpenRouter model slug."""
    digest = hashlib.sha256(f"openrouter:{model}".encode()).hexdigest()
    return f"sha256:{digest}"


def resolved_openrouter_identity(model: str) -> str:
    """Return the exact identity the call-cache key will use for ``model``."""
    return _identity_for(model)


def _retry_after_seconds(header: str | None, *, now: float) -> float:
    """Bound a rate-limit delay; malformed headers use the normal transport backoff."""
    if header is None:
        return float(_RETRY_BACKOFF_SECONDS)
    value = header.strip()
    if value.isascii() and value.isdigit():
        return min(float(value), float(_MAX_RETRY_AFTER_SECONDS))
    try:
        deadline = parsedate_to_datetime(value)
        if deadline.tzinfo is None:
            return float(_RETRY_BACKOFF_SECONDS)
        return min(max(deadline.timestamp() - now, 0.0), float(_MAX_RETRY_AFTER_SECONDS))
    except (TypeError, ValueError, OverflowError):
        return float(_RETRY_BACKOFF_SECONDS)


def _post_chat_raw(
    payload: dict[str, Any],
    *,
    api_key: str,
    endpoint: str = OPENROUTER_URL,
    budget: BudgetRequest | None = None,
) -> dict[str, Any]:
    """One raw HTTPS POST to the OpenRouter chat completions API.

    ``endpoint`` mirrors :func:`ailm.benchmarks.ollama.post_generate_uncached`'s
    injectable endpoint: it defaults to :data:`OPENROUTER_URL` so every existing
    caller and cache key is unchanged, and a test can point it at a loopback
    server instead of the network.

    Retries transient transport failures and HTTP 429 up to the attempt cap.
    Rate-limit responses honor Retry-After seconds or an HTTP date, bounded by
    the retry-delay cap; malformed headers use the normal transport backoff.
    Other HTTP refusals are immediate. Each budgeted attempt reserves separately,
    and failed attempts retain their reservations because their charge is unknown.
    """
    for attempt in range(_MAX_ATTEMPTS):
        reservation = budget.reserve() if budget is not None else None
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=_DEFAULT_TIMEOUT) as response:
                body = response.read(_MAX_RESPONSE_BYTES + 1)
                if len(body) > _MAX_RESPONSE_BYTES:
                    raise ValueError(f"OpenRouter response exceeds {_MAX_RESPONSE_BYTES} bytes")
                decoded = body.decode("utf-8")
                parsed = parse_budget_json(decoded) if budget is not None else json.loads(decoded)
        except urllib.error.HTTPError as exc:
            with exc:
                error_body = exc.read().decode("utf-8", errors="replace")[:500]
            if exc.code == 429 and attempt < _MAX_ATTEMPTS - 1:
                time.sleep(_retry_after_seconds(exc.headers.get("Retry-After"), now=time.time()))
                continue
            raise RuntimeError(f"OpenRouter API returned {exc.code}: {error_body}") from exc
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            if attempt == _MAX_ATTEMPTS - 1:
                raise RuntimeError("OpenRouter POST exhausted its retry attempts") from exc
            time.sleep(_RETRY_BACKOFF_SECONDS)
            continue
        if budget is not None and reservation is not None:
            budget.settle(reservation, parsed)
        return parsed
    raise RuntimeError("OpenRouter POST exhausted its retry attempts")


def _extract_response_text(body: dict[str, Any]) -> str:
    """Extract the completion text from an OpenAI-compatible response."""
    choices = body.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return str(message.get("content", ""))


def _response_identity(model: str, body: dict[str, Any]) -> str:
    """The identity recorded in the cache for this response."""
    served = body.get("model", model)
    digest = hashlib.sha256(f"openrouter:{model}:{served}".encode()).hexdigest()
    return f"sha256:{digest}"


def post_chat(
    payload: dict[str, Any],
    *,
    api_key: str,
    cache_protocol: str,
    cache_read: bool = True,
    endpoint: str = OPENROUTER_URL,
) -> dict[str, Any]:
    """Cache-aware chat completions call.

    The key covers the payload, the endpoint, the protocol version and the
    model identity, so a reply is replayed only for a question that is
    identical in every respect that could change it. ``endpoint`` defaults to
    :data:`OPENROUTER_URL`, so default behavior and existing cache keys are
    unchanged; an override changes the key exactly the way a different
    endpoint should.
    """
    budget = budget_for(payload)
    if budget is not None:
        payload = {**payload, "provider": budget.provider_options()}
    cache = _CALL_CACHE
    if cache is None:
        body = _post_chat_raw(payload, api_key=api_key, endpoint=endpoint, budget=budget)
        return {"response": _extract_response_text(body), "_raw": body}

    model = str(payload.get("model", ""))
    identity = _identity_for(model)
    request = CacheRequest(
        payload=payload,
        endpoint=endpoint,
        protocol=cache_protocol,
        identity=identity,
    )
    if cache_read:
        hit = cache.get(request)
        if hit is not None:
            return hit
    cache.stats.misses += 1
    raw_body = _post_chat_raw(payload, api_key=api_key, endpoint=endpoint, budget=budget)
    text = _extract_response_text(raw_body)
    # The receipt contract binds a server-reported prompt token count per call
    # (`maximum_prompt_eval_tokens` refuses anything else), so the OpenRouter
    # reply's usage block is carried under the same field name the Ollama
    # transport stores. A reply without it is a refused call, not a hole in the
    # receipt.
    usage = raw_body.get("usage")
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool) or prompt_tokens <= 0:
        raise RuntimeError(
            f"OpenRouter reply for {model!r} has no positive usage.prompt_tokens "
            "count; the receipt contract requires a server-reported prompt token "
            "count on every scored call"
        )
    body = {
        "response": text,
        "_raw": raw_body,
        "_identity": _response_identity(model, raw_body),
        "prompt_eval_count": prompt_tokens,
    }
    cache.put(request, body)
    return body


def ask_openrouter(
    model: str,
    prompt: str,
    task: str,
    num_predict: int = 32,
    seed: int = DEFAULT_SEED,
    *,
    api_key: str | None = None,
    cache_protocol: str = "openrouter-v1",
    cache_read: bool = True,
    endpoint: str = OPENROUTER_URL,
) -> str:
    """Send one chat completions request and return the response text.

    Cache-aware, mirroring :func:`ailm.benchmarks.ollama.ask_ollama`'s
    discipline, including its injectable ``endpoint`` (default
    :data:`OPENROUTER_URL`). The response text is what the scorer parses — the
    same strict JSON contract the Ollama transport feeds.
    """
    key = api_key or _get_api_key()
    payload = _chat_payload(model, prompt, task, num_predict, seed)
    return str(
        post_chat(
            payload,
            api_key=key,
            cache_protocol=cache_protocol,
            cache_read=cache_read,
            endpoint=endpoint,
        ).get("response", "")
    )
