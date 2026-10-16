"""Transport to a local Ollama server, with the content-addressed call cache."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ailm.benchmarks.constants import (
    DEFAULT_SEED,
    OLLAMA_NUM_CTX,
    OLLAMA_PROMPT_BYTE_TOKEN_FACTOR,
    OLLAMA_TEMPLATE_TOKEN_RESERVE,
    OLLAMA_URL,
    OLLAMA_USE_MMAP,
    PROTOCOL_VERSION,
    TOP_LOGPROBS,
)
from ailm.benchmarks.prompts import SYNTHETIC_LETTER_INVENTORY_SYMBOLS
from ailm.call_cache import CacheRequest, CallCache, default_cache_path, model_digest
from ailm.levels.letter import ALPHA_LIST, ALPHA_TO_IDX
from ailm.levels.phoneme import (
    PHONES,
)
from ailm.loopback_http import (
    LoopbackTransportError,
    loopback_json_request,
    validate_loopback_endpoint,
)

CALL_CACHE_PATH_ENV = "AILM_CALL_CACHE_PATH"


def uses_current_ollama_loader(payload: Mapping[str, object]) -> bool:
    """Require the explicit publication loader and context choices, without JSON coercion."""
    options = payload.get("options")
    return (
        isinstance(options, dict)
        and options.get("use_mmap") is OLLAMA_USE_MMAP
        and type(options.get("num_ctx")) is int
        and options["num_ctx"] == OLLAMA_NUM_CTX
    )


def current_ollama_cache_entries(
    store: CallCache, *, endpoint: str = OLLAMA_URL
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Filter authenticated endpoint identity before current-population sampling."""
    origin = validate_loopback_endpoint(endpoint)
    candidates = [entry for entry in store.entries() if uses_current_ollama_loader(entry[1])]
    if not candidates:
        return []
    endpoints = store.provenance_binding({entry[0] for entry in candidates})["endpoints"]
    if endpoints == [origin]:
        return candidates
    if origin not in endpoints:
        return []
    return [
        entry
        for entry in candidates
        if store.provenance_binding({entry[0]})["endpoints"] == [origin]
    ]


def default_context_token_bound(prompt: str, num_predict: int) -> int:
    """Conservative registered-prompt bound, including response and template reserve.

    Byte BPE encoding and fallback use at most two bytes per input byte;
    merges only concatenate those symbols. The shipped
    SentencePiece path can expand spaces into a multibyte marker before byte
    fallback, so the byte multiplier covers that expansion. The separate
    reserve covers the inspected registered templates, special tokens and
    prefix insertion; it is not a universal claim about arbitrary model templates.
    """
    return (
        OLLAMA_PROMPT_BYTE_TOKEN_FACTOR * len(prompt.encode("utf-8"))
        + num_predict
        + OLLAMA_TEMPLATE_TOKEN_RESERVE
    )


def _project_root(source: Path | None = None) -> Path:
    """The repository root, found by walking up rather than counting hops.

    A ``parents[N]`` count is wrong the moment a module changes depth, and it
    fails in the least useful way: ``src/vendor/...`` is a plausible-looking
    path, so callers report "no such file" for a corpus that never moved. This
    broke exactly that way when ``benchmarks.py`` became ``benchmarks/``.

    Raises:
        RuntimeError: If no ancestor holds both ``vendor/`` and ``src/``.
    """
    source = source or Path(__file__)
    for candidate in source.resolve().parents:
        if (candidate / "vendor").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError(f"no project root above {source} contains both vendor/ and src/")


def ollama_available(model: str, *, endpoint: str = OLLAMA_URL) -> bool:
    """True when the Ollama endpoint answers /api/tags and serves the model.

    The probe's contract is reachability, not speed: a freshly started (or
    freshly idle) server scans the models directory before its first
    ``/api/tags`` answer, a measured 13.4 s cold path on this workstation, so
    the timeout must clear that scan or a healthy endpoint reads as absent.
    """
    try:
        body = loopback_json_request(endpoint, "/api/tags", timeout=30)
        if not isinstance(body, dict) or not isinstance(body.get("models"), list):
            return False
        names = {str(item.get("name", "")) for item in body["models"] if isinstance(item, dict)}
        return model in names or model.split(":", 1)[0] in {name.split(":", 1)[0] for name in names}
    except (OSError, ValueError, LoopbackTransportError):
        return False


def _response_format(task: str) -> dict[str, Any]:
    schema: dict[str, Any]
    if task == "letter":
        schema = {"type": "string", "enum": sorted(ALPHA_LIST)}
    elif task == "synthetic_letter":
        schema = {"type": "string", "enum": sorted(SYNTHETIC_LETTER_INVENTORY_SYMBOLS)}
    elif task == "phoneme":
        schema = {"type": "string", "enum": sorted(PHONES)}
    elif task == "g2p":
        schema = {
            "type": "array",
            "items": {"type": "string", "enum": sorted(PHONES)},
            "minItems": 1,
        }
    else:
        raise ValueError(f"unknown task {task!r}")
    return {
        "type": "object",
        "properties": {"answer": schema},
        "required": ["answer"],
        "additionalProperties": False,
    }


def _generate_payload(
    model: str,
    prompt: str,
    task: str,
    num_predict: int,
    seed: int,
    num_ctx: int | None = None,
) -> dict[str, Any]:
    if isinstance(num_predict, bool) or not isinstance(num_predict, int) or num_predict <= 0:
        raise ValueError("num_predict must be positive")
    effective_context = OLLAMA_NUM_CTX if num_ctx is None else num_ctx
    if isinstance(effective_context, bool) or not isinstance(effective_context, int):
        raise ValueError("num_ctx must be an integer")
    if effective_context <= num_predict:
        raise ValueError("num_ctx must be larger than num_predict")
    if num_ctx is None and default_context_token_bound(prompt, num_predict) > effective_context:
        raise ValueError("default local context budget exceeded; no request sent")
    options = {
        "temperature": 0,
        "seed": seed,
        "num_predict": num_predict,
        "use_mmap": OLLAMA_USE_MMAP,
        "num_ctx": effective_context,
    }
    return {
        "model": model,
        "prompt": prompt,
        "format": _response_format(task),
        "stream": False,
        "options": options,
    }


def post_generate_uncached(
    payload: dict[str, Any], *, endpoint: str = OLLAMA_URL
) -> dict[str, Any]:
    """One request to the server. The only function here that touches the network."""
    body = loopback_json_request(
        endpoint,
        "/api/generate",
        payload=payload,
        timeout=1800,
    )
    if not isinstance(body, dict):
        raise LoopbackTransportError("Ollama /api/generate response must be a JSON object")
    return dict(body)


def warm_ollama_model(
    model: str,
    prompt: str,
    task: str,
    num_predict: int,
    *,
    endpoint: str = OLLAMA_URL,
    num_ctx: int | None = None,
) -> str:
    """Issue one deliberately uncached generation before a scored call series.

    Availability and model-identity endpoints do not load the same generation
    path that the benchmark scores.  A cached scored request is not a warm-up
    either.  This helper therefore uses the exact deterministic generation
    payload while bypassing the call store, and refuses a malformed server
    envelope before an arm-order comparison begins.
    """

    if num_predict <= 0:
        raise ValueError("warm-up num_predict must be positive")
    body = post_generate_uncached(
        _generate_payload(
            model,
            prompt,
            task,
            num_predict,
            DEFAULT_SEED,
            num_ctx,
        ),
        endpoint=endpoint,
    )
    response = body.get("response")
    if not isinstance(response, str):
        raise LoopbackTransportError("Ollama warm-up response has no string 'response' field")
    return response


#: The process-wide call store, or ``None`` when caching is off. Every model call
#: in this module funnels through :func:`_post_generate`, so this one object is
#: the whole memo: the three ``ask_ollama*`` functions differ in what they read
#: out of the reply, not in how they obtain it.
_CALL_CACHE: CallCache | None = None

#: Digest per model tag, resolved once per process. ``None`` records a model whose
#: identity could not be read, so the lookup is not retried on every position of
#: a census while the server keeps failing to answer it.
_MODEL_IDENTITY: dict[tuple[str, str], str | None] = {}


def call_cache() -> CallCache | None:
    """The active store, if caching is enabled for this process."""
    return _CALL_CACHE


def enable_call_cache(
    path: Path | None = None,
    *,
    enabled: bool = True,
    require_default_path: bool = False,
) -> CallCache | None:
    """Turn the call store on or off for this process.

    Off by default, and deliberately so. A cache that switches itself on is a
    cache that can answer a question nobody knew it was answering; a runner that
    wants resumability asks for it, and every other caller — tests, ad-hoc
    scripts, the oracle — keeps the previous behavior exactly.

    ``AILM_CALL_CACHE_PATH`` gives tests and validation runs an explicit,
    process-local destination without changing a recorder's CLI. Both that
    environment value and an explicit ``path`` argument must be absolute: a
    working-directory-relative store is not a stable provenance input. An
    explicit argument takes precedence, which lets a caller establish isolation
    even when its parent process exported a broader default.

    Publication recorders pass ``require_default_path=True``. That closes the
    isolation escape hatch at the publication boundary: a noncanonical store
    cannot be measured while the receipt validator later reads the repository's
    default store.
    """
    global _CALL_CACHE
    if not enabled:
        _CALL_CACHE = None
        return None
    root = _project_root()
    canonical_path = default_cache_path(root)
    configured = os.environ.get(CALL_CACHE_PATH_ENV)
    selected = path if path is not None else Path(configured) if configured is not None else None
    if selected is None:
        selected = canonical_path
    elif not selected.is_absolute():
        origin = (
            CALL_CACHE_PATH_ENV if path is None and configured is not None else "call-cache path"
        )
        raise ValueError(f"{origin} must be an absolute path, got {str(selected)!r}")

    # Normalize ``..`` without resolving symlinks. CallCache performs the
    # descriptor-level no-follow checks; resolving here would hide the lexical
    # path whose trust boundary it is supposed to validate.
    selected = Path(os.path.abspath(selected))
    if require_default_path and selected != canonical_path:
        raise ValueError(
            "publication call-cache path must be the canonical repository store "
            f"{canonical_path}; got {selected}"
        )
    _CALL_CACHE = CallCache(path=selected)
    return _CALL_CACHE


def _identity_for(model: str, endpoint: str) -> str | None:
    """This model's content digest, resolved once and remembered.

    ``None`` propagates as "do not cache", which is why it is stored rather than
    treated as a cache miss on the identity lookup itself: retrying ``/api/show``
    for every one of claim:bench_letter_positions positions against a server
    that cannot answer it would turn a degraded cache into a slow one.
    """
    key = (endpoint.rstrip("/"), model)
    if key not in _MODEL_IDENTITY:
        _MODEL_IDENTITY[key] = model_digest(model, key[0])
    return _MODEL_IDENTITY[key]


def resolved_model_identity(model: str, *, endpoint: str = OLLAMA_URL) -> str | None:
    """Return the exact identity the call-cache key will use for ``model``."""
    return _identity_for(model, endpoint.rstrip("/"))


def _post_generate(
    payload: dict[str, Any],
    *,
    endpoint: str = OLLAMA_URL,
    cache_protocol: str = PROTOCOL_VERSION,
    cache_read: bool = True,
) -> dict[str, Any]:
    """A request, answered from the store when the identical one was made before.

    The key covers the payload, the endpoint, the protocol version and the
    model's digest, so a reply is replayed only for a question that is identical
    in every respect that could change it. An unresolvable digest yields no key,
    and a call with no key is measured and not stored — an unknown model identity
    degrades to no caching rather than to caching under a guess.
    """
    cache = _CALL_CACHE
    if cache is None:
        return post_generate_uncached(payload, endpoint=endpoint)
    normalized_endpoint = endpoint.rstrip("/")
    identity = _identity_for(str(payload.get("model", "")), normalized_endpoint)
    request = (
        None
        if identity is None
        else CacheRequest(
            payload=payload,
            endpoint=normalized_endpoint,
            protocol=cache_protocol,
            identity=identity,
        )
    )
    if cache_read:
        hit = cache.get(request)
        if hit is not None:
            return hit
    cache.stats.misses += 1
    body = post_generate_uncached(payload, endpoint=normalized_endpoint)
    cache.put(request, body)
    return body


def ask_ollama(
    model: str,
    prompt: str,
    task: str,
    num_predict: int = 4,
    seed: int = DEFAULT_SEED,
    *,
    endpoint: str = OLLAMA_URL,
    cache_protocol: str = PROTOCOL_VERSION,
    cache_read: bool = True,
    num_ctx: int | None = None,
) -> str:
    """Send one generate request and return its response text, cache-aware."""
    payload = _generate_payload(model, prompt, task, num_predict, seed, num_ctx)
    return str(
        _post_generate(
            payload,
            endpoint=endpoint,
            cache_protocol=cache_protocol,
            cache_read=cache_read,
        ).get("response", "")
    )


#: The JSON envelope preceding the answer slot, with insignificant whitespace
#: removed. Models differ in whether they emit ``{"answer":"`` or
#: ``{"answer": "``, and both are the same JSON.
_ANSWER_PREFIX = '{"answer":"'


def _answer_token_index(tokens: list[str]) -> int | None:
    """Index of the token that carries the ``answer`` value, or ``None``.

    The answer slot is the token beginning where the accumulated reply equals
    the JSON envelope. Matching on the accumulated string rather than a fixed
    index keeps this correct across tokenisers, and whitespace inside the
    envelope is ignored because it is not part of the JSON: a model that emits
    ``{"answer": "`` would otherwise never match, and the distribution for every
    one of its positions would silently fall back to uniform.
    """
    prefix = ""
    for index, token in enumerate(tokens):
        if prefix == _ANSWER_PREFIX:
            return index
        prefix += "".join(token.split())
        if len(prefix) > 32:
            break
    return None


def observed_symbol_mass(candidates: list[dict[str, Any]]) -> dict[str, float]:
    """Per-symbol probability the server actually reported, before any completion.

    This is the raw material both completion rules start from, so it is factored
    out rather than reimplemented beside them: a second copy of the token-to-
    symbol folding could drift from the one the shipped reconstruction uses, and
    a rule comparison built on a different reading of the same candidate list
    would measure the drift instead of the rules.

    Symbols absent from the list are absent here too — completing them is what
    the rules disagree about. An empty result means the reply's answer slot put
    no reported mass on the inventory at all.
    """
    observed: dict[str, float] = {}
    for candidate in candidates:
        symbol = _candidate_symbol(str(candidate.get("token", "")))
        if symbol is None:
            continue
        probability = math.exp(float(candidate.get("logprob", -math.inf)))
        # Two tokens can spell the same answer — the bare symbol and the symbol
        # merged with the closing quote — and they are distinct paths to one
        # outcome, so their probabilities add.
        observed[symbol] = observed.get(symbol, 0.0) + probability
    return observed


def _distribution_from_candidates(candidates: list[dict[str, Any]]) -> dict[str, float]:
    """Turn the server's ``top_logprobs`` list into a distribution over ALPHABET.

    Two facts shape the arithmetic. The candidate list is the *raw* next-token
    distribution rather than the schema-masked one, so most of its mass sits on
    tokens the response grammar forbids; that mass belongs to no inventory
    symbol and is discarded, not redistributed. And the list is truncated at
    ``top_logprobs``, which is below the inventory size, so some inventory
    symbols are missing entirely. A missing symbol fell outside the top-k, so
    its probability is at most the smallest reported one; each is assigned that
    bound and the whole thing is renormalized. That is conservative in the only
    direction available — it can overstate an unreported symbol, never
    understate it — and it keeps the completion from dominating the result the
    way spreading the discarded mass would.
    """
    observed = observed_symbol_mass(candidates)
    if not observed:
        return {symbol: 1.0 / len(ALPHA_LIST) for symbol in ALPHA_LIST}
    mass = sum(observed.values())
    missing = [symbol for symbol in ALPHA_LIST if symbol not in observed]
    if not missing:
        return {symbol: value / mass for symbol, value in observed.items()}
    bound = min(observed.values())
    combined = {**observed, **{symbol: bound for symbol in missing}}
    total = sum(combined.values())
    return {symbol: value / total for symbol, value in combined.items()}


def _candidate_symbol(token: str) -> str | None:
    """The answer symbol a candidate token would produce, or ``None``.

    The answer slot sits inside a JSON string, and a tokeniser is free to merge
    the symbol with the closing quote: ``."`` is the token that answers ``.``.
    Matching candidates against the inventory literally therefore drops the very
    token the model emitted, which both understates the observed mass and lets
    the reconstructed distribution disagree with the scored answer — calibration
    would then describe a decision the benchmark never scored. Everything up to
    the string terminator is the answer; anything longer than one symbol is a
    word the response grammar could not have emitted here.
    """
    symbol = token.split('"', 1)[0]
    return symbol if len(symbol) == 1 and symbol in ALPHA_TO_IDX else None


def captured_mass(candidates: list[dict[str, Any]]) -> float:
    """Probability mass the server actually reported over the scored inventory.

    The candidate list is the raw next-token distribution, not the
    schema-masked one: it carries multi-character tokens the grammar could
    never emit, and it is truncated at ``top_logprobs``. Whatever those two
    facts leave uncaptured is filled in by assumption, so the share that was
    genuinely observed is recorded next to every calibration number rather than
    left for a reader to wonder about.
    """
    return sum(
        math.exp(float(candidate.get("logprob", -math.inf)))
        for candidate in candidates
        if _candidate_symbol(str(candidate.get("token", ""))) is not None
    )


def ask_ollama_answer_candidates(
    model: str,
    prompt: str,
    task: str,
    num_predict: int = 32,
    seed: int = DEFAULT_SEED,
    top_logprobs: int = TOP_LOGPROBS,
    *,
    endpoint: str = OLLAMA_URL,
) -> tuple[str, list[dict[str, Any]]]:
    """Return the reply text and the server's candidate list for the answer slot.

    The request is the scored request plus ``logprobs``: same endpoint, same
    schema, same temperature and seed, so the candidates belong to the same
    decode the accuracy numbers come from rather than to a second system.

    The candidate list is returned rather than only the reconstruction built from
    it because more than one completion rule can be applied to it, and comparing
    two rules on one position requires the position's own candidates. An empty
    list means the answer slot was never located in the token stream; callers
    reconstruct from it exactly as they would from a located slot that reported
    nothing on the inventory, because those two cases carry the same information.
    """
    payload = {
        **_generate_payload(model, prompt, task, num_predict, seed),
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }
    body = _post_generate(payload, endpoint=endpoint)
    text = str(body.get("response", ""))
    entries = body.get("logprobs") or []
    tokens = [str(entry.get("token", "")) for entry in entries]
    index = _answer_token_index(tokens)
    if index is None or index >= len(entries):
        return text, []
    return text, list(entries[index].get("top_logprobs") or [])


def ask_ollama_distribution(
    model: str,
    prompt: str,
    task: str,
    num_predict: int = 32,
    seed: int = DEFAULT_SEED,
    top_logprobs: int = TOP_LOGPROBS,
    *,
    endpoint: str = OLLAMA_URL,
) -> tuple[str, dict[str, float], float]:
    """Return the reply text, the answer-slot distribution, and its captured mass.

    A thin reading of :func:`ask_ollama_answer_candidates`: the reconstruction is
    the shipped min-bound rule and the mass is the share of it that was observed.
    An unlocated answer slot yields no candidates, which
    :func:`_distribution_from_candidates` reconstructs as the uniform column and
    :func:`captured_mass` scores as zero — the same pair this returned when it
    special-cased that branch itself.
    """
    text, candidates = ask_ollama_answer_candidates(
        model, prompt, task, num_predict, seed, top_logprobs, endpoint=endpoint
    )
    return text, _distribution_from_candidates(candidates), captured_mass(candidates)
