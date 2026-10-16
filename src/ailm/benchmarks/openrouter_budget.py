"""Process-shared reservations for explicitly budgeted OpenRouter runs.

The runner supplies a fresh ledger with pinned provider pricing and token bounds.
Every HTTP attempt reserves its worst-case charge before transmission. Unknown
charges retain that reservation; only explicit API usage costs release money.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TypeAlias
from uuid import uuid4

Json: TypeAlias = str | int | float | bool | None | list["Json"] | dict[str, "Json"]
BUDGET_FILE_ENV = "AILM_OPENROUTER_BUDGET_FILE"
BUDGET_LIMIT_ENV = "AILM_OPENROUTER_BUDGET_LIMIT_USD"
SCHEMA = "ailm-openrouter-budget-v1"


class OpenRouterBudgetError(RuntimeError):
    """A budget cannot authorize a request or reconcile its charge."""


def parse_budget_json(text: str) -> dict[str, Json]:
    """Refuse ambiguous accounting fields instead of keeping their last occurrence."""

    def unique_object(pairs: list[tuple[str, Json]]) -> dict[str, Json]:
        result: dict[str, Json] = {}
        for key, value in pairs:
            if key in result:
                raise OpenRouterBudgetError(f"duplicate JSON key {key!r} in budget accounting")
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise OpenRouterBudgetError("budget accounting requires a JSON object")
    return value


def _money(value: Json, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise OpenRouterBudgetError(f"{label} must be a finite nonnegative USD amount")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise OpenRouterBudgetError(f"{label} is not a decimal USD amount") from exc
    if not amount.is_finite() or amount < 0:
        raise OpenRouterBudgetError(f"{label} must be a finite nonnegative USD amount")
    return amount


def _positive_int(value: Json, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise OpenRouterBudgetError(f"{label} must be a positive integer")
    return value


def _price_number(amount: Decimal) -> float:
    number = float(amount)
    if not math.isfinite(number) or (amount > 0 and number == 0):
        raise OpenRouterBudgetError("provider price is outside JSON numeric range")
    if Decimal(str(number)) > amount:
        number = math.nextafter(number, 0.0)
    return number


@dataclass(frozen=True, slots=True)
class BudgetRequest:
    path: Path
    limit: Decimal
    model: str
    provider: str
    context_length: int
    max_tokens: int
    prompt_price: float
    completion_price: float
    request_price: float
    price_identity: str

    @property
    def reservation(self) -> Decimal:
        return (
            Decimal(str(self.prompt_price)) * self.context_length / 1_000_000
            + Decimal(str(self.completion_price)) * self.max_tokens / 1_000_000
            + Decimal(str(self.request_price))
        )

    def provider_options(self) -> dict[str, Json]:
        return {
            "only": [self.provider],
            "allow_fallbacks": False,
            "require_parameters": True,
            "max_price": {
                "prompt": self.prompt_price,
                "completion": self.completion_price,
                "request": self.request_price,
            },
        }

    def reserve(self) -> str:
        with _locked(self.path) as state:
            _validate_ledger(state, self.limit)
            prices = state["models"]
            if not isinstance(prices, dict):
                raise OpenRouterBudgetError("budget models must be an object")
            if _identity(prices.get(self.model)) != self.price_identity:
                raise OpenRouterBudgetError("pinned provider pricing changed during the run")
            calls = state["calls"]
            if not isinstance(calls, list):
                raise OpenRouterBudgetError("budget calls must be a list")
            total = Decimal(0)
            for call in calls:
                if not isinstance(call, dict):
                    raise OpenRouterBudgetError("malformed reservation")
                cost = call["cost_usd"]
                total += _money(call["reserved_usd"] if cost is None else cost, "call charge")
            if total + self.reservation > self.limit:
                raise OpenRouterBudgetError("OpenRouter budget exhausted before HTTP request")
            identity = uuid4().hex
            calls.append(
                {
                    "id": identity,
                    "model": self.model,
                    "reserved_usd": str(self.reservation),
                    "cost_usd": None,
                }
            )
            _save(self.path, state)
            return identity

    def settle(self, identity: str, response: Mapping[str, Json]) -> None:
        usage = response.get("usage")
        if not isinstance(usage, dict) or "cost" not in usage:
            raise OpenRouterBudgetError("OpenRouter usage.cost missing; reservation retained")
        cost = _money(usage["cost"], "usage.cost")
        with _locked(self.path) as state:
            _validate_ledger(state, self.limit)
            calls = state["calls"]
            if not isinstance(calls, list):
                raise OpenRouterBudgetError("budget calls must be a list")
            matches = [call for call in calls if isinstance(call, dict) and call["id"] == identity]
            if len(matches) != 1 or matches[0]["cost_usd"] is not None:
                raise OpenRouterBudgetError("reservation missing or already reconciled")
            matches[0]["cost_usd"] = str(cost)
            exceeded = cost > _money(matches[0]["reserved_usd"], "reservation")
            if exceeded:
                state["blocked"] = True
            _save(self.path, state)
            if exceeded:
                raise OpenRouterBudgetError("API cost exceeded reserved bound; budget blocked")


def _identity(value: Json) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def budget_for(payload: Mapping[str, Json]) -> BudgetRequest | None:
    location, limit_text = os.environ.get(BUDGET_FILE_ENV), os.environ.get(BUDGET_LIMIT_ENV)
    if location is None and limit_text is None:
        return None
    if not location or not limit_text:
        raise OpenRouterBudgetError(
            "budget file and independently configured USD limit are required"
        )
    limit = _money(limit_text, BUDGET_LIMIT_ENV)
    if limit == 0:
        raise OpenRouterBudgetError("budget limit must be positive")
    if set(payload) - {"model", "messages", "max_tokens", "temperature", "seed"}:
        raise OpenRouterBudgetError("budgeted calls accept only bounded text chat payloads")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise OpenRouterBudgetError("budgeted calls require text messages")
    for message in messages:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] != "user"
            or not isinstance(message["content"], str)
        ):
            raise OpenRouterBudgetError("budgeted calls require plain user text messages")
    model = payload.get("model")
    if not isinstance(model, str):
        raise OpenRouterBudgetError("budgeted model must be an explicitly pinned slug")
    path = Path(location)
    with _locked(path) as state:
        _validate_ledger(state, limit)
        models = state["models"]
        if not isinstance(models, dict):
            raise OpenRouterBudgetError("budget models must be an object")
        price = models.get(model)
        if not isinstance(price, dict):
            raise OpenRouterBudgetError(f"no pinned provider price for {model!r}")
        provider, source = price.get("provider"), price.get("source_sha256")
        if (
            not isinstance(provider, str)
            or not provider
            or not isinstance(source, str)
            or (len(source) != 64 or any(c not in "0123456789abcdef" for c in source))
        ):
            raise OpenRouterBudgetError("provider identity and metadata SHA-256 are required")
        maximum = _positive_int(price.get("max_completion_tokens"), "max_completion_tokens")
        tokens = _positive_int(payload.get("max_tokens"), "max_tokens")
        if tokens > maximum:
            raise OpenRouterBudgetError("max_tokens exceeds pinned provider output bound")
        return BudgetRequest(
            path,
            limit,
            model,
            provider,
            _positive_int(price.get("context_length"), "context_length"),
            tokens,
            _price_number(_money(price.get("prompt_usd_per_token"), "prompt price") * 1_000_000),
            _price_number(
                _money(price.get("completion_usd_per_token"), "completion price") * 1_000_000
            ),
            _price_number(_money(price.get("request_usd"), "request price")),
            _identity(price),
        )


def configured_provider_options(model: str) -> dict[str, Json] | None:
    """Snapshot the run's route for its receipt without reserving or sending a call."""
    budget = budget_for(
        {
            "model": model,
            "messages": [{"role": "user", "content": ""}],
            "max_tokens": 1,
        }
    )
    return None if budget is None else budget.provider_options()


def validate_provider_options(value: Json) -> dict[str, Json] | None:
    """Validate a receipt's route independently of the live budget environment."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "only",
        "allow_fallbacks",
        "require_parameters",
        "max_price",
    }:
        raise OpenRouterBudgetError("invalid recorded OpenRouter provider constraints")
    providers, prices = value["only"], value["max_price"]
    if (
        value["allow_fallbacks"] is not False
        or value["require_parameters"] is not True
        or not isinstance(providers, list)
        or len(providers) != 1
        or not isinstance(providers[0], str)
        or not providers[0]
        or not isinstance(prices, dict)
        or set(prices) != {"prompt", "completion", "request"}
    ):
        raise OpenRouterBudgetError("recorded provider constraints must pin one capped route")
    for name, amount in prices.items():
        if not isinstance(amount, (int, float)):
            raise OpenRouterBudgetError("recorded provider prices must be JSON numbers")
        _money(amount, f"recorded provider {name} price")
    return parse_budget_json(json.dumps(value, allow_nan=False))


def _validate_ledger(state: dict[str, Json], limit: Decimal) -> None:
    if state.get("schema") != SCHEMA or _money(state.get("limit_usd"), "ledger limit") != limit:
        raise OpenRouterBudgetError("budget schema or independently configured limit disagrees")
    if state.get("blocked") is not False:
        raise OpenRouterBudgetError("budget blocked or missing its blocked flag")
    calls = state.get("calls")
    if not isinstance(state.get("models"), dict) or not isinstance(calls, list):
        raise OpenRouterBudgetError("budget models and calls must be present")
    identities: set[str] = set()
    for call in calls:
        if not isinstance(call, dict):
            raise OpenRouterBudgetError("malformed reservation")
        identity = call.get("id")
        if not isinstance(identity, str):
            raise OpenRouterBudgetError("malformed reservation identity")
        if identity in identities or "cost_usd" not in call:
            raise OpenRouterBudgetError("duplicate or incomplete reservation")
        identities.add(identity)
        reserved = _money(call.get("reserved_usd"), "reservation")
        if call["cost_usd"] is not None and _money(call["cost_usd"], "cost") > reserved:
            raise OpenRouterBudgetError("stored API cost exceeds reservation")


@contextmanager
def _locked(path: Path) -> Iterator[dict[str, Json]]:
    try:
        import fcntl
    except ImportError as exc:
        raise OpenRouterBudgetError("budgeted calls require POSIX process locking") from exc
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            state = parse_budget_json(path.read_text(encoding="utf-8"))
            yield state
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _save(path: Path, state: dict[str, Json]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as out:
        json.dump(state, out, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
        temporary = out.name
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
