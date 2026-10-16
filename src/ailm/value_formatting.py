"""Shared ledger-value formatting for documentation and manuscript tokens.

This boundary uses only the standard library. Reading an already recorded value
must not initialize corpus scorers, model agents, or a numerical runtime.
"""

from __future__ import annotations

import json
from typing import Any


class ManuscriptVariableError(RuntimeError):
    """Base error for a variable map that cannot be generated unambiguously."""


def format_ledger_value(value: Any) -> str:
    """Render a ledger value for prose.

    Integers stay integral (a position count of ``42820`` must not appear as
    ``42820.0``). Non-integral floats use Python's shortest round-tripping
    representation, which preserves the recorded binary value rather than the
    six-significant-digit default of ``:g``. Very small p-values naturally stay
    in scientific notation.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}" if abs(value) >= 10000 else str(value)
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return repr(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    raise ManuscriptVariableError(
        f"manuscript values must be JSON scalars, lists, or objects; got {type(value).__name__}"
    )
