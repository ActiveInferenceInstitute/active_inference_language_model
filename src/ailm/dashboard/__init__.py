"""Public dashboard API, loaded when a caller requests an entry point."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ailm.dashboard.engine import infer_sentence_hierarchy
    from ailm.dashboard.server import run_dashboard

__all__ = ["infer_sentence_hierarchy", "run_dashboard"]

_EXPORT_MODULES = {
    "infer_sentence_hierarchy": "ailm.dashboard.engine",
    "run_dashboard": "ailm.dashboard.server",
}


def __getattr__(name: str) -> Any:
    """Resolve the original function object without importing unused engines."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
