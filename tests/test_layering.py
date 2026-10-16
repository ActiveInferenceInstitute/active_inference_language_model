"""Static layering guard: the declared core stays free of ``ailm.benchmarks``.

``ailm.benchmarks`` is model-and-network-adjacent (Ollama/OpenRouter probes,
recorder scaffolding); the modules checked here are declared core — generative
model, level channels, gates, claims ledger, shared statistics primitives, and
the non-benchmark figure renderer — and must be importable, and usable, without
ever pulling ``ailm.benchmarks`` in. This is an AST check, not an import-time
check: it never executes the target modules, so it cannot be fooled by a lazy
``__getattr__`` import and it cannot be slowed down by one either.

If a module below starts genuinely needing a benchmark symbol, that is a real
design question (move the symbol to a lower layer, as this same change moved
``wilson_interval`` / ``paired_difference_interval`` / ``paired_power`` out of
``ailm.benchmarks.scoring`` and into ``ailm.statistics``) rather than a reason
to relax this test. Do not add a module to ``KNOWN_BENCHMARK_LEAKS`` without
also filing the reason in the same commit.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

#: Package or module names declared to be core: reachable without importing
#: ``ailm.benchmarks``. A package name is checked recursively over every
#: ``.py`` file it contains; a plain module name is checked by itself.
CORE_LAYER_TARGETS: tuple[str, ...] = (
    "ailm.levels",
    "ailm.gates",
    "ailm.claims",
    "ailm.statistics",
    "ailm.generative",
    "ailm.channels",
    "ailm.figures.results",
)

#: Modules inside a target above that are allowed to import ``ailm.benchmarks``,
#: paired with the reason. Empty today: every module named in
#: ``CORE_LAYER_TARGETS`` is clean as of the ``wilson_interval`` /
#: ``paired_difference_interval`` / ``paired_power`` migration to
#: ``ailm.statistics``. A future entry here is a deliberate, documented
#: exception, not a silent allowance.
KNOWN_BENCHMARK_LEAKS: dict[str, str] = {}


def _benchmark_import_names(tree: ast.AST) -> list[tuple[int, str]]:
    """Return ``(lineno, dotted_name)`` for every import of ``ailm.benchmarks``."""
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ailm.benchmarks" or alias.name.startswith("ailm.benchmarks."):
                    hits.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            if module and (module == "ailm.benchmarks" or module.startswith("ailm.benchmarks.")):
                hits.append((node.lineno, module))
    return hits


def _module_source_files(module_name: str) -> list[Path]:
    """Resolve a module or package name to the ``.py`` files it owns.

    Uses import machinery (``importlib.util.find_spec``) rather than a
    hand-built path, both to fail loudly if a target module has been renamed
    or removed and to walk a package the same way Python itself would resolve
    it, ignoring caches and namespace-package edge cases.
    """
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(
            f"layering target {module_name!r} does not resolve to an importable module; "
            "update CORE_LAYER_TARGETS if it was renamed or removed"
        )
    origin = Path(spec.origin)
    if spec.submodule_search_locations is None:
        # A plain module: exactly one file.
        return [origin]
    # A package: every .py file under its directory, skipping caches.
    files: list[Path] = []
    for location in spec.submodule_search_locations:
        files.extend(
            path for path in Path(location).rglob("*.py") if "__pycache__" not in path.parts
        )
    return sorted(files)


def _module_name_for(file_path: Path, package_root: Path, package_name: str) -> str:
    """Best-effort dotted module name for a file under a resolved package."""
    relative = file_path.relative_to(package_root)
    parts = relative.with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((package_name, *parts)) if parts else package_name


@pytest.mark.parametrize("target", CORE_LAYER_TARGETS)
def test_core_layer_target_resolves(target: str) -> None:
    """Every declared core target must still exist and be importable by name.

    A target that silently stopped resolving (renamed module, deleted file)
    would make the benchmark-free assertion below vacuously true for it —
    this guards against exactly that.
    """
    files = _module_source_files(target)
    assert files, f"{target!r} resolved to no .py files; layering check would be vacuous"


@pytest.mark.parametrize("target", CORE_LAYER_TARGETS)
def test_core_layer_target_does_not_import_benchmarks(target: str) -> None:
    """No file reachable from a declared core target imports ``ailm.benchmarks``."""
    spec = importlib.util.find_spec(target)
    assert spec is not None and spec.origin is not None
    origin = Path(spec.origin)
    package_root = origin.parent if spec.submodule_search_locations is not None else origin.parent

    violations: list[str] = []
    for path in _module_source_files(target):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        hits = _benchmark_import_names(tree)
        if not hits:
            continue
        module_name = _module_name_for(path, package_root, target)
        if module_name in KNOWN_BENCHMARK_LEAKS:
            continue
        for lineno, imported in hits:
            violations.append(f"{path}:{lineno} imports {imported!r} (module {module_name!r})")

    assert not violations, (
        f"{target!r} is declared free of ailm.benchmarks, but:\n"
        + "\n".join(violations)
        + "\nEither undo the import, move the needed symbol to a lower layer "
        "(ailm.statistics is the precedent), or add a documented exception to "
        "KNOWN_BENCHMARK_LEAKS."
    )


def test_known_benchmark_leaks_are_empty_until_justified() -> None:
    """No leak is exempted without a reason string explaining why.

    A ``KNOWN_BENCHMARK_LEAKS`` entry with an empty reason would defeat the
    documentation requirement in the module docstring.
    """
    for module_name, reason in KNOWN_BENCHMARK_LEAKS.items():
        assert reason.strip(), f"{module_name!r} is exempted with no reason recorded"


def test_ailm_statistics_module_itself_has_no_runtime_benchmarks_import() -> None:
    """Belt-and-suspenders: the exact module the migration targeted.

    ``paired_power``'s default alpha is a duplicated literal rather than an
    import of ``ailm.benchmarks.constants.POWER_ALPHA`` for exactly this
    reason; this test is the executable form of that comment.
    """
    import ailm.statistics as statistics_module

    assert not hasattr(statistics_module, "ailm_benchmarks"), (
        "ailm.statistics must not bind a name from ailm.benchmarks at module scope"
    )
    source = Path(statistics_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=statistics_module.__file__)
    hits = _benchmark_import_names(tree)
    assert not hits, f"ailm.statistics imports ailm.benchmarks: {hits}"
