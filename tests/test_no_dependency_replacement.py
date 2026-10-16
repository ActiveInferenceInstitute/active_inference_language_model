"""Enforce the repository's no-test-double contract over executable tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent


def _bound_name(alias: ast.alias) -> str:
    return alias.asname or alias.name.split(".", 1)[0]


def _is_call_to(node: ast.AST, names: set[str], attributes: set[str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id in names
    return isinstance(node.func, ast.Attribute) and node.func.attr in attributes


def _simple_name_targets(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        return {name for element in node.elts for name in _simple_name_targets(element)}
    return set()


def _module_factory_names(tree: ast.Module) -> set[str]:
    """Find local helpers whose return value is an imported module object."""
    factories: set[str] = set()
    for function in (
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        module_values: set[str] = set()
        changed = True
        while changed:
            changed = False
            for node in ast.walk(function):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                value = node.value
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = {name for target in targets for name in _simple_name_targets(target)}
                is_module = _is_call_to(
                    value,
                    {"import_module", "module_from_spec"},
                    {"import_module", "module_from_spec"},
                ) or (isinstance(value, ast.Name) and value.id in module_values)
                if is_module and not names <= module_values:
                    module_values.update(names)
                    changed = True
        if any(
            isinstance(node, ast.Return)
            and (
                (isinstance(node.value, ast.Name) and node.value.id in module_values)
                or _is_call_to(
                    node.value,
                    {"import_module", "module_from_spec"},
                    {"import_module", "module_from_spec"},
                )
            )
            for node in ast.walk(function)
        ):
            factories.add(function.name)
    return factories


def _is_modules_registry(
    node: ast.AST,
    *,
    sys_aliases: set[str],
    registry_aliases: set[str],
) -> bool:
    return (
        isinstance(node, ast.Name)
        and node.id in registry_aliases
        or isinstance(node, ast.Attribute)
        and node.attr == "modules"
        and isinstance(node.value, ast.Name)
        and node.value.id in sys_aliases
    )


def _syntax_violations(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        return [f"{path}:{error.lineno or 1}: syntax-error"]

    found: list[tuple[int, str]] = []
    imported_modules: set[str] = set()
    sys_aliases: set[str] = set()
    registry_aliases: set[str] = set()
    monkeypatch_aliases = {"monkeypatch"}
    mocker_aliases = {"mocker"}
    import_module_names: set[str] = set()
    module_from_spec_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = _bound_name(alias)
                imported_modules.add(bound)
                if alias.name == "sys":
                    sys_aliases.add(bound)
                if alias.name == "unittest.mock":
                    found.append((node.lineno, "unittest.mock import"))
        elif isinstance(node, ast.ImportFrom):
            if node.module == "unittest.mock" or (
                node.module == "unittest" and any(alias.name == "mock" for alias in node.names)
            ):
                found.append((node.lineno, "unittest.mock import"))
            if node.module == "sys":
                registry_aliases.update(
                    alias.asname or alias.name for alias in node.names if alias.name == "modules"
                )
            if node.module == "importlib":
                import_module_names.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "import_module"
                )
            if node.module == "importlib.util":
                module_from_spec_names.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name == "module_from_spec"
                )

    module_factories = _module_factory_names(tree)
    dynamic_modules: set[str] = set()
    #: String module names this file itself creates through the import
    #: factories — the only names whose sys.modules.pop cleanup is allowed.
    created_module_names: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {name for target in targets for name in _simple_name_targets(target)}
            if isinstance(value, ast.Name):
                for aliases in (monkeypatch_aliases, mocker_aliases, registry_aliases):
                    if value.id in aliases and not names <= aliases:
                        aliases.update(names)
                        changed = True
                if value.id in imported_modules | dynamic_modules and not names <= dynamic_modules:
                    dynamic_modules.update(names)
                    changed = True
            if (
                _is_modules_registry(
                    value,
                    sys_aliases=sys_aliases,
                    registry_aliases=registry_aliases,
                )
                and not names <= registry_aliases
            ):
                registry_aliases.update(names)
                changed = True
            if isinstance(value, ast.Call):
                call_name = value.func.id if isinstance(value.func, ast.Name) else None
                call_attr = value.func.attr if isinstance(value.func, ast.Attribute) else None
                returns_module = (
                    call_name in import_module_names | module_from_spec_names | module_factories
                    or call_attr in {"import_module", "module_from_spec"}
                )
                if returns_module and not names <= dynamic_modules:
                    dynamic_modules.update(names)
                    changed = True
                if returns_module:
                    for argument in value.args:
                        if (
                            isinstance(argument, ast.Constant)
                            and isinstance(argument.value, str)
                            and argument.value not in created_module_names
                        ):
                            created_module_names.add(argument.value)
                            changed = True

    dependency_modules = imported_modules | dynamic_modules
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            if (
                isinstance(owner, ast.Name)
                and owner.id in monkeypatch_aliases
                and node.func.attr in {"setattr", "delattr"}
            ):
                found.append((node.lineno, "monkeypatch dependency replacement"))
            if (
                isinstance(owner, ast.Name)
                and owner.id in mocker_aliases
                and node.func.attr in {"patch", "spy", "stub"}
            ):
                found.append((node.lineno, "mocker dependency replacement"))
            if _is_modules_registry(
                owner,
                sys_aliases=sys_aliases,
                registry_aliases=registry_aliases,
            ) and node.func.attr in {"clear", "pop", "popitem", "setdefault", "update"}:
                # Cleanup of a module this test itself created — a string
                # passed to import_module/module_from_spec, popped with the
                # None default in a finally — cannot replace a dependency:
                # the module never existed for any other code to import. Every
                # other shape stays flagged, including popping an imported
                # name or a name this file never created.
                is_probe_cleanup = (
                    node.func.attr == "pop"
                    and len(node.args) == 2
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and node.args[0].value in created_module_names
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value is None
                )
                if not is_probe_cleanup:
                    found.append((node.lineno, "sys.modules mutation"))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"setattr", "delattr"}
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in dependency_modules
        ):
            found.append((node.lineno, "direct dependency assignment"))

        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = list(node.targets)
        for target in targets:
            if isinstance(target, ast.Subscript) and _is_modules_registry(
                target.value,
                sys_aliases=sys_aliases,
                registry_aliases=registry_aliases,
            ):
                found.append((target.lineno, "sys.modules mutation"))
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id in dependency_modules
            ):
                found.append((target.lineno, "direct dependency assignment"))

    return [f"{path}:{line}: {kind}" for line, kind in sorted(set(found))]


def _violations(paths: list[Path]) -> list[str]:
    """Return every syntax-level dependency replacement with its location."""
    return [violation for path in paths for violation in _syntax_violations(path)]


def test_executable_tests_do_not_replace_dependencies():
    """Every top-level executable test is covered, including newly added files."""
    test_files = sorted(TESTS_DIR.glob("*.py"))
    assert len(test_files) > 50, "the test glob is unexpectedly narrow or empty"
    assert _violations(test_files) == []


@pytest.mark.parametrize(
    ("source", "kind"),
    (
        (
            "import unittest.mock as doubles\nreplacement = doubles.MagicMock()\n",
            "unittest.mock import",
        ),
        (
            "def test_x(monkeypatch):\n"
            "    substitute = monkeypatch\n"
            "    substitute.setattr(dependency, 'call', lambda: None)\n",
            "monkeypatch dependency replacement",
        ),
        (
            "import sys as runtime\nregistry = runtime.modules\nregistry['dependency'] = object()\n",
            "sys.modules mutation",
        ),
        (
            "import json as dependency\ndependency.loads = lambda value: {}\n",
            "direct dependency assignment",
        ),
        (
            "import importlib as loader\n"
            "dependency = loader.import_module('json')\n"
            "dependency.loads = lambda value: {}\n",
            "direct dependency assignment",
        ),
        (
            "import sys\nimport json\nsys.modules.pop('json', None)\n",
            "sys.modules mutation",
        ),
        (
            "import sys\nsys.modules.pop('never_created_here', None)\n",
            "sys.modules mutation",
        ),
    ),
)
def test_the_guard_detects_planted_aliased_replacements(
    tmp_path: Path,
    source: str,
    kind: str,
):
    planted = tmp_path / "planted_forbidden_test.py"
    assert planted.parent != TESTS_DIR
    planted.write_text(source, encoding="utf-8")
    violations = _violations([planted])
    assert any(kind in violation for violation in violations), violations


def test_the_guard_allows_environment_isolation_and_local_state(tmp_path: Path):
    planted = tmp_path / "allowed_test.py"
    planted.write_text(
        "class State:\n"
        "    pass\n\n"
        "def test_x(monkeypatch):\n"
        "    monkeypatch.setenv('AILM_TEST_VALUE', '1')\n"
        "    state = State()\n"
        "    state.value = 1\n\n"
        "import importlib as loader\n"
        "probe = loader.import_module('local_probe_module')\n"
        "import sys\n"
        "sys.modules.pop('local_probe_module', None)\n",
        encoding="utf-8",
    )
    assert _violations([planted]) == []
