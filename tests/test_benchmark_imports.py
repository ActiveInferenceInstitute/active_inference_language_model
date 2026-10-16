"""Real import boundaries: registry access is light, estimators retain identity."""

from __future__ import annotations

import ast
import importlib
import json
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.timeout(240)
def test_probe_registry_and_warming_inventory_work_without_site_packages(tmp_path):
    # -I -S removes third-party dependencies altogether. An accidental numerical
    # import fails rather than merely running faster against a warmed cache.
    code = (
        "import json, sys; "
        f"sys.path[:0] = {[str(PROJECT_ROOT / 'src'), str(PROJECT_ROOT / 'scripts')]!r}; "
        "import ailm.benchmarks as package; "
        "from ailm.benchmarks.constants import PROBE_MODEL_TAGS; "
        "import warm_call_cache; "
        "assert package.PROBE_MODEL_TAGS is PROBE_MODEL_TAGS; "
        "assert warm_call_cache._declared_probes() == PROBE_MODEL_TAGS; "
        "assert all(tag in warm_call_cache._recorder_command(audit_strata_only=True) "
        "for tag in PROBE_MODEL_TAGS); "
        "assert not {'numpy', 'scipy', 'jax', 'pymdp', 'requests'} & set(sys.modules); "
        "print(json.dumps(PROBE_MODEL_TAGS))"
    )
    result = subprocess.run(
        [str(PROJECT_ROOT / ".venv/bin/python"), "-I", "-S", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout), "an empty registry cannot certify the import boundary"


def test_lazy_registry_matches_the_static_reexport_declarations():
    import ailm.benchmarks as package

    tree = ast.parse(Path(package.__file__).read_text())
    declared = {
        alias.asname or alias.name: (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("ailm.benchmarks.")
        for alias in node.names
    }
    assert declared
    assert package._LAZY_EXPORTS == declared
    assert set(package.__all__) <= set(declared)
    assert set(declared) <= set(dir(package))
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = package.not_a_declared_benchmark_export


@pytest.mark.timeout(300)
def test_every_lazy_export_is_the_original_owner_object():
    import ailm.benchmarks as package

    for name, (module_name, attribute) in package._LAZY_EXPORTS.items():
        owner = importlib.import_module(module_name)
        exported = getattr(package, name)
        assert exported is getattr(owner, attribute), name
        assert package.__dict__[name] is exported, name
