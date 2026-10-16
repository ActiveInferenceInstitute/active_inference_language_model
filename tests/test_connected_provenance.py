"""Independent inventory and real filesystem refusal controls for GUM provenance."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ailm.verification.connected_provenance import (
    ExecutionBindingError,
    canonical_execution_binding,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inventory_root(tmp_path):
    root = tmp_path / "project"
    for relative in canonical_execution_binding(ROOT):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    return root


def test_canonical_inventory_is_complete_sorted_and_hashes_actual_bytes():
    binding = canonical_execution_binding(ROOT)
    assert list(binding) == sorted(binding)
    required = {
        "scripts/record_connected_gum_claims.py",
        "scripts/_artifact_io.py",
        "src/ailm/levels/connected.py",
        "src/ailm/corpus/connected_gum.py",
        "src/ailm/verification/connected_provenance.py",
        "src/ailm/verification/__init__.py",
        "src/ailm/verification/oracle.py",
        "data/case_word/ud_ewt_case_word_counts.json",
        "data/widened/ud_ewt_clause_widened_counts.json",
        "pyproject.toml",
        "uv.lock",
    }
    assert required <= binding.keys()
    for relative, digest in binding.items():
        assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    for directory, pattern in (
        ("src/ailm/levels", "*.py"),
        ("src/ailm/corpus", "*.py"),
        ("src/ailm/gnn", "*.py"),
        ("gnn/models", "*.gnn.md"),
        ("vendor", "*.json"),
    ):
        present = {path.relative_to(ROOT).as_posix() for path in (ROOT / directory).glob(pattern)}
        assert present and present <= binding.keys()


def test_new_matching_source_is_bound_and_changed_bytes_change_digest(inventory_root):
    prior = canonical_execution_binding(inventory_root)
    added = inventory_root / "src/ailm/levels/added_dependency.py"
    added.write_text("VALUE = 1\n")
    current = canonical_execution_binding(inventory_root)
    assert set(current) - set(prior) == {"src/ailm/levels/added_dependency.py"}
    assert all(current[key] == value for key, value in prior.items())
    added.write_text("VALUE = 2\n")
    assert (
        canonical_execution_binding(inventory_root)["src/ailm/levels/added_dependency.py"]
        != current["src/ailm/levels/added_dependency.py"]
    )


def test_missing_fixed_input_is_refused(inventory_root):
    (inventory_root / "scripts/_artifact_io.py").unlink()
    with pytest.raises(ExecutionBindingError, match="unavailable or unsafe"):
        canonical_execution_binding(inventory_root)


def test_empty_discovered_inventory_is_refused(inventory_root):
    for path in (inventory_root / "gnn/models").glob("*.gnn.md"):
        path.unlink()
    with pytest.raises(ExecutionBindingError, match="empty execution input inventory"):
        canonical_execution_binding(inventory_root)


@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_nonregular_input_is_refused_without_blocking(inventory_root, kind):
    path = inventory_root / "scripts/_artifact_io.py"
    path.unlink()
    if kind == "directory":
        path.mkdir()
    else:
        os.mkfifo(path)
    with pytest.raises(ExecutionBindingError, match="regular file"):
        canonical_execution_binding(inventory_root)


@pytest.mark.parametrize("outside", [False, True])
def test_symlinked_leaf_is_refused_even_if_content_matches(inventory_root, tmp_path, outside):
    path = inventory_root / "scripts/_artifact_io.py"
    target = (tmp_path if outside else inventory_root) / "target.py"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(ExecutionBindingError, match="unavailable or unsafe"):
        canonical_execution_binding(inventory_root)


@pytest.mark.parametrize("outside", [False, True])
def test_symlinked_parent_directory_is_refused(inventory_root, tmp_path, outside):
    directory = inventory_root / "vendor"
    moved = (tmp_path if outside else inventory_root) / "moved-vendor"
    directory.rename(moved)
    directory.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ExecutionBindingError, match="unavailable or unsafe"):
        canonical_execution_binding(inventory_root)


def test_symlinked_project_root_is_refused(inventory_root, tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(inventory_root, target_is_directory=True)
    with pytest.raises(ExecutionBindingError, match="invalid execution root"):
        canonical_execution_binding(alias)


def test_nonexistent_project_root_is_refused(tmp_path):
    with pytest.raises(ExecutionBindingError, match="invalid execution root"):
        canonical_execution_binding(tmp_path / "missing")


def test_helper_import_preserves_inference_and_script_independence():
    code = f"""
import json, sys
sys.path.insert(0, {str(ROOT / "src")!r})
from ailm.verification.connected_provenance import canonical_execution_binding
forbidden = [name for name in sys.modules if name.split('.')[0] in {{'jax','jaxlib','pymdp'}} or name.startswith('ailm.levels') or name.startswith('scripts.') or name == 'record_connected_gum_claims']
assert not forbidden, forbidden
print(json.dumps({{'independent': True}}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {"independent": True}


def test_recorder_public_function_delegates_and_preserves_error_contract(inventory_root):
    code = f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(ROOT / "scripts")!r})
import record_connected_gum_claims as recorder
from ailm.verification.connected_provenance import canonical_execution_binding
from ailm.corpus.connected_gum import GumProtocolError
root = Path(sys.argv[1])
assert recorder.execution_binding(root) == canonical_execution_binding(root)
try:
    recorder.execution_binding(root / 'missing')
except GumProtocolError:
    pass
else:
    raise AssertionError('recorder accepted missing root')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(inventory_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
