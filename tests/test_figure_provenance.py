"""The figure build authenticates the environment, inputs, sources, and PNGs.

``scripts/generate_figures.py`` writes ``figure_provenance.json`` next to the
figures. The v3 receipt binds the numeric environment, the exact canonical
manuscript-variable map, each generator's source module, and every output PNG.
Selective builds authenticate the complete prior receipt before rendering, and
all builds render in a sibling stage before publishing PNGs and the receipt.

Real subprocesses throughout (no mocks): each test invokes the actual script
and mutates real files to prove the checks can fail.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "generate_figures.py"


def _pin(package: str) -> str:
    """The exact ``package==version`` pin declared in pyproject.toml."""
    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(rf'"{re.escape(package)}==([^"]+)"', text)
    assert match, f"pyproject.toml no longer pins {package} exactly — update this test"
    return match.group(1)


def _write_variables(project_root: Path, variables: dict[str, str]) -> Path:
    """Write the same canonical JSON form as the manuscript-variable generator."""
    target = project_root / "output" / "data" / "manuscript_variables.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(variables, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generate(
    output_dir: Path,
    only: str,
    *,
    project_root: Path | None = None,
    script: Path = SCRIPT,
    timeout: float = 600,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    if project_root is not None:
        environment["AILM_PROJECT_ROOT"] = str(project_root)
    return subprocess.run(
        [sys.executable, str(script), "--only", only, "--output-dir", str(output_dir)],
        cwd=script.parent.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


@pytest.mark.timeout(1200)
def test_provenance_records_the_pinned_real_stack(tmp_path: Path) -> None:
    result = _generate(tmp_path, "letter_transition")
    assert result.returncode == 0, result.stderr[-2000:]
    figure = tmp_path / "letter_transition.png"
    assert figure.exists()
    assert result.stdout.splitlines() == [str(figure.absolute())]
    assert ".stage-" not in result.stdout

    payload = json.loads((tmp_path / "figure_provenance.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "ailm-figure-provenance-v3"

    env = payload["environment"]
    assert env["jax"] == _pin("jax"), "figures were built on an unpinned JAX"
    assert env["jaxlib"] == _pin("jaxlib")
    assert env["pymdp"] == _pin("inferactively-pymdp")
    assert env["jax_backend"] == "cpu", (
        "the environment of record is CPU; a GPU/Metal backend changes float "
        "results and must never silently produce the shipped figures"
    )
    assert env["jax_devices"], "no JAX devices recorded — the stack did not initialize"

    record = payload["figures"]["letter_transition"]
    assert record["module"] == "matrices"
    assert record["source"]["path"] == "src/ailm/figures/matrices.py"
    assert record["source"]["sha256"] == _sha256(PROJECT_ROOT / record["source"]["path"])
    assert record["outputs"] == [{"path": "letter_transition.png", "sha256": _sha256(figure)}]
    variables = payload["manuscript_variables"]
    assert variables["path"] == "output/data/manuscript_variables.json"
    assert variables["sha256"] == _sha256(PROJECT_ROOT / variables["path"])
    assert [source["path"] for source in payload["shared_sources"]] == [
        "scripts/generate_figures.py",
        "scripts/_artifact_io.py",
        "src/ailm/figures/_common.py",
        "src/ailm/viz_config.py",
    ]
    assert all(
        source["sha256"] == _sha256(PROJECT_ROOT / source["path"])
        for source in payload["shared_sources"]
    )
    code_dependencies = payload["code_dependencies"]
    assert [record["path"] for record in code_dependencies] == sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "src" / "ailm").rglob("*.py")
    )
    assert all(
        record["sha256"] == _sha256(PROJECT_ROOT / record["path"]) for record in code_dependencies
    )
    computation_inputs = payload["computation_inputs"]
    input_paths = [record["path"] for record in computation_inputs]
    assert "data/claims.json" in input_paths
    assert "data/learning_agreement.npz" in input_paths
    assert "vendor/ud_ewt_clause_counts.json" in input_paths
    assert "gnn/models/L2b_grapheme_to_phoneme.gnn.md" in input_paths
    assert "gnn/mutations/L4_case@frozen_transition.gnn.md" in input_paths
    assert input_paths == sorted(input_paths)
    assert "wall_seconds" not in record
    receipt_text = (tmp_path / "figure_provenance.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in receipt_text

    # The build log states the environment out loud, not only in the sidecar.
    assert "environment: jax" in result.stderr

    # POSITIVE CONTROL: a repeat must be byte-identical. Absolute paths or
    # wall-clock timings in the sidecar make this fail even if the PNG is
    # unchanged.
    first_receipt = (tmp_path / "figure_provenance.json").read_bytes()
    repeated = _generate(tmp_path, "letter_transition")
    assert repeated.returncode == 0, repeated.stderr[-2000:]
    assert (tmp_path / "figure_provenance.json").read_bytes() == first_receipt

    # A later --only run merges rather than erasing prior records.
    second = _generate(tmp_path, "wiring_comparison")
    assert second.returncode == 0, second.stderr[-2000:]
    merged = json.loads((tmp_path / "figure_provenance.json").read_text(encoding="utf-8"))
    assert set(merged["figures"]) >= {"letter_transition", "wiring_comparison"}


@pytest.mark.timeout(1200)
def test_selective_build_refuses_stale_manuscript_input_before_touching_outputs(
    tmp_path: Path,
) -> None:
    """NEGATIVE CONTROL: a receipt cannot relabel a changed token map."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "figures"
    variables = _write_variables(project_root, {"PLANTED_TOKEN": "before"})
    initial = _generate(output_dir, "case_matrices", project_root=project_root)
    assert initial.returncode == 0, initial.stderr[-2000:]

    figure = output_dir / "case_matrices.png"
    receipt = output_dir / "figure_provenance.json"
    previous_figure = figure.read_bytes()
    previous_receipt = receipt.read_bytes()
    variables.write_text(
        json.dumps({"PLANTED_TOKEN": "after"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = _generate(output_dir, "case_matrices", project_root=project_root)

    assert result.returncode != 0
    assert "manuscript variable map digest changed" in result.stderr
    assert result.stdout == ""
    assert figure.read_bytes() == previous_figure
    assert receipt.read_bytes() == previous_receipt


@pytest.mark.timeout(1200)
def test_selective_build_refuses_mutated_png_before_touching_outputs(tmp_path: Path) -> None:
    """NEGATIVE CONTROL: output bytes are authenticated, not merely named."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "figures"
    _write_variables(project_root, {"PLANTED_TOKEN": "stable"})
    initial = _generate(output_dir, "case_matrices", project_root=project_root)
    assert initial.returncode == 0, initial.stderr[-2000:]

    figure = output_dir / "case_matrices.png"
    receipt = output_dir / "figure_provenance.json"
    receipt_before = receipt.read_bytes()
    planted_mutation = figure.read_bytes() + b"planted output mutation"
    figure.write_bytes(planted_mutation)

    result = _generate(output_dir, "case_matrices", project_root=project_root)

    assert result.returncode != 0
    assert "figure output digest changed" in result.stderr
    assert result.stdout == ""
    assert figure.read_bytes() == planted_mutation
    assert receipt.read_bytes() == receipt_before


@pytest.mark.timeout(1200)
def test_selective_build_refuses_planted_source_digest_before_touching_outputs(
    tmp_path: Path,
) -> None:
    """NEGATIVE CONTROL: a changed generator receipt cannot be merged."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "figures"
    _write_variables(project_root, {"PLANTED_TOKEN": "stable"})
    initial = _generate(output_dir, "case_matrices", project_root=project_root)
    assert initial.returncode == 0, initial.stderr[-2000:]

    figure = output_dir / "case_matrices.png"
    receipt = output_dir / "figure_provenance.json"
    figure_before = figure.read_bytes()
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["figures"]["case_matrices"]["source"]["sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    planted_receipt = receipt.read_bytes()

    result = _generate(output_dir, "case_matrices", project_root=project_root)

    assert result.returncode != 0
    assert "generator source path or digest changed" in result.stderr
    assert result.stdout == ""
    assert figure.read_bytes() == figure_before
    assert receipt.read_bytes() == planted_receipt


@pytest.mark.timeout(1200)
def test_selective_build_refuses_planted_shared_source_digest(tmp_path: Path) -> None:
    """NEGATIVE CONTROL: shared save/layout/style code is receipt-bound too."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "figures"
    _write_variables(project_root, {"PLANTED_TOKEN": "stable"})
    initial = _generate(output_dir, "case_matrices", project_root=project_root)
    assert initial.returncode == 0, initial.stderr[-2000:]

    figure = output_dir / "case_matrices.png"
    receipt = output_dir / "figure_provenance.json"
    figure_before = figure.read_bytes()
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["shared_sources"][1]["sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    planted_receipt = receipt.read_bytes()

    result = _generate(output_dir, "case_matrices", project_root=project_root)

    assert result.returncode != 0
    assert "shared rendering source path or digest changed" in result.stderr
    assert result.stdout == ""
    assert figure.read_bytes() == figure_before
    assert receipt.read_bytes() == planted_receipt


@pytest.mark.timeout(1200)
def test_canonical_input_symlink_is_rejected_without_publication(tmp_path: Path) -> None:
    """NEGATIVE CONTROL: descriptor hashing must not follow a real symlink."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "figures"
    real_variables = tmp_path / "real_variables.json"
    real_variables.write_text('{"PLANTED_TOKEN": "stable"}\n', encoding="utf-8")
    canonical = project_root / "output" / "data" / "manuscript_variables.json"
    canonical.parent.mkdir(parents=True)
    canonical.symlink_to(real_variables)

    result = _generate(output_dir, "case_matrices", project_root=project_root)

    assert result.returncode != 0
    assert "regular non-symlink file" in result.stderr
    assert result.stdout == ""
    assert not output_dir.exists()


@pytest.mark.timeout(1200)
def test_generator_failure_publishes_neither_png_nor_receipt(tmp_path: Path) -> None:
    """NEGATIVE CONTROL: a real missing-token failure never crosses the stage."""
    project_root = tmp_path / "project"
    output_dir = tmp_path / "figures"
    _write_variables(project_root, {})
    output_dir.mkdir()
    figure = output_dir / "clause_null.png"

    result = _generate(output_dir, "clause_null", project_root=project_root)

    assert result.returncode != 0
    assert "MissingVariableError" in result.stderr
    assert "published 0" in result.stderr
    assert result.stdout == ""
    assert not figure.exists()
    assert not (output_dir / "figure_provenance.json").exists()


@pytest.mark.timeout(1200)
def test_selective_build_refuses_malformed_provenance_before_touching_png(
    tmp_path: Path,
) -> None:
    """NEGATIVE CONTROL: corruption is not an empty receipt to merge into."""
    figure = tmp_path / "letter_transition.png"
    receipt = tmp_path / "figure_provenance.json"
    previous_figure = b"previous figure bytes"
    previous_receipt = b'{"schema": "truncated"'
    figure.write_bytes(previous_figure)
    receipt.write_bytes(previous_receipt)

    result = _generate(tmp_path, "letter_transition")

    assert result.returncode != 0
    assert "existing provenance is unreadable" in result.stderr
    assert figure.read_bytes() == previous_figure
    assert receipt.read_bytes() == previous_receipt


@pytest.mark.timeout(1200)
def test_selective_build_refuses_the_retired_v2_receipt_contract(tmp_path: Path) -> None:
    """NEGATIVE CONTROL: the expanded v3 contract cannot reinterpret a v2 receipt."""
    receipt = tmp_path / "figure_provenance.json"
    retired = b'{"schema":"ailm-figure-provenance-v2"}\n'
    receipt.write_bytes(retired)

    result = _generate(tmp_path, "hierarchy")

    assert result.returncode != 0
    assert "unsupported schema" in result.stderr
    assert result.stdout == ""
    assert receipt.read_bytes() == retired
    assert not (tmp_path / "hierarchy.png").exists()


@pytest.mark.timeout(1200)
def test_selective_build_refuses_to_relabel_another_numeric_environment(
    tmp_path: Path,
) -> None:
    initial = _generate(tmp_path, "letter_transition")
    assert initial.returncode == 0, initial.stderr[-2000:]
    figure = tmp_path / "letter_transition.png"
    receipt = tmp_path / "figure_provenance.json"
    previous_figure = figure.read_bytes()
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["environment"]["numpy"] = "planted-different-version"
    receipt.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    previous_receipt = receipt.read_bytes()

    result = _generate(tmp_path, "letter_transition")

    assert result.returncode != 0
    assert "different numeric environment" in result.stderr
    assert figure.read_bytes() == previous_figure
    assert receipt.read_bytes() == previous_receipt


@pytest.mark.timeout(600)
def test_selective_build_rejects_a_fifo_receipt_without_blocking(tmp_path: Path) -> None:
    """NEGATIVE CONTROL: O_NONBLOCK reaches fstat instead of waiting for a writer."""
    receipt = tmp_path / "figure_provenance.json"
    os.mkfifo(receipt)

    result = _generate(tmp_path, "hierarchy", timeout=600)

    assert result.returncode != 0
    assert "existing provenance is unreadable" in result.stderr
    assert result.stdout == ""
    assert receipt.is_fifo()
    assert not (tmp_path / "hierarchy.png").exists()


@pytest.mark.timeout(600)
def test_selective_build_rejects_a_claimed_fifo_png_without_blocking(tmp_path: Path) -> None:
    """NEGATIVE CONTROL: the historical claimed-output FIFO hang is closed."""
    initial = _generate(tmp_path, "hierarchy")
    assert initial.returncode == 0, initial.stderr[-2000:]
    figure = tmp_path / "hierarchy.png"
    receipt = tmp_path / "figure_provenance.json"
    receipt_before = receipt.read_bytes()
    figure.unlink()
    os.mkfifo(figure)

    result = _generate(tmp_path, "hierarchy", timeout=600)

    assert result.returncode != 0
    assert "figure output hierarchy.png must be a regular file" in result.stderr
    assert result.stdout == ""
    assert figure.is_fifo()
    assert receipt.read_bytes() == receipt_before


def _copy_isolated_figure_runtime(destination: Path) -> Path:
    """Copy the bounded real runtime needed for a mutation/reuse control."""
    (destination / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, destination / "scripts" / SCRIPT.name)
    shutil.copy2(
        PROJECT_ROOT / "scripts" / "_artifact_io.py",
        destination / "scripts" / "_artifact_io.py",
    )
    shutil.copytree(
        PROJECT_ROOT / "src" / "ailm",
        destination / "src" / "ailm",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    from scripts.generate_figures import _computation_input_paths

    for relative in _computation_input_paths(PROJECT_ROOT):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / relative, target)
    return destination / "scripts" / SCRIPT.name


@pytest.mark.timeout(1200)
def test_selective_build_refuses_real_code_and_input_mutations(tmp_path: Path) -> None:
    """NEGATIVE CONTROL: untouched PNGs cannot inherit changed dependency bytes."""
    isolated_root = tmp_path / "isolated-runtime"
    isolated_script = _copy_isolated_figure_runtime(isolated_root)
    _write_variables(isolated_root, {"PLANTED_TOKEN": "stable"})
    output_dir = isolated_root / "output" / "figures"

    initial = _generate(
        output_dir,
        "hierarchy",
        project_root=isolated_root,
        script=isolated_script,
    )
    assert initial.returncode == 0, initial.stderr[-2000:]
    from scripts.generate_figure_registry import load_provenance

    assert load_provenance(isolated_root) == {"hierarchy.png": "ailm.figures.architecture"}
    figure = output_dir / "hierarchy.png"
    receipt = output_dir / "figure_provenance.json"
    figure_before = figure.read_bytes()
    receipt_before = receipt.read_bytes()

    claims = isolated_root / "data" / "claims.json"
    claims_before = claims.read_bytes()
    claims.write_bytes(claims_before + b"\n")
    changed_input = _generate(
        output_dir,
        "hierarchy",
        project_root=isolated_root,
        script=isolated_script,
    )
    assert changed_input.returncode != 0
    assert "computation-input inventory or digest changed" in changed_input.stderr
    assert figure.read_bytes() == figure_before
    assert receipt.read_bytes() == receipt_before

    claims.write_bytes(claims_before)
    for relative in ("src/ailm/dashboard/static/js/main.js", "output/web/ailm.html"):
        asset = isolated_root / relative
        original_asset = asset.read_bytes()
        asset.write_bytes(original_asset + b"changed capture source")
        changed_capture_source = _generate(
            output_dir, "hierarchy", project_root=isolated_root, script=isolated_script
        )
        assert changed_capture_source.returncode != 0
        assert "computation-input inventory or digest changed" in changed_capture_source.stderr
        assert figure.read_bytes() == figure_before
        assert receipt.read_bytes() == receipt_before
        asset.write_bytes(original_asset)

    computation = isolated_root / "src" / "ailm" / "cases.py"
    computation.write_text(
        computation.read_text(encoding="utf-8") + "\n# planted computation mutation\n",
        encoding="utf-8",
    )
    changed_code = _generate(
        output_dir,
        "hierarchy",
        project_root=isolated_root,
        script=isolated_script,
    )
    assert changed_code.returncode != 0
    assert "code dependency inventory or digest changed" in changed_code.stderr
    assert figure.read_bytes() == figure_before
    assert receipt.read_bytes() == receipt_before
