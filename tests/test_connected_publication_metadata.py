from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Literal, assert_never

import pytest
from coverage import Coverage
from coverage.exceptions import ConfigError

from ailm import _publication_inventory as inventory
from ailm._publication_inventory import (
    file_hash,
    publication_source_bytes,
    validate_publication_execution_binding,
)
from ailm._publication_types import DerivativeError
from ailm.figures.connected import connected_core_data
from ailm.value_formatting import ManuscriptVariableError
from ailm.verification.connected_provenance import canonical_execution_binding

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = """[project]
name='fixture'
version='1.0.0'
requires-python='>=3.12'
dependencies=['jax==0.4.38']
[build-system]
requires=['hatchling==1.32.0']
build-backend='hatchling.build'
[tool.coverage.report]
fail_under=90
"""
URLS = "\n[project.urls]\nRepository='https://example.org/release'\n"
TOML_SHAPES = """
[tool.metadata_control]
boolean = true
integer = 1
floating = 1.0
text = 'one'
day = 2026-10-14
clock = 12:34:56
local = 2026-10-14T12:34:56
offset = 2026-10-14T12:34:56Z
nan = nan
infinity = inf
empty_table = {}
empty_array = []
nested = [{items = [true, 1, 1.0, {value = true}]}]
"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.org",
            "-C",
            str(root),
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def _binding(root: Path) -> dict[str, str]:
    return {
        name: file_hash(root / name)
        for name in (
            "pyproject.toml",
            "uv.lock",
            "src/ailm/levels/connected.py",
            "gnn/models/L1.gnn.md",
        )
    }


@pytest.fixture
def package_source(tmp_path, request):
    package = getattr(request, "param", PACKAGE)
    if package == "actual-original":
        declaration = json.loads((ROOT / "data/publication_derivative.json").read_bytes())
        package = publication_source_bytes(
            ROOT, declaration["measurement_commit"], "pyproject.toml"
        ).decode("utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "src/ailm/levels").mkdir(parents=True)
    (tmp_path / "gnn/models").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(package)
    for name in ("uv.lock", "src/ailm/levels/connected.py", "gnn/models/L1.gnn.md"):
        (tmp_path / name).write_text("original\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "original fixture")
    commit = _git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "data/publication_derivative.json").write_text(
        json.dumps({"measurement_commit": commit})
    )
    recorded = _binding(tmp_path)
    if package == PACKAGE:
        publication = package + URLS
    else:
        publication = package.replace(
            "https://github.com/docxology/active_inference_language_model",
            "https://github.com/ActiveInferenceInstitute/Active_Inference_Language_Model",
        )
        if publication == package:
            publication += URLS
    (tmp_path / "pyproject.toml").write_text(publication)
    return tmp_path, recorded


def test_url_only_bridge_authenticates_original_blob(package_source):
    root, recorded = package_source
    current = _binding(root)
    assert recorded["pyproject.toml"] != current["pyproject.toml"]
    assert recorded["pyproject.toml"] == hashlib.sha256(PACKAGE.encode()).hexdigest()
    validate_publication_execution_binding(root, recorded, current)
    assert _binding(root) == current
    assert recorded["pyproject.toml"] == hashlib.sha256(PACKAGE.encode()).hexdigest()


def test_unchanged_original_package_is_accepted(package_source):
    root, recorded = package_source
    (root / "pyproject.toml").write_text(PACKAGE)
    validate_publication_execution_binding(root, recorded, _binding(root))


@pytest.mark.parametrize("package_source", [PACKAGE + TOML_SHAPES], indirect=True)
@pytest.mark.parametrize("urls_only", [False, True])
def test_all_toml_shapes_preserved(package_source, urls_only):
    root, recorded = package_source
    if not urls_only:
        declaration = json.loads((root / "data/publication_derivative.json").read_bytes())
        original = _git(root, "show", declaration["measurement_commit"] + ":pyproject.toml")
        (root / "pyproject.toml").write_text(original + "\n")
    validate_publication_execution_binding(root, recorded, _binding(root))


@pytest.mark.parametrize("package_source", [PACKAGE + TOML_SHAPES], indirect=True)
@pytest.mark.parametrize(
    "before,after",
    [
        ("boolean = true", "boolean = 1"),
        ("integer = 1\n", "integer = true\n"),
        ("integer = 1\n", "integer = 1.0\n"),
        ("floating = 1.0", "floating = 1"),
        ("[true, 1, 1.0,", "[1, 1, 1.0,"),
        ("[true, 1, 1.0,", "[true, 1.0, 1.0,"),
        ("[true, 1, 1.0,", "[true, 1, 1,"),
        ("{value = true}", "{value = 1}"),
        ("{value = true}", "{value = true, extra = 1}"),
        ("empty_array = []", "empty_array = [1]"),
        ("empty_table = {}", "empty_table = []"),
        ("text = 'one'", "text = 'two'"),
        ("day = 2026-10-14", "day = 2026-10-15"),
        ("clock = 12:34:56", "clock = 12:34:57"),
        ("local = 2026-10-14T12:34:56", "local = 2026-10-14T12:34:57"),
        ("offset = 2026-10-14T12:34:56Z", "offset = 2026-10-14T12:34:57Z"),
        ("nan = nan", "nan = inf"),
        ("infinity = inf", "infinity = -inf"),
    ],
)
def test_recursive_toml_changes_refused(package_source, before, after):
    root, recorded = package_source
    path = root / "pyproject.toml"
    original = path.read_text()
    changed = original.replace(before, after)
    assert changed != original
    path.write_text(changed)
    with pytest.raises(DerivativeError, match="outside publication URLs"):
        validate_publication_execution_binding(root, recorded, _binding(root))


@pytest.mark.parametrize("package_source", ["actual-original"], indirect=True)
@pytest.mark.parametrize(
    "before,after",
    [("branch = true", "branch = 1"), ("fail_under = 90", "fail_under = 90.0")],
)
def test_real_original_coverage_type_changes_refused(package_source, before, after):
    root, recorded = package_source
    path = root / "pyproject.toml"
    assert recorded["pyproject.toml"] == (
        "d5a78054b8b0ad44a11d67d2ed0accd1a2a6acfeda2401e4f40d38ceb5fea353"
    )
    Coverage(config_file=str(path))
    validate_publication_execution_binding(root, recorded, _binding(root))
    original = path.read_text()
    changed = original.replace(before, after)
    assert changed != original
    path.write_text(changed)
    if before == "branch = true":
        with pytest.raises(ConfigError, match="branch is not a boolean"):
            Coverage(config_file=str(path))
    else:
        assert Coverage(config_file=str(path)).config.fail_under == 90.0
    with pytest.raises(DerivativeError, match="outside publication URLs"):
        validate_publication_execution_binding(root, recorded, _binding(root))


@pytest.mark.parametrize("change", ["wrong", "current", "missing"])
def test_recorded_package_hash_cannot_be_rebound(package_source, change):
    root, recorded = package_source
    if change == "missing":
        del recorded["pyproject.toml"]
    else:
        recorded["pyproject.toml"] = (
            file_hash(root / "pyproject.toml") if change == "current" else "f" * 64
        )
    with pytest.raises(DerivativeError, match="original measurement source"):
        validate_publication_execution_binding(root, recorded, _binding(root))


@pytest.mark.parametrize(
    "before,after",
    [
        ("jax==0.4.38", "jax==999"),
        (">=3.12", ">=3.13"),
        ("version='1.0.0'", "version='2.0.0'"),
        ("hatchling==1.32.0", "hatchling==999"),
        ("hatchling.build", "other.build"),
        ("fail_under=90", "fail_under=0"),
    ],
)
def test_non_url_configuration_refused_with_refreshed_hash(package_source, before, after):
    root, recorded = package_source
    (root / "pyproject.toml").write_text((PACKAGE + URLS).replace(before, after))
    with pytest.raises(DerivativeError, match="outside publication URLs"):
        validate_publication_execution_binding(root, recorded, _binding(root))


@pytest.mark.parametrize(
    "name", ["uv.lock", "src/ailm/levels/connected.py", "gnn/models/L1.gnn.md"]
)
def test_real_execution_input_changes_refused(package_source, name):
    root, recorded = package_source
    (root / name).write_text("changed\n")
    with pytest.raises(DerivativeError, match="inventory is incomplete or changed"):
        validate_publication_execution_binding(root, recorded, _binding(root))


@pytest.mark.parametrize(
    "change", ["missing", "extra", "current_hash", "unknown_commit", "bad_toml"]
)
def test_incomplete_or_unauthenticated_bridge_refused(package_source, change):
    root, recorded = package_source
    current = _binding(root)
    if change == "missing":
        del recorded["uv.lock"]
    elif change == "extra":
        recorded["extra.py"] = "f" * 64
    elif change == "current_hash":
        current["pyproject.toml"] = "f" * 64
    elif change == "unknown_commit":
        (root / "data/publication_derivative.json").write_text(
            json.dumps({"measurement_commit": "f" * 40})
        )
    else:
        (root / "pyproject.toml").write_text("invalid TOML")
        current = _binding(root)
    with pytest.raises(DerivativeError):
        validate_publication_execution_binding(root, recorded, current)


def _retained_report() -> Path:
    return Path(
        os.environ.get("AILM_CONNECTED_RETAINED_REPORT", ROOT / "output/reports/connected_gum.json")
    )


def _portable_checkout(tmp_path: Path):
    root = tmp_path / "public"
    (root / "data").mkdir(parents=True)
    for name in (
        "pyproject.toml",
        "data/publication_derivative.json",
        "data/publication_sources.json",
    ):
        (root / name).write_bytes((ROOT / name).read_bytes())
    _git(root, "init", "--quiet")
    _git(root, "add", "-f", "--", "pyproject.toml", "data")
    _git(root, "commit", "--quiet", "-m", "Public source fixture")
    assert _git(root, "rev-list", "--count", "HEAD") == "1"
    recorded = json.loads(_retained_report().read_bytes())["execution_binding"]
    return root, recorded, canonical_execution_binding(ROOT)


def test_public_orphan_source_preserves_original_package_binding(tmp_path: Path) -> None:
    root, recorded, current = _portable_checkout(tmp_path)
    validate_publication_execution_binding(root, recorded, current)


@pytest.mark.parametrize("mutation", ["corrupt", "missing"])
def test_public_orphan_source_refuses_unauthenticated_archive(
    tmp_path: Path, mutation: Literal["corrupt", "missing"]
) -> None:
    root, recorded, current = _portable_checkout(tmp_path)
    archive = root / "data/publication_sources.json"
    match mutation:
        case "corrupt":
            archive.write_bytes(archive.read_bytes() + b" ")
        case "missing":
            archive.rename(root / "data/held-publication-sources.json")
        case unreachable:
            assert_never(unreachable)
    with pytest.raises(DerivativeError):
        validate_publication_execution_binding(root, recorded, current)


def test_public_orphan_source_refuses_non_url_configuration_change(tmp_path: Path) -> None:
    root, recorded, current = _portable_checkout(tmp_path)
    package = root / "pyproject.toml"
    before = package.read_bytes()
    assert b"inferactively-pymdp==1.0.4" in before
    package.write_bytes(
        before.replace(b"inferactively-pymdp==1.0.4", b"inferactively-pymdp==1.0.3")
    )
    current["pyproject.toml"] = file_hash(package)
    with pytest.raises(DerivativeError, match="outside publication URLs"):
        validate_publication_execution_binding(root, recorded, current)


def test_corrupt_archive_refuses_available_git_route(package_source) -> None:
    root, recorded = package_source
    archive = root / "data/publication_sources.json"
    archive.write_bytes((ROOT / "data/publication_sources.json").read_bytes() + b" ")
    with pytest.raises(DerivativeError, match="archive digest mismatch"):
        validate_publication_execution_binding(root, recorded, _binding(root))


def test_portable_source_refuses_unregistered_ref(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data/publication_sources.json").write_bytes(
        (ROOT / "data/publication_sources.json").read_bytes()
    )
    with pytest.raises(DerivativeError, match="does not authorize requested ref/path"):
        publication_source_bytes(tmp_path, "f" * 40, "pyproject.toml")


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "duplicate", "extent", "digest", "base64", "schema", "ref"]
)
def test_portable_source_entry_controls(
    mutation: Literal[
        "missing", "extra", "duplicate", "extent", "digest", "base64", "schema", "ref"
    ],
) -> None:
    raw = (ROOT / "data/publication_sources.json").read_bytes()
    archive = json.loads(raw)
    key = next(iter(archive["blobs"]))
    match mutation:
        case "missing":
            archive["blobs"].pop(key)
        case "extra":
            archive["blobs"]["f" * 40 + ":extra.py"] = archive["blobs"][key]
        case "duplicate":
            raw = raw.replace(b'"schema":', b'"schema": "duplicate", "schema":', 1)
        case "extent":
            archive["blobs"][key]["bytes"] += 1
        case "digest":
            archive["blobs"][key]["sha256"] = "0" * 64
        case "base64":
            archive["blobs"][key]["base64"] = "!"
        case "schema":
            archive["schema"] = "unknown"
        case "ref":
            archive["blobs"]["f" * 40 + ":pyproject.toml"] = archive["blobs"].pop(key)
        case unreachable:
            assert_never(unreachable)
    if mutation != "duplicate":
        raw = json.dumps(archive).encode("utf-8")
    with pytest.raises(DerivativeError):
        inventory._publication_source_blobs(raw)


def test_retained_original_report_reconciles_current_metadata():
    report_path = _retained_report()
    content = report_path.read_bytes()
    report = json.loads(content)
    claims = json.loads((ROOT / "data/claims.json").read_bytes())["claims"]
    assert hashlib.sha256(content).hexdigest() == claims["connected_gum_report_sha256"]["value"]
    current = canonical_execution_binding(ROOT)
    assert {name for name in current if current[name] != report["execution_binding"][name]} == {
        "pyproject.toml"
    }
    values = connected_core_data(report_path=report_path)
    for attribute, suffix in [
        ("gain", "document_mean_gain"),
        ("low", "document_ci_low"),
        ("high", "document_ci_high"),
        ("documents", "documents"),
    ]:
        assert getattr(values, attribute) == claims["connected_gum_" + suffix]["value"]
    assert report_path.read_bytes() == content
    assert canonical_execution_binding(ROOT) == current


@pytest.mark.parametrize("change", ["package", "lock", "core", "model", "missing", "extra"])
def test_consumer_refuses_refreshed_artifact_and_ledger_hashes(tmp_path, change):
    report = json.loads(_retained_report().read_bytes())
    ledger = json.loads((ROOT / "data/claims.json").read_bytes())
    ledger["claims"] = {k: v for k, v in ledger["claims"].items() if k.startswith("connected_gum_")}
    binding = report["execution_binding"]
    names = {
        "package": "pyproject.toml",
        "lock": "uv.lock",
        "core": "src/ailm/levels/connected.py",
        "model": "gnn/models/L1_letters.gnn.md",
    }
    if change == "missing":
        del binding["uv.lock"]
    elif change == "extra":
        binding["extra.py"] = "f" * 64
    else:
        name = names[change]
        if change == "model":
            name = next(key for key in binding if key.startswith("gnn/models/"))
        binding[name] = "f" * 64
    execution = (json.dumps(binding, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    report["execution_sha256"] = hashlib.sha256(execution).hexdigest()
    ledger["claims"]["connected_gum_execution_sha256"]["value"] = report["execution_sha256"]
    report["runtime_sha256"] = binding["src/ailm/levels/connected.py"]
    ledger["claims"]["connected_gum_runtime_sha256"]["value"] = report["runtime_sha256"]
    report_path, ledger_path = tmp_path / "report.json", tmp_path / "claims.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    ledger["claims"]["connected_gum_report_sha256"]["value"] = file_hash(report_path)
    ledger_path.write_text(json.dumps(ledger))
    with pytest.raises(
        ManuscriptVariableError, match="execution inventory is incomplete or changed"
    ):
        connected_core_data(ledger_path, report_path)
