"""Real-process tests for the release receipt orchestrator."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import inspect
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import threading
import time
import tomllib
from dataclasses import asdict
from pathlib import Path

import pytest

import scripts.verify_network_claim_bindings as network_binding_verifier
import scripts.verify_publication_derivative as publication_verifier
import scripts.warm_call_cache as canonical_warmer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "verify_release.py"
CI_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
APPROVED_CI_ACTIONS = {
    "actions/checkout": "9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0",
    "astral-sh/setup-uv": "c771a70e6277c0a99b617c7a806ffedaca235ff9",
    "actions/upload-artifact": "b7c566a772e6b6bfb58ed0dc250532a479d7789f",
}


def _git(root: Path, *args: str, environment: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _init_git_repository(root: Path, *, with_release_script: bool = False) -> str:
    root.mkdir()
    shutil.copy2(PROJECT_ROOT / "release-uv.toml", root / "release-uv.toml")
    if with_release_script:
        scripts = root / "scripts"
        scripts.mkdir()
        shutil.copy2(SCRIPT, scripts / SCRIPT.name)
        interpreter = root / ".venv" / "bin" / "python"
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(Path(sys.executable).resolve())
        (root / ".venv" / "pyvenv.cfg").write_text(
            f"home = {Path(sys._base_executable).parent}\n"
            "include-system-site-packages = false\n"
            f"version_info = {sys.version_info.major}.{sys.version_info.minor}\n",
            encoding="utf-8",
        )
        (
            root
            / ".venv"
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        ).mkdir(parents=True)
    (root / "tracked.txt").write_text("initial\n", encoding="utf-8")
    _git(root, "init", "-q")
    # The release runner reads Git with GIT_CONFIG_GLOBAL=/dev/null, so a
    # developer's user-level ignore file (for example one listing `.venv/`)
    # must not decide what this fixture commits: otherwise `git add .` skips
    # the planted interpreter and the runner then refuses the tree as dirty.
    _git(root, "config", "core.excludesFile", os.devnull)
    _git(root, "config", "user.name", "AILM Release Test")
    _git(root, "config", "user.email", "release-test@example.invalid")
    _git(root, "add", ".")
    commit_environment = {
        **os.environ,
        "GIT_AUTHOR_DATE": "2023-11-14T22:13:20+00:00",
        "GIT_COMMITTER_DATE": "2023-11-14T22:13:20+00:00",
    }
    _git(root, "commit", "-q", "-m", "initial", environment=commit_environment)
    return _git(root, "rev-parse", "HEAD")


def _repository_python(root: Path) -> str:
    """Exact interpreter path required by a copied release orchestrator."""

    return str(root / ".venv" / "bin" / "python")


def _stdlib_sitecustomize(repository: Path) -> Path | None:
    """The stdlib ``sitecustomize`` that shadows a venv one, when the interpreter ships it.

    Debian-family builds install ``sitecustomize.py`` in the standard library,
    which precedes site-packages on ``sys.path``; a module planted in the venv
    can then never be imported, so the positive control must read the module
    that actually did.
    """

    candidate = Path(sysconfig.get_path("stdlib")) / "sitecustomize.py"
    if not candidate.exists():
        return None
    probe = subprocess.run(
        [
            _repository_python(repository),
            "-I",
            "-c",
            "import sitecustomize; print(sitecustomize.__file__)",
        ],
        cwd=repository,
        env=_release_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    assert Path(probe.stdout.strip()).resolve() == candidate.resolve()
    return candidate


def _project_python_probe() -> str:
    """Execute the module's real interpreter check through a spawned child.

    The child is the exact sidecar python (.venv/bin/python), so
    ``sys.executable`` inside it matches the contract under test. In-process we
    cannot satisfy it because uv-built venvs expose ``python3`` while the
    contract names ``python`` -- and patching ``sys.executable`` would be a
    dependency replacement, which the house rule forbids.
    """
    code = (
        "import sys; sys.path.insert(0, 'scripts'); "
        "import verify_release as m; print(m._project_python())"
    )
    result = subprocess.run(
        [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-I", "-S", "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=480,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _release_environment(**overrides: object) -> dict[str, str]:
    """Caller environment without uv-run bookkeeping leaked into children.

    ``uv run`` exports ``UV_RUN_RECURSION_DEPTH``; pytest itself launched via
    ``uv run`` passes it to every child it spawns.  The release orchestrator
    treats that variable as a *direct* uv-run launch — correct for a human
    invocation, wrong for a test subprocess whose interpreter identity is the
    thing under test — so its refusal fires before the condition the negative
    control actually examines.  A test that plants the variable deliberately
    passes ``uv_run_recursion_depth="1"`` explicitly instead.
    """

    env = {key: value for key, value in os.environ.items() if key != "UV_RUN_RECURSION_DEPTH"}
    depth = overrides.pop("uv_run_recursion_depth", None)
    if depth is not None:
        env["UV_RUN_RECURSION_DEPTH"] = str(depth)
    env.update({key: str(value) for key, value in overrides.items()})
    return env


def _ci_release_contract_violations(source: str) -> list[str]:
    """Return high-signal hosted-CI release-policy violations."""

    import yaml

    violations: list[str] = []
    try:
        workflow = yaml.safe_load(source)
        jobs = workflow["jobs"]
        steps = [
            step for job in jobs.values() for step in job.get("steps", ()) if isinstance(step, dict)
        ]
        run_commands = [str(step["run"]) for step in steps if "run" in step]
    except (TypeError, KeyError, yaml.YAMLError):
        return ["invalid-workflow-yaml"]

    if workflow.get("permissions") != {"contents": "read"}:
        violations.append("permissions-not-contents-read")
    if any("permissions" in job for job in jobs.values() if isinstance(job, dict)):
        violations.append("job-permissions-override")

    for step in steps:
        if "uses" not in step:
            continue
        uses = str(step["uses"])
        match = re.fullmatch(r"([^@]+)@([0-9a-f]{40})", uses)
        if match is None:
            violations.append(f"action-not-full-sha:{uses}")
            continue
        action, revision = match.groups()
        if APPROVED_CI_ACTIONS.get(action) != revision:
            violations.append(f"action-not-approved:{uses}")
        if (
            action == "actions/checkout"
            and step.get("with", {}).get("persist-credentials") is not False
        ):
            violations.append("checkout-persists-credentials")

    if any(re.search(r"\$\{\{\s*github\.event(?:\.|\s*\}\})", command) for command in run_commands):
        violations.append("event-expression-in-run")

    if any("uv run" in command for command in run_commands):
        violations.append("uv-run-sidecar")
    if any(re.search(r"verify_release\.py\s+full(?:\s|$)", command) for command in run_commands):
        violations.append("hosted-full-without-local-inputs")
    if any(
        "verify_release.py" in command
        and re.search(r"\.venv/bin/python\s+-I\s+-S\s+scripts/verify_release\.py", command) is None
        for command in run_commands
    ):
        violations.append("release-python-not-isolated")
    expected_quick = ".venv/bin/python -I -S scripts/verify_release.py quick"
    if run_commands.count(expected_quick) != 1:
        violations.append("explicit-project-python-quick")
    for prerequisite in (
        "external-call cache",
        "Ollama server",
        "every declared probe",
        "docxology/template",
        "Lean",
        "qpdf",
    ):
        if prerequisite not in source:
            violations.append(f"missing-full-prerequisite:{prerequisite}")
    return violations


def _init_template_repository(root: Path, *, project_root: Path = PROJECT_ROOT) -> str:
    root.mkdir()
    required = {
        "infrastructure/__init__.py": '"""Test template infrastructure."""\n',
        "scripts/pipeline/stage_03_render.py": "raise SystemExit(0)\n",
        "scripts/pipeline/stage_04_validate.py": "raise SystemExit(0)\n",
        "pyproject.toml": "[project]\nname = 'test-template'\nversion = '0.0.0'\n",
        "uv.lock": "version = 1\nrevision = 1\nrequires-python = '>=3.11'\n",
        ".gitignore": "projects/working/active_inference_language_model\n",
    }
    for relative, content in required.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    project_entry = root / "projects/working/active_inference_language_model"
    project_entry.parent.mkdir(parents=True)
    project_entry.symlink_to(project_root.resolve(), target_is_directory=True)
    _git(root, "init", "-q")
    # The release runner reads Git with GIT_CONFIG_GLOBAL=/dev/null, so a
    # developer's user-level ignore file (for example one listing `.venv/`)
    # must not decide what this fixture commits: otherwise `git add .` skips
    # the planted interpreter and the runner then refuses the tree as dirty.
    _git(root, "config", "core.excludesFile", os.devnull)
    _git(root, "config", "user.name", "AILM Release Test")
    _git(root, "config", "user.email", "release-test@example.invalid")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "template")
    return _git(root, "rev-parse", "HEAD")


def _module():
    return importlib.import_module("scripts.verify_release")


def _build_profiles_in_sidecar(template: Path):
    """Build quick+full step inventories inside the real sidecar python.

    Returns plain dicts so assertions run against data, not live objects; the
    builder itself executed for real, with its own interpreter-identity check
    passing because the child IS .venv/bin/python.
    """
    code = (
        "import sys, json\n"
        "sys.path.insert(0, 'scripts')\n"
        "import verify_release as m\n"
        f"from pathlib import Path as _P; template = _P(r'{template}')\n"
        "def enc(steps):\n"
        "    return [\n"
        "        {\n"
        "            'name': s.name,\n"
        "            'argv': list(s.argv),\n"
        "            'cwd': s.cwd,\n"
        "            'env_ov': [list(o) for o in (s.environment_overrides or ())],\n"
        "            'forbid_skips': bool(s.forbid_pytest_skips),\n"
        "            'depends_on': list(s.depends_on or ()),\n"
        "        }\n"
        "        for s in steps\n"
        "    ]\n"
        "payload = {'quick': enc(m.build_steps('quick')), "
        "'full': enc(m.build_steps('full', template_root=template))}\n"
        "print(json.dumps(payload))\n"
    )
    result = subprocess.run(
        [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-I", "-S", "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=480,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)["quick"], json.loads(result.stdout)["full"]


@pytest.mark.timeout(960)
def test_profiles_have_distinct_bounded_acceptance_surfaces(tmp_path, monkeypatch):
    # The pytest process itself may have been launched through `uv run`; and
    # uv-built venvs name sys.executable "python3" while the sidecar pins
    # "python", so in-process build_steps would refuse. We therefore execute the
    # builder inside the REAL sidecar python via the probe helper — no mocks,
    # no setattr, and the interpreter-identity half of the contract is covered
    # by the subprocess controls in this file.
    monkeypatch.delenv("UV_RUN_RECURSION_DEPTH", raising=False)
    module = _module()
    template = tmp_path / "template"
    _init_template_repository(template)
    quick, full = _build_profiles_in_sidecar(template)
    expected_python = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    assert _project_python_probe() == expected_python
    quick_names = [step["name"] for step in quick]
    full_names = [step["name"] for step in full]
    assert (
        quick_names[:6]
        == full_names[:6]
        == [
            "lock",
            "environment",
            "environment-compatibility",
            "format",
            "lint",
            "types",
        ]
    )
    assert "tests-quick" in quick_names
    assert "tests-full" in full_names
    assert quick_names.index("web-model") < quick_names.index("web-page")
    assert quick_names.index("web-page") < quick_names.index("tests-quick")
    assert full_names.index("web-page") < full_names.index("tests-full")
    assert full_names.index("pymdp-sparse-bundle-generate") < full_names.index(
        "pymdp-sparse-bundle-check"
    )
    assert full_names.index("pymdp-sparse-bundle-check") < full_names.index("tests-full")
    assert full_names.index("release-infrastructure-coverage") < full_names.index("tests-full")
    assert "package-distributions" in full_names
    assert full_names.index("package-distributions") < full_names.index("tests-full")
    assert full_names.index("network-claim-bindings") < full_names.index("cache-audit")
    assert full_names.index("cache-audit") < full_names.index("tests-full"), (
        "the expensive full suite starts before its ignored cache prerequisite is audited"
    )
    assert full_names.index("raw-corpus-reproducibility") < full_names.index("tests-full")
    assert {
        "gate-audit-full",
        "lean-ledger-check",
        "lean-build",
        "lean-negative-control",
        "manuscript-variables",
        "hydrated-docs",
        "hydrated-docs-check",
        "figures",
        "figure-registry",
        "figure-registry-check",
        "template-render",
        "publication-html-references",
        "manuscript-render-contract",
        "pdf-qpdf",
        "pdf-integrity",
        "template-validate",
    } <= set(full_names)
    # Order, not just presence. A set cannot see order, and the ordering here is
    # the one the pipeline docs call "the pair people get wrong": the figure
    # modules read output/data/manuscript_variables.json, that file is tracked, and
    # a stale copy does not raise -- it silently supplies the previous ledger and
    # the figures draw it. A shipped figure once contradicted the ledger this way,
    # past a green suite, because this assertion was a set.
    assert full_names.index("manuscript-variables") < full_names.index("figures"), (
        "the full profile regenerates figures before the manuscript variable map, so "
        "figures would be drawn from whatever map was last committed rather than from "
        "the current ledger"
    )
    assert full_names.index("manuscript-variables") < full_names.index("hydrated-docs")
    assert "cover" in full_names and "cover-check" in full_names, (
        "the release plan can publish a stale title-page cover"
    )
    cover = next(step for step in full if step["name"] == "cover")
    cover_check = next(step for step in full if step["name"] == "cover-check")
    render = next(step for step in full if step["name"] == "template-render")
    assert "manuscript-variables" in cover["depends_on"]
    assert "cover" in cover_check["depends_on"]
    assert "--check" in cover_check["argv"]
    assert "cover-check" in render["depends_on"]
    assert full_names.index("hydrated-docs") < full_names.index("hydrated-docs-check")
    assert full_names.index("template-render") < full_names.index("publication-html-references")
    assert full_names.index("publication-html-references") < full_names.index("hydrated-docs-check")
    assert full_names.index("hydrated-docs-check") < full_names.index("template-validate")
    references = next(step for step in full if step["name"] == "publication-html-references")
    assert references["depends_on"] == ["template-render"]
    assert "--publication-references" in references["argv"]
    docs_check = next(step for step in full if step["name"] == "hydrated-docs-check")
    assert "--check" in docs_check["argv"]
    assert "publication-html-references" in docs_check["depends_on"]
    assert "hydrated-docs" in render["depends_on"]
    assert "hydrated-docs-check" not in render["depends_on"]
    assert full_names.index("figures") < full_names.index("figure-registry")
    assert full_names.index("figure-registry") < full_names.index("figure-registry-check")
    assert full_names.index("lean-ledger-check") < full_names.index("lean-build")
    assert full_names.index("lean-build") < full_names.index("lean-negative-control")
    assert full_names.index("template-render") < full_names.index("manuscript-render-contract")
    assert full_names.index("manuscript-render-contract") < full_names.index("pdf-qpdf")
    render_contract = next(step for step in full if step["name"] == "manuscript-render-contract")
    assert "template-render" in render_contract["depends_on"]
    assert (
        "tests/test_manuscript_tokens.py::test_real_resolved_tree_is_exactly_current"
        in render_contract["argv"]
    )
    qpdf_step = next(step for step in full if step["name"] == "pdf-qpdf")
    assert "manuscript-render-contract" in qpdf_step["depends_on"]
    assert full_names.index("pdf-qpdf") < full_names.index("pdf-integrity")
    assert full_names.index("pdf-integrity") < full_names.index("template-validate")
    manuscript_step = next(step for step in full if step["name"] == "manuscript-variables")
    assert "--print-only" not in manuscript_step["argv"], (
        "the release profile refreshed the token map but skipped the resolved manuscript tree"
    )
    registry_check = next(step for step in full if step["name"] == "figure-registry-check")
    assert "--check" in registry_check["argv"], (
        "the registry was written but never rebuilt in non-writing validation mode"
    )
    assert "not recorder_replay" in next(
        " ".join(step["argv"]) for step in quick if step["name"] == "tests-quick"
    )
    assert "not raw_corpus_regeneration" in next(
        " ".join(step["argv"]) for step in quick if step["name"] == "tests-quick"
    )
    assert "not requires_cuda12" in next(
        " ".join(step["argv"]) for step in quick if step["name"] == "tests-quick"
    ), "the CUDA-only tests skip deterministically on a CPU host; both profiles deselect them"
    raw_replay = next(step for step in full if step["name"] == "raw-corpus-reproducibility")
    assert raw_replay["env_ov"] == [["AILM_RUN_RAW_CORPUS_REGENERATION", "1"]]
    assert raw_replay["forbid_skips"] is True
    assert "raw_corpus_regeneration" in raw_replay["argv"]
    assert "--runxfail" in raw_replay["argv"]
    assert "not raw_corpus_regeneration" in next(
        " ".join(step["argv"]) for step in full if step["name"] == "tests-full"
    )
    assert "not requires_cuda12" in next(
        " ".join(step["argv"]) for step in full if step["name"] == "tests-full"
    ), "a forbid-skips profile that selects the CUDA-only tests can never pass on a CPU host"
    tests_full = next(step for step in full if step["name"] == "tests-full")
    assert "--ignore=tests/test_release_orchestration.py" in tests_full["argv"]
    assert "pymdp-sparse-bundle-check" in tests_full["depends_on"]
    assert "release-infrastructure-coverage" in tests_full["depends_on"]
    assert tests_full["env_ov"] == [
        ["AILM_RUN_LIVE_OLLAMA_TESTS", "1"],
        ["AILM_RELEASE_MANDATORY_ENVIRONMENT", "1"],
    ], (
        "the full profile forbids skips but did not explicitly opt into its live Ollama "
        "tests and tell environment-dependent tests to fail rather than skip"
    )
    release_coverage = next(
        step for step in full if step["name"] == "release-infrastructure-coverage"
    )
    for covered_module in (
        "scripts.verify_release",
        "scripts.verify_network_claim_bindings",
        "scripts.verify_publication_derivative",
        "scripts._publication_execution",
        "scripts._publication_bootstrap",
        "scripts._publication_artists",
        "scripts._publication_controls",
        "scripts._publication_snapshot",
        "scripts.orchestrate_claim_recorders",
        "scripts.warm_call_cache",
    ):
        assert f"--cov={covered_module}" in release_coverage["argv"]
    for infrastructure_tests in (
        "tests/test_release_orchestration.py",
        "tests/test_replay_provenance.py",
        "tests/test_claim_recorder_orchestration.py",
        "tests/test_publication_derivative.py",
    ):
        assert infrastructure_tests in release_coverage["argv"]
    assert release_coverage["argv"][release_coverage["argv"].index("-k") + 1] == (
        "not real_reversed_dependency"
    )
    assert "--cov-branch" in release_coverage["argv"]
    assert "--cov-fail-under=60" in release_coverage["argv"]
    pytest_steps = [step for step in full if "pytest" in step["argv"]]
    assert pytest_steps
    assert all(step["forbid_skips"] for step in pytest_steps), (
        "a pytest-based publication step can exit zero after skipping mandatory evidence"
    )
    environment = next(step for step in full if step["name"] == "environment")
    quick_environment = next(step for step in quick if step["name"] == "environment")
    assert quick_environment["argv"] == [
        environment["argv"][0],
        "sync",
        "--check",
        "--locked",
        "--python",
        expected_python,
    ]
    assert environment["argv"] == [
        quick_environment["argv"][0],
        "sync",
        "--check",
        "--locked",
        "--extra",
        "gnn-interop",
        "--python",
        expected_python,
    ], "full requires actual GNN interoperability and must select its declared extra"
    assert list(environment["depends_on"]) == list(quick_environment["depends_on"]) == ["lock"]
    tests_quick = next(step for step in quick if step["name"] == "tests-quick")
    assert tests_quick["env_ov"] == []
    assert not tests_quick["forbid_skips"]
    # Quick parallelizes without coverage; full stays serial so its receipt
    # parser sees exactly one coverage TOTAL row and the recorder replay keeps
    # its single ledger lock.
    assert tests_quick["argv"][tests_quick["argv"].index("-n") + 1] == "4"
    assert "--no-cov" in tests_quick["argv"]
    assert "-n" not in tests_full["argv"]
    assert tests_quick["argv"][-1] == (
        "not recorder_replay and not raw_corpus_regeneration and not requires_cuda12 "
        "and not publication_context and not template_context"
    )
    assert tests_full["argv"][tests_full["argv"].index("-m") + 1] == (
        "not raw_corpus_regeneration and not requires_cuda12"
    )
    for name in ("format", "lint", "types"):
        step = next(candidate for candidate in full if candidate["name"] == name)
        assert list(step["depends_on"]) == ["environment-compatibility"], (
            f"{name} could run through --keep-going in an unverified environment"
        )
    python_steps = {
        "format": ("module", "ruff"),
        "lint": ("module", "ruff"),
        "types": ("module", "mypy"),
        "web-model": ("script", "scripts/export_web_model.py"),
        "web-page": ("script", "scripts/build_web_ailm.py"),
        "package-distributions": ("module", "pytest"),
        "network-claim-bindings": ("script", "scripts/verify_network_claim_bindings.py"),
        "cache-audit": ("script", "scripts/record_external_benchmark_claims.py"),
        "raw-corpus-reproducibility": ("module", "pytest"),
        "pymdp-sparse-bundle-generate": ("script", "scripts/generate_pymdp_sparse_patch.py"),
        "pymdp-sparse-bundle-check": ("script", "scripts/generate_pymdp_sparse_patch.py"),
        "release-infrastructure-coverage": ("module", "pytest"),
        "tests-full": ("module", "pytest"),
        "gate-audit-full": ("script", "scripts/audit_gates.py"),
        "lean-ledger-check": ("script", "scripts/generate_lean_ledger.py"),
        "lean-negative-control": ("module", "pytest"),
        "manuscript-variables": ("script", "scripts/z_generate_manuscript_variables.py"),
        "hydrated-docs": ("script", "scripts/z_render_docs.py"),
        "hydrated-docs-check": ("script", "scripts/z_render_docs.py"),
        "publication-html-references": ("script", "scripts/z_render_docs.py"),
        "figures": ("script", "scripts/generate_figures.py"),
        "figure-registry": ("script", "scripts/generate_figure_registry.py"),
        "figure-registry-check": ("script", "scripts/generate_figure_registry.py"),
        "pdf-integrity": ("module", "pytest"),
        "manuscript-render-contract": ("module", "pytest"),
        "cover": ("script", "scripts/generate_cover.py"),
        "cover-check": ("script", "scripts/generate_cover.py"),
    }
    assert {
        step["name"] for step in full if step["argv"] and step["argv"][0] == expected_python
    } == set(python_steps)
    for step in full:
        if step["name"] in python_steps:
            assert step["argv"][:5] == [
                expected_python,
                "-I",
                "-S",
                "-B",
                str((PROJECT_ROOT / "scripts" / "verify_release.py").resolve()),
            ]
            assert step["argv"][5:8] == [
                module.CONTROLLED_RUNNER_MARKER,
                *python_steps[step["name"]],
            ]
    for name, arguments in (
        ("hydrated-docs", ["--supporting-protocols"]),
        ("hydrated-docs-check", ["--check", "--supporting-protocols"]),
    ):
        step = next(step for step in full if step["name"] == name)
        assert step["argv"][5:] == [
            module.CONTROLLED_RUNNER_MARKER,
            "script",
            "scripts/z_render_docs.py",
            *arguments,
        ], "full acceptance must generate and check portable protocol destinations"
    type_step = next(step for step in full if step["name"] == "types")
    assert "--explicit-package-bases" in type_step["argv"]
    # The typing gate covers all of src/ailm. A directory-wide `exclude` once hid
    # every figure module (a quarter of the package) behind a green gate; the
    # step passes the directory, so pyproject's exclude would apply silently.
    mypy_config = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )["tool"]["mypy"]
    assert not mypy_config.get("exclude"), (
        f"[tool.mypy] exclude = {mypy_config['exclude']!r} silently narrows the release "
        "typing gate; remove it and fix the excluded modules instead"
    )
    for release_surface in (
        "src/ailm",
        "scripts/verify_release.py",
        "scripts/verify_network_claim_bindings.py",
        "scripts/orchestrate_claim_recorders.py",
        "scripts/warm_call_cache.py",
        "scripts/record_external_benchmark_claims.py",
        "scripts/record_synthetic_benchmark_claims.py",
        "scripts/record_incontext_benchmark_claims.py",
        "scripts/record_logprob_drift_claims.py",
    ):
        assert release_surface in type_step["argv"]
    gate_audit = next(step for step in full if step["name"] == "gate-audit-full")
    assert "--json" in gate_audit["argv"]
    assert "--full" in gate_audit["argv"]
    assert "--limit" not in gate_audit["argv"]
    manuscript_dependencies = set(manuscript_step["depends_on"])
    assert {"tests-full", "gate-audit-full", "lean-build", "lean-negative-control"} <= (
        manuscript_dependencies
    )
    assert "package-distributions" in manuscript_dependencies
    assert "pymdp-sparse-bundle-generate" in manuscript_dependencies
    assert "pymdp-sparse-bundle-check" in manuscript_dependencies
    assert "release-infrastructure-coverage" in manuscript_dependencies


def test_release_infrastructure_modules_import_under_their_coverage_names() -> None:
    """Keep coverage collection bound to the production module paths.

    Several adversarial tests load a copied script under a temporary module
    name.  Canonical imports ensure coverage still attributes those executed
    lines to the same production files named by the release gate.
    """

    assert network_binding_verifier.PROJECT_ROOT == PROJECT_ROOT
    assert canonical_warmer.PROJECT_ROOT == PROJECT_ROOT
    assert publication_verifier.PROJECT_ROOT == PROJECT_ROOT


def test_network_binding_cli_accepts_repeatable_discovered_checkpoint_selectors() -> None:
    discovered = network_binding_verifier._selected_recorders(PROJECT_ROOT, [])
    assert len(discovered) >= 2
    requested = [discovered[1], discovered[0]]
    parsed = network_binding_verifier._parser().parse_args(
        [argument for recorder in requested for argument in ("--recorder", recorder)]
    )
    assert parsed.recorder == requested
    assert network_binding_verifier._selected_recorders(PROJECT_ROOT, parsed.recorder) == tuple(
        requested
    )


def test_network_binding_cli_rejects_duplicate_and_undiscovered_selectors(capsys) -> None:
    discovered = network_binding_verifier._selected_recorders(PROJECT_ROOT, [])
    with pytest.raises(ValueError, match="must be unique"):
        network_binding_verifier._selected_recorders(PROJECT_ROOT, [discovered[0], discovered[0]])

    invalid = "scripts/record_not_discovered_claims.py"
    assert network_binding_verifier.main(["--recorder", invalid]) == 1
    assert "does not name a discovered network recorder" in capsys.readouterr().err


def test_network_binding_verifier_requires_evidence_and_an_executable_semantic_validator(
    tmp_path: Path,
) -> None:
    from ailm.call_cache import CallCache

    root = tmp_path / "semantic-verifier"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    relative = "scripts/record_probe_claims.py"
    recorder = root / relative
    store = CallCache(root / "cache" / "calls.jsonl")

    recorder.write_text(
        "def validate_publication_measurement(receipt, store):\n"
        "    if receipt['measurement'] != {'ok': True}:\n"
        "        raise ValueError('semantic control fired')\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no semantic measurement evidence"):
        network_binding_verifier._validate_semantic_measurements(
            root, {relative: {"measurement": None}}, store
        )

    recorder.write_text('"""No semantic validator."""\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no semantic measurement validator"):
        network_binding_verifier._validate_semantic_measurements(
            root, {relative: {"measurement": {"ok": True}}}, store
        )

    recorder.write_text(
        "def validate_publication_measurement(receipt, store):\n"
        "    if receipt['measurement'] != {'ok': True}:\n"
        "        raise ValueError('semantic control fired')\n",
        encoding="utf-8",
    )
    network_binding_verifier._validate_semantic_measurements(
        root, {relative: {"measurement": {"ok": True}}}, store
    )


def test_distribution_artifact_receipt_parser_is_live_and_fail_closed():
    module = _module()
    wheel = {
        "filename": "ailm-0.1.0-py3-none-any.whl",
        "sha256": "a" * 64,
        "bytes": 123,
        "source_date_epoch": "1700000000",
    }
    sdist = {
        "filename": "ailm-0.1.0.tar.gz",
        "sha256": "b" * 64,
        "bytes": 456,
        "source_date_epoch": "1700000000",
    }
    stdout = "noise\n" + "\n".join(
        module.DISTRIBUTION_MARKER + json.dumps(row, sort_keys=True) for row in (wheel, sdist)
    )
    assert module._distribution_artifacts_from_output(
        stdout, expected_source_date_epoch="1700000000"
    ) == [wheel, sdist]

    # Pytest progress output can overwrite its own line via carriage return,
    # gluing progress text onto the marker line; the parser must still find it.
    glued = stdout.replace(
        "\n" + module.DISTRIBUTION_MARKER,
        " tests/test_wheel_resources.py AILM_DISTRIBUTION_ARTIFACT ".replace(
            "AILM_DISTRIBUTION_ARTIFACT ", module.DISTRIBUTION_MARKER
        ),
        1,
    )
    assert glued != stdout
    assert module._distribution_artifacts_from_output(
        glued, expected_source_date_epoch="1700000000"
    ) == [wheel, sdist]

    planted_bad_digest = stdout.replace("a" * 64, "not-a-digest")
    with pytest.raises(ValueError, match="sha256"):
        module._distribution_artifacts_from_output(planted_bad_digest)
    with pytest.raises(ValueError, match="exactly one wheel"):
        module._distribution_artifacts_from_output(module.DISTRIBUTION_MARKER + json.dumps(wheel))
    with pytest.raises(ValueError, match="disagrees with the release environment"):
        module._distribution_artifacts_from_output(stdout, expected_source_date_epoch="1787361436")


@pytest.mark.timeout(960)
def test_full_profile_uses_exact_template_render_then_validation(tmp_path, monkeypatch):
    # In-process under a uv-launched pytest: scrub the marker (environment
    # isolation). build_steps runs inside the real sidecar python because the
    # in-process sys.executable is "python3" while the sidecar pins "python".
    monkeypatch.delenv("UV_RUN_RECURSION_DEPTH", raising=False)
    module = _module()
    template = tmp_path / "template"
    _init_template_repository(template)
    assert module._validate_template_root(template) == template.resolve()

    code = (
        "import sys, json\n"
        "sys.path.insert(0, 'scripts')\n"
        "import verify_release as m\nfrom pathlib import Path as _P\n"
        f"template = r'{template.resolve()}'\n"
        "full = [{'name': s.name, 'argv': list(s.argv), 'cwd': s.cwd, 'depends_on': list(s.depends_on or ())} "
        "for s in m.build_steps('full', template_root=_P(template))]\n"
        "print(json.dumps(full))\n"
    )
    child = subprocess.run(
        [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-I", "-S", "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=480,
    )
    assert child.returncode == 0, child.stderr
    full = json.loads(child.stdout)
    render = next(step for step in full if step["name"] == "template-render")
    validate = next(step for step in full if step["name"] == "template-validate")
    lean = next(step for step in full if step["name"] == "lean-build")
    lean_negative = next(step for step in full if step["name"] == "lean-negative-control")
    qpdf = next(step for step in full if step["name"] == "pdf-qpdf")

    assert render["cwd"] == validate["cwd"] == str(template.resolve())
    assert render["argv"][1:4] == ["run", "--locked", "python"]
    assert list(render["argv"][-3:]) == [
        "scripts/pipeline/stage_03_render.py",
        "--project",
        "working/active_inference_language_model",
    ]
    assert list(validate["argv"][-3:]) == [
        "scripts/pipeline/stage_04_validate.py",
        "--project",
        "working/active_inference_language_model",
    ]
    assert lean["cwd"] == str(PROJECT_ROOT / "formal")
    assert list(lean["depends_on"]) == ["lean-ledger-check"]
    # The negative-control argv embeds the pinned interpreter, so it is verified
    # structurally here; the identity contract runs in the probe subprocesses.
    assert lean_negative["argv"][0] == str(PROJECT_ROOT / ".venv" / "bin" / "python")
    assert "-I" in lean_negative["argv"] and "-S" in lean_negative["argv"]
    assert (
        "tests/test_formal_isomorphism.py::test_the_negative_control_does_not_compile"
        in lean_negative["argv"]
    )
    assert list(qpdf["argv"][1:]) == [
        "--check",
        str(PROJECT_ROOT / "output/pdf/active_inference_language_model_combined.pdf"),
    ]


def test_step_runner_records_a_real_success_and_failure(tmp_path):
    module = _module()
    success = module._run_step(
        module.Step("success", (sys.executable, "-c", "print('ok')"), 10),
        tmp_path,
    )
    failure = module._run_step(
        module.Step(
            "failure",
            (sys.executable, "-c", "raise SystemExit(7)"),
            10,
        ),
        tmp_path,
    )
    assert success["exit_code"] == 0
    assert success["status"] == "passed"
    assert success["stdout_tail"].strip() == "ok"
    assert failure["exit_code"] == 7
    assert failure["status"] == "failed"
    assert (tmp_path / "success.log").is_file()
    assert (tmp_path / "failure.log").is_file()


def test_step_runner_records_missing_executable_and_requested_cwd(tmp_path):
    module = _module()
    working = tmp_path / "working"
    working.mkdir()
    result = module._run_step(
        module.Step("missing", ("ailm-command-that-does-not-exist",), 10, cwd=str(working)),
        tmp_path / "logs",
    )
    assert result["exit_code"] == 127
    assert result["status"] == "failed"
    assert result["cwd"] == str(working)
    assert "Could not execute" in result["stderr_tail"]


def test_step_runner_applies_only_declared_environment_overrides(tmp_path):
    module = _module()
    step = module.Step(
        "explicit-environment",
        (
            sys.executable,
            "-c",
            "import os; print(os.environ['AILM_RELEASE_TEST_VALUE'])",
        ),
        10,
        environment_overrides=(("AILM_RELEASE_TEST_VALUE", "declared"),),
    )
    result = module._run_step(
        step,
        tmp_path / "logs",
        environment={"AILM_RELEASE_TEST_VALUE": "inherited", "PATH": os.environ["PATH"]},
    )
    assert result["status"] == "passed"
    assert result["stdout_tail"].strip() == "declared"
    assert result["environment_overrides"] == (("AILM_RELEASE_TEST_VALUE", "declared"),)


def test_keep_going_blocks_failed_descendants_but_runs_independent_steps(tmp_path, monkeypatch):
    module = _module()
    blocked_marker = tmp_path / "blocked.txt"
    independent_marker = tmp_path / "independent.txt"
    steps = (
        module.Step("fails", (sys.executable, "-c", "raise SystemExit(9)"), 10),
        module.Step(
            "dependent",
            (
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(blocked_marker)!r}).write_text('bad')",
            ),
            10,
            depends_on=("fails",),
        ),
        module.Step(
            "independent",
            (
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(independent_marker)!r}).write_text('ok')",
            ),
            10,
        ),
        module.Step(
            "transitive",
            (sys.executable, "-c", "raise SystemExit('must not run')"),
            10,
            depends_on=("dependent",),
        ),
    )

    monkeypatch.delenv("UV_RUN_RECURSION_DEPTH", raising=False)
    results = module._execute_steps(
        steps,
        tmp_path / "logs",
        environment=_release_environment(),
        keep_going=True,
    )

    assert [result["status"] for result in results] == [
        "failed",
        "blocked",
        "passed",
        "blocked",
    ]
    assert results[1]["blocked_by"] == ["fails"]
    assert results[3]["blocked_by"] == ["dependent"]
    assert not blocked_marker.exists()
    assert independent_marker.read_text(encoding="utf-8") == "ok"


def test_step_graph_rejects_dependencies_it_cannot_enforce():
    module = _module()
    with pytest.raises(ValueError, match="missing or forward"):
        module._validate_step_graph(
            (
                module.Step("consumer", (sys.executable, "-c", "pass"), 10, ("producer",)),
                module.Step("producer", (sys.executable, "-c", "pass"), 10),
            )
        )
    with pytest.raises(ValueError, match="duplicate"):
        module._validate_step_graph(
            (
                module.Step("same", (sys.executable, "-c", "pass"), 10),
                module.Step("same", (sys.executable, "-c", "pass"), 10),
            )
        )
    with pytest.raises(ValueError, match="environment overrides"):
        module._validate_step_graph(
            (
                module.Step(
                    "invalid-environment",
                    (sys.executable, "-c", "pass"),
                    10,
                    environment_overrides=(("DUPLICATE", "one"), ("DUPLICATE", "two")),
                ),
            )
        )


def test_list_mode_is_read_only_and_complete():
    result = subprocess.run(
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "-I",
            "-S",
            str(SCRIPT),
            "quick",
            "--list",
        ],
        cwd=PROJECT_ROOT,
        env=_release_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "tests-quick" in result.stdout
    assert "gate-audit-bounded" in result.stdout
    assert "tests-full" not in result.stdout


def test_isolated_startup_blocks_inherited_sitecustomize(tmp_path):
    """NEGATIVE CONTROL: caller Python code cannot run before the inventory."""

    poison = tmp_path / "poison"
    poison.mkdir()
    marker = tmp_path / "sitecustomize-ran"
    (poison / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )
    environment = {**_release_environment(), "PYTHONPATH": str(poison)}

    project_python = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    nonisolated = subprocess.run(
        [project_python, str(SCRIPT), "quick", "--list"],
        cwd=PROJECT_ROOT,
        env=_release_environment(**(dict(environment) if isinstance(environment, dict) else {})),
        capture_output=True,
        text=True,
        check=False,
    )
    assert nonisolated.returncode == 2
    assert "requires isolated, stdlib-only Python startup" in nonisolated.stderr
    assert marker.read_text(encoding="utf-8") == "executed"

    marker.unlink()
    isolated = subprocess.run(
        [project_python, "-I", "-S", str(SCRIPT), "quick", "--list"],
        cwd=PROJECT_ROOT,
        env=_release_environment(**(dict(environment) if isinstance(environment, dict) else {})),
        capture_output=True,
        text=True,
        check=False,
    )
    assert isolated.returncode == 0, isolated.stderr
    assert not marker.exists()
    assert "tests-quick" in isolated.stdout

    site_enabled = subprocess.run(
        [project_python, "-I", str(SCRIPT), "quick", "--list"],
        cwd=PROJECT_ROOT,
        env=_release_environment(**(dict(environment) if isinstance(environment, dict) else {})),
        capture_output=True,
        text=True,
        check=False,
    )
    assert site_enabled.returncode == 2
    assert "-I -S" in site_enabled.stderr


@pytest.mark.parametrize("customizer", ["sitecustomize.py", "usercustomize.py"])
def test_stdlib_startup_rejects_venv_customization_without_executing_it(tmp_path, customizer):
    repository = tmp_path / "repository"
    _init_git_repository(repository, with_release_script=True)
    site_packages = (
        repository
        / ".venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    marker = tmp_path / f"{customizer}-executed"
    (site_packages / customizer).write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )

    site_enabled = subprocess.run(
        [
            _repository_python(repository),
            "-I",
            str(repository / "scripts" / "verify_release.py"),
            "quick",
            "--list",
        ],
        cwd=repository,
        env=_release_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert site_enabled.returncode == 2
    if customizer == "sitecustomize.py":
        if _stdlib_sitecustomize(repository) is None:
            assert marker.read_text(encoding="utf-8") == "executed"
            marker.unlink()
        else:
            assert not marker.exists(), "a shadowed venv sitecustomize must not execute"
    else:
        assert not marker.exists(), "-I should already suppress user-site customization"

    result = subprocess.run(
        [
            _repository_python(repository),
            "-I",
            "-S",
            str(repository / "scripts" / "verify_release.py"),
            "quick",
            "--list",
        ],
        cwd=repository,
        env=_release_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "unsafe virtual-environment startup" in result.stderr
    assert customizer.removesuffix(".py") in result.stderr
    assert not marker.exists(), "-S failed to keep the planted customizer inert"


@pytest.mark.parametrize(
    ("name", "content", "expected"),
    [
        ("planted_exec.pth", "import pathlib; pathlib.Path('bad').touch()\n", "not owned"),
        ("planted_path.pth", "uncontrolled-import-root\n", "not owned"),
    ],
)
def test_stdlib_startup_rejects_unowned_pth_surfaces(tmp_path, name, content, expected):
    repository = tmp_path / "repository"
    _init_git_repository(repository, with_release_script=True)
    site_packages = (
        repository
        / ".venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    (site_packages / name).write_text(content, encoding="utf-8")

    result = subprocess.run(
        [
            _repository_python(repository),
            "-I",
            "-S",
            str(repository / "scripts" / "verify_release.py"),
            "quick",
            "--list",
        ],
        cwd=repository,
        env=_release_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert expected in result.stderr


def test_distribution_record_cannot_authorize_changed_executable_pth(tmp_path):
    module = _module()
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "src").mkdir()
    site_packages = (
        repository
        / ".venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    site_packages.mkdir(parents=True)
    content = b"import pathlib; pathlib.Path('planted').touch()\n"
    planted = site_packages / "a1_coverage.pth"
    planted.write_bytes(content)
    distribution = site_packages / "coverage-999.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: coverage\nVersion: 999\n",
        encoding="utf-8",
    )
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip("=")
    (distribution / "RECORD").write_text(
        f"a1_coverage.pth,sha256={digest},{len(content)}\n",
        encoding="utf-8",
    )

    state = module._venv_startup_state(repository)
    assert state["integrity_ok"] is False
    assert any("approved executable entry bytes changed" in error for error in state["errors"])


def test_startup_audit_rejects_symlinked_pth_and_venv_components(tmp_path):
    module = _module()
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "src").mkdir()
    site_packages = (
        repository
        / ".venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    site_packages.mkdir(parents=True)
    external = tmp_path / "external.pth"
    external.write_text("import pathlib\n", encoding="utf-8")
    (site_packages / "planted.pth").symlink_to(external)

    pth_state = module._venv_startup_state(repository)
    assert pth_state["integrity_ok"] is False
    assert any("not a real regular file" in error for error in pth_state["errors"])

    routed_repository = tmp_path / "routed-repository"
    routed_repository.mkdir()
    (routed_repository / "src").mkdir()
    external_venv = tmp_path / "external-venv"
    (
        external_venv
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    ).mkdir(parents=True)
    (routed_repository / ".venv").symlink_to(external_venv, target_is_directory=True)

    routed_state = module._venv_startup_state(routed_repository)
    assert routed_state["integrity_ok"] is False
    assert any("not a real directory" in error for error in routed_state["errors"])


@pytest.mark.timeout(960)
def test_controlled_python_bootstrap_runs_pinned_modules_without_site_startup(
    tmp_path, monkeypatch
):
    # In-process under a uv-launched pytest: scrub the marker (environment
    # isolation only — no dependency replacement). Interpreter identity is
    # exercised through the spawned-sidecar probe helper.
    monkeypatch.delenv("UV_RUN_RECURSION_DEPTH", raising=False)
    assert _project_python_probe() == str(PROJECT_ROOT / ".venv" / "bin" / "python")
    module = _module()
    state = module._venv_startup_state()
    assert state["integrity_ok"] is True
    assert state["pth_entries"]
    assert all(entry["inert_under_controlled_startup"] for entry in state["pth_entries"])

    # Build the controlled argv inside the spawned sidecar python: in-process
    # sys.executable is "python3" and _controlled_python_argv embeds the pinned
    # interpreter path, so the real check must run where identity holds.
    code = (
        "import sys; sys.path.insert(0, 'scripts'); "
        "import verify_release as m; print(' '.join(m._controlled_python_argv('module', 'ruff', '--version')))"
    )
    probe = subprocess.run(
        [str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-I", "-S", "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=480,
    )
    assert probe.returncode == 0, probe.stderr
    argv_shlexed = probe.stdout.strip()

    result = module._run_step(
        module.Step(
            "controlled-ruff",
            tuple(argv_shlexed.split(" ")),
            # A cold isolated interpreter start on the external drive exceeds 30s
            # under concurrent recorder I/O; the bound tests the step contract,
            # not start-up latency.
            300,
        ),
        tmp_path / "logs",
        environment=module._base_release_environment(),
    )
    assert result["status"] == "passed", result["stderr_tail"]
    assert "ruff " in result["stdout_tail"]


def test_wrong_interpreter_cannot_list_or_run_a_release_inventory(tmp_path):
    """NEGATIVE CONTROL: a base interpreter must fail before any step is trusted."""

    # The refusal receipt fingerprints its checkout. Exercise the real CLI in
    # a bounded repository rather than hashing unrelated project work here.
    repository = tmp_path / "repository"
    _init_git_repository(repository, with_release_script=True)
    script = repository / "scripts" / SCRIPT.name
    expected = repository / ".venv" / "bin" / "python"
    base_python = Path(sys._base_executable).absolute()
    assert base_python != expected, "negative control requires a distinct base interpreter"

    result = subprocess.run(
        [str(base_python), "-I", "-S", str(script), "quick", "--list"],
        cwd=repository,
        env=_release_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "must be invoked as" in result.stderr
    assert "Do not use uv run" in result.stderr
    assert "tests-quick" not in result.stdout

    receipt_path = tmp_path / "wrong-interpreter-receipt.json"
    run = subprocess.run(
        [
            str(base_python),
            "-I",
            "-S",
            str(script),
            "quick",
            "--receipt",
            str(receipt_path),
        ],
        cwd=repository,
        env=_release_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 1, run.stdout + run.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["steps"] == []
    assert "interpreter prerequisite failed" in receipt["refusal"]
    assert receipt["interpreter"] == str(base_python)
    assert receipt["expected_interpreter"] == str(expected)


def test_uv_run_marker_is_refused_even_with_the_exact_interpreter(tmp_path):
    """NEGATIVE CONTROL: path identity alone does not detect a sidecar uv run."""

    repository = tmp_path / "repository"
    _init_git_repository(repository, with_release_script=True)
    receipt_path = tmp_path / "uv-run-receipt.json"
    result = subprocess.run(
        [
            _repository_python(repository),
            "-I",
            "-S",
            str(repository / "scripts" / "verify_release.py"),
            "quick",
            "--receipt",
            str(receipt_path),
        ],
        cwd=repository,
        env=_release_environment(uv_run_recursion_depth="1"),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["steps"] == []
    assert "launched through uv run" in receipt["refusal"]
    assert receipt["interpreter"] == _repository_python(repository)
    assert receipt["publication_acceptance"] is False


def test_hosted_ci_uses_the_project_interpreter_and_does_not_fake_full_acceptance():
    """CI is quick-only because its checkout lacks two full-profile inputs."""

    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    assert _ci_release_contract_violations(workflow) == []
    assert workflow.index("run: uv sync --locked --extra dev") < workflow.index(
        "run: .venv/bin/python -I -S scripts/verify_release.py quick"
    )
    for prerequisite in (
        "external-call cache",
        "Ollama server",
        "every declared probe",
        "docxology/template",
        "Lean",
        "qpdf",
    ):
        assert prerequisite in workflow

    # NEGATIVE CONTROLS: both historical CI defects must make the policy fail.
    wrong_python = workflow.replace(
        "run: .venv/bin/python -I -S scripts/verify_release.py quick",
        "run: uv run --locked python scripts/verify_release.py quick",
    )
    assert "uv-run-sidecar" in _ci_release_contract_violations(wrong_python)
    nonisolated_python = workflow.replace(".venv/bin/python -I -S", ".venv/bin/python -I")
    assert "release-python-not-isolated" in _ci_release_contract_violations(nonisolated_python)
    false_full = (
        workflow
        + "\n  full:\n    steps:\n      - run: .venv/bin/python -I -S scripts/verify_release.py full\n"
    )
    assert "hosted-full-without-local-inputs" in _ci_release_contract_violations(false_full)
    undocumented_live_input = workflow.replace("Ollama server", "model service")
    assert "missing-full-prerequisite:Ollama server" in _ci_release_contract_violations(
        undocumented_live_input
    )
    floating_action = workflow.replace(
        "actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0",
        "actions/checkout@main",
    )
    assert any(
        violation.startswith("action-not-full-sha:actions/checkout@main")
        for violation in _ci_release_contract_violations(floating_action)
    )
    write_permission = workflow.replace("contents: read", "contents: write")
    assert "permissions-not-contents-read" in _ci_release_contract_violations(write_permission)
    persisted_credentials = workflow.replace(
        "persist-credentials: false", "persist-credentials: true"
    )
    assert "checkout-persists-credentials" in _ci_release_contract_violations(persisted_credentials)
    event_in_shell = workflow.replace(
        "run: .venv/bin/python -I -S scripts/verify_release.py quick",
        "run: echo '${{ github.event.pull_request.title }}'",
    )
    assert "event-expression-in-run" in _ci_release_contract_violations(event_in_shell)


def test_release_environment_uses_audited_commit_and_project_cache_dirs(tmp_path, monkeypatch):
    module = _module()
    repository = tmp_path / "repository"
    commit = _init_git_repository(repository)
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    monkeypatch.delenv("FORCE_SOURCE_DATE", raising=False)

    environment, recorded = module._release_environment(commit, root=repository)

    assert recorded["SOURCE_DATE_EPOCH"] == "1700000000"
    assert recorded["source_date_epoch_origin"] == "audited-input-commit"
    assert environment["SOURCE_DATE_EPOCH"] == recorded["SOURCE_DATE_EPOCH"]
    assert environment["FORCE_SOURCE_DATE"] == recorded["FORCE_SOURCE_DATE"] == "1"
    assert recorded["force_source_date_origin"] == "release-policy"
    assert recorded["inherited_force_source_date_overridden"] is False
    for variable in ("UV_CACHE_DIR", "MPLCONFIGDIR"):
        path = Path(recorded[variable])
        assert path.is_dir()
        assert repository.resolve() in path.parents
        # Dependency caches can contain symlinks. The publication inventory
        # rejects those even below hidden output directories.
        assert (repository.resolve() / "output") not in path.parents
        assert (repository.resolve() / ".venv") in path.parents
        assert environment[variable] == str(path)
    assert environment["MPLBACKEND"] == recorded["MPLBACKEND"] == "Agg"
    assert environment["UV_CONFIG_FILE"] == recorded["UV_CONFIG_FILE"]
    assert Path(recorded["UV_CONFIG_FILE"]) == repository / "release-uv.toml"
    assert (
        recorded["uv_config_sha256"]
        == hashlib.sha256((repository / "release-uv.toml").read_bytes()).hexdigest()
    )

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000001")
    inherited, inherited_record = module._release_environment(commit, root=repository)
    assert inherited["SOURCE_DATE_EPOCH"] == "1700000000"
    assert inherited_record["source_date_epoch_origin"] == "audited-input-commit"
    assert inherited_record["inherited_source_date_epoch_overridden"] is True

    # NEGATIVE CONTROLS: malformed or negative caller values cannot perturb the
    # release timestamp because the audited commit is the only accepted source.
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "not-an-epoch")
    malformed, malformed_record = module._release_environment(commit, root=repository)
    assert malformed["SOURCE_DATE_EPOCH"] == "1700000000"
    assert malformed_record["inherited_source_date_epoch_overridden"] is True
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "-1")
    negative, _ = module._release_environment(commit, root=repository)
    assert negative["SOURCE_DATE_EPOCH"] == "1700000000"

    # Caller flags cannot opt XeLaTeX calendar primitives back into wall time.
    for caller_flag in ("0", "malformed"):
        monkeypatch.setenv("FORCE_SOURCE_DATE", caller_flag)
        assert "FORCE_SOURCE_DATE" not in module._base_release_environment()
        forced, forced_record = module._release_environment(commit, root=repository)
        assert forced["SOURCE_DATE_EPOCH"] == "1700000000"
        assert forced["FORCE_SOURCE_DATE"] == forced_record["FORCE_SOURCE_DATE"] == "1"
        assert forced_record["force_source_date_origin"] == "release-policy"
        assert forced_record["inherited_force_source_date_overridden"] is True


def test_repository_snapshot_ignores_ambient_git_repository_routing(tmp_path, monkeypatch):
    """NEGATIVE CONTROL: caller Git routing cannot substitute another checkout."""

    module = _module()
    repository = tmp_path / "repository"
    expected_commit = _init_git_repository(repository)
    alternate = tmp_path / "alternate"
    alternate_commit = _init_git_repository(alternate)
    assert alternate_commit != expected_commit or alternate.resolve() != repository.resolve()

    monkeypatch.setenv("GIT_DIR", str(alternate / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(alternate))
    monkeypatch.setenv("GIT_INDEX_FILE", str(alternate / ".git" / "index"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "attacker.gitconfig"))

    initial = module._repository_state(repository)
    assert initial.head == expected_commit
    assert initial.dirty is False

    (repository / "tracked.txt").write_text("mutated project input\n", encoding="utf-8")
    final = module._repository_state(repository)
    assert final.head == expected_commit
    assert final.dirty is True
    assert final.fingerprint != initial.fingerprint

    # The commit epoch used by the release environment must be read from the
    # same explicitly supplied checkout under the hostile caller environment.
    environment, recorded = module._release_environment(
        expected_commit, root=repository, create_cache_dirs=False
    )
    assert environment["SOURCE_DATE_EPOCH"] == "1700000000"
    assert recorded["SOURCE_DATE_EPOCH"] == "1700000000"


def test_repository_snapshot_forces_real_filemode_and_neutralizes_external_excludes(
    tmp_path,
):
    module = _module()
    repository = tmp_path / "repository"
    _init_git_repository(repository)

    tracked = repository / "tracked.txt"
    tracked.chmod(tracked.stat().st_mode | stat.S_IXUSR)
    _git(repository, "config", "core.filemode", "false")
    assert _git(repository, "status", "--porcelain=v1") == "", (
        "negative control requires local core.filemode=false to hide the planted mode change"
    )
    mode_state = module._repository_state(repository)
    assert mode_state.dirty is True
    assert mode_state.unstaged_sha256 != hashlib.sha256(b"").hexdigest()

    tracked.chmod(tracked.stat().st_mode & ~0o111)
    external_excludes = tmp_path / "external-excludes"
    external_excludes.write_text("hidden-by-global-rule.txt\n", encoding="utf-8")
    _git(repository, "config", "core.excludesFile", str(external_excludes))
    hidden = repository / "hidden-by-global-rule.txt"
    hidden.write_text("release input\n", encoding="utf-8")
    assert _git(repository, "status", "--porcelain=v1") == "", (
        "negative control requires core.excludesFile to hide the planted input"
    )
    excluded_state = module._repository_state(repository)
    assert excluded_state.dirty is True
    assert any("hidden-by-global-rule.txt" in row for row in excluded_state.status)


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_repository_boundary_rejects_index_visibility_flags(tmp_path, flag):
    module = _module()
    repository = tmp_path / "repository"
    _init_git_repository(repository)
    _git(repository, "update-index", flag, "tracked.txt")

    state = module._repository_state(repository)
    assert state.index_flags
    refusal = module._repository_boundary_refusal(state, label="test checkout")
    assert refusal is not None
    assert "assume-unchanged or skip-worktree" in refusal
    assert "tracked.txt" in refusal


def test_release_refuses_hidden_index_state_before_any_step(tmp_path):
    repository = tmp_path / "repository"
    _init_git_repository(repository, with_release_script=True)
    _git(repository, "update-index", "--assume-unchanged", "tracked.txt")
    receipt_path = tmp_path / "index-refusal.json"

    result = subprocess.run(
        [
            _repository_python(repository),
            "-I",
            "-S",
            str(repository / "scripts" / "verify_release.py"),
            "quick",
            "--receipt",
            str(receipt_path),
        ],
        cwd=repository,
        env=_release_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["steps"] == []
    assert "assume-unchanged or skip-worktree" in receipt["refusal"]
    assert receipt["git"]["initial_state"]["index_flags"]


def test_external_tools_and_release_path_ignore_a_poisoned_caller_path(tmp_path, monkeypatch):
    module = _module()
    # This test needs a real commit for SOURCE_DATE_EPOCH, not a digest of the
    # caller's entire dirty worktree. Repository-state integrity has its own
    # real-file mutation tests below.
    repository = tmp_path / "repository"
    head = _init_git_repository(repository)
    poison = tmp_path / "poison-bin"
    poison.mkdir()
    markers: list[Path] = []
    for name in module.TOOL_VERSION_ARGUMENTS:
        marker = tmp_path / f"{name}-executed"
        markers.append(marker)
        executable = poison / name
        executable.write_text(
            f"#!/bin/sh\nprintf planted > {str(marker)!r}\nexit 0\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(poison))

    toolchain = module._toolchain_state(check_versions=True)
    environment, recorded = module._release_environment(
        head,
        root=repository,
        create_cache_dirs=False,
    )
    assert str(poison) not in environment["PATH"].split(os.pathsep)
    assert recorded["PATH"] == environment["PATH"] == toolchain["PATH"]
    assert environment["HOME"] == str(module._login_home())
    assert recorded["home_origin"] == "login-account-database"
    for row in toolchain["tools"].values():
        if row["available"]:
            assert poison not in Path(row["realpath"]).parents
            assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
            assert row["size_bytes"] > 0
            assert row["version_argv"][0] == row["path"]
    assert not any(marker.exists() for marker in markers)


def test_tool_execution_preserves_a_basename_sensitive_dispatch_proxy(tmp_path):
    """NEGATIVE CONTROL: resolving a dispatcher target changes its command."""
    module = _module()
    target = tmp_path / "dispatcher"
    target.write_text(
        "#!/bin/sh\n"
        'if [ "${0##*/}" = lake ]; then\n'
        "  printf 'Lake proxy invoked\\n'\n"
        "  exit 0\n"
        "fi\n"
        "printf 'wrong dispatcher basename: %s\\n' \"${0##*/}\" >&2\n"
        "exit 41\n",
        encoding="utf-8",
    )
    target.chmod(0o755)
    proxy = tmp_path / "lake"
    proxy.symlink_to(target.name)
    toolchain = {
        "tools": {
            "lake": {
                "usable": True,
                "path": str(proxy),
                "realpath": str(target.resolve()),
            }
        }
    }

    selected = module._tool_path(toolchain, "lake")
    correct = subprocess.run([selected, "--version"], capture_output=True, text=True, check=False)
    planted_wrong = subprocess.run(
        [str(target.resolve()), "--version"], capture_output=True, text=True, check=False
    )

    assert selected == str(proxy)
    assert correct.returncode == 0 and "Lake proxy invoked" in correct.stdout
    assert planted_wrong.returncode == 41
    assert "wrong dispatcher basename" in planted_wrong.stderr


@pytest.mark.timeout(480)
def test_receipt_binds_tool_realpaths_digests_versions_and_availability(tmp_path):
    repository = tmp_path / "repository"
    _init_git_repository(repository, with_release_script=True)
    project = "[project]\nname = 'ailm-tool-receipt-fixture'\nversion = '0.0.0'\n"
    project += "requires-python = '>=3.11'\n"
    invalid_lock = "version = definitely-invalid-fixture-value\n"
    (repository / "pyproject.toml").write_text(project, encoding="utf-8")
    (repository / "uv.lock").write_text(invalid_lock, encoding="utf-8")
    _git(repository, "add", "pyproject.toml", "uv.lock")
    _git(repository, "commit", "-q", "-m", "isolate the tool receipt lock failure")
    receipt_path = tmp_path / "tool-receipt.json"

    result = subprocess.run(
        [
            _repository_python(repository),
            "-I",
            "-S",
            str(repository / "scripts" / "verify_release.py"),
            "quick",
            "--receipt",
            str(receipt_path),
        ],
        cwd=repository,
        env=_release_environment(PATH=str(tmp_path / "caller-path-is-not-used")),
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert [step["name"] for step in receipt["steps"]] == ["lock"]
    assert receipt["steps"][0]["status"] == "failed"
    assert receipt["steps"][0]["exit_code"] != 0
    assert "uv.lock" in receipt["steps"][0]["stderr_tail"]
    assert "definitely-invalid-fixture-value" in receipt["steps"][0]["stderr_tail"]
    assert (repository / "pyproject.toml").read_text(encoding="utf-8") == project
    assert (repository / "uv.lock").read_text(encoding="utf-8") == invalid_lock
    toolchain = receipt["toolchain"]["initial_state"]
    assert toolchain["path_policy"] == "fixed-search-path-preserve-dispatch-basename-v2"
    for name in ("git", "uv"):
        row = toolchain["tools"][name]
        assert row["available"] is True
        assert row["usable"] is True
        assert Path(row["realpath"]).is_absolute()
        assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
        assert row["version_exit_code"] == 0
        assert row["version_output"]


def test_release_environment_defeats_collect_only_and_other_inherited_poison(tmp_path, monkeypatch):
    """NEGATIVE CONTROL: an inherited pytest option must not turn tests into collection."""
    module = _module()
    repository = tmp_path / "repository"
    commit = _init_git_repository(repository)
    poisons = {
        "PYTEST_ADDOPTS": "--collect-only",
        "PYTHONOPTIMIZE": "1",
        "PYTHONPATH": str(tmp_path / "injected"),
        "COVERAGE_FILE": str(tmp_path / "fake.coverage"),
        "HTTP_PROXY": "http://127.0.0.1:9",
        "UV_CONFIG_FILE": str(tmp_path / "untrusted-uv.toml"),
    }
    for name, value in poisons.items():
        monkeypatch.setenv(name, value)

    environment, recorded = module._release_environment(commit, root=repository)
    assert (set(poisons) - {"UV_CONFIG_FILE"}).isdisjoint(environment)
    assert environment["UV_CONFIG_FILE"] != poisons["UV_CONFIG_FILE"]
    assert recorded["inherited_environment_policy"] == "allowlist-v1"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONHASHSEED"] == "0"
    assert environment["UV_CONFIG_FILE"] == str(repository / "release-uv.toml")

    planted_failure = tmp_path / "test_release_poison_control.py"
    planted_failure.write_text("def test_load_bearing():\n    assert False\n", encoding="utf-8")
    step = module.Step(
        "pytest-poison-negative-control",
        (sys.executable, "-m", "pytest", str(planted_failure), "-q", "--no-cov"),
        60,
        cwd=str(tmp_path),
    )
    result = module._run_step(step, tmp_path / "logs", environment=environment)
    assert result["exit_code"] != 0
    assert result["status"] == "failed"
    assert "1 failed" in result["stdout_tail"]


@pytest.mark.timeout(480)
def test_release_uv_config_defeats_a_poisoned_user_config(tmp_path, monkeypatch):
    """NEGATIVE CONTROL: caller HOME cannot inject uv behavior into release steps."""

    uv = shutil.which("uv")
    if uv is None:
        pytest.fail("uv is required to verify configuration isolation")
    repository = tmp_path / "repository"
    commit = _init_git_repository(repository)
    (repository / "pyproject.toml").write_text(
        "[project]\n"
        "name = 'ailm-uv-config-control'\n"
        "version = '0.0.0'\n"
        "requires-python = '>=3.11'\n"
        "dependencies = []\n\n"
        "[tool.uv]\n"
        "package = false\n",
        encoding="utf-8",
    )
    locked = subprocess.run(
        [uv, "--config-file", str(repository / "release-uv.toml"), "lock", "--offline"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    assert locked.returncode == 0, locked.stderr
    # The production runner starts inside an existing project environment.
    # Give this real-uv fixture the same prerequisite before caches are placed
    # beneath .venv; an empty directory is not a Python environment.
    synced = subprocess.run(
        [
            uv,
            "--config-file",
            str(repository / "release-uv.toml"),
            "sync",
            "--locked",
            "--offline",
            "--python",
            sys.executable,
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    assert synced.returncode == 0, synced.stderr
    poison_home = tmp_path / "poison-home"
    poison = poison_home / ".config" / "uv" / "uv.toml"
    poison.parent.mkdir(parents=True)
    poison.write_text("this is invalid TOML = [\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(poison_home))

    environment, recorded = _module()._release_environment(commit, root=repository)
    assert recorded["UV_CONFIG_FILE"] == str(repository / "release-uv.toml")
    checked = subprocess.run(
        [uv, "lock", "--check"],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    assert checked.returncode == 0, checked.stderr
    assert str(poison) not in checked.stderr


@pytest.mark.timeout(480)
def test_exact_environment_check_rejects_an_undeclared_distribution(tmp_path):
    """NEGATIVE CONTROL: release sync must detect an importable extra package."""

    uv = shutil.which("uv")
    if uv is None:
        pytest.fail("uv is required to verify the release environment contract")

    project = tmp_path / "exact-environment"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        "[project]\n"
        "name = 'ailm-exact-environment-control'\n"
        "version = '0.0.0'\n"
        "requires-python = '>=3.11'\n"
        "dependencies = []\n\n"
        "[tool.uv]\n"
        "package = false\n",
        encoding="utf-8",
    )
    locked = subprocess.run(
        [uv, "lock", "--offline"],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    assert locked.returncode == 0, locked.stderr
    synced = subprocess.run(
        [uv, "sync", "--locked", "--python", sys.executable],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    assert synced.returncode == 0, synced.stderr

    interpreter = project / ".venv" / "bin" / "python"
    purelib_result = subprocess.run(
        [str(interpreter), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert purelib_result.returncode == 0, purelib_result.stderr
    planted = Path(purelib_result.stdout.strip()) / "ailm_extraneous_fixture-1.0.dist-info"
    planted.mkdir()
    (planted / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: ailm-extraneous-fixture\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (planted / "RECORD").write_text("", encoding="utf-8")

    exact = subprocess.run(
        [uv, "sync", "--check", "--locked", "--python", str(interpreter)],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    assert exact.returncode != 0, (
        "exact locked sync accepted an undeclared distribution:\n" + exact.stdout + exact.stderr
    )

    # The historical flag is the live control: it deliberately tolerates extras.
    inexact = subprocess.run(
        [
            uv,
            "sync",
            "--check",
            "--locked",
            "--inexact",
            "--python",
            str(interpreter),
        ],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    assert inexact.returncode == 0, inexact.stderr


def test_release_coverage_gate_is_read_from_pyproject_not_hardcoded(tmp_path):
    """The full-profile floor is the pyproject floor, so one edit cannot drift.

    The floor was previously a literal in the runner. When the owner lowered
    ``fail_under`` in ``pyproject.toml`` the release gate silently kept the old
    value, so the publication profile enforced a number no source file declared.
    """
    module = _module()
    declared = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    expected = declared["tool"]["coverage"]["report"]["fail_under"]
    assert module._coverage_floor() == expected
    assert module._coverage_gate_argument() == f"--cov-fail-under={expected}"

    # POSITIVE CONTROL: a different declared floor is what the gate carries.
    other = tmp_path / "other"
    other.mkdir()
    (other / "pyproject.toml").write_text(
        "[tool.coverage.report]\nfail_under = 42\n", encoding="utf-8"
    )
    assert module._coverage_gate_argument(other) == "--cov-fail-under=42"

    # NEGATIVE CONTROLS: a missing or malformed floor refuses rather than
    # defaulting to any remembered number.
    missing = tmp_path / "missing"
    missing.mkdir()
    (missing / "pyproject.toml").write_text("[tool.coverage.report]\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no .*fail_under"):
        module._coverage_floor(missing)
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "pyproject.toml").write_text(
        "[tool.coverage.report]\nfail_under = 'ninety'\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="not a percentage"):
        module._coverage_floor(malformed)


def test_pytest_receipt_requires_execution_and_coverage_evidence():
    module = _module()
    step = module.Step(
        "tests-full",
        (
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "--cov=ailm",
            # A synthetic floor: this test exercises the evidence parser's
            # cross-check of the reported total against the flag it was run
            # with, independent of the project's declared pyproject floor.
            "--cov-fail-under=95",
        ),
        60,
    )
    output = "123 passed in 4.2s\nTOTAL                         1000     40    96%\n"
    assert module._pytest_execution_evidence(step, output, "") == (123, 0, 96.0)

    with pytest.raises(ValueError, match="no executed passing tests"):
        module._pytest_execution_evidence(step, "123 tests collected in 0.2s", "")
    with pytest.raises(ValueError, match="below 95.0%"):
        module._pytest_execution_evidence(
            step,
            "123 passed in 4.2s\nTOTAL                         1000     60    94%\n",
            "",
        )

    mandatory = module.Step(
        "raw-corpus-reproducibility",
        (sys.executable, "-m", "pytest", "tests/"),
        60,
        forbid_pytest_skips=True,
    )
    with pytest.raises(ValueError, match="mandatory release evidence"):
        module._pytest_execution_evidence(mandatory, "4 passed, 1 skipped in 1.0s", "")
    with pytest.raises(ValueError, match="xfailed tests"):
        module._pytest_execution_evidence(mandatory, "4 passed, 1 xfailed in 1.0s", "")


def test_mandatory_pytest_step_cannot_hide_a_failure_behind_xfail(tmp_path):
    module = _module()
    planted = tmp_path / "test_expected_failure.py"
    planted.write_text(
        "import pytest\n\n"
        "def test_control_passes():\n"
        "    assert True\n\n"
        "@pytest.mark.xfail(reason='planted mandatory failure')\n"
        "def test_mandatory_failure():\n"
        "    assert False\n",
        encoding="utf-8",
    )
    hidden = module.Step(
        "mandatory-hidden-xfail",
        (sys.executable, "-m", "pytest", str(planted), "-q", "--no-cov"),
        60,
        cwd=str(tmp_path),
        forbid_pytest_skips=True,
    )
    hidden_result = module._run_step(hidden, tmp_path / "hidden-logs")
    assert hidden_result["status"] == "failed"
    assert "xfailed tests" in hidden_result["stderr_tail"]

    exposed = module.Step(
        "mandatory-runxfail",
        (
            sys.executable,
            "-m",
            "pytest",
            str(planted),
            "-q",
            "--no-cov",
            "--runxfail",
        ),
        60,
        cwd=str(tmp_path),
        forbid_pytest_skips=True,
    )
    exposed_result = module._run_step(exposed, tmp_path / "exposed-logs")
    assert exposed_result["status"] == "failed"
    assert "1 failed" in exposed_result["stdout_tail"]


def test_mandatory_pytest_step_rejects_a_planted_unexpected_skip(tmp_path):
    """NEGATIVE CONTROL: pytest's successful skip exit is not publication evidence."""
    module = _module()
    planted = tmp_path / "test_unexpected_skip.py"
    planted.write_text(
        "import pytest\n\n"
        "def test_control_passes():\n"
        "    assert True\n\n"
        "def test_mandatory_evidence():\n"
        "    pytest.skip('planted missing publication input')\n",
        encoding="utf-8",
    )
    mandatory = module.Step(
        "mandatory-unexpected-skip",
        (sys.executable, "-m", "pytest", str(planted), "-q", "--no-cov"),
        60,
        cwd=str(tmp_path),
        forbid_pytest_skips=True,
    )

    result = module._run_step(mandatory, tmp_path / "logs")

    assert "1 skipped" in result["stdout_tail"], (
        "the planted test did not exercise pytest's successful skip path"
    )
    assert result["exit_code"] == 1, "the release wrapper did not convert the skip to failure"
    assert result["status"] == "failed"
    assert "reported 1 skipped tests" in result["stderr_tail"]


def test_publication_acceptance_is_distinct_from_development_success():
    module = _module()

    assert module._is_publication_acceptance(
        "full", successful=True, development_allow_dirty_full=False
    )
    assert not module._is_publication_acceptance(
        "full", successful=True, development_allow_dirty_full=True
    )
    assert not module._is_publication_acceptance(
        "quick", successful=True, development_allow_dirty_full=False
    )
    assert not module._is_publication_acceptance(
        "full", successful=False, development_allow_dirty_full=False
    )

    with pytest.raises(ValueError, match="valid only with the full profile"):
        module.run_profile("quick", development_allow_dirty_full=True)


def test_repository_state_fingerprints_unstaged_staged_and_untracked_content(tmp_path):
    module = _module()
    repository = tmp_path / "repository"
    _init_git_repository(repository)

    clean = module._repository_state(repository)
    assert not clean.dirty

    tracked = repository / "tracked.txt"
    tracked.write_text("unstaged\n", encoding="utf-8")
    unstaged = module._repository_state(repository)
    assert unstaged.dirty
    assert unstaged.unstaged_sha256 != clean.unstaged_sha256
    assert unstaged.fingerprint != clean.fingerprint

    _git(repository, "add", "tracked.txt")
    staged = module._repository_state(repository)
    assert staged.staged_sha256 != clean.staged_sha256
    assert staged.fingerprint != unstaged.fingerprint

    untracked_path = repository / "untracked.txt"
    untracked_path.write_text("one\n", encoding="utf-8")
    untracked_one = module._repository_state(repository)
    untracked_path.write_text("two\n", encoding="utf-8")
    untracked_two = module._repository_state(repository)
    assert untracked_one.untracked_sha256 != untracked_two.untracked_sha256
    assert untracked_one.fingerprint != untracked_two.fingerprint


def test_call_cache_receipt_fingerprints_ignored_content(tmp_path):
    module = _module()
    missing = module._call_cache_state(tmp_path)
    assert missing["schema"] == "ailm-call-cache-file-state-v2"
    assert missing["path"] == "cache/bench_calls.jsonl"
    assert missing["exists"] is False
    assert missing["path_kind"] == "missing"
    assert missing["integrity_ok"] is True
    assert missing["records"] == missing["size_bytes"] == 0
    assert missing["sha256"] is None
    assert re.fullmatch(r"[0-9a-f]{64}", str(missing["fingerprint"]))

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache = cache_dir / "bench_calls.jsonl"
    cache.write_text('{"key":"one"}\n', encoding="utf-8")
    first = module._call_cache_state(tmp_path)
    assert first["exists"] is True
    assert first["records"] == 1
    assert first["size_bytes"] == cache.stat().st_size

    cache.write_text('{"key":"two"}\n', encoding="utf-8")
    second = module._call_cache_state(tmp_path)
    assert second["sha256"] != first["sha256"]
    assert second["fingerprint"] != first["fingerprint"]

    binding_dir = cache_dir / "network_claim_bindings"
    binding_dir.mkdir()
    binding = binding_dir / "record_probe_claims.json"
    binding.write_text('{"receipt":"one"}\n', encoding="utf-8")
    with_binding = module._call_cache_state(tmp_path)
    binding_state = with_binding["network_claim_bindings"]
    assert binding_state["integrity_ok"] is True
    assert [row["path"] for row in binding_state["files"]] == [
        "cache/network_claim_bindings/record_probe_claims.json"
    ]
    binding.write_text('{"receipt":"two"}\n', encoding="utf-8")
    changed_binding = module._call_cache_state(tmp_path)
    assert changed_binding["fingerprint"] != with_binding["fingerprint"]


def test_call_cache_receipt_refuses_symlinked_network_binding(tmp_path):
    """NEGATIVE CONTROL: ignored provenance receipts cannot escape the checkout."""

    module = _module()
    binding_dir = tmp_path / "cache" / "network_claim_bindings"
    binding_dir.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    (binding_dir / "record_probe_claims.json").symlink_to(outside)

    state = module._call_cache_state(tmp_path)
    assert state["integrity_ok"] is False
    assert state["network_claim_bindings"]["integrity_ok"] is False
    assert "not a real file" in state["error"]


def test_call_cache_receipt_refuses_symlink_and_special_paths(tmp_path):
    """NEGATIVE CONTROLS: the release snapshot never follows an ignored cache."""

    module = _module()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"external":true}\n', encoding="utf-8")
    cache = cache_dir / "bench_calls.jsonl"
    cache.symlink_to(outside)

    linked = module._call_cache_state(tmp_path)
    assert linked["exists"] is True
    assert linked["path_kind"] == "symlink"
    assert linked["symlink_target"] == str(outside)
    assert linked["resolved_path"] == str(outside.resolve())
    assert linked["integrity_ok"] is False
    assert linked["sha256"] is None

    cache.unlink()
    os.mkfifo(cache)
    special = module._call_cache_state(tmp_path)
    assert special["exists"] is True
    assert special["path_kind"] == "special"
    assert special["integrity_ok"] is False
    assert special["sha256"] is None
    assert special["fingerprint"] != linked["fingerprint"]


def test_external_file_state_fingerprints_ignored_content_and_path_identity(tmp_path):
    """NEGATIVE CONTROL: ignored raw input mutation cannot escape Git snapshots."""

    module = _module()
    first_path = tmp_path / "raw-one.bin"
    second_path = tmp_path / "raw-two.bin"
    first_path.write_bytes(b"canonical-one")
    second_path.write_bytes(b"canonical-two")

    def specification(label: str, path: Path, expected: bytes) -> dict[str, object]:
        return {
            "label": label,
            "path": str(path),
            "expected_sha256": hashlib.sha256(expected).hexdigest(),
            "expected_size_bytes": len(expected),
        }

    specifications = [
        specification("one", first_path, b"canonical-one"),
        specification("two", second_path, b"canonical-two"),
    ]
    initial = module._external_file_set_state(specifications)
    assert initial["integrity_ok"] is True
    assert len(initial["inputs"]) == 2

    second_path.write_bytes(b"forged-value")
    mutated = module._external_file_set_state(specifications)
    assert mutated["integrity_ok"] is False
    assert mutated["fingerprint"] != initial["fingerprint"]

    second_path.write_bytes(b"canonical-two")
    target = tmp_path / "alternate.bin"
    target.write_bytes(b"canonical-one")
    first_path.unlink()
    first_path.symlink_to(target)
    retargeted = module._external_file_set_state(specifications)
    assert retargeted["integrity_ok"] is True
    assert retargeted["fingerprint"] != initial["fingerprint"]
    first_entry = next(row for row in retargeted["inputs"] if row["label"] == "one")
    assert first_entry["path_kind"] == "symlink"
    assert first_entry["resolved_path"] == str(target.resolve())


@pytest.mark.timeout(240)
def test_full_refuses_dirty_input_before_running_steps_and_records_the_environment(tmp_path):
    repository = tmp_path / "repository"
    commit = _init_git_repository(repository, with_release_script=True)
    template = tmp_path / "template"
    _init_template_repository(template, project_root=repository)
    (repository / "untracked.txt").write_text("dirty input\n", encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"
    environment = {
        key: value
        for key, value in _release_environment().items()
        if key not in {"SOURCE_DATE_EPOCH", "TEMPLATE_REPO_ROOT"}
    }

    result = subprocess.run(
        [
            _repository_python(repository),
            "-I",
            "-S",
            str(repository / "scripts" / "verify_release.py"),
            "full",
            "--receipt",
            str(receipt_path),
        ],
        cwd=repository,
        env=_release_environment(**(dict(environment) if isinstance(environment, dict) else {})),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["steps"] == [], "dirty-input refusal happened after release steps ran"
    assert receipt["refusal"]
    assert "clean tracked, staged, and untracked" in receipt["refusal"]
    assert receipt["git"]["commit"] == commit
    assert receipt["git"]["initial_dirty"] is True
    assert receipt["external_inputs"]["call_cache_unchanged"] is True
    assert receipt["environment"]["SOURCE_DATE_EPOCH"] == "1700000000"
    assert receipt["environment"]["source_date_epoch_origin"] == "audited-input-commit"
    assert receipt["publication_acceptance"] is False
    assert receipt["success"] is False

    development_receipt_path = tmp_path / "development-receipt.json"
    development = subprocess.run(
        [
            _repository_python(repository),
            "-I",
            "-S",
            str(repository / "scripts" / "verify_release.py"),
            "full",
            "--development-allow-dirty-full",
            "--receipt",
            str(development_receipt_path),
        ],
        cwd=repository,
        env=_release_environment(
            **(
                dict({**environment, "TEMPLATE_REPO_ROOT": str(template)})
                if isinstance({**environment, "TEMPLATE_REPO_ROOT": str(template)}, dict)
                else {}
            )
        ),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert development.returncode == 1, development.stdout + development.stderr
    development_receipt = json.loads(development_receipt_path.read_text(encoding="utf-8"))
    assert development_receipt["development_allow_dirty_full"] is True
    assert development_receipt["publication_acceptance"] is False
    assert "external toolchain prerequisite failed" in development_receipt["refusal"], (
        "the dirty override did not advance to the synthetic checkout's next real prerequisite"
    )
    assert development_receipt["steps"] == []
    assert "clean tracked, staged, and untracked" not in development_receipt["refusal"]
    assert development_receipt["external_inputs"]["template"]["root"] == str(template.resolve())


@pytest.mark.timeout(480)
def test_isolated_cli_trace_executes_transaction_receipt_and_exit_paths(tmp_path):
    """Semantic coverage gate for the load-bearing isolated subprocess path.

    Pytest-cov cannot attach to ``-I -S`` children without installing startup
    hooks that the release boundary intentionally forbids. A stdlib trace in
    that child instead proves the real transaction, final-state assessment,
    receipt publication, and CLI exit lines executed. This complements the
    aggregate branch percentage rather than pretending copied subprocess lines
    contributed to it.
    """
    repository = tmp_path / "repository"
    _init_git_repository(repository, with_release_script=True)
    (repository / "dirty.txt").write_text("force full-profile refusal\n", encoding="utf-8")
    receipt_path = tmp_path / "traced-receipt.json"
    counts_path = tmp_path / "trace-counts.json"
    bootstrap = tmp_path / "trace-release.py"
    release_script = repository / "scripts" / "verify_release.py"
    bootstrap.write_text(
        "import json, runpy, sys, trace\n"
        "from pathlib import Path\n"
        f"target = {str(release_script)!r}\n"
        f"counts_path = Path({str(counts_path)!r})\n"
        f"sys.argv = [target, 'full', '--receipt', {str(receipt_path)!r}]\n"
        "tracer = trace.Trace(count=True, trace=False)\n"
        "exit_code = 0\n"
        "try:\n"
        "    tracer.runctx(\"runpy.run_path(target, run_name='__main__')\", globals(), globals())\n"
        "except SystemExit as error:\n"
        "    exit_code = int(error.code or 0)\n"
        "finally:\n"
        "    counts = {f'{path}:{line}': count for (path, line), count in tracer.results().counts.items()}\n"
        "    counts_path.write_text(json.dumps(counts, sort_keys=True), encoding='utf-8')\n"
        "raise SystemExit(exit_code)\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [_repository_python(repository), "-I", "-S", str(bootstrap)],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["success"] is False and receipt["steps"] == []
    counts = json.loads(counts_path.read_text(encoding="utf-8"))
    source_lines = release_script.read_text(encoding="utf-8").splitlines()
    required_fragments = (
        "initial_state = _repository_state()",
        "results = _execute_steps(",
        "final_state = _repository_state()",
        "success = (",
        "_write_receipt(receipt_path, receipt)",
        'return 0 if receipt["success"] else 1',
    )
    missed = []
    for fragment in required_fragments:
        line_number = next(index for index, line in enumerate(source_lines, 1) if fragment in line)
        key = f"{release_script}:{line_number}"
        if int(counts.get(key, 0)) <= 0:
            missed.append((fragment, line_number))
    assert not missed, f"isolated release transaction paths were not executed: {missed}"


def test_dirty_refusal_never_executes_the_repository_raw_corpus_locator(tmp_path):
    repository = tmp_path / "repository"
    _init_git_repository(repository, with_release_script=True)
    marker = tmp_path / "raw-locator-executed"
    locator = repository / "scripts" / "list_raw_corpus_inputs.py"
    locator.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n"
        "print('{}')\n",
        encoding="utf-8",
    )
    _git(repository, "add", "scripts/list_raw_corpus_inputs.py")
    _git(repository, "commit", "-q", "-m", "raw locator control")
    (repository / "dirty-input.txt").write_text("refuse before helpers\n", encoding="utf-8")
    receipt_path = tmp_path / "dirty-helper-refusal.json"

    result = subprocess.run(
        [
            _repository_python(repository),
            "-I",
            "-S",
            str(repository / "scripts" / "verify_release.py"),
            "full",
            "--receipt",
            str(receipt_path),
        ],
        cwd=repository,
        env=_release_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["steps"] == []
    assert receipt["external_inputs"]["raw_corpora"]["initial_state"]["available"] is False
    assert (
        "prerequisite was refused"
        in receipt["external_inputs"]["raw_corpora"]["initial_state"]["locator_error"]
    )
    assert not marker.exists(), "dirty-source refusal occurred after executing repository code"


def test_template_root_file_is_honored_by_the_release_verifier_and_is_final(tmp_path, monkeypatch):
    """`.template-root` selects the provider when the environment is silent, and is final."""
    module = _module()
    monkeypatch.delenv("TEMPLATE_REPO_ROOT", raising=False)
    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    template = tmp_path / "template"
    _init_template_repository(template, project_root=sidecar)
    configured = sidecar / ".template-root"
    configured.write_text(f"# chosen checkout\n{template}\n", encoding="utf-8")
    assert module._resolve_template_root(project_root=sidecar) == template.resolve()

    configured.write_text(f"{tmp_path / 'absent-template'}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing required publication files"):
        module._resolve_template_root(project_root=sidecar)
    configured.write_text("one\ntwo\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exactly one template checkout path"):
        module._resolve_template_root(project_root=sidecar)
    # The environment value remains the first and final source.
    monkeypatch.setenv("TEMPLATE_REPO_ROOT", str(template))
    assert module._resolve_template_root(project_root=sidecar) == template.resolve()


def test_explicit_template_root_is_final_and_invalid_root_is_receipted(tmp_path, monkeypatch):
    module = _module()
    invalid = tmp_path / "not-a-template"
    invalid.mkdir()
    monkeypatch.setenv("TEMPLATE_REPO_ROOT", str(invalid))
    with pytest.raises(RuntimeError, match="missing required publication files"):
        module._resolve_template_root()

    repository = tmp_path / "repository"
    _init_git_repository(repository, with_release_script=True)
    receipt_path = tmp_path / "invalid-template-receipt.json"
    environment = {
        **_release_environment(),
        "TEMPLATE_REPO_ROOT": str(invalid),
    }
    environment.pop("SOURCE_DATE_EPOCH", None)
    result = subprocess.run(
        [
            _repository_python(repository),
            "-I",
            "-S",
            str(repository / "scripts" / "verify_release.py"),
            "full",
            "--receipt",
            str(receipt_path),
        ],
        cwd=repository,
        env=_release_environment(**(dict(environment) if isinstance(environment, dict) else {})),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["steps"] == []
    assert "template publication prerequisite failed" in receipt["refusal"]
    assert receipt["external_inputs"]["template"]["root"] is None
    assert (
        "missing required publication files"
        in receipt["external_inputs"]["template"]["resolution_error"]
    )


def test_template_root_must_route_the_requested_project_to_this_sidecar(tmp_path):
    module = _module()
    other_project = tmp_path / "different-sidecar"
    other_project.mkdir()
    template = tmp_path / "template"
    _init_template_repository(template, project_root=other_project)

    with pytest.raises(RuntimeError, match="not the audited sidecar"):
        module._validate_template_root(template)


def test_final_template_audit_rechecks_the_ignored_project_route(tmp_path):
    """NEGATIVE CONTROL: a route mutation is invisible to the Git fingerprint."""

    module = _module()
    template = tmp_path / "template"
    _init_template_repository(template)
    initial = module._repository_state(template)

    final, unchanged, route_valid, error = module._audit_final_template_input(template, initial)
    assert final is not None
    assert unchanged is True
    assert route_valid is True
    assert error is None

    other_project = tmp_path / "different-sidecar"
    other_project.mkdir()
    project_entry = template / "projects/working/active_inference_language_model"
    project_entry.unlink()
    project_entry.symlink_to(other_project.resolve(), target_is_directory=True)

    final, unchanged, route_valid, error = module._audit_final_template_input(template, initial)
    assert final is not None
    assert unchanged is True, "the planted route is ignored, so Git alone cannot detect it"
    assert route_valid is False
    assert error is not None and "not the audited sidecar" in error


def test_full_refuses_dirty_template_and_records_both_git_inputs(tmp_path):
    repository = tmp_path / "repository"
    _init_git_repository(repository, with_release_script=True)
    template = tmp_path / "template"
    template_commit = _init_template_repository(template, project_root=repository)
    (template / "untracked-renderer-input.txt").write_text("dirty\n", encoding="utf-8")
    receipt_path = tmp_path / "dirty-template-receipt.json"
    environment = {**_release_environment(), "TEMPLATE_REPO_ROOT": str(template)}
    environment.pop("SOURCE_DATE_EPOCH", None)

    result = subprocess.run(
        [
            _repository_python(repository),
            "-I",
            "-S",
            str(repository / "scripts" / "verify_release.py"),
            "full",
            "--receipt",
            str(receipt_path),
        ],
        cwd=repository,
        env=_release_environment(**(dict(environment) if isinstance(environment, dict) else {})),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["steps"] == []
    assert "clean template checkout" in receipt["refusal"]
    template_input = receipt["external_inputs"]["template"]
    assert template_input["root"] == str(template.resolve())
    assert template_input["initial_state"]["head"] == template_commit
    assert template_input["initial_state"]["status"]
    assert template_input["repository_state_unchanged"] is True


def test_receipt_destination_must_be_outside_or_gitignored(tmp_path):
    module = _module()
    repository = tmp_path / "repository"
    _init_git_repository(repository)
    (repository / ".gitignore").write_text("output/reports/\n", encoding="utf-8")

    assert module._receipt_destination_is_safe(tmp_path / "external.json", repository)
    assert module._receipt_destination_is_safe(
        repository / "output/reports/release-receipt.json",
        repository,
        step_names=("lock",),
    )
    assert not module._receipt_destination_is_safe(repository / "tracked.txt", repository)
    assert not module._receipt_destination_is_safe(repository / "visible-receipt.json", repository)

    (repository / ".gitignore").write_text(
        "output/reports/release-receipt.json\noutput/reports/logs/00000000000000000000000000000000/release-step.log\n",
        encoding="utf-8",
    )
    assert not module._receipt_destination_is_safe(
        repository / "output/reports/release-receipt.json", repository
    ), "an exact receipt ignore hid the atomic temporary receipt that could survive a crash"

    (repository / ".gitignore").write_text(
        "output/reports/release-receipt.json\n"
        "output/reports/.release-receipt.json.*.tmp\n"
        "output/reports/logs/00000000000000000000000000000000/lock.log\n",
        encoding="utf-8",
    )
    assert not module._receipt_destination_is_safe(
        repository / "output/reports/release-receipt.json",
        repository,
        step_names=("lock",),
    ), "the final log was ignored but the log writer's matching scratch path was visible"
    with_log_scratch = (repository / ".gitignore").read_text(
        encoding="utf-8"
    ) + "output/reports/logs/00000000000000000000000000000000/.lock.log.*.tmp\n"
    (repository / ".gitignore").write_text(with_log_scratch, encoding="utf-8")
    assert module._receipt_destination_is_safe(
        repository / "output/reports/release-receipt.json",
        repository,
        step_names=("lock",),
    )

    for token in ("0" * 32, "f" * 32):
        temporary = module._atomic_temporary_name("release-receipt.json", token)
        assert re.fullmatch(r"\.release-receipt\.json\.[0-9a-f]{32}\.tmp", temporary)
    with pytest.raises(ValueError, match="exactly 32"):
        module._atomic_temporary_name("release-receipt.json", "release-probe-a")


def test_receipt_safety_rejects_lexical_tracked_and_symlink_components(tmp_path):
    module = _module()
    repository = tmp_path / "repository"
    _init_git_repository(repository)
    external = tmp_path / "external"
    external.mkdir()
    tracked_route = repository / "tracked-route"
    tracked_route.symlink_to(external, target_is_directory=True)
    _git(repository, "add", "tracked-route")
    _git(repository, "commit", "-q", "-m", "tracked route")

    assert not module._receipt_destination_is_safe(
        tracked_route / "receipt.json",
        repository,
    ), "resolving first incorrectly makes the tracked lexical route look external"

    untracked_route = repository / "untracked-route"
    untracked_route.symlink_to(external, target_is_directory=True)
    assert not module._receipt_destination_is_safe(
        untracked_route / "receipt.json",
        repository,
    )
    assert not module._receipt_destination_is_safe(
        repository / "tracked.txt" / "receipt.json",
        repository,
    )

    reports = repository / "output" / "reports"
    reports.mkdir(parents=True)
    (repository / ".gitignore").write_text("output/reports/\n", encoding="utf-8")
    (reports / "logs").symlink_to(external, target_is_directory=True)
    assert not module._receipt_destination_is_safe(
        reports / "receipt.json",
        repository,
        step_names=("lock",),
    )


def test_receipt_writer_atomically_publishes_complete_json(tmp_path):
    module = _module()
    receipt_path = tmp_path / "receipt.json"
    module._write_receipt(receipt_path, {"schema": "test", "success": True})
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == {
        "schema": "test",
        "success": True,
    }
    assert not list(tmp_path.glob(".receipt.json.*.tmp"))


def test_receipt_writer_refuses_a_symlink_destination_directory(tmp_path):
    module = _module()
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(OSError):
        module._write_receipt(linked / "receipt.json", {"schema": "test"})

    assert not (real / "receipt.json").exists()


def test_receipt_writer_anchors_creation_of_an_absent_nested_parent(tmp_path):
    """The isolated receipt writer must close the same mkdir/open swap race."""
    module = _module()
    publication = tmp_path / "publication"
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    moved_publication = tmp_path / "anchored-publication"
    components = tuple(f"d{index:03d}" for index in range(120))
    relative_parent = Path(*components)
    receipt_path = publication / relative_parent / "receipt.json"
    payload = {"schema": "test", "evidence": "anchored"}
    outcome: list[BaseException | None] = []

    def write() -> None:
        try:
            module._write_receipt(receipt_path, payload)
            outcome.append(None)
        except BaseException as error:  # surfaced below with its original type
            outcome.append(error)

    thread = threading.Thread(target=write)
    thread.start()
    deadline = time.monotonic() + 10
    first_descendant = publication / components[0]
    while time.monotonic() < deadline and not first_descendant.is_dir():
        time.sleep(0.0001)
    assert first_descendant.is_dir(), (
        "receipt writer did not expose the new ancestor; the race was not exercised"
    )

    publication.rename(moved_publication)
    publication.symlink_to(attacker, target_is_directory=True)
    thread.join(timeout=20)

    assert not thread.is_alive()
    assert outcome == [None]
    published = moved_publication / relative_parent / "receipt.json"
    assert json.loads(published.read_text(encoding="utf-8")) == payload
    assert not (attacker / relative_parent / "receipt.json").exists()


def test_dirty_input_policy_keeps_quick_usable_and_requires_explicit_full_override(tmp_path):
    module = _module()
    repository = tmp_path / "repository"
    _init_git_repository(repository)
    (repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    dirty = module._repository_state(repository)
    assert dirty.dirty

    assert module._dirty_input_refusal("quick", dirty, development_allow_dirty_full=False) is None
    refusal = module._dirty_input_refusal("full", dirty, development_allow_dirty_full=False)
    assert refusal is not None
    assert "--development-allow-dirty-full" in refusal
    assert module._dirty_input_refusal("full", dirty, development_allow_dirty_full=True) is None


def test_the_call_cache_lives_outside_the_regenerable_tree():
    """An artifact that cannot be rebuilt is an input, whatever directory it sits in.

    The memoized model calls lived under `output/`, which promises to be
    entirely regenerable and to hold no source of truth. Clearing that tree to
    prove the project rebuilds destroyed roughly 190 MB that no script can
    recreate without a live model server, and the release profile's coverage
    floor silently depended on it: about 166 statements across five benchmark
    modules stop executing on a cold store, dropping the total from ~93% to
    91.87% with no failing test naming the cause.

    NEGATIVE CONTROL: the path must still be *somewhere*, and it must still be
    ignored. A check that only forbids `output/` would pass on a cache moved
    into the tracked tree, which trades one problem for a worse one.
    """
    from pathlib import Path

    from ailm.call_cache import default_cache_path

    root = Path(__file__).resolve().parents[1]
    path = default_cache_path(root)

    assert "output" not in path.relative_to(root).parts, (
        f"the call cache is back under output/: {path.relative_to(root)}. That "
        f"directory means regenerable, and this store is not."
    )
    assert path.name.endswith(".jsonl") and path.parent != root

    ignored = subprocess.run(
        ["git", "check-ignore", str(path.relative_to(root))],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert ignored.returncode == 0, (
        f"{path.relative_to(root)} is not gitignored; the store is large and "
        f"machine-specific, and the claims it supports are in the ledger"
    )


def test_the_cold_cache_prerequisite_is_written_down_where_it_bites():
    """A prerequisite nobody states is a defect report waiting to be filed.

    The first clean rebuild spent real time diagnosing a coverage drop with
    exactly one cause and no symptom pointing at it.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    verifier = (root / "scripts" / "verify_release.py").read_text(encoding="utf-8")
    assert "A cold, partial, unreachable, or changed store is a hard failure" in verifier
    assert "_call_cache_state" in verifier, (
        "the release receipt no longer fingerprints the ignored cache it consumes"
    )

    quickstart = (root / "docs" / "guides" / "quickstart.md").read_text(encoding="utf-8")
    assert "warm_call_cache.py" in quickstart, (
        "quickstart no longer tells a new checkout how to warm the cache. It names "
        "the runbook rather than the recorder now: the raw recipe was four ways to "
        "get it subtly wrong, and the script checks all four before spending an hour."
    )
    assert "ollama serve" in quickstart
    assert "resumes" in quickstart.lower(), (
        "quickstart no longer says the warming run resumes, which is the fact that "
        "makes an interruption survivable"
    )


def _warming_ledger_payload(values: dict[str, object]) -> dict[str, object]:
    """One strict v2 ledger fixture for real warmer subprocess tests."""

    return {
        "_schema": "ailm-claims-v2",
        "_description": "warming test fixture",
        "claims": {
            label: {
                "value": value,
                "run_id": "warming-test",
                "source": "tests/test_release_orchestration.py",
            }
            for label, value in values.items()
        },
    }


@pytest.mark.timeout(960)
def test_the_warming_runbook_preflights_before_it_spends_hours():
    """A run that fails on the third model after two hours should have failed at once.

    `scripts/warm_call_cache.py` is the documented way to satisfy the `full`
    profile's prerequisite. Its value is entirely in the checks it runs first:
    a live server, every declared probe present, an isolated ledger as the write
    target, and a resume that reports what it already holds.

    NEGATIVE CONTROL: it must NOT write the real ledger. A warming run that
    re-measures into `data/claims.json` overwrites recorded claims with fresh
    numbers, and whether a value moved is the question worth asking rather than
    one to answer silently in place.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "warm_call_cache.py").read_text(encoding="utf-8")

    for required in (
        "api/tags",
        "ollama serve",
        "ollama pull",
        "AILM_CLAIMS_PATH",
        "--check",
        "--audit-strata-only",
    ):
        assert required in source, f"the runbook no longer handles {required!r}"

    assert "ISOLATED_LEDGER" in source

    spec = importlib.util.spec_from_file_location(
        "ailm_warm_call_cache", root / "scripts" / "warm_call_cache.py"
    )
    assert spec is not None and spec.loader is not None
    warmer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(warmer)
    audit_command = warmer._recorder_command(audit_strata_only=True)
    assert "--ignore-sidecars" in audit_command, (
        "audit repair could reuse a reduced sidecar and make no missing cache call"
    )
    assert sorted(warmer._declared_probes()) == sorted(
        part for part in audit_command if part in warmer._declared_probes()
    )
    run_source = inspect.getsource(warmer._run_recorder_with_isolated_ledger)
    assert "atomic_write_bytes(isolated_target, recorded)" in run_source, (
        "the warming run no longer initializes a complete isolated ledger"
    )
    assert run_source.count("single_writer_lock(") == 3, (
        "the initializer needs the warming-session lock and the recorder's lock both "
        "before and after child execution"
    )
    assert "subprocess.run" in run_source
    assert "_run_recorder_with_isolated_ledger(command)" in inspect.getsource(warmer.main), (
        "main bypasses the initializer-plus-recorder critical section"
    )
    assert "APPROXIMATE_TOTAL_CALLS" not in source, (
        "warm progress regressed to a hand-copied workload denominator"
    )
    assert "shutil.copy2" not in source, (
        "copy2 follows an existing target symlink and is not an atomic ledger publication"
    )
    assert "report_ledger_delta" in source, (
        "the run no longer reports whether a re-measured value moved, which is its "
        "entire scientific interest"
    )
    assert "rm -rf" not in source, (
        "the resumable runbook must not instruct an agent to delete the output tree; "
        "the artifact writers already replace their owned files transactionally"
    )
    next_steps = inspect.getsource(warmer.next_steps)
    assert "orchestrate_claim_recorders.py network" in next_steps
    assert ".venv/bin/python scripts/record_external_benchmark_claims.py" not in next_steps, (
        "the runbook duplicated a recorder inventory instead of delegating to "
        "dependency-ordered dynamic discovery"
    )

    # Reach the real CLI through a deterministic local-state refusal. Checking
    # a live server also plans the full publication workload and hashes the
    # user's cache; those operations have separate fixture-based tests. An
    # invalid store selection must fail before any model-server request.
    result = subprocess.run(
        [
            str(root / ".venv" / "bin" / "python"),
            str(root / "scripts" / "warm_call_cache.py"),
            "--check",
        ],
        cwd=root,
        env=_release_environment(AILM_CALL_CACHE_PATH="relative-invalid-store.jsonl"),
        capture_output=True,
        text=True,
        timeout=480,
    )
    assert result.returncode == 1, f"preflight did not refuse invalid state:\n{result.stderr}"
    assert "AILM_CALL_CACHE_PATH selects a different store" in result.stderr
    assert "Warming. This resumes" not in result.stdout
    assert "Traceback" not in result.stderr, result.stderr


def test_warming_inventory_rejects_unsupported_v1_before_reporting_progress(tmp_path):
    """NEGATIVE CONTROL: progress cannot hide an unsupported row behind an index."""
    from ailm.call_cache import CacheCorruptionError, CacheRequest, CallCache

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "ailm_warm_inventory", root / "scripts" / "warm_call_cache.py"
    )
    assert spec is not None and spec.loader is not None
    warmer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(warmer)

    unsupported_path = tmp_path / "unsupported-v1.jsonl"
    unsupported_path.write_text(
        json.dumps(
            {
                "schema": "ailm-call-cache-v1",
                "key": "0" * 64,
                "payload": {"model": "unsupported"},
                "value": {"response": "untrusted"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CacheCorruptionError, match="unsupported schema 'ailm-call-cache-v1'"):
        warmer._warm_inventory(unsupported_path)

    path = tmp_path / "current.jsonl"
    current_request = CacheRequest(
        payload={"model": "current"},
        endpoint="http://127.0.0.1:11434",
        protocol="generate-v2",
        identity="sha256:" + "b" * 64,
    )
    cache = CallCache(path)
    cache.put(current_request, {"response": "bound"})
    assert warmer._warm_inventory(path) == 1


def test_warming_progress_is_the_exact_protocol_key_union(tmp_path) -> None:
    """Progress counts matching keys, not a stale scalar or unrelated records.

    NEGATIVE CONTROL: the raw number of calls is not the number of cache keys.
    The long-context arm overlaps the short arm at early positions and repeated
    phoneme prompts share a key, so summing declared sample sizes overcounts the
    resumable workload even when every individual sample size is current.
    """
    from ailm.call_cache import CacheRequest, CallCache

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "ailm_warm_exact_progress", root / "scripts" / "warm_call_cache.py"
    )
    assert spec is not None and spec.loader is not None
    warmer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(warmer)

    configuration = warmer._publication_configuration()
    model_tags = configuration["model_tags"]
    assert isinstance(model_tags, list) and model_tags
    identities = {tag: f"sha256:{index:064x}" for index, tag in enumerate(model_tags, start=1)}
    expected = warmer._publication_call_keys(identities=identities)
    raw_calls = sum(len(warmer._publication_payloads(tag, configuration)) for tag in model_tags)
    assert len(expected) < raw_calls, (
        "planted duplicate-key control disappeared; a raw workload sum could now "
        "masquerade as an exact cache denominator"
    )

    endpoint = configuration["endpoint"]
    protocol = configuration["call_protocol"]
    assert isinstance(endpoint, str) and isinstance(protocol, str)
    first_tag = model_tags[0]
    first_payload = warmer._publication_payloads(first_tag, configuration)[0]
    matching = CacheRequest(
        payload=first_payload,
        endpoint=endpoint,
        protocol=protocol,
        identity=identities[first_tag],
    )
    assert matching.key in expected
    unrelated = CacheRequest(
        payload={"model": first_tag, "request": "outside-publication-protocol"},
        endpoint=endpoint,
        protocol=protocol,
        identity=identities[first_tag],
    )
    assert unrelated.key not in expected

    path = tmp_path / "calls.jsonl"
    cache = CallCache(path)
    cache.put(matching, {"response": "publication"})
    cache.put(unrelated, {"response": "unrelated"})
    covered, required, trusted = warmer._publication_inventory(expected, path)
    assert (covered, required, trusted) == (1, len(expected), 2)

    with pytest.raises(ValueError, match="cover the publication registry exactly"):
        warmer._publication_call_keys(identities={first_tag: identities[first_tag]})


@pytest.mark.timeout(240)
def test_warming_initialization_and_recorder_are_one_real_locked_session(tmp_path) -> None:
    """A second process must refuse before it replaces the active ledger.

    This uses OS-visible PID locks and real subprocesses.  The first invocation
    is the planted positive control: it copies the source ledger, the fake child
    acquires the recorder's own lock, and it observes the outer session lock.
    The two blocked invocations then hold each lock independently and prove that
    neither an active recorder nor another warmer can reach initialization.
    """
    from ailm.ledger_lock import LOCK_FILENAME, single_writer_lock

    root = Path(__file__).resolve().parents[1]
    warmer_path = root / "scripts" / "warm_call_cache.py"
    spec = importlib.util.spec_from_file_location("ailm_warm_lock_test", warmer_path)
    assert spec is not None and spec.loader is not None
    warmer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(warmer)

    source = tmp_path / "claims.json"
    isolated = tmp_path / "warming_claims.json"
    source_payload = _warming_ledger_payload({"baseline": 1})
    sentinel_payload = {"sentinel": "active-recorder-result"}
    source.write_text(json.dumps(source_payload), encoding="utf-8")
    isolated.write_text(json.dumps(sentinel_payload), encoding="utf-8")

    fake_recorder = tmp_path / "fake_recorder.py"
    fake_recorder.write_text(
        """from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from ailm.ledger_lock import ledger_lock_path, single_writer_lock

isolated = Path(os.environ["AILM_CLAIMS_PATH"])
session_lock = Path(sys.argv[1])
if not session_lock.is_file():
    raise SystemExit("outer warming-session lock was not held during the child")
with single_writer_lock(ledger_lock_path(Path.cwd())):
    payload = json.loads(isolated.read_text(encoding="utf-8"))
    payload["claims"]["fake_recorder"] = {
        "value": "completed",
        "run_id": "warming-test",
        "source": "tests/test_release_orchestration.py",
    }
    isolated.write_text(json.dumps(payload), encoding="utf-8")
""",
        encoding="utf-8",
    )
    runner = (
        "import importlib.util, sys\n"
        "from pathlib import Path\n"
        "warmer_path, root, source, isolated, child = map(Path, sys.argv[1:])\n"
        "spec = importlib.util.spec_from_file_location('warm_runner', warmer_path)\n"
        "assert spec is not None and spec.loader is not None\n"
        "warmer = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(warmer)\n"
        "command = [sys.executable, str(child), "
        "str(warmer._warming_session_lock_path(isolated))]\n"
        "result, _recorded, _snapshot = warmer._run_recorder_with_isolated_ledger(\n"
        "    command, ledger=source, isolated_ledger=isolated, project_root=root\n"
        ")\n"
        "raise SystemExit(result.returncode)\n"
    )

    def invoke() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(root / ".venv" / "bin" / "python"),
                "-c",
                runner,
                str(warmer_path),
                str(root),
                str(source),
                str(isolated),
                str(fake_recorder),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )

    completed = invoke()
    assert completed.returncode == 0, completed.stderr
    positive = json.loads(isolated.read_text(encoding="utf-8"))
    expected_positive = json.loads(json.dumps(source_payload))
    expected_positive["claims"]["fake_recorder"] = {
        "value": "completed",
        "run_id": "warming-test",
        "source": "tests/test_release_orchestration.py",
    }
    assert positive == expected_positive, (
        "the planted normal warm did not initialize and execute its real child"
    )

    isolated.write_text(json.dumps(sentinel_payload), encoding="utf-8")
    with single_writer_lock(isolated.parent / LOCK_FILENAME):
        blocked_by_recorder = invoke()
    assert blocked_by_recorder.returncode != 0
    assert "warming session refused" in blocked_by_recorder.stderr
    assert json.loads(isolated.read_text(encoding="utf-8")) == sentinel_payload

    with single_writer_lock(warmer._warming_session_lock_path(isolated)):
        blocked_by_warmer = invoke()
    assert blocked_by_warmer.returncode != 0
    assert "warming session refused" in blocked_by_warmer.stderr
    assert json.loads(isolated.read_text(encoding="utf-8")) == sentinel_payload


def test_warming_child_cannot_follow_a_replaced_leaf_symlink(tmp_path) -> None:
    """The child must write the lexical isolated path, never a former link target."""
    root = Path(__file__).resolve().parents[1]
    warmer_path = root / "scripts" / "warm_call_cache.py"
    spec = importlib.util.spec_from_file_location("ailm_warm_symlink_test", warmer_path)
    assert spec is not None and spec.loader is not None
    warmer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(warmer)

    source = tmp_path / "claims.json"
    source.write_text(
        json.dumps(_warming_ledger_payload({"source": 1})),
        encoding="utf-8",
    )
    outside = tmp_path / "outside.json"
    outside.write_text("outside-sentinel", encoding="utf-8")
    isolated = tmp_path / "warming_claims.json"
    isolated.symlink_to(outside)
    child = (
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "target = Path(os.environ['AILM_CLAIMS_PATH'])\n"
        "payload = json.loads(target.read_text(encoding='utf-8'))\n"
        "payload['claims']['child'] = {\n"
        "    'value': 'wrote-lexical-target',\n"
        "    'run_id': 'warming-test',\n"
        "    'source': 'tests/test_release_orchestration.py',\n"
        "}\n"
        "target.write_text(json.dumps(payload), encoding='utf-8')\n"
    )

    result, recorded, snapshot = warmer._run_recorder_with_isolated_ledger(
        [str(root / ".venv" / "bin" / "python"), "-c", child],
        ledger=source,
        isolated_ledger=isolated,
        project_root=root,
    )

    assert result.returncode == 0
    assert outside.read_text(encoding="utf-8") == "outside-sentinel", (
        "the child followed the former isolated-ledger symlink outside its write target"
    )
    assert not isolated.is_symlink()
    isolated_payload = json.loads(isolated.read_text(encoding="utf-8"))
    assert isolated_payload["claims"]["child"]["value"] == "wrote-lexical-target"
    assert recorded == source.read_bytes()
    assert snapshot is not None
    assert json.loads(snapshot)["claims"]["child"]["value"] == "wrote-lexical-target"


@pytest.mark.parametrize(
    "content",
    (
        b"not-json",
        b"[]",
        b'{"_schema":"wrong","_description":"fixture","claims":{}}',
        b'{"_schema":"ailm-claims-v2","_description":"fixture","claims":[]}',
        b'{"_schema":"ailm-claims-v2","_description":"fixture","claims":{"x":{"value":1}}}',
    ),
)
def test_warming_refuses_malformed_source_ledgers_before_launch(
    tmp_path: Path, content: bytes
) -> None:
    """NEGATIVE CONTROL — invalid source evidence cannot reach the child process."""

    root = Path(__file__).resolve().parents[1]
    warmer_path = root / "scripts" / "warm_call_cache.py"
    spec = importlib.util.spec_from_file_location("ailm_warm_invalid_ledger", warmer_path)
    assert spec is not None and spec.loader is not None
    warmer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(warmer)
    source = tmp_path / "claims.json"
    source.write_bytes(content)
    marker = tmp_path / "child-launched"
    isolated = tmp_path / "warming_claims.json"

    with pytest.raises(SystemExit, match="warming session refused"):
        warmer._run_recorder_with_isolated_ledger(
            [
                str(root / ".venv" / "bin" / "python"),
                "-c",
                "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ran')",
                str(marker),
            ],
            ledger=source,
            isolated_ledger=isolated,
            project_root=root,
        )
    assert not marker.exists()


def test_warming_child_is_forced_to_the_preflighted_cache_path(tmp_path: Path) -> None:
    """NEGATIVE CONTROL — an inherited override cannot redirect the real child."""

    root = Path(__file__).resolve().parents[1]
    warmer_path = root / "scripts" / "warm_call_cache.py"
    spec = importlib.util.spec_from_file_location("ailm_warm_cache_env", warmer_path)
    assert spec is not None and spec.loader is not None
    warmer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(warmer)
    source = tmp_path / "claims.json"
    source.write_text(
        json.dumps(_warming_ledger_payload({"source": 1})),
        encoding="utf-8",
    )
    isolated = tmp_path / "warming_claims.json"
    canonical = tmp_path / "canonical-calls.jsonl"
    wrong = tmp_path / "wrong-calls.jsonl"
    child = (
        "import json, os\n"
        "from pathlib import Path\n"
        "target = Path(os.environ['AILM_CLAIMS_PATH'])\n"
        "payload = json.loads(target.read_text(encoding='utf-8'))\n"
        "payload['claims']['cache_path_seen'] = {\n"
        "    'value': os.environ['AILM_CALL_CACHE_PATH'],\n"
        "    'run_id': 'warming-test',\n"
        "    'source': 'tests/test_release_orchestration.py',\n"
        "}\n"
        "target.write_text(json.dumps(payload), encoding='utf-8')\n"
    )
    environment = dict(os.environ)
    environment["AILM_CALL_CACHE_PATH"] = str(wrong)

    result, _recorded, snapshot = warmer._run_recorder_with_isolated_ledger(
        [str(root / ".venv" / "bin" / "python"), "-c", child],
        ledger=source,
        isolated_ledger=isolated,
        project_root=root,
        environment=environment,
        cache_path=canonical,
    )

    assert result.returncode == 0
    assert snapshot is not None
    measured = json.loads(snapshot)
    assert measured["claims"]["cache_path_seen"]["value"] == str(canonical)
    assert measured["claims"]["cache_path_seen"]["value"] != str(wrong)


def test_warming_rejects_non_digest_and_aliased_model_identities() -> None:
    """Progress cannot promise replay for identities publication binding rejects."""
    root = Path(__file__).resolve().parents[1]
    warmer_path = root / "scripts" / "warm_call_cache.py"
    spec = importlib.util.spec_from_file_location("ailm_warm_identity_test", warmer_path)
    assert spec is not None and spec.loader is not None
    warmer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(warmer)
    tags = list(warmer._publication_configuration()["model_tags"])
    assert len(tags) >= 2

    non_digest = dict.fromkeys(tags, "not-a-content-digest")
    with pytest.raises(ValueError, match="full sha256 digest"):
        warmer._publication_call_keys(identities=non_digest)

    shared = "sha256:" + "a" * 64
    aliased = dict.fromkeys(tags, shared)
    with pytest.raises(ValueError, match="duplicate immutable identities"):
        warmer._publication_call_keys(identities=aliased)


def test_warming_delta_reports_only_the_external_recorder_namespace() -> None:
    """An overlapping core replay must not look like network measurement drift."""

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "ailm_warm_delta", root / "scripts" / "warm_call_cache.py"
    )
    assert spec is not None and spec.loader is not None
    warmer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(warmer)

    external = "scripts/record_external_benchmark_claims.py"
    core = "scripts/record_parameter_claims.py"
    recorded = {
        "bench_value": {"value": 1, "source": external},
        "bench_retired": {"value": 3, "source": external},
        "params_total": {"value": 10, "source": core},
    }
    measured = {
        "bench_value": {"value": 2, "source": external},
        "params_total": {"value": 11, "source": core},
    }

    assert warmer.external_claim_delta(recorded, measured) == {
        "bench_retired": (3, None),
        "bench_value": (1, 2),
    }


# Syntax and cross-surface consistency do not establish public DOI availability.
_ZENODO_DOI_SYNTAX = re.compile(r"10\.5281/zenodo\.[0-9]+")


def _doi_metadata_agrees(citation: dict, publication: dict, urls: dict) -> bool:
    """Optional DOI fields must all be absent or identify the same valid DOI."""
    presence = ("doi" in citation, "doi" in publication, "DOI" in urls)
    if not any(presence):
        return True
    if not all(presence):
        return False
    value = citation["doi"]
    return (
        isinstance(value, str)
        and _ZENODO_DOI_SYNTAX.fullmatch(value) is not None
        and publication["doi"] == value
        and urls["DOI"] == f"https://doi.org/{value}"
    )


@pytest.mark.parametrize(
    "citation,publication,urls,expected",
    [
        ({}, {}, {}, True),
        (
            {"doi": "10.5281/zenodo.123"},
            {"doi": "10.5281/zenodo.123"},
            {"DOI": "https://doi.org/10.5281/zenodo.123"},
            True,
        ),
        ({"doi": "10.5281/zenodo.123"}, {}, {}, False),
        ({}, {"doi": "10.5281/zenodo.123"}, {}, False),
        ({}, {}, {"DOI": "https://doi.org/10.5281/zenodo.123"}, False),
        ({"doi": None}, {"doi": None}, {"DOI": None}, False),
        (
            {"doi": "10.5281/zenodo.123"},
            {"doi": "10.5281/zenodo.456"},
            {"DOI": "https://doi.org/10.5281/zenodo.123"},
            False,
        ),
        (
            {"doi": "10.5281/zenodo.123"},
            {"doi": "10.5281/zenodo.123"},
            {"DOI": "https://doi.org/10.5281/zenodo.456"},
            False,
        ),
        (
            {"doi": "10.5281/zenodo.TBD"},
            {"doi": "10.5281/zenodo.TBD"},
            {"DOI": "https://doi.org/10.5281/zenodo.TBD"},
            False,
        ),
        ({"doi": ""}, {"doi": ""}, {"DOI": "https://doi.org/"}, False),
    ],
)
def test_optional_doi_metadata_controls(citation, publication, urls, expected):
    assert _doi_metadata_agrees(citation, publication, urls) is expected


def test_the_citation_file_stays_in_step_with_the_paper():
    """Titles, versions, source identities and optional DOI fields stay coherent."""
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    citation = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    config = yaml.safe_load(
        (root / "docs" / "manuscript" / "config.yaml").read_text(encoding="utf-8")
    )
    package = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    init_source = (root / "src" / "ailm" / "__init__.py").read_text(encoding="utf-8")
    version_match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_source, re.MULTILINE)
    assert version_match is not None, "src/ailm/__init__.py has no literal __version__"

    paper = config["paper"] if "paper" in config else config
    assert citation["title"] == paper["title"], (
        "CITATION.cff and docs/manuscript/config.yaml disagree about the title"
    )
    versions = {
        "CITATION.cff": str(citation["version"]),
        "docs/manuscript/config.yaml": str(paper["version"]),
        "pyproject.toml": str(package["project"]["version"]),
        "src/ailm/__init__.py": version_match.group(1),
    }
    assert len(set(versions.values())) == 1, f"Release version drift: {versions}"
    version = next(iter(versions.values()))
    assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version), (
        f"Release version is not a stable three-part version: {version!r}"
    )

    # NEGATIVE CONTROL: the comparison rejects a stale package declaration.
    planted = {**versions, "pyproject.toml": "0.1.0"}
    assert len(set(planted.values())) > 1

    author = citation["authors"][0]
    declared = config["authors"][0]
    assert declared["name"].endswith(author["family-names"])
    assert author["orcid"].endswith(declared["orcid"])

    assert _doi_metadata_agrees(
        citation, config.get("publication") or {}, package["project"]["urls"]
    ), "DOI presence or identity differs across citation, title page and package URLs"

    assert citation.get("repository-code") == (
        "https://github.com/ActiveInferenceInstitute/Active_Inference_Language_Model"
    ), "CITATION.cff must name the declared Institute release destination"
    for absent in ("commit", "date-released"):
        value = citation.get(absent)
        assert value is None, (
            f"CITATION.cff declares {absent}={value!r}. These are pushed or tagged "
            f"by the principal, and a citation field that resolves to nothing is "
            f"worse than an absent one."
        )


def test_the_license_states_the_constraint_that_forces_it():
    """The distribution must disclose every narrower vendored-data term.

    The repository is a mixed-license collection, not one work that can be
    relabeled MIT or even uniformly CC BY-SA. Brown is non-commercial in NLTK's
    own package metadata and Czech-PDTC is CC BY-NC-SA; omitting either would
    advertise a commercial-use permission the distributed data do not grant.

    The split itself: original source code (``src/``, ``scripts/``, ``tests/``,
    ``formal/``, ``gnn/``) is MIT, while the manuscript, documentation, figures,
    and AILM-authored data stay CC BY-SA 4.0. Neither license touches the
    vendored third-party terms below, and the code being MIT does not relax the
    commercial-use boundary the vendored data impose on the packaged
    distribution as a whole.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    license_text = (root / "LICENSE").read_text(encoding="utf-8")
    notices = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    provenance = (root / "docs" / "provenance" / "sources.md").read_text(encoding="utf-8")
    package = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    def license_problems(text: str) -> list[str]:
        """Every way a LICENSE text fails to state the split and the vendored terms."""
        problems = [
            f"omits {required}"
            for required in (
                "MIT",
                "CC BY-SA 4.0",
                "CC BY-NC-SA 4.0",
                "BSD-2-Clause",
                "non-commercial",
            )
            if required.casefold() not in text.casefold()
        ]
        # The split is stated for both kinds, not just asserted by keyword presence.
        if re.search(r"Original AILM source code[^\n]*?—\s*MIT License", text) is None:
            problems.append("does not state that the code itself is MIT")
        if (
            re.search(
                r"manuscript, documentation, figures, and authored data[^\n]*?CC BY-SA 4\.0",
                text,
                re.IGNORECASE,
            )
            is None
        ):
            problems.append("does not state that manuscript/docs/figures/data are CC BY-SA 4.0")
        return problems

    assert license_problems(license_text) == []
    assert "Copyright (C) 1993-2015 Carnegie Mellon University" in notices
    assert "May be used for non-commercial purposes" in notices
    assert "CC BY-NC-SA 4.0" in provenance

    brown_public_domain = re.compile(r"Brown[^\n]{0,120}public domain", re.IGNORECASE)
    assert brown_public_domain.search(license_text) is None
    assert brown_public_domain.search(provenance) is None

    # NEGATIVE CONTROL: a plain relabel of the whole distribution as MIT, with no
    # split and no vendored-data disclosure, is refused by the same check.
    planted_plain_mit = (
        "MIT License\n\nCopyright (c) 2026 Daniel Ari Friedman\n\n"
        "Permission is hereby granted, free of charge, to any person obtaining a "
        "copy of this software..."
    )
    refused = license_problems(planted_plain_mit)
    assert "omits CC BY-NC-SA 4.0" in refused
    assert "does not state that the code itself is MIT" in refused

    project = package["project"]
    assert project["license"] == "LicenseRef-AILM-Mixed"
    assert set(project["license-files"]) == {"LICENSE", "THIRD_PARTY_NOTICES.md"}

    # POSITIVE CONTROL: the old notice is detected rather than accepted merely
    # because it also happens to name CC BY-SA and BSD.
    planted_old = "Brown Corpus, via NLTK | Public domain | CC BY-SA 4.0 | BSD-2-Clause"
    assert brown_public_domain.search(planted_old)


def test_release_runs_keep_their_own_logs_when_receipts_share_a_parent(tmp_path):
    module = _module()
    receipts = (tmp_path / "first.json", tmp_path / "second.json")
    logs = [
        module._receipt_log_directory(path, token)
        for path, token in zip(receipts, ("0" * 32, "f" * 32), strict=True)
    ]
    first = module._run_step(
        module.Step("same-step", (sys.executable, "-c", "print('first run')"), 30), logs[0]
    )
    first_path = Path(first["log"])
    original = first_path.read_bytes()
    second = module._run_step(
        module.Step("same-step", (sys.executable, "-c", "print('second run')"), 30), logs[1]
    )
    assert first["exit_code"] == second["exit_code"] == 0
    assert first["log"] != second["log"]
    assert first_path.read_bytes() == original
    assert b"second run" in Path(second["log"]).read_bytes()
    # Reusing a receipt filename still cannot overwrite a previous run's logs.
    assert module._receipt_log_directory(receipts[0], "f" * 32) == logs[1]
    with pytest.raises(ValueError, match="32 lowercase hex"):
        module._receipt_log_directory(receipts[0], "../outside")


def test_publication_derivative_is_refused_by_quick_profile(tmp_path):
    # Given an explicit historical-evidence contract in the development profile.
    module = _module()
    # When its command inventory is requested.
    with pytest.raises(ValueError, match="valid only with the full profile"):
        module.build_steps("quick", publication_derivative=tmp_path / "contract.json")


def test_derivative_finish_requires_both_execution_results(tmp_path):
    # Given no executed offline or live prerequisite.
    module = _module()
    # When finish-state acceptance is evaluated.
    errors = module._publication_results_errors((), ())
    # Then neither missing prerequisite becomes a successful empty-loop check.
    assert len(errors) == 2
    assert any("network-claim-bindings" in error for error in errors)
    assert any("cache-audit" in error for error in errors)


def test_derivative_finish_refuses_failed_prerequisites():
    # Given real scheduled modes whose subprocesses failed.
    module = _module()
    steps = tuple(
        module.Step(name, ("unused",), 1) for name in ("network-claim-bindings", "cache-audit")
    )
    rows = tuple({"name": step.name, "status": "failed"} for step in steps)
    # When acceptance is evaluated.
    errors = module._publication_results_errors(steps, rows)
    # Then a failed result cannot supply typed evidence.
    assert len(errors) == 2
    assert all("did not pass" in error for error in errors)


def test_derivative_finish_refuses_forged_successful_nonproof_steps():
    # Given two successful statuses from commands that emitted no derivative proof.
    module = _module()
    steps = tuple(
        module.Step(name, ("unused",), 1) for name in ("network-claim-bindings", "cache-audit")
    )
    rows = tuple({"name": step.name, "status": "passed"} for step in steps)
    # When their result artifacts are reauthenticated.
    errors = module._publication_results_errors(steps, rows)
    # Then statuses alone cannot satisfy either prerequisite.
    assert len(errors) == 2
    assert all("changed or is absent" in error for error in errors)


def test_derivative_step_refuses_preexisting_attestation_before_process(tmp_path):
    # Given a prior artifact and a real subprocess that would leave a launch marker.
    module = _module()
    output = tmp_path / "old-attestation.json"
    output.write_text("{}\n", encoding="utf-8")
    launched = tmp_path / "launched"
    step = module.Step(
        "network-claim-bindings",
        (
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(launched)!r}).write_text('launched')",
            "scripts/verify_publication_derivative.py",
            "--output",
            str(output),
        ),
        10,
    )
    # When the step is executed through the real release surface.
    result = module._run_step(step, tmp_path / "logs")
    # Then it fails before process launch and preserves the prior evidence.
    assert result["status"] == "failed"
    assert not launched.exists()
    assert output.read_text(encoding="utf-8") == "{}\n"
    assert "must be newly emitted" in (tmp_path / "logs/network-claim-bindings.log").read_text()


def test_derivative_command_inventory_keeps_both_real_prerequisites(tmp_path):
    # Given the release checks and a dependent full-suite step.
    code = (
        "import json, sys\n"
        "sys.path.insert(0, 'scripts')\n"
        "from pathlib import Path\n"
        "from dataclasses import asdict\n"
        "import verify_release as m\n"
        "steps = (m.Step('environment-compatibility', ('unused',), 1), "
        "m.Step('network-claim-bindings', ('strict',), 1, depends_on=('environment-compatibility',)), "
        "m.Step('cache-audit', ('strict-live',), 1, depends_on=('network-claim-bindings',)), "
        "m.Step('tests-full', ('tests',), 1, depends_on=('cache-audit',)))\n"
        f"result = m._publication_derivative_steps(steps, Path({str(tmp_path / 'contract.json')!r}), "
        f"Path({str(tmp_path / 'proofs')!r}))\n"
        "m._validate_step_graph(result)\n"
        "print(json.dumps([asdict(step) for step in result]))\n"
    )
    # When the real isolated sidecar interpreter builds the derivative plan.
    completed = subprocess.run(
        [str(PROJECT_ROOT / ".venv/bin/python"), "-I", "-S", "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    # Then offline proof and live audit remain mandatory in their original order.
    assert completed.returncode == 0, completed.stderr
    steps = json.loads(completed.stdout)
    assert [step["name"] for step in steps] == [
        "environment-compatibility",
        "network-claim-bindings",
        "cache-audit",
        "tests-full",
    ]
    for step, mode in zip(steps[1:3], ("offline", "live"), strict=True):
        argv = step["argv"]
        assert argv[1:3] == ["-I", "-S"]
        assert "scripts/verify_publication_derivative.py" in argv
        assert argv[argv.index("--mode") + 1] == mode
        assert argv[argv.index("--contract") + 1] == str(tmp_path / "contract.json")
    assert steps[2]["depends_on"] == ["network-claim-bindings"]
    assert steps[3]["depends_on"] == ["cache-audit"]


def test_fresh_exit_zero_derivative_cannot_launch_descendant_without_proof(tmp_path):
    # Given a real exit-zero process that emits no mandatory derivative evidence.
    module = _module()
    contract = tmp_path / "contract.json"
    contract.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "missing-proof.json"
    launched = tmp_path / "descendant-launched"
    prerequisite = module.Step(
        "network-claim-bindings",
        (
            sys.executable,
            "-c",
            "print('real subprocess exited zero without a proof')",
            "scripts/verify_publication_derivative.py",
            "--contract",
            str(contract),
            "--mode",
            "offline",
            "--output",
            str(output),
        ),
        10,
    )
    descendant = module.Step(
        "cache-audit",
        (sys.executable, "-c", f"from pathlib import Path; Path({str(launched)!r}).touch()"),
        10,
        depends_on=("network-claim-bindings",),
    )
    # When keep-going executes the real graph, proof failure must still block descendants.
    rows = module._execute_steps(
        (prerequisite, descendant),
        tmp_path / "logs",
        environment=dict(os.environ),
        keep_going=True,
    )
    # Then successful subprocess status cannot replace proof-before-dependency authentication.
    assert [row["status"] for row in rows] == ["failed", "blocked"]
    assert not output.exists()
    assert not launched.exists()
    log = (tmp_path / "logs/network-claim-bindings.log").read_text()
    assert "real subprocess exited zero without a proof" in log
    assert "Publication derivative proof validation failed" in log


def test_real_typing_step_lists_derivative_source_without_executing_proof(tmp_path):
    module = _module()
    common_typing = next(step for step in module.build_steps("quick") if step.name == "types")
    steps = module._publication_derivative_steps(
        (common_typing,), tmp_path / "unused-contract.json", tmp_path / "proofs"
    )
    typing = next(step for step in steps if step.name == "types")
    assert "scripts/verify_publication_derivative.py" in typing.argv
    result = module._run_step(typing, tmp_path / "logs", environment=dict(os.environ))
    assert result["status"] == "passed", result
    assert "Success: no issues found" in result["stdout_tail"]
    assert "publication_derivative" not in result
    assert not (tmp_path / "proofs").exists()


@pytest.mark.parametrize(
    "kind", ["typing", "ordinary_argument", "direct", "controlled", "required"]
)
def test_derivative_classification_uses_command_role_and_script_target(tmp_path, kind):
    module = _module()
    target = "scripts/verify_publication_derivative.py"
    if kind == "typing":
        argv = module._controlled_python_argv("module", "mypy", target)
    elif kind == "ordinary_argument":
        argv = (sys.executable, "-c", "print('ordinary argument')", target)
    elif kind == "direct":
        argv = (sys.executable, "-I", "-S", "-B", target)
    elif kind == "controlled":
        argv = module._controlled_python_argv("script", target)
    else:
        argv = (sys.executable, "-c", "print('no proof')", target)
    step = module.Step("network-claim-bindings" if kind == "required" else "control", argv, 10)
    if kind in {"typing", "ordinary_argument"}:
        assert module._publication_output_path(step) is None
        assert module._publication_step_evidence(step) is None
    else:
        with pytest.raises(OSError, match="exactly one evidence output"):
            module._publication_output_path(step)
        with pytest.raises(ValueError, match="ambiguous --contract"):
            module._publication_step_evidence(step)


@pytest.mark.timeout(1500)
def test_derivative_cli_execution_consumes_real_proof_and_tracks_child_coverage(tmp_path):
    """Execute the reviewed full selfproof when the derivative release supplies it.

    Ordinary development runs drive the genuine missing-contract refusal. The
    explicit derivative release supplies its reviewed input and already executed
    live proof; then this test drives the positive CLI and finish-tamper controls.
    Standard Coverage records actual current-source child execution. Historical
    snapshot paths are never mapped onto current source or assigned synthetic hits.
    """
    import coverage

    module = _module()
    configured = os.environ.get("AILM_PUBLICATION_DERIVATIVE_CONTRACT")
    contract = Path(configured) if configured else tmp_path / "absent-contract.json"
    proof = tmp_path / "fresh-offline.json"
    child_data_path = tmp_path / "child.coverage"
    shim = tmp_path / "covered-derivative.py"
    measured_modules = (
        "scripts.verify_publication_derivative",
        "scripts._publication_execution",
        "scripts._publication_bootstrap",
        "scripts._publication_artists",
        "scripts._publication_controls",
        "scripts._publication_snapshot",
        "ailm.publication_derivative",
        "ailm._publication_types",
        "ailm._publication_inventory",
        "ailm._publication_contract",
        "ailm._publication_proof",
        "ailm._publication_attestation",
    )
    shim.write_text(
        "import sys\nfrom pathlib import Path\n"
        f"root = Path({str(PROJECT_ROOT)!r})\n"
        "sys.path.insert(0, str(root))\n"
        "from scripts._publication_bootstrap import audited_startup\n"
        "site, startup = audited_startup(root)\n"
        "sys.path[:0] = [str(root / 'src'), str(root), str(site)]\n"
        "import coverage\n"
        f"collector = coverage.Coverage(data_file={str(child_data_path)!r}, "
        f"branch=True, source={list(measured_modules)!r}, config_file=False)\n"
        "collector.start()\n"
        "try:\n"
        "    from scripts import verify_publication_derivative\n"
        "    result = verify_publication_derivative.main(sys.argv[2:])\n"
        "finally:\n"
        "    collector.stop()\n    collector.save()\n"
        "raise SystemExit(result)\n",
        encoding="utf-8",
    )
    prerequisite = module.Step(
        "network-claim-bindings",
        (
            str(PROJECT_ROOT / ".venv/bin/python"),
            "-I",
            "-S",
            "-B",
            str(shim),
            "scripts/verify_publication_derivative.py",
            "--contract",
            str(contract),
            "--mode",
            "offline",
            "--output",
            str(proof),
        ),
        1400,
    )
    launched = tmp_path / "after-authentication"
    descendant = module.Step(
        "after-proof",
        (sys.executable, "-c", f"from pathlib import Path; Path({str(launched)!r}).touch()"),
        10,
        depends_on=("network-claim-bindings",),
    )
    rows = module._execute_steps(
        (prerequisite, descendant),
        tmp_path / "logs",
        environment=dict(os.environ),
        keep_going=True,
    )
    assert child_data_path.is_file(), rows
    child_data = coverage.CoverageData(basename=str(child_data_path))
    child_data.read()
    measured_files = sorted(child_data.measured_files())
    assert str(PROJECT_ROOT / "scripts/verify_publication_derivative.py") in measured_files
    parent = coverage.Coverage.current()
    if parent is not None:
        # Retain only actual child arcs belonging to the parent's declared scope.
        declared = tuple(parent.config.source or ())
        selected = {}
        for filename in measured_files:
            relative = Path(filename).relative_to(PROJECT_ROOT).with_suffix("")
            dotted = ".".join(relative.parts)
            if dotted.startswith("src."):
                dotted = dotted.removeprefix("src.")
            if any(dotted == name or dotted.startswith(name + ".") for name in declared):
                selected[filename] = child_data.arcs(filename) or []
        actual_subset = coverage.CoverageData()
        actual_subset.add_arcs(selected)
        parent.get_data().update(actual_subset)
    if configured is None:
        assert [row["status"] for row in rows] == ["failed", "blocked"], rows
        assert not launched.exists()
        assert not proof.exists()
        return
    assert [row["status"] for row in rows] == ["passed", "passed"], rows
    assert launched.exists()
    assert rows[0]["publication_derivative"]["accepted"] is True
    assert (
        asdict(module._publication_step_evidence(prerequisite)) == rows[0]["publication_derivative"]
    )
    live_path = Path(os.environ["AILM_PUBLICATION_DERIVATIVE_LIVE_PROOF"])
    live_step = module.Step(
        "cache-audit",
        (
            "already-executed",
            "scripts/verify_publication_derivative.py",
            "--contract",
            str(contract),
            "--mode",
            "live",
            "--output",
            str(live_path),
        ),
        1,
    )
    live_evidence = module._publication_step_evidence(live_step)
    assert live_evidence is not None and live_evidence.accepted
    live_row = {
        "name": "cache-audit",
        "status": "passed",
        "publication_derivative": asdict(live_evidence),
    }
    assert module._publication_results_errors((prerequisite, live_step), (rows[0], live_row)) == ()
    (tmp_path / "pristine-offline.json").write_bytes(proof.read_bytes())
    proof.write_text("{}\n", encoding="utf-8")
    errors = module._publication_results_errors((prerequisite, live_step), (rows[0], live_row))
    assert any("network-claim-bindings" in error for error in errors), errors


def _dependency_identity_fixture(tmp_path):
    module = _module()
    root = (tmp_path / "dependency-project").resolve()
    _init_git_repository(root, with_release_script=True)
    for name in (
        "_publication_execution.py",
        "_publication_bootstrap.py",
        "_publication_artists.py",
        "_publication_controls.py",
        "_publication_snapshot.py",
    ):
        shutil.copy2(PROJECT_ROOT / "scripts" / name, root / "scripts" / name)
    package = root / "src/ailm"
    package.mkdir(parents=True)
    for path in (PROJECT_ROOT / "src/ailm").glob("_publication_*.py"):
        shutil.copy2(path, package / path.name)
    for name in ("__init__.py", "publication_derivative.py"):
        shutil.copy2(PROJECT_ROOT / "src/ailm" / name, package / name)
    site = module._venv_site_packages(root)
    return module, root, site


@pytest.mark.parametrize(
    "relative",
    ("review_dependency.py", "native_dependency.dylib", "__pycache__/hidden_source.py"),
)
def test_derivative_finish_rehashes_dependencies_after_retained_proof(tmp_path, relative):
    """A real dependency receipt refuses drift; full semantic acceptance is separate."""
    module, root, site = _dependency_identity_fixture(tmp_path)
    dependency = site / relative
    dependency.parent.mkdir(parents=True, exist_ok=True)
    dependency.write_bytes(b"VALUE = 1\n")
    before = module._publication_dependency_state(root)
    proof = tmp_path / "retained-dependency-proof.json"
    proof.write_text(json.dumps({"dependency_before": before, "dependency_after": before}))
    proof_hash = hashlib.sha256(proof.read_bytes()).hexdigest()
    assert module._publication_dependency_errors(proof, before) == ()
    dependency.write_bytes(b"VALUE = 2\n")
    after = module._publication_dependency_state(root)
    assert before["startup"] == after["startup"]
    assert before["inventory_sha256"] != after["inventory_sha256"]
    assert hashlib.sha256(proof.read_bytes()).hexdigest() == proof_hash
    assert module._publication_dependency_errors(proof, after) == (
        "actual current dependency bytes differ from retained proof",
    )


@pytest.mark.parametrize("kind", ["source-cache", "sourceless-cache"])
def test_dependency_executable_bytecode_drift_is_detected_under_isolation(tmp_path, kind):
    import marshal
    import py_compile

    module, root, site = _dependency_identity_fixture(tmp_path)
    source = site / "review_dependency.py"
    source.write_text("VALUE = 1\n")
    cache = Path(
        py_compile.compile(
            str(source), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP
        )
    )
    if kind == "sourceless-cache":
        destination = site / "review_dependency.pyc"
        cache.rename(destination)
        cache = destination
        source.unlink()
    source_before = source.read_bytes() if source.exists() else None

    def actual_import():
        command = (
            str(root / ".venv/bin/python"),
            "-I",
            "-S",
            "-B",
            "-c",
            f"import sys; sys.path.insert(0, {str(site)!r}); "
            "import review_dependency; print(review_dependency.VALUE)",
        )
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    assert actual_import() == "1"
    before = module._publication_dependency_state(root)
    assert before["bytecode_policy"] == "all-regular-executable-caches-included-v1"
    proof = tmp_path / "retained-bytecode-dependency-proof.json"
    proof.write_text(json.dumps({"dependency_before": before, "dependency_after": before}))
    proof_hash = hashlib.sha256(proof.read_bytes()).hexdigest()
    assert module._publication_dependency_errors(proof, before) == ()
    header = cache.read_bytes()[:16]
    changed_code = compile("VALUE = 2\n", str(source), "exec", dont_inherit=True)
    cache.write_bytes(header + marshal.dumps(changed_code))
    assert actual_import() == "2"
    after = module._publication_dependency_state(root)
    assert (source.read_bytes() if source.exists() else None) == source_before
    assert before["startup"] == after["startup"]
    assert before["inventory_sha256"] != after["inventory_sha256"]
    assert hashlib.sha256(proof.read_bytes()).hexdigest() == proof_hash
    assert module._publication_dependency_errors(proof, after) == (
        "actual current dependency bytes differ from retained proof",
    )


def test_derivative_bytecode_policy_controls_actual_python_startup(tmp_path):
    module = _module()
    original = module.Step(
        "python-probe",
        (
            str(PROJECT_ROOT / ".venv/bin/python"),
            "-I",
            "-S",
            "-X",
            f"pycache_prefix={tmp_path / 'legacy-pycache'}",
            "-c",
            "import sys; print(sys.dont_write_bytecode)",
        ),
        10,
    )
    derivative = module._publication_no_bytecode_step(original)
    for step, expected in ((original, "False"), (derivative, "True")):
        result = subprocess.run(step.argv, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == expected
    assert "-B" not in original.argv
    assert ("PYTHONDONTWRITEBYTECODE", "1") in derivative.environment_overrides


@pytest.mark.parametrize("kind", ["module", "script"])
def test_controlled_factory_bytecode_is_default(tmp_path, kind):
    module = _module()
    target = "ruff" if kind == "module" else "scripts/list_raw_corpus_inputs.py"
    option = "--version" if kind == "module" else "--help"
    argv = module._controlled_python_argv(kind, target, option)
    assert argv[1:4] == ("-I", "-S", "-B")
    original = module.Step("factory-probe", argv, 15)
    once = module._publication_no_bytecode_step(original)
    twice = module._publication_no_bytecode_step(once)
    assert once.argv == twice.argv == argv
    assert argv.count("-B") == 1
    environment = _release_environment(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS="")
    result = subprocess.run(
        argv,
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    (tmp_path / "command.json").write_text(
        json.dumps(
            {
                "argv": argv,
                "exit": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        ),
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    assert ("ruff " if kind == "module" else "usage:") in result.stdout.lower()


def test_controlled_factory_bytecode_preserves_assertions(tmp_path):
    module = _module()
    fixture = tmp_path / "standalone"
    fixture.mkdir()
    config = fixture / "pytest.ini"
    config.write_text("[pytest]\n", encoding="utf-8")
    test = fixture / "test_flags.py"
    test.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "def test_flags():\n"
        "    proof = dict(isolated=sys.flags.isolated, no_site=sys.flags.no_site, "
        "optimize=sys.flags.optimize, dont_write_bytecode=sys.dont_write_bytecode, "
        "prefix=sys.prefix, base_prefix=sys.base_prefix, executable=sys.executable)\n"
        "    Path(os.environ['RUNTIME_PROOF']).write_text(json.dumps(proof))\n"
        "    assert sys.flags.isolated and sys.flags.no_site\n"
        "    assert sys.flags.optimize == 0\n"
        "    assert sys.dont_write_bytecode == (os.environ['EXPECTED_B'] == '1')\n"
        "    assert int(os.environ['EXPECTED_SUM']) == 4\n",
        encoding="utf-8",
    )
    for name, enabled, expected_sum, expected_exit in (
        ("positive", True, 4, 0),
        ("wrong-assertion", True, 5, 1),
        ("legacy-counterexample", False, 4, 0),
    ):
        cache = tmp_path / (name + "-pycache")
        cache.mkdir()
        proof = tmp_path / (name + "-runtime.json")
        argv = list(
            module._controlled_python_argv(
                "module",
                "pytest",
                "-c",
                str(config),
                "--rootdir",
                str(fixture),
                "-o",
                "addopts=",
                "-p",
                "no:cacheprovider",
                "-q",
                str(test),
            )
        )
        if not enabled:
            argv.remove("-B")
        argv[1:1] = ["-X", f"pycache_prefix={cache}"]
        environment = _release_environment(
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            PYTEST_ADDOPTS="",
            EXPECTED_B="1" if enabled else "0",
            EXPECTED_SUM=str(expected_sum),
            RUNTIME_PROOF=str(proof),
        )
        result = subprocess.run(
            argv,
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        pycs = sorted(str(path.relative_to(cache)) for path in cache.rglob("*.pyc"))
        (tmp_path / (name + "-result.json")).write_text(
            json.dumps(
                {
                    "argv": argv,
                    "exit": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "pycs": pycs,
                }
            ),
            encoding="utf-8",
        )
        assert result.returncode == expected_exit, result.stdout + result.stderr
        runtime = json.loads(proof.read_text())
        assert runtime["isolated"] == runtime["no_site"] == 1
        assert runtime["optimize"] == 0 and runtime["dont_write_bytecode"] is enabled
        assert runtime["prefix"] == str(PROJECT_ROOT / ".venv")
        if enabled:
            assert pycs == []
        else:
            assert any("test_flags." in path and "-pytest-" in path for path in pycs)
        if expected_exit:
            assert "1 failed" in result.stdout and "assert 5 == 4" in result.stdout
        else:
            assert "1 passed" in result.stdout


@pytest.mark.timeout(480)
def test_quick_context_filter_has_real_collection_controls(tmp_path):
    template = tmp_path / "template"
    _init_template_repository(template)
    quick, full = _build_profiles_in_sidecar(template)
    quick_step = next(step for step in quick if step["name"] == "tests-quick")
    full_step = next(step for step in full if step["name"] == "tests-full")
    quick_expression = quick_step["argv"][quick_step["argv"].index("-m") + 1]
    full_expression = full_step["argv"][full_step["argv"].index("-m") + 1]
    selectors = [
        "tests/test_claims_status.py::test_registration_provenance_is_bound_to_git_history",
        "tests/test_publication_derivative.py::test_v5_original_raw_exact_identity_controls",
        "tests/test_publication_derivative.py::test_v5_unknown_raw_is_not_grandfathered",
        "tests/test_publication_derivative.py::test_v5_archival_authentication_controls",
        "tests/test_publication_derivative.py::test_v5_archival_declaration_strict_schema",
        "tests/test_publication_derivative.py::test_v5_actual_archival_omission_and_before_after_drift",
        "tests/test_publication_derivative.py::test_v5_raw_original_source_extraction_controls",
        "tests/test_publication_derivative.py::test_v5_archival_blob_framing_is_independent_of_declared_api",
        "tests/test_publication_derivative.py::test_v5_archival_and_raw_files_never_become_semantic_overlays",
        "tests/test_publication_derivative.py::test_v5_archival_missing_duplicate_and_escape_controls",
        "tests/test_publication_derivative.py::test_exact_original_worker_bytecode_remains_archival_and_refuses_drift",
        "tests/test_manuscript_tokens.py::TestCanonicalGrammar::test_grammar_is_isomorphic_to_the_template_renderer",
        "tests/test_manuscript_tokens.py::TestPublicationBoundary::test_resolved_tree_is_fully_staged_without_touching_live_tree",
        "tests/test_manuscript_tokens.py::test_real_resolved_tree_is_exactly_current",
        "tests/test_manuscript_tokens.py::test_staging_resolves_copied_auxiliary_preamble",
        "tests/test_manuscript_tokens.py::test_article_style_survives_actual_provider_preamble_refresh_and_compiles",
        "tests/test_manuscript_tokens.py::test_article_style_auxiliary_refuses_missing_or_stale_bytes",
        "tests/test_manuscript_tokens.py::test_missing_article_style_refuses_actual_latex_compile",
        "tests/test_manuscript_tokens.py::test_article_style_beamer_branch_preserves_slide_geometry",
        "tests/test_manuscript_tokens.py::test_article_style_is_not_a_standalone_manuscript_source",
        "tests/test_manuscript_tokens.py::test_combined_provider_boundary_refuses_invalid_interpreter",
        "tests/test_manuscript_tokens.py::test_standalone_tex_auxiliary_is_detected_and_refused",
        "tests/test_manuscript_tokens.py::test_article_style_transition_replaces_legacy_tree_without_standalone_auxiliary",
        "tests/test_reference_docs.py::test_whole_protocol_page_and_actual_pdf_destinations_survive_relocation",
        "tests/test_supporting_package.py::test_actual_citeproc_modes_prefixes_suffixes_and_multi_citations_are_transported",
        "tests/test_supporting_package.py::test_complete_relocated_package_and_production_mutations",
        "tests/test_supporting_package.py::test_production_cli_delivers_and_checks_both_editions_and_actual_consumers",
    ]
    portable = "tests/test_documentation.py::test_runtime_vendor_inventory_has_one_code_owned_source_of_truth"

    def collect(expression):
        child = subprocess.run(
            [
                str(PROJECT_ROOT / ".venv/bin/python"),
                "-B",
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-o",
                "addopts=",
                "--no-cov",
                "-m",
                expression,
                *selectors,
                portable,
            ],
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=80,
        )
        assert child.returncode == 0, child.stdout + child.stderr
        assert "PytestUnknownMarkWarning" not in child.stdout + child.stderr
        return {
            line for line in child.stdout.splitlines() if line.startswith("tests/") and "::" in line
        }

    contexts = collect("publication_context or template_context")
    assert len(contexts) == 103
    assert portable not in contexts
    assert collect(quick_expression) == {portable}
    assert collect(full_expression) == contexts | {portable}
    for marker in ("publication_context", "template_context"):
        broken_expression = quick_expression.replace(" and not " + marker, "")
        assert broken_expression != quick_expression
        leaked = collect(broken_expression)
        assert leaked > {portable}
        assert leaked - {portable} <= contexts
