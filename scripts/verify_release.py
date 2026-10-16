#!/usr/bin/env python3
"""Run reproducible AILM release profiles and emit an evidence receipt.

``quick`` is the developer loop: environment, formatting, lint, typing, all
portable tests excluding recorder replay and prepared publication/template contexts,
and a bounded gate audit. ``full`` is publication
acceptance: the same static checks, the complete coverage suite (including
isolated recorder replay), a repeated locked offline wheel/source-distribution
build with artifact digests, network-recorder claim bindings plus the live cache
audit, a machine-readable gate audit, the generated Lean
bridge plus its build and negative control, hydrated manuscript and documentation
trees, figures and their registry, template Stage 03 rendering, qpdf and structural
PDF checks, then template Stage 04 validation.

The receipt records every command, exit status, duration, log path, interpreter,
deterministic build environment, Git commit, content-sensitive snapshots of
unstaged/staged/untracked state, and the ignored external-call cache that the
full profile consumes. Git, uv, Lake, and qpdf are resolved without caller
``PATH`` and bound by availability, realpath, bytes, digest, and version result.
The distribution step records each package filename,
byte size, and SHA-256 digest in its receipt row. The template checkout is an equally fingerprinted release
input; an explicit ``TEMPLATE_REPO_ROOT`` is final and never silently replaced by
a fallback checkout. ``full`` refuses either dirty input unless the
development-only override is explicit; ``quick`` may start dirty but still fails
if the run changes any recorded input state. Generated output remains under
``output/``; stable publication surfaces are tracked and transient render
products are ignored.

The sidecar interpreter identity and stdlib-only startup are prerequisites:
invoke this script as ``.venv/bin/python -I -S scripts/verify_release.py ...``.
``-I -S`` prevents caller or virtual-environment startup code from executing
before the script can audit the release boundary. Sidecar Python steps use the
same flags and a controlled bootstrap that adds only the audited source and
site-packages directories without processing ``.pth`` files. A non-isolated,
site-enabled, bare-interpreter, or sidecar ``uv run`` invocation is refused
before any inventory or step runs. The two template steps still use that
repository's locked ``uv run`` contract.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import pwd
import re
import secrets
import stat
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:
    from ailm.publication_derivative import (
        AttestationResult,
        DerivativeContract,
        DerivativeState,
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECEIPT = PROJECT_ROOT / "output" / "reports" / "release-receipt.json"
SCHEMA = "ailm-release-receipt-v7"
TEMPLATE_PROJECT = "working/active_inference_language_model"
COMBINED_PDF = PROJECT_ROOT / "output" / "pdf" / "active_inference_language_model_combined.pdf"
DISTRIBUTION_MARKER = "AILM_DISTRIBUTION_ARTIFACT "
RELEASE_UV_CONFIG = PROJECT_ROOT / "release-uv.toml"
CONTROLLED_RUNNER_MARKER = "--_ailm-controlled-python"
ATOMIC_TEMP_HEX_LENGTH = 32
TOOL_VERSION_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "git": ("--version",),
    "uv": ("--version",),
    "lake": ("--version",),
    "qpdf": ("--version",),
}
APPROVED_EXECUTABLE_PTH_SHA256 = {
    (
        "coverage",
        "a1_coverage.pth",
    ): "ef2ed06d19867ec669c09a804060666a9cd5e383af0a9d11aa2de79b77d448e8",
    ("setuptools", "distutils-precedence.pth"): (
        "2638ce9e2500e572a5e0de7faed6661eb569d1b696fcba07b0dd223da5f5d224"
    ),
}

# Release steps inherit only operating-system context needed to launch local
# tools. Test/build behavior is set below, never inherited. In particular,
# PYTEST_ADDOPTS=--collect-only can otherwise make every pytest gate exit zero
# without executing a test; PYTHONOPTIMIZE can erase assert statements; proxy
# variables can reroute nominally loopback benchmark traffic; and tool-specific
# config variables can replace the checked-in configuration. The dedicated local
# benchmark origin is deliberately inherited, recorded and validated by its owner.
RELEASE_ENV_ALLOWLIST = frozenset(
    {
        "HOME",
        "AILM_OLLAMA_ENDPOINT",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "OPENROUTER_API_KEY",
        "SHELL",
        "TEMP",
        "TERM",
        "TMP",
        "TMPDIR",
        "USER",
    }
)
PYTEST_PASSED_RE = re.compile(r"(?<!\d)(?P<count>\d+)\s+passed\b")
PYTEST_SKIPPED_RE = re.compile(r"(?<!\d)(?P<count>\d+)\s+skipped\b")
PYTEST_XFAILED_RE = re.compile(r"(?<!\d)(?P<count>\d+)\s+xfailed\b")
COVERAGE_TOTAL_RE = re.compile(r"^TOTAL\s+.*?\s(?P<percent>\d+(?:\.\d+)?)%\s*$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class Step:
    """One bounded subprocess in a release profile."""

    name: str
    argv: tuple[str, ...]
    timeout_seconds: int
    depends_on: tuple[str, ...] = ()
    cwd: str | None = None
    environment_overrides: tuple[tuple[str, str], ...] = ()
    forbid_pytest_skips: bool = False


@dataclass(frozen=True, slots=True)
class RepositoryState:
    """Content-sensitive snapshot of every Git-visible repository state."""

    head: str
    branch: str
    status: tuple[str, ...]
    unstaged_sha256: str
    staged_sha256: str
    untracked_sha256: str
    index_flags: tuple[str, ...]
    fingerprint: str

    @property
    def dirty(self) -> bool:
        """Whether tracked, staged, or untracked state differs from ``HEAD``."""
        return bool(self.status)


def _login_home() -> Path:
    """Return the account home without trusting an inherited ``HOME`` value."""

    return Path(pwd.getpwuid(os.getuid()).pw_dir)


def _trusted_tool_directories() -> tuple[Path, ...]:
    """Return the fixed executable search path, independent of caller ``PATH``."""

    home = _login_home()
    system_root = Path(Path(sys.executable).anchor)
    configured_system_paths = tuple(
        Path(path) for path in (os.confstr("CS_PATH") or "").split(os.pathsep) if path
    )
    candidates = (
        *configured_system_paths,
        system_root / "usr" / "local" / "bin",
        system_root / "usr" / "local" / "sbin",
        home / ".local" / "bin",
        home / ".cargo" / "bin",
        home / ".elan" / "bin",
        system_root / "opt" / "homebrew" / "bin",
        system_root / "opt" / "homebrew" / "sbin",
        system_root / "System" / "Cryptexes" / "App" / "usr" / "bin",
        system_root / "usr" / "sbin",
        system_root / "sbin",
        system_root / "Library" / "Apple" / "usr" / "bin",
        system_root / "Library" / "TeX" / "texbin",
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = str(candidate)
        if text not in seen:
            seen.add(text)
            unique.append(candidate)
    return tuple(unique)


def _sanitized_path() -> str:
    """Return the release ``PATH`` assembled only from fixed locations."""

    return os.pathsep.join(str(path) for path in _trusted_tool_directories())


def _base_release_environment() -> dict[str, str]:
    """Return approved OS/network context plus a caller-independent executable path."""

    environment = {
        name: value for name, value in os.environ.items() if name in RELEASE_ENV_ALLOWLIST
    }
    environment["HOME"] = str(_login_home())
    environment["PATH"] = _sanitized_path()
    return environment


def _resolve_executable(name: str) -> Path | None:
    """Select one executable only through the fixed release search path.

    The returned path deliberately preserves the requested basename. Dispatch
    proxies such as elan's ``lake`` change behavior from ``argv[0]``; executing
    their resolved ``elan-init`` target is a different command. The tool receipt
    separately resolves and fingerprints the exact regular-file target.
    """

    if name not in TOOL_VERSION_ARGUMENTS:
        raise ValueError(f"unsupported release tool: {name!r}")
    for directory in _trusted_tool_directories():
        candidate = directory / name
        try:
            metadata = candidate.stat()
            resolved = candidate.resolve(strict=True)
            resolved_metadata = resolved.stat()
        except (FileNotFoundError, OSError, RuntimeError):
            continue
        if (
            stat.S_ISREG(metadata.st_mode)
            and stat.S_ISREG(resolved_metadata.st_mode)
            and os.access(candidate, os.X_OK)
        ):
            return candidate.absolute()
    return None


def _read_regular_file_identity(path: Path) -> dict[str, object]:
    """Fingerprint one resolved regular file through the descriptor that is read."""

    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"executable is not a regular file: {path}")
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    finally:
        os.close(descriptor)
    return {
        "realpath": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": size,
        "mode": metadata.st_mode,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
    }


def _tool_identity(name: str, *, check_version: bool) -> dict[str, object]:
    """Bind one external executable's path, bytes, and version result."""

    invocation = _resolve_executable(name)
    row: dict[str, object] = {
        "name": name,
        "available": invocation is not None,
        "usable": False,
        "path": str(invocation) if invocation is not None else None,
        "realpath": None,
        "sha256": None,
        "size_bytes": 0,
        "mode": None,
        "device": None,
        "inode": None,
        "version_argv": None,
        "version_checked": check_version,
        "version_exit_code": None,
        "version_output": None,
        "error": None,
    }
    if invocation is None:
        row["error"] = f"{name} was not found in the fixed release search path"
        return row
    try:
        resolved = invocation.resolve(strict=True)
        row.update(_read_regular_file_identity(resolved))
    except (OSError, ValueError) as error:
        row["available"] = False
        row["error"] = str(error)
        return row
    if not check_version:
        row["error"] = "version invocation suppressed after an earlier refusal"
        return row

    argv = (str(invocation), *TOOL_VERSION_ARGUMENTS[name])
    version_cwd = PROJECT_ROOT / "formal" if name == "lake" else PROJECT_ROOT
    try:
        result = subprocess.run(
            argv,
            cwd=version_cwd,
            env=_base_release_environment(),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        row["version_argv"] = list(argv)
        row["error"] = f"version invocation failed: {error}"
        return row
    output = (result.stdout + result.stderr).strip()
    row.update(
        {
            "version_argv": list(argv),
            "version_exit_code": result.returncode,
            "version_output": output[-2000:],
            "usable": result.returncode == 0,
            "error": None if result.returncode == 0 else "version command returned nonzero",
        }
    )
    return row


def _toolchain_state(*, check_versions: bool) -> dict[str, object]:
    """Fingerprint the fixed external toolchain for the release receipt."""

    rows = {
        name: _tool_identity(name, check_version=check_versions) for name in TOOL_VERSION_ARGUMENTS
    }
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "ailm-release-toolchain-v2",
        "path_policy": "fixed-search-path-preserve-dispatch-basename-v2",
        "PATH": _sanitized_path(),
        "tools": rows,
        "fingerprint": hashlib.sha256(encoded).hexdigest(),
    }


def _tool_path(toolchain: Mapping[str, object], name: str) -> str:
    """Return a usable fingerprinted tool path or a fail-closed sentinel."""

    tools = toolchain.get("tools")
    if not isinstance(tools, Mapping):
        raise ValueError("toolchain receipt has no tool map")
    row = tools.get(name)
    if not isinstance(row, Mapping):
        raise ValueError(f"toolchain receipt has no {name!r} entry")
    path = row.get("path")
    if row.get("usable") is True and isinstance(path, str):
        return path
    return str(Path(Path(sys.executable).anchor) / "__ailm_missing_or_unusable_tool__" / name)


def _venv_site_packages(root: Path = PROJECT_ROOT) -> Path:
    """Return the sidecar site-packages path without importing :mod:`site`."""

    return (
        root.resolve()
        / ".venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )


def _record_owned_paths(site_packages: Path) -> dict[str, tuple[str, str | None]]:
    """Map installed paths to their distribution and RECORD digest."""

    owned: dict[str, tuple[str, str | None]] = {}
    for record in sorted(site_packages.glob("*.dist-info/RECORD")):
        try:
            metadata = record.lstat()
            if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                continue
            distribution = record.parent.name.removesuffix(".dist-info")
            metadata_path = record.parent / "METADATA"
            if metadata_path.is_file() and not metadata_path.is_symlink():
                for line in metadata_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines():
                    if line.startswith("Name: "):
                        distribution = line.removeprefix("Name: ").strip().casefold()
                        break
            with record.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.reader(handle):
                    if len(row) < 3:
                        continue
                    raw_path, digest, _size = row[:3]
                    if (
                        raw_path
                        and not Path(raw_path).is_absolute()
                        and ".." not in Path(raw_path).parts
                    ):
                        owned[Path(raw_path).as_posix()] = (distribution, digest or None)
        except (OSError, UnicodeError, csv.Error):
            continue
    return owned


def _record_digest_matches(content: bytes, recorded: str | None) -> bool:
    """Whether a RECORD sha256 entry authenticates ``content``."""

    if recorded is None or not recorded.startswith("sha256="):
        return False
    encoded = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode("ascii")
    return encoded.rstrip("=") == recorded.removeprefix("sha256=")


def _startup_customization_candidates(site_packages: Path) -> tuple[Path, ...]:
    """Return venv modules that :mod:`site` would import implicitly."""

    candidates: list[Path] = []
    for stem in ("sitecustomize", "usercustomize"):
        candidates.extend(sorted(site_packages.glob(f"{stem}.*")))
        package = site_packages / stem
        if package.exists() or package.is_symlink():
            candidates.append(package)
    return tuple(dict.fromkeys(candidates))


def _venv_startup_state(root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Audit every implicit venv startup surface without executing one.

    Sidecar commands run with ``-S`` and therefore do not process these files.
    The inventory nevertheless fails closed on customizers, symlinked, unsafe,
    unowned, or unknown executable ``.pth`` entries so a later command cannot
    silently regress to ordinary :mod:`site` startup.
    """

    site_packages = _venv_site_packages(root)
    errors: list[str] = []
    entries: list[dict[str, object]] = []
    project = root.resolve()
    current = project
    for component in site_packages.relative_to(project).parts:
        current /= component
        try:
            component_metadata = current.lstat()
        except OSError as error:
            errors.append(f"virtual-environment path component is unavailable: {current}: {error}")
            break
        if not stat.S_ISDIR(component_metadata.st_mode) or stat.S_ISLNK(component_metadata.st_mode):
            errors.append(f"virtual-environment path component is not a real directory: {current}")
            break
        if component_metadata.st_uid not in {0, os.getuid()}:
            errors.append(
                f"virtual-environment path component has unsafe owner uid "
                f"{component_metadata.st_uid}: {current}"
            )
            break
    try:
        directory_metadata = site_packages.lstat()
    except OSError as error:
        return {
            "schema": "ailm-venv-startup-surface-v1",
            "site_packages": str(site_packages),
            "controlled_import_paths": [],
            "pth_entries": [],
            "customization_modules": [],
            "integrity_ok": False,
            "errors": [f"site-packages is unavailable: {error}"],
            "fingerprint": None,
        }
    if not stat.S_ISDIR(directory_metadata.st_mode) or stat.S_ISLNK(directory_metadata.st_mode):
        errors.append("site-packages is not a real directory")
    if directory_metadata.st_uid not in {0, os.getuid()}:
        errors.append(f"site-packages has unsafe owner uid {directory_metadata.st_uid}")

    customizers = _startup_customization_candidates(site_packages)
    if customizers:
        errors.extend(
            f"implicit Python customization module is present: {path.name}" for path in customizers
        )

    owned = _record_owned_paths(site_packages)
    project_source = (root.resolve() / "src").resolve()
    for path in sorted(site_packages.glob("*.pth")):
        entry: dict[str, object] = {
            "name": path.name,
            "owner": None,
            "record_sha256": None,
            "sha256": None,
            "size_bytes": 0,
            "mode": None,
            "uid": None,
            "executable_directives": [],
            "path_directives": [],
            "inert_under_controlled_startup": True,
            "integrity_ok": False,
            "errors": [],
        }
        entry_errors: list[str] = []
        try:
            metadata = path.lstat()
            entry["mode"] = metadata.st_mode
            entry["uid"] = metadata.st_uid
            if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise ValueError("entry is not a real regular file")
            if metadata.st_uid not in {0, os.getuid()}:
                raise ValueError(f"entry has unsafe owner uid {metadata.st_uid}")
            if stat.S_IMODE(metadata.st_mode) & 0o111:
                raise ValueError("entry has executable filesystem mode")
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0),
            )
            try:
                opened_metadata = os.fstat(descriptor)
                if not stat.S_ISREG(opened_metadata.st_mode):
                    raise ValueError("opened entry is not a regular file")
                if (opened_metadata.st_dev, opened_metadata.st_ino) != (
                    metadata.st_dev,
                    metadata.st_ino,
                ):
                    raise ValueError("entry changed while its identity was captured")
                bounded = bytearray()
                while chunk := os.read(descriptor, 1024 * 1024 + 1 - len(bounded)):
                    bounded.extend(chunk)
                    if len(bounded) > 1024 * 1024:
                        raise ValueError("entry exceeds the bounded startup-surface size")
                content = bytes(bounded)
            finally:
                os.close(descriptor)
            entry["size_bytes"] = len(content)
            entry["sha256"] = hashlib.sha256(content).hexdigest()

            owner = owned.get(path.name)
            if owner is None and path.name == "_virtualenv.pth":
                sibling = site_packages / "_virtualenv.py"
                if content not in {b"import _virtualenv", b"import _virtualenv\n"}:
                    raise ValueError("canonical uv virtualenv bootstrap content changed")
                sibling_metadata = sibling.lstat()
                if (
                    not stat.S_ISREG(sibling_metadata.st_mode)
                    or stat.S_ISLNK(sibling_metadata.st_mode)
                    or sibling_metadata.st_uid not in {0, os.getuid()}
                ):
                    raise ValueError("canonical uv virtualenv bootstrap module is unsafe")
                distribution = "uv-virtualenv-bootstrap"
                recorded_digest = None
            elif owner is None:
                raise ValueError("entry is not owned by an installed distribution RECORD")
            else:
                distribution, recorded_digest = owner
                if not _record_digest_matches(content, recorded_digest):
                    raise ValueError("entry bytes do not match their distribution RECORD")
            entry["owner"] = distribution
            entry["record_sha256"] = recorded_digest

            executable_lines: list[str] = []
            path_lines: list[str] = []
            for raw_line in content.decode("utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("import ") or line.startswith("import\t"):
                    executable_lines.append(line)
                else:
                    path_lines.append(line)
            entry["executable_directives"] = executable_lines
            entry["path_directives"] = path_lines
            if executable_lines and distribution != "uv-virtualenv-bootstrap":
                approved_digest = APPROVED_EXECUTABLE_PTH_SHA256.get((distribution, path.name))
                if approved_digest is None:
                    raise ValueError("entry contains an unapproved executable directive")
                if entry["sha256"] != approved_digest:
                    raise ValueError("approved executable entry bytes changed")
            for directive in path_lines:
                candidate = Path(directive)
                if not candidate.is_absolute():
                    candidate = site_packages / candidate
                resolved = candidate.resolve(strict=True)
                if resolved != project_source and site_packages not in resolved.parents:
                    raise ValueError(f"entry exposes an uncontrolled import path: {resolved}")
            entry["integrity_ok"] = True
        except (OSError, RuntimeError, UnicodeError, ValueError) as error:
            entry_errors.append(str(error))
        entry["errors"] = entry_errors
        errors.extend(f"{path.name}: {error}" for error in entry_errors)
        entries.append(entry)

    payload = {
        "schema": "ailm-venv-startup-surface-v1",
        "site_packages": str(site_packages),
        "controlled_import_paths": [
            str(root.resolve() / "src"),
            str(root.resolve()),
            str(site_packages),
        ],
        "pth_entries": entries,
        "customization_modules": [str(path) for path in customizers],
        "integrity_ok": not errors,
        "errors": errors,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _startup_errors(state: Mapping[str, object]) -> list[str]:
    """Narrow the startup receipt's error field at each consumption boundary."""

    value = state.get("errors")
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return ["virtual-environment startup receipt has a malformed errors field"]
    return value


def _project_python_error(root: Path = PROJECT_ROOT) -> str | None:
    """Explain why this process is not the sidecar's explicitly addressed Python."""

    expected = root.resolve() / ".venv" / "bin" / "python"
    actual = Path(sys.executable).absolute()
    if os.environ.get("UV_RUN_RECURSION_DEPTH") is not None:
        return (
            "release orchestration was launched through uv run; invoke the sidecar "
            f"interpreter directly as {expected}"
        )
    if actual != expected:
        return (
            f"release orchestration must be invoked as {expected}; this process is "
            f"{actual}. Do not use uv run or a bare Python executable in the sidecar"
        )
    if not expected.is_file() or not os.access(expected, os.X_OK):
        return f"project interpreter is missing or not executable: {expected}"
    return None


def _project_python(root: Path = PROJECT_ROOT) -> str:
    """Return the exact sidecar interpreter or fail before building an inventory."""

    error = _project_python_error(root)
    if error is not None:
        raise RuntimeError(error)
    return str(root.resolve() / ".venv" / "bin" / "python")


def _isolated_startup_error() -> str | None:
    """Explain why this process began before stdlib-only isolation was established."""

    if not sys.flags.isolated or not sys.flags.no_site:
        return (
            "release orchestration requires isolated, stdlib-only Python startup; invoke "
            ".venv/bin/python -I -S scripts/verify_release.py so PYTHONPATH, .pth files, "
            "sitecustomize, and usercustomize cannot execute before the release boundary"
        )
    return None


def _controlled_python_argv(kind: str, target: str, *arguments: str) -> tuple[str, ...]:
    """Build one sidecar Python command that never imports :mod:`site`."""

    if kind not in {"module", "script"}:
        raise ValueError(f"unsupported controlled Python target kind: {kind!r}")
    return (
        _project_python(),
        "-I",
        "-S",
        "-B",
        str(PROJECT_ROOT / "scripts" / "verify_release.py"),
        CONTROLLED_RUNNER_MARKER,
        kind,
        target,
        *arguments,
    )


def _run_controlled_python(arguments: Sequence[str]) -> int:
    """Run one module or project script with explicit audited import roots."""

    startup_error = _isolated_startup_error()
    if startup_error is not None:
        print(f"controlled Python startup error: {startup_error}", file=sys.stderr)
        return 2
    interpreter_error = _project_python_error()
    if interpreter_error is not None:
        print(f"controlled Python startup error: {interpreter_error}", file=sys.stderr)
        return 2
    startup_state = _venv_startup_state()
    if startup_state["integrity_ok"] is not True:
        print(
            "controlled Python startup error: unsafe virtual-environment startup "
            + "; ".join(_startup_errors(startup_state)),
            file=sys.stderr,
        )
        return 2
    if len(arguments) < 2 or arguments[0] not in {"module", "script"}:
        print("controlled Python startup error: malformed target", file=sys.stderr)
        return 2

    kind, target, *target_arguments = arguments
    root = PROJECT_ROOT.resolve()
    site_packages = _venv_site_packages(root)
    sys.path[:0] = [str(root / "src"), str(root), str(site_packages)]
    sys.prefix = str(root / ".venv")
    sys.exec_prefix = sys.prefix
    import runpy

    if kind == "module":
        if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", target) is None:
            print(f"controlled Python startup error: invalid module {target!r}", file=sys.stderr)
            return 2
        sys.argv = [target, *target_arguments]
        runpy.run_module(target, run_name="__main__", alter_sys=True)
        return 0

    relative = Path(target)
    if relative.is_absolute() or ".." in relative.parts:
        print(f"controlled Python startup error: invalid script path {target!r}", file=sys.stderr)
        return 2
    script = (root / relative).resolve(strict=True)
    try:
        script.relative_to(root)
    except ValueError:
        print(
            f"controlled Python startup error: script escapes project root: {script}",
            file=sys.stderr,
        )
        return 2
    metadata = script.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        print(
            f"controlled Python startup error: script is not a regular file: {script}",
            file=sys.stderr,
        )
        return 2
    sys.argv = [str(script), *target_arguments]
    runpy.run_path(str(script), run_name="__main__")
    return 0


def _validate_template_root(candidate: Path) -> Path:
    """Return one exact template checkout or fail without searching elsewhere."""
    root = candidate.expanduser().resolve()
    project_entry = root / "projects" / TEMPLATE_PROJECT
    required = (
        root / "infrastructure" / "__init__.py",
        root / "scripts" / "pipeline" / "stage_03_render.py",
        root / "scripts" / "pipeline" / "stage_04_validate.py",
        root / "pyproject.toml",
        root / "uv.lock",
        project_entry,
    )
    missing = [
        str(path.relative_to(root))
        for path in required
        if not (path.is_dir() if path == project_entry else path.is_file())
    ]
    if missing:
        raise RuntimeError(
            f"template root {root} is missing required publication files: " + ", ".join(missing)
        )
    routed_project = project_entry.resolve()
    if routed_project != PROJECT_ROOT.resolve():
        raise RuntimeError(
            f"template root {root} routes {TEMPLATE_PROJECT} to {routed_project}, "
            f"not the audited sidecar {PROJECT_ROOT.resolve()}"
        )
    return root


def _resolve_template_root() -> Path:
    """Resolve the template provider, treating an explicit environment value as final."""
    explicit = os.environ.get("TEMPLATE_REPO_ROOT", "").strip()
    if explicit:
        return _validate_template_root(Path(explicit))

    # The candidate order is byte-identical in scripts/z_generate_manuscript_variables.py
    # and scripts/generate_figure_registry.py; tests/test_figure_registry.py holds them to it.
    candidates = [
        *PROJECT_ROOT.parents,
        *(parent / "template" for parent in PROJECT_ROOT.parents[2:3]),
        _login_home() / "Documents" / "GitHub" / "template",
    ]
    for candidate in candidates:
        if (candidate / "infrastructure" / "__init__.py").is_file():
            return _validate_template_root(candidate)
    raise RuntimeError(
        "could not locate the template checkout required for publication; "
        "set TEMPLATE_REPO_ROOT to its exact root"
    )


def _coverage_floor(root: Path = PROJECT_ROOT) -> int:
    """Return the branch-coverage floor declared once in ``pyproject.toml``.

    The floor is ``[tool.coverage.report] fail_under``. ``scripts/record_stage0_claims.py``
    publishes the same key as ``claim:stage0_coverage_floor_pct``, and pytest-cov
    applies it whenever ``--cov`` is passed, so the release step spells the value
    out only so the receipt shows which gate ran; it is never a second copy.
    """
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    try:
        floor = pyproject["tool"]["coverage"]["report"]["fail_under"]
    except KeyError as error:
        raise RuntimeError(
            "pyproject.toml declares no [tool.coverage.report] fail_under; the release "
            "coverage gate has no floor to enforce"
        ) from error
    if isinstance(floor, bool) or not isinstance(floor, int) or not 0 < floor <= 100:
        raise RuntimeError(f"pyproject.toml fail_under is not a percentage: {floor!r}")
    return floor


def _coverage_gate_argument(root: Path = PROJECT_ROOT) -> str:
    """The pytest-cov flag for the declared floor, as recorded in the receipt."""
    return f"--cov-fail-under={_coverage_floor(root)}"


def build_steps(
    profile: str,
    *,
    template_root: Path | None = None,
    toolchain: Mapping[str, object] | None = None,
    publication_derivative: Path | None = None,
    derivative_output: Path | None = None,
) -> tuple[Step, ...]:
    """Return the deterministic command inventory for a profile."""
    if publication_derivative is not None and profile != "full":
        raise ValueError("--publication-derivative is valid only with the full profile")
    resolved_toolchain = _toolchain_state(check_versions=True) if toolchain is None else toolchain
    uv = _tool_path(resolved_toolchain, "uv")
    lake = _tool_path(resolved_toolchain, "lake")
    qpdf = _tool_path(resolved_toolchain, "qpdf")
    # Full requires the upstream GNN parser; quick checks the base environment.
    environment_extras = ("--extra", "gnn-interop") if profile == "full" else ()
    common = (
        Step("lock", (uv, "lock", "--check"), 120),
        Step(
            "environment",
            (
                uv,
                "sync",
                "--check",
                "--locked",
                *environment_extras,
                "--python",
                _project_python(),
            ),
            300,
            depends_on=("lock",),
        ),
        Step(
            "environment-compatibility",
            (uv, "pip", "check", "--python", _project_python()),
            120,
            depends_on=("environment",),
        ),
        Step(
            "format",
            _controlled_python_argv(
                "module", "ruff", "format", "--check", "src", "scripts", "tests"
            ),
            120,
            depends_on=("environment-compatibility",),
        ),
        Step(
            "lint",
            _controlled_python_argv("module", "ruff", "check", "src", "scripts", "tests"),
            120,
            depends_on=("environment-compatibility",),
        ),
        Step(
            "types",
            _controlled_python_argv(
                "module",
                "mypy",
                "--explicit-package-bases",
                "src/ailm",
                "scripts/verify_release.py",
                "scripts/verify_network_claim_bindings.py",
                "scripts/verify_publication_derivative.py",
                "scripts/_publication_execution.py",
                "scripts/_publication_bootstrap.py",
                "scripts/_publication_artists.py",
                "scripts/_publication_controls.py",
                "scripts/_publication_snapshot.py",
                "scripts/orchestrate_claim_recorders.py",
                "scripts/warm_call_cache.py",
                "scripts/record_external_benchmark_claims.py",
                "scripts/record_synthetic_benchmark_claims.py",
                "scripts/record_incontext_benchmark_claims.py",
                "scripts/record_logprob_drift_claims.py",
            ),
            1800,
            depends_on=("environment-compatibility",),
        ),
        Step(
            "web-model",
            _controlled_python_argv("script", "scripts/export_web_model.py"),
            1800,
            depends_on=("environment-compatibility",),
        ),
        Step(
            "web-page",
            _controlled_python_argv("script", "scripts/build_web_ailm.py"),
            300,
            depends_on=("web-model",),
        ),
    )
    if profile == "quick":
        return common + (
            Step(
                "tests-quick",
                _controlled_python_argv(
                    "module",
                    "pytest",
                    "tests/",
                    "-q",
                    "--no-cov",
                    "-m",
                    "not recorder_replay and not raw_corpus_regeneration and not requires_cuda12 "
                    "and not publication_context and not template_context",
                ),
                3600,
                depends_on=("environment-compatibility", "web-page"),
            ),
            Step(
                "gate-audit-bounded",
                _controlled_python_argv("script", "scripts/audit_gates.py", "--limit", "400"),
                1800,
                depends_on=("environment-compatibility",),
            ),
        )
    if profile == "full":
        resolved_template = (
            _resolve_template_root()
            if template_root is None
            else _validate_template_root(template_root)
        )
        source_acceptance = (
            "lock",
            "environment",
            "environment-compatibility",
            "format",
            "lint",
            "types",
            "web-model",
            "web-page",
            "package-distributions",
            "network-claim-bindings",
            "cache-audit",
            "raw-corpus-reproducibility",
            "pymdp-sparse-bundle-generate",
            "pymdp-sparse-bundle-check",
            "release-infrastructure-coverage",
            "tests-full",
            "gate-audit-full",
            "lean-ledger-check",
            "lean-build",
            "lean-negative-control",
        )
        full_steps = common + (
            Step(
                "package-distributions",
                _controlled_python_argv(
                    "module",
                    "pytest",
                    "tests/test_wheel_resources.py::test_installed_wheel_embeds_exact_artifacts_and_default_loaders_work",
                    "-q",
                    "--no-cov",
                    "-s",
                ),
                # The test itself carries `@pytest.mark.timeout(900)` (building
                # two wheels and smoke-testing one is genuinely slow); the outer
                # bound must exceed that or this step can be killed by
                # verify_release before pytest's own timeout ever fires.
                1800,
                depends_on=("environment-compatibility",),
                forbid_pytest_skips=True,
            ),
            # Fail fast on the only non-Git input the coverage suite consumes.
            # The audit requires a reachable, matching replay from every
            # registered model × request-kind cell. Replies are exact;
            # answer-slot logprob values use the recorder's declared tolerance.
            # A cold, partial, unreachable, or changed store is a hard failure.
            Step(
                "network-claim-bindings",
                _controlled_python_argv("script", "scripts/verify_network_claim_bindings.py"),
                1800,
                depends_on=("environment-compatibility",),
            ),
            Step(
                "cache-audit",
                _controlled_python_argv(
                    "script",
                    "scripts/record_external_benchmark_claims.py",
                    "--verify-cache",
                    "25",
                ),
                1800,
                depends_on=("network-claim-bindings",),
            ),
            Step(
                "raw-corpus-reproducibility",
                _controlled_python_argv(
                    "module",
                    "pytest",
                    "tests/test_corpus_source_integrity.py",
                    "tests/test_cmudict_load.py",
                    "tests/test_provenance_checksums.py",
                    "-q",
                    "--no-cov",
                    "-m",
                    "raw_corpus_regeneration",
                    "--runxfail",
                    "-rs",
                ),
                1800,
                depends_on=("environment-compatibility",),
                environment_overrides=(("AILM_RUN_RAW_CORPUS_REGENERATION", "1"),),
                forbid_pytest_skips=True,
            ),
            Step(
                "pymdp-sparse-bundle-generate",
                _controlled_python_argv("script", "scripts/generate_pymdp_sparse_patch.py"),
                300,
                depends_on=("environment-compatibility",),
            ),
            Step(
                "pymdp-sparse-bundle-check",
                _controlled_python_argv(
                    "script", "scripts/generate_pymdp_sparse_patch.py", "--check"
                ),
                300,
                depends_on=("pymdp-sparse-bundle-generate",),
            ),
            Step(
                "release-infrastructure-coverage",
                _controlled_python_argv(
                    "module",
                    "pytest",
                    "tests/test_release_orchestration.py",
                    "tests/test_replay_provenance.py",
                    "tests/test_claim_recorder_orchestration.py",
                    "tests/test_publication_derivative.py",
                    "-q",
                    "--cov=scripts.verify_release",
                    "--cov=scripts.verify_network_claim_bindings",
                    "--cov=scripts.verify_publication_derivative",
                    "--cov=scripts._publication_execution",
                    "--cov=scripts._publication_bootstrap",
                    "--cov=scripts._publication_artists",
                    "--cov=scripts._publication_controls",
                    "--cov=scripts._publication_snapshot",
                    "--cov=scripts.orchestrate_claim_recorders",
                    "--cov=scripts.warm_call_cache",
                    "--cov-branch",
                    "--cov-report=term-missing",
                    "--cov-fail-under=60",
                    "-k",
                    "not real_reversed_dependency",
                    "-rsx",
                ),
                1800,
                depends_on=("environment-compatibility",),
                forbid_pytest_skips=True,
            ),
            Step(
                "tests-full",
                _controlled_python_argv(
                    "module",
                    "pytest",
                    "tests/",
                    "-q",
                    "--cov=ailm",
                    "--cov-report=term-missing",
                    _coverage_gate_argument(),
                    "-rsx",
                    "--ignore=tests/test_release_orchestration.py",
                    "-m",
                    "not raw_corpus_regeneration and not requires_cuda12",
                ),
                43_200,
                depends_on=(
                    "cache-audit",
                    "raw-corpus-reproducibility",
                    "web-page",
                    "pymdp-sparse-bundle-check",
                    "release-infrastructure-coverage",
                ),
                # The full profile forbids skips, so every environment-dependent
                # test must be told to fail loudly rather than skip when its
                # prerequisite is missing: the mandatory-environment flag turns
                # "not installed here" into a release failure that names the gap.
                environment_overrides=(
                    ("AILM_RUN_LIVE_OLLAMA_TESTS", "1"),
                    ("AILM_RELEASE_MANDATORY_ENVIRONMENT", "1"),
                ),
                forbid_pytest_skips=True,
            ),
            Step(
                "gate-audit-full",
                _controlled_python_argv("script", "scripts/audit_gates.py", "--full", "--json"),
                3600,
                depends_on=("environment-compatibility",),
            ),
            Step(
                "lean-ledger-check",
                _controlled_python_argv("script", "scripts/generate_lean_ledger.py", "--check"),
                300,
                depends_on=("environment-compatibility",),
            ),
            Step(
                "lean-build",
                (lake, "build"),
                600,
                depends_on=("lean-ledger-check",),
                cwd=str(PROJECT_ROOT / "formal"),
            ),
            Step(
                "lean-negative-control",
                _controlled_python_argv(
                    "module",
                    "pytest",
                    "tests/test_formal_isomorphism.py::test_the_negative_control_does_not_compile",
                    "-q",
                    "--no-cov",
                ),
                600,
                depends_on=("lean-build",),
                forbid_pytest_skips=True,
            ),
            # ORDER IS LOAD-BEARING: manuscript-variables before figures.
            #
            # The figure modules read `output/data/manuscript_variables.json` and
            # refuse to draw a number that is not in it. That refusal is only a
            # guard while the map is current. The map is a tracked file, so a
            # stale one never raises MissingVariableError -- it silently supplies
            # the *previous* ledger, and the figures draw it.
            #
            # That is not hypothetical. Harmonizing the coverage levels moved the
            # recorded share intervals from 0.99 to 0.95, and because figures ran
            # first, `multiplicity_forest.png` shipped drawing 99% whiskers over a
            # ledger holding 95% -- past a green suite, in the one figure whose
            # subject is the multiplicity correction.
            #
            Step(
                "manuscript-variables",
                _controlled_python_argv("script", "scripts/z_generate_manuscript_variables.py"),
                300,
                depends_on=source_acceptance,
            ),
            Step(
                "hydrated-docs",
                _controlled_python_argv(
                    "script", "scripts/z_render_docs.py", "--supporting-protocols"
                ),
                300,
                depends_on=("manuscript-variables",),
            ),
            Step(
                "figures",
                _controlled_python_argv("script", "scripts/generate_figures.py"),
                1800,
                depends_on=("manuscript-variables",),
            ),
            Step(
                "figure-registry",
                _controlled_python_argv("script", "scripts/generate_figure_registry.py"),
                300,
                depends_on=("figures",),
            ),
            Step(
                "figure-registry-check",
                _controlled_python_argv("script", "scripts/generate_figure_registry.py", "--check"),
                300,
                depends_on=("figure-registry",),
            ),
            Step(
                "cover",
                _controlled_python_argv("script", "scripts/generate_cover.py"),
                300,
                depends_on=("manuscript-variables",),
            ),
            Step(
                "cover-check",
                _controlled_python_argv("script", "scripts/generate_cover.py", "--check"),
                300,
                depends_on=("cover",),
            ),
            Step(
                "template-render",
                (
                    uv,
                    "run",
                    "--locked",
                    "python",
                    "scripts/pipeline/stage_03_render.py",
                    "--project",
                    TEMPLATE_PROJECT,
                ),
                7200,
                depends_on=(
                    "manuscript-variables",
                    "hydrated-docs",
                    "figure-registry-check",
                    "cover-check",
                ),
                cwd=str(resolved_template),
            ),
            Step(
                "publication-html-references",
                _controlled_python_argv(
                    "script", "scripts/z_render_docs.py", "--publication-references"
                ),
                300,
                depends_on=("template-render",),
            ),
            Step(
                "hydrated-docs-check",
                _controlled_python_argv(
                    "script", "scripts/z_render_docs.py", "--check", "--supporting-protocols"
                ),
                300,
                depends_on=("hydrated-docs", "publication-html-references"),
            ),
            Step(
                "manuscript-render-contract",
                _controlled_python_argv(
                    "module",
                    "pytest",
                    "tests/test_manuscript_tokens.py::test_real_resolved_tree_is_exactly_current",
                    "-q",
                    "--no-cov",
                ),
                600,
                depends_on=("template-render", "hydrated-docs-check"),
                forbid_pytest_skips=True,
            ),
            Step(
                "pdf-qpdf",
                (qpdf, "--check", str(COMBINED_PDF)),
                600,
                depends_on=("manuscript-render-contract",),
            ),
            Step(
                "pdf-integrity",
                _controlled_python_argv(
                    "module",
                    "pytest",
                    "tests/test_pdf_integrity.py",
                    "-q",
                    "--no-cov",
                ),
                600,
                depends_on=("pdf-qpdf",),
                forbid_pytest_skips=True,
            ),
            Step(
                "template-validate",
                (
                    uv,
                    "run",
                    "--locked",
                    "python",
                    "scripts/pipeline/stage_04_validate.py",
                    "--project",
                    TEMPLATE_PROJECT,
                ),
                1800,
                depends_on=("pdf-integrity", "hydrated-docs-check"),
                cwd=str(resolved_template),
            ),
        )
        if publication_derivative is None:
            return full_steps
        return _publication_derivative_steps(
            full_steps,
            publication_derivative,
            derivative_output or DEFAULT_RECEIPT.parent / "publication-derivative",
        )
    raise ValueError(f"unknown profile {profile!r}")


def _publication_derivative_steps(
    steps: Sequence[Step], contract_path: Path, output_directory: Path
) -> tuple[Step, ...]:
    """Replace the measurement checks only for an explicitly declared derivative."""
    modes = {"network-claim-bindings": "offline", "cache-audit": "live"}
    transformed: list[Step] = []
    for step in steps:
        if step.name == "release-infrastructure-coverage":
            step = replace(
                step,
                environment_overrides=(
                    *step.environment_overrides,
                    ("AILM_PUBLICATION_DERIVATIVE_CONTRACT", str(contract_path.absolute())),
                    (
                        "AILM_PUBLICATION_DERIVATIVE_LIVE_PROOF",
                        str((output_directory / "live.json").absolute()),
                    ),
                ),
            )
        mode = modes.get(step.name)
        if mode is None:
            transformed.append(step)
            continue
        transformed.append(
            replace(
                step,
                argv=_controlled_python_argv(
                    "script",
                    "scripts/verify_publication_derivative.py",
                    "--contract",
                    str(contract_path.absolute()),
                    "--mode",
                    mode,
                    "--output",
                    str((output_directory / f"{mode}.json").absolute()),
                ),
                timeout_seconds=3600,
            )
        )
    return tuple(_publication_no_bytecode_step(step) for step in transformed)


def _publication_no_bytecode_step(step: Step) -> Step:
    """Keep derivative validation from creating unreviewed rewritten code caches."""
    argv = step.argv
    if argv[:3] == (str(PROJECT_ROOT / ".venv/bin/python"), "-I", "-S") and argv[3:4] != ("-B",):
        argv = (*argv[:3], "-B", *argv[3:])
    return replace(
        step,
        argv=argv,
        environment_overrides=(*step.environment_overrides, ("PYTHONDONTWRITEBYTECODE", "1")),
    )


def _publication_contract(path: Path) -> DerivativeContract:
    """Load a reviewed contract after the stdlib-only release startup audit."""
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from ailm.publication_derivative import load_contract

    return load_contract(path, PROJECT_ROOT)


def _publication_state(contract: DerivativeContract) -> DerivativeState:
    """Capture immutable derivative inputs, separate from newly emitted evidence."""
    from ailm.publication_derivative import capture_derivative_state

    return capture_derivative_state(contract, PROJECT_ROOT)


def _publication_dependency_state(root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Rehash actual dependency bytes using the same audited producer bootstrap."""
    environment = _base_release_environment()
    for key in tuple(environment):
        if key == "OPENROUTER_API_KEY" or key.startswith("AILM_OPENROUTER_BUDGET_"):
            del environment[key]
    command = (
        str(root / ".venv/bin/python"),
        "-I",
        "-S",
        "-B",
        str(root / "scripts/_publication_execution.py"),
        "internal-dependencies",
    )
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
            close_fds=True,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError("actual dependency capture timed out") from error
    if completed.returncode:
        raise ValueError(
            f"actual dependency capture failed: exit={completed.returncode}; {completed.stderr}"
        )
    value: object = json.loads(completed.stdout)
    if not isinstance(value, dict) or value.get("integrity_ok") is not True:
        raise ValueError("actual dependency capture returned no authenticated inventory")
    return {str(key): item for key, item in value.items()}


def _publication_dependency_errors(path: Path, current: Mapping[str, object]) -> tuple[str, ...]:
    """Compare retained execution identities with freshly captured dependencies."""
    value: object = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        return ("derivative dependency evidence is not an object",)
    if current.get("integrity_ok") is not True or any(
        value.get(name) != current for name in ("dependency_before", "dependency_after")
    ):
        return ("actual current dependency bytes differ from retained proof",)
    return ()


def _publication_derivative_command(step: Step) -> bool:
    """Recognize proof prerequisites or their script target, not a typing input."""
    target = "scripts/verify_publication_derivative.py"
    if target not in step.argv:
        return False
    if step.name in {"network-claim-bindings", "cache-audit"}:
        return True
    marker = "--_ailm-controlled-python"
    if step.argv.count(marker) == 1:
        index = step.argv.index(marker)
        return step.argv[index + 1 : index + 3] == ("script", target)
    for argument in step.argv[1:]:
        if argument in {"-I", "-S", "-B"}:
            continue
        return argument == target
    return False


def _publication_step_evidence(step: Step) -> AttestationResult | None:
    """Require the new typed proof, rather than a derivative CLI's exit status."""
    if not _publication_derivative_command(step):
        return None
    values: dict[str, str] = {}
    for flag in ("--contract", "--mode", "--output"):
        if step.argv.count(flag) != 1:
            raise ValueError(f"derivative step has an ambiguous {flag} argument")
        index = step.argv.index(flag) + 1
        if index == len(step.argv):
            raise ValueError(f"derivative step is missing {flag}")
        values[flag] = step.argv[index]
    contract = _publication_contract(Path(values["--contract"]))
    expected_modes = {"network-claim-bindings": "offline", "cache-audit": "live"}
    if expected_modes.get(step.name) != values["--mode"]:
        raise ValueError("derivative step does not execute its required verification mode")
    from ailm.publication_derivative import validate_attestation

    result = validate_attestation(
        Path(values["--output"]), contract, _publication_state(contract), values["--mode"]
    )
    if result.accepted is not True:
        raise ValueError(f"derivative evidence refused: {result.errors}")
    path = Path(values["--output"])
    errors = _publication_dependency_errors(path, _publication_dependency_state())
    if errors:
        raise ValueError(f"derivative evidence refused: {errors}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != result.attestation_sha256:
        raise ValueError("derivative proof changed during dependency authentication")
    return result


def _publication_output_path(step: Step) -> Path | None:
    """Require a new per-run destination for freshly executed derivative evidence."""
    if not _publication_derivative_command(step):
        return None
    if step.argv.count("--output") != 1:
        raise OSError("derivative command must name exactly one evidence output")
    index = step.argv.index("--output") + 1
    if index == len(step.argv):
        raise OSError("derivative command is missing its evidence output")
    path = Path(step.argv[index])
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"derivative evidence must be newly emitted: {path}")
    return path


def _publication_results_errors(
    steps: Sequence[Step], results: Sequence[Mapping[str, object]]
) -> tuple[str, ...]:
    """Reauthenticate both derivative artifacts at finish, including their digests."""
    errors: list[str] = []
    for name in ("network-claim-bindings", "cache-audit"):
        matching_steps = [step for step in steps if step.name == name]
        matching_rows = [row for row in results if row["name"] == name]
        if len(matching_steps) != 1 or len(matching_rows) != 1:
            errors.append(f"missing or ambiguous derivative result: {name}")
            continue
        row = matching_rows[0]
        if row["status"] != "passed":
            errors.append(f"derivative prerequisite did not pass: {name}")
            continue
        try:
            evidence = _publication_step_evidence(matching_steps[0])
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(f"derivative finish authentication failed for {name}: {error}")
            continue
        if evidence is None or asdict(evidence) != row.get("publication_derivative"):
            errors.append(f"derivative evidence changed or is absent: {name}")
    return tuple(errors)


def _git_environment() -> dict[str, str]:
    """Return a non-interactive Git environment without caller repository routing.

    Repository identity is a release boundary.  Inherited ``GIT_DIR`` or
    ``GIT_WORK_TREE`` values can otherwise make a command launched from this
    checkout inspect an entirely different repository.  Global/system Git
    configuration is excluded as well so pager, diff, or fsmonitor helpers
    cannot replace a read-only snapshot operation.
    """

    environment = _base_release_environment()
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _git_bytes(*args: str, root: Path = PROJECT_ROOT) -> bytes:
    project = root.expanduser().resolve()
    git = _resolve_executable("git")
    if git is None:
        raise RuntimeError("git is unavailable in the fixed release search path")
    result = subprocess.run(
        [
            str(git),
            "-C",
            str(project),
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-c",
            "core.filemode=true",
            "-c",
            f"core.excludesFile={os.devnull}",
            *args,
        ],
        cwd=project,
        env=_git_environment(),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr}")
    return result.stdout


def _git(*args: str, root: Path = PROJECT_ROOT) -> str:
    return _git_bytes(*args, root=root).decode(errors="replace").strip()


def _network_binding_files_state(project: Path) -> dict[str, object]:
    """Fingerprint ignored network-claim receipts without following links."""

    directory = project / "cache" / "network_claim_bindings"
    state: dict[str, object] = {
        "schema": "ailm-network-claim-binding-file-state-v1",
        "path": str(directory.relative_to(project)),
        "exists": False,
        "path_kind": "missing",
        "files": [],
        "integrity_ok": True,
        "error": None,
    }
    try:
        directory_stat = directory.lstat()
        state["exists"] = True
        state["path_kind"] = (
            "symlink"
            if stat.S_ISLNK(directory_stat.st_mode)
            else "directory"
            if stat.S_ISDIR(directory_stat.st_mode)
            else "special"
        )
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise ValueError("network claim binding path is not a real directory")
        files: list[dict[str, object]] = []
        for path in sorted(directory.glob("*.json")):
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"network claim binding is not a real file: {path.name}")
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0),
            )
            try:
                opened = os.fstat(descriptor)
                if not stat.S_ISREG(opened.st_mode):
                    raise ValueError(f"opened network claim binding is not regular: {path.name}")
                if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                    raise ValueError(f"network claim binding changed while opening: {path.name}")
                content = bytearray()
                while chunk := os.read(descriptor, 1024 * 1024):
                    content.extend(chunk)
            finally:
                os.close(descriptor)
            files.append(
                {
                    "path": path.relative_to(project).as_posix(),
                    "device": opened.st_dev,
                    "inode": opened.st_ino,
                    "size_bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        state["files"] = files
    except FileNotFoundError:
        pass
    except (OSError, RuntimeError, ValueError) as error:
        state["integrity_ok"] = False
        state["error"] = str(error)
    encoded = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    state["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return state


def _call_cache_state(root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Fingerprint the ignored JSONL input without following cache symlinks."""

    project = root.expanduser().resolve()
    parent = project / "cache"
    path = parent / "bench_calls.jsonl"
    state: dict[str, object] = {
        "schema": "ailm-call-cache-file-state-v2",
        "path": str(path.relative_to(project)),
        "parent_path": str(parent.relative_to(project)),
        "parent_lstat_mode": None,
        "parent_kind": "missing",
        "parent_resolved_path": None,
        "exists": False,
        "lstat_mode": None,
        "path_kind": "missing",
        "symlink_target": None,
        "resolved_path": None,
        "device": None,
        "inode": None,
        "records": 0,
        "size_bytes": 0,
        "sha256": None,
        "integrity_ok": False,
        "error": None,
        "network_claim_bindings": _network_binding_files_state(project),
    }
    try:
        parent_stat = parent.lstat()
        state["parent_lstat_mode"] = parent_stat.st_mode
        state["parent_kind"] = (
            "symlink"
            if stat.S_ISLNK(parent_stat.st_mode)
            else "directory"
            if stat.S_ISDIR(parent_stat.st_mode)
            else "special"
        )
        state["parent_resolved_path"] = str(parent.resolve(strict=True))
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise ValueError("cache parent is not a real directory")
    except FileNotFoundError:
        # An absent cache and absent cache directory are safe for quick runs;
        # the full profile's cache-audit step independently requires coverage.
        state["integrity_ok"] = True
    except (OSError, RuntimeError, ValueError) as error:
        state["error"] = str(error)
    else:
        try:
            path_stat = path.lstat()
        except FileNotFoundError:
            state["integrity_ok"] = True
        except OSError as error:
            state["error"] = str(error)
        else:
            state["exists"] = True
            state["lstat_mode"] = path_stat.st_mode
            state["path_kind"] = (
                "symlink"
                if stat.S_ISLNK(path_stat.st_mode)
                else "file"
                if stat.S_ISREG(path_stat.st_mode)
                else "special"
            )
            state["symlink_target"] = os.readlink(path) if stat.S_ISLNK(path_stat.st_mode) else None
            try:
                state["resolved_path"] = str(path.resolve(strict=True))
                if not stat.S_ISREG(path_stat.st_mode):
                    raise ValueError("cache entry is not a real regular file")
                descriptor = os.open(
                    path,
                    os.O_RDONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NONBLOCK", 0),
                )
                try:
                    opened_stat = os.fstat(descriptor)
                    if not stat.S_ISREG(opened_stat.st_mode):
                        raise ValueError("opened cache entry is not a regular file")
                    if (opened_stat.st_dev, opened_stat.st_ino) != (
                        path_stat.st_dev,
                        path_stat.st_ino,
                    ):
                        raise ValueError("cache entry changed while its identity was captured")
                    content = bytearray()
                    while chunk := os.read(descriptor, 1024 * 1024):
                        content.extend(chunk)
                finally:
                    os.close(descriptor)
                state["device"] = opened_stat.st_dev
                state["inode"] = opened_stat.st_ino
                state["records"] = len(bytes(content).splitlines())
                state["size_bytes"] = len(content)
                state["sha256"] = hashlib.sha256(content).hexdigest()
                state["integrity_ok"] = True
            except (OSError, RuntimeError, ValueError) as error:
                state["error"] = str(error)

    binding_state = state["network_claim_bindings"]
    if isinstance(binding_state, dict) and not bool(binding_state.get("integrity_ok")):
        state["integrity_ok"] = False
        state["error"] = binding_state.get("error") or "network claim binding path is invalid"
    encoded = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    state["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return state


def _external_file_set_state(specifications: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Fingerprint exact external files, including path type and declared pin."""

    labels: set[str] = set()
    inputs: list[dict[str, object]] = []
    for specification in specifications:
        if set(specification) != {
            "label",
            "path",
            "expected_sha256",
            "expected_size_bytes",
        }:
            raise ValueError(f"invalid external-file specification: {specification!r}")
        label = specification["label"]
        raw_path = specification["path"]
        expected_sha256 = specification["expected_sha256"]
        expected_size = specification["expected_size_bytes"]
        if not isinstance(label, str) or not label or label in labels:
            raise ValueError(f"invalid or duplicate external-file label: {label!r}")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"external-file path for {label} must be a non-empty string")
        if (
            not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        ):
            raise ValueError(f"external-file digest for {label} is invalid")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size <= 0
        ):
            raise ValueError(f"external-file size for {label} is invalid")
        labels.add(label)
        path = Path(raw_path).expanduser().absolute()
        entry: dict[str, object] = {
            "label": label,
            "path": str(path),
            "expected_sha256": expected_sha256,
            "expected_size_bytes": expected_size,
        }
        try:
            path_stat = path.lstat()
            entry["exists"] = True
            entry["lstat_mode"] = path_stat.st_mode
            entry["path_kind"] = (
                "symlink"
                if stat.S_ISLNK(path_stat.st_mode)
                else "file"
                if stat.S_ISREG(path_stat.st_mode)
                else "special"
            )
            entry["symlink_target"] = os.readlink(path) if path.is_symlink() else None
            resolved = path.resolve(strict=True)
            entry["resolved_path"] = str(resolved)
            if not resolved.is_file():
                raise ValueError(f"resolved input is not a regular file: {resolved}")
            content = resolved.read_bytes()
            entry["size_bytes"] = len(content)
            entry["sha256"] = hashlib.sha256(content).hexdigest()
            entry["integrity_ok"] = (
                entry["size_bytes"] == expected_size and entry["sha256"] == expected_sha256
            )
            entry["error"] = None
        except (OSError, RuntimeError, ValueError) as error:
            entry.setdefault("exists", False)
            entry.setdefault("lstat_mode", None)
            entry.setdefault("path_kind", None)
            entry.setdefault("symlink_target", None)
            entry.setdefault("resolved_path", None)
            entry.setdefault("size_bytes", 0)
            entry.setdefault("sha256", None)
            entry["integrity_ok"] = False
            entry["error"] = str(error)
        inputs.append(entry)

    inputs.sort(key=lambda row: str(row["label"]))
    encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "ailm-external-file-set-state-v1",
        "inputs": inputs,
        "integrity_ok": bool(inputs) and all(bool(row["integrity_ok"]) for row in inputs),
        "fingerprint": hashlib.sha256(encoded).hexdigest(),
    }


def _raw_corpus_input_state(
    environment: Mapping[str, str], root: Path = PROJECT_ROOT
) -> dict[str, object]:
    """Resolve under the sanitized environment, then fingerprint every raw input."""

    project = root.resolve()
    locator = project / "scripts" / "list_raw_corpus_inputs.py"
    python = project / ".venv" / "bin" / "python"
    result = subprocess.run(
        [
            str(python),
            "-I",
            "-S",
            "-B",
            str(project / "scripts" / "verify_release.py"),
            CONTROLLED_RUNNER_MARKER,
            "script",
            str(locator.relative_to(project)),
            "--project-root",
            str(project),
        ],
        cwd=root,
        env=dict(environment),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        return {
            "schema": "ailm-raw-corpus-input-state-v1",
            "available": False,
            "integrity_ok": False,
            "fingerprint": None,
            "inputs": [],
            "locator_error": (result.stderr or result.stdout).strip()[-2000:],
        }
    try:
        inventory = json.loads(result.stdout)
        if not isinstance(inventory, dict) or inventory.get("schema") != (
            "ailm-raw-corpus-input-inventory-v1"
        ):
            raise ValueError("raw-corpus locator emitted an unsupported schema")
        specifications = inventory.get("inputs")
        if not isinstance(specifications, list):
            raise ValueError("raw-corpus locator did not emit an input list")
        file_state = _external_file_set_state(specifications)
    except (json.JSONDecodeError, ValueError) as error:
        return {
            "schema": "ailm-raw-corpus-input-state-v1",
            "available": False,
            "integrity_ok": False,
            "fingerprint": None,
            "inputs": [],
            "locator_error": str(error),
        }
    return {
        "schema": "ailm-raw-corpus-input-state-v1",
        "available": True,
        "integrity_ok": file_state["integrity_ok"],
        "fingerprint": file_state["fingerprint"],
        "inputs": file_state["inputs"],
        "locator_error": None,
    }


def _framed_digest(parts: Sequence[bytes]) -> str:
    """Hash byte strings without concatenation-boundary ambiguity."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, byteorder="big"))
        digest.update(part)
    return digest.hexdigest()


def _untracked_digest(root: Path, raw_paths: Sequence[bytes]) -> str:
    """Hash untracked names, types, modes, and contents in stable path order."""
    digest = hashlib.sha256()
    for raw_path in sorted(raw_paths):
        path = root / os.fsdecode(raw_path)
        stat_result = path.lstat()
        digest.update(len(raw_path).to_bytes(8, byteorder="big"))
        digest.update(raw_path)
        digest.update(stat_result.st_mode.to_bytes(8, byteorder="big"))
        if path.is_symlink():
            target = os.fsencode(os.readlink(path))
            digest.update(b"symlink")
            digest.update(len(target).to_bytes(8, byteorder="big"))
            digest.update(target)
        elif path.is_file():
            digest.update(b"file")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            # Git normally reports regular files and symlinks here. Keeping the
            # type marker makes a socket/FIFO replacement visible without ever
            # opening a special file and potentially blocking the release run.
            digest.update(b"special")
    return digest.hexdigest()


def _repository_state(root: Path = PROJECT_ROOT) -> RepositoryState:
    """Snapshot ``HEAD`` plus unstaged, staged, and untracked content."""
    head = _git("rev-parse", "HEAD", root=root)
    branch = _git("branch", "--show-current", root=root)
    status_bytes = _git_bytes("status", "--porcelain=v1", "-z", "--untracked-files=all", root=root)
    status = tuple(
        entry.decode(errors="surrogateescape") for entry in status_bytes.split(b"\0") if entry
    )
    unstaged = _git_bytes(
        "diff", "--no-ext-diff", "--no-textconv", "--binary", "--", ".", root=root
    )
    staged = _git_bytes(
        "diff",
        "--cached",
        "--no-ext-diff",
        "--no-textconv",
        "--binary",
        "--",
        ".",
        root=root,
    )
    untracked_paths = tuple(
        path
        for path in _git_bytes("ls-files", "--others", "--exclude-standard", "-z", root=root).split(
            b"\0"
        )
        if path
    )
    untracked_sha256 = _untracked_digest(root, untracked_paths)
    flag_rows = tuple(
        row
        for row in _git_bytes("ls-files", "-v", "-z", root=root).split(b"\0")
        if row and (row[:1] == b"S" or row[:1].islower())
    )
    index_flags = tuple(row.decode(errors="surrogateescape") for row in sorted(flag_rows))
    unstaged_sha256 = hashlib.sha256(unstaged).hexdigest()
    staged_sha256 = hashlib.sha256(staged).hexdigest()
    fingerprint = _framed_digest(
        (
            head.encode(),
            branch.encode(),
            status_bytes,
            unstaged,
            staged,
            untracked_sha256.encode(),
            b"\0".join(flag_rows),
        )
    )
    return RepositoryState(
        head=head,
        branch=branch,
        status=status,
        unstaged_sha256=unstaged_sha256,
        staged_sha256=staged_sha256,
        untracked_sha256=untracked_sha256,
        index_flags=index_flags,
        fingerprint=fingerprint,
    )


def _repository_boundary_refusal(state: RepositoryState, *, label: str) -> str | None:
    """Reject index flags that make Git-visible snapshots incomplete."""

    if state.index_flags:
        return f"{label} has assume-unchanged or skip-worktree index entries: " + ", ".join(
            state.index_flags
        )
    return None


def _audit_final_template_input(
    root: Path,
    initial_state: RepositoryState,
) -> tuple[RepositoryState | None, bool, bool, str | None]:
    """Recheck both the template Git state and its ignored sidecar route.

    The template checkout intentionally ignores ``projects/`` entries. A route
    can therefore be retargeted without changing the template Git fingerprint;
    publication acceptance must validate that non-Git input again after all
    template commands finish.
    """

    try:
        final_state = _repository_state(root)
    except RuntimeError as error:
        return None, False, False, f"could not snapshot template checkout at finish: {error}"

    repository_state_unchanged = initial_state.fingerprint == final_state.fingerprint
    try:
        _validate_template_root(root)
    except RuntimeError as error:
        return final_state, repository_state_unchanged, False, str(error)
    return final_state, repository_state_unchanged, True, None


#: Git pathspec naming the release inputs: every tracked path except the
#: regenerable ``output/`` tree. Mirrors ``INPUT_EPOCH_PATHSPEC`` in
#: ``scripts/z_generate_manuscript_variables.py`` (this runner cannot import it:
#: it starts with ``-I -S`` and no sidecar imports). Deriving the epoch from the
#: last input-touching commit reachable from the audited commit, rather than
#: from the audited commit itself, is what lets a render-only commit converge.
INPUT_EPOCH_PATHSPEC: tuple[str, ...] = (":/", ":(exclude,top)output")


def _commit_epoch(commit: str, root: Path = PROJECT_ROOT) -> str:
    """Return the committer timestamp of the last input-touching commit at ``commit``."""
    epoch = _git("log", "-1", "--format=%ct", commit, "--", *INPUT_EPOCH_PATHSPEC, root=root)
    if not epoch:
        raise RuntimeError(
            f"no commit reachable from {commit} touches a release input outside output/"
        )
    try:
        parsed = int(epoch)
    except ValueError as error:
        raise RuntimeError(f"git returned a non-integer commit timestamp: {epoch!r}") from error
    if parsed < 0:
        raise RuntimeError(f"git returned a negative commit timestamp: {parsed}")
    return str(parsed)


def _release_environment(
    commit: str,
    root: Path = PROJECT_ROOT,
    *,
    create_cache_dirs: bool = True,
    template_root: Path | None = None,
) -> tuple[dict[str, str], dict[str, object]]:
    """Build and describe the deterministic, project-writable step environment."""
    inherited_source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    inherited_force_source_date = os.environ.get("FORCE_SOURCE_DATE")
    environment = _base_release_environment()
    source_date_epoch = _commit_epoch(commit, root=root)
    release_uv_config = root.resolve() / RELEASE_UV_CONFIG.name
    if not release_uv_config.is_file():
        raise RuntimeError(f"reviewed release uv configuration is missing: {release_uv_config}")
    release_uv_config_sha256 = hashlib.sha256(release_uv_config.read_bytes()).hexdigest()

    # Dependency caches contain symlinks and non-repository .git sentinels.
    # Keep them outside publication outputs and inside the environment tree
    # excluded by project-source provenance scans.
    cache_root = root.resolve() / ".venv" / "release-cache"
    uv_cache_dir = cache_root / "uv"
    mpl_config_dir = cache_root / "matplotlib"
    if create_cache_dirs:
        uv_cache_dir.mkdir(parents=True, exist_ok=True)
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
    recorded: dict[str, object] = {
        "inherited_environment_policy": "allowlist-v1",
        "inherited_environment_keys": sorted(set(environment) - {"PATH"}),
        "AILM_OLLAMA_ENDPOINT": environment.get("AILM_OLLAMA_ENDPOINT"),
        "HOME": environment["HOME"],
        "home_origin": "login-account-database",
        "PATH": environment["PATH"],
        "path_origin": "fixed-search-path-v1",
        "SOURCE_DATE_EPOCH": source_date_epoch,
        "source_date_epoch_origin": "audited-input-commit",
        "FORCE_SOURCE_DATE": "1",
        "force_source_date_origin": "release-policy",
        "inherited_force_source_date_overridden": inherited_force_source_date is not None,
        "inherited_source_date_epoch_overridden": inherited_source_date_epoch is not None,
        "UV_CACHE_DIR": str(uv_cache_dir),
        "UV_CONFIG_FILE": str(release_uv_config),
        "uv_config_sha256": release_uv_config_sha256,
        "MPLCONFIGDIR": str(mpl_config_dir),
        "MPLBACKEND": "Agg",
    }
    environment.update(
        {
            "SOURCE_DATE_EPOCH": source_date_epoch,
            "FORCE_SOURCE_DATE": "1",
            "UV_CACHE_DIR": str(uv_cache_dir),
            "UV_CONFIG_FILE": str(release_uv_config),
            "MPLCONFIGDIR": str(mpl_config_dir),
            "MPLBACKEND": "Agg",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "TZ": "UTC",
        }
    )
    recorded.update(
        {
            "PYTHONHASHSEED": environment["PYTHONHASHSEED"],
            "PYTHONNOUSERSITE": environment["PYTHONNOUSERSITE"],
            "TZ": environment["TZ"],
        }
    )
    if template_root is not None:
        recorded["TEMPLATE_REPO_ROOT"] = str(template_root)
        environment["TEMPLATE_REPO_ROOT"] = str(template_root)
    return environment, recorded


def _dirty_input_refusal(
    profile: str, state: RepositoryState, *, development_allow_dirty_full: bool
) -> str | None:
    """Refuse dirty publication acceptance unless development explicitly opts in."""
    if profile == "full" and state.dirty and not development_allow_dirty_full:
        return (
            "full publication acceptance requires a clean tracked, staged, and untracked "
            "input state; use --development-allow-dirty-full only for a non-release "
            "development run"
        )
    return None


def _template_input_refusal(
    state: RepositoryState, *, development_allow_dirty_full: bool
) -> str | None:
    """Refuse a dirty template provider under the same publication policy."""
    if state.dirty and not development_allow_dirty_full:
        return (
            "full publication acceptance requires a clean template checkout because "
            "its renderer and validator are release inputs; use "
            "--development-allow-dirty-full only for a non-release development run"
        )
    return None


def _toolchain_refusal(profile: str, toolchain: Mapping[str, object]) -> str | None:
    """Explain which profile-required external executable is unusable."""

    required = ("git", "uv") if profile == "quick" else ("git", "uv", "lake", "qpdf")
    tools = toolchain.get("tools")
    if not isinstance(tools, Mapping):
        return "release toolchain inventory is malformed"
    failures: list[str] = []
    for name in required:
        row = tools.get(name)
        if not isinstance(row, Mapping) or row.get("usable") is not True:
            detail = row.get("error") if isinstance(row, Mapping) else "missing inventory row"
            failures.append(f"{name}: {detail}")
    if failures:
        return "external toolchain prerequisite failed: " + "; ".join(failures)
    return None


def _is_publication_acceptance(
    profile: str,
    *,
    successful: bool,
    development_allow_dirty_full: bool,
) -> bool:
    """Whether a successful run is eligible to serve as publication evidence."""

    return profile == "full" and successful and not development_allow_dirty_full


def _validate_step_graph(steps: Sequence[Step]) -> None:
    """Reject duplicate, missing, or forward dependencies before executing."""
    seen: set[str] = set()
    for step in steps:
        if step.name in seen:
            raise ValueError(f"duplicate release step name: {step.name}")
        unresolved = [dependency for dependency in step.depends_on if dependency not in seen]
        if unresolved:
            raise ValueError(
                f"release step {step.name!r} has missing or forward dependencies: "
                + ", ".join(unresolved)
            )
        override_names = [name for name, _value in step.environment_overrides]
        if len(set(override_names)) != len(override_names) or any(
            not name or "=" in name or "\x00" in name for name in override_names
        ):
            raise ValueError(f"release step {step.name!r} has invalid environment overrides")
        seen.add(step.name)


def _git_path_is_tracked_exact(relative: Path, *, root: Path) -> bool:
    """Whether Git tracks the exact lexical path rather than a descendant."""

    expected = os.fsencode(relative.as_posix())
    rows = _git_bytes("ls-files", "-z", "--", relative.as_posix(), root=root).split(b"\0")
    return expected in rows


def _atomic_temporary_name(final_name: str, token: str) -> str:
    """Return the one scratch-name grammar shared by writer and safety probes."""

    if re.fullmatch(rf"[0-9a-f]{{{ATOMIC_TEMP_HEX_LENGTH}}}", token) is None:
        raise ValueError("atomic scratch tokens must contain exactly 32 lowercase hex digits")
    return f".{final_name}.{token}.tmp"


def _receipt_log_directory(receipt_path: Path, run_token: str) -> Path:
    """Keep each execution's logs separate even when receipts share a parent."""
    if re.fullmatch(r"[0-9a-f]{32}", run_token) is None:
        raise ValueError("release run tokens must contain exactly 32 lowercase hex digits")
    return receipt_path.parent / "logs" / run_token


def _receipt_destination_is_safe(
    receipt_path: Path,
    root: Path = PROJECT_ROOT,
    *,
    step_names: Sequence[str] = ("release-step",),
    run_token: str = "0" * 32,
) -> bool:
    """Whether lexical receipt/log paths cannot alter Git-visible project state.

    The check deliberately precedes path resolution: a tracked symlink may
    resolve outside the checkout and look external even though replacing or
    traversing that lexical component changes a repository-owned path.
    """

    expanded = receipt_path.expanduser()
    if ".." in expanded.parts:
        return False
    receipt = expanded if expanded.is_absolute() else Path.cwd() / expanded
    receipt = receipt.absolute()
    project = root.expanduser().absolute()
    try:
        relative = receipt.relative_to(project)
    except ValueError:
        return True

    temporary_probes = tuple(
        relative.parent / _atomic_temporary_name(relative.name, character * ATOMIC_TEMP_HEX_LENGTH)
        for character in ("0", "f")
    )
    log_paths = tuple(
        _receipt_log_directory(relative, run_token) / f"{step_name}.log" for step_name in step_names
    )
    log_temporary_probes = tuple(
        log_path.parent / _atomic_temporary_name(log_path.name, character * ATOMIC_TEMP_HEX_LENGTH)
        for log_path in log_paths
        for character in ("0", "f")
    )
    probes = (
        relative,
        *temporary_probes,
        *log_paths,
        *log_temporary_probes,
    )
    for probe in probes:
        current = project
        for component in probe.parts:
            current /= component
            try:
                metadata = current.lstat()
            except FileNotFoundError:
                continue
            except OSError:
                return False
            if stat.S_ISLNK(metadata.st_mode):
                return False
            prefix = current.relative_to(project)
            if _git_path_is_tracked_exact(prefix, root=project):
                return False
        if _git_path_is_tracked_exact(probe, root=project):
            return False
        try:
            _git_bytes("check-ignore", "--quiet", "--", str(probe), root=project)
        except RuntimeError:
            return False
    return True


def _blocked_step_result(step: Step, failed_dependencies: Sequence[str]) -> dict[str, object]:
    """Create an explicit receipt row for a dependency-blocked step."""
    return {
        "name": step.name,
        "argv": list(step.argv),
        "timeout_seconds": step.timeout_seconds,
        "depends_on": list(step.depends_on),
        "cwd": step.cwd or str(PROJECT_ROOT),
        "environment_overrides": [list(item) for item in step.environment_overrides],
        "forbid_pytest_skips": step.forbid_pytest_skips,
        "status": "blocked",
        "blocked_by": list(failed_dependencies),
        "exit_code": None,
        "timed_out": False,
        "duration_seconds": 0.0,
        "log": None,
        "stdout_tail": "",
        "stderr_tail": "",
    }


def _distribution_artifacts_from_output(
    stdout: str, *, expected_source_date_epoch: str | None = None
) -> list[dict[str, object]]:
    """Parse and validate the package verifier's machine-readable artifact rows."""
    artifacts: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_line in stdout.splitlines():
        # Pytest progress output can share a line with the marker (carriage-
        # return overwrite), so locate the marker anywhere in the line.
        marker_index = raw_line.find(DISTRIBUTION_MARKER)
        if marker_index < 0:
            continue
        line = raw_line[marker_index:]
        try:
            row = json.loads(line.removeprefix(DISTRIBUTION_MARKER))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid distribution artifact JSON: {line!r}") from exc
        if not isinstance(row, dict) or set(row) != {
            "filename",
            "sha256",
            "bytes",
            "source_date_epoch",
        }:
            raise ValueError(f"invalid distribution artifact fields: {row!r}")
        filename = row["filename"]
        digest = row["sha256"]
        size = row["bytes"]
        source_date_epoch = row["source_date_epoch"]
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not (filename.endswith(".whl") or filename.endswith(".tar.gz"))
        ):
            raise ValueError(f"invalid distribution artifact filename: {filename!r}")
        if filename in seen:
            raise ValueError(f"duplicate distribution artifact row: {filename}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"invalid distribution artifact sha256: {digest!r}")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"invalid distribution artifact byte size: {size!r}")
        if (
            not isinstance(source_date_epoch, str)
            or not source_date_epoch.isdigit()
            or str(int(source_date_epoch)) != source_date_epoch
        ):
            raise ValueError(f"invalid distribution SOURCE_DATE_EPOCH: {source_date_epoch!r}")
        if (
            expected_source_date_epoch is not None
            and source_date_epoch != expected_source_date_epoch
        ):
            raise ValueError(
                "distribution artifact epoch disagrees with the release environment: "
                f"{source_date_epoch} != {expected_source_date_epoch}"
            )
        seen.add(filename)
        artifacts.append(
            {
                "filename": filename,
                "sha256": digest,
                "bytes": size,
                "source_date_epoch": source_date_epoch,
            }
        )

    suffixes = sorted(
        ".whl" if str(row["filename"]).endswith(".whl") else ".tar.gz" for row in artifacts
    )
    if suffixes != [".tar.gz", ".whl"]:
        raise ValueError(
            "package verifier must report exactly one wheel and one source distribution; "
            f"reported suffixes={suffixes}"
        )
    return artifacts


def _is_pytest_step(step: Step) -> bool:
    """Whether a release step invokes pytest through the audited interpreter."""
    legacy = any(
        step.argv[index : index + 2] == ("-m", "pytest")
        for index in range(max(0, len(step.argv) - 1))
    )
    controlled = any(
        step.argv[index : index + 3] == (CONTROLLED_RUNNER_MARKER, "module", "pytest")
        for index in range(max(0, len(step.argv) - 2))
    )
    return legacy or controlled


def _pytest_execution_evidence(
    step: Step, stdout: str, stderr: str
) -> tuple[int, int, float | None]:
    """Require executed passes, plus a coverage total when coverage is declared.

    Pytest exits successfully for ``--collect-only``. Release acceptance cannot
    trust the exit code alone, so every pytest receipt must contain a nonzero
    executed-pass summary. A coverage-bearing step must additionally expose its
    TOTAL row at or above the command's own ``--cov-fail-under`` threshold.
    """
    combined = f"{stdout}\n{stderr}"
    pass_counts = [int(match["count"]) for match in PYTEST_PASSED_RE.finditer(combined)]
    skip_counts = [int(match["count"]) for match in PYTEST_SKIPPED_RE.finditer(combined)]
    xfail_counts = [int(match["count"]) for match in PYTEST_XFAILED_RE.finditer(combined)]
    passed = max(pass_counts, default=0)
    skipped = max(skip_counts, default=0)
    xfailed = max(xfail_counts, default=0)
    if passed == 0:
        raise ValueError(
            f"pytest step {step.name!r} reported no executed passing tests; "
            "collection-only and all-skipped runs are not release evidence"
        )
    if step.forbid_pytest_skips and skipped:
        raise ValueError(
            f"pytest step {step.name!r} reported {skipped} skipped tests, but every selected "
            "test is mandatory release evidence"
        )
    if step.forbid_pytest_skips and xfailed:
        raise ValueError(
            f"pytest step {step.name!r} reported {xfailed} xfailed tests, but expected "
            "failures cannot replace mandatory release evidence; select --runxfail"
        )

    threshold_arguments = [
        argument.removeprefix("--cov-fail-under=")
        for argument in step.argv
        if argument.startswith("--cov-fail-under=")
    ]
    if not threshold_arguments:
        return passed, skipped, None
    if len(threshold_arguments) != 1:
        raise ValueError(f"pytest step {step.name!r} has ambiguous coverage thresholds")
    try:
        threshold = float(threshold_arguments[0])
    except ValueError as exc:
        raise ValueError(f"pytest step {step.name!r} has an invalid coverage threshold") from exc
    totals = [float(match["percent"]) for match in COVERAGE_TOTAL_RE.finditer(combined)]
    if len(totals) != 1:
        raise ValueError(
            f"pytest step {step.name!r} must report exactly one coverage TOTAL row; "
            f"found {len(totals)}"
        )
    if totals[0] < threshold:
        raise ValueError(
            f"pytest step {step.name!r} reported {totals[0]}% coverage below {threshold}%"
        )
    return passed, skipped, totals[0]


def _run_step(
    step: Step,
    logs_dir: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{step.name}.log"
    started = time.monotonic()
    timed_out = False
    stdout: str
    stderr: str
    working_directory = Path(step.cwd) if step.cwd is not None else PROJECT_ROOT
    try:
        _publication_output_path(step)
        step_environment = (
            dict(environment) if environment is not None else {**os.environ, "MPLBACKEND": "Agg"}
        )
        step_environment.update(dict(step.environment_overrides))
        result = subprocess.run(
            list(step.argv),
            cwd=working_directory,
            env=step_environment,
            capture_output=True,
            text=True,
            timeout=step.timeout_seconds,
            check=False,
        )
        exit_code = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = 124
        stdout = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else exc.stdout or ""
        )
        stderr = (
            exc.stderr.decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else exc.stderr or ""
        )
        stderr += f"\nTimed out after {step.timeout_seconds} seconds.\n"
    except OSError as exc:
        exit_code = 127
        stdout = ""
        stderr = f"Could not execute {step.argv[0]!r}: {exc}\n"

    distribution_artifacts: list[dict[str, object]] | None = None
    derivative_evidence: AttestationResult | None = None
    pytest_passed: int | None = None
    pytest_skipped: int | None = None
    coverage_percent: float | None = None
    if step.name == "package-distributions" and exit_code == 0:
        try:
            distribution_artifacts = _distribution_artifacts_from_output(
                stdout,
                expected_source_date_epoch=step_environment.get("SOURCE_DATE_EPOCH"),
            )
        except ValueError as exc:
            exit_code = 1
            stderr += f"\nPackage artifact receipt validation failed: {exc}\n"
    if _is_pytest_step(step) and exit_code == 0:
        try:
            pytest_passed, pytest_skipped, coverage_percent = _pytest_execution_evidence(
                step, stdout, stderr
            )
        except ValueError as exc:
            exit_code = 1
            stderr += f"\nPytest execution receipt validation failed: {exc}\n"
    if exit_code == 0:
        try:
            derivative_evidence = _publication_step_evidence(step)
        except (OSError, RuntimeError, ValueError) as exc:
            exit_code = 1
            stderr += f"\nPublication derivative proof validation failed: {exc}\n"

    duration = time.monotonic() - started
    combined = (
        f"$ {' '.join(step.argv)}\n"
        f"exit={exit_code} duration_seconds={duration:.3f} timed_out={timed_out}\n\n"
        f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    )
    _atomic_write_text(log_path, combined)
    try:
        receipt_log = str(log_path.relative_to(PROJECT_ROOT))
    except ValueError:
        receipt_log = str(log_path)
    receipt_row = {
        **asdict(step),
        "argv": list(step.argv),
        "depends_on": list(step.depends_on),
        "cwd": str(working_directory),
        "status": "passed" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_seconds": round(duration, 3),
        "log": receipt_log,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
    }
    if distribution_artifacts is not None:
        receipt_row["artifacts"] = distribution_artifacts
    if derivative_evidence is not None:
        receipt_row["publication_derivative"] = asdict(derivative_evidence)
    if pytest_passed is not None:
        receipt_row["pytest_passed"] = pytest_passed
    if pytest_skipped is not None:
        receipt_row["pytest_skipped"] = pytest_skipped
    if coverage_percent is not None:
        receipt_row["coverage_percent"] = coverage_percent
    return receipt_row


def _execute_steps(
    steps: Sequence[Step],
    logs_dir: Path,
    *,
    environment: Mapping[str, str],
    keep_going: bool,
) -> list[dict[str, object]]:
    """Execute a validated DAG, blocking unsafe descendants after failures."""
    _validate_step_graph(steps)
    results: list[dict[str, object]] = []
    statuses: dict[str, str] = {}
    for step in steps:
        blocked_by = [
            dependency for dependency in step.depends_on if statuses.get(dependency) != "passed"
        ]
        if blocked_by:
            result = _blocked_step_result(step, blocked_by)
            results.append(result)
            statuses[step.name] = "blocked"
            print(f"[blocked] {step.name} <- {', '.join(blocked_by)}", flush=True)
            continue

        result = _run_step(step, logs_dir, environment=environment)
        results.append(result)
        statuses[step.name] = str(result["status"])
        print(
            f"[{result['exit_code']}] {step.name} "
            f"({result['duration_seconds']}s) -> {result['log']}",
            flush=True,
        )
        if result["exit_code"] != 0 and not keep_going:
            break
    return results


def _atomic_write_text(path: Path, content: str) -> None:
    """Self-contained atomic writer used before project imports are trusted."""
    if not path.name:
        raise ValueError("atomic publication requires a file name")

    required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
    missing_flags = [name for name in required_flags if not hasattr(os, name)]
    if missing_flags:
        raise NotImplementedError(
            "secure release-receipt publication requires " + ", ".join(missing_flags)
        )
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW

    directory = path.parent.expanduser()
    parts = directory.parts
    if directory.is_absolute():
        parent_descriptor = os.open(directory.anchor, directory_flags)
        parts = parts[1:]
    else:
        parent_descriptor = os.open(".", directory_flags)
    try:
        for component in parts:
            if component in ("", "."):
                continue
            if component == "..":
                raise ValueError("release receipt paths must not contain parent traversal")
            try:
                child_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=parent_descriptor,
                )
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o755, dir_fd=parent_descriptor)
                    os.fsync(parent_descriptor)
                except FileExistsError:
                    pass
                child_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=parent_descriptor,
                )
            os.close(parent_descriptor)
            parent_descriptor = child_descriptor
    except BaseException:
        os.close(parent_descriptor)
        raise

    descriptor = -1
    temporary_name: str | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        for _attempt in range(128):
            temporary_name = _atomic_temporary_name(path.name, secrets.token_hex(16))
            try:
                descriptor = os.open(
                    temporary_name,
                    flags,
                    0o600,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:
                continue
            break
        else:
            raise FileExistsError(f"could not reserve a unique scratch beside {path.name!r}")

        try:
            metadata = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            mode = 0o644
        else:
            mode = stat.S_IMODE(metadata.st_mode) if stat.S_ISREG(metadata.st_mode) else 0o644
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            os.fchmod(handle.fileno(), mode)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        temporary_name = None
        os.fsync(parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)


def _write_receipt(receipt_path: Path, receipt: Mapping[str, object]) -> None:
    """Atomically publish one complete receipt after final state capture."""
    _atomic_write_text(receipt_path, json.dumps(receipt, indent=2) + "\n")


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def run_profile(
    profile: str,
    *,
    receipt_path: Path = DEFAULT_RECEIPT,
    keep_going: bool = False,
    development_allow_dirty_full: bool = False,
    publication_derivative: Path | None = None,
) -> dict[str, object]:
    """Run ``profile`` and write its complete receipt."""
    if development_allow_dirty_full and profile != "full":
        raise ValueError("--development-allow-dirty-full is valid only with the full profile")
    if publication_derivative is not None and profile != "full":
        raise ValueError("--publication-derivative is valid only with the full profile")
    startup_error = _isolated_startup_error()
    if startup_error is not None:
        raise RuntimeError(startup_error)
    expanded_receipt = receipt_path.expanduser()
    receipt_path = (
        expanded_receipt if expanded_receipt.is_absolute() else Path.cwd() / expanded_receipt
    ).absolute()
    started = _timestamp()
    run_token = secrets.token_hex(16)
    logs_dir = _receipt_log_directory(receipt_path, run_token)
    initial_state = _repository_state()
    initial_call_cache = _call_cache_state()
    interpreter_error = _project_python_error()
    initial_startup_state = _venv_startup_state()
    refusal: str | None = None
    if interpreter_error is not None:
        refusal = f"interpreter prerequisite failed: {interpreter_error}"
    if refusal is None and initial_startup_state["integrity_ok"] is not True:
        refusal = "virtual-environment startup prerequisite failed: " + "; ".join(
            _startup_errors(initial_startup_state)
        )
    if refusal is None:
        refusal = _repository_boundary_refusal(initial_state, label="sidecar checkout")
    if refusal is None:
        refusal = _dirty_input_refusal(
            profile,
            initial_state,
            development_allow_dirty_full=development_allow_dirty_full,
        )
    initial_toolchain = _toolchain_state(check_versions=refusal is None)
    if profile == "full" and refusal is None and not bool(initial_call_cache["integrity_ok"]):
        refusal = "call-cache publication prerequisite failed: " + str(
            initial_call_cache.get("error") or "cache path identity is invalid"
        )
    template_root: Path | None = None
    initial_template_state: RepositoryState | None = None
    template_resolution_error: str | None = None
    if profile == "full" and refusal is None:
        try:
            template_root = _resolve_template_root()
            initial_template_state = _repository_state(template_root)
        except RuntimeError as error:
            template_resolution_error = str(error)
            refusal = f"template publication prerequisite failed: {template_resolution_error}"
        else:
            refusal = _repository_boundary_refusal(
                initial_template_state,
                label="template checkout",
            )
            if refusal is None:
                refusal = _template_input_refusal(
                    initial_template_state,
                    development_allow_dirty_full=development_allow_dirty_full,
                )
    if refusal is None:
        refusal = _toolchain_refusal(profile, initial_toolchain)

    derivative_contract: DerivativeContract | None = None
    initial_derivative_state: DerivativeState | None = None
    initial_derivative_dependencies: dict[str, object] | None = None
    derivative_state_error: str | None = None
    if publication_derivative is not None and refusal is None:
        publication_derivative = publication_derivative.expanduser().absolute()
        try:
            derivative_contract = _publication_contract(publication_derivative)
            initial_derivative_state = _publication_state(derivative_contract)
            initial_derivative_dependencies = _publication_dependency_state()
        except (OSError, RuntimeError, ValueError) as error:
            derivative_state_error = str(error)
            refusal = f"publication derivative prerequisite failed: {error}"

    steps = (
        ()
        if refusal
        else build_steps(
            profile,
            template_root=template_root,
            toolchain=initial_toolchain,
            publication_derivative=publication_derivative,
            derivative_output=logs_dir / "publication-derivative",
        )
    )
    if not _receipt_destination_is_safe(
        receipt_path,
        step_names=tuple(step.name for step in steps),
        run_token=run_token,
    ):
        raise ValueError(
            "receipt and log destinations inside the project must be untracked and "
            f"gitignored: {receipt_path}"
        )

    environment, recorded_environment = _release_environment(
        initial_state.head,
        create_cache_dirs=refusal is None,
        template_root=template_root,
    )
    if publication_derivative is not None:
        environment.pop("OPENROUTER_API_KEY", None)
        recorded_environment["inherited_environment_keys"] = sorted(
            RELEASE_ENV_ALLOWLIST.intersection(environment)
        )
        recorded_environment["hosted_credentials_excluded"] = True
        if derivative_contract is not None:
            environment["AILM_OLLAMA_ENDPOINT"] = derivative_contract.endpoint
            recorded_environment["AILM_OLLAMA_ENDPOINT"] = derivative_contract.endpoint
            recorded_environment["ollama_endpoint_origin"] = "authenticated-original-receipts"
    raw_inputs_required = profile == "full"
    raw_inventory_captured = raw_inputs_required and refusal is None
    if raw_inventory_captured:
        initial_raw_corpus_inputs = _raw_corpus_input_state(environment)
    else:
        initial_raw_corpus_inputs = {
            "schema": "ailm-raw-corpus-input-state-v1",
            "available": False,
            "integrity_ok": False,
            "fingerprint": None,
            "inputs": [],
            "locator_error": (
                "not required by quick profile"
                if not raw_inputs_required
                else "not captured because a release prerequisite was refused"
            ),
        }
    results = _execute_steps(
        steps,
        logs_dir,
        environment=environment,
        keep_going=keep_going,
    )

    final_state = _repository_state()
    final_derivative_state: DerivativeState | None = None
    final_derivative_dependencies: dict[str, object] | None = None
    if derivative_contract is not None:
        try:
            final_derivative_state = _publication_state(derivative_contract)
            final_derivative_dependencies = _publication_dependency_state()
        except (OSError, RuntimeError, ValueError) as error:
            derivative_state_error = str(error)
    derivative_state_unchanged = publication_derivative is None or (
        initial_derivative_state is not None
        and final_derivative_state is not None
        and initial_derivative_state == final_derivative_state
        and derivative_state_error is None
    )
    derivative_result_errors = (
        _publication_results_errors(steps, results) if publication_derivative is not None else ()
    )
    derivative_results_valid = not derivative_result_errors
    derivative_dependencies_unchanged = publication_derivative is None or (
        initial_derivative_dependencies is not None
        and initial_derivative_dependencies == final_derivative_dependencies
    )
    final_call_cache = _call_cache_state()
    final_startup_state = _venv_startup_state()
    toolchain_rechecked_at_finish = refusal is None
    final_toolchain = (
        _toolchain_state(check_versions=True)
        if toolchain_rechecked_at_finish
        else initial_toolchain
    )
    final_raw_corpus_inputs = (
        _raw_corpus_input_state(environment)
        if raw_inventory_captured
        else initial_raw_corpus_inputs
    )
    final_template_state: RepositoryState | None = None
    template_state_unchanged = profile != "full"
    template_route_valid_at_finish: bool | None = None
    template_finish_error: str | None = None
    if initial_template_state is not None and template_root is not None:
        (
            final_template_state,
            template_state_unchanged,
            template_route_valid_at_finish,
            template_finish_error,
        ) = _audit_final_template_input(template_root, initial_template_state)
    repository_state_unchanged = initial_state.fingerprint == final_state.fingerprint
    startup_state_unchanged = initial_startup_state == final_startup_state
    startup_state_valid = bool(initial_startup_state["integrity_ok"]) and bool(
        final_startup_state["integrity_ok"]
    )
    toolchain_unchanged = (
        initial_toolchain == final_toolchain if toolchain_rechecked_at_finish else None
    )
    call_cache_unchanged = initial_call_cache == final_call_cache
    call_cache_valid = profile != "full" or (
        bool(initial_call_cache["integrity_ok"]) and bool(final_call_cache["integrity_ok"])
    )
    raw_corpus_inputs_unchanged = initial_raw_corpus_inputs == final_raw_corpus_inputs
    raw_corpus_inputs_valid = not raw_inputs_required or (
        bool(initial_raw_corpus_inputs["integrity_ok"])
        and bool(final_raw_corpus_inputs["integrity_ok"])
    )
    success = (
        refusal is None
        and len(results) == len(steps)
        and all(result["status"] == "passed" for result in results)
        and repository_state_unchanged
        and startup_state_unchanged
        and startup_state_valid
        and toolchain_unchanged is True
        and call_cache_unchanged
        and call_cache_valid
        and raw_corpus_inputs_unchanged
        and raw_corpus_inputs_valid
        and template_state_unchanged
        and (profile != "full" or template_route_valid_at_finish is True)
        and derivative_state_unchanged
        and derivative_results_valid
        and derivative_dependencies_unchanged
    )
    publication_acceptance = _is_publication_acceptance(
        profile,
        successful=success,
        development_allow_dirty_full=development_allow_dirty_full,
    )
    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "profile": profile,
        "run_id": run_token,
        "started_at": started,
        "finished_at": _timestamp(),
        "project_root": str(PROJECT_ROOT),
        "interpreter": sys.executable,
        "expected_interpreter": str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        "python_version": sys.version,
        "python_startup": {
            "required_flags": ["-I", "-S"],
            "isolated": bool(sys.flags.isolated),
            "no_site": bool(sys.flags.no_site),
            "state_unchanged": startup_state_unchanged,
            "integrity_ok": startup_state_valid,
            "initial_state": initial_startup_state,
            "final_state": final_startup_state,
        },
        "toolchain": {
            "rechecked_at_finish": toolchain_rechecked_at_finish,
            "state_unchanged": toolchain_unchanged,
            "initial_state": initial_toolchain,
            "final_state": final_toolchain,
        },
        "environment": recorded_environment,
        "git": {
            "commit": initial_state.head,
            "branch": initial_state.branch,
            "initial_dirty": initial_state.dirty,
            "final_dirty": final_state.dirty,
            "repository_state_unchanged": repository_state_unchanged,
            "initial_state": asdict(initial_state),
            "final_state": asdict(final_state),
        },
        "external_inputs": {
            "publication_derivative": {
                "required": publication_derivative is not None,
                "contract_path": (
                    str(publication_derivative) if publication_derivative is not None else None
                ),
                "state_unchanged": derivative_state_unchanged,
                "results_valid": derivative_results_valid,
                "dependencies_unchanged": derivative_dependencies_unchanged,
                "initial_dependencies": initial_derivative_dependencies,
                "final_dependencies": final_derivative_dependencies,
                "result_errors": derivative_result_errors,
                "error": derivative_state_error,
                "initial_state": (
                    asdict(initial_derivative_state)
                    if initial_derivative_state is not None
                    else None
                ),
                "final_state": (
                    asdict(final_derivative_state) if final_derivative_state is not None else None
                ),
            },
            "call_cache_unchanged": call_cache_unchanged,
            "call_cache_integrity_ok": call_cache_valid,
            "initial_call_cache": initial_call_cache,
            "final_call_cache": final_call_cache,
            "raw_corpora": {
                "required": raw_inputs_required,
                "integrity_ok": raw_corpus_inputs_valid,
                "state_unchanged": raw_corpus_inputs_unchanged,
                "initial_state": initial_raw_corpus_inputs,
                "final_state": final_raw_corpus_inputs,
            },
            "template": {
                "required": profile == "full",
                "root": str(template_root) if template_root is not None else None,
                "resolution_error": template_resolution_error,
                "route_valid_at_finish": template_route_valid_at_finish,
                "finish_validation_error": template_finish_error,
                "repository_state_unchanged": template_state_unchanged,
                "initial_state": (
                    asdict(initial_template_state) if initial_template_state is not None else None
                ),
                "final_state": (
                    asdict(final_template_state) if final_template_state is not None else None
                ),
            },
        },
        "development_allow_dirty_full": development_allow_dirty_full,
        "publication_acceptance": publication_acceptance,
        "refusal": refusal,
        "steps": results,
        "success": success,
    }
    _write_receipt(receipt_path, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("quick", "full"))
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument(
        "--publication-derivative",
        type=Path,
        help="full only: require a reviewed two-source publication derivative contract",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="run later steps after a failure so the receipt captures all independent gates",
    )
    parser.add_argument(
        "--development-allow-dirty-full",
        action="store_true",
        help=(
            "development only: allow the full profile to start from dirty tracked, "
            "staged, or untracked state; never use for publication acceptance"
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the command inventory without executing it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    startup_error = _isolated_startup_error()
    if startup_error is not None:
        print(f"release startup error: {startup_error}", file=sys.stderr)
        return 2
    args = _parser().parse_args(argv)
    if args.list:
        try:
            _project_python()
            startup_state = _venv_startup_state()
            if startup_state["integrity_ok"] is not True:
                raise RuntimeError(
                    "unsafe virtual-environment startup: "
                    + "; ".join(_startup_errors(startup_state))
                )
            toolchain = _toolchain_state(check_versions=True)
            template_root = _resolve_template_root() if args.profile == "full" else None
            toolchain_error = _toolchain_refusal(args.profile, toolchain)
            if toolchain_error is not None:
                raise RuntimeError(toolchain_error)
            steps = build_steps(
                args.profile,
                template_root=template_root,
                toolchain=toolchain,
                publication_derivative=args.publication_derivative,
            )
            _validate_step_graph(steps)
        except (RuntimeError, ValueError) as error:
            print(f"release inventory error: {error}", file=sys.stderr)
            return 2
        for step in steps:
            dependencies = ",".join(step.depends_on) or "-"
            working_directory = step.cwd or str(PROJECT_ROOT)
            print(
                f"{step.name}\t{step.timeout_seconds}\t{dependencies}\t"
                f"{working_directory}\t{' '.join(step.argv)}"
            )
        return 0
    try:
        receipt = run_profile(
            args.profile,
            receipt_path=args.receipt,
            keep_going=args.keep_going,
            development_allow_dirty_full=args.development_allow_dirty_full,
            publication_derivative=args.publication_derivative,
        )
    except (RuntimeError, ValueError) as error:
        print(f"release orchestration error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "receipt": str(args.receipt),
                "success": receipt["success"],
                "publication_acceptance": receipt["publication_acceptance"],
            }
        )
    )
    return 0 if receipt["success"] else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == CONTROLLED_RUNNER_MARKER:
        raise SystemExit(_run_controlled_python(sys.argv[2:]))
    raise SystemExit(main())
