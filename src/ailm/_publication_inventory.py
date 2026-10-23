"""Bounded publication derivative inventory responsibility."""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import io
import json
import marshal
import math
import os
import pwd
import re
import stat
import subprocess
import sys
import tempfile
import tomllib
from datetime import date, datetime, time
from pathlib import Path
from types import CodeType
from typing import Final, Mapping, TypeAlias, assert_never

from . import _publication_types as t
from ._publication_types import parse_hashes as parse_hashes
from ._publication_types import parse_inventory as parse_inventory

TOMLValue: TypeAlias = (
    str | bool | int | float | date | datetime | time | list["TOMLValue"] | dict[str, "TOMLValue"]
)

_PUBLICATION_SOURCE_ARCHIVE_SHA256: Final = (
    "e96d0ae70c7c4af0f5037a3144a2018313fcd4fe76cad90e3537196762d26129"
)
_PUBLICATION_SOURCE_KEYS: Final = frozenset(
    {
        "9f8842d93c9d3c463589fb6fae45087e7b2048a5:pyproject.toml",
        "9f8842d93c9d3c463589fb6fae45087e7b2048a5:src/ailm/__init__.py",
        "9f8842d93c9d3c463589fb6fae45087e7b2048a5:scripts/warm_call_cache.py",
        "6b1f24c7300a24d50e82a9b3b8a92f5c7d1bca8e:scripts/warm_call_cache.py",
    }
)


def publication_source_bytes(root: Path, commit: str, name: str) -> bytes:
    key = t.hash_string(commit, 40) + ":" + t.relative_path(name)
    archive = root / "data/publication_sources.json"
    if not os.path.lexists(archive):
        return git_bytes(root, "show", key)
    content = regular_bytes(archive)
    t.require(
        t.digest(content) == _PUBLICATION_SOURCE_ARCHIVE_SHA256,
        "portable publication source archive digest mismatch",
    )
    blobs = _publication_source_blobs(content)
    t.require(key in blobs, "source archive does not authorize requested ref/path")
    return blobs[key]


def _publication_source_blobs(content: bytes) -> Mapping[str, bytes]:
    fields = t.exact_keys(read_json_bytes(content), {"schema", "blobs"}, "source archive")
    t.require(fields["schema"] == "ailm-publication-source-blobs-v1", "source archive schema")
    blobs = t.json_map(fields["blobs"])
    t.require(set(blobs) == _PUBLICATION_SOURCE_KEYS, "source archive exact entry set differs")
    decoded: dict[str, bytes] = {}
    for identifier, value in blobs.items():
        row = t.exact_keys(t.json_map(value), {"bytes", "sha256", "base64"}, "source blob")
        try:
            data = base64.b64decode(t.json_text(row["base64"]), validate=True)
        except ValueError as error:
            raise t.DerivativeError("source blob base64 is invalid") from error
        t.require(len(data) == t.json_integer(row["bytes"]), "source blob byte extent differs")
        t.require(t.digest(data) == t.hash_string(row["sha256"]), "source blob digest differs")
        decoded[identifier] = data
    return decoded


def _toml_equal(original: TOMLValue, current: TOMLValue) -> bool:
    if type(original) is not type(current):
        return False
    match original:
        case dict():
            return (
                isinstance(current, dict)
                and original.keys() == current.keys()
                and all(_toml_equal(value, current[key]) for key, value in original.items())
            )
        case list():
            return (
                isinstance(current, list)
                and len(original) == len(current)
                and all(_toml_equal(a, b) for a, b in zip(original, current, strict=True))
            )
        case float():
            return original == current or (
                isinstance(current, float) and math.isnan(original) and math.isnan(current)
            )
        case str() | bool() | int() | datetime() | date() | time():
            return original == current
        case unreachable:
            assert_never(unreachable)


def validate_package_publication_urls(root: Path, measurement_commit: str) -> None:
    """Allow URL-only package metadata changes; preserve every other TOML field."""
    _validate_package_publication_bytes(
        root, git_bytes(root, "show", measurement_commit + ":pyproject.toml")
    )


def _validate_package_publication_bytes(root: Path, original_bytes: bytes) -> None:
    try:
        original = tomllib.loads(original_bytes.decode("utf-8"))
        current = tomllib.loads(regular_bytes(root / "pyproject.toml").decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as error:
        raise t.DerivativeError(
            "numerical/source/input package configuration is invalid"
        ) from error
    original_project, current_project = original.get("project"), current.get("project")
    if not isinstance(original_project, dict) or not isinstance(current_project, dict):
        raise t.DerivativeError("numerical/source/input package project table missing")
    original_project.pop("urls", None)
    current_project.pop("urls", None)
    t.require(
        _toml_equal(original, current),
        "numerical/source/input configuration changed outside publication URLs",
    )


def validate_publication_execution_binding(
    root: Path, recorded: dict[str, str], current: dict[str, str]
) -> None:
    """Authenticate the original package bytes before admitting URL-only equivalence."""
    declaration = read_json(root / "data/publication_derivative.json")
    commit = t.hash_string(declaration.get("measurement_commit"), 40)
    original = publication_source_bytes(root, commit, "pyproject.toml")
    t.require(
        recorded.get("pyproject.toml") == hashlib.sha256(original).hexdigest(),
        "recorded package hash differs from original measurement source",
    )
    t.require(
        set(recorded) == set(current)
        and all(recorded[name] == current[name] for name in current if name != "pyproject.toml"),
        "connected report execution inventory is incomplete or changed",
    )
    _validate_package_publication_bytes(root, original)
    t.require(
        current.get("pyproject.toml") == file_hash(root / "pyproject.toml"),
        "current package hash differs from execution inventory",
    )


def regular_bytes(path: Path) -> bytes:
    """Read a regular file without following any leaf or ancestor symlink."""
    target = path.absolute()
    for parent in (target, *target.parents):
        t.require(not parent.is_symlink(), f"symlink refused: {parent}")
    fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        t.require(stat.S_ISREG(before.st_mode), f"not regular: {path}")
        content = stream.read()
        after = os.fstat(stream.fileno())
    t.require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
        f"file changed during read: {path}",
    )
    t.require(path.lstat().st_ino == after.st_ino, f"file replaced during read: {path}")
    return content


def file_hash(path: Path) -> str:
    # Cache inputs are large; stream without accepting symlinks or read drift.
    target = path.absolute()
    for parent in (target, *target.parents):
        t.require(not parent.is_symlink(), f"symlink refused: {parent}")
    fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
    h = hashlib.sha256()
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        t.require(stat.S_ISREG(before.st_mode), f"not regular: {path}")
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
        after = os.fstat(stream.fileno())
    t.require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
        f"file changed during hash: {path}",
    )
    t.require(path.lstat().st_ino == after.st_ino, f"file replaced during hash: {path}")
    return h.hexdigest()


def file_identity(path: Path) -> t.FileIdentity:
    metadata = path.lstat()
    t.require(stat.S_ISREG(metadata.st_mode), f"not regular: {path}")
    return t.FileIdentity(
        file_hash(path), metadata.st_size, "100755" if metadata.st_mode & 0o111 else "100644"
    )


def read_json(path: Path) -> t.JSONMap:
    return read_json_bytes(regular_bytes(path))


def read_json_bytes(content: bytes) -> t.JSONMap:
    def unique(pairs: list[tuple[str, t.JSONValue]]) -> t.JSONMap:
        result: t.JSONMap = {}
        for key, value in pairs:
            t.require(key not in result, f"duplicate JSON field {key}")
            result[key] = value
        return result

    value = json.loads(
        content,
        object_pairs_hook=unique,
        parse_constant=lambda x: (_ for _ in ()).throw(t.DerivativeError(f"nonfinite JSON: {x}")),
    )
    return t.json_map(value)


def trusted_git(default_path: str = os.defpath) -> Path:
    """Select the system Git from the fixed default path, never from ``PATH``.

    Directories are tried in ``default_path`` order and the first one holding a
    ``git`` entry decides. A directory that is itself a symlink is admitted only
    when it resolves to another directory of the same default path — the
    merged-``/usr`` layout, where ``/bin`` is the operating system's alias for
    ``/usr/bin`` — and the executable is then taken from the resolved directory,
    so the returned path has no symlink component. An alias resolving anywhere
    else, and a ``git`` entry that is not a regular executable, are refused
    rather than skipped.
    """
    directories = [Path(entry).absolute() for entry in default_path.split(os.pathsep) if entry]
    admitted = {str(directory) for directory in directories}
    for directory in directories:
        if directory.is_symlink():
            resolved = directory.resolve()
            t.require(
                str(resolved) in admitted,
                f"symlink refused: {directory} resolves outside the default path",
            )
            directory = resolved
        path = directory / "git"
        try:
            metadata = path.lstat()
        except (FileNotFoundError, NotADirectoryError):
            continue
        t.require(
            stat.S_ISREG(metadata.st_mode) and bool(metadata.st_mode & 0o111),
            "system Git is not a regular executable",
        )
        file_hash(path)  # Reject ancestor/leaf symlinks and concurrent executable drift.
        return path
    raise t.DerivativeError("trusted system Git unavailable")


def git_command(
    root: Path, *args: str, input_bytes: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    tool = trusted_git()
    identity = file_identity(tool)
    environment = {
        "PATH": os.defpath,
        "HOME": pwd.getpwuid(os.getuid()).pw_dir,
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    # A full-tree batch can fill both pipes: feed requests from a regular file
    # so collecting Git's output never waits on a reciprocal stdin pipe write.
    with tempfile.TemporaryFile() as requests:
        if input_bytes is not None:
            requests.write(input_bytes)
            requests.seek(0)
        result = subprocess.run(
            [
                str(tool),
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.hooksPath=" + os.devnull,
                "-C",
                str(root),
                *args,
            ],
            stdin=requests if input_bytes is not None else None,
            env=environment,
            capture_output=True,
            check=False,
            close_fds=True,
        )
    t.require(file_identity(tool) == identity, "trusted Git executable changed during object read")
    t.require(
        result.returncode == 0,
        f"Git {' '.join(args)} failed: {result.stderr.decode(errors='replace')}",
    )
    return result


def git_bytes(root: Path, *args: str) -> bytes:
    return git_command(root, *args).stdout


def git_inventory(root: Path, commit: str) -> dict[str, t.FileIdentity]:
    t.hash_string(commit, 40)
    result: dict[str, t.FileIdentity] = {}
    # One batch process avoids a subprocess per blob, including large full trees.
    rows = git_bytes(root, "ls-tree", "-rz", "--full-tree", commit).split(b"\0")
    objects: list[tuple[str, str, str]] = []
    for row in rows:
        if not row:
            continue
        metadata, raw_path = row.split(b"\t", 1)
        mode, kind, oid = metadata.decode().split()
        path = t.relative_path(raw_path.decode())
        t.require(
            kind == "blob" and mode in {"100644", "100755"}, f"nonregular pinned Git entry: {path}"
        )
        objects.append((path, mode, oid))
    t.require(bool(objects), "empty pinned tree")
    proc = git_command(
        root, "cat-file", "--batch", input_bytes=("\n".join(x[2] for x in objects) + "\n").encode()
    )
    raw = proc.stdout
    offset = 0
    for path, mode, oid in objects:
        end = raw.index(b"\n", offset)
        head = raw[offset:end].decode().split()
        t.require(head[0] == oid and head[1] == "blob", "Git batch object mismatch")
        size = int(head[2])
        start = end + 1
        result[path] = t.FileIdentity(t.digest(raw[start : start + size]), size, mode)
        offset = start + size + 1
    t.require(offset == len(raw), "Git batch trailing bytes")
    return result


def scan_untracked_implementation(root: Path, tracked: set[str]) -> dict[str, t.FileIdentity]:
    result = {}
    for directory, directories, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in list(directories):
            link = base / name
            if link.is_symlink():
                relative_link = link.relative_to(root).as_posix()
                historical_figure_links = {
                    ".omo/evidence/publication-layout-beamer-20261006/figures",
                    ".omo/evidence/publication-crop-edge-20261006/figures",
                }
                t.require(
                    relative_link in historical_figure_links
                    and os.readlink(link) == str(root / "output/figures"),
                    f"untracked directory symlink: {link}",
                )
                directories.remove(name)  # Exact historical alias; target is scanned under output.
            elif link.relative_to(root).as_posix() in t.SCAN_EXCLUDED:
                directories.remove(name)  # Only these exact root environment/tool paths.

        for name in files:
            path = base / name
            relative = path.relative_to(root).as_posix()
            t.require(not path.is_symlink(), f"symlink refused: {path}")
            if relative in tracked or relative in t.SUPPORT_PATHS:
                continue
            if generated_bytecode_product(root, path, tracked | set(t.SUPPORT_PATHS)):
                continue
            if generated_lean_product(relative, tracked):
                t.require(
                    file_identity(path).mode == "100644", "generated Lean product executable mode"
                )
                continue
            if (
                relative.startswith(
                    ("src/", "scripts/", "vendor/", "gnn/", "formal/", "data/", "tests/")
                )
                or "__pycache__" in path.relative_to(root).parts
                or path.suffix in t.IMPLEMENTATION_SUFFIXES
                or (path.lstat().st_mode & 0o111)
            ):
                result[relative] = file_identity(path)
    return result


def generated_bytecode_product(root: Path, path: Path, sources: set[str]) -> bool:
    if path.parent.name != "__pycache__":
        return False
    match = re.fullmatch(
        r"(.+)\." + re.escape(str(sys.implementation.cache_tag)) + r"(?:\.opt-([12]))?\.pyc",
        path.name,
    )
    if match is None:
        return False
    source = path.parent.parent / (match[1] + ".py")
    if source.relative_to(root).as_posix() not in sources:
        return False
    t.require(file_identity(path).mode == "100644", "generated bytecode executable mode")
    content = regular_bytes(path)
    t.require(
        len(content) >= 16 and content[:4] == importlib.util.MAGIC_NUMBER,
        "invalid generated bytecode header",
    )
    try:
        t.require(int.from_bytes(content[4:8], "little") in {0, 1, 3}, "invalid bytecode flags")
        stream = io.BytesIO(content[16:])
        code = marshal.load(stream)
        t.require(not stream.read(), "generated bytecode trailing bytes")
        t.require(isinstance(code, CodeType), "invalid generated bytecode code")
        expected = compile(
            regular_bytes(source),
            code.co_filename,
            "exec",
            dont_inherit=True,
            optimize=int(match[2] or 0),
        )
        t.require(code == expected, "generated bytecode differs from source")
    except (EOFError, ValueError, TypeError, SyntaxError) as error:
        raise t.DerivativeError("invalid generated bytecode: " + str(error)) from error
    return True


def execution_snapshot_inventory(snapshot: Path) -> t.SnapshotInventory:
    result: t.SnapshotInventory = {}
    for directory, directories, files in os.walk(snapshot, followlinks=False):
        base = Path(directory)
        t.require(
            not base.is_symlink() and stat.S_IMODE(base.lstat().st_mode) == 0o555,
            "snapshot directory integrity changed",
        )
        for name in directories:
            t.require(not (base / name).is_symlink(), "snapshot directory symlink")
        for name in files:
            p = base / name
            metadata = p.lstat()
            t.require(stat.S_ISREG(metadata.st_mode), "snapshot nonregular file")
            result[p.relative_to(snapshot).as_posix()] = {
                "sha256": file_hash(p),
                "bytes": metadata.st_size,
                "permissions": f"{stat.S_IMODE(metadata.st_mode):04o}",
            }
    return result


def generated_lean_product(name: str, tracked: set[str]) -> bool:
    for prefix, suffixes in (
        ("formal/.lake/build/ir/", (".c.hash", ".setup.json", ".c")),
        (
            "formal/.lake/build/lib/lean/",
            (".ilean.hash", ".olean.hash", ".ilean", ".olean", ".trace"),
        ),
    ):
        if name.startswith(prefix):
            relative = name[len(prefix) :]
            for suffix in suffixes:
                if relative.endswith(suffix):
                    return "formal/" + relative[: -len(suffix)] + ".lean" in tracked
    return False


def original_literal(
    root: Path, contract: t.DerivativeContract, source: str, name: str
) -> t.JSONValue:
    content = git_bytes(root, "show", contract.measurement_commit + ":" + source)
    t.require(
        t.digest(content) == contract.original_inventory[source].sha256,
        "original source literal identity mismatch",
    )
    try:
        values = [
            node.value
            for node in ast.parse(content).body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        ]
        t.require(len(values) == 1, "original source literal missing/duplicate")
        return t.json_value(ast.literal_eval(values[0]))
    except (SyntaxError, ValueError, TypeError) as error:
        raise t.DerivativeError("original source literal parse failed: " + str(error)) from error
