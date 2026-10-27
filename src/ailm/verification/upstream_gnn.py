"""Offline custody of the released upstream parser used by interoperability tests."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path


class UpstreamReleaseError(ValueError):
    """The selected checkout does not supply the pinned release's source bytes."""


@dataclass(frozen=True)
class UpstreamRelease:
    repository: str
    tag: str
    commit: str


def load_release_pin(path: Path) -> UpstreamRelease:
    """Read the authored release identity, refusing ambiguous or malformed pins."""

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise UpstreamReleaseError(f"duplicate upstream pin field: {key}")
            result[key] = value
        return result

    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UpstreamReleaseError(f"cannot read upstream release pin: {error}") from error
    if not isinstance(data, dict) or set(data) != {"repository", "tag", "commit"}:
        raise UpstreamReleaseError("upstream pin must name repository, tag and commit only")
    if not all(isinstance(value, str) for value in data.values()):
        raise UpstreamReleaseError("upstream pin fields must be strings")
    if not data["repository"].startswith("https://github.com/"):
        raise UpstreamReleaseError("upstream repository must be an HTTPS GitHub URL")
    if not re.fullmatch(r"v\d+\.\d+\.\d+", data["tag"]):
        raise UpstreamReleaseError("upstream tag must name a stable version")
    if not re.fullmatch(r"[0-9a-f]{40}", data["commit"]):
        raise UpstreamReleaseError("upstream commit must be a complete Git object identity")
    return UpstreamRelease(**data)


def upstream_candidates(project: Path) -> tuple[Path, ...]:
    """Durable conventional checkouts, independent of a user's home directory."""
    namespace = (
        (project.parents[2] / "ActiveInferenceInstitute/Public/Generalized_Notation_Notation",)
        if len(project.parents) > 2
        else ()
    )
    return (*namespace, project.parent / "Generalized_Notation_Notation")


def default_upstream_root(candidates: tuple[Path, ...]) -> Path:
    """Choose a checkout supplying a parser; its identity is checked before import."""
    for candidate in candidates:
        if (candidate / "src/gnn/parsers/markdown_parser.py").is_file():
            return candidate
    return candidates[0]


def selected_upstream_root(project: Path) -> Path:
    """An explicit selection is final, including refusal of an empty value."""
    if "AILM_GNN_UPSTREAM" in os.environ:
        value = os.environ["AILM_GNN_UPSTREAM"].strip()
        if not value:
            raise UpstreamReleaseError("AILM_GNN_UPSTREAM must not be empty")
        return Path(value).expanduser().resolve()
    return default_upstream_root(upstream_candidates(project)).resolve()


def _git(root: Path, *arguments: str) -> bytes:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    try:
        result = subprocess.run(
            [
                "git",
                "--no-optional-locks",
                "-c",
                "core.fsmonitor=false",
                "-C",
                str(root),
                *arguments,
            ],
            env=environment,
            capture_output=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        detail = (
            error.stderr.decode(errors="replace")
            if isinstance(error, subprocess.CalledProcessError)
            else str(error)
        )
        raise UpstreamReleaseError(f"cannot inspect upstream checkout: {detail.strip()}") from error
    return result.stdout


def verify_release_checkout(root: Path, pin: UpstreamRelease) -> dict[str, str | int]:
    """Verify HEAD, tag and every source file against Git objects before import.

    Reading actual files catches changes hidden by assume-unchanged or
    skip-worktree flags. Unexpected source files and symlinked paths also fail.
    Interpreter bytecode caches are excluded from the source inventory.
    """
    root = root.resolve()
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    if commit != pin.commit:
        raise UpstreamReleaseError(
            f"upstream HEAD {commit} differs from pinned {pin.tag} ({pin.commit}); "
            "select its isolated release checkout with AILM_GNN_UPSTREAM"
        )
    tag = _git(root, "rev-parse", f"refs/tags/{pin.tag}^{{commit}}").decode().strip()
    if tag != pin.commit:
        raise UpstreamReleaseError(f"upstream tag {pin.tag} differs from its pinned commit")
    expected: dict[str, str] = {}
    for record in _git(root, "ls-tree", "-r", "-z", "HEAD", "--", "src").split(b"\0"):
        if not record:
            continue
        header, raw_path = record.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        relative = raw_path.decode("utf-8")
        if mode not in {"100644", "100755"} or kind != "blob":
            raise UpstreamReleaseError(f"nonregular upstream source: {relative}")
        expected[relative] = oid
    if "src/gnn/parsers/markdown_parser.py" not in expected:
        raise UpstreamReleaseError("upstream release has no Markdown parser")
    if (root / "src").is_symlink():
        raise UpstreamReleaseError("symlinked upstream source: src")
    actual: set[str] = set()
    for path in (root / "src").rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise UpstreamReleaseError(f"symlinked upstream source: {relative}")
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if not path.is_dir():
            actual.add(relative)
    if actual != expected.keys():
        raise UpstreamReleaseError(
            f"upstream source inventory differs: missing={sorted(expected.keys() - actual)}, "
            f"unexpected={sorted(actual - expected.keys())}"
        )
    digests: dict[str, str] = {}
    for relative, oid in sorted(expected.items()):
        path = root / relative
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise UpstreamReleaseError(f"nonregular upstream source: {relative}")
        content = path.read_bytes()
        after = path.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise UpstreamReleaseError(f"upstream source changed while reading: {relative}")
        blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if blob != oid:
            raise UpstreamReleaseError(f"modified upstream source: {relative}")
        digests[relative] = hashlib.sha256(content).hexdigest()
    return {
        "repository": pin.repository,
        "tag": pin.tag,
        "commit": commit,
        "tree": _git(root, "rev-parse", "HEAD^{tree}").decode().strip(),
        "source_files": len(digests),
        "source_sha256": hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest(),
        "checkout": str(root),
    }
