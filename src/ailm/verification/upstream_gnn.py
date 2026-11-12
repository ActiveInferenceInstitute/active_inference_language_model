"""Offline custody of the released upstream parser used by interoperability tests.

The pin itself is read by ``gnn.importers``, the reader the pinned release ships;
:func:`load_importer_pin` imports that reader from the checkout and adapts its
normalized result. What remains here is custody: which checkout is selected, and
whether its HEAD, tag ref and every source byte still equal the pinned commit.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


class UpstreamReleaseError(ValueError):
    """The selected checkout does not supply the pinned release's source bytes."""


#: Where the conformance arm materializes the pinned release. The path carries
#: no tag, so a pin bump replaces one cache entry instead of leaving a family of
#: release directories behind, and no consumer has to read the pin to find it.
CHECKOUT_RELATIVE_PATH = "refs/gnn-upstream/checkout"


@dataclass(frozen=True)
class UpstreamRelease:
    repository: str
    tag: str
    commit: str


def load_importer_pin(path: Path, *, checkout_root: Path) -> UpstreamRelease:
    """Read the pin with the reader the pinned release ships.

    ``gnn.importers.read_pin`` is the estate's only pin reader: it decides which
    key holds the revision, which shapes are acceptable, and what the schema
    means. This function adds nothing to those decisions; it imports that reader
    from the checkout the pin names and converts its normalized result into the
    identity the custody check compares against Git objects.
    """

    source = (checkout_root / "src").resolve()
    if not source.is_dir():
        raise UpstreamReleaseError(f"{checkout_root} has no src directory to read the pin with")
    module_path = source / "gnn" / "importers.py"
    if not module_path.is_file():
        raise UpstreamReleaseError(
            f"the checkout at {checkout_root} does not ship gnn.importers: {module_path} is absent"
        )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    try:
        from gnn.importers import PinError, read_pin
    except ImportError as error:
        raise UpstreamReleaseError(
            f"the checkout at {checkout_root} does not ship a usable gnn.importers: {error}"
        ) from error
    reader = Path(sys.modules["gnn.importers"].__file__ or "").resolve()
    if source not in reader.parents:
        raise UpstreamReleaseError(
            f"the pin reader already imported from {reader} is not the one in {checkout_root}"
        )
    try:
        pin = read_pin(path)
    except PinError as error:
        raise UpstreamReleaseError(str(error)) from error
    if pin.tag is None:
        raise UpstreamReleaseError(f"{path}: the pin records no release tag to verify")
    return UpstreamRelease(
        repository=pin.repository or "",
        tag=pin.tag,
        commit=pin.commit,
    )


def checkout_head(root: Path) -> str:
    """Return the commit the selected checkout has checked out."""

    return _git(root, "rev-parse", "HEAD").decode().strip()


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
    environment.update(
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_NO_REPLACE_OBJECTS="1",
    )
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
    Interpreter __pycache__ directories are excluded from the source inventory;
    unversioned sourceless modules elsewhere are refused.
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
        if "__pycache__" in path.parts:
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
