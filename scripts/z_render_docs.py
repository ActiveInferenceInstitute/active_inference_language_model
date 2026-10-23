#!/usr/bin/env python3
"""Render the documentation tree with every ``claim:<label>`` resolved.

The source documents under ``docs/`` carry ledger references rather than
numerals, which is what keeps them from drifting — but a reference is not
readable. `claim:stage3_trie_nodes_v2000` tells a reader nothing about the
size of a trie.

So the sources stay tagged and this writes a hydrated mirror to
``output/docs/``, where every tag has become its recorded value. Same input,
two readings: the tagged tree is what is edited and audited, the hydrated tree
is what is read.

The alternative — typing the number and putting the tag beside it — is what the
project did before, and it is how decision 13's vocabulary table came to
disagree with all four of the measurements it summarized while looking entirely
correct.

Usage:
    .venv/bin/python scripts/z_render_docs.py
    .venv/bin/python scripts/z_render_docs.py --check   # exit 1 if stale
    .venv/bin/python scripts/z_render_docs.py --output-dir output/docs-probe
    .venv/bin/python scripts/z_render_docs.py --prune-retired --dry-run
    .venv/bin/python scripts/z_render_docs.py --prune-retired

``--output-dir`` is published through the atomic writer, which refuses any
symlinked path component; on macOS ``/tmp`` and ``/var`` are symlinks, so a
probe directory there must be given as ``os.path.realpath()`` of the
destination.

Every replaced publication leaves the previous tree in place as a hidden
retained sibling — ``.docs.generation.<token>`` after a native exchange,
``.docs.retired.<token>`` after the two-rename fallback — so a reader that
already resolved the old directory keeps a coherent view. Retention is a
bound, not immortality, and the tradeoff is explicit: a reader holding an
*open file* from a pruned generation keeps reading it (an unlinked inode
lives until its last descriptor closes), but a reader anchored to a pruned
generation — one that will open further files by path inside it — cannot
survive deletion. ``--prune-retired`` enforces the bound reviewably: it
keeps the newest ``--keep`` retained trees and deletes the rest, refusing
anything in the namespace that is not a publisher-shaped real directory.
``--min-age`` keeps siblings younger than the grace period out of every
prune, because such a sibling may be the staging tree of a concurrent
render, and ``--dry-run`` prints the inventory without deleting.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import math
import os
import re
import secrets
import shutil
import stat
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _artifact_io import _open_or_create_directory, atomic_write_text  # noqa: E402

from ailm.docs_hydration import hydrate  # noqa: E402
from ailm.reference_presentation import (  # noqa: E402
    PROTOCOL_SUPPORTING_HTML,
    _load_reference_variables,
)

#: Documents to hydrate, relative to the repository root.
SOURCES = ("docs", "README.md", "AGENTS.md", "CHANGELOG.md")

OUTPUT_DIR = PROJECT_ROOT / "output" / "docs"


class RetiredDocumentationError(ValueError):
    """Published Markdown has no counterpart in the current source inventory."""


@dataclass(frozen=True)
class _PublicationResult:
    """Outcome after the public directory has, or has not, been committed."""

    committed: bool
    reader_atomic: bool
    durability_error: str | None = None
    cleanup_error: str | None = None
    verification_error: str | None = None


def _markdown_sources() -> list[Path]:
    found: list[Path] = []
    for entry in SOURCES:
        path = PROJECT_ROOT / entry
        if path.is_dir():
            found.extend(sorted(path.rglob("*.md")))
        elif path.exists():
            found.append(path)
    return found


def _validate_output_dir(output_dir: Path) -> Path:
    """Return an absolute output path that cannot overwrite authored sources."""
    expanded = output_dir.expanduser()
    resolved = Path(os.path.abspath(expanded))
    source_roots = tuple((PROJECT_ROOT / entry).resolve() for entry in SOURCES)
    if resolved == PROJECT_ROOT.resolve() or any(
        source.is_dir() and (resolved == source or source in resolved.parents)
        for source in source_roots
    ):
        raise ValueError(
            f"output directory {resolved} overlaps authored documentation; "
            "choose output/docs or a separate temporary directory"
        )
    return resolved


def _generation_prefix(target: Path) -> str:
    return f".{target.name}.generation."


def _existing_publication_tree(
    target: Path,
) -> Path | None:
    """Return the real public tree; every link or non-directory is invalid."""
    try:
        metadata = target.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISDIR(metadata.st_mode):
        return target
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(
            f"unmanaged symlink at output tree {target}: the publication root must "
            "be a real directory managed by this renderer"
        )
    raise ValueError(f"output tree {target} must be a real directory, not a link or file")


def _expected_directories(expected_files: set[Path]) -> set[Path]:
    """Return every directory relative to the publication root."""
    directories = {Path(".")}
    for relative in expected_files:
        directories.update(relative.parents)
    return directories


def _validated_tree_inventory(root: Path, expected_files: set[Path]) -> set[Path]:
    """Validate a complete regular-file tree without following any symlink."""
    if not root.exists():
        return set()
    root_metadata = root.lstat()
    if not stat.S_ISDIR(root_metadata.st_mode) or stat.S_ISLNK(root_metadata.st_mode):
        reason = (
            "unmanaged symlink: the publication root must be managed by this renderer"
            if stat.S_ISLNK(root_metadata.st_mode)
            else "not a directory"
        )
        raise ValueError(
            f"output tree {root} must be a real directory ({reason}), not a link or file"
        )

    expected_directories = _expected_directories(expected_files)
    found: set[Path] = set()
    retired_documents: set[Path] = set()
    unexpected_directories: set[Path] = set()
    for candidate in sorted(root.rglob("*")):
        relative = candidate.relative_to(root)
        metadata = candidate.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"output tree contains a symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            if relative not in expected_directories:
                unexpected_directories.add(relative)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"output tree contains a special file: {relative}")
        if relative not in expected_files:
            if relative.parts[0] == "docs" and relative.suffix == ".md":
                retired_documents.add(relative)
                continue
            raise ValueError(f"output tree contains an unexpected file: {relative}")
        found.add(relative)
    # Obsolete Markdown can carry a directory retired along with its source.
    # Empty or unrelated directories remain structural errors, as before.
    unexplained = unexpected_directories - _expected_directories(retired_documents)
    if unexplained:
        raise ValueError(f"output tree contains an unexpected directory: {sorted(unexplained)[0]}")
    if retired_documents:
        raise RetiredDocumentationError(
            "stale hydrated Markdown has no current source; review these retained "
            "outputs before moving them out of the publication tree:\n  "
            + "\n  ".join(str(path) for path in sorted(retired_documents))
        )
    return found


def _fsync_tree(root: Path) -> None:
    """Synchronize directory entries throughout a fully staged tree."""
    directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        descriptor = os.open(
            directory,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _atomic_exchange(parent_descriptor: int, left: str, right: str) -> bool:
    """Atomically exchange two sibling names when the host filesystem can.

    Darwin exposes ``renameatx_np(RENAME_SWAP)`` and Linux exposes
    ``renameat2(RENAME_EXCHANGE)``.  Returning ``False`` means the platform or
    filesystem lacks an exchange primitive; other errors remain failures.
    """
    library = ctypes.CDLL(None, use_errno=True)
    function = None
    if sys.platform == "darwin" and hasattr(library, "renameatx_np"):
        function = library.renameatx_np
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        function = library.renameat2
    if function is None:
        return False

    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    result = function(
        parent_descriptor,
        os.fsencode(left),
        parent_descriptor,
        os.fsencode(right),
        0x2,  # RENAME_SWAP on Darwin; RENAME_EXCHANGE on Linux.
    )
    if result == 0:
        return True
    error_number = ctypes.get_errno()
    unsupported = {
        errno.EINVAL,
        errno.ENOSYS,
        errno.ENOTSUP,
        getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
    }
    if error_number in unsupported:
        return False
    raise OSError(error_number, os.strerror(error_number), f"{left} <-> {right}")


def _unique_sibling_name(parent_descriptor: int, prefix: str) -> str:
    """Return a currently absent unpredictable sibling basename."""
    for _attempt in range(128):
        candidate = f"{prefix}{secrets.token_hex(16)}"
        try:
            os.stat(candidate, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return candidate
    raise FileExistsError(f"could not reserve a unique name with prefix {prefix!r}")


def _publish_tree(
    staged: Path,
    target: Path,
    expected_files: set[Path],
) -> _PublicationResult:
    """Publish a fully staged real directory under the canonical name.

    A missing target is one atomic rename. Replacing a directory uses native
    rename exchange where available. The portable fallback performs two
    descriptor-anchored renames with rollback and reports its reader-visible
    interval. Old real directories are retained under hidden sibling names so
    readers holding them open remain valid.
    """
    if staged.parent != target.parent or not staged.name.startswith(_generation_prefix(target)):
        raise ValueError("staged hydrated tree is not a managed sibling generation")
    staged_files = _validated_tree_inventory(staged, expected_files)
    if staged_files != expected_files:
        missing = sorted(str(path) for path in expected_files - staged_files)
        raise RuntimeError(f"staged hydrated tree is incomplete: {missing}")

    staged_metadata = staged.lstat()
    staged_identity = (staged_metadata.st_dev, staged_metadata.st_ino)
    current = _existing_publication_tree(target)
    current_metadata = None if current is None else target.lstat()
    current_identity = (
        None
        if current_metadata is None
        else (current_metadata.st_dev, current_metadata.st_ino, current_metadata.st_mode)
    )
    parent_descriptor = _open_or_create_directory(target.parent)
    committed = False
    reader_atomic = True
    durability_error: str | None = None
    cleanup_error: str | None = None
    verification_error: str | None = None
    try:
        try:
            target_metadata = os.stat(target.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            target_metadata = None

        observed_identity = (
            None
            if target_metadata is None
            else (target_metadata.st_dev, target_metadata.st_ino, target_metadata.st_mode)
        )
        if observed_identity != current_identity:
            raise RuntimeError(
                "hydrated documentation publication changed while replacement staged"
            )

        if target_metadata is None:
            os.replace(
                staged.name,
                target.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            committed = True
        elif stat.S_ISDIR(target_metadata.st_mode):
            if _atomic_exchange(parent_descriptor, staged.name, target.name):
                committed = True
            else:
                reader_atomic = False
                backup_name = _unique_sibling_name(parent_descriptor, f".{target.name}.retired.")
                os.replace(
                    target.name,
                    backup_name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                )
                try:
                    os.replace(
                        staged.name,
                        target.name,
                        src_dir_fd=parent_descriptor,
                        dst_dir_fd=parent_descriptor,
                    )
                    committed = True
                except BaseException:
                    os.replace(
                        backup_name,
                        target.name,
                        src_dir_fd=parent_descriptor,
                        dst_dir_fd=parent_descriptor,
                    )
                    raise
        else:
            raise ValueError(f"output tree {target} is not a real directory")

        try:
            published_metadata = os.stat(
                target.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            verification_error = f"cannot authenticate committed public directory: {error}"
        else:
            if (
                not stat.S_ISDIR(published_metadata.st_mode)
                or (
                    published_metadata.st_dev,
                    published_metadata.st_ino,
                )
                != staged_identity
            ):
                verification_error = (
                    "published hydrated tree is not the authenticated staged directory"
                )

        try:
            os.fsync(parent_descriptor)
        except OSError as error:
            durability_error = str(error)

        return _PublicationResult(
            committed=committed,
            reader_atomic=reader_atomic,
            durability_error=durability_error,
            cleanup_error=cleanup_error,
            verification_error=verification_error,
        )
    finally:
        os.close(parent_descriptor)


def render(check_only: bool = False, *, output_dir: Path = OUTPUT_DIR) -> int:
    """Write (or verify) the hydrated documentation tree.

    Args:
        check_only: Compare against what is on disk instead of writing, and
            return non-zero if any file would change. This is the mode a gate
            runs in: a hydrated tree that is stale reports numbers from a
            previous ledger, which is the failure the whole mechanism exists to
            prevent.
        output_dir: Root of the hydrated mirror. An explicit directory lets
            clean-checkout tests generate and verify a disposable tree instead
            of depending on ignored artifacts left by an earlier local run.

    Returns:
        A process exit status.
    """
    try:
        output_dir = _validate_output_dir(output_dir)
        publication_tree = _existing_publication_tree(output_dir)
        sources = _markdown_sources()
        expected_files = {source.relative_to(PROJECT_ROOT) for source in sources}
        existing_files = (
            set()
            if publication_tree is None
            else _validated_tree_inventory(publication_tree, expected_files)
        )
    except RetiredDocumentationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1 if check_only else 2
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    ledger_path = PROJECT_ROOT / "data" / "claims.json"
    claims = json.loads(ledger_path.read_text(encoding="utf-8"))["claims"]

    try:
        reference_variables = _load_reference_variables(PROJECT_ROOT)
    except (ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    rendered_tree: dict[Path, str] = {}
    for source in sources:
        relative = source.relative_to(PROJECT_ROOT)
        try:
            rendered = hydrate(
                source.read_text(encoding="utf-8"),
                claims,
                strict=True,
                reference_variables=reference_variables,
            )
        except (KeyError, ValueError) as error:
            # Never write a document with an unresolved tag. It reads as a
            # fabricated result the same way an unresolved {{TOKEN}} does in the
            # rendered PDF, and it is the one failure that must be loud.
            print(f"ERROR: {relative}: {error}", file=sys.stderr)
            return 1
        rendered_tree[relative] = rendered

    stale = sorted(
        str(relative)
        for relative, rendered in rendered_tree.items()
        if relative not in existing_files
        or (output_dir / relative).read_text(encoding="utf-8") != rendered
    )
    if not stale and publication_tree is not None:
        # Preserve the directory and its inodes on an unchanged rerun. Replacing
        # an identical tree only creates another retained reader generation.
        print(f"All {len(rendered_tree)} hydrated documents are current.")
        return 0
    if check_only:
        print(
            f"{len(stale)} hydrated document(s) are stale; re-run "
            f"scripts/z_render_docs.py:\n  " + "\n  ".join(stale),
            file=sys.stderr,
        )
        return 1

    parent_descriptor = _open_or_create_directory(output_dir.parent)
    try:
        staged_name = _unique_sibling_name(parent_descriptor, _generation_prefix(output_dir))
        os.mkdir(staged_name, mode=0o755, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    staged = output_dir.parent / staged_name
    committed = False
    try:
        for relative, rendered in rendered_tree.items():
            atomic_write_text(staged / relative, rendered)
        staged_files = _validated_tree_inventory(staged, expected_files)
        if staged_files != expected_files:
            missing = sorted(str(path) for path in expected_files - staged_files)
            raise RuntimeError(f"staged hydrated tree is incomplete: {missing}")
        _fsync_tree(staged)
        publication = _publish_tree(staged, output_dir, expected_files)
        committed = publication.committed
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: hydrated-tree publication failed: {error}", file=sys.stderr)
        return 1
    finally:
        if not committed and staged.exists():
            shutil.rmtree(staged)

    if publication.durability_error is not None:
        print(
            "ERROR: hydrated-tree publication committed, but its parent directory "
            f"could not be synchronized: {publication.durability_error}",
            file=sys.stderr,
        )
        return 1
    if not publication.reader_atomic:
        print(
            "WARNING: hydrated-tree publication committed, but this filesystem has "
            "no native atomic directory exchange; replacing the prior publication "
            "contained a reader-visible rename interval",
            file=sys.stderr,
        )
    if publication.cleanup_error is not None:
        print(f"ERROR: {publication.cleanup_error}", file=sys.stderr)
        return 1
    if publication.verification_error is not None:
        print(f"ERROR: {publication.verification_error}", file=sys.stderr)
        return 1

    try:
        published_files = _validated_tree_inventory(output_dir, expected_files)
        if published_files != expected_files:
            missing = sorted(str(path) for path in expected_files - published_files)
            raise RuntimeError(f"published hydrated tree is incomplete: {missing}")
        drifted = [
            str(relative)
            for relative, rendered in rendered_tree.items()
            if (output_dir / relative).read_text(encoding="utf-8") != rendered
        ]
        if drifted:
            raise RuntimeError(f"published hydrated tree changed after commit: {drifted}")
    except (OSError, RuntimeError, ValueError) as error:
        print(
            f"ERROR: hydrated-tree post-publication verification failed: {error}", file=sys.stderr
        )
        return 1

    try:
        display_dir = output_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        display_dir = output_dir
    print(f"Rendered {len(rendered_tree)} document(s) into {display_dir}/")
    return 0


# ── Retained-generation retention ────────────────────────────────────────────


#: How many retired sibling generations ``--prune-retired`` keeps by default.
DEFAULT_RETAINED_GENERATIONS = 3
#: Siblings younger than this are presumed to be the in-flight staging tree of
#: a concurrent render and are never prune candidates, whatever the bound says.
DEFAULT_MIN_RETENTION_AGE_SECONDS = 3600.0


@dataclass(frozen=True)
class RetainedGeneration:
    """One entry of the managed sibling namespace beside the publication.

    ``action`` is the reviewable verdict: ``prune`` deletes the entry when the
    plan is applied, ``keep`` retains it, and ``refuse`` never touches it
    because it is not a publisher-shaped retained generation — a symlink, a
    non-directory, or a name the publisher would not have created.
    """

    name: str
    path: Path
    mtime: float
    mtime_ns: int
    size_bytes: int
    action: str
    reason: str
    identity: tuple[int, int]


@dataclass(frozen=True)
class RetiredGenerationCleanup:
    """The retained-generation inventory and, once applied, its outcome.

    ``entries`` is the plan; ``deleted`` and ``failed`` are filled by
    :func:`apply_retired_generation_cleanup` and stay empty for a dry run.
    """

    target: Path
    keep: int
    min_age_seconds: float
    entries: tuple[RetainedGeneration, ...]
    deleted: tuple[str, ...] = ()
    failed: tuple[tuple[str, str], ...] = ()

    @property
    def prune_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.entries if entry.action == "prune")


def _generation_prefixes(target_name: str) -> tuple[str, str]:
    """The managed sibling names: the exchange-retained and fallback-retired trees."""
    return (f".{target_name}.generation.", f".{target_name}.retired.")


def _classify_managed_name(name: str, target_name: str) -> str | None:
    """Classify one sibling name as ``managed``, ``lookalike``, or foreign.

    Every publisher-created sibling ends in the 32-hex-digit token that
    :func:`_unique_sibling_name` reserves; anything else wearing the prefix is
    a lookalike that cleanup must report instead of manage.
    """
    for prefix in _generation_prefixes(target_name):
        if not name.startswith(prefix):
            continue
        if re.fullmatch(rf"{re.escape(prefix)}[0-9a-f]{{32}}", name):
            return "managed"
        return "lookalike"
    return None


def _retained_tree_bytes(root: Path) -> int:
    """Sum one retained tree's regular-file bytes without following any link."""
    total = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as siblings:
            for sibling in siblings:
                metadata = sibling.stat(follow_symlinks=False)
                if stat.S_ISDIR(metadata.st_mode):
                    pending.append(Path(sibling.path))
                elif stat.S_ISREG(metadata.st_mode):
                    total += metadata.st_size
    return total


def plan_retired_generation_cleanup(
    output_dir: Path,
    *,
    keep: int = DEFAULT_RETAINED_GENERATIONS,
    min_age_seconds: float = DEFAULT_MIN_RETENTION_AGE_SECONDS,
    now: float | None = None,
) -> RetiredGenerationCleanup:
    """Inventory the retained generations beside ``output_dir`` and mark each one.

    This is the reviewable interface: the sibling namespace is read with
    ``lstat`` semantics only, the publication itself is reported as protected,
    and every deletion candidate is a publisher-shaped real directory that is
    both older than the grace period and beyond the ``keep`` bound.
    """
    if keep < 0:
        raise ValueError(f"retention bound must not be negative, got {keep}")
    if not math.isfinite(min_age_seconds):
        raise ValueError(f"retention grace period must be finite, got {min_age_seconds}")
    if min_age_seconds < 0:
        raise ValueError(f"retention grace period must not be negative, got {min_age_seconds}")
    current_time = time.time() if now is None else now
    if not math.isfinite(current_time):
        raise ValueError(f"retention clock must be finite, got {current_time}")
    target = _validate_output_dir(output_dir)
    if target.name.startswith("."):
        raise ValueError(
            f"output directory {target.name!r} sits inside the retained-generation "
            "namespace; prune the visible publication root instead"
        )
    publication = _existing_publication_tree(target)
    entries: list[RetainedGeneration] = []
    if publication is not None:
        metadata = publication.lstat()
        entries.append(
            RetainedGeneration(
                name=publication.name,
                path=publication,
                mtime=metadata.st_mtime,
                mtime_ns=metadata.st_mtime_ns,
                size_bytes=_retained_tree_bytes(publication),
                action="keep",
                reason="active publication; never a prune candidate",
                identity=(metadata.st_dev, metadata.st_ino),
            )
        )
    grace: list[RetainedGeneration] = []
    eligible: list[RetainedGeneration] = []
    refusals: list[RetainedGeneration] = []
    try:
        siblings = list(os.scandir(target.parent))
    except FileNotFoundError:
        siblings = []
    for sibling in siblings:
        verdict = _classify_managed_name(sibling.name, target.name)
        if verdict is None:
            continue
        metadata = sibling.stat(follow_symlinks=False)
        identity = (metadata.st_dev, metadata.st_ino)
        entry_path = target.parent / sibling.name
        if verdict == "lookalike" or not stat.S_ISDIR(metadata.st_mode):
            if stat.S_ISLNK(metadata.st_mode):
                reason = "symlink in the managed namespace; never followed or removed"
                size_bytes = 0
            elif not stat.S_ISDIR(metadata.st_mode):
                reason = "not a real directory"
                size_bytes = metadata.st_size if stat.S_ISREG(metadata.st_mode) else 0
            else:
                reason = "name is not a publisher generation (32-hex token required)"
                size_bytes = _retained_tree_bytes(entry_path)
            refusals.append(
                RetainedGeneration(
                    name=sibling.name,
                    path=entry_path,
                    mtime=metadata.st_mtime,
                    mtime_ns=metadata.st_mtime_ns,
                    size_bytes=size_bytes,
                    action="refuse",
                    reason=reason,
                    identity=identity,
                )
            )
            continue
        entry = RetainedGeneration(
            name=sibling.name,
            path=entry_path,
            mtime=metadata.st_mtime,
            mtime_ns=metadata.st_mtime_ns,
            size_bytes=_retained_tree_bytes(entry_path),
            action="keep",
            reason="younger than the in-flight grace period",
            identity=identity,
        )
        if current_time - metadata.st_mtime < min_age_seconds:
            grace.append(entry)
        else:
            eligible.append(entry)
    ordered = sorted(eligible, key=lambda entry: (entry.mtime, entry.name))
    prunable_count = max(len(ordered) - keep, 0)
    entries.extend(
        replace(entry, action="prune", reason="beyond the retention bound")
        for entry in ordered[:prunable_count]
    )
    entries.extend(
        replace(entry, reason="within the retention bound") for entry in ordered[prunable_count:]
    )
    entries.extend(grace)
    entries.extend(sorted(refusals, key=lambda entry: entry.name))
    return RetiredGenerationCleanup(
        target=target,
        keep=keep,
        min_age_seconds=min_age_seconds,
        entries=tuple(entries),
    )


def apply_retired_generation_cleanup(
    plan: RetiredGenerationCleanup, *, dry_run: bool = False
) -> RetiredGenerationCleanup:
    """Delete exactly the entries the plan marked ``prune``; touch nothing else.

    Each deletion re-authenticates the planned identity first: an entry that
    vanished or was replaced between planning and application is reported as
    failed with a fresh-plan requirement instead of being deleted blind.
    """
    if dry_run:
        return plan
    deleted: list[str] = []
    failed: list[tuple[str, str]] = []
    for entry in plan.entries:
        if entry.action != "prune":
            continue
        try:
            metadata = entry.path.lstat()
        except FileNotFoundError:
            failed.append(
                (entry.name, "vanished after planning; re-run --prune-retired for a fresh plan")
            )
            continue
        except OSError as error:
            failed.append((entry.name, str(error)))
            continue
        if (
            (metadata.st_dev, metadata.st_ino) != entry.identity
            or metadata.st_mtime_ns != entry.mtime_ns
            or not stat.S_ISDIR(metadata.st_mode)
        ):
            failed.append(
                (entry.name, "changed after planning; re-run --prune-retired for a fresh plan")
            )
            continue
        try:
            shutil.rmtree(entry.path)
        except OSError as error:
            failed.append((entry.name, f"deletion failed: {error}"))
            continue
        deleted.append(entry.name)
    return replace(plan, deleted=tuple(deleted), failed=tuple(failed))


def _human_bytes(count: int) -> str:
    """Render a byte count for the reviewable inventory."""
    size = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0:
            return f"{count} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PiB"


def _print_retention_inventory(plan: RetiredGenerationCleanup, *, dry_run: bool) -> None:
    mode = "dry-run inventory" if dry_run else "cleanup"
    print(
        f"Retained documentation generations beside {plan.target} "
        f"(keep={plan.keep}, min_age={plan.min_age_seconds:g}s), {mode}:"
    )
    for entry in plan.entries:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.mtime))
        print(
            f"  {entry.action.upper():<6} {entry.name}  "
            f"{_human_bytes(entry.size_bytes):>10}  {stamp}  {entry.reason}"
        )
    if dry_run:
        prunable = [entry for entry in plan.entries if entry.action == "prune"]
        print(
            f"dry-run: {len(prunable)} generation(s), {_human_bytes(plan.prune_bytes)} "
            "reclaimable; nothing deleted"
        )


def prune_retired(
    *,
    output_dir: Path,
    keep: int = DEFAULT_RETAINED_GENERATIONS,
    min_age_seconds: float = DEFAULT_MIN_RETENTION_AGE_SECONDS,
    dry_run: bool = False,
) -> int:
    """Bound the retained sibling generations through the reviewable interface.

    Returns ``0`` when the requested work completed (a dry run always does),
    ``1`` when any deletion failed, and ``2`` when the output path itself is
    invalid. Refused namespace entries are reported on stderr and left for
    manual review; they never block pruning the entries the plan did mark.
    """
    try:
        plan = plan_retired_generation_cleanup(
            output_dir, keep=keep, min_age_seconds=min_age_seconds
        )
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    _print_retention_inventory(plan, dry_run=dry_run)
    if dry_run:
        return 0
    applied = apply_retired_generation_cleanup(plan, dry_run=False)
    for name, reason in applied.failed:
        print(f"ERROR: {name}: {reason}", file=sys.stderr)
    for entry in plan.entries:
        if entry.action == "refuse":
            print(
                f"WARNING: left for manual review, {entry.name}: {entry.reason}",
                file=sys.stderr,
            )
    reclaimed = sum(
        entry.size_bytes for entry in plan.entries if entry.name in set(applied.deleted)
    )
    print(
        f"pruned {len(applied.deleted)} retained generation(s), "
        f"{_human_bytes(reclaimed)} reclaimed, {len(applied.failed)} failure(s)"
    )
    return 1 if applied.failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the hydrated tree matches the sources and the ledger, without writing",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="hydrated mirror root (default: output/docs)",
    )
    parser.add_argument(
        "--supporting-protocols",
        action="store_true",
        help="also generate/check the portable protocol bundle beside PDF and web consumers",
    )
    parser.add_argument(
        "--publication-references",
        action="store_true",
        help="after manuscript rendering, resolve individual HTML references from the "
        "combined index; with --check, reject stale or invalid HTML without writing",
    )
    parser.add_argument(
        "--web-dir",
        type=Path,
        default=None,
        help="with --publication-references, the rendered web edition (default: output/web)",
    )
    parser.add_argument(
        "--prune-retired",
        action="store_true",
        help="bound the retained sibling generations: keep the newest --keep, "
        "delete the rest; the active publication is never a candidate",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=None,
        metavar="N",
        help=f"with --prune-retired, how many retained generations to keep "
        f"(default: {DEFAULT_RETAINED_GENERATIONS})",
    )
    parser.add_argument(
        "--min-age",
        type=float,
        default=None,
        metavar="SECONDS",
        help="with --prune-retired, siblings younger than this are presumed "
        f"in-flight staging trees and are kept (default: {DEFAULT_MIN_RETENTION_AGE_SECONDS:g})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="with --prune-retired, print the inventory and delete nothing",
    )
    args = parser.parse_args(argv)
    if args.publication_references:
        if args.prune_retired or args.supporting_protocols or args.output_dir != OUTPUT_DIR:
            parser.error("--publication-references is a separate post-render phase")
        if args.keep is not None or args.min_age is not None or args.dry_run:
            parser.error("retention options cannot be combined with --publication-references")
        from ailm.supporting_package import (
            publication_reference_repairs,
            validate_publication_html,
        )

        web = args.web_dir or PROJECT_ROOT / "output" / "web"
        try:
            repairs = publication_reference_repairs(web)
            if args.check and repairs:
                raise ValueError(
                    "unresolved publication references in "
                    + ", ".join(page.name for page in repairs)
                )
            for page, content in repairs.items():
                atomic_write_text(page, content)
            if publication_reference_repairs(web):
                raise ValueError("publication references changed during post-render verification")
            validate_publication_html(web)
            print(f"Publication HTML references verified; repaired {len(repairs)} page(s).")
            return 0
        except (OSError, ValueError) as error:
            print(f"ERROR: publication HTML references: {error}", file=sys.stderr)
            return 1
    if args.web_dir is not None:
        parser.error("--web-dir requires --publication-references")
    if args.prune_retired:
        if args.check or args.supporting_protocols:
            parser.error(
                "--prune-retired cannot be combined with publication checks or supporting HTML"
            )
        return prune_retired(
            output_dir=args.output_dir,
            keep=DEFAULT_RETAINED_GENERATIONS if args.keep is None else args.keep,
            min_age_seconds=(
                DEFAULT_MIN_RETENTION_AGE_SECONDS if args.min_age is None else args.min_age
            ),
            dry_run=args.dry_run,
        )
    if args.keep is not None or args.min_age is not None or args.dry_run:
        parser.error("--keep, --min-age, and --dry-run require --prune-retired")
    status = render(check_only=args.check, output_dir=args.output_dir)
    if status or not args.supporting_protocols:
        return status
    # The full release explicitly requests this artifact; the stdlib-only
    # documentation mirror remains usable without the external template provider.
    from z_generate_manuscript_variables import render_supporting_protocol_html

    from ailm.supporting_package import validate_publication_html

    try:
        supporting_dir = Path(PROTOCOL_SUPPORTING_HTML).parent
        status = render_supporting_protocol_html(
            args.output_dir / "docs" / "extension_protocols.md",
            PROJECT_ROOT / "output" / "pdf" / supporting_dir,
            check_only=args.check,
            additional_output_dirs=(PROJECT_ROOT / "output" / "web" / supporting_dir,),
        )
        if status:
            return status
        # Generation checks its freshly produced companions. The complete web
        # edition is checked after Stage 03 and the global-reference repair.
        if args.check:
            from ailm.supporting_package import publication_reference_repairs

            if publication_reference_repairs(PROJECT_ROOT / "output" / "web"):
                raise ValueError("unresolved publication references after manuscript render")
            validate_publication_html(PROJECT_ROOT / "output" / "web")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: supporting protocol HTML: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
