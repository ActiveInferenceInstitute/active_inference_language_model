"""Canonical file inventory for connected GUM execution provenance.

This module inspects bytes only. It neither imports the model nor executes a
recorder, so publication readers can share the inventory without reversing the
package-to-script dependency boundary.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import stat
from pathlib import Path


class ExecutionBindingError(ValueError):
    """A canonical execution input is missing, unsafe or changed during reading."""


_FIXED_INPUTS = (
    "scripts/record_connected_gum_claims.py",
    "scripts/_artifact_io.py",
    "pyproject.toml",
    "uv.lock",
    "src/ailm/__init__.py",
    "src/ailm/cases.py",
    "src/ailm/config.py",
    "src/ailm/resources.py",
    "src/ailm/spec_agent.py",
    "src/ailm/statistics.py",
    "src/ailm/claims.py",
    "src/ailm/ledger_lock.py",
    "src/ailm/verification/__init__.py",
    "src/ailm/verification/oracle.py",
    "src/ailm/verification/connected_provenance.py",
    "data/case_word/ud_ewt_case_word_counts.json",
    "data/widened/ud_ewt_clause_widened_counts.json",
)
_DISCOVERED_INPUTS = (
    ("src/ailm/levels", "*.py"),
    ("src/ailm/corpus", "*.py"),
    ("src/ailm/gnn", "*.py"),
    ("gnn/models", "*.gnn.md"),
    ("vendor", "*.json"),
)


def _directory(root_fd: int, parts: tuple[str, ...]) -> int:
    descriptor = os.dup(root_fd)
    try:
        for part in parts:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError:
        os.close(descriptor)
        raise


def _file_digest(root_fd: int, relative: str) -> str:
    parts = Path(relative).parts
    parent = _directory(root_fd, parts[:-1])
    try:
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    finally:
        os.close(parent)
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        os.close(descriptor)
        raise ExecutionBindingError(f"execution input is not a regular file: {relative}")
    with os.fdopen(descriptor, "rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
        after = os.fstat(source.fileno())

        def signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )

        if signature(before) != signature(after):
            raise ExecutionBindingError(f"execution input changed while reading: {relative}")
        return digest


def canonical_execution_binding(root: Path) -> dict[str, str]:
    """Hash the complete canonical inventory, refusing symlinked or nonregular inputs.

    Directory-relative opens keep all traversals beneath the opened project root.
    Every directory and file is opened without following symbolic links. The
    existing model/input discovery patterns remain exhaustive, and this helper's
    package import dependencies are explicit members of the same binding.
    """
    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise ExecutionBindingError(f"invalid execution root: {root}") from error
    try:
        paths = list(_FIXED_INPUTS)
        for directory, pattern in _DISCOVERED_INPUTS:
            descriptor = _directory(root_fd, Path(directory).parts)
            try:
                names = sorted(
                    name for name in os.listdir(descriptor) if fnmatch.fnmatchcase(name, pattern)
                )
            finally:
                os.close(descriptor)
            if not names:
                raise ExecutionBindingError(f"empty execution input inventory: {directory}")
            paths.extend(f"{directory}/{name}" for name in names)
        if len(set(paths)) != len(paths):
            raise ExecutionBindingError("duplicated execution input inventory")
        return {relative: _file_digest(root_fd, relative) for relative in sorted(paths)}
    except OSError as error:
        raise ExecutionBindingError(
            f"execution inventory input is unavailable or unsafe: {error}"
        ) from error
    finally:
        os.close(root_fd)
