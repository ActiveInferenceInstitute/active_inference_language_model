#!/usr/bin/env python3
"""Refuse a commit whose figure provenance no longer matches its source.

The v3 figure receipt at ``output/figures/figure_provenance.json`` records the
SHA-256 of every Python module under ``src/ailm``. Editing any of those modules
invalidates the receipt even though nothing in the receipt's own directory
changed, so the mismatch is invisible to the person making the edit and usually
surfaces only in a release-profile run.

This check reads the receipt and the source blobs **from one commit**, recomputes
each digest, and refuses on the first disagreement. It reads the commit tree
rather than the working tree because the object being pushed is the object that
must be verified: a dirty checkout must neither mask a stale receipt nor
manufacture a failure the commit does not have.

Usage
-----
    .venv/bin/python -I -S scripts/verify_figure_provenance.py --commit HEAD
    .venv/bin/python -I -S scripts/verify_figure_provenance.py --commit origin/main

Install the hook once per checkout:

    git config core.hooksPath githooks

Exit status is 0 when the inventory and every digest match the commit, and 1 when
the receipt, the inventory, or a digest disagrees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECEIPT_PATH = "output/figures/figure_provenance.json"
SOURCE_ROOT = "src/ailm"
SCHEMA = "ailm-figure-provenance-v3"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ProvenanceCheckError(RuntimeError):
    """The commit, the receipt, or the recorded dependencies cannot be trusted."""


def _git(root: Path, *arguments: str, binary: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=not binary,
        check=False,
    )


def _resolve_commit(root: Path, ref: str) -> str:
    result = _git(root, "rev-parse", "--verify", f"{ref}^{{commit}}")
    if result.returncode != 0:
        raise ProvenanceCheckError(
            f"{ref}: not a readable commit ({result.stderr.strip() or 'git refused'})"
        )
    return result.stdout.strip()


def _committed_dependencies(root: Path, commit: str) -> list[tuple[str, str]]:
    """Return ``(path, blob)`` for every Python module under ``src/ailm``."""
    result = _git(root, "ls-tree", "-r", "-z", commit, "--", SOURCE_ROOT, binary=True)
    if result.returncode != 0:
        raise ProvenanceCheckError(
            f"{commit}: cannot list {SOURCE_ROOT} ({result.stderr.decode().strip()})"
        )
    entries: list[tuple[str, str]] = []
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        metadata, path = record.split(b"\t", 1)
        mode, _kind, blob = metadata.decode().split(" ")
        if not path.endswith(b".py"):
            continue
        if mode == "120000":
            raise ProvenanceCheckError(
                f"{path.decode()}: code dependency is a symlink; the receipt pins regular files"
            )
        entries.append((path.decode(), blob))
    entries.sort()
    return entries


def _recorded_dependencies(root: Path, commit: str) -> list[dict[str, str]]:
    result = _git(root, "show", f"{commit}:{RECEIPT_PATH}")
    if result.returncode != 0:
        raise ProvenanceCheckError(
            f"{commit}: no {RECEIPT_PATH} in the commit; the figure receipt must be committed "
            "beside the code it authenticates"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ProvenanceCheckError(f"{commit}:{RECEIPT_PATH} is not valid JSON: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ProvenanceCheckError(f"{commit}:{RECEIPT_PATH} does not carry schema {SCHEMA!r}")
    records = payload.get("code_dependencies")
    if not isinstance(records, list) or any(
        not isinstance(record, dict)
        or set(record) != {"path", "sha256"}
        or not isinstance(record.get("path"), str)
        or not _SHA256.match(str(record.get("sha256")))
        for record in records
    ):
        raise ProvenanceCheckError(f"{commit}:{RECEIPT_PATH} has malformed code-dependency records")
    return records


def _blob_digests(root: Path, blobs: list[str]) -> list[str]:
    """Return the SHA-256 of each blob, read through one ``cat-file --batch`` stream.

    One process rather than one per file: the inventory is the whole of
    ``src/ailm`` and a pre-push hook that spawns a subprocess per module is slow
    enough to be disabled.
    """
    request = "".join(f"{blob}\n" for blob in blobs).encode()
    result = subprocess.run(
        ["git", "-C", str(root), "cat-file", "--batch"],
        input=request,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ProvenanceCheckError(
            f"git cat-file --batch failed: {result.stderr.decode().strip() or 'git refused'}"
        )
    data = result.stdout
    digests: list[str] = []
    cursor = 0
    for blob in blobs:
        header_end = data.find(b"\n", cursor)
        if header_end < 0:
            raise ProvenanceCheckError(f"blob {blob}: cannot be read from the object store")
        header = data[cursor:header_end].decode(errors="replace").split(" ")
        if len(header) != 3 or header[1] != "blob":
            raise ProvenanceCheckError(f"blob {blob}: cannot be read from the object store")
        start = header_end + 1
        size = int(header[2])
        digests.append(hashlib.sha256(data[start : start + size]).hexdigest())
        cursor = start + size + 1
    return digests


def check(root: Path, ref: str) -> tuple[str, int]:
    """Return the resolved commit and dependency count, or raise on a mismatch."""
    commit = _resolve_commit(root, ref)
    committed = _committed_dependencies(root, commit)
    recorded = _recorded_dependencies(root, commit)

    recorded_paths = [record["path"] for record in recorded]
    committed_paths = [path for path, _blob in committed]
    if recorded_paths != committed_paths:
        added = sorted(set(committed_paths) - set(recorded_paths))
        removed = sorted(set(recorded_paths) - set(committed_paths))
        detail = []
        if added:
            detail.append(f"not recorded: {added[:3]}{'...' if len(added) > 3 else ''}")
        if removed:
            detail.append(f"recorded but absent: {removed[:3]}{'...' if len(removed) > 3 else ''}")
        raise ProvenanceCheckError(
            f"{commit}: AILM code dependency inventory changed since the receipt was written "
            f"({'; '.join(detail) or 'order differs'}); run a full figure build"
        )

    digests = _blob_digests(root, [blob for _path, blob in committed])
    for record, (path, _blob), digest in zip(recorded, committed, digests, strict=True):
        if digest != record["sha256"]:
            raise ProvenanceCheckError(
                f"{commit}: {path}: AILM code dependency digest changed since the receipt was "
                f"written (receipt {record['sha256'][:12]}..., commit {digest[:12]}...); "
                "run a full figure build"
            )
    return commit, len(recorded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commit",
        default="HEAD",
        help="commit to verify (default: HEAD); the receipt and sources are read from it",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="repository root containing the commit (default: this script's repository)",
    )
    arguments = parser.parse_args(argv)
    try:
        commit, count = check(arguments.project_root, arguments.commit)
    except ProvenanceCheckError as error:
        print(f"figure provenance check failed: {error}", file=sys.stderr)
        return 1
    print(f"{count} AILM code dependencies match {RECEIPT_PATH} at {commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
