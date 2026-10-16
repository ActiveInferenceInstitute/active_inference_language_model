"""Stdlib-first bounded publication snapshot helpers."""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ailm._publication_types import (
        DerivativeContract,
        FileIdentityData,
        Materialization,
        SnapshotInventory,
    )


def snapshot_inventory(root: Path) -> SnapshotInventory:
    from ailm.publication_derivative import execution_snapshot_inventory

    return execution_snapshot_inventory(root)


def materialize(
    contract: DerivativeContract, project_root: Path, destination: Path
) -> Materialization:
    from ailm.publication_derivative import file_identity, git_bytes, git_command, require

    destination.mkdir()  # Never reuse or overwrite an old source snapshot.
    objects = []
    for row in git_bytes(
        project_root, "ls-tree", "-rz", "--full-tree", contract.measurement_commit
    ).split(b"\0"):
        if row:
            metadata, path = row.split(b"\t", 1)
            mode, kind, oid = metadata.decode().split()
            require(
                kind == "blob" and mode in {"100644", "100755"}, "nonregular Git snapshot entry"
            )
            objects.append((path.decode(), mode, oid))
    proc = git_command(
        project_root,
        "cat-file",
        "--batch",
        input_bytes=("\n".join(x[2] for x in objects) + "\n").encode(),
    )
    offset = 0
    for name, mode, oid in objects:
        end = proc.stdout.index(b"\n", offset)
        head = proc.stdout[offset:end].decode().split()
        require(head[:2] == [oid, "blob"], "Git snapshot batch object mismatch")
        size = int(head[2])
        offset = end + 1
        p = destination / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("xb") as stream:
            stream.write(proc.stdout[offset : offset + size])
        p.chmod(0o755 if mode == "100755" else 0o644)
        offset += size + 1
    pre_overlay: dict[str, FileIdentityData] = {
        name: {"sha256": identity.sha256, "bytes": identity.bytes, "mode": identity.mode}
        for name in contract.original_inventory
        for identity in (file_identity(destination / name),)
    }
    require(
        pre_overlay == {p: asdict(i) for p, i in contract.original_inventory.items()},
        "pre-overlay pinned tree mismatch",
    )
    for name, expected in contract.inputs.items():
        if name in contract.original_inventory:
            require(file_identity(destination / name).sha256 == expected, "tracked input mismatch")
            continue
        p = destination / name
        p.parent.mkdir(parents=True, exist_ok=True)
        require(not p.exists(), "overlay replaces tracked source")
        shutil.copyfile(project_root / name, p, follow_symlinks=False)
        require(file_identity(p).sha256 == expected, f"overlay hash mismatch: {name}")
    for directory, _directories, files in os.walk(destination, topdown=False):
        for name in files:
            p = Path(directory) / name
            p.chmod(0o555 if p.stat().st_mode & 0o111 else 0o444)
        Path(directory).chmod(0o555)
    before = snapshot_inventory(destination)
    require(
        set(before) == set(contract.original_inventory) | set(contract.inputs),
        "snapshot extra/missing path",
    )
    return {"snapshot": str(destination), "pre_overlay": pre_overlay, "before": before, "after": {}}


def mutation_tree(contract: DerivativeContract, root: Path, destination: Path) -> Path:
    """Real temporary current tree, including inputs; never touches canonical files."""
    from ailm.publication_derivative import file_identity, git_bytes, require

    destination.mkdir()
    names = (
        {x.decode() for x in git_bytes(root, "ls-files", "-z").split(b"\0") if x}
        | set(contract.baseline_untracked)
        | set(contract.inputs)
        | set(contract.worker_evidence)
        | {x for x in contract.support if (root / x).exists()}
    )
    for name in sorted(names):
        p = root / name
        identity = file_identity(p)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target, follow_symlinks=False)
        target.chmod(0o755 if identity.mode == "100755" else 0o644)
    # Read-only Git queries use the shared object database/index; no Git mutation.
    (destination / ".git").write_text("gitdir: " + str(root / ".git") + "\n")
    candidates = [
        p
        for p in contract.original_inventory
        if p.startswith("src/ailm/levels/") and p.endswith(".py")
    ]
    require(bool(candidates), "no numerical source for negative control")
    target = destination / sorted(candidates)[0]
    with target.open("ab") as stream:
        stream.write(b"\n# numerical-byte refusal control\n")
    return target
