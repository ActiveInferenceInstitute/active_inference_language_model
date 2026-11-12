"""Materialize the pinned upstream GNN release the conformance arm reads.

The conformance gate needs the upstream parser at exactly the release recorded in
``gnn/upstream.json``. The hosted job used to clone it with an inline shell block
into a sibling directory of the checkout, which put a second repository beside a
vault-owned working copy and made the hosted job and a local run two different
procedures. This script is the single procedure.

The release identity is *not* read here: ``gnn.importers.read_pin`` — the reader
the pinned release ships — decides the shape, the schema and the revision, and
:func:`verify_release_checkout` refuses a checkout whose HEAD, tag ref or source
bytes disagree with it. A checkout has to exist before that reader can be
imported from it, so the one thing this script reads from the pin is the pair of
coordinates a clone needs (the repository URL and the release tag), and only when
there is no checkout yet.

``refs/`` is ignored, so materializing never changes the tracked or untracked
state the release runner fingerprints. The checkout lives at one tag-independent
path: a checkout that holds a *different* commit than the pin is a cache entry
for a previous pin and is replaced, while a checkout of the pinned commit whose
source bytes were modified is a refusal naming the file, never a silent repair.

Usage:
    .venv/bin/python scripts/fetch_gnn_upstream.py
    .venv/bin/python scripts/fetch_gnn_upstream.py --print-path
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ailm.verification.upstream_gnn import (  # noqa: E402
    CHECKOUT_RELATIVE_PATH,
    UpstreamRelease,
    UpstreamReleaseError,
    checkout_head,
    load_importer_pin,
    verify_release_checkout,
)

PIN_PATH = ROOT / "gnn" / "upstream.json"
CHECKOUT = ROOT / CHECKOUT_RELATIVE_PATH


def _clone_coordinates(pin_path: Path) -> tuple[str, str]:
    """Return the repository and tag a clone needs, before the reader exists.

    This is the materialization input, not a pin reader: it takes two strings and
    makes no decision about the pin's shape. The pinned reader re-reads the same
    file as soon as the checkout exists, and the custody check then refuses
    unless HEAD, the tag ref and every source byte agree with what it says.
    """

    try:
        payload = json.loads(pin_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UpstreamReleaseError(f"cannot read the clone coordinates: {error}") from error
    repository = payload.get("repository") if isinstance(payload, dict) else None
    tag = payload.get("tag") if isinstance(payload, dict) else None
    if not isinstance(repository, str) or not isinstance(tag, str):
        raise UpstreamReleaseError(f"{pin_path} does not name a repository and a tag to clone")
    return repository, tag


def _clone(repository: str, tag: str, target: Path) -> Path:
    """Clone the tag's ``src/`` into ``target``; the reader decides if it is right.

    Only ``src/`` is materialized: it is what supplies the parser and the reader,
    and what custody verifies. Nothing else about the release is read, and a
    checkout that carried the upstream's own ``*.gnn.md`` files would make this
    repository's specification count a function of the pinned release.
    """

    staging = target.with_name(f"{target.name}.incoming-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    for command in (
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            tag,
            "--filter=blob:none",
            "--sparse",
            repository,
            str(staging),
        ],
        ["git", "-C", str(staging), "sparse-checkout", "set", "src"],
    ):
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            shutil.rmtree(staging, ignore_errors=True)
            raise UpstreamReleaseError(
                f"cannot clone {repository} at {tag}: {result.stderr.strip()}"
            )
    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)
    return target


def _materialize() -> Path:
    """Return the checkout of the pinned release, cloning it when it is absent."""

    if CHECKOUT.is_dir():
        pin = load_importer_pin(PIN_PATH, checkout_root=CHECKOUT)
        if checkout_head(CHECKOUT) == pin.commit:
            return CHECKOUT
        # A cache entry for a previous pin, not a claim about this one: replace it.
    repository, tag = _clone_coordinates(PIN_PATH)
    return _clone(repository, tag, CHECKOUT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--print-path",
        action="store_true",
        help="print only the verified checkout path, for a shell consumer",
    )
    args = parser.parse_args(argv)
    try:
        target = _materialize()
        pin: UpstreamRelease = load_importer_pin(PIN_PATH, checkout_root=target)
        evidence = verify_release_checkout(target, pin)
    except (OSError, UpstreamReleaseError) as error:
        print(f"GNN upstream checkout refused: {error}", file=sys.stderr)
        return 1
    if args.print_path:
        print(target)
        return 0
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
