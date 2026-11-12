"""Materialize the pinned upstream GNN release the conformance arm reads.

The conformance gate needs the upstream Markdown parser at exactly the tag and
commit recorded in ``gnn/upstream.json``. The hosted job used to clone it with an
inline shell block into a sibling directory of the checkout, which put a second
repository beside a vault-owned working copy and made the hosted job and a local
run two different procedures. This script is the single procedure: read the pin,
clone that tag into the Git-ignored ``refs/gnn-upstream/<tag>`` directory, and
refuse a checkout whose HEAD, tag ref, or source bytes disagree with the pin.

``refs/`` is ignored, so materializing the checkout never changes the tracked or
untracked state the release runner fingerprints. An existing checkout is verified,
never re-cloned or deleted: a mismatch is a refusal naming the drift, and the
directory stays for inspection.

Usage:
    .venv/bin/python scripts/fetch_gnn_upstream.py
    .venv/bin/python scripts/fetch_gnn_upstream.py --print-path
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ailm.verification.upstream_gnn import (  # noqa: E402
    UpstreamRelease,
    UpstreamReleaseError,
    load_release_pin,
    verify_release_checkout,
)

PIN_PATH = ROOT / "gnn" / "upstream.json"
CHECKOUT_ROOT = ROOT / "refs" / "gnn-upstream"


def checkout_path(tag: str) -> Path:
    """The one checkout location for a pinned tag, inside the ignored tree."""

    return CHECKOUT_ROOT / tag


def _clone(pin: UpstreamRelease) -> Path:
    """Clone the pinned tag when its checkout is absent; never replace one."""

    target = checkout_path(pin.tag)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", pin.tag, pin.repository, str(target)],
        stdin=subprocess.DEVNULL,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise UpstreamReleaseError(
            f"cannot clone {pin.repository} at {pin.tag}: {result.stderr.strip()}"
        )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--print-path",
        action="store_true",
        help="print only the verified checkout path, for a shell consumer",
    )
    args = parser.parse_args(argv)
    try:
        pin = load_release_pin(PIN_PATH)
        target = _clone(pin)
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
