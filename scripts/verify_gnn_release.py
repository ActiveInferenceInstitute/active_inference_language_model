"""Verify the pinned upstream GNN release offline, without importing its parser."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ailm.verification.upstream_gnn import (  # noqa: E402
    UpstreamReleaseError,
    load_release_pin,
    selected_upstream_root,
    verify_release_checkout,
)


def main() -> int:
    try:
        evidence = verify_release_checkout(
            selected_upstream_root(ROOT), load_release_pin(ROOT / "gnn/upstream.json")
        )
    except (OSError, UpstreamReleaseError) as error:
        print(f"GNN release verification refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
