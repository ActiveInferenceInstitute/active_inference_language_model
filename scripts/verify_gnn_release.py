"""Verify the pinned upstream GNN release offline, without importing its parser.

The pin is read by ``gnn.importers`` — the reader the pinned release ships, which
is the same reader the conformance arm runs — and the selected checkout is then
held against Git objects: HEAD, the tag ref, and every source byte under ``src/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ailm.verification.upstream_gnn import (  # noqa: E402
    UpstreamReleaseError,
    load_importer_pin,
    selected_upstream_root,
    verify_release_checkout,
)


def main() -> int:
    try:
        root = selected_upstream_root(ROOT)
        pin = load_importer_pin(ROOT / "gnn/upstream.json", checkout_root=root)
        evidence = verify_release_checkout(root, pin)
    except (OSError, UpstreamReleaseError) as error:
        print(f"GNN release verification refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
