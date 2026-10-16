#!/usr/bin/env python3
"""Describe the exact ignored raw-corpus files consumed by release replay.

This command is intentionally read-only and emits one JSON object.  The release
orchestrator runs it inside the sanitized release environment, then independently
fingerprints the returned paths before and after all steps.  Brown is resolved
through its authenticated search contract; the other writers consume fixed
sidecar-relative paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ailm.corpus.brown import (
    BROWN_RAW_SHA256,
    BROWN_RAW_SIZE_BYTES,
    _read_verified_brown_source,
)
from ailm.corpus.cmudict import CMUDICT_RAW_SHA256, CMUDICT_RAW_SIZE_BYTES
from ailm.corpus.ud import UD_EWT_RAW_SHA256, UD_EWT_RAW_SIZE_BYTES
from ailm.corpus.ud_czech import UD_CZECH_PDTC_SHARDS


def describe(project_root: Path) -> dict[str, object]:
    """Return a complete path-and-pin inventory for all canonical raw inputs."""

    root = project_root.expanduser().resolve()
    if not (root / "pyproject.toml").is_file():
        raise ValueError(f"project root lacks pyproject.toml: {root}")
    brown_path, _brown_content = _read_verified_brown_source()
    inputs: list[dict[str, object]] = [
        {
            "label": "brown_zip",
            "path": str(brown_path),
            "expected_sha256": BROWN_RAW_SHA256,
            "expected_size_bytes": BROWN_RAW_SIZE_BYTES,
        },
        {
            "label": "cmudict_dict",
            "path": str(root / "vendor" / "cmudict.dict"),
            "expected_sha256": CMUDICT_RAW_SHA256,
            "expected_size_bytes": CMUDICT_RAW_SIZE_BYTES,
        },
        {
            "label": "ud_ewt_dev",
            "path": str(root / "vendor" / "en_ewt-ud-dev.conllu"),
            "expected_sha256": UD_EWT_RAW_SHA256,
            "expected_size_bytes": UD_EWT_RAW_SIZE_BYTES,
        },
    ]
    staging = root / "vendor" / "staging_ud_czech"
    inputs.extend(
        {
            "label": f"ud_czech_{split}",
            "path": str(staging / shard.filename),
            "expected_sha256": shard.sha256,
            "expected_size_bytes": shard.size_bytes,
        }
        for split, shard in sorted(UD_CZECH_PDTC_SHARDS.items())
    )
    return {"schema": "ailm-raw-corpus-input-inventory-v1", "inputs": inputs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(describe(args.project_root), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
