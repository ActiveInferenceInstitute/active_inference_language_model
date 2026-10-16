#!/usr/bin/env python3
"""Opt-in live verification of exact Zenodo records and deposited PDFs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ailm.reference_sources import (
    DEFAULT_TIMEOUT_SECONDS,
    ReferenceVerificationError,
    verify_reference_sources,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare data/reference_sources.json with its exact official Zenodo records. "
            "Metadata is checked by default; --full-files also authenticates deposited PDFs."
        )
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=PROJECT_ROOT / "data" / "reference_sources.json",
        help="reference snapshot to verify (official zenodo.org targets only)",
    )
    parser.add_argument(
        "--full-files",
        action="store_true",
        help="stream and verify each complete deposited PDF",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="per-operation connection and read timeout",
    )
    parser.add_argument(
        "--pdfinfo",
        help="explicit Poppler pdfinfo executable; otherwise resolve pdfinfo from PATH",
    )
    parser.add_argument(
        "--require-pdfinfo",
        action="store_true",
        help="fail unless full-file mode verifies PDF page extents with pdfinfo",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        verified = verify_reference_sources(
            args.snapshot,
            full_files=args.full_files,
            timeout_seconds=args.timeout_seconds,
            pdfinfo_path=args.pdfinfo,
            require_pdfinfo=args.require_pdfinfo,
        )
    except ReferenceVerificationError as exc:
        print(f"reference verification failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema": "ailm-reference-verification-v1",
                "mode": "full-files" if args.full_files else "metadata-only",
                "references": [entry.to_dict() for entry in verified],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
