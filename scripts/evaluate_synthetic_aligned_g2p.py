#!/usr/bin/env python3
"""Offline evaluation of the stipulated aligned spelling-to-pronunciation arm.

Loads the validated artifact (default ``vendor/synthetic_aligned_g2p_counts.json``),
fits the shipped G2P engine on the stipulated train pairs, and reports every
arm of protocol P2 §3.3-3.4 on the identical held-out positions:

- the native stipulated engine (emission + transition + start filter decode);
- the separately named ``T``-lookup baseline (the trivial closed-form map);
- the A1 English-transfer control (same engine on the CMUdict stride split);
- the A2 broken-correspondence control (engine refit on ``sigma . T`` pairs);
- the A3 analytic uniform floor.

Same-task mechanisms may legitimately tie; the report states the comparison
instead of constructing a beat-the-lookup gate, and carries no transferred
thresholds.  The report is printed as JSON (or written with ``--report``) and
**no claim is recorded anywhere** — this is the arm's offline half; probe arms
need their own execution handoff under the existing Ollama discipline.

Usage
-----
    .venv/bin/python scripts/evaluate_synthetic_aligned_g2p.py [--artifact PATH]
        [--english-counts PATH] [--report PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _artifact_io import atomic_write_json  # noqa: E402 - project path established above

from ailm.benchmarks.aligned_g2p import (  # noqa: E402 - project path established above
    ALIGNED_G2P_ARTIFACT_NAME,
    artifact_pairs,
    evaluate_aligned_g2p,
    load_aligned_g2p,
)
from ailm.levels.phoneme import g2p_split, load_phoneme_counts  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline evaluation of the stipulated aligned spelling-to-"
            "pronunciation arm; records no claims."
        )
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=REPO_ROOT / "vendor" / ALIGNED_G2P_ARTIFACT_NAME,
        help="the validated artifact to evaluate",
    )
    parser.add_argument(
        "--english-counts",
        type=Path,
        default=REPO_ROOT / "vendor" / "cmudict_counts.json",
        help=(
            "vendored CMUdict counts feeding the A1 English-transfer control "
            "(default: the shipped artifact)"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="also write the JSON report to this path",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = load_aligned_g2p(vendor_dir=args.artifact.resolve().parent)
    correspondence, train, test = artifact_pairs(payload)

    english_train_pairs: list[tuple[str, list[str]]] | None = None
    if args.english_counts is not None:
        counts = load_phoneme_counts(vendor_dir=args.english_counts.resolve().parent)
        english_train_pairs, _english_test = g2p_split(counts)

    report = evaluate_aligned_g2p(
        correspondence, train, test, english_train_pairs=english_train_pairs
    )
    report["artifact"] = {
        "path": str(args.artifact),
        "schema": payload["schema"],
        "train_words": payload["provenance"]["train_words"],
        "test_words": payload["provenance"]["test_words"],
        "word_length_L": payload["provenance"]["word_length_L"],
    }
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.report is not None:
        atomic_write_json(args.report, report, indent=1, sort_keys=True)
        print(f"[report written to {args.report}]")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
