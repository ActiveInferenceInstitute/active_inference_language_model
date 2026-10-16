#!/usr/bin/env python3
"""Generate the stipulated aligned spelling-to-pronunciation artifact.

Writes ``vendor/synthetic_aligned_g2p_counts.json`` (schema
``ailm-synthetic-aligned-g2p-v1``): a seeded per-letter correspondence from the
26 lowercase letters onto a fixed subset of the canonical 39-phone ARPAbet
inventory, and train/test splits of fixed-length spelling words cut from
independently seeded trajectories of the shipped stipulated letter chain
(``ailm.levels.synthetic.generate_sequence``), one word per trajectory, with
each pronunciation the correspondence applied per position.

Protocol P2 of ``docs/extension_protocols.md`` §3 (frozen:
``output/reports/omp-dispatch-20260908/evidence/extensions-E1R-frozen/``).
Deliberately not an English reading; the table is recorded and digested.

Determinism: fixed seeds, one bulk uniform draw per trajectory walked against
precomputed cumulative columns, sorted canonical JSON digests, no timestamps.
A second run reproduces the file byte-for-byte, which is what makes the sha256
in ``docs/provenance/sources.md`` meaningful.  Changing any protocol constant
changes the hash and requires updating that document in the same change.

Refusals: within-split duplicates and cross-split collisions are checked
against the union of accepted words on every draw; exhausting the declared
64-attempt resample budget exits nonzero with a refusal message and writes
nothing.  No claim is recorded here, ever.

Usage
-----
    .venv/bin/python scripts/generate_synthetic_aligned_g2p.py [--output PATH]
        [--train-words N] [--test-words N] [--word-length N]
        [--correspondence-seed S]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _artifact_io import atomic_write_json  # noqa: E402 - project path established above

from ailm.benchmarks.aligned_g2p import (  # noqa: E402 - project path established above
    ALIGNED_G2P_ARTIFACT_NAME,
    ALIGNED_G2P_CORRESPONDENCE_SEED,
    ALIGNED_G2P_TEST_WORDS,
    ALIGNED_G2P_TRAIN_WORDS,
    ALIGNED_G2P_WORD_LENGTH,
    build_artifact_payload,
    build_splits,
    stipulated_correspondence,
    validate_artifact_payload,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the stipulated aligned spelling-to-pronunciation artifact; records no claims."
        )
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "vendor" / ALIGNED_G2P_ARTIFACT_NAME
    )
    parser.add_argument("--train-words", type=int, default=ALIGNED_G2P_TRAIN_WORDS)
    parser.add_argument("--test-words", type=int, default=ALIGNED_G2P_TEST_WORDS)
    parser.add_argument("--word-length", type=int, default=ALIGNED_G2P_WORD_LENGTH)
    parser.add_argument("--correspondence-seed", type=int, default=ALIGNED_G2P_CORRESPONDENCE_SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if (
        args.train_words != ALIGNED_G2P_TRAIN_WORDS
        or args.test_words != ALIGNED_G2P_TEST_WORDS
        or args.word_length != ALIGNED_G2P_WORD_LENGTH
        or args.correspondence_seed != ALIGNED_G2P_CORRESPONDENCE_SEED
        or args.output != REPO_ROOT / "vendor" / ALIGNED_G2P_ARTIFACT_NAME
    ):
        print(
            "fixture-sized generation: the written artifact records the actual "
            "split parameters and is NOT the canonical vendored input; register "
            "nothing from it.",
            file=sys.stderr,
        )
    correspondence = stipulated_correspondence(seed=args.correspondence_seed)
    train, test, stats = build_splits(
        correspondence,
        n_train=args.train_words,
        n_test=args.test_words,
        word_length=args.word_length,
    )
    payload = build_artifact_payload(
        correspondence,
        train,
        test,
        stats,
        word_length=args.word_length,
        generator_path="scripts/generate_synthetic_aligned_g2p.py",
    )
    validate_artifact_payload(payload)
    atomic_write_json(args.output, payload, indent=1, sort_keys=True)
    provenance = payload["provenance"]
    print(f"artifact written: {args.output}")
    print(f"schema: {payload['schema']}")
    print(
        f"train/test words: {provenance['train_words']}/{provenance['test_words']} at L={provenance['word_length_L']}"
    )
    print(f"correspondence codomain: {len(set(correspondence.values()))} of 39 phones")
    print(f"train_words_digest: {provenance['train_words_digest']}")
    print(f"test_words_digest: {provenance['test_words_digest']}")
    print(f"correspondence_table_digest: {provenance['correspondence_table_digest']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
