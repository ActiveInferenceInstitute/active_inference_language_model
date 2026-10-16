#!/usr/bin/env python3
"""Generate the stipulated synthetic letter-bigram artifact.

Writes ``vendor/synthetic_counts.json`` as a flat mapping ``{"<a>,<b>": count}``
matching the ``counts`` mapping inside ``vendor/brown_counts.json``, so
:func:`ailm.levels.letter.build_letter_transition` consumes the numerical payload with
no adapter. Brown's file additionally carries an external-source envelope.

Unlike every other artifact under ``vendor/``, this one is derived from no
corpus at all. The language is stipulated in :mod:`ailm.levels.synthetic`: a
column-stochastic transition written down in code, sampled to produce a training
stream. Brown and CMUdict may sit inside a reference model's training data, so a
benchmark on them cannot separate inference from recall. A locally generated
artifact narrows the claim to exact-artifact overlap; it does not reveal what a
reference model encountered during pretraining.

Determinism
-----------
Fixed seed, bulk uniform draw, cumulative-column walk, keys sorted in ALPHABET
order, no timestamps. A second run reproduces the file byte-for-byte, which is
what makes the sha256 in ``docs/provenance/sources.md`` meaningful.
Regenerating with different ``--chars`` or ``--seed`` changes the hash and
requires updating that document in the same change.

Usage
-----
    .venv/bin/python scripts/generate_synthetic_counts.py [--output PATH]
                                                          [--chars N] [--seed S]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from _artifact_io import atomic_write_json  # noqa: E402

from ailm.levels.synthetic import (  # noqa: E402  (path setup must precede the import)
    DEFAULT_CORPUS_CHARS,
    DEFAULT_CORPUS_SEED,
    SYNTHETIC_LETTERS,
    build_synthetic_counts,
)

DEFAULT_OUTPUT = REPO_ROOT / "vendor" / "synthetic_counts.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chars", type=int, default=DEFAULT_CORPUS_CHARS)
    parser.add_argument("--seed", type=int, default=DEFAULT_CORPUS_SEED)
    args = parser.parse_args()

    payload = build_synthetic_counts(n_chars=args.chars, seed=args.seed)

    atomic_write_json(args.output, payload, indent=2)

    cells = len(SYNTHETIC_LETTERS) ** 2
    total = sum(payload.values())
    print(
        f"seed={args.seed} chars={args.chars} distinct_bigrams={len(payload)}/{cells} total={total}"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
