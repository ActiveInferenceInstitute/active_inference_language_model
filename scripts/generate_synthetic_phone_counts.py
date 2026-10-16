#!/usr/bin/env python3
"""Generate the stipulated synthetic phone-bigram artifact.

Writes ``vendor/synthetic_phone_counts.json`` as a flat mapping
``{"<prev>,<next>": count}`` matching the ``phone_bigrams`` key format inside
``vendor/cmudict_counts.json``, so
:func:`ailm.levels.phoneme.build_phone_transition` consumes the numerical
payload with no adapter. CMUdict's file additionally carries unigram, G2P and
provenance sections this artifact deliberately does not.

Like the letter artifact, this one is derived from no corpus at all. The
language is stipulated in :mod:`ailm.levels.synthetic`: a column-stochastic
transition over the 39 canonical ARPAbet phones written down in code, sampled
to produce a training stream. CMUdict may sit inside a reference model's
training data, so a benchmark on it cannot separate inference from recall. A
locally generated artifact narrows the claim to exact-artifact overlap; it
does not reveal what a reference model encountered during pretraining.

Determinism
-----------
Fixed seed, bulk uniform draw, cumulative-column walk, keys sorted in
inventory order, no timestamps. A second run reproduces the file
byte-for-byte, which is what makes the sha256 in
``docs/provenance/sources.md`` meaningful. Regenerating with different
``--symbols`` or ``--seed`` changes the hash and requires updating that
document in the same change.

Usage
-----
    .venv/bin/python scripts/generate_synthetic_phone_counts.py
        [--output PATH] [--symbols N] [--seed S]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from _artifact_io import atomic_write_json  # noqa: E402  (path setup must precede the import)

from ailm.levels.synthetic import (  # noqa: E402
    DEFAULT_PHONE_CORPUS_SEED,
    DEFAULT_PHONE_CORPUS_SYMBOLS,
    SYNTHETIC_PHONES,
    build_synthetic_phone_counts,
)

DEFAULT_OUTPUT = REPO_ROOT / "vendor" / "synthetic_phone_counts.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--symbols", type=int, default=DEFAULT_PHONE_CORPUS_SYMBOLS)
    parser.add_argument("--seed", type=int, default=DEFAULT_PHONE_CORPUS_SEED)
    args = parser.parse_args()

    payload = build_synthetic_phone_counts(n_symbols=args.symbols, seed=args.seed)

    atomic_write_json(args.output, payload, indent=2)

    cells = len(SYNTHETIC_PHONES) ** 2
    total = sum(payload.values())
    print(
        f"seed={args.seed} symbols={args.symbols} distinct_bigrams={len(payload)}/{cells} "
        f"total={total}"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
