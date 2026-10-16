#!/usr/bin/env python3
"""Quickstart: the AILM Python API, no CLI.

Everything here calls public functions in :mod:`ailm.demo` — the same code
`python -m ailm` runs. Two calls, both offline and read-only:

  1. `ailm.demo.load_gate_audit()` — the committed Stage 1-6 gate verdicts,
     read from `output/reports/gate_audit.json`. No recomputation.
  2. `ailm.demo.decode_word_with_posteriors()` — a live Stage 2B G2P decode:
     a real `pymdp` agent built from the vendored CMUdict artifact, run
     letter-by-letter, keeping the full posterior at each position.

Run from a source checkout (this needs `output/reports/gate_audit.json` and
`vendor/cmudict_counts.json`, which do not ship in an installed wheel):

    .venv/bin/python examples/quickstart_demo.py
    .venv/bin/python examples/quickstart_demo.py galaxy
"""

from __future__ import annotations

import sys

from ailm.demo import (
    decode_word_with_posteriors,
    format_decode_table,
    format_gate_table,
    load_gate_audit,
)


def main() -> int:
    word = sys.argv[1] if len(sys.argv) > 1 else "galaxy"

    # 1. Gate verdicts: baseline, ceiling, and the resulting PASS/FAIL verdict
    #    for every stage, exactly as recorded in the committed report.
    rows = load_gate_audit()
    print(format_gate_table(rows))
    print()

    # 2. A live decode. `positions` is a list of dicts:
    #    {"position": int, "letter": str, "top_phones": [(phone, probability), ...]}
    #    — plain data, not model-internal objects, so it is easy to consume
    #    downstream without any AILM-specific type.
    positions = decode_word_with_posteriors(word, top_k=3)
    print(format_decode_table(word, positions))

    # A caller who only wants the top phone per position, ignoring the rest of
    # the posterior, reads it directly off that same structure:
    top_guess = "".join(entry["top_phones"][0][0] for entry in positions)
    print(f"\ntop-1 phone sequence: {top_guess}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
