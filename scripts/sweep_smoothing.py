#!/usr/bin/env python3
"""Sweep the add-k pseudocount and report Level 1 held-out perplexity per value.

Why this exists
---------------
Add-1 Laplace used to be written into ``build_letter_transition`` as ``np.ones(...)``, so
the model's sensitivity to it could not be inspected without editing source. With
``ailm.config.LevelConfig`` the pseudocount is a parameter, and this script is the
inspectable sensitivity table: one row per ``k``, each row a full held-out
measurement rather than an interpolation.

What to read off it
-------------------
The curve is flat-then-steep, not monotone. Between ``k = 0.001`` and ``k = 10``
the perplexity moves by well under 0.1 on this eval set — the letter bigram table
is dense enough that the pseudocount barely matters. Past ``k = 1000`` the
smoothing mass dominates the corpus counts and the model collapses toward the
uniform ceiling of ``exp(log 30) = 30``. So the shipped default is not delicately
tuned: it sits in a wide basin, which is the honest robustness statement and the
only one this sweep supports.

Eval set
--------
``ailm.evaluation.STAGE1_EVALUATION_TEXT`` — imported rather than re-typed, so the
``k = 1`` row is the same measurement the Stage 1 gate enforces and the ledger
records. Every row scores the **whole** string; nothing is subsampled, so no
stride question arises (see ``ailm.levels.phoneme._iter_pairs`` for why a prefix
subsample would be reproducible but not representative).

Claims discipline
-----------------
This script writes **nothing**. Claim labels have exactly one writer and the
recorders own them (``tests/test_claims_single_writer.py`` enforces it). If a row
here is worth citing in prose, it needs a recorder, not a copy-paste.

Run:  .venv/bin/python scripts/sweep_smoothing.py
      .venv/bin/python scripts/sweep_smoothing.py --grid 0.5,1,2
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# pymdp emits a benign "JAX array is being set as static" UserWarning per Agent().
warnings.filterwarnings("ignore", message=".*JAX array is being set as static.*")

from ailm.config import DEFAULT_LEVEL_CONFIG, LevelConfig  # noqa: E402
from ailm.evaluation import STAGE1_EVALUATION_TEXT  # noqa: E402
from ailm.levels.letter import (  # noqa: E402
    N_SYMBOLS,
    build_letter_agent,
    build_letter_transition,
    compute_perplexity,
    load_brown_counts,
)

#: Default pseudocount grid: five orders of magnitude below the default and six
#: above, so both the flat basin and the collapse toward uniform are visible.
DEFAULT_GRID: tuple[float, ...] = (
    0.001,
    0.01,
    0.1,
    0.5,
    1.0,
    2.0,
    10.0,
    100.0,
    1000.0,
    1_000_000.0,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--grid",
        default=None,
        help=(
            "Comma-separated pseudocounts to sweep. Every value must be finite "
            "and > 0; LevelConfig rejects the rest. Defaults to "
            f"{','.join(repr(k) for k in DEFAULT_GRID)}."
        ),
    )
    return parser.parse_args(argv)


def _grid_from_args(raw: str | None) -> list[float]:
    """Turn ``--grid`` into a list of floats, failing loudly on junk."""
    if raw is None:
        return list(DEFAULT_GRID)
    values = [float(token) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError(f"--grid parsed to no values from {raw!r}")
    return values


def main(argv: list[str] | None = None) -> int:
    """Measure and print the sensitivity table."""
    args = parse_args(argv)
    grid = _grid_from_args(args.grid)

    counts = load_brown_counts()
    uniform = float(N_SYMBOLS)

    print("Level 1 letter model — held-out perplexity vs add-k smoothing")
    print(
        f"  eval set : ailm.evaluation.STAGE1_EVALUATION_TEXT "
        f"({len(STAGE1_EVALUATION_TEXT)} chars, scored in full)"
    )
    print(f"  training : vendor/brown_counts.json ({len(counts)} observed bigrams)")
    print(f"  uniform  : {uniform:.4f} (exp(log {N_SYMBOLS}))")
    print(f"  default  : smoothing = {DEFAULT_LEVEL_CONFIG.smoothing}")
    print()

    measured: list[tuple[float, float]] = []
    for k in grid:
        agent = build_letter_agent(build_letter_transition(counts, config=LevelConfig(smoothing=k)))
        measured.append((k, compute_perplexity(agent, STAGE1_EVALUATION_TEXT)))

    best_k, best_pp = min(measured, key=lambda row: row[1])
    default_pp = next((pp for k, pp in measured if k == DEFAULT_LEVEL_CONFIG.smoothing), None)

    header = f"{'smoothing k':>14}  {'perplexity':>11}  {'vs best':>9}  {'vs uniform':>11}"
    print(header)
    print("-" * len(header))
    for k, pp in measured:
        marks = []
        if k == DEFAULT_LEVEL_CONFIG.smoothing:
            marks.append("default")
        if k == best_k:
            marks.append("best")
        suffix = f"   <- {', '.join(marks)}" if marks else ""
        print(f"{k:>14g}  {pp:>11.4f}  {pp - best_pp:>+9.4f}  {pp - uniform:>+11.4f}{suffix}")

    print()
    if default_pp is not None:
        print(
            f"Default k={DEFAULT_LEVEL_CONFIG.smoothing} scores {default_pp:.4f}; the best "
            f"grid point k={best_k:g} scores {best_pp:.4f} "
            f"(difference {default_pp - best_pp:+.4f})."
        )
    print(
        "Nothing here is recorded. Claim labels have exactly one writer; if a row "
        "belongs in prose, add it to a recorder under scripts/."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
