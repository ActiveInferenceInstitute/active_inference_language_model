#!/usr/bin/env python3
"""Generate ``formal/AILM/LedgerData.lean`` from ``data/claims.json``.

Why this script exists
----------------------
The formal layer needs the *measured* numbers to say anything about this
project rather than about arithmetic in general. But a measured number typed
by hand into a Lean file is exactly the defect the claims ledger exists to
prevent: the ledger moves when a recorder re-runs, the hand-typed copy does
not, and the proof then certifies a value nobody measured.

So no ledger value is ever hand-typed into Lean. This script is the only
writer of ``formal/AILM/LedgerData.lean``, and
``tests/test_formal_isomorphism.py`` fails unless a fresh run reproduces the
committed file **byte for byte**. A hand edit cannot survive the suite, and a
ledger change that is not regenerated cannot either.

Exact conversion policy — the whole point of the file
-----------------------------------------------------
The ledger stores decimals. Lean's ``Rat`` is exact. Rounding twice, or
converting through a binary float, would put a number in Lean that is *close
to* the recorded one, and "close to" is precisely the property a formal proof
must not silently assume.

1. ``data/claims.json`` is parsed with ``parse_float=str``, so the source is
   the literal decimal **text** in the file, never a re-rounded float.
2. That text becomes a ``fractions.Fraction`` exactly: ``"68.34"`` is
   ``6834/100``, which ``Fraction`` reduces to ``3417/50``.
3. The reduced pair is emitted as ``AILM.lit n d``. ``lit`` is built on
   ``Rat.mk'``, which carries a ``Nat.Coprime`` proof field discharged by
   ``decide`` at elaboration time — so a non-reduced or zero-denominator
   literal is a **compile error**, not a silently wrong number. The gcd
   reduction has to happen here, on the Python side, and the Lean kernel
   checks that it did.

No rounding, no tolerance, no float arithmetic anywhere in this file.

Determinism
-----------
The output has a fixed header with **no timestamp**, no absolute path, and no
hostname. Blocks are emitted in a fixed order and entries are sorted by Lean
identifier within each block. Two runs on the same ledger produce identical
bytes; a run on a changed ledger produces different bytes. Both directions are
tested.

What this script deliberately does NOT emit
-------------------------------------------
* **Metric direction** (higher/lower is better). That is structural — a
  property of the metric, not a measurement — so it is written by hand in
  ``formal/AILM/Instances.lean`` where it mirrors ``scripts/audit_gates.py``.
  Emitting it here would dress a hand-maintained constant as ledger data.
* **The Stage 6 and generation gate rows.** ``scripts/audit_gates.py`` audits
  them, but neither typed comparator has a ledger label (Stage 6's is
  fitted-on-test, generation's is the analytic 0.0). A triple is emitted only when all of its
  components exist in the ledger; inventing the missing third value would be
  the fabrication this layer exists to make impossible.
* **The full coupling sweep.** Only three of its six grid points are recorded
  (see ``COUPLING_SCOPE`` below), so only three are emitted, and
  ``Instances.lean`` scopes its coupling theorems to what those three support.

Usage
-----
    .venv/bin/python scripts/generate_lean_ledger.py
    .venv/bin/python scripts/generate_lean_ledger.py --out output/probe.lean
    .venv/bin/python scripts/generate_lean_ledger.py --check

``--out`` goes through the atomic publication writer, which refuses any
symlinked path component; on macOS ``/tmp`` and ``/var`` are symlinks, so a
probe path there must be given as ``os.path.realpath()`` of the destination.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUT = _PROJECT_ROOT / "formal" / "AILM" / "LedgerData.lean"
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from _artifact_io import atomic_write_text  # noqa: E402

#: The lexical channel's mixing-weight grid, in the order the sweep was run.
#: Structural (it is the grid ``scripts/record_lexical_channel_claims.py``
#: fixed in advance), but every *value* at each weight comes from the ledger.
LEXICAL_GRID: tuple[str, ...] = ("0.0", "0.1", "0.2", "0.3", "0.5", "0.7", "0.9")

#: Why the coupling channel gets three points and the lexical channel gets
#: seven. Reproduced into the generated file so a reader of the Lean sees the
#: scope without leaving it.
COUPLING_SCOPE = (
    "Only three points of the six-point coupling grid are recorded: the "
    "alpha=0 anchor, the BEST positive alpha, and alpha=1. The middle entry "
    "is the grid MAXIMUM over alpha > 0 (record_coupling_claims.py: "
    "best_alpha = max(positives, key=...)), so 'no entry beats the anchor' on "
    "this three-point list is exactly 'no positive weight in the full "
    "six-point grid beats the anchor' -- a stronger statement than its length "
    "suggests. What it is NOT is a claim about the SHAPE between the points: "
    "alphas 0.25, 0.5 and 0.75 have no recorded values, so no monotonicity "
    "claim about the full sweep can be read off this list."
)

#: Gate rows audited by ``scripts/audit_gates.py`` for which the ledger holds
#: every component. ``threshold`` is not listed because the audit sets it to
#: the baseline for every row; the generated file emits it as an alias so no
#: second literal pretends to be an independent measurement.
#:
#: ``scope`` records the population each triple describes. One row differs from
#: the audit's own run: 2B uses a strided subsample while the ledger labels name
#: the full held-out population. Stage 2A is closed-form on the full dictionary
#: in both places.
#: The VERDICT agrees in both cases (checked against the audit output), but the
#: numbers describe different position sets and saying so is not optional.
#:
#: These strings are **qualitative on purpose**. They used to quote the audit's
#: own subsample measurements ("25.0024/15.6530" and friends) as digits inside
#: this constant. Those digits resolved to no ledger label, so byte-identity
#: regeneration *preserved* them: a re-measurement of those populations would
#: leave them stale while every gate stayed green, inside the one file whose
#: header promises that nothing in it is typed by hand. A scope note may name a
#: population; it may not carry a measurement.
GATES: tuple[dict[str, Any], ...] = (
    {
        "prefix": "stage1",
        "gate_id": "stage1_letter",
        "baseline": "stage1_unigram_baseline",
        "comparator": "stage1_bigram_floor_perplexity",
        "model": "stage1_letter_perplexity",
        "scope": "held-out Brown text; same population as audit_gates.py",
    },
    {
        "prefix": "stage2Phoneme",
        "gate_id": "stage2_phoneme",
        "baseline": "stage2_phoneme_unigram_baseline",
        "comparator": "stage2_phoneme_all_position_floor_perplexity",
        "model": "stage2_phoneme_perplexity",
        "scope": "FULL CMUdict, closed form; same population as audit_gates.py",
    },
    {
        "prefix": "stage2G2p",
        "gate_id": "stage2_g2p",
        "baseline": "stage2_g2p_baseline_accuracy",
        "comparator": "stage2_g2p_smoothing_ceiling",
        "model": "stage2_g2p_accuracy",
        "scope": (
            "FULL held-out G2P test set. audit_gates.py scores this row on an "
            "evenly-spaced strided subsample instead; same verdict, different "
            "position set, and therefore different numbers than the ones below"
        ),
    },
    {
        "prefix": "stage3Word",
        "gate_id": "stage3_word",
        "baseline": "stage3_bigram_baseline",
        "comparator": "stage3_ceiling_perplexity",
        "model": "stage3_word_perplexity",
        "scope": "in-vocabulary held-out Brown; same population as audit_gates.py",
    },
    {
        "prefix": "stage4Derived",
        "gate_id": "stage4_case",
        "baseline": "stage4_derived_baseline",
        "comparator": "stage4_derived_smoothing",
        "model": "stage4_derived_agreement",
        "scope": (
            "derived case oracle, held-out UD EWT; same population as "
            "audit_gates.py, which prints these same three measurements at "
            "full precision where the ledger stores them rounded. READ THIS "
            "BEFORE CITING THE COMPARATOR: on this row the shipped model scores "
            "ABOVE the recorded forward-backward SMOOTHING reference. It is a "
            "reference comparator, not a bound over every model in the class. "
            "Nothing in this tree assumes otherwise -- G6's "
            "dominance hypothesis is about the BASELINE, which does sit below "
            "it -- but the manuscript sentence that once justified that "
            "hypothesis 'by construction' was wrong on exactly this row"
        ),
    },
    {
        "prefix": "stage5Clause",
        "gate_id": "stage5_clause",
        "baseline": "stage5_typefree_perplexity",
        "comparator": "stage5_type_oracle_perplexity",
        "model": "stage5_mixture_perplexity",
        "scope": "held-out clause corpus; same population as audit_gates.py",
    },
)

#: Historical declared thresholds and the information-theoretic bounds they
#: were judged against. These are what make the file interesting: the same
#: declared bar classifies VACUOUS on one population and ABOVE_REFERENCE on
#: another, and both populations are in the ledger.
DECLARED: tuple[tuple[str, str, str], ...] = (
    (
        "stage1DeclaredGateNats",
        "stage1_declared_gate_nats",
        "the >=2 nats Stage 1 originally declared",
    ),
    (
        "stage1GainOverUniformNats",
        "stage1_gain_over_uniform_nats",
        "what the shipped letter bigram achieved",
    ),
    (
        "stage1MaxBigramGainNats",
        "stage1_max_bigram_gain_nats",
        "the order-1 ceiling: no bigram model beats this",
    ),
    (
        "stage3DeclaredGate",
        "stage3_declared_gate",
        "the perplexity < 500 Stage 3 originally declared",
    ),
    ("stage4DeclaredGate", "stage4_declared_gate", "the >=60% Stage 4 originally declared"),
    (
        "stage4GoldBaseline",
        "stage4_gold_baseline",
        "gold-morphology population: context-free argmax",
    ),
    (
        "stage4GoldCeiling",
        "stage4_gold_smoothing",
        "gold-morphology population: forward-backward smoothing",
    ),
    ("stage4GoldModel", "stage4_gold_agreement", "gold-morphology population: the shipped agent"),
)

#: Scalar lexical/coupling labels, as (Lean name, ledger label, comment).
LEXICAL_SCALARS: tuple[tuple[str, str, str], ...] = (
    ("lexicalAnchorAccuracy", "lexical_anchor_accuracy", "the uncoupled alpha=0 anchor"),
    ("lexicalBestAccuracy", "lexical_best_accuracy", "best accuracy over the grid"),
    ("lexicalBestAlpha", "lexical_best_alpha", "the weight that achieved it"),
)

COUPLING_SCALARS: tuple[tuple[str, str, str], ...] = (
    ("couplingAlpha0Accuracy", "coupling_alpha0_accuracy", "the uncoupled alpha=0 anchor"),
    ("couplingAlpha1Accuracy", "coupling_alpha1_accuracy", "the fully-coupled endpoint"),
    ("couplingBestAccuracy", "coupling_best_accuracy", "MAXIMUM over every alpha > 0 in the grid"),
    ("couplingBestAlpha", "coupling_best_alpha", "the weight that achieved it"),
)


class GeneratorError(RuntimeError):
    """A refusal: the ledger cannot support the file this script would write."""


def load_claims(ledger_path: Path) -> dict[str, Any]:
    """Load claim values with decimals kept as their exact source text.

    ``parse_float=str`` is the load-bearing argument. Without it every value
    would round-trip through a binary float before reaching ``Fraction``, and
    the "exact" conversion would be exact only about the float.
    """
    with ledger_path.open(encoding="utf-8") as handle:
        payload = json.load(handle, parse_float=str)
    claims = payload.get("claims")
    if not isinstance(claims, dict) or not claims:
        raise GeneratorError(f"{ledger_path} holds no claims; nothing to generate from.")
    return {label: entry["value"] for label, entry in claims.items()}


def exact_fraction(label: str, value: Any) -> Fraction:
    """Convert one ledger value to an exact reduced fraction.

    Accepts the decimal text produced by ``parse_float=str`` and plain JSON
    integers. Anything else (a bool, a list, a string that is not a number) is
    a refusal rather than a coercion: silently rendering ``True`` as ``1``
    would put a number in Lean that the ledger never recorded.
    """
    if isinstance(value, bool):
        raise GeneratorError(f"{label} is a boolean; it is not a rational measurement.")
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, str):
        try:
            return Fraction(value)
        except ValueError as exc:
            raise GeneratorError(f"{label} = {value!r} is not an exact decimal: {exc}") from exc
    raise GeneratorError(f"{label} has non-numeric value {value!r} of type {type(value).__name__}")


def lit(label: str, value: Any) -> str:
    """Render one ledger value as an ``AILM.lit`` literal in lowest terms."""
    fraction = exact_fraction(label, value)
    numerator = fraction.numerator
    denominator = fraction.denominator
    rendered = f"({numerator})" if numerator < 0 else str(numerator)
    return f"lit {rendered} {denominator}"


def nat_value(label: str, value: Any) -> int:
    """Render one ledger value as a Lean ``Nat``, refusing anything non-integral."""
    fraction = exact_fraction(label, value)
    if fraction.denominator != 1 or fraction.numerator < 0:
        raise GeneratorError(f"{label} = {value!r} is not a natural number.")
    return fraction.numerator


def require(claims: dict[str, Any], *labels: str) -> None:
    """Refuse to generate when any required label is missing from the ledger."""
    missing = sorted(label for label in labels if label not in claims)
    if missing:
        raise GeneratorError(
            f"the ledger is missing {missing}. Run the recorders that own those "
            f"labels before regenerating; emitting a placeholder would put an "
            f"unmeasured number inside a machine-checked proof."
        )


def _decl(name: str, literal: str, comment: str, label: str) -> str:
    """One ``def`` with its provenance comment. Fixed shape, no timestamps."""
    return f"/-- {comment} (ledger: `{label}`). -/\ndef {name} : Rat := {literal}"


def _wrap(text: str, width: int = 74, indent: str = "  ") -> list[str]:
    """Deterministic greedy wrap for the scope paragraphs in the header."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) > width and current:
            lines.append(indent + current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(indent + current)
    return lines


def _sweep_block(claims: dict[str, Any]) -> str:
    """The lexical sweep as a ``List Rat``, one grid point per line."""
    labels = [f"lexical_sweep_alpha_{alpha.replace('.', '_')}_accuracy" for alpha in LEXICAL_GRID]
    require(claims, *labels)
    literals = [lit(label, claims[label]) for label in labels]
    width = max(len(text) for text in literals)
    rows = []
    for index, (alpha, literal, label) in enumerate(
        zip(LEXICAL_GRID, literals, labels, strict=True)
    ):
        opener = "  [ " if index == 0 else "  , "
        rows.append(f"{opener}{literal.ljust(width)}  -- alpha {alpha}  ({label})")
    return "\n".join(rows) + "\n  ]"


def _grid_block() -> str:
    """The prospectively specified lexical grid, as a ``List Rat``.

    The one **structural** list in the generated file: these are the mixing
    weights the sweep was run at, fixed in advance by
    ``scripts/record_lexical_channel_claims.py``, not values anybody measured.
    Every entry is marked ``[structural]`` in place, and — deliberately — none
    of them carries a trailing ``(label)``, so the ledger-literal scanner in
    ``tests/test_formal_isomorphism.py`` does not try to resolve them against
    ``data/claims.json`` and find nothing.

    It is emitted for one reason: without it, ``lexicalPeakIndex`` is a number
    Python derived and Lean took on trust. With it, ``Instances.lean`` can
    ``decide`` that the index really does select ``lexical_best_alpha``.
    """
    fractions = [Fraction(alpha) for alpha in LEXICAL_GRID]
    literals = [f"lit {value.numerator} {value.denominator}" for value in fractions]
    width = max(len(text) for text in literals)
    rows = []
    for index, (alpha, literal) in enumerate(zip(LEXICAL_GRID, literals, strict=True)):
        opener = "  [ " if index == 0 else "  , "
        rows.append(f"{opener}{literal.ljust(width)}  -- alpha {alpha}  [structural]")
    return "\n".join(rows) + "\n  ]"


def _peak_index(claims: dict[str, Any]) -> int:
    """Index of the best recorded weight within the fixed grid.

    Derived, not asserted: the grid order is structural and
    ``lexical_best_alpha`` is a ledger value, so the index is a lookup. Lean
    re-decides the peak independently — this only spares ``Instances.lean`` a
    hand-typed ``3``.
    """
    best = exact_fraction("lexical_best_alpha", claims["lexical_best_alpha"])
    grid = [Fraction(alpha) for alpha in LEXICAL_GRID]
    if best not in grid:
        raise GeneratorError(
            f"lexical_best_alpha = {best} is not in the recorded grid {LEXICAL_GRID}. "
            f"The sweep grid and the ledger disagree; one of them moved."
        )
    return grid.index(best)


def _coupling_block(claims: dict[str, Any]) -> str:
    """The three recorded coupling points, ordered by mixing weight."""
    best_alpha = exact_fraction("coupling_best_alpha", claims["coupling_best_alpha"])
    if not Fraction(0) < best_alpha < Fraction(1):
        raise GeneratorError(
            f"coupling_best_alpha = {best_alpha} is not strictly between the two "
            f"recorded endpoints, so it cannot be placed between them in an "
            f"alpha-ordered list. Refusing to emit a mis-ordered sweep."
        )
    entries = (
        ("coupling_alpha0_accuracy", "alpha 0 (anchor)"),
        ("coupling_best_accuracy", f"alpha {best_alpha} (best over alpha > 0)"),
        ("coupling_alpha1_accuracy", "alpha 1 (fully coupled)"),
    )
    literals = [lit(label, claims[label]) for label, _ in entries]
    width = max(len(text) for text in literals)
    rows = []
    for index, (literal, (label, note)) in enumerate(zip(literals, entries, strict=True)):
        opener = "  [ " if index == 0 else "  , "
        rows.append(f"{opener}{literal.ljust(width)}  -- {note}  ({label})")
    return "\n".join(rows) + "\n  ]"


#: The generated file's fixed preamble. A constant, so no run-dependent value
#: (timestamp, path, hostname, user) can reach the output.
HEADER = """/-
  AILM.LedgerData — GENERATED FILE. DO NOT EDIT BY HAND.

  Written by `scripts/generate_lean_ledger.py` from `data/claims.json`.
  Regenerate with:

      .venv/bin/python scripts/generate_lean_ledger.py

  `tests/test_formal_isomorphism.py` requires a fresh run to reproduce this
  file byte for byte, so a hand edit here cannot survive the suite, and neither
  can a ledger change that was never regenerated.

  ## Every number below is a recorded measurement, with one marked exception

  Each `def` names the ledger label it came from. Nothing here is typed by
  hand, rounded, or adjusted.

  The exception is `lexicalGrid`: the mixing weights the sweep was *run at*,
  fixed in advance rather than measured. It is structure, in the same sense the
  `Direction` annotations in `Instances.lean` are structure, and every one of
  its entries says `[structural]` in place. It is emitted so the kernel can
  check `lexicalPeakIndex` against the recorded `lexical_best_alpha` instead of
  trusting the Python that derived it.

  The synthetic shape witnesses in `GateVerdict.lean`, `SweepShape.lean`,
  `Filtering.lean` and `Bonferroni.lean` are a third thing again: invented
  numbers chosen to exhibit a shape. Do not confuse the three — this is the only
  module in the tree whose literals describe the world.

  No prose in this file may carry a measurement. A number written into a comment
  here resolves to no ledger label, so byte-identity regeneration preserves it
  forever: it would go stale on the next re-measurement while every proof below
  it stayed green. Scope notes name populations, never values.

  ## Exact conversion policy

  `data/claims.json` is parsed with `parse_float=str`, so the source is the
  literal decimal text in the file rather than a re-rounded binary float. That
  text becomes an exact `Fraction` (an invented 68.34 -> 6834/100), reduced to
  lowest terms (3417/50) and emitted as `lit n d`. `lit` is built on `Rat.mk'`,
  whose `Nat.Coprime` field is discharged by `decide` at elaboration time — so
  if this script ever emitted an unreduced fraction the file would fail to
  compile rather than quietly hold a wrong number. No float arithmetic, no
  rounding, and no tolerance anywhere in the conversion.

  ## What is deliberately NOT here

  * Metric direction (higher/lower is better) is structural, not measured, so
    it is hand-written in `Instances.lean` where it mirrors
    `scripts/audit_gates.py`.
  * The Stage 6 and generation gate rows. `audit_gates.py` audits both, but
    neither typed comparator has a ledger label (Stage 6's is fitted on test,
    generation's is the analytic 0.0). A triple is emitted only when all three
    components exist; inventing the third would be the fabrication this layer
    exists to make impossible.
  * The unrecorded interior of the coupling grid — see the scope note on
    `couplingRecorded`.
-/
import AILM.Core

namespace AILM"""


def render(claims: dict[str, Any]) -> str:
    """Render the whole generated module. A pure function of the ledger.

    Chunks are joined by exactly one blank line, so the file's layout is a
    consequence of its structure rather than of post-hoc whitespace repair.
    """
    for _, label, _ in LEXICAL_SCALARS + COUPLING_SCALARS + DECLARED:
        require(claims, label)
    require(claims, "lexical_optimum_is_interior", "coupling_grid_size")
    for gate in GATES:
        require(claims, gate["baseline"], gate["comparator"], gate["model"])

    chunks: list[str] = [HEADER]

    chunks.append("/-! ## Lexical channel — `sec:hierarchy`, the novel-message sweep -/")
    for name, label, comment in sorted(LEXICAL_SCALARS):
        chunks.append(_decl(name, lit(label, claims[label]), comment, label))
    flag = nat_value("lexical_optimum_is_interior", claims["lexical_optimum_is_interior"])
    chunks.append(
        "/-- The recorder's own interior-optimum flag, `1` for true. `Instances.lean`\n"
        "re-decides the shape from `lexicalSweep` and requires the two to agree, so\n"
        "this is a cross-check on the recorder rather than an input to any proof\n"
        "(ledger: `lexical_optimum_is_interior`). -/\n"
        f"def lexicalOptimumIsInteriorFlag : Nat := {flag}"
    )
    chunks.append(
        "/-- The prospectively specified lexical mixing-weight grid.\n\n"
        "STRUCTURAL, NOT MEASURED — the only such list in this file. These are the\n"
        "weights the sweep was run at, fixed in advance by\n"
        "`scripts/record_lexical_channel_claims.py`; no recorder measured them, so no\n"
        "entry names a ledger label and every entry is marked `[structural]`.\n\n"
        "It is here so `lexicalPeakIndex` stops being a number Lean takes on trust:\n"
        "`Instances.lean` decides that the index really selects `lexical_best_alpha`. -/\n"
        "def lexicalGrid : List Rat :=\n" + _grid_block()
    )
    chunks.append(
        "/-- Index of `lexical_best_alpha` within `lexicalGrid`. Derived here from the\n"
        "ledger and the grid order, then re-checked in the kernel: `Instances.lean`\n"
        "decides both that the sweep peaks at this index and that the grid holds the\n"
        "recorded best alpha there. -/\n"
        f"def lexicalPeakIndex : Nat := {_peak_index(claims)}"
    )
    chunks.append(
        "/-- The lexical mixing-weight sweep, index 0 = the uncoupled anchor. Every\n"
        "entry is a recorded accuracy; the grid order is the one\n"
        "`scripts/record_lexical_channel_claims.py` fixed in advance. -/\n"
        "def lexicalSweep : List Rat :=\n" + _sweep_block(claims)
    )

    chunks.append("/-! ## Coupling channel — `sec:hierarchy`, the redundant-message sweep -/")
    for name, label, comment in sorted(COUPLING_SCALARS):
        chunks.append(_decl(name, lit(label, claims[label]), comment, label))
    chunks.append(
        "/-- Size of the coupling alpha grid (ledger: `coupling_grid_size`). Only three\n"
        "of these points have recorded accuracies. -/\n"
        "def couplingGridSize : Nat := "
        f"{nat_value('coupling_grid_size', claims['coupling_grid_size'])}"
    )
    chunks.append(
        "/-- The recorded coupling points, ordered by mixing weight.\n\n"
        "SCOPE — read this before citing the theorems about it:\n\n"
        + "\n".join(_wrap(COUPLING_SCOPE, indent=""))
        + " -/\n"
        "def couplingRecorded : List Rat :=\n" + _coupling_block(claims)
    )

    chunks.append(
        "/-! ## Stage gate triples — the rows `scripts/audit_gates.py` classifies\n\n"
        "Each row is (baseline, typed comparator, model). The audit sets the THRESHOLD to the\n"
        "baseline for every row, so the threshold is emitted as an alias rather than\n"
        "as a second literal pretending to be its own measurement. The retained\n"
        "`*Ceiling` Lean identifiers are historical API names: their values are typed\n"
        "comparators and are not necessarily bounds. -/"
    )
    for gate in GATES:
        prefix = gate["prefix"]
        scope = "\n".join(_wrap(gate["scope"], indent=""))
        chunks.append(f"/-! ### `{gate['gate_id']}`\n\n{scope} -/")
        chunks.append(
            _decl(
                f"{prefix}Baseline",
                lit(gate["baseline"], claims[gate["baseline"]]),
                f"`{gate['gate_id']}` trivial baseline",
                gate["baseline"],
            )
        )
        chunks.append(
            _decl(
                f"{prefix}Ceiling",
                lit(gate["comparator"], claims[gate["comparator"]]),
                f"`{gate['gate_id']}` typed comparator (legacy `Ceiling` identifier)",
                gate["comparator"],
            )
        )
        chunks.append(
            _decl(
                f"{prefix}Model",
                lit(gate["model"], claims[gate["model"]]),
                f"`{gate['gate_id']}` shipped model",
                gate["model"],
            )
        )
        chunks.append(
            f"/-- `{gate['gate_id']}` threshold. `audit_gates.py` sets this row's\n"
            "threshold to its baseline, so this is an alias, not a new number. -/\n"
            f"def {prefix}Threshold : Rat := {prefix}Baseline"
        )

    chunks.append(
        "/-! ## Declared gates and information-theoretic bounds\n\n"
        "The historical thresholds Stages 1, 3 and 4 originally declared, plus the\n"
        "bounds they are judged against. `stage4DeclaredGate` is the interesting one:\n"
        "the same bar classifies differently on the two Stage 4 populations, and both\n"
        "populations are recorded. -/"
    )
    for name, label, comment in sorted(DECLARED):
        chunks.append(_decl(name, lit(label, claims[label]), comment, label))

    chunks.append("end AILM")
    return "\n\n".join(chunks) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT, help="output path")
    parser.add_argument(
        "--ledger", type=Path, default=_PROJECT_ROOT / "data" / "claims.json", help="ledger path"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if the target is not byte-identical to a fresh run",
    )
    args = parser.parse_args()

    try:
        text = render(load_claims(args.ledger))
    except GeneratorError as exc:
        print(f"REFUSING TO GENERATE: {exc}", file=sys.stderr)
        return 1

    if args.check:
        if not args.out.exists():
            print(f"{args.out} does not exist", file=sys.stderr)
            return 1
        current = args.out.read_text(encoding="utf-8")
        if current != text:
            print(
                f"{args.out} is stale: regenerate with "
                f"`.venv/bin/python scripts/generate_lean_ledger.py`",
                file=sys.stderr,
            )
            return 1
        print(f"{args.out} is up to date ({len(text)} bytes).")
        return 0

    atomic_write_text(args.out, text)
    print(f"Wrote {args.out} ({len(text)} bytes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
