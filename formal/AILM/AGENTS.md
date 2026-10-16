# AGENTS.md — formal/AILM/

Operating rules for the Lean library. Quick reference: [`README.md`](README.md).
The layer-wide guide — what is proved, what is deliberately not, the isomorphism
table, and the axiom policy — is [`../README.md`](../README.md), which this file
does not restate.

## Overview

Seven modules of plain Lean 4 core. Six are authored theory plus their witnesses;
one is generated from the claims ledger. The split between those two categories is
the point of the directory: a proof here can never be repaired by adjusting a
number, because the numbers are not editable from inside Lean.

## Ownership boundary

| Module | Status | Written by |
|---|---|---|
| `Core.lean` | authored | hand; definition site only, no theorems |
| `GateVerdict.lean` | authored | hand; synthetic witnesses only |
| `SweepShape.lean` | authored | hand; synthetic witnesses only |
| `Filtering.lean` | authored | hand; synthetic witnesses only |
| `Bonferroni.lean` | authored | hand; synthetic witnesses only |
| `LedgerData.lean` | **GENERATED** | `scripts/generate_lean_ledger.py`, from `data/claims.json` |
| `Instances.lean` | authored | hand — *structure* only: metric direction, claimed predicate, expected verdict |

## Invariants (never violate)

| Invariant | Why |
| --- | --- |
| **Never hand-edit `LedgerData.lean`** | A fresh generator run must reproduce it byte for byte, and a second independent oracle re-derives every literal from the ledger. A hand edit fails both; a stale file fails the first |
| **No measured number is typed into any other module** | Recorded values reach the theory only through `LedgerData`. Confusing a synthetic witness with a ledger value is the failure this layer exists to prevent |
| **`Instances.lean` holds structure, never numbers** | Metric direction, the predicate a sweep is claimed to satisfy, and the verdict a triple must classify to are properties of the design, not of a run |
| **`AILM.lit` is the only literal constructor in the tree** | Duplicating it collides at the root import, because auto-bound tactic proofs lift to top-level names |
| **Every `decide`d fact is a pure order comparison** | `Rat` arithmetic gets stuck at `Rat.add` in the kernel; `Rat.normalize` is opaque. Arithmetic facts are closed by `grind` |
| **`native_decide` appears nowhere** | It would add the `Lean.ofReduceBool` trusted-code axiom |
| **Every conditional theorem has a satisfiability witness** | One instance proving the hypothesis is inhabited, and one firing the theorem on it. A conditional theorem with an unsatisfiable hypothesis compiles clean and proves nothing |
| **Every decidable predicate is exhibited satisfiable *and* refutable** | Otherwise a `Decidable` instance may be deciding a constant, which happened here to two of the tree's instances |
| **No placeholder-tactic name as a word, in code or comment** | Lean warns rather than errors on an unfinished proof, so the lexical scan is a real gate; it reads comments on purpose |
| **Declaration keywords never start a line of prose** | `scripts/record_formal_claims.py` matches `^theorem` and `^instance`. A wrapped doc comment landing at column zero becomes a phantom declaration and fails the coverage check — loudly, at record time |
| **Every module reachable from `../AILM.lean`** | An unimported module escapes the build, the axiom report, and the coverage check together |

## The witness-count convention

The per-module counts in [`../README.md`](../README.md)'s "What is proved" table are
hand-maintained under one rule: the first number is the module's theory theorems,
the second its *named* witness theorems — the `theorem` declarations under that
module's `## Satisfiability witnesses` marker. Anonymous `example` witnesses count
in neither. `Filtering.lean` has two witness blocks (one per half of
`eq:filtering`), and both count toward its witness number.

The whole-tree totals are not hand-maintained: they are ledger tags written by
`scripts/record_formal_claims.py` from a real build.

## Adding a theorem

1. State it, with a doc comment naming its manuscript object and **what would
   falsify it**. A theorem that excludes no possible world is decoration.
2. If it is conditional, add the satisfiability witness pair under the module's
   witness marker.
3. If a hypothesis is load-bearing, add a negative control that drops exactly that
   hypothesis, plus a companion theorem proving the remaining hypotheses still hold
   on the same input. Without the companion, a multi-fault witness reads as evidence
   for whichever hypothesis it was filed under.
4. Rebuild, regenerate the axiom report, and re-record:

```bash
# From the repository root
(cd formal && lake build && ./Scripts/axiom_report.sh)
.venv/bin/python scripts/record_formal_claims.py
```

`record_formal_claims.py` refuses to record unless the report's entries and the
declarations in `AILM/*.lean` are the same set, checked in both directions.

## Changing a recorded value

Regenerate before re-recording — the other order records counts for a tree that no
longer exists:

```bash
# From the repository root
.venv/bin/python scripts/generate_lean_ledger.py
.venv/bin/python scripts/record_formal_claims.py
```

If a remeasurement moves a value across a threshold, the proof stops compiling.
That is the intended alarm, not a maintenance burden.

## Testing

```bash
# From the repository root
.venv/bin/python scripts/generate_lean_ledger.py --check
.venv/bin/python -m pytest tests/test_formal_isomorphism.py -q --no-cov
```

The ledger-binding checks run without a toolchain; the Lean-dependent ones skip
cleanly when `lake` is absent, naming the pinned toolchain in the skip reason.

## See Also

- [`README.md`](README.md) — module map and quick commands
- [`../README.md`](../README.md) — what is proved, the isomorphism table, the scope list
- [`../AGENTS.md`](../AGENTS.md) — layer-wide operating rules
- [`../Controls/AGENTS.md`](../Controls/AGENTS.md) — the negative control that keeps a green build meaningful
- [`../../docs/formalism_map.md`](../../docs/reference/formalism_map.md) — equation-to-code-to-test map
