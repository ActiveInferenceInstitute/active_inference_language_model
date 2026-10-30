# `formal/AILM/` — the Lean library

The whole machine-checked layer. Plain Lean 4 core only: no Mathlib, no
dependencies, no network. `lake build` from `../` type-checks every module here.

## Quick start

```bash
# From the repository root
(cd formal && lake build)
```

## Modules

| Module | What it holds |
|---|---|
| `Core.lean` | `AILM.lit` — the single reduced-rational literal constructor. No theorems; a definition site |
| `GateVerdict.lean` | `eq:gate_satisfies` and `eq:gate_verdict`: the comparator typing and the gate taxonomy |
| `SweepShape.lean` | The two pre-registered sweep-shape classes of `sec:hierarchy`, and their mutual exclusion |
| `Filtering.lean` | Both halves of `eq:filtering` — the Bayes correction step and the prediction step |
| `Bonferroni.lean` | The finite union-bound arithmetic behind `sec:multiplicity` |
| `Holm.lean` | The step-down comparison in the Holm paragraph of `98_supplement_statistics.md`. Holm's rank-`i` bar `α/(m-i+1)` is never stricter than Bonferroni's `α/m`, is nondecreasing in the rank, and coincides with it at rank 1; on a **sorted** family the step-down walk rejects every member the fixed bar rejects, with a synthetic witness on which it rejects strictly more. Proves **no** familywise error control, and **not** that halting at the first failure is the right stopping rule |
| `PairedDifference.lean` | `eq:paired_difference` of `98_supplement_statistics.md`. The two forms the manuscript writes for the paired difference `δ` agree — the share-weighted `(2s − 1)(b + c)/n` equals `(c − b)/n` — and the statistic's sign and zero behavior are pinned. Arithmetic on the definition only: proves **no** sampling distribution, interval coverage, or McNemar-test validity |
| `LedgerData.lean` | **GENERATED** from `data/claims.json`. Recorded values, no theorems |
| `Instances.lean` | The recorded values run through the theory above. Structure hand-written, numbers never |

`../AILM.lean` is the root module that imports all of them, so `#print axioms`
in `../Scripts/AxiomReport.lean` sees every named declaration behind one import.

## Key rules

- **`LedgerData.lean` is generated. Never edit it.** `scripts/generate_lean_ledger.py`
  is its only writer, and `tests/test_formal_isomorphism.py` requires a fresh run to
  reproduce it byte for byte.
- **Two kinds of number, never a third.** `LedgerData.lean` holds recorded
  measurements; every literal in the theory modules is a synthetic witness chosen to
  exhibit a shape, and each witness block says so in place.
- **`AILM.lit` is the only literal constructor.** It emits `Rat.mk'`, whose
  `Nat.Coprime` field is discharged by `decide`, so an unreduced fraction like
  `lit 2 8` is a compile error rather than a silently wrong number.
- **Every `decide`d fact is an order comparison.** `Rat` arithmetic does not
  kernel-reduce; arithmetic facts use `grind`.
- **No placeholder-tactic name may appear as a word**, in code or in a comment.
  The scan is word-boundary anchored and deliberately reads comments.

## Common commands

```bash
# From the repository root
(cd formal && lake build && ./Scripts/axiom_report.sh)
.venv/bin/python scripts/generate_lean_ledger.py --check
.venv/bin/python -m pytest tests/test_formal_isomorphism.py -q --no-cov
```

## More information

See [`AGENTS.md`](AGENTS.md) for the rules in full, [`../README.md`](../README.md)
for what is proved and what is deliberately not, and
[`../../docs/formal.md`](../../docs/formal.md) for the documentation-side account.
