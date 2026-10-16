# AGENTS.md — formal/

Operating rules for the Lean 4 layer. [`README.md`](README.md) is the full guide:
what is proved, what is deliberately not, the isomorphism table, the axiom policy,
and the conversion policy for the generated ledger bridge. This file is the short
set of rules an agent needs before touching anything here, and it does not restate
the README.

## Overview

A small, dependency-free Lean 4 tree that machine-checks the parts of AILM's
argument that are actually mathematical: the gate taxonomy, the sweep-shape
predicates, the filtering recursion, the Bonferroni arithmetic, and the Holm
step-down comparison that justifies keeping it. Everything else
is deliberately out of scope, and the exclusion list in the README is part of the
deliverable rather than an apology for it.

## Ownership boundary

| Path | Owner |
| --- | --- |
| `AILM/LedgerData.lean` | **Generated.** `scripts/generate_lean_ledger.py` is its only writer |
| `AILM/Core.lean`, `GateVerdict.lean`, `SweepShape.lean`, `Filtering.lean`, `Bonferroni.lean`, `Holm.lean` | Authored theory; every literal in them is a synthetic witness |
| `AILM/Instances.lean` | Authored, but every value it reasons about comes from `LedgerData` |
| `axiom_report.txt` | **Generated.** `Scripts/axiom_report.sh` is its only writer |
| `Scripts/AxiomReport.lean` | **Generated.** `scripts/record_formal_claims.py --generate-axiom-script` uses the shared declaration scan |
| `Controls/MustFail.lean` | The negative control; outside the lake target on purpose |
| `lean-toolchain`, `lakefile.toml`, `lake-manifest.json` | The pinned, dependency-free build |
| The `formal_*` ledger labels | `scripts/record_formal_claims.py` |

## Invariants (never violate)

| Invariant | Why |
| --- | --- |
| **`AILM/LedgerData.lean` is never hand-edited** | `tests/test_formal_isomorphism.py` requires a fresh generator run to reproduce it byte for byte, and a second independent oracle re-derives every literal from the ledger |
| **No ledger value is ever hand-typed into Lean** | There are exactly two kinds of number here — generated ledger values and synthetic witnesses — and never a third |
| **No `[[require]]` stanza in `lakefile.toml`** | No dependencies, not even Mathlib, so `lake build` is offline by construction |
| **`native_decide` is used nowhere** | It would add the `Lean.ofReduceBool` trusted-code axiom, trading kernel checking for compiler trust |
| **Every fact closed by `decide` is a pure order comparison** | `Rat` arithmetic does not kernel-reduce; arithmetic facts are closed by `grind` instead |
| **`AILM.lit` is the only literal constructor** | It emits `Rat.mk'` directly, so its literals are `decide`-reducible and an unreduced fraction is a compile error |
| **`Controls/MustFail.lean` must not compile** | Without it, a green `lake build` is an assertion nobody has made fail |
| **Every conditional theorem carries a satisfiability witness** | A conditional theorem with an unsatisfiable hypothesis compiles clean and proves nothing — a defect this project has shipped |
| **No identifier or comment in `AILM/` contains a placeholder tactic's name as a word** | The scan is word-boundary anchored and deliberately reads comments too |
| **Declaration keywords stay off the left margin of prose** | `record_formal_claims.py` finds declarations with column-anchored patterns; a wrapped doc comment starting at column zero reads as a declaration and fails coverage |
| **Allowed axioms: `propext`, `Classical.choice`, `Quot.sound`, `funext`** | Tightening past `Classical.choice` is unsatisfiable against Lean core's own `Rat` lemmas |

## File organization

| Path | What it is |
| --- | --- |
| `AILM/` | The library — see [`AILM/AGENTS.md`](AILM/AGENTS.md) |
| `AILM.lean` | The library root that imports every module |
| `Controls/` | The negative control — see [`Controls/AGENTS.md`](Controls/AGENTS.md) |
| `Scripts/` | The axiom-report tooling — see [`Scripts/AGENTS.md`](Scripts/AGENTS.md) |
| `axiom_report.txt` | Generated `#print axioms` output for every named declaration |
| `lakefile.toml` | The build target; deliberately dependency-free |
| `lean-toolchain` | The pinned toolchain (`leanprover/lean4:v4.28.0`) |
| `lake-manifest.json` | The empty dependency manifest that pin implies |

## The change order that keeps the bridge honest

Changing a recorded value that the Lean reads is two commands in one order:

```bash
# From the repository root
.venv/bin/python scripts/generate_lean_ledger.py
.venv/bin/python scripts/record_formal_claims.py
```

Regenerate first, re-record second. Running them the other way records counts for a
tree that no longer exists.

## Adding a theorem, or a whole module

A new named `theorem` or `instance` under `AILM/` is not finished when it
compiles. `record_formal_claims.py` checks axiom-report coverage in **both**
directions, so a declaration with no `#print axioms` line makes the recorder
refuse. That is by design: a theorem the report does not cover is one the axiom
gate keeps reporting clean for. The order is:

```bash
# Add the declaration; the report script regenerates its driver and builds.
# From the repository root:
(cd formal && ./Scripts/axiom_report.sh)
.venv/bin/python scripts/record_formal_claims.py
```

A new module additionally needs the `import` in `AILM.lean`, a row in the module
tables of `README.md`, `AILM/README.md` and `../docs/formal.md`, and its own
entry in the scope list. What it does **not** prove is as much of the
deliverable as what it does.

## Testing

```bash
# From the repository root
(cd formal && lake build && ./Scripts/axiom_report.sh)
.venv/bin/python scripts/generate_lean_ledger.py --check
.venv/bin/python -m pytest tests/test_formal_isomorphism.py -q --no-cov
```

`tests/test_formal_isomorphism.py` skips its Lean-dependent checks cleanly when
`lake` is absent, naming the pinned toolchain in the skip reason. Its ledger-binding
checks need no toolchain and always run, so a machine without `elan` still cannot
desynchronise the Lean from the measurements.

## Best practices

- A theorem that restates a definition is decoration. Every theorem here carries a
  doc comment naming its manuscript object and stating what would falsify it.
- When a hypothesis is load-bearing, add a negative control showing the conclusion
  fails without it — and a companion theorem proving the *other* hypotheses still
  hold on that same input, so a multi-fault witness cannot read as evidence for
  whichever hypothesis it was filed under.
- Before filing something as "deferred, needs a lemma core lacks", check whether it
  is a claim about the chosen representation instead. The prediction half of
  `eq:filtering` sat on that list until the column-major representation made it two
  one-line inductions.

## See Also

- [`README.md`](README.md) — the full guide, the isomorphism table, and the scope list
- [`../docs/formal.md`](../docs/formal.md) — what is and is not formalized, from the documentation side
- [`../docs/formalism_map.md`](../docs/reference/formalism_map.md) — every labeled equation mapped to code and test
- [`../AGENTS.md`](../AGENTS.md) — project invariants, including the generated-file rule for `LedgerData.lean`
