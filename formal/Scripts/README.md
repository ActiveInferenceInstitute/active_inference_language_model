# `formal/Scripts/` — the axiom-report generator

Produces `../axiom_report.txt`: the `#print axioms` verdict for every named
declaration in the library. This directory sits **outside** the `AILM/` library
root, so `lake build` never compiles it — the report is a separate, explicit step.

## Quick start

The report workflow requires Bash, the pinned Lake/Lean toolchain, the project's
Python interpreter, and ripgrep (`rg`) on `PATH`. `AILM_LAKE` and `AILM_PYTHON`
can select explicit tool locations. The Lean library itself retains its
dependency-free `lake build` contract.

```bash
# From formal/
./Scripts/axiom_report.sh
```

The script generates the declaration driver, builds, and stages the report. It
publishes only a successful Lean run with verdict lines. A failure prints the
diagnostics and preserves the previous report; it does not make that previous
report current. The recorder independently checks freshness before recording.

Failure modes are distinct, and a tool failure never reads as an empty report:

- `lake` or `rg` missing exits 2 before any work — `command -v` proves presence,
  not operation, so the verdict step still catches an `rg` that runs and fails;
- a failing `rg` exits 4, naming the status it died with;
- a successful Lean run whose output carries no verdict lines exits 3 — the only
  path that refuses as vacuity.

## Files

| File | What it is |
|---|---|
| `AxiomReport.lean` | Generated from the shared declaration scan; one `#print axioms` line per named declaration, grouped by module, plus the allowed-axiom policy |
| `axiom_report.sh` | Generates the driver, builds, and publishes only a successful, nonempty report |

## Key rules

- **Every declaration in `../AILM/` needs a generated line here.** Do not edit the
  driver by hand. `record_formal_claims.py --check-axiom-script` detects drift; the
  generator refuses unsupported declaration forms.
- **Coverage is enforced, not trusted.** `scripts/record_formal_claims.py` refuses
  to record unless the report's entries and the library's declarations are the same
  set, checked in both directions.
- **Allowed axioms: `propext`, `Classical.choice`, `Quot.sound`, `funext`.**
  `Classical.choice` is on the list because Lean core's own `Rat.add_comm` reports
  it; tightening past it is unsatisfiable.
- **Two names must never appear in a verdict:** the incomplete-proof axiom and
  `Lean.ofReduceBool`. Lean warns rather than errors on an unfinished proof, so a
  zero exit code alone would not catch the first.
- **`AxiomReport.lean` mentions the incomplete-proof tactic by name** in its policy
  comment, which is why the lexical placeholder scan covers `../AILM/` only.

## Common commands

```bash
# From formal/
./Scripts/axiom_report.sh
AILM_LAKE=/path/to/lake ./Scripts/axiom_report.sh   # a non-PATH toolchain

# From the repository root, after regenerating
.venv/bin/python scripts/record_formal_claims.py
```

## More information

See [`AGENTS.md`](AGENTS.md) for the rules in full and
[`../README.md`](../README.md) § "Axiom policy" for why the allowlist is what it is.
