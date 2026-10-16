# `formal/Controls/` — the negative control

One file, `MustFail.lean`, and its entire job is to **not compile**.

"`lake build` succeeded" is only evidence if the build is capable of failing.
Without a file the toolchain is asked to reject and does reject, every green result
in `formal/` rests on an assertion nobody has made fail. This is the Lean-side
analogue of `verify_negative_control_fails` in `src/ailm/control.py`.

## Quick start

```bash
# From the repository root
.venv/bin/python -m pytest \
  "tests/test_formal_isomorphism.py::test_the_negative_control_does_not_compile" \
  -v --no-cov
```

The test asserts on the **rejection text**, not merely a nonzero exit status: it
requires the expected distinct refutation diagnostics and the ledger-backed one. A
raw shell negation would also accept a typo, a missing import, a wrong path, or a
broken toolchain.

## Files

| File | What it is |
|---|---|
| `MustFail.lean` | Four false statements, each false for a different reason |

## What the four cover

| Statement | Why it is false |
|---|---|
| equality passing a strict gate | `passes_irrefl` — a metric exactly on its bar does not pass |
| a degenerate population classifying as either bracketed verdict | `degenerate_no_gate` |
| a sweep in both shape classes at once | the shape classes are mutually exclusive |
| `InteriorOptimum couplingRecorded` | false on the **recorded** coupling accuracies |

The fourth is the one that covers the ledger bridge. The first three are refuted on
synthetic witnesses, so on their own they show only that `decide` can reject
*invented* numbers. Without a control refuted on real ledger literals, every
`by decide` in `../AILM/Instances.lean` would be an assertion that the kernel
evaluated something.

## Key rules

- **This directory sits outside the `AILM/` library root**, so `lake build` never
  sees it. Do not add it to `../lakefile.toml`.
- **Never "fix" a failure here.** A statement in this file compiling is the alarm.
- **Keep the four reasons distinct.** A typo or a missing import breaks all four the
  same way, which is why the test reads the diagnostics rather than the exit code.

## More information

See [`AGENTS.md`](AGENTS.md) for the rules in full and
[`../README.md`](../README.md) § "The negative control".
