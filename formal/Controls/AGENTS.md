# AGENTS.md — formal/Controls/

Operating rules for the Lean negative control. Quick reference:
[`README.md`](README.md). The layer-wide guide is [`../README.md`](../README.md).

## Overview

`MustFail.lean` states four propositions the library proves false. The toolchain
must reject it. The file exists because a passing build is not evidence unless
something is known to fail — the same reason `ControlHarness` raises
`GateVacuousError` when a positive and a negative control score identically.

## Ownership boundary

| Artifact | Owner |
| --- | --- |
| `MustFail.lean` | Hand-written; changed only when a theorem it names changes |
| Its invocation and the diagnostic assertions | `tests/test_formal_isomorphism.py::test_the_negative_control_does_not_compile` |
| The Python-side analogue | `verify_negative_control_fails` in `src/ailm/control.py` |

## Invariants (never violate)

| Invariant | Why |
| --- | --- |
| **`MustFail.lean` must not compile** | It is the only thing standing between "green build" and "green build that could not have been red" |
| **This directory stays outside the `AILM/` library root** | `lake build` must never see it. It is invoked directly by the test |
| **Never repair a statement here to make something pass** | A statement compiling is the alarm, not the bug |
| **The control is asserted on rejection text, not exit status** | A nonzero exit also results from a typo, a missing import, a wrong path, or a broken toolchain |
| **Each statement is false for a different reason** | An unrelated single failure would break all of them identically, which the diagnostics — not the exit code — distinguish |
| **At least one statement is refuted on recorded ledger values** | `control_coupling_interior` is refuted on `LedgerData.couplingRecorded`. Without it the controls show only that `decide` can reject invented numbers, which would leave every ledger-facing `by decide` in `AILM/Instances.lean` unattested |

## The four statements

| Declaration | Refuted by |
|---|---|
| `control_equality_passes` | `passes_irrefl` — equality does not clear a strict bar |
| `control_degenerate_bracketed` | `degenerate_no_gate` — a population whose baseline equals its ceiling admits no bracketed verdict |
| `control_both_shapes` | `interior_excludes_no_gain` — the pre-registered shape classes are mutually exclusive |
| `control_coupling_interior` | the recorded coupling accuracies themselves |

## Usage

```bash
# From the repository root
.venv/bin/python -m pytest \
  "tests/test_formal_isomorphism.py::test_the_negative_control_does_not_compile" \
  -v --no-cov
```

The test skips cleanly when `lake` is absent, naming the pinned toolchain in the
skip reason.

## Maintenance

When a theorem this file leans on is renamed or restated, update the corresponding
control in the same change and re-run the test. A control that no longer refers to a
live theorem fails for the wrong reason, which the diagnostic assertions catch —
but only after someone reads the failure.

Adding a fifth statement is fine; removing one is a scope decision, because the
diagnostic count the test requires moves with it.

## See Also

- [`README.md`](README.md) — quick reference and what each statement covers
- [`../README.md`](../README.md) — the layer's full guide
- [`../AGENTS.md`](../AGENTS.md) — layer-wide operating rules
- [`../AILM/AGENTS.md`](../AILM/AGENTS.md) — the library these statements are refuted against
- [`../../docs/testing_philosophy.md`](../../docs/testing_philosophy.md) — why every gate here carries a control
