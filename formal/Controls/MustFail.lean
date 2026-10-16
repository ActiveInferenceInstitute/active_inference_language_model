/-
  NEGATIVE CONTROL — THIS FILE MUST NOT COMPILE.

  It is deliberately excluded from the `lake` target (`Controls/` sits outside
  the `AILM/` library root), so `lake build` never sees it. It is invoked
  through the diagnostic-validating pytest control, run from the repository
  root:

      .venv/bin/python -m pytest tests/test_formal_isomorphism.py::test_the_negative_control_does_not_compile -v --no-cov

  The test requires the intended refutation diagnostics and the ledger-backed
  diagnostic, not merely a nonzero Lean exit status. A raw shell negation would
  also accept a typo, missing import, wrong path, or broken toolchain.

  Why this exists: "lake build succeeded" is only evidence if the build is
  capable of failing. Without a control that the toolchain is asked to reject
  and does reject, every green result in `formal/` rests on an assertion nobody
  has made fail. This is the Lean-side analogue of
  `verify_negative_control_fails` in `src/ailm/control.py`.

  Each of the four statements below is false for a *different* reason, so the
  control cannot pass by accident on a single unrelated error (a typo or a
  missing import would break all four the same way, which is detectable by the
  error message rather than the exit code alone).

  The fourth is the one that covers the ledger bridge. The first three are
  refuted on synthetic witnesses, so on their own they would show only that
  `decide` can reject *invented* numbers. `control_coupling_interior` is
  refuted on `AILM.LedgerData.couplingRecorded` — real recorded accuracies —
  which is what makes the ledger-backed `by decide` results in
  `AILM/Instances.lean` evidence rather than assertion.
-/
import AILM

namespace AILM.Controls

/-- FALSE by `passes_irrefl`: a metric exactly on its bar does not pass. If this
compiles, the strictness of `eq:gate_satisfies` is not being enforced. -/
theorem control_equality_passes : passes Direction.higherIsBetter (lit 1 2) (lit 1 2) := by
  decide

/-- FALSE by `degenerate_no_gate`: a population whose baseline equals its ceiling
admits no sound threshold. If this compiles, the degenerate-population result of
`sec:gate_design` is not being checked. -/
theorem control_degenerate_bracketed :
    (classify Direction.higherIsBetter (lit 1 2)
      (some (lit 1 2, ComparatorKind.modelClassBound)) (lit 1 4)).isBracketed = true := by
  decide

/-- FALSE by `interior_excludes_no_gain`: the two pre-registered sweep shapes are
mutually exclusive. If this compiles, `fig:channel_comparison`'s band system is
not well-defined and S2 proves nothing. -/
theorem control_both_shapes : NoInteriorGain interiorWitness := by decide

/-- FALSE on **recorded data**: the coupling channel's recorded points have no
interior optimum (`couplingRecorded_not_interiorOptimum`). If this compiles,
`decide` is not actually evaluating the ledger literals, and every ledger-backed
result in `AILM/Instances.lean` is worthless — including the ones that read as
findings. -/
theorem control_coupling_interior : InteriorOptimum couplingRecorded := by decide

end AILM.Controls
