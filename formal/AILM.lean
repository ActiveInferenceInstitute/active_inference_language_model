/-
  AILM — root module of the formal layer.

  Imports every core-theory module so that `lake build` (default target `AILM`)
  type-checks the whole tree, and so that `#print axioms` in
  `Scripts/AxiomReport.lean` can see every named theorem behind one import.

  Module map:

  * `AILM.Core`        — the single `Rat` literal constructor (`lit`).
  * `AILM.GateVerdict` — `eq:gate_satisfies` / `eq:gate_verdict`, the gate taxonomy.
  * `AILM.SweepShape`  — `sec:hierarchy`, the two pre-registered sweep-shape classes.
  * `AILM.Filtering`   — `eq:filtering`, the Bayes correction step.
  * `AILM.Bonferroni`  — `sec:multiplicity`, the finite union-bound arithmetic.
  * `AILM.Holm`        : the step-down comparison of `98_supplement_statistics.md`.
    Holm's per-rank bar is never stricter than Bonferroni's, and on a sorted
    family Holm rejects every member Bonferroni rejects, sometimes strictly more.
    Rejection-count arithmetic only: it proves **no** familywise error control,
    the same caveat `AILM.Bonferroni` carries and for the same reason.
  * `AILM.PairedDifference` — `eq:paired_difference`, the paired accuracy
    difference. It proves the two forms the manuscript writes for `δ` agree and
    pins the statistic's sign and zero behavior. Arithmetic only: no sampling
    distribution, no interval coverage, and no McNemar-test validity.
  * `AILM.LedgerData`  — GENERATED from `data/claims.json`; the recorded
    numbers, nothing else.
  * `AILM.Instances`   — the recorded numbers run through the theory above.

  ## Where the numbers come from

  Exactly one module in this tree contains measured values: `AILM.LedgerData`,
  which is written by `scripts/generate_lean_ledger.py` and required by
  `tests/test_formal_isomorphism.py` to be byte-identical to a fresh
  regeneration. Every literal in the six theory modules is a **synthetic
  witness** — a number invented to exhibit a shape — and each witness block
  says so. The two kinds must never be confused: a synthetic witness proves a
  theorem is non-vacuous, a ledger value proves something about the world.
-/
import AILM.Core
import AILM.GateVerdict
import AILM.SweepShape
import AILM.Filtering
import AILM.Bonferroni
import AILM.Holm
import AILM.PairedDifference
import AILM.LedgerData
import AILM.Instances
