/-
  AILM.Bonferroni — the finite union-bound arithmetic behind `sec:multiplicity`.

  `manuscript/10_verification.md` § "One family, one corrected threshold"
  declares one multiplicity family, enumerated executably by
  `SIGNIFICANCE_FAMILY` in `src/ailm/statistics.py`, and judges every member
  against the Bonferroni per-test threshold α/n.

  ## What this file does NOT prove — read this before citing it

  This module proves **arithmetic**, not familywise error control. There is no
  probability measure here, no event algebra, and no measure-theoretic union
  bound: those are out of reach without Mathlib, and Mathlib is out of scope by
  construction (see README.md).

  What `sum_le_length_mul` and `bonferroni_union_bound` actually say is:

  > if each recorded p-value is at most `t`, and `n · t ≤ α`, then their sum is
  > at most `α`.

  That is the *arithmetic core* of the Bonferroni correction and nothing more.
  Claiming it establishes FWER control would be exactly the "necessary condition
  is not sufficient" defect `sec:gate_design` diagnoses elsewhere in this
  project. The probabilistic step — from "the sum of the per-test levels is at
  most α" to "the probability of at least one false rejection is at most α" —
  is Boole's inequality applied to the rejection events, and it is assumed here,
  not proved.
-/
import AILM.Core

namespace AILM

/-- The Bonferroni per-test threshold α/n.

Definition only. No instance fact ever evaluates this: `Rat` division does not
kernel-reduce, so a `decide`d statement about ledger p-values must compare
against a threshold constant supplied as an exact reduced literal rather than
computed inside Lean. -/
def bonferroniThreshold (alpha : Rat) (n : Nat) : Rat := alpha / (n : Rat)

/--
**B1.** A family of values each bounded by `t` has total at most `n · t`.

The finite counting step of the union bound.

FALSIFIER: a family of per-test bounds whose total exceeds the count times the
bound — impossible for `≤` on an ordered field, which is precisely why this
step is safe to assume and worth pinning: every remaining risk in the
multiplicity argument sits in the *probabilistic* step, not here.

Note the degenerate case: on the empty family both sides are `0`, so B1 alone is
satisfied vacuously by an empty family. `witnessFamily_bounded` below supplies a
nonempty one so the theorem is exercised where it has content.
-/
theorem sum_le_length_mul (t : Rat) : ∀ ps : List Rat, (∀ p ∈ ps, p ≤ t) →
    ps.sum ≤ (ps.length : Rat) * t
  | [], _ => by simp; grind
  | p :: rest, h => by
      have hrest := sum_le_length_mul t rest (fun q hq => h q (by simp [hq]))
      have hp : p ≤ t := h p (by simp)
      simp only [List.sum_cons, List.length_cons]
      push_cast
      grind

/--
**B2.** The division-free Bonferroni statement.

Manuscript: `sec:multiplicity`, "At familywise α = 0.01 the Bonferroni per-test
threshold is 0.000625, and 8 of the 16 declared tests cross it."

FALSIFIER: a correction that accepts a per-test threshold too large for its
family size — i.e. an `(n, t, α)` triple with `n · t > α` still being reported
as a valid Bonferroni configuration. The `hscale` hypothesis is what forbids it,
and `scaleHypothesisIsLoadBearing` below shows the conclusion genuinely fails
when it is dropped.

The division-free form is the one to use against ledger constants. The recorded
`stat_bonferroni_alpha` is a *rounded* float, so `n × recorded` is not exactly
α; discharging `hscale` against the recorded value rather than against an exact
α/n literal would quietly assert something false.
-/
theorem bonferroni_union_bound (alpha t : Rat) (ps : List Rat)
    (hscale : (ps.length : Rat) * t ≤ alpha) (h : ∀ p ∈ ps, p ≤ t) : ps.sum ≤ alpha :=
  Rat.le_trans (sum_le_length_mul t ps h) hscale

/--
**B2a.** `n` copies of the Bonferroni threshold recover α exactly, for `n > 0`.

FALSIFIER: an off-by-one in the family count, or a threshold definition that is
not α/n. This is the equation the *divided* form of the correction rests on, and
it is stated separately because it is the step `grind` cannot do on its own — it
is nonlinear, and needs `Rat.mul_inv_cancel` by hand.
-/
theorem bonferroni_threshold_scales (alpha : Rat) (n : Nat) (hn : 0 < n) :
    (n : Rat) * bonferroniThreshold alpha n = alpha := by
  have hne : ((n : Rat)) ≠ 0 := by
    have : (0 : Rat) < (n : Rat) := by exact_mod_cast hn
    grind
  have hcancel := Rat.mul_inv_cancel ((n : Rat)) hne
  unfold bonferroniThreshold
  rw [Rat.div_def, ← Rat.mul_assoc, Rat.mul_comm ((n : Rat)) alpha, Rat.mul_assoc, hcancel]
  grind

/--
**B3.** Bonferroni in the divided form the manuscript writes.

Manuscript: `sec:multiplicity`, "counting the three nested subsample re-analyzes
as one member instead of three gives a family of fourteen and a corrected
threshold of 0.01/14".

FALSIFIER: the same as B2 — a family whose members each clear α/n but whose
total exceeds α. `hn : 0 < ps.length` is required and non-removable: at `n = 0`
the threshold `α/0` is `0` in `Rat`, and the statement would be about a
different object than the correction.
-/
theorem bonferroni_union_bound_div (alpha : Rat) (ps : List Rat) (hn : 0 < ps.length)
    (h : ∀ p ∈ ps, p ≤ bonferroniThreshold alpha ps.length) : ps.sum ≤ alpha :=
  bonferroni_union_bound alpha _ ps
    (by rw [bonferroni_threshold_scales alpha ps.length hn]; exact Rat.le_refl) h

/-! ## Satisfiability witnesses

B1, B2, B2a and B3 are all conditional. Each hypothesis is exhibited inhabited
on a **nonempty** family — the empty family satisfies B1 and B2 vacuously, so a
witness that did not exclude it would prove nothing.

The p-values below are **synthetic**, not recorded. They reproduce the declared
family's *size* (sixteen members) and the exact per-test threshold shape
(α = 1/100, n = 16, so t = 1/1600) without hand-typing a recorded p-value into
Lean; the ledger-backed survivor and non-survivor instances belong in the
generated `AILM/LedgerData.lean` + `AILM/Instances.lean` pair. -/

/-- Sixteen distinct synthetic p-values, each strictly below the exact
Bonferroni per-test threshold `1/1600` for a family of sixteen at α = 1/100. -/
def witnessFamily : List Rat :=
  [1/20000, 1/15000, 1/12000, 1/10000, 1/9000, 1/8000, 1/7000, 1/6000,
   1/5500, 1/5000, 1/4500, 1/4000, 1/3500, 1/3000, 1/2500, 1/2000]

theorem witnessFamily_length : witnessFamily.length = 16 := by decide

/-- B1's and B2's per-member hypothesis is inhabited, on a nonempty family. -/
theorem witnessFamily_bounded : ∀ p ∈ witnessFamily, p ≤ (1 : Rat) / 1600 := by
  intro p hp
  simp only [witnessFamily, List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> grind

/-- B2's `hscale` hypothesis is inhabited, and at the exact boundary
`16 · (1/1600) = 1/100` rather than with slack — the tightest configuration the
correction allows. -/
theorem witnessFamily_scale :
    ((witnessFamily.length : Nat) : Rat) * ((1 : Rat) / 1600) ≤ 1 / 100 := by
  rw [witnessFamily_length]
  push_cast
  grind

/-- B2 fires on a nonempty family with both hypotheses discharged. -/
theorem witnessFamily_bound : witnessFamily.sum ≤ (1 : Rat) / 100 :=
  bonferroni_union_bound _ _ _ witnessFamily_scale witnessFamily_bounded

/-- B3's positivity hypothesis is inhabited. -/
theorem witnessFamily_nonempty : 0 < witnessFamily.length := by decide

/-- B2a is inhabited at the family's own size: sixteen copies of the corrected
threshold recover the familywise level exactly. -/
example : (16 : Rat) * bonferroniThreshold (1 / 100) 16 = 1 / 100 :=
  bonferroni_threshold_scales _ 16 (by decide)

/-- The witness family's corrected threshold, as an exact reduced literal.

This is the bridge B3 needs. `Rat` division does not kernel-reduce, so B3's
per-member hypothesis cannot be `decide`d against `bonferroniThreshold`
directly; it has to be rewritten to a literal first, and that rewrite is an
arithmetic fact rather than an order comparison.

FALSIFIER: a `bonferroniThreshold` that is not α/n — the same falsifier as B2a,
instantiated at the family size this module's witnesses use. -/
theorem witnessFamily_corrected_threshold :
    bonferroniThreshold (1 / 100) witnessFamily.length = 1 / 1600 := by
  rw [witnessFamily_length]
  unfold bonferroniThreshold
  push_cast
  grind

/-- **B3's per-member hypothesis is inhabited**, stated against the *divided*
threshold rather than against a hand-supplied literal.

This witness is the point of the block. Until it existed, `witnessFamily_bounded`
bounded the family against the literal `1/1600` and nothing in the tree ever
supplied B3's `∀ p ∈ ps, p ≤ bonferroniThreshold α ps.length`, so B3 was a
conditional theorem no instance anywhere satisfied — which compiles clean and
proves nothing. -/
theorem witnessFamily_bounded_by_corrected_threshold :
    ∀ p ∈ witnessFamily, p ≤ bonferroniThreshold (1 / 100) witnessFamily.length := by
  intro p hp
  rw [witnessFamily_corrected_threshold]
  exact witnessFamily_bounded p hp

/-- B3 fires on a nonempty family with both of its hypotheses discharged, so the
divided form of the correction — the form `sec:multiplicity` actually writes — is
exercised rather than merely stated. -/
theorem witnessFamily_bound_div : witnessFamily.sum ≤ (1 : Rat) / 100 :=
  bonferroni_union_bound_div _ _ witnessFamily_nonempty
    witnessFamily_bounded_by_corrected_threshold

/-- **Negative control.** B2's `hscale` hypothesis is load-bearing: here every
member clears the per-test threshold `t = 1/100`, yet the family total exceeds
`α = 1/100`, because `2 · t > α`. Dropping `hscale` would make B2 false. -/
theorem scaleHypothesisIsLoadBearing :
    (∀ p ∈ [(1 : Rat) / 100, 1 / 100], p ≤ (1 : Rat) / 100)
      ∧ ¬ (([(1 : Rat) / 100, 1 / 100] : List Rat).sum ≤ 1 / 100) := by
  constructor
  · intro p hp
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl <;> grind
  · simp only [List.sum_cons, List.sum_nil]
    grind

/-- **Negative control.** B3's `0 < ps.length` hypothesis is load-bearing:
`bonferroniThreshold α 0 = α / 0 = 0` in `Rat`, so at `n = 0` the "threshold"
collapses to zero and B2a's conclusion fails for any nonzero α. -/
theorem zeroFamilyThresholdCollapses : bonferroniThreshold (1 / 100) 0 = 0 := by
  unfold bonferroniThreshold
  simp [Rat.div_def]

example : (0 : Rat) * bonferroniThreshold (1 / 100) 0 ≠ 1 / 100 := by
  rw [zeroFamilyThresholdCollapses]
  grind

end AILM
