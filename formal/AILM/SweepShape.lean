/-
  AILM.SweepShape — the two pre-registered shape classes of `sec:hierarchy`.

  `manuscript/09_hierarchy.md` states a two-clause working hypothesis whose
  first clause is diagnosed by the *shape* of a mixing-weight sweep:

  * a **monotone decline** in the mixing weight says the message is redundant —
    "there is no weight at which saying it more loudly helps";
  * an **interior optimum** says the message carries something the receiver
    lacks — "an interior optimum being the signature of novel rather than
    redundant information".

  `fig:channel_comparison` draws both classes as pre-registered bands and its
  generator refuses to draw if the recomputed verdict contradicts the band. That
  refusal is only meaningful if the two classes are genuinely exclusive. This
  module proves they are.

  Scores are a `List Rat` indexed so that **index 0 is the anchor** (mixing
  weight 0). Working on the score list alone rather than on weight/score pairs
  is deliberate: it needs no well-formedness hypothesis about distinct or sorted
  weights, and the exclusion proof stays two lines.

  What is deliberately NOT here: any theorem asserting that interior-optimum
  shape *implies* novelty, or *predicts* channel benefit. That is the empirical
  working hypothesis of `sec:hierarchy`, supported by two measured channels of
  each kind in one language. Lean can pin the logical relationship between the
  shape predicates; dressing an inductive conjecture in theorem syntax would be
  decoration with a proof attached.
-/
import AILM.Core

namespace AILM

/-- A mixing-weight sweep's accuracies, indexed with the uncoupled anchor
(weight 0) at index 0. -/
abbrev Scores := List Rat

/-- `xs` has a strict argmax at index `i`: every other in-range index scores
strictly lower. Strictness matters — a tie would make the "optimum" ambiguous
and the interior/boundary distinction undecidable in practice. -/
def PeakAt (xs : Scores) (i : Nat) : Prop :=
  ∀ j, j < xs.length → j ≠ i → xs.getD j 0 < xs.getD i 0

instance instDecidablePeakAt (xs : Scores) (i : Nat) : Decidable (PeakAt xs i) := by
  unfold PeakAt; infer_instance

/-- The **interior optimum** class: a strict argmax at a non-endpoint index.
`0 < i` excludes the anchor; `i + 1 < xs.length` excludes the final weight.

Manuscript: `sec:hierarchy`, "the optimum is interior: at weight 0.9 the effect
*reverses*"; and the `fig:channel_comparison` band recomputed as "the
interior-vs-boundary argmax over the recorded lexical weights". -/
def InteriorOptimum (xs : Scores) : Prop :=
  ∃ i, 0 < i ∧ i + 1 < xs.length ∧ PeakAt xs i

/-- Decidability by bounding the existential. The added `i < xs.length` is
implied by `i + 1 < xs.length`, so this is a decision procedure for the
statement as written, not for a weaker one. -/
instance instDecidableInteriorOptimum (xs : Scores) : Decidable (InteriorOptimum xs) :=
  decidable_of_iff (∃ i, i < xs.length ∧ 0 < i ∧ i + 1 < xs.length ∧ PeakAt xs i)
    (by
      constructor
      · rintro ⟨i, _, h1, h2, h3⟩
        exact ⟨i, h1, h2, h3⟩
      · rintro ⟨i, h1, h2, h3⟩
        exact ⟨i, by omega, h1, h2, h3⟩)

/-- The **no interior gain** class: no non-anchor weight beats the anchor.

Manuscript: `fig:channel_comparison`'s generator recomputes exactly this —
"whether any recorded weight improves on the uncoupled anchor".

The `i < xs.length` bound is written first on purpose: with the bound in any
other position the `Nat.decidableBallLT` instance fails to synthesize. -/
def NoInteriorGain (xs : Scores) : Prop :=
  ∀ i, i < xs.length → 0 < i → xs.getD i 0 < xs.getD 0 0

instance instDecidableNoInteriorGain (xs : Scores) : Decidable (NoInteriorGain xs) := by
  unfold NoInteriorGain; infer_instance

/-- The **monotone decline** class: every index scores strictly below every
earlier index.

Manuscript: `sec:hierarchy`, "Accuracy fell *monotonically* in the mixing
weight, which is what redundancy looks like". Same bound-first ordering. -/
def MonotoneDecline (xs : Scores) : Prop :=
  ∀ j, j < xs.length → ∀ i, i < j → xs.getD j 0 < xs.getD i 0

instance instDecidableMonotoneDecline (xs : Scores) : Decidable (MonotoneDecline xs) := by
  unfold MonotoneDecline; infer_instance

/--
**S1.** Monotone decline implies no interior gain.

Manuscript: `sec:hierarchy`, the redundant-channel half — "there is no weight at
which saying it more loudly helps".

FALSIFIER: a monotone-declining sweep in which some non-anchor weight
nonetheless beats the anchor. The implication is *strict*, not an equivalence:
`noGainButNotMonotone` below exhibits a sweep with no interior gain that is not
monotone, so this theorem carries content rather than renaming one predicate to
the other.
-/
theorem monotone_decline_no_interior_gain (xs : Scores) (h : MonotoneDecline xs) :
    NoInteriorGain xs :=
  fun i hlen hi => h i hlen 0 hi

/--
**S2.** Interior optimum excludes no-interior-gain. The two pre-registered shape
classes are mutually exclusive.

Manuscript: `sec:hierarchy`'s two-clause working hypothesis, and
`fig:channel_comparison`'s pre-registered bands — "*interior optimum* for the
novel channel, *no interior gain* for the redundant one".

FALSIFIER: a single score list satisfying both predicates. Such a list would
make the figure's band system ill-defined (one sweep could legitimately be drawn
in either band) and would demote the manuscript's shape diagnostic from
well-defined to merely suggestive. This is the load-bearing theorem of the
module.

NON-VACUITY: both predicates are inhabited and both are refutable — see the
witness block below.
-/
theorem interior_excludes_no_gain (xs : Scores) (h : InteriorOptimum xs) :
    ¬ NoInteriorGain xs := by
  obtain ⟨i, hi, hlen, hpeak⟩ := h
  intro hng
  have h1 := hpeak 0 (by omega) (by omega)
  have h2 := hng i (by omega) hi
  grind

/--
**S3.** Interior optimum excludes monotone decline.

Manuscript: the two bands of `fig:channel_comparison` cannot co-fire on one
sweep, which is what licenses the generator's refusal-to-draw check.

FALSIFIER: a monotone-declining sweep with a strict interior argmax. Corollary
of S1 composed with S2, stated separately because it is the form the figure
generator's contradiction check actually needs.
-/
theorem interior_excludes_monotone (xs : Scores) (h : InteriorOptimum xs) :
    ¬ MonotoneDecline xs :=
  fun hm => interior_excludes_no_gain xs h (monotone_decline_no_interior_gain xs hm)

/-! ## Satisfiability witnesses

Every predicate here is proved **both inhabited and refutable**, so neither S1
nor S2 is protected by a class nothing (or everything) belongs to.

The numbers below are **synthetic shape witnesses, not measurements**. They
reproduce the *shape* of the two recorded sweeps — a seven-point sweep with an
interior peak, and a three-point monotone decline — without hand-typing any
recorded value into Lean. Ledger-backed instances of these predicates belong in
the generated `AILM/LedgerData.lean` + `AILM/Instances.lean` pair, which is a
separate deliverable; no recorded number appears anywhere in this module. -/

/-- Synthetic seven-point sweep peaking at index 2 (interior). Shaped like the
lexical channel's recorded sweep; the values are not the recorded ones. -/
def interiorWitness : Scores :=
  [lit 1 2, lit 5 8, lit 3 4, lit 11 16, lit 5 8, lit 9 16, lit 1 2]

/-- Synthetic three-point monotone decline. Shaped like the coupling channel's
recorded anchors; the values are not the recorded ones. -/
def monotoneWitness : Scores :=
  [lit 3 4, lit 5 8, lit 1 2]

/-- Synthetic sweep with no interior gain that is **not** monotone: the anchor
still wins, but index 2 rises above index 1. This is what makes S1 a strict
implication rather than a renaming. -/
def noGainButNotMonotone : Scores :=
  [lit 3 4, lit 1 2, lit 5 8]

/-- `InteriorOptimum` is inhabited. -/
theorem interiorWitness_interior : InteriorOptimum interiorWitness := by decide

/-- ... and S2 then fires on it, non-vacuously. -/
theorem interiorWitness_not_no_gain : ¬ NoInteriorGain interiorWitness :=
  interior_excludes_no_gain _ interiorWitness_interior

/-- ... and so does S3. -/
theorem interiorWitness_not_monotone : ¬ MonotoneDecline interiorWitness :=
  interior_excludes_monotone _ interiorWitness_interior

/-- `MonotoneDecline` is inhabited. -/
theorem monotoneWitness_monotone : MonotoneDecline monotoneWitness := by decide

/-- ... so S1 fires non-vacuously. -/
theorem monotoneWitness_no_gain : NoInteriorGain monotoneWitness :=
  monotone_decline_no_interior_gain _ monotoneWitness_monotone

/-- `InteriorOptimum` is refutable: the monotone witness has its argmax at the
anchor, which is a boundary index. -/
theorem monotoneWitness_not_interior : ¬ InteriorOptimum monotoneWitness := by decide

/-- S1's converse fails: no interior gain does not imply monotone decline. -/
theorem noGainButNotMonotone_no_gain : NoInteriorGain noGainButNotMonotone := by decide

theorem noGainButNotMonotone_not_monotone : ¬ MonotoneDecline noGainButNotMonotone := by
  decide

/-- `NoInteriorGain` is refutable directly, not only via S2. -/
example : ¬ NoInteriorGain interiorWitness := by decide

/-- `MonotoneDecline` is refutable directly, not only via S3. -/
example : ¬ MonotoneDecline interiorWitness := by decide

/-- `PeakAt` is satisfiable at a boundary index too, so `InteriorOptimum`'s
`0 < i` and `i + 1 < xs.length` conjuncts are doing real work rather than being
implied by the peak condition. -/
example : PeakAt monotoneWitness 0 := by decide

/-- ... and `PeakAt` is refutable: index 1 is not the monotone witness's argmax.
`PeakAt` is the predicate `InteriorOptimum` is built from and it is the only one
in this module that was exhibited holding but never failing, which would leave
`instDecidablePeakAt` exercised in one direction only. -/
example : ¬ PeakAt monotoneWitness 1 := by decide

end AILM
