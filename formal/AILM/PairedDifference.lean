/-
  AILM.PairedDifference — the paired-difference statistic of
  `eq:paired_difference` in `98_supplement_statistics.md`.

  The manuscript writes the paired accuracy difference two ways and treats them
  as one object: the discordant-win-share form `δ = (2s − 1)(b + c)/n`, with
  `s = c/(b + c)` the share of discordant cells won, and the difference form
  `δ = (c − b)/n`. This module proves the two agree and pins the sign and the
  zero behaviour of the statistic.

  ## What this file does NOT prove — read this before citing it

  This is arithmetic on the statistic's *definition*, not statistics. There is
  no probability space here, so it proves nothing about the sampling
  distribution of `δ`, nothing about the coverage of any interval built from it,
  and nothing about the validity of the McNemar test the discordant counts feed.
  `paired_difference_interval` in `src/ailm/benchmarks/scoring.py` builds a
  bootstrap interval; that interval's coverage is a property of the bootstrap,
  not of the identity proved here. The reason nothing probabilistic is in scope
  is the same as `AILM.Bonferroni`'s: a measure and Mathlib are both out of
  scope by construction (see ../README.md).

  Every literal below is a **synthetic** witness chosen to exhibit a shape, not
  a recorded value; the recorded counts live in the generated
  `AILM/LedgerData.lean`, where the ledger places them.
-/
import AILM.Core

namespace AILM

/-- The paired accuracy difference `δ = (c − b)/n`: the discordant cells won
(`c`) less the cells lost (`b`), over the `n` scored positions.

Definition only. At `n = 0` this is `0` in `Rat`, which is why every sign
statement below carries `0 < n` rather than assuming the division is
meaningful. -/
def pairedDelta (b c n : Nat) : Rat := ((c : Rat) - (b : Rat)) / (n : Rat)

/-- The discordant-win share `s = w/(w + l)` over the `w + l` discordant cells.

Definition only; the denominator is `0` when there are no discordant cells,
which is why PD1 carries `0 < c + b`. -/
def discordantShare (w l : Nat) : Rat := (w : Rat) / (((w + l : Nat)) : Rat)

/-- **PD1.** The manuscript's two forms of `eq:paired_difference` agree: the
share-weighted form `(2s − 1)(b + c)/n`, with `s = c/(b + c)`, equals the
difference form `(c − b)/n`.

Manuscript: `98_supplement_statistics.md`, `eq:paired_difference`.

FALSIFIER: a nonempty discordant set on which the two forms diverge — which
would make the manuscript's sentence "the two numbers are never interchangeable"
about one object, `δ`, false. The identity is what licenses using either form.

The hypothesis `0 < c + b` keeps the share `s` well defined; it is not
load-bearing for the *equality*, since at `c + b = 0` both sides are `0`, so
the theorem also holds degenerately. It is carried because the statement is
about a share, and a share over an empty cell set is not a number this project
reports. -/
theorem pairedDelta_discordant_form (b c n : Nat) (hbc : 0 < c + b) :
    pairedDelta b c n
      = (2 * discordantShare c b - 1) * (((c + b : Nat)) : Rat) / (n : Rat) := by
  unfold pairedDelta discordantShare
  rw [Rat.div_def, Rat.div_def, Rat.div_def]
  have hneq : (((c + b : Nat)) : Rat) ≠ 0 := by
    have : (0 : Rat) < (((c + b : Nat)) : Rat) := by exact_mod_cast hbc
    grind
  have hcancel : (((c + b : Nat)) : Rat) * ((((c + b : Nat)) : Rat))⁻¹ = 1 :=
    Rat.mul_inv_cancel (((c + b : Nat)) : Rat) hneq
  have hcore : (2 * ((c : Rat) * ((((c + b : Nat)) : Rat))⁻¹) - 1) * (((c + b : Nat)) : Rat)
      = (c : Rat) - (b : Rat) := by grind
  rw [hcore]

/-- **PD2.** A positive discordant imbalance on a nonempty scored population
makes `δ` positive: `b < c` and `0 < n` imply `0 < δ`, so the sign of the
statistic tracks the sign of `c − b`.

Manuscript: `98_supplement_statistics.md`, `eq:paired_difference`.

FALSIFIER: a scored population with more discordant wins than losses and a
non-positive `δ`. Both hypotheses are load-bearing — `pairedDelta_pos_drops_imbalance`
drops the imbalance at `n = 1`, and `pairedDelta_pos_drops_nonempty` drops the
scored-position hypothesis at `b < c` — and each control's companion proves the
remaining hypothesis still holds on that input. -/
theorem pairedDelta_pos (b c n : Nat) (hn : 0 < n) (h : b < c) :
    0 < pairedDelta b c n := by
  unfold pairedDelta
  rw [Rat.div_def]
  have hn' : (0 : Rat) < (n : Rat) := by exact_mod_cast hn
  have hnum : (0 : Rat) < (c : Rat) - (b : Rat) := by
    have : (b : Rat) < (c : Rat) := by exact_mod_cast h
    grind
  exact Rat.mul_pos hnum (Rat.inv_pos.mpr hn')

/-- **PD3.** A negative discordant imbalance on a nonempty scored population
makes `δ` negative — the mirror of PD2.

Manuscript: `98_supplement_statistics.md`, `eq:paired_difference`.

FALSIFIER: a scored population with more discordant losses than wins and a
non-negative `δ`. Both hypotheses are load-bearing, with a control and a
companion for each, exactly as in PD2. -/
theorem pairedDelta_neg (b c n : Nat) (hn : 0 < n) (h : c < b) :
    pairedDelta b c n < 0 := by
  unfold pairedDelta
  rw [Rat.div_def]
  have hn' : (0 : Rat) < (n : Rat) := by exact_mod_cast hn
  have hnum : (c : Rat) - (b : Rat) < 0 := by
    have : (c : Rat) < (b : Rat) := by exact_mod_cast h
    grind
  have hstep := Rat.mul_lt_mul_of_pos_right hnum (Rat.inv_pos.mpr hn')
  simpa using hstep

/-- **PD4.** Equal discordant counts make `δ` vanish, for every `n` including
`0`: no imbalance, no difference.

Manuscript: `98_supplement_statistics.md`, `eq:paired_difference`.

FALSIFIER: a population with `b = c` and a nonzero `δ` — which would mean the
statistic reports a difference where the discordant cells carry none. This
statement is unconditional, so it carries no satisfiability witness. -/
theorem pairedDelta_zero (b n : Nat) : pairedDelta b b n = 0 := by
  unfold pairedDelta
  rw [Rat.div_def]
  grind

/-! ## Satisfiability witnesses

PD1, PD2 and PD3 are conditional. Each witness below applies its theorem to a
concrete nondegenerate input, which both exhibits the hypotheses as inhabited
and fires the conclusion. The numbers are synthetic. -/

/-- PD2's hypotheses are inhabited and its conclusion holds: at `(b, c, n) =
(0, 1, 1)`, `δ = 1`. -/
theorem witness_pos : 0 < pairedDelta 0 1 1 :=
  pairedDelta_pos 0 1 1 (by decide) (by decide)

/-- PD3's hypotheses are inhabited and its conclusion holds: at `(b, c, n) =
(1, 0, 1)`, `δ = −1`. -/
theorem witness_neg : pairedDelta 1 0 1 < 0 :=
  pairedDelta_neg 1 0 1 (by decide) (by decide)

/-- PD1's hypothesis is inhabited and its conclusion holds: at `(b, c, n) =
(0, 1, 1)` the share form and the difference form both give `1`. -/
theorem witness_discordant_form :
    pairedDelta 0 1 1 = (2 * discordantShare 1 0 - 1) * (((1 + 0 : Nat)) : Rat) / 1 :=
  pairedDelta_discordant_form 0 1 1 (by decide)

/-! ## Load-bearing hypotheses

PD2 and PD3 each carry two hypotheses and both are load-bearing. Every control
below drops exactly one of them, and its companion proves the other still holds
on the same input, so no control is a multi-fault witness that could be filed
under the wrong hypothesis. -/

/-- PD2's imbalance hypothesis `b < c` is load-bearing: at `(1, 0, 1)` the
remaining hypothesis `0 < n` still holds but the conclusion is false. -/
theorem pairedDelta_pos_drops_imbalance : ¬ (0 < pairedDelta 1 0 1) := by
  unfold pairedDelta
  rw [Rat.div_def]
  grind

/-- The companion for `pairedDelta_pos_drops_imbalance`: on the same input
`(b, c, n) = (1, 0, 1)`, PD2's remaining hypothesis `0 < n` still holds, so the
control is a one-fault input. -/
theorem pairedDelta_pos_drops_imbalance_companion : 0 < 1 := by decide

/-- PD2's scored-position hypothesis `0 < n` is load-bearing: at `(0, 1, 0)` the
remaining hypothesis `b < c` still holds but the conclusion is false, because
`δ = 0`. -/
theorem pairedDelta_pos_drops_nonempty : ¬ (0 < pairedDelta 0 1 0) := by
  unfold pairedDelta
  rw [Rat.div_def]
  grind

/-- The companion for `pairedDelta_pos_drops_nonempty`: on the same input
`(b, c, n) = (0, 1, 0)`, PD2's remaining hypothesis `b < c` still holds. -/
theorem pairedDelta_pos_drops_nonempty_companion : (0 : Nat) < 1 := by decide

/-- PD3's imbalance hypothesis `c < b` is load-bearing: at `(0, 1, 1)` the
remaining hypothesis `0 < n` still holds but the conclusion is false. -/
theorem pairedDelta_neg_drops_imbalance : ¬ (pairedDelta 0 1 1 < 0) := by
  unfold pairedDelta
  rw [Rat.div_def]
  grind

/-- The companion for `pairedDelta_neg_drops_imbalance`: on the same input
`(b, c, n) = (0, 1, 1)`, PD3's remaining hypothesis `0 < n` still holds. -/
theorem pairedDelta_neg_drops_imbalance_companion : 0 < 1 := by decide

/-- PD3's scored-position hypothesis `0 < n` is load-bearing: at `(1, 0, 0)` the
remaining hypothesis `c < b` still holds but the conclusion is false, because
`δ = 0`. -/
theorem pairedDelta_neg_drops_nonempty : ¬ (pairedDelta 1 0 0 < 0) := by
  unfold pairedDelta
  rw [Rat.div_def]
  grind

/-- The companion for `pairedDelta_neg_drops_nonempty`: on the same input
`(b, c, n) = (1, 0, 0)`, PD3's remaining hypothesis `c < b` still holds. -/
theorem pairedDelta_neg_drops_nonempty_companion : (0 : Nat) < 1 := by decide

end AILM
