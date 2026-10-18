/-
  AILM.Holm: the step-down arithmetic behind `sec:multiplicity`'s Holm paragraph.

  `docs/manuscript/98_supplement_statistics.md` § "Why Bonferroni" states, in
  prose, the one comparative fact that justifies keeping a fixed threshold:

  > Holm's step-down procedure is the real alternative: it also controls the
  > familywise rate under arbitrary dependence and is uniformly at least as
  > powerful, so every member Bonferroni admits Holm admits too, and the
  > converse can fail.

  The second clause of that sentence, *uniformly at least as powerful and
  strictly so sometimes*, is arithmetic, and this module is where it stops
  being prose a reader has to re-derive. The Python it transcribes is
  `holm_step_down` in `src/ailm/statistics.py`: sort ascending, compare the
  rank-`i` p-value of a family of `m` against `α / (m - i + 1)`, halt at the
  first member that fails, and retain every hypothesis from there on.

  ## What this file does NOT prove: read this before citing it

  The same caveat `AILM/Bonferroni.lean` carries applies here **unchanged, and
  for the same reason**. This module proves arithmetic and comparison-counting.
  It does not prove familywise error control:

  * there is no probability measure here, no event algebra, and no
    measure-theoretic union bound;
  * "Holm controls the FWER under arbitrary dependence" is a theorem of
    Holm (1979) resting on Boole's inequality applied to rejection events, and
    it is **assumed here, not proved**;
  * consequently, "uniformly at least as powerful" is established here only in
    the sense of *rejection counts on a given recorded family*, which is the
    sense the manuscript sentence uses, and not as a statement about power
    functions under any distribution.

  Citing this module as establishing FWER control for Holm would be exactly the
  "necessary condition is not sufficient" defect `sec:gate_design` diagnoses
  elsewhere in this project.

  Two further exclusions, stated so the boundary is not left to inference:

  * **The step-down halt is not proved to be the right stopping rule.** That the
    procedure must retain every hypothesis after the first failure is Holm's
    closed-testing argument, which needs the event algebra above. Here it is a
    *definition* (`stepDownCount` returns `0` on the first failure), and the
    theorems are about that definition.
  * **Nothing here proves that `holm_step_down` in Python computes
    `stepDownCount`.** The Python sorts floats and divides floats; this file
    counts over exact `List Rat`. The honest reading of the isomorphism row is
    "the same object, stated twice", and `tests/test_statistics.py` is what
    exercises the Python. This is the same gap `AILM/Filtering.lean` records
    against `oracle.forward_filter`, and it is a gap, not an oversight.
-/
import AILM.Core
import AILM.Bonferroni

namespace AILM

/-! ## Two order facts about `Rat` division that Lean core does not supply

Core `Rat` has `mul_le_mul_of_nonneg_left`, `le_of_mul_le_mul_left`,
`mul_pos` and `mul_inv_cancel`, but no antitonicity of `x ↦ a / x`. Both
statements below are ordinary ordered-field facts; they are proved here rather
than assumed because Mathlib is out of scope by construction (see README.md),
and `grind` cannot do the nonlinear step on its own: it reduces the goal to
exactly the inverse comparison and stops. -/

/--
**H0a.** Inversion is antitone on the positive rationals.

FALSIFIER: a pair `0 < x ≤ y` with `x⁻¹ < y⁻¹`. This is the step every
threshold comparison below rests on: a Holm bar and a Bonferroni bar are the
same numerator over different denominators, so their order is decided entirely
here.

The `0 < x` hypothesis is not decorative. At `x = 0` core `Rat` sets
`0⁻¹ = 0`, so `0 ≤ y` with `y⁻¹ ≤ 0⁻¹` fails for every `0 < y`;
`inversionPositivityIsLoadBearing` exhibits that failure.
-/
theorem inv_antitone_of_pos (x y : Rat) (hx : 0 < x) (hxy : x ≤ y) : y⁻¹ ≤ x⁻¹ := by
  have hy : 0 < y := by grind
  have hxne : x ≠ 0 := by grind
  have hyne : y ≠ 0 := by grind
  have hc : 0 < x * y := Rat.mul_pos hx hy
  refine Rat.le_of_mul_le_mul_left ?_ hc
  have ex : x * y * y⁻¹ = x := by
    rw [Rat.mul_assoc, Rat.mul_inv_cancel y hyne, Rat.mul_one]
  have ey : x * y * x⁻¹ = y := by
    rw [Rat.mul_comm x y, Rat.mul_assoc, Rat.mul_inv_cancel x hxne, Rat.mul_one]
  rw [ex, ey]
  exact hxy

/--
**H0b.** A nonnegative numerator over a larger positive denominator is no
larger.

FALSIFIER: a nonnegative `a` and denominators `0 < x ≤ y` with `a / x < a / y`.

This is H0a with the numerator carried through, and it is the only shape in
which division is ever compared in this module. -/
theorem div_antitone_in_denominator (a x y : Rat) (ha : 0 ≤ a) (hx : 0 < x) (hxy : x ≤ y) :
    a / y ≤ a / x := by
  rw [Rat.div_def, Rat.div_def]
  exact Rat.mul_le_mul_of_nonneg_left (inv_antitone_of_pos x y hx hxy) ha

/-! ## The per-rank threshold -/

/--
Holm's per-rank threshold: for rank `i` in a family of `m`, the bar is
`α / (m - i + 1)`.

This transcribes `holm_step_down`'s `threshold = level / (size - index)` with
`index` the zero-based position and `i = index + 1` the rank, so the rank-1 bar
divides by `m` and the rank-`m` bar divides by `1`.

Definition only. As with `bonferroniThreshold`, no `decide`d fact ever
evaluates it: `Rat` division does not kernel-reduce, so every statement about a
concrete bar rewrites it to an exact literal first and compares literals.

The subtraction is `Nat` truncated subtraction, which saturates rather than
wrapping: at `i > m` the bar is `α / 1 = α`. That is not a rank the procedure
ever visits on a family of `m`, and no theorem below pretends otherwise; it is
recorded because it is why `holm_threshold_monotone_in_rank` needs no upper
bound on the rank. -/
def holmThreshold (alpha : Rat) (m i : Nat) : Rat := alpha / ((m - i + 1 : Nat) : Rat)

/--
**H1.** Holm's bar at any valid rank is never stricter than Bonferroni's.

Manuscript: `98_supplement_statistics.md`, "uniformly at least as powerful, so
every member Bonferroni admits Holm admits too".

FALSIFIER: a family size `m`, a valid rank `i`, and a positive `α` for which the
step-down bar `α/(m-i+1)` falls *below* the fixed bar `α/m`, which is to say a
rank at which Holm is stricter than Bonferroni. That world would break the domination
claim the manuscript uses to justify reporting the conservative survivor set,
and it is the entire arithmetic content of the first half of that sentence.

`0 < alpha` is required: at `α = 0` both bars are `0` and the statement is true
but empty, and at `α < 0` the inequality reverses. `1 ≤ i` is required and is
where `Nat` truncated subtraction bites: at `i = 0` the bar is `α/(m+1)`, which
is strictly *below* the Bonferroni bar, and
`holm_rank_positivity_is_load_bearing` exhibits that. -/
theorem bonferroni_le_holm_threshold (alpha : Rat) (m i : Nat)
    (halpha : 0 < alpha) (hi : 1 ≤ i) (him : i ≤ m) :
    bonferroniThreshold alpha m ≤ holmThreshold alpha m i := by
  have hnat : m - i + 1 ≤ m := by omega
  have hden : ((m - i + 1 : Nat) : Rat) ≤ (m : Rat) := by exact_mod_cast hnat
  have hpos : (0 : Rat) < ((m - i + 1 : Nat) : Rat) := by
    have : 0 < m - i + 1 := Nat.succ_pos _
    exact_mod_cast this
  exact div_antitone_in_denominator alpha _ _ (Rat.le_of_lt halpha) hpos hden

/--
**H1'.** The same domination, with the upper bound on the rank dropped.

FALSIFIER: the same as H1, at a rank past the end of the family.

This is the form the step-down count needs. `stepDownCount` walks ranks
`1, 2, 3, …` for as long as the list runs, and nothing in its type ties the
list's length to the declared family size `m`, so a proof that only covers
`i ≤ m` cannot be applied at every rank the walk reaches. Beyond the family the
`Nat` subtraction saturates and the bar sits at `α / 1 = α`, which for `1 ≤ m`
is still at least `α / m`. `1 ≤ m` is what does the work there and is required:
at `m = 0` the Bonferroni bar collapses to zero (`zeroFamilyThresholdCollapses`)
and the comparison stops being about a correction at all. -/
theorem bonferroni_le_holm_threshold_at_every_rank (alpha : Rat) (m i : Nat)
    (halpha : 0 < alpha) (hm : 1 ≤ m) (hi : 1 ≤ i) :
    bonferroniThreshold alpha m ≤ holmThreshold alpha m i := by
  have hnat : m - i + 1 ≤ m := by omega
  have hden : ((m - i + 1 : Nat) : Rat) ≤ (m : Rat) := by exact_mod_cast hnat
  have hpos : (0 : Rat) < ((m - i + 1 : Nat) : Rat) := by
    have : 0 < m - i + 1 := Nat.succ_pos _
    exact_mod_cast this
  exact div_antitone_in_denominator alpha _ _ (Rat.le_of_lt halpha) hpos hden

/--
**H2.** The step-down bar is nondecreasing in the rank.

Manuscript: the "step-down" structure itself. The bar loosens as the procedure
walks down the sorted family, which is why halting at the first failure is a
stopping rule rather than an arbitrary truncation.

FALSIFIER: a rank at which the next bar is *stricter* than the current one. In
that world the procedure could reject a member and then reject a strictly larger
p-value under a tighter bar, and "step-down" would name nothing.

No rank bounds are assumed, and that is deliberate rather than an oversight:
`Nat` truncated subtraction saturates at `i ≥ m`, pinning every out-of-range bar
at `α / 1`, so the sequence is nondecreasing over all of `Nat`. Adding
`1 ≤ i ≤ m` here would state something weaker while looking more careful. -/
theorem holm_threshold_monotone_in_rank (alpha : Rat) (m i : Nat) (halpha : 0 ≤ alpha) :
    holmThreshold alpha m i ≤ holmThreshold alpha m (i + 1) := by
  have hnat : m - (i + 1) + 1 ≤ m - i + 1 := by omega
  have hden : ((m - (i + 1) + 1 : Nat) : Rat) ≤ ((m - i + 1 : Nat) : Rat) := by
    exact_mod_cast hnat
  have hpos : (0 : Rat) < ((m - (i + 1) + 1 : Nat) : Rat) := by
    have : 0 < m - (i + 1) + 1 := Nat.succ_pos _
    exact_mod_cast this
  exact div_antitone_in_denominator alpha _ _ halpha hpos hden

/--
**H3.** At rank 1 the two corrections coincide exactly.

FALSIFIER: an off-by-one in the rank convention. If `holmThreshold` divided by
`m - i` or by `m - i + 2`, the first bar would not be `α/m` and this equation
would fail; it is the check that this module's rank indexing matches
`holm_step_down`'s zero-based `size - index` at the one rank where the two
procedures are required to agree.

`1 ≤ m` is load-bearing; see `holm_rank_one_hypothesis_is_load_bearing`. -/
theorem holm_rank_one_is_bonferroni (alpha : Rat) (m : Nat) (hm : 1 ≤ m) :
    holmThreshold alpha m 1 = bonferroniThreshold alpha m := by
  have hnat : m - 1 + 1 = m := by omega
  unfold holmThreshold bonferroniThreshold
  rw [hnat]

/-! ## Counting rejections

Two counters, deliberately different shapes, because the two corrections have
different shapes.

`countBelow` is a *set* count: a fixed bar admits a member wherever it sits in
the list, so Bonferroni's rejection set is "every member below `α/m`".

`stepDownCount` is a *prefix* count: Holm halts at the first failure, so its
rejection set is a leading run and nothing after the halt can be rejected no
matter how small it is. Keeping the two definitions distinct is what makes the
sortedness hypothesis below visible instead of hidden. -/

/-- How many members of `ps` fall strictly below the fixed bar `t`, anywhere in
the list. This is the Bonferroni rejection set's size. -/
def countBelow (t : Rat) : List Rat → Nat
  | [] => 0
  | p :: rest => (if p < t then 1 else 0) + countBelow t rest

/-- The length of the leading run of `ps` that clears its own rank's bar, with
ranks counted from `i`. `bar` is the schedule; `holmRejectCount` supplies Holm's.

The `else 0` is the step-down halt: everything after the first failure is
retained, whatever its p-value. -/
def stepDownCount (bar : Nat → Rat) : Nat → List Rat → Nat
  | _, [] => 0
  | i, p :: rest => if p < bar i then stepDownCount bar (i + 1) rest + 1 else 0

/-- Holm's rejection count on a family, with the family size read off the list
and ranks starting at 1. -/
def holmRejectCount (alpha : Rat) (ps : List Rat) : Nat :=
  stepDownCount (holmThreshold alpha ps.length) 1 ps

/-- Bonferroni's rejection count on the same family: every member below the
single corrected bar. -/
def bonferroniRejectCount (alpha : Rat) (ps : List Rat) : Nat :=
  countBelow (bonferroniThreshold alpha ps.length) ps

/--
**H4.** A step-down walk under a uniformly looser schedule rejects at least as
many leading members.

FALSIFIER: two bar schedules, one everywhere at least the other from rank `i`
on, whose leading runs are ordered the wrong way. This is the counting core of
"uniformly at least as powerful", and it is where the *step-down* structure
earns its keep: a looser bar can only ever extend the run, never truncate it,
because the halt is triggered by a failure that the looser bar may not see.

The hypothesis is stated from rank `i` upward rather than for all of `Nat`
because rank `0` is not a rank the walk ever visits and the Holm bar there is
strictly tighter (see H1's note). Quantifying over every natural would leave
this result unusable at exactly the instantiation it exists for. -/
theorem stepDownCount_mono_in_bar (b c : Nat → Rat) :
    ∀ (i : Nat) (ps : List Rat), (∀ j, i ≤ j → b j ≤ c j) →
      stepDownCount b i ps ≤ stepDownCount c i ps
  | _, [], _ => Nat.le_refl 0
  | i, p :: rest, h => by
      by_cases hp : p < b i
      · have hbc : b i ≤ c i := h i (Nat.le_refl i)
        have hc : p < c i := by grind
        have hrest := stepDownCount_mono_in_bar b c (i + 1) rest
          (fun j hj => h j (Nat.le_trans (Nat.le_succ i) hj))
        simp only [stepDownCount, ite_eq_left hp, ite_eq_left hc]
        omega
      · simp only [stepDownCount, ite_eq_right hp]
        exact Nat.zero_le _

/--
**H5.** Nothing below the fixed bar is left in a list whose members all sit at
or above it.

FALSIFIER: a list every member of which is at least `t`, yet with a member
counted as strictly below `t`. Used only to discharge the halt case of H6; it is
the step that turns "the head fails" into "no member later can succeed", and it
is exactly where sortedness enters. -/
theorem countBelow_eq_zero_of_all_at_least (t : Rat) :
    ∀ ps : List Rat, (∀ q ∈ ps, t ≤ q) → countBelow t ps = 0
  | [], _ => rfl
  | p :: rest, h => by
      have hp : ¬ p < t := by
        have := h p (List.mem_cons_self ..)
        grind
      have hrest := countBelow_eq_zero_of_all_at_least t rest
        (fun q hq => h q (List.mem_cons_of_mem p hq))
      simp only [countBelow, ite_eq_right hp, hrest]

/--
**H6.** On a sorted family, the fixed bar's rejection *set* and its leading
*run* are the same size.

FALSIFIER: an ascending list on which some member below the bar sits after a
member at or above it. Impossible when sorted, and possible otherwise, which is
the whole content of the sortedness hypothesis and the reason
`sortednessIsLoadBearing` below is not decoration.

`List.Pairwise (· ≤ ·)` is the sortedness notion, matching `holm_step_down`'s
`sorted(...)` step. Note the direction this lemma runs: it is an *equality*, so
it also rules out the leading run over-counting, not merely under-counting. -/
theorem sorted_countBelow_eq_stepDown_const (t : Rat) :
    ∀ (i : Nat) (ps : List Rat), List.Pairwise (· ≤ ·) ps →
      countBelow t ps = stepDownCount (fun _ => t) i ps
  | _, [], _ => rfl
  | i, p :: rest, hsorted => by
      have hhead : ∀ q ∈ rest, p ≤ q := (List.pairwise_cons.mp hsorted).1
      have htail : List.Pairwise (· ≤ ·) rest := (List.pairwise_cons.mp hsorted).2
      by_cases hp : p < t
      · have hrest := sorted_countBelow_eq_stepDown_const t (i + 1) rest htail
        simp only [countBelow, stepDownCount, ite_eq_left hp, hrest]
        omega
      · have hall : ∀ q ∈ rest, t ≤ q := by
          intro q hq
          have := hhead q hq
          grind
        simp only [countBelow, stepDownCount, ite_eq_right hp,
          countBelow_eq_zero_of_all_at_least t rest hall]

/--
**H7.** On a sorted family, Holm rejects every member Bonferroni rejects.

Manuscript: `98_supplement_statistics.md`, "every member Bonferroni admits Holm
admits too". This is that clause, machine-checked, in the counting sense stated
in the module header, not as a statement about familywise error rates.

FALSIFIER: an ascending family of p-values and a positive `α` at which the fixed
bar `α/m` rejects strictly more members than the step-down walk does. That world
would falsify the manuscript's justification for keeping the fixed threshold,
which is that it costs power and nothing else: the fixed threshold would
sometimes be reporting a *larger* survivor set than the procedure it defers to.

`1 ≤ ps.length` is required: on the empty family `bonferroniThreshold α 0` is
`0` rather than a correction, and both counts are `0` for reasons unrelated to
the correction. `0 < alpha` is required by H1'. -/
theorem holm_rejects_every_bonferroni_rejection (alpha : Rat) (ps : List Rat)
    (halpha : 0 < alpha) (hlen : 1 ≤ ps.length) (hsorted : List.Pairwise (· ≤ ·) ps) :
    bonferroniRejectCount alpha ps ≤ holmRejectCount alpha ps := by
  unfold bonferroniRejectCount holmRejectCount
  rw [sorted_countBelow_eq_stepDown_const _ 1 ps hsorted]
  exact stepDownCount_mono_in_bar _ _ 1 ps
    (fun j hj => bonferroni_le_holm_threshold_at_every_rank alpha ps.length j halpha hlen hj)

/-! ## Satisfiability witnesses

Every theorem above is conditional, so each hypothesis is exhibited inhabited,
and H7's conclusion is exhibited *strict*. A containment witness that only ever
showed equality would leave the "and the converse can fail" half of the
manuscript sentence unexercised.

The p-values below are **synthetic**, not recorded. They are chosen to have the
right shape, an ascending family of four at a familywise level of `1/100` with
a Bonferroni bar of `1/400` that two members clear and two do not, and no
recorded p-value is hand-typed into Lean anywhere in this file. The ledger-backed
literals live in the generated `AILM/LedgerData.lean`. -/

/-- Four ascending synthetic p-values. Two sit below the fixed bar `1/400`; all
four clear their own Holm rank bar, so the step-down walk runs to the end. -/
def witnessFamilySorted : List Rat := [1/1000, 1/500, 1/250, 1/150]

theorem witnessFamilySorted_length : witnessFamilySorted.length = 4 := by decide

/-- H6's and H7's sortedness hypothesis is inhabited. -/
theorem witnessFamilySorted_ascending : List.Pairwise (· ≤ ·) witnessFamilySorted := by
  unfold witnessFamilySorted
  refine List.pairwise_cons.mpr ⟨?_, List.pairwise_cons.mpr ⟨?_,
    List.pairwise_cons.mpr ⟨?_, List.pairwise_singleton ..⟩⟩⟩
  · intro q hq
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
    rcases hq with rfl | rfl | rfl <;> grind
  · intro q hq
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
    rcases hq with rfl | rfl <;> grind
  · intro q hq
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
    rcases hq with rfl <;> grind

/-- The witness family's four Holm bars, as exact reduced literals.

`Rat` division does not kernel-reduce, so a concrete step-down walk cannot be
`decide`d against `holmThreshold` directly; each bar is rewritten to a literal
first, and that rewrite is an arithmetic fact rather than an order comparison.

FALSIFIER: a `holmThreshold` that is not `α/(m-i+1)` at the family size these
witnesses use: the same falsifier as H3, instantiated four times. -/
theorem witnessFamilySorted_bars :
    holmThreshold (1 / 100) 4 1 = 1 / 400 ∧ holmThreshold (1 / 100) 4 2 = 1 / 300 ∧
      holmThreshold (1 / 100) 4 3 = 1 / 200 ∧ holmThreshold (1 / 100) 4 4 = 1 / 100 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> (unfold holmThreshold; push_cast; grind)

/-- The corrected fixed bar at this family's size, as an exact reduced literal.
The bridge H7's Bonferroni side needs, for the same reason as above. -/
theorem witnessFamilySorted_fixed_bar : bonferroniThreshold (1 / 100) 4 = 1 / 400 := by
  unfold bonferroniThreshold
  push_cast
  grind

/-- Holm rejects all four members: the step-down walk never halts. -/
theorem witnessFamilySorted_holm_count : holmRejectCount (1 / 100) witnessFamilySorted = 4 := by
  obtain ⟨b1, b2, b3, b4⟩ := witnessFamilySorted_bars
  unfold holmRejectCount
  rw [witnessFamilySorted_length]
  simp only [witnessFamilySorted, stepDownCount, b1, b2, b3, b4]
  rw [ite_eq_left (by grind : (1 : Rat) / 150 < 1 / 100)]
  rw [ite_eq_left (by grind : (1 : Rat) / 250 < 1 / 200)]
  rw [ite_eq_left (by grind : (1 : Rat) / 500 < 1 / 300)]
  rw [ite_eq_left (by grind : (1 : Rat) / 1000 < 1 / 400)]

/-- Bonferroni rejects two of the four: the third and fourth members clear the
per-test bar under Holm but not under the fixed one. -/
theorem witnessFamilySorted_bonferroni_count :
    bonferroniRejectCount (1 / 100) witnessFamilySorted = 2 := by
  unfold bonferroniRejectCount
  rw [witnessFamilySorted_length, witnessFamilySorted_fixed_bar]
  simp only [witnessFamilySorted, countBelow]
  rw [ite_eq_right (by grind : ¬ (1 : Rat) / 150 < 1 / 400)]
  rw [ite_eq_right (by grind : ¬ (1 : Rat) / 250 < 1 / 400)]
  rw [ite_eq_left (by grind : (1 : Rat) / 500 < 1 / 400)]
  rw [ite_eq_left (by grind : (1 : Rat) / 1000 < 1 / 400)]

/-- H7 fires on a nonempty sorted family with every hypothesis discharged, so
the containment is exercised rather than merely stated. -/
theorem witnessFamilySorted_containment :
    bonferroniRejectCount (1 / 100) witnessFamilySorted ≤
      holmRejectCount (1 / 100) witnessFamilySorted :=
  holm_rejects_every_bonferroni_rejection _ _ (by grind)
    (by rw [witnessFamilySorted_length]; omega) witnessFamilySorted_ascending

/--
**The converse can fail.** On this family the containment of H7 is *strict*:
Holm rejects four where Bonferroni rejects two.

FALSIFIER: none, because this is a computation, and that is the point. Without it H7
would be satisfied by every family on which the two procedures agree, and the
second half of the manuscript sentence ("and the converse can fail") would rest
on a theorem that never distinguishes them. A containment witness that only
exhibits equality proves the containment is inhabited, not that it is proper. -/
theorem witnessFamilySorted_holm_rejects_strictly_more :
    bonferroniRejectCount (1 / 100) witnessFamilySorted <
      holmRejectCount (1 / 100) witnessFamilySorted := by
  rw [witnessFamilySorted_bonferroni_count, witnessFamilySorted_holm_count]
  omega

/-- H1's and H2's hypotheses are inhabited at this family's size and every one of
its four ranks, so the threshold domination is exercised on real ranks rather
than only stated. -/
theorem witnessFamilySorted_bars_dominate_the_fixed_bar :
    ∀ i, 1 ≤ i → i ≤ 4 → bonferroniThreshold (1 / 100) 4 ≤ holmThreshold (1 / 100) 4 i :=
  fun i hi him => bonferroni_le_holm_threshold (1 / 100) 4 i (by grind) hi him

/-! ### Negative controls

Each shows a hypothesis above is load-bearing: drop it and the conclusion fails
on a concrete input. They sit under the witness marker rather than beside it so
that the module's witness count in `../README.md` covers them, matching how
`Bonferroni.lean` files `scaleHypothesisIsLoadBearing`. -/

/--
**Negative control.** H6's and H7's sortedness hypothesis is load-bearing.

On the *unsorted* family `[1/2, 1/1000]` at `α = 1/100` and `m = 2` the fixed bar
is `1/200`. The second member clears it, so Bonferroni rejects one; the first
member does not, so the step-down walk halts immediately and Holm rejects none.
Containment fails in the direction H7 forbids.

This is the failure mode the sorting step in `holm_step_down` exists to prevent,
and it is why H6 is stated as an equality between a set count and a prefix count
rather than assumed. -/
theorem sortednessIsLoadBearing :
    ¬ List.Pairwise (· ≤ ·) [(1 : Rat) / 2, 1 / 1000] ∧
      holmRejectCount (1 / 100) [(1 : Rat) / 2, 1 / 1000] = 0 ∧
      bonferroniRejectCount (1 / 100) [(1 : Rat) / 2, 1 / 1000] = 1 := by
  refine ⟨?_, ?_, ?_⟩
  · intro hsorted
    have := (List.pairwise_cons.mp hsorted).1 (1 / 1000) (by simp)
    grind
  · have hbar : holmThreshold (1 / 100) 2 1 = 1 / 200 := by
      unfold holmThreshold; push_cast; grind
    unfold holmRejectCount
    simp only [List.length_cons, List.length_nil, stepDownCount, hbar]
    rw [ite_eq_right (by grind : ¬ (1 : Rat) / 2 < 1 / 200)]
  · have hbar : bonferroniThreshold (1 / 100) 2 = 1 / 200 := by
      unfold bonferroniThreshold; push_cast; grind
    unfold bonferroniRejectCount
    simp only [List.length_cons, List.length_nil, countBelow, hbar]
    rw [ite_eq_right (by grind : ¬ (1 : Rat) / 2 < 1 / 200)]
    rw [ite_eq_left (by grind : (1 : Rat) / 1000 < 1 / 200)]

/--
**Negative control.** H3's `1 ≤ m` hypothesis is load-bearing.

At `m = 0` the `Nat` subtraction saturates: `0 - 1 + 1 = 1`, so the rank-1 Holm
bar is `α / 1 = α`, while `bonferroniThreshold α 0` collapses to `0`
(`zeroFamilyThresholdCollapses`). The two corrections do not coincide at rank 1
on the empty family, and the equation H3 states is false there. -/
theorem holm_rank_one_hypothesis_is_load_bearing :
    holmThreshold (1 / 100) 0 1 ≠ bonferroniThreshold (1 / 100) 0 := by
  rw [zeroFamilyThresholdCollapses]
  unfold holmThreshold
  push_cast
  grind

/--
**Negative control.** H1's `1 ≤ i` hypothesis is load-bearing.

At rank `0` the bar is `α/(m+1)`, strictly *below* the Bonferroni bar `α/m`, so
the domination reverses. Rank `0` is not a rank the procedure visits, since
ranks are one-based, but nothing in `holmThreshold`'s type says so, which is
precisely why the hypothesis has to be carried rather than assumed away. -/
theorem holm_rank_positivity_is_load_bearing :
    holmThreshold (1 / 100) 4 0 < bonferroniThreshold (1 / 100) 4 := by
  have hholm : holmThreshold (1 / 100) 4 0 = 1 / 500 := by
    unfold holmThreshold; push_cast; grind
  rw [hholm, witnessFamilySorted_fixed_bar]
  grind

/--
**Negative control.** H0a's positivity hypothesis is load-bearing.

Core `Rat` defines `0⁻¹ = 0`, so at `x = 0` the conclusion `y⁻¹ ≤ x⁻¹` reads
`y⁻¹ ≤ 0`, which is false for every positive `y`. Without `0 < x` the lemma
would be false, and every threshold comparison in this module rests on it. -/
theorem inversionPositivityIsLoadBearing : ¬ ((2 : Rat)⁻¹ ≤ (0 : Rat)⁻¹) := by
  have hzero : (0 : Rat)⁻¹ = 0 := by grind
  rw [hzero]
  grind

end AILM
