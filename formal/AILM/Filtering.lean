/-
  AILM.Filtering — both halves of `eq:filtering`.

  `manuscript/02_background.md` states the filter step as

      ρ_t(j) = Σ_k B_jk q_{t-1}(k),
      q_t(j) = A_{o_t j} ρ_t(j) / Σ_k A_{o_t k} ρ_t(k).

  The **second** equation is the correction step, F1–F4 below. `Evidence` is the
  vector of numerators `A_{o_t j} ρ_t(j)` indexed by state `j`; `normalize`
  divides by the denominator `Σ_k A_{o_t k} ρ_t(k)` — which is also, by
  `eq:perplexity`, the predictive probability of the observation, so it is the
  one quantity in the recursion that both the posterior and the perplexity read.

  The **first** equation is the prediction step, P1–P12 below. It says the
  transition operator maps the simplex into itself, so `ρ_t` is a legitimate
  input to the correction step and the recursion closes.

  Python counterpart: `oracle.forward_filter` (`manuscript/10_verification.md`,
  `eq:forward_recursion`) — the `transition @ pred` line for the prediction
  half, the `unnormalized / norm` lines for the correction half.

  ## Deliberately NOT proved here (see README.md for the full list)

  * **Perplexity** (`eq:perplexity`) and the **order-1 floor**
    (`eq:conditional_entropy`). Both need `exp`/`log`. `Rat` has no
    transcendental functions and there is no `Real` without Mathlib. A Lean
    "theorem about perplexity" here would quantify over an uninterpreted symbol
    and prove nothing about the measured quantity.
  * **pymdp ↔ oracle numerical agreement** (`eq:decoder_equivalence`): float
    equality within a tolerance, not a Lean object. It stays in
    `scripts/record_stage2_claims.py`.
-/
import AILM.Core

namespace AILM

/-- The unnormalized evidence vector `A_{o_t j} · ρ_t(j)`, indexed by hidden
state `j`. These are the *numerators* of `eq:filtering`. -/
abbrev Evidence := List Rat

/-- The Bayes correction of `eq:filtering`: divide each numerator by the total
evidence mass. `ev.sum` is exactly the denominator `Σ_k A_{o_t k} ρ_t(k)`. -/
def normalize (ev : Evidence) : List Rat := ev.map (fun x => x / ev.sum)

/-- A list is a probability distribution when it is pointwise nonnegative and
sums to one. -/
def IsDistribution (xs : List Rat) : Prop := (∀ x ∈ xs, 0 ≤ x) ∧ xs.sum = 1

/--
**F1.** Dividing every element by a fixed `Z` divides the sum by `Z`.

Linearity of the normalizer. Holds unconditionally, including at `Z = 0`, where
both sides are `0` because `Rat` division by zero is zero.

FALSIFIER: an implementation that renormalizes element-by-element against a
denominator that drifts as it goes — i.e. recomputing the total inside the loop
after already having modified part of it. That is a real and easy defect in an
in-place filter step, and it breaks exactly this equation.
-/
theorem sum_div (Z : Rat) : ∀ xs : List Rat, (xs.map (fun x => x / Z)).sum = xs.sum / Z
  | [] => by simp [Rat.div_def]
  | _ :: rest => by
      simp only [List.map_cons, List.sum_cons, sum_div Z rest]
      grind

/--
**F2.** A filter step on evidence with nonzero total mass returns something that
sums to one.

Manuscript: `eq:filtering`; `eq:forward_recursion`'s per-step renormalization in
`oracle.forward_filter`.

FALSIFIER: a filtering implementation whose output is not a probability
distribution — the defect class the oracle's per-step renormalization exists to
prevent, and precisely what a transposed emission matrix produces (the wrong
slot still sums to *something*, just not to the right thing; see
`gotcha: generative matrix in wrong slot is anti-informative`).

The hypothesis `ev.sum ≠ 0` is not removable: `zeroMassCounterexample` below
exhibits an evidence vector with zero mass whose normalization sums to `0`, not
`1`. So this is a conditional theorem with a *witnessed* hypothesis and a
*witnessed* failure outside it.
-/
theorem normalize_sum (ev : Evidence) (h : ev.sum ≠ 0) : (normalize ev).sum = 1 := by
  unfold normalize
  rw [sum_div]
  grind

/--
**F3.** Normalizing nonnegative evidence with positive mass keeps it
nonnegative.

FALSIFIER: a normalizer that flips signs — which is what dividing by a
*negative* total would do. Nothing in `sum_div` rules that out; positivity of
the denominator, not merely its nonzeroness, is what this theorem consumes.
Without F3, F2 alone would license calling the output a "distribution" when it
could contain negative entries summing to one.
-/
theorem normalize_nonneg (ev : Evidence) (hnn : ∀ x ∈ ev, 0 ≤ x) (hpos : 0 < ev.sum) :
    ∀ y ∈ normalize ev, 0 ≤ y := by
  unfold normalize
  intro y hy
  simp only [List.mem_map] at hy
  obtain ⟨x, hx, rfl⟩ := hy
  have hinv : (0 : Rat) ≤ (ev.sum)⁻¹ := Rat.le_of_lt (Rat.inv_pos.mpr hpos)
  rw [Rat.div_def]
  exact Rat.mul_nonneg (hnn x hx) hinv

/--
**F4.** Positive evidence yields a posterior.

The full statement of the filter-step invariant: nonnegative evidence with
positive total mass normalizes to an actual probability distribution.

FALSIFIER: any nonnegative evidence vector with positive mass whose
normalization is not a distribution. This is the statement a downstream step is
entitled to assume, and the one that makes `q_t` a legitimate input to the next
prediction step.
-/
theorem normalize_isDistribution (ev : Evidence) (hnn : ∀ x ∈ ev, 0 ≤ x)
    (hpos : 0 < ev.sum) : IsDistribution (normalize ev) :=
  ⟨normalize_nonneg ev hnn hpos, normalize_sum ev (by grind)⟩

/-! ## Satisfiability witnesses

F2, F3 and F4 are conditional, so each hypothesis is exhibited inhabited below,
and the zero-mass case is exhibited as a genuine counterexample so the
hypothesis is visibly load-bearing rather than defensive.

Numerals here are written as ordinary `Rat` division (`1/4`) rather than through
`AILM.lit`. That is a deliberate, module-local choice: this file proves
*arithmetic* facts, `grind` reasons about `Rat` numerals, and no measured value
appears in it. `lit` is reserved for the exact reduced literals the ledger
generator emits, where `decide`-reducibility (not `grind`-arithmetic) is what
matters. -/

/-- A genuinely **unnormalized** evidence vector: three equal masses summing to
`3/4`. Chosen so that `normalize` has real work to do — a witness that already
summed to one would satisfy F2 with `normalize` acting as the identity, which
would make the witness prove nothing about renormalization. -/
def evWitness : Evidence := [1/4, 1/4, 1/4]

theorem evWitness_sum : evWitness.sum = 3/4 := by
  simp only [evWitness, List.sum_cons, List.sum_nil]
  grind

/-- The witness is not already a distribution — so F2's conclusion is not free. -/
theorem evWitness_not_distribution : evWitness.sum ≠ 1 := by
  rw [evWitness_sum]; grind

/-- F2's hypothesis is inhabited. -/
theorem evWitness_sum_ne_zero : evWitness.sum ≠ 0 := by
  rw [evWitness_sum]; grind

/-- F3's and F4's positivity hypothesis is inhabited.

Note the shape this proof is forced into: the positivity of the *sum* cannot be
`decide`d, because `Rat` addition does not kernel-reduce. It is derived from the
computed sum instead. Any witness added later must follow the same pattern. -/
theorem evWitness_pos : (0 : Rat) < evWitness.sum := by
  rw [evWitness_sum]; grind

/-- F3's and F4's nonnegativity hypothesis is inhabited. -/
theorem evWitness_nonneg : ∀ x ∈ evWitness, (0 : Rat) ≤ x := by
  intro x hx
  simp only [evWitness, List.mem_cons, List.not_mem_nil, or_false] at hx
  rcases hx with rfl | rfl | rfl <;> grind

/-- All three hypotheses fire together, and F4's conclusion is reached on a
vector that actually required renormalizing. -/
theorem evWitness_normalizes : IsDistribution (normalize evWitness) :=
  normalize_isDistribution _ evWitness_nonneg evWitness_pos

/-- **Negative control.** F2's hypothesis is not decoration: zero total evidence
mass normalizes to a list summing to `0`, not `1`. Dropping `ev.sum ≠ 0` would
make F2 false, so the conditional is a real restriction rather than a
precaution. -/
theorem zeroMassCounterexample : (normalize [0, 0]).sum ≠ 1 := by
  simp only [normalize, List.sum_cons, List.sum_nil, List.map_cons, List.map_nil]
  grind

/-- `sum_div` is not vacuous at the degenerate denominator either: at `Z = 0`
both sides are `0`, which is why F1 needs no hypothesis while F2 does. -/
example : (evWitness.map (fun x => x / 0)).sum = evWitness.sum / 0 := sum_div 0 evWitness

/-! ## The prediction half of `eq:filtering`

`ρ_t = B q_{t-1}`, proved from Lean core alone. The result this module used to
name as out of reach is `predict_sum` (P7): the exchange of summation order

    Σ_j Σ_k B_jk q(k)  =  Σ_k q(k) Σ_j B_jk

which core provides no Fubini-over-lists for. It is reachable without one,
because `B` is stored here as its **list of columns** rather than as a matrix of
rows. Under that representation `B q` is a linear combination of columns, the
outer sum is a fold, and the exchange collapses into two one-line inductions —
`smul_sum` (P4), pulling a scalar out of a sum, and `vadd_sum` (P6), splitting a
pointwise sum in two. The column representation also matches the Python: in
`oracle.forward_filter`, `transition[i, j] = P(next = i | cur = j)` is
column-stochastic over `i`, so a column is exactly a distribution over next
states.

`vadd` is `List.zipWith (+)`, which **truncates** to the shorter operand. That is
the whole reason widths appear as hypotheses rather than as a type index: a
column narrower than the state space silently drops mass, and P3/P7 are where
that is excluded rather than assumed. `raggedColumnLosesMass` exhibits it.
-/

/-- A transition operator, stored as its list of columns: `B[k]` is the
distribution over next states given current state `k`. Column-major to match
`oracle.forward_filter`, whose `transition[i, j]` is `P(next = i | cur = j)`. -/
abbrev Transition := List (List Rat)

/-- Scalar multiple of a vector, `a • v`. -/
def smul (a : Rat) (v : List Rat) : List Rat := v.map (fun x => a * x)

/-- Pointwise sum of two vectors. `List.zipWith` truncates to the shorter
operand, so every result about `vadd` that mentions mass carries a length
hypothesis. -/
def vadd (u v : List Rat) : List Rat := List.zipWith (· + ·) u v

/-- The all-zero vector of width `n`, the empty linear combination. -/
def zeroVec (n : Nat) : List Rat := List.replicate n 0

/-- The prediction step `ρ = B q`, as the linear combination `Σ_k q(k) • B[k]`.

`n` is the width of the state space, needed because the empty combination has no
column to read a width from. Columns and prior entries are consumed in lockstep;
a mismatch falls through to `zeroVec n`, which is why `predict_sum` requires
`B.length = q.length` rather than inferring it. -/
def predict (n : Nat) : Transition → List Rat → List Rat
  | c :: cs, x :: xs => vadd (smul x c) (predict n cs xs)
  | _, _ => zeroVec n

/--
**P1.** Scaling a vector does not change its width.

FALSIFIER: a scalar multiply that drops zero entries — a sparse representation
leaking into a dense one. P3 would then read a short column and truncate.
-/
theorem smul_length (a : Rat) (v : List Rat) : (smul a v).length = v.length := by
  simp [smul]

/--
**P2.** Pointwise addition truncates to the shorter operand.

Stated rather than hidden, because truncation is the defect this module's width
hypotheses exist to exclude, and a reader who assumes `zipWith` pads with zeros
will read P3 and P7 as decoration.

FALSIFIER: any `vadd` that padded instead of truncating. The theorem would be
false, and `raggedColumnLosesMass` would stop being a counterexample.
-/
theorem vadd_length (u v : List Rat) : (vadd u v).length = min u.length v.length := by
  simp [vadd]

/--
**P3.** A prediction over columns of width `n` has width `n`.

The invariant that makes the recursion composable: `ρ_t` is the right shape to be
multiplied by the emission row and handed to `normalize`, and the right shape to
be the input of the next prediction step.

FALSIFIER: a transition operator with a short column. By P2 the combination
would truncate to that column's width, and every later step would run on a
narrower state space than the model declares — silently, since a shorter vector
still normalizes to something.
-/
theorem predict_length (n : Nat) :
    ∀ (B : Transition) (q : List Rat), (∀ c ∈ B, c.length = n) → (predict n B q).length = n
  | [], [], _ => by simp [predict, zeroVec]
  | [], _ :: _, _ => by simp [predict, zeroVec]
  | _ :: _, [], _ => by simp [predict, zeroVec]
  | c :: cs, x :: xs, hw => by
      have hc : c.length = n := hw c (by simp)
      have ih := predict_length n cs xs (fun d hd => hw d (by simp [hd]))
      simp only [predict, vadd_length, smul_length, hc, ih]
      simp

/--
**P4.** A scalar factors out of a sum: `Σ_j a·v(j) = a·Σ_j v(j)`.

Half of the sum exchange. Induction on the list; core supplies no distributivity
over `List.sum`, so this is the base case of the scaffolding.

FALSIFIER: a weighted-mixture implementation that scales each column but
accumulates the weights separately — double-counting `a` — or one that scales
only the first entry of a column. Both preserve the shape and break this
equation.
-/
theorem smul_sum (a : Rat) : ∀ v : List Rat, (smul a v).sum = a * v.sum
  | [] => by simp [smul]
  | x :: rest => by
      have ih := smul_sum a rest
      simp only [smul, List.map_cons, List.sum_cons] at *
      rw [ih]; grind

/--
**P5.** The empty linear combination has zero mass.

FALSIFIER: an accumulator initialized to anything but zero — a prediction seeded
with the previous step's `ρ` instead of a fresh vector, which is a real defect in
an in-place filter and adds one extra unit of mass per step.
-/
theorem zeroVec_sum (n : Nat) : (zeroVec n).sum = 0 := by
  induction n with
  | zero => simp [zeroVec]
  | succ m ih =>
      unfold zeroVec at ih ⊢
      rw [List.replicate_succ, List.sum_cons, ih]
      grind

/--
**P6.** Mass is additive over a pointwise sum of vectors of equal width.

The other half of the sum exchange. The width hypothesis is not removable: by P2
a mismatched `vadd` truncates, and the dropped entries take their mass with them.

FALSIFIER: accumulating a mixture with a `zip`-style combinator over operands of
different width — the exact shape of the ragged-column defect, where the totals
stop agreeing without any array ever being the wrong type.
-/
theorem vadd_sum : ∀ u v : List Rat, u.length = v.length → (vadd u v).sum = u.sum + v.sum
  | [], [], _ => by simp [vadd]; grind
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | x :: xs, y :: ys, h => by
      have hl : xs.length = ys.length := by simpa using h
      have ih := vadd_sum xs ys hl
      simp only [vadd, List.zipWith_cons_cons, List.sum_cons] at *
      rw [ih]; grind

/--
**P7.** THE SUM EXCHANGE. A column-stochastic operator preserves total mass:
`Σ_j (B q)(j) = Σ_k q(k)`.

This is the statement the module's deferral note used to name as unreachable
from core. Reading the proof as the exchange it is: the outer induction walks
`k`, P4 turns `Σ_j q(k)·B_jk` into `q(k)·Σ_j B_jk = q(k)·1`, and P6 lets the
walk add that to the accumulated mass instead of interleaving with it. There is
no Fubini step because the column representation has already done the exchange.

All three hypotheses are load-bearing, and each has its own counterexample
below: `B.length = q.length` (`missingColumnLosesMass`), width (`raggedColumn-
LosesMass`), column-stochasticity (`leakyColumnLosesMass`).

FALSIFIER: a transition matrix used in the wrong orientation. A row-stochastic
matrix fed to a column-major consumer sums each row to one instead of each
column, so the predicted mass drifts away from one — the same defect class as
`gotcha: generative matrix in wrong slot is anti-informative`, one step earlier
in the recursion, and invisible downstream because the correction step
renormalizes whatever it is handed.
-/
theorem predict_sum (n : Nat) :
    ∀ (B : Transition) (q : List Rat), B.length = q.length → (∀ c ∈ B, c.length = n) →
      (∀ c ∈ B, c.sum = 1) → (predict n B q).sum = q.sum
  | [], [], _, _, _ => by simp [predict, zeroVec_sum]
  | [], _ :: _, h, _, _ => by simp at h
  | _ :: _, [], h, _, _ => by simp at h
  | c :: cs, x :: xs, hl, hw, hs => by
      have hc : c.length = n := hw c (by simp)
      have hcs : c.sum = 1 := hs c (by simp)
      have hl' : cs.length = xs.length := by simpa using hl
      have ihlen : (predict n cs xs).length = n :=
        predict_length n cs xs (fun d hd => hw d (by simp [hd]))
      have ihsum := predict_sum n cs xs hl' (fun d hd => hw d (by simp [hd]))
        (fun d hd => hs d (by simp [hd]))
      have hlen : (smul x c).length = (predict n cs xs).length := by
        rw [smul_length, ihlen, hc]
      simp only [predict, List.sum_cons]
      rw [vadd_sum _ _ hlen, smul_sum, ihsum, hcs]
      grind

/--
**P8.** Scaling nonnegative entries by a nonnegative factor stays nonnegative.

FALSIFIER: a prior carrying a negative entry — which a subtractive update, or a
float renormalization that underflowed past zero, can produce. Mass alone would
not notice: P7 holds for signed vectors summing to one.
-/
theorem smul_nonneg (a : Rat) (v : List Rat) (ha : 0 ≤ a) (hv : ∀ x ∈ v, 0 ≤ x) :
    ∀ y ∈ smul a v, 0 ≤ y := by
  intro y hy
  simp only [smul, List.mem_map] at hy
  obtain ⟨x, hx, rfl⟩ := hy
  exact Rat.mul_nonneg ha (hv x hx)

/-- **P9.** The empty combination is nonnegative.

FALSIFIER: an accumulator seeded with anything negative; see P5 for the mass
half of the same defect. -/
theorem zeroVec_nonneg (n : Nat) : ∀ y ∈ zeroVec n, (0 : Rat) ≤ y := by
  intro y hy
  simp only [zeroVec, List.mem_replicate] at hy
  grind

/-- **P10.** Pointwise sums of nonnegative vectors are nonnegative.

FALSIFIER: an accumulator that subtracts an already-added contribution to
"correct" a running total, which can go negative in a coordinate while the total
stays right. -/
theorem vadd_nonneg : ∀ u v : List Rat, (∀ x ∈ u, 0 ≤ x) → (∀ y ∈ v, 0 ≤ y) →
    ∀ z ∈ vadd u v, 0 ≤ z
  | [], _, _, _ => by simp [vadd]
  | _ :: _, [], _, _ => by simp [vadd]
  | x :: xs, y :: ys, hu, hv => by
      intro z hz
      have ih := vadd_nonneg xs ys (fun a ha => hu a (by simp [ha]))
        (fun b hb => hv b (by simp [hb]))
      simp only [vadd, List.zipWith_cons_cons, List.mem_cons] at hz
      rcases hz with rfl | hz
      · have hx := hu x (by simp)
        have hy := hv y (by simp)
        grind
      · exact ih z (by simpa [vadd] using hz)

/--
**P11.** A prediction from nonnegative columns and a nonnegative prior is
nonnegative.

Note what this does **not** need: no width hypothesis, and no column-sum
hypothesis. Truncation cannot manufacture a negative entry. Keeping the two
halves of `IsDistribution` on separate hypothesis sets is what makes P12's
hypotheses honest about which of them each half consumes.

FALSIFIER: any signed transition column, e.g. a matrix that was mean-centerd or
whitened before being used as a probability operator.
-/
theorem predict_nonneg (n : Nat) :
    ∀ (B : Transition) (q : List Rat), (∀ c ∈ B, ∀ y ∈ c, 0 ≤ y) → (∀ x ∈ q, 0 ≤ x) →
      ∀ z ∈ predict n B q, 0 ≤ z
  | [], [], _, _ => by simp [predict]; exact zeroVec_nonneg n
  | [], _ :: _, _, _ => by simp [predict]; exact zeroVec_nonneg n
  | _ :: _, [], _, _ => by simp [predict]; exact zeroVec_nonneg n
  | c :: cs, x :: xs, hB, hq => by
      have ih := predict_nonneg n cs xs (fun d hd => hB d (by simp [hd]))
        (fun a ha => hq a (by simp [ha]))
      exact vadd_nonneg _ _ (smul_nonneg x c (hq x (by simp)) (hB c (by simp))) ih

/--
**P12.** The prediction step maps the simplex into itself.

The prediction half of `eq:filtering` in full, and the reason the recursion
closes: `q_{t-1}` a distribution gives `ρ_t` a distribution, `ρ_t` a
distribution with positive evidence mass gives `q_t` a distribution by F4, and
the argument repeats. Before this theorem the two halves met only in prose.

FALSIFIER: a transition operator satisfying every hypothesis whose prediction is
not a distribution. The three counterexamples below each drop exactly one
hypothesis and produce exactly that, so the hypothesis set is tight in the
direction that matters — none of the three is decoration.
-/
theorem predict_isDistribution (n : Nat) (B : Transition) (q : List Rat)
    (hl : B.length = q.length) (hw : ∀ c ∈ B, c.length = n)
    (hcol : ∀ c ∈ B, IsDistribution c) (hq : IsDistribution q) :
    IsDistribution (predict n B q) :=
  ⟨predict_nonneg n B q (fun c hc => (hcol c hc).1) hq.1,
   by rw [predict_sum n B q hl hw (fun c hc => (hcol c hc).2)]; exact hq.2⟩

/-! ### Satisfiability witnesses for the prediction half

Same discipline as the correction half above: every hypothesis of P12 exhibited
inhabited, the conclusion reached on a case where prediction does real mixing,
and each hypothesis refuted in turn. Synthetic numerals throughout — no measured
value appears in this module. -/

/-- A genuinely mixing two-state operator: neither column is a point mass, so
the prediction is not a permutation of the prior and P12's conclusion is not
reachable by inspection. -/
def transWitness : Transition := [[3/4, 1/4], [1/2, 1/2]]

/-- An asymmetric prior, so a bug that swapped the two columns would change the
result. A uniform prior would hide one. -/
def priorWitness : List Rat := [1/3, 2/3]

/-- P12's `B.length = q.length` hypothesis is inhabited. -/
theorem transWitness_length : transWitness.length = priorWitness.length := by
  simp [transWitness, priorWitness]

/-- P12's width hypothesis is inhabited at `n = 2`. -/
theorem transWitness_width : ∀ c ∈ transWitness, c.length = 2 := by
  intro c hc
  simp only [transWitness, List.mem_cons, List.not_mem_nil, or_false] at hc
  rcases hc with rfl | rfl <;> simp

/-- P12's column-stochasticity hypothesis is inhabited. -/
theorem transWitness_columns : ∀ c ∈ transWitness, IsDistribution c := by
  intro c hc
  simp only [transWitness, List.mem_cons, List.not_mem_nil, or_false] at hc
  rcases hc with rfl | rfl <;>
    refine ⟨?_, ?_⟩ <;>
    first
      | (intro x hx
         simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
         rcases hx with rfl | rfl <;> grind)
      | (simp only [List.sum_cons, List.sum_nil]; grind)

/-- P12's prior hypothesis is inhabited. -/
theorem priorWitness_isDistribution : IsDistribution priorWitness := by
  refine ⟨?_, ?_⟩
  · intro x hx
    simp only [priorWitness, List.mem_cons, List.not_mem_nil, or_false] at hx
    rcases hx with rfl | rfl <;> grind
  · simp only [priorWitness, List.sum_cons, List.sum_nil]; grind

/-- The witness prediction is `[7/12, 5/12]` — equal to neither column nor to
the prior. Pinned because a witness on which `predict` acted as the identity, or
as a column lookup, would satisfy P12 while exercising no mixing at all; that is
the same defect `evWitness` guards against for `normalize`. -/
theorem predictWitness_mixes : predict 2 transWitness priorWitness = [7/12, 5/12] := by
  simp only [predict, transWitness, priorWitness, smul, vadd, zeroVec,
    List.map_cons, List.map_nil, List.replicate, List.zipWith_cons_cons,
    List.zipWith_nil_right, List.cons.injEq]
  grind

/-- All four hypotheses fire together and P12's conclusion is reached. -/
theorem predictWitness_isDistribution : IsDistribution (predict 2 transWitness priorWitness) :=
  predict_isDistribution 2 transWitness priorWitness transWitness_length transWitness_width
    transWitness_columns priorWitness_isDistribution

/-- A ragged operator: the second column is a distribution over **one** state in
a two-state model. Nothing about it is type-incorrect, and every check that
looks at one column at a time passes. -/
def raggedTransition : Transition := [[0, 1], [1]]

/-- A uniform two-state prior, shared by two of the counterexamples. -/
def uniformPrior : List Rat := [1/2, 1/2]

/-- **Negative control for P12's width hypothesis.** The prediction carries mass
`1/2`: `vadd` truncated against the short column and the first state's
contribution was dropped.

The other three hypotheses hold (`raggedTransition_meets_the_other_hypotheses`),
so this refutes the width hypothesis specifically rather than exhibiting a
generally malformed input. -/
theorem raggedColumnLosesMass : ¬ IsDistribution (predict 2 raggedTransition uniformPrior) := by
  rintro ⟨-, h⟩
  simp only [predict, raggedTransition, uniformPrior, smul, vadd, zeroVec,
    List.map_cons, List.map_nil, List.replicate, List.zipWith_cons_cons,
    List.zipWith_nil_left, List.zipWith_nil_right, List.sum_cons, List.sum_nil] at h
  grind

/-- The ragged operator satisfies everything P12 asks except width. With the
previous theorem, exactly one hypothesis is doing the work. -/
theorem raggedTransition_meets_the_other_hypotheses :
    raggedTransition.length = uniformPrior.length ∧
      (∀ c ∈ raggedTransition, IsDistribution c) ∧ IsDistribution uniformPrior := by
  refine ⟨by simp [raggedTransition, uniformPrior], ?_, ?_⟩
  · intro c hc
    simp only [raggedTransition, List.mem_cons, List.not_mem_nil, or_false] at hc
    rcases hc with rfl | rfl <;>
      refine ⟨?_, ?_⟩ <;>
      first
        | (intro x hx
           simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
           rcases hx with rfl | rfl <;> grind)
        | (simp only [List.sum_cons, List.sum_nil]; grind)
  · refine ⟨?_, ?_⟩
    · intro x hx
      simp only [uniformPrior, List.mem_cons, List.not_mem_nil, or_false] at hx
      rcases hx with rfl | rfl <;> grind
    · simp only [uniformPrior, List.sum_cons, List.sum_nil]; grind

/-- An operator whose first column sums to `1/2` — the shape a row-stochastic
matrix presents to a column-major consumer, and the shape a truncated or
renormalization-skipped column has. -/
def leakyTransition : Transition := [[1/2, 0], [0, 1]]

/-- A point mass on the state whose column leaks, so the defect is not averaged
away by the prior. -/
def pointPrior : List Rat := [1, 0]

/-- **Negative control for P12's column-stochasticity hypothesis.** Widths are
right, lengths are right, the prior is a distribution, and the prediction still
carries mass `1/2`. This is the one the correction step would hide: `normalize`
divides by whatever mass it is handed, so the leak leaves no trace in `q_t`. -/
theorem leakyColumnLosesMass : ¬ IsDistribution (predict 2 leakyTransition pointPrior) := by
  rintro ⟨-, h⟩
  simp only [predict, leakyTransition, pointPrior, smul, vadd, zeroVec,
    List.map_cons, List.map_nil, List.replicate, List.zipWith_cons_cons,
    List.zipWith_nil_right, List.sum_cons, List.sum_nil] at h
  grind

/-- The leaky operator satisfies everything P12 asks except column
stochasticity. -/
theorem leakyTransition_meets_the_other_hypotheses :
    leakyTransition.length = pointPrior.length ∧
      (∀ c ∈ leakyTransition, c.length = 2) ∧ IsDistribution pointPrior := by
  refine ⟨by simp [leakyTransition, pointPrior], ?_, ?_⟩
  · intro c hc
    simp only [leakyTransition, List.mem_cons, List.not_mem_nil, or_false] at hc
    rcases hc with rfl | rfl <;> simp
  · refine ⟨?_, ?_⟩
    · intro x hx
      simp only [pointPrior, List.mem_cons, List.not_mem_nil, or_false] at hx
      rcases hx with rfl | rfl <;> grind
    · simp only [pointPrior, List.sum_cons, List.sum_nil]; grind

/-- A one-state operator, paired below with a two-state prior: the state whose
column is missing is the state whose mass disappears. -/
def shortTransition : Transition := [[1]]

/-- **Negative control for P12's `B.length = q.length` hypothesis.** A prior over
two states against an operator with one column: the second state's `1/2` has
nowhere to go and the prediction carries mass `1/2`.

This is what an off-by-one state space does — a vocabulary extended in the prior
but not in the transition operator — and it is silent, because `predict`'s
fall-through returns a well-formed vector of the declared width. -/
theorem missingColumnLosesMass : ¬ IsDistribution (predict 1 shortTransition uniformPrior) := by
  rintro ⟨-, h⟩
  simp only [predict, shortTransition, uniformPrior, smul, vadd, zeroVec,
    List.map_cons, List.map_nil, List.replicate, List.zipWith_cons_cons,
    List.zipWith_nil_right, List.sum_cons, List.sum_nil] at h
  grind

/-- The short operator satisfies everything P12 asks except the length match:
its single column is a width-1 distribution and the prior is a distribution. -/
theorem shortTransition_meets_the_other_hypotheses :
    (∀ c ∈ shortTransition, c.length = 1) ∧
      (∀ c ∈ shortTransition, IsDistribution c) ∧ IsDistribution uniformPrior := by
  refine ⟨?_, ?_, ?_⟩
  · intro c hc
    simp only [shortTransition, List.mem_cons, List.not_mem_nil, or_false] at hc
    rcases hc with rfl; simp
  · intro c hc
    simp only [shortTransition, List.mem_cons, List.not_mem_nil, or_false] at hc
    rcases hc with rfl
    refine ⟨?_, ?_⟩
    · intro x hx
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
      rcases hx with rfl; grind
    · simp only [List.sum_cons, List.sum_nil]; grind
  · refine ⟨?_, ?_⟩
    · intro x hx
      simp only [uniformPrior, List.mem_cons, List.not_mem_nil, or_false] at hx
      rcases hx with rfl | rfl <;> grind
    · simp only [uniformPrior, List.sum_cons, List.sum_nil]; grind

end AILM
