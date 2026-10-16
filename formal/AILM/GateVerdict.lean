/-
  AILM.GateVerdict — the gate taxonomy of `manuscript/03a_gate_design.md`.

  Formalises three manuscript objects:

  * `eq:gate_satisfies` — the declared (inclusive) relation `x ⪰ t` and the
    strict audit relation `x ≻ t`, with the strictness that makes equality a
    failure in *both* directions;
  * the comparator typing of `tbl:gate_audit` — every measured ceiling carries
    a `ComparatorKind`, and only a bound-kind comparator licenses an
    impossibility verdict;
  * `eq:gate_verdict` — the population-indexed classification
    VACUOUS / UNREACHABLE / ABOVE_REFERENCE / UNAUDITED /
    BRACKETED_BOUND / BRACKETED_REFERENCE.

  `classify` below is a direct transcription of `classify_gate` in
  `src/ailm/gates.py`, including the UNAUDITED branch and the *case order*.
  The case order is itself a claim the manuscript argues for in prose
  ("The order of the cases in [@eq:gate_verdict] is not arbitrary"); `no_co_fire`
  is that argument machine-checked.

  The comparator typing exists because an external audit found the previous
  four-verdict taxonomy licensing `UNREACHABLE` from any measured ceiling while
  the prose admitted some ceilings are exceedable references. `c_ref ≺ t` does
  not imply that every model of the class sits below `t`; only a bound does
  that. G10 and G11 below are that repair machine-checked: an impossibility
  verdict is *derivable only* from a bound-kind comparator, and a
  reference-kind comparator failing the bar yields `ABOVE_REFERENCE`, which
  asserts placement relative to that comparator and nothing more.

  What is deliberately NOT here: an `evaluate_gate` analogue returning a signed
  `margin`. `margin` is `x - t`, a `Rat` subtraction, and `Rat` arithmetic does
  not kernel-reduce (see README.md § "Why every instance fact is an order
  comparison"), so no `decide`d instance could ever use it. A theorem
  "margin is positive exactly when passes" would additionally be a restatement
  of the definition of `gate_margin`, which is decoration, not content.
-/
import AILM.Core

namespace AILM

/-- Which way a metric improves. `perplexity` is `lowerIsBetter`; `accuracy` and
`agreement` are `higherIsBetter`. Mirrors the `Direction` type alias in
`src/ailm/gates.py`. -/
inductive Direction where
  | higherIsBetter
  | lowerIsBetter
  deriving DecidableEq, Repr

/-- What kind of quantity a gate's ceiling is. Mirrors `ComparatorType` in
`src/ailm/gates.py` and the comparator-type column of `tbl:gate_audit`.

The distinction that matters is `isBound`: whether the number bounds every
model of the class on the scored population. Only a bound turns "the ceiling
fails the threshold" into "no implementation could pass". -/
inductive ComparatorKind where
  /-- A theorem-backed bound on every model of the class, measured on the same
  population the gate scores (Stage 1's evaluation-passage conditional
  entropy, via `eq:heldout_bound`). -/
  | modelClassBound
  /-- A closed-form bound needing no measurement (generation's zero
  total-variation distance). -/
  | analyticBound
  /-- A measured comparator given information the model forgoes (Stage 3's
  held-out word model, Stage 5's type oracle, Stage 6's fitted-on-test
  distribution). Strong, but exceedable in principle. -/
  | privilegedOracle
  /-- A measured system with no privileged access (Stage 2B's and Stage 4's
  forward-backward smoothing). Explicitly exceedable — the Stage 4 derived
  row's shipped agent scores above its smoothing reference. -/
  | referenceModel
  deriving DecidableEq, Repr

/-- Whether a comparator kind licenses an impossibility verdict. -/
def ComparatorKind.isBound : ComparatorKind → Bool
  | .modelClassBound => true
  | .analyticBound => true
  | .privilegedOracle => false
  | .referenceModel => false

/-- Epistemic status of a *threshold's placement* — never of a model's outcome.
Mirrors `GateVerdict` in `src/ailm/gates.py`. -/
inductive Verdict where
  | vacuous
  | unreachable
  | aboveReference
  | unaudited
  | bracketedBound
  | bracketedReference
  deriving DecidableEq, Repr

/-- Whether a verdict places the threshold strictly between the baseline and
the measured ceiling — of either comparator strength. Mirrors
`verdict_is_bracketed` in `src/ailm/gates.py`. -/
def Verdict.isBracketed : Verdict → Bool
  | .bracketedBound => true
  | .bracketedReference => true
  | _ => false

/-- The bracketed verdict a comparator of kind `k` produces. -/
def bracketedFor (k : ComparatorKind) : Verdict :=
  if k.isBound then .bracketedBound else .bracketedReference

/-- The failing-ceiling verdict a comparator of kind `k` produces. Only a
bound-kind comparator yields the impossibility verdict. -/
def beyondFor (k : ComparatorKind) : Verdict :=
  if k.isBound then .unreachable else .aboveReference

/-- The strict audit relation `x ≻ t` of `eq:gate_satisfies`. This is
`gate_passes` in `src/ailm/gates.py`: equality is a failure in both directions. -/
def passes : Direction → Rat → Rat → Prop
  | .higherIsBetter, x, t => t < x
  | .lowerIsBetter, x, t => x < t

/-- The inclusive declared relation `x ⪰ t` of `eq:gate_satisfies` — what a
threshold's own wording (`>=2 nats`, `>=60%`) means by "pass". No Python
function implements this; it exists only as the declared-threshold wording, and
`passes_iff_satisfies_and_ne` is what ties the two together. -/
def satisfies : Direction → Rat → Rat → Prop
  | .higherIsBetter, x, t => t ≤ x
  | .lowerIsBetter, x, t => x ≤ t

instance instDecidablePasses (d : Direction) (x t : Rat) : Decidable (passes d x t) := by
  cases d <;> simp only [passes] <;> infer_instance

instance instDecidableSatisfies (d : Direction) (x t : Rat) : Decidable (satisfies d x t) := by
  cases d <;> simp only [satisfies] <;> infer_instance

/-- `eq:gate_verdict`, transcribed from `classify_gate` in `src/ailm/gates.py`
with the same case order: VACUOUS is decided *before* the ceiling is consulted,
and a missing ceiling yields UNAUDITED rather than a guess.

`b` is the measured trivial baseline `b_P`; `c` the measured ceiling *paired
with its declared comparator kind* (`none` when no ceiling was measured); `t`
the declared threshold. The pairing is the audit repair: a ceiling cannot reach
the classifier without its kind, so an untyped comparator cannot silently
acquire bound semantics. -/
def classify (d : Direction) (b : Rat) (c : Option (Rat × ComparatorKind)) (t : Rat) :
    Verdict :=
  if passes d b t then .vacuous
  else
    match c with
    | none => .unaudited
    | some (cv, k) => if passes d cv t then bracketedFor k else beyondFor k

/--
**G1.** No metric strictly passes a threshold equal to itself.

Manuscript: `eq:gate_satisfies`, and the sentence "equality fails in both
directions, so a baseline sitting exactly on the bar is never certified sound
and a ceiling sitting exactly on the bar never certifies reachability".

FALSIFIER: an implementation of `passes` built on `≤`/`≥` instead of `<`/`>`.
Such a gate would certify a baseline that merely ties its bar, which is exactly
the boundary case `src/ailm/gates.py` documents as load-bearing. Unconditional,
so there is no hypothesis to witness.
-/
theorem passes_irrefl (d : Direction) (x : Rat) : ¬ passes d x x := by
  cases d <;> simp only [passes] <;> grind

/--
**G2.** The strict audit relation refines the declared inclusive relation.

Manuscript: `eq:gate_satisfies` — the two relations disagree only on the
boundary `x = t`.

That boundary is measure-zero as a *set* and is anything but rare *here*:
`scripts/audit_gates.py` sets every audited row's threshold to its measured
baseline, so `b = t` exactly on all eight rows of `AUDITED_GATE_IDS` — of which
GT1-GT6 below are the six instantiated here. Under `satisfies` each of those
baselines would clear its own bar and every audited gate would read vacuous;
under `passes` none does. The refinement direction proved below is the one that
makes the audit able to say anything, which is why it is stated as an
implication rather than an equivalence.

FALSIFIER: a direction-inverted implementation, where `passes` and `satisfies`
compare on opposite sides. Then a metric could strictly pass while failing the
threshold as declared, and the audit would be reporting about a different gate
than the one that was written down.
-/
theorem satisfies_of_passes {d : Direction} {x t : Rat} (h : passes d x t) :
    satisfies d x t := by
  cases d <;> simp only [passes] at h <;> simp only [satisfies] <;> grind

/--
**G3.** `x ≻ t` is *exactly* `x ⪰ t ∧ x ≠ t`.

Manuscript: `eq:gate_satisfies`, final clause, verbatim.

FALSIFIER: any strict relation that is not the inclusive one minus the boundary
— e.g. one that also excluded some `x ≠ t`, or one that accepted `x = t` in a
single direction. This is the theorem that makes G1 and G2 a *characterization*
rather than two separate facts.
-/
theorem passes_iff_satisfies_and_ne (d : Direction) (x t : Rat) :
    passes d x t ↔ (satisfies d x t ∧ x ≠ t) := by
  cases d <;> simp only [passes, satisfies] <;> constructor <;> grind

/-! Reduction lemmas: `classify` applied to a literal `some`/`none` ceiling is
definitionally an if-cascade, and stating that once as `rfl` lets every proof
below work by ordinary `rw` on `ite` instead of fighting the match. -/

theorem classify_some_cases (d : Direction) (b cv t : Rat) (k : ComparatorKind) :
    classify d b (some (cv, k)) t
      = if passes d b t then Verdict.vacuous
        else if passes d cv t then bracketedFor k else beyondFor k := rfl

theorem classify_none_cases (d : Direction) (b t : Rat) :
    classify d b none t
      = if passes d b t then Verdict.vacuous else Verdict.unaudited := rfl

/-- The two bracketed verdicts are the image of `bracketedFor`, and the two
failing-ceiling verdicts the image of `beyondFor`; the images are disjoint.
Small structural facts the characterization theorems below lean on. -/
theorem bracketedFor_isBracketed (k : ComparatorKind) : (bracketedFor k).isBracketed = true := by
  cases k <;> rfl

theorem beyondFor_not_isBracketed (k : ComparatorKind) :
    (beyondFor k).isBracketed = false := by
  cases k <;> rfl

theorem bracketedFor_ne_beyondFor (k k' : ComparatorKind) :
    bracketedFor k ≠ beyondFor k' := by
  cases k <;> cases k' <;> decide

theorem beyondFor_eq_unreachable_iff (k : ComparatorKind) :
    beyondFor k = Verdict.unreachable ↔ k.isBound = true := by
  cases k <;> decide

theorem beyondFor_eq_aboveReference_iff (k : ComparatorKind) :
    beyondFor k = Verdict.aboveReference ↔ k.isBound = false := by
  cases k <;> decide

/--
**G4.** A bracketed verdict fires exactly when the baseline fails and the
ceiling clears — and the verdict's *flavor* is fixed by the comparator's kind,
never by the numbers.

Manuscript: Definition *Threshold placement* — "bracketed-by-bound if the
comparator is a bound and bracketed-by-reference otherwise".

FALSIFIER: a classifier that reports a bracketed verdict when the baseline
already clears the bar (a vacuous gate mislabeled sound), when the ceiling does
not clear it (a beyond-ceiling gate mislabeled sound), or that lets a
reference-kind comparator produce `bracketedBound`. Any of these would license
a threshold with a strength its comparator does not support.
-/
theorem bracketed_iff (d : Direction) (b cv t : Rat) (k : ComparatorKind) :
    classify d b (some (cv, k)) t = bracketedFor k ↔ (¬ passes d b t ∧ passes d cv t) := by
  rw [classify_some_cases]
  by_cases hb : passes d b t
  · rw [if_pos hb]
    constructor
    · intro h
      exact absurd h (by cases k <;> decide)
    · rintro ⟨hb', -⟩
      exact absurd hb hb'
  · rw [if_neg hb]
    by_cases hc : passes d cv t
    · rw [if_pos hc]
      exact ⟨fun _ => ⟨hb, hc⟩, fun _ => rfl⟩
    · rw [if_neg hc]
      constructor
      · intro h
        exact absurd h.symm (bracketedFor_ne_beyondFor k k)
      · rintro ⟨-, hc'⟩
        exact absurd hc' hc

/--
**G5.** VACUOUS fires exactly when the baseline clears the bar — whatever the
ceiling is, including when there is none, and whatever kind it carries.

Manuscript: `eq:gate_verdict`, first case; and the Stage 2B diagnosis "the
declared threshold sits below the trivial baseline, so a model doing nothing
hierarchical passes ... and it would have remained vacuous on the correct
corpus".

FALSIFIER: a classifier whose VACUOUS branch consults `c`. The `∀ c` here is the
content: it says the vacuity diagnosis survives *any* ceiling measurement,
including no measurement at all, which is why a vacuous gate can be diagnosed
before a ceiling exists.
-/
theorem vacuous_iff (d : Direction) (b : Rat) (c : Option (Rat × ComparatorKind)) (t : Rat) :
    classify d b c t = .vacuous ↔ passes d b t := by
  by_cases hb : passes d b t
  · refine ⟨fun _ => hb, fun _ => ?_⟩
    cases c with
    | none => rw [classify_none_cases, if_pos hb]
    | some pair =>
      obtain ⟨cv, k⟩ := pair
      rw [classify_some_cases, if_pos hb]
  · refine ⟨fun h => ?_, fun h => absurd h hb⟩
    exfalso
    cases c with
    | none =>
      rw [classify_none_cases, if_neg hb] at h
      exact absurd h (by decide)
    | some pair =>
      obtain ⟨cv, k⟩ := pair
      rw [classify_some_cases, if_neg hb] at h
      by_cases hc : passes d cv t
      · rw [if_pos hc] at h
        exact absurd h (by cases k <;> decide)
      · rw [if_neg hc] at h
        exact absurd h (by cases k <;> decide)

/--
**G6.** Whenever the ceiling dominates the baseline, VACUOUS and the
failing-ceiling verdicts cannot both apply.

Manuscript: `sec:gate_design`, "The order of the cases in [@eq:gate_verdict] is
not arbitrary. Under the strict relation, VACUOUS and UNREACHABLE cannot
co-fire whenever `c_P ⪰ b_P`: if `b_P ≻ t` and `c_P ⪰ b_P`, then `c_P ≻ t` ...
Reporting VACUOUS first therefore loses no information."

The hypothesis `dom : ¬ passes d b cv` says the baseline does not strictly beat
the ceiling, i.e. the ceiling bounds the *baseline*. It holds on every recorded
row, and it is checked there rather than assumed.

Do **not** strengthen that to "the ceiling bounds every model in the class".
It is false on recorded data for the reference-kind comparators: the Stage 4
derived row's shipped agent scores above its recorded
`stage4_derived_smoothing` ceiling. That distinction is exactly what
`ComparatorKind.isBound` now carries, and G10 is where it becomes
verdict-bearing. G6 needs only the weaker fact about the baseline and gets it.

FALSIFIER: a triple `(b, cv, t)` with the ceiling dominating the baseline for
which the baseline clears the bar but the ceiling does not. Such a triple would
make the case order of `eq:gate_verdict` information-destroying: reporting
VACUOUS first would hide a live failing-ceiling diagnosis.

NON-VACUITY: `dom` is inhabited — see `no_co_fire_hypothesis_inhabited` below,
and note that the degenerate `b = cv` case satisfies it by `passes_irrefl`.
-/
theorem no_co_fire {d : Direction} {b cv t : Rat} (dom : ¬ passes d b cv)
    (hb : passes d b t) : passes d cv t := by
  cases d <;> simp only [passes] at * <;> grind

/--
**G7.** When the baseline equals the ceiling, *no* threshold is sound —
whatever kind the degenerate comparator is declared to be.

Manuscript: `sec:gate_design`, "when `b_P = c_P` the sound region is empty, and
no threshold on that population can be bracketed" — a degenerate population is
itself a measurement worth having before building the level.

The `∀ t` is the whole point. A per-threshold statement would only say that
some particular bar fails; this says the *population* supports no gate at all,
so no amount of threshold tuning rescues it.

FALSIFIER: a single rational `t` at which a population whose baseline equals
its ceiling nonetheless classifies bracketed.

This is instantiable on real data, and is: `stage4Gold_no_threshold_is_sound`
in `AILM/Instances.lean` applies it to the Stage 4 **gold-morphology**
population, whose recorded baseline, ceiling and model are all one number.
-/
theorem degenerate_no_gate (d : Direction) (b t : Rat) (k : ComparatorKind) :
    (classify d b (some (b, k)) t).isBracketed = false := by
  rw [classify_some_cases]
  by_cases hp : passes d b t
  · rw [if_pos hp]
    rfl
  · rw [if_neg hp, if_neg hp]
    exact beyondFor_not_isBracketed k

/--
**G8.** When neither the baseline nor a *bound-kind* ceiling clears the bar,
the verdict is UNREACHABLE.

Manuscript: `eq:gate_verdict`; applied at Stage 1, where "the declared target
exceeds that ceiling ... No letter bigram model — not a better smoothed one,
not a better implemented one — could have passed it." The kind hypothesis is
what makes that sentence sayable: Stage 1's comparator is the evaluation
passage's own conditional entropy, a model-class bound.

FALSIFIER: a classifier reporting UNAUDITED or a bracketed verdict in this
configuration. Combined with a `∀ b`, this is what lets the Stage 1 diagnosis
be stated for *every* baseline rather than for the one that happened to be
measured.

NON-VACUITY: both hypotheses are inhabited — see
`unreachable_hypotheses_inhabited` below.
-/
theorem unreachable_of_bound_fails {d : Direction} {b cv t : Rat} {k : ComparatorKind}
    (hk : k.isBound = true) (hb : ¬ passes d b t) (hc : ¬ passes d cv t) :
    classify d b (some (cv, k)) t = .unreachable := by
  rw [classify_some_cases, if_neg hb, if_neg hc]
  exact (beyondFor_eq_unreachable_iff k).mpr hk

/--
**G8b.** The same configuration with a *reference-kind* ceiling yields
ABOVE_REFERENCE — the threshold sits above that comparator, and no
impossibility claim follows, because another model in the admissible class may
still exceed a reference.

This is the external audit's central correction, stated positively:
`c_ref ≺ t` licenses exactly "above that reference" and nothing stronger.

FALSIFIER: a classifier that emits UNREACHABLE here. That is what the previous
four-verdict taxonomy did, and it is the overreach this kind-split exists to
remove.
-/
theorem above_reference_of_reference_fails {d : Direction} {b cv t : Rat}
    {k : ComparatorKind} (hk : k.isBound = false) (hb : ¬ passes d b t)
    (hc : ¬ passes d cv t) :
    classify d b (some (cv, k)) t = .aboveReference := by
  rw [classify_some_cases, if_neg hb, if_neg hc]
  exact (beyondFor_eq_aboveReference_iff k).mpr hk

/--
**G9.** When the threshold *is* the baseline, a bracketed verdict reduces to a
**single** order comparison: the ceiling must strictly beat the baseline.

This is the disclosure that `AILM/Instances.lean` § "Stage gates" needs, stated
as a theorem instead of as prose. `scripts/audit_gates.py` sets
`threshold := baseline` on every audited row, so `LedgerData.lean` emits each
`…Threshold` as a definitional alias of its `…Baseline`. Under that convention
the first conjunct of G4 — "the baseline does not clear the bar" — is discharged
by G1 unconditionally, for every population and every direction. It is a
tautology, not a measurement.

So the bracketed GT1-GT5 in `Instances.lean` are **one-sided**: they say the
recorded comparator strictly beats the recorded baseline, and nothing else.
GT6 supplies the complementary measured case: its comparator does not beat the
baseline and the reference-typed row classifies ABOVE_REFERENCE. NC5-NC10
separately show `classify` is not constant on the same recorded population.
A reader who took a bracketed verdict here to mean two independent measurements
cleared two independent hurdles would be wrong, and this theorem is where that
is written down.

FALSIFIER: a classifier for which some triple `(b, cv, b)` classifies bracketed
while `cv` does not strictly beat `b`, or fails to while it does. Either would
mean the threshold-equals-baseline convention carries information this theorem
says it cannot — and the disclosure above would be overstating the loss.
-/
theorem bracketed_at_baseline_bar_iff (d : Direction) (b cv : Rat) (k : ComparatorKind) :
    classify d b (some (cv, k)) b = bracketedFor k ↔ passes d cv b := by
  rw [bracketed_iff]
  exact ⟨fun h => h.2, fun h => ⟨passes_irrefl d b, h⟩⟩

/--
**G10.** An impossibility verdict is derivable **only** from a bound-kind
comparator.

This is the audit repair as a single citable statement: whatever the numbers
are, `classify` cannot emit UNREACHABLE unless the comparator was declared a
model-class or analytic bound. `c_reference ≺ t` therefore never reads as
"no model of the class passes" — the inference the external audit showed the
previous taxonomy licensing without support.

FALSIFIER: any `(d, b, cv, t)` and non-bound `k` for which the classifier
emits UNREACHABLE. G8b shows the configuration that used to produce one now
produces ABOVE_REFERENCE instead.
-/
theorem unreachable_requires_bound {d : Direction} {b cv t : Rat} {k : ComparatorKind}
    (h : classify d b (some (cv, k)) t = .unreachable) : k.isBound = true := by
  rw [classify_some_cases] at h
  by_cases hb : passes d b t
  · rw [if_pos hb] at h
    exact absurd h (by decide)
  · rw [if_neg hb] at h
    by_cases hc : passes d cv t
    · rw [if_pos hc] at h
      exact absurd h (by cases k <;> decide)
    · rw [if_neg hc] at h
      exact (beyondFor_eq_unreachable_iff k).mp h

/--
**G11.** Dually, ABOVE_REFERENCE is derivable only from an exceedable
comparator: a bound that fails the bar is never softened into a reference
statement.

FALSIFIER: a bound-kind comparator producing ABOVE_REFERENCE. That would
discard a valid impossibility diagnosis, which is the loss in the *other*
direction — the taxonomy must neither overclaim from a reference nor
underclaim from a bound.
-/
theorem above_reference_requires_exceedable {d : Direction} {b cv t : Rat}
    {k : ComparatorKind} (h : classify d b (some (cv, k)) t = .aboveReference) :
    k.isBound = false := by
  rw [classify_some_cases] at h
  by_cases hb : passes d b t
  · rw [if_pos hb] at h
    exact absurd h (by decide)
  · rw [if_neg hb] at h
    by_cases hc : passes d cv t
    · rw [if_pos hc] at h
      exact absurd h (by cases k <;> decide)
    · rw [if_neg hc] at h
      exact (beyondFor_eq_aboveReference_iff k).mp h

/-! ## Satisfiability witnesses

Every conditional theorem above needs an instance proving its hypothesis is
inhabited; a conditional theorem with an unsatisfiable hypothesis builds clean
and proves nothing.

The numbers below are **synthetic shape witnesses**, not measurements: this
module is the general theory, and the ledger-backed instances (Stage 1, 3, 4)
land in a separate generated `AILM/LedgerData.lean` + `AILM/Instances.lean`
pair. No recorded value is hand-typed anywhere in this tree. -/

/-- `no_co_fire`'s hypothesis is inhabited, in the degenerate form that matters:
a population whose ceiling equals its baseline. -/
example : ¬ passes Direction.higherIsBetter (lit 1 2) (lit 1 2) :=
  passes_irrefl _ _

/-- `no_co_fire`'s hypothesis is inhabited in the strict form too: a ceiling
that genuinely dominates its baseline (higher-is-better, `1/2 < 3/4`). -/
theorem no_co_fire_hypothesis_inhabited :
    ¬ passes Direction.higherIsBetter (lit 1 2) (lit 3 4) := by decide

/-- ... and the conclusion of `no_co_fire` is then non-trivially derivable at a
threshold the baseline actually clears. -/
example : passes Direction.higherIsBetter (lit 3 4) (lit 1 4) :=
  no_co_fire no_co_fire_hypothesis_inhabited (by decide)

/-- `unreachable_of_bound_fails` has both hypotheses inhabited simultaneously:
baseline `1/2`, ceiling `3/4`, threshold `9/10` — neither clears. -/
theorem unreachable_hypotheses_inhabited :
    ¬ passes Direction.higherIsBetter (lit 1 2) (lit 9 10)
      ∧ ¬ passes Direction.higherIsBetter (lit 3 4) (lit 9 10) := by decide

example :
    classify Direction.higherIsBetter (lit 1 2)
      (some (lit 3 4, ComparatorKind.modelClassBound)) (lit 9 10)
    = Verdict.unreachable :=
  unreachable_of_bound_fails rfl unreachable_hypotheses_inhabited.1
    unreachable_hypotheses_inhabited.2

/-- G8b fires on the same numbers with the kind flipped: the comparator's
declared kind, not any measurement, is what separates the impossibility verdict
from the reference one. -/
example :
    classify Direction.higherIsBetter (lit 1 2)
      (some (lit 3 4, ComparatorKind.referenceModel)) (lit 9 10)
    = Verdict.aboveReference :=
  above_reference_of_reference_fails rfl unreachable_hypotheses_inhabited.1
    unreachable_hypotheses_inhabited.2

/-! Each verdict is *reachable* — a classifier that could never emit one of
these would make the corresponding theorem vacuously safe. All six fire. -/

example : classify Direction.higherIsBetter (lit 9 10)
    (some (lit 1 1, ComparatorKind.modelClassBound)) (lit 1 2)
    = Verdict.vacuous := by decide

example : classify Direction.higherIsBetter (lit 1 2)
    (some (lit 3 4, ComparatorKind.modelClassBound)) (lit 9 10)
    = Verdict.unreachable := by decide

example : classify Direction.higherIsBetter (lit 1 2)
    (some (lit 3 4, ComparatorKind.privilegedOracle)) (lit 9 10)
    = Verdict.aboveReference := by decide

example : classify Direction.higherIsBetter (lit 1 2) none (lit 9 10)
    = Verdict.unaudited := by decide

example : classify Direction.higherIsBetter (lit 1 2)
    (some (lit 3 4, ComparatorKind.analyticBound)) (lit 3 5)
    = Verdict.bracketedBound := by decide

example : classify Direction.higherIsBetter (lit 1 2)
    (some (lit 3 4, ComparatorKind.referenceModel)) (lit 3 5)
    = Verdict.bracketedReference := by decide

/-! `passes` is both satisfiable and refutable in each direction — otherwise the
`Decidable` instance would be deciding a constant. -/

example : passes Direction.higherIsBetter (lit 3 4) (lit 1 2) := by decide
example : ¬ passes Direction.higherIsBetter (lit 1 2) (lit 3 4) := by decide
example : passes Direction.lowerIsBetter (lit 1 2) (lit 3 4) := by decide
example : ¬ passes Direction.lowerIsBetter (lit 3 4) (lit 1 2) := by decide

/-! `satisfies` likewise, in both polarities. The boundary examples below exhibit
it holding; these exhibit it **failing**, which is what stops
`instDecidableSatisfies` from being exercised only in the `true` direction. -/

example : satisfies Direction.higherIsBetter (lit 3 4) (lit 1 2) := by decide
example : ¬ satisfies Direction.higherIsBetter (lit 1 2) (lit 3 4) := by decide
example : satisfies Direction.lowerIsBetter (lit 1 2) (lit 3 4) := by decide
example : ¬ satisfies Direction.lowerIsBetter (lit 3 4) (lit 1 2) := by decide

/-! G2 is the one theorem here with no other consumer, so its hypothesis is
inhabited and the implication is fired explicitly, in each direction. Without
these it would rest on `passes` examples that happen to exist for other
reasons. -/

example : satisfies Direction.higherIsBetter (lit 3 4) (lit 1 2) :=
  satisfies_of_passes (by decide)

example : satisfies Direction.lowerIsBetter (lit 1 2) (lit 3 4) :=
  satisfies_of_passes (by decide)

/-! G9's reduction fires in both directions on synthetic triples: with the bar
on the baseline, a ceiling that beats it gives the kind's bracketed verdict and
one that does not gives the kind's failing verdict. Nothing in between is
reachable, which is the content of G9. -/

example : classify Direction.higherIsBetter (lit 1 2)
    (some (lit 3 4, ComparatorKind.modelClassBound)) (lit 1 2)
    = Verdict.bracketedBound := by decide

example : classify Direction.higherIsBetter (lit 1 2)
    (some (lit 1 4, ComparatorKind.modelClassBound)) (lit 1 2)
    = Verdict.unreachable := by decide

example : classify Direction.higherIsBetter (lit 1 2)
    (some (lit 1 4, ComparatorKind.referenceModel)) (lit 1 2)
    = Verdict.aboveReference := by decide

/-- The boundary case, both directions: `satisfies` holds where `passes` fails.
This is the `x = t` disagreement of `eq:gate_satisfies`, exhibited rather than
asserted — and it is the configuration every audited stage-gate row is in, not
an edge case, since the audit sets each threshold to its own baseline. -/
example : satisfies Direction.higherIsBetter (lit 1 2) (lit 1 2)
    ∧ ¬ passes Direction.higherIsBetter (lit 1 2) (lit 1 2) := by decide

example : satisfies Direction.lowerIsBetter (lit 1 2) (lit 1 2)
    ∧ ¬ passes Direction.lowerIsBetter (lit 1 2) (lit 1 2) := by decide

/-- `degenerate_no_gate` is not vacuous for want of a degenerate population:
here is one, and the verdict it does produce is the kind's failing verdict. -/
example : classify Direction.higherIsBetter (lit 1 2)
    (some (lit 1 2, ComparatorKind.modelClassBound)) (lit 3 4)
    = Verdict.unreachable := by decide

example : classify Direction.higherIsBetter (lit 1 2)
    (some (lit 1 2, ComparatorKind.privilegedOracle)) (lit 3 4)
    = Verdict.aboveReference := by decide

end AILM
