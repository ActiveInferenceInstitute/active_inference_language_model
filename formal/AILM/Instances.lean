/-
  AILM.Instances — the recorded measurements, run through the general theory.

  This is the module that makes the formal layer about *this project* rather
  than about arithmetic. Every other module proves a general fact about gate
  taxonomies, sweep shapes, Bayes normalization or union bounds, and exhibits
  synthetic witnesses to keep those facts non-vacuous. Here the witnesses are
  the real ones: every number reaching a theorem below comes from
  `AILM.LedgerData`, which is generated from `data/claims.json` and cannot be
  edited by hand (`tests/test_formal_isomorphism.py` enforces byte identity).

  ## The division of labour, and why it is not cosmetic

  * `LedgerData.lean` — GENERATED. Numbers only, each naming its ledger label.
  * `Instances.lean`  — HAND-WRITTEN. Structure only: which direction each
    metric improves in, which predicate each sweep is claimed to satisfy, which
    verdict each triple must classify to.

  A number can therefore never be adjusted to make a proof go through — it
  would have to be adjusted in the ledger, by a recorder, from a measurement,
  and the regeneration test would notice. That is the entire point.

  ## Metric direction is structural, not measured

  The `Direction` written against each gate below mirrors the
  `lower_is_better` flag on the corresponding row of `scripts/audit_gates.py`.
  It is a property of the metric (perplexity improves downward, accuracy and
  agreement upward), not something a run measures, so it is hand-written here
  rather than dressed up as ledger data.

  ## What every theorem here is NOT

  None of these are claims that a measurement is *correct*. Lean cannot check
  that the Stage 3 baseline was computed on the right positions; the recorder's
  own guards do that. What Lean checks is that the recorded numbers have the
  logical properties the manuscript and `audit_gates.py` assert of them —
  the interior optimum really is interior, the degenerate population really
  supports no sound threshold, the declared Stage 4 bar really does fail in two
  opposite directions on two recorded populations. Those are exactly the
  statements that were previously asserted in prose and checked by nothing.

  ## A naming constraint

  The incomplete-proof gate is a whole-file text search over `AILM/*.lean`,
  deliberately including comments — a scanner that parsed comments out would be
  one more thing that could be wrong in the direction of reporting clean. The
  cost is that no identifier or comment in this directory may contain either
  placeholder tactic's name as a word. `formal/README.md` § "The naming
  constraint" records the near-miss that established this.
-/
import AILM.Core
import AILM.GateVerdict
import AILM.SweepShape
import AILM.LedgerData

namespace AILM

/-! ## Lexical channel — the novel-message sweep has an interior optimum

`manuscript/09_hierarchy.md` diagnoses the lexical channel as carrying *novel*
information on the strength of its sweep shape: accuracy rises off the anchor,
peaks at an interior weight, and falls again. `fig:channel_comparison` draws it
in the pre-registered "interior optimum" band and its generator refuses to draw
if the recomputed verdict contradicts the band.

Both the figure generator and the recorder decide that shape in Python. The
theorems below decide it again, in the kernel, from the recorded accuracies. -/

/--
**LX1.** The recorded lexical sweep has a strict interior optimum.

This is the manuscript's shape diagnosis, machine-checked on the recorded
values rather than on a synthetic witness shaped like them.

FALSIFIER: a recorded sweep whose argmax sits at the anchor or at the final
weight, or one with a tie at the peak. Any of those would put the channel in
the *other* pre-registered band, and `fig:channel_comparison` would be drawing
a claim its own data contradicts. Note this is falsifiable by a *re-run*: if
`scripts/record_lexical_channel_claims.py` remeasures and the peak moves to a
boundary, regeneration changes `LedgerData.lean` and this proof stops
compiling. That is the intended failure mode.
-/
theorem lexicalSweep_interiorOptimum : InteriorOptimum lexicalSweep := by decide

/-! Reading note (decision 110). After the lexical backoff lookup was corrected
and the sweep re-recorded, LX1 still closes: the recorded argmax is interior.
What the manuscript now says about that fact is narrower than "optimum": the
peak leads the highest tested weight by a margin the paired test does not
resolve, so the theorem is read as the ordering it states, and the pre-registered
"interior optimum" band it licenses is a class the argmax satisfies without
being distinguished from a plateau. The proof is unchanged; the interpretation
in `manuscript/09_hierarchy.md` is where the narrowing lives. -/

/--
**LX2.** The peak sits exactly where the ledger says it does, and it is the
accuracy the ledger calls best.

`lexical_best_alpha`, `lexical_best_accuracy` and the seven
`lexical_sweep_alpha_*_accuracy` rows are separate ledger labels written by one
recorder. Nothing in the ledger format makes them agree.

FALSIFIER: a ledger in which `lexical_best_accuracy` does not equal the sweep
row at `lexical_best_alpha` — i.e. the summary and the table disagreeing about
which weight won. This has a name in this project's own history: two producers
under one label, the value decided by whichever ran last.
-/
theorem lexicalSweep_peak_is_recorded_best :
    PeakAt lexicalSweep lexicalPeakIndex
      ∧ lexicalSweep.getD lexicalPeakIndex 0 = lexicalBestAccuracy := by decide

/--
**LX3.** The anchor row of the sweep is the recorded anchor accuracy.

FALSIFIER: a sweep whose `alpha = 0` row differs from `lexical_anchor_accuracy`.
Since every `delta` in the recorded table is measured *against* the anchor, a
disagreement here would make the whole margin column describe a comparison
nobody ran.
-/
theorem lexicalSweep_anchor_is_recorded :
    lexicalSweep.getD 0 0 = lexicalAnchorAccuracy := by decide

/--
**LX3a.** The peak index selects the recorded best mixing *weight*.

`lexicalPeakIndex` is derived in Python: `generate_lean_ledger.py` looks
`lexical_best_alpha` up in the pre-registered grid and emits the position.
LX2 binds that index to the sweep's argmax; this binds it to the weight the
ledger names, so the derivation is checked in the kernel rather than trusted.
It is also the only thing in the tree that reads `lexicalBestAlpha`, which was
otherwise a recorded value no proof consumed.

FALSIFIER: a ledger whose `lexical_best_alpha` is not the grid entry at the
emitted index — the sweep would then peak at one weight while the ledger's
summary named another, and every "at weight α = …" sentence in `sec:hierarchy`
would be citing a different column than the one that won.
-/
theorem lexical_peak_index_selects_the_recorded_best_alpha :
    lexicalGrid.getD lexicalPeakIndex 0 = lexicalBestAlpha := by decide

/--
**LX4.** The recorder's interior-optimum flag agrees with the kernel's verdict
on the recorder's own numbers.

`lexical_optimum_is_interior` is a boolean the recorder computed in Python and
stored as `1`. This theorem re-derives it in Lean from the recorded sweep and
requires the two to match.

FALSIFIER: a ledger where the flag says `1` and the recorded accuracies do not
in fact have an interior optimum (or the reverse). That is the "metric green by
construction" defect in its most literal form — a stored verdict that no longer
describes the stored data — and it is invisible to any test that reads only the
flag.
-/
theorem lexical_interior_flag_agrees_with_the_sweep :
    (if InteriorOptimum lexicalSweep then 1 else 0) = lexicalOptimumIsInteriorFlag := by
  decide

/-- **LX5.** The lexical channel is therefore *not* in the redundant band. Via
S2, so this is the general exclusion theorem firing on recorded data. -/
theorem lexicalSweep_not_noInteriorGain : ¬ NoInteriorGain lexicalSweep :=
  interior_excludes_no_gain _ lexicalSweep_interiorOptimum

/-- **LX6.** ... and not monotone declining either. Via S3. -/
theorem lexicalSweep_not_monotone : ¬ MonotoneDecline lexicalSweep :=
  interior_excludes_monotone _ lexicalSweep_interiorOptimum

/-! ## Coupling channel — no recorded weight beats the anchor

Read the SCOPE note on `couplingRecorded` in `LedgerData.lean` first. Only
three of the six grid points are recorded, so the honest predicate here is
`NoInteriorGain`, not a shape claim about the whole sweep. -/

/--
**CP1.** The three recorded points strictly decline in the mixing weight.

SCOPE, and it matters: this is a fact about the three recorded points ordered
by alpha, **not** about the full sweep. Alphas 0.25, 0.5 and 0.75 have no
recorded accuracies, so nothing here rules out a bump between them (CP5 pins
that gap as a ledger fact rather than a comment). Citing this as "the coupling
sweep declines monotonically" would be an overclaim of exactly the kind this
repository audits for.

FALSIFIER: a recorded `alpha = 1` accuracy above the recorded best, or a best
above the anchor. What it excludes is the world in which full coupling is not
the worst of the three recorded settings — which is the qualitative story the
manuscript tells about a message the receiver already has.

This is the *strongest* of the three coupling shape facts, so it is the one
decided from the data; CP2 and CP3 are derived from it rather than re-decided.
Stating it and then separately `decide`-ing its own consequence would put two
theorems in `formal_theorem_count` where the second excludes no world the first
had not already excluded.
-/
theorem couplingRecorded_declines_across_recorded_points :
    MonotoneDecline couplingRecorded := by decide

/--
**CP2.** No recorded coupling weight improves on the uncoupled anchor.

This is stronger than a three-point check. The middle entry is the *maximum*
over every `alpha > 0` in the six-point grid (`record_coupling_claims.py`
computes `best_alpha = max(positives, ...)`), so "no entry beats the anchor"
here is exactly "no positive weight in the full grid beats the anchor" — which
is the manuscript's redundant-channel finding.

Derived from CP1 through S1 (`monotone_decline_no_interior_gain`), because it
is strictly implied by it. The manuscript cites this form, so it keeps its own
name; what it does not get is its own independent `decide`.

FALSIFIER: a recorded `coupling_best_accuracy` above `coupling_alpha0_accuracy`.
That would mean some weight did help, the coupling channel would no longer be
the redundant half of the two-clause hypothesis, and
`fig:channel_comparison`'s band assignment would be wrong. Being a corollary,
it now fails by CP1 failing first.
-/
theorem couplingRecorded_noInteriorGain : NoInteriorGain couplingRecorded :=
  monotone_decline_no_interior_gain _ couplingRecorded_declines_across_recorded_points

/--
**CP3.** The coupling channel is therefore excluded from the interior-optimum
band. Contrapositive of S2 on recorded data.

FALSIFIER: the same as CP2. Stated separately because this is the form
`fig:channel_comparison`'s contradiction check needs: the two channels must
land in *different* bands, and LX5/CP3 together are that separation, proved
rather than drawn.
-/
theorem couplingRecorded_not_interiorOptimum : ¬ InteriorOptimum couplingRecorded :=
  fun h => interior_excludes_no_gain _ h couplingRecorded_noInteriorGain

/--
**CP4.** The recorded list is exactly the three recorded scalars, in
mixing-weight order.

`couplingRecorded` and the `coupling*Accuracy` defs above it are emitted by two
different code paths in `scripts/generate_lean_ledger.py` — the list block and
the scalar loop. Nothing in the generator forces them to agree, and before this
was proved, `couplingBestAccuracy` and `couplingAlpha1Accuracy` were ledger
values no proof in the tree ever read.

FALSIFIER: a generator emitting the list in a different order than the scalars
name. Swapping the anchor and the fully-coupled endpoint would turn CP1's
declining shape into a rising one while every proof still compiled, because
nothing else in the tree looks at both representations.
-/
theorem couplingRecorded_entries_are_the_recorded_scalars :
    couplingRecorded.getD 0 0 = couplingAlpha0Accuracy
      ∧ couplingRecorded.getD 1 0 = couplingBestAccuracy
      ∧ couplingRecorded.getD 2 0 = couplingAlpha1Accuracy := by decide

/--
**CP5.** Fewer weights are recorded than the grid holds — the scope caveat as a
checked fact rather than a comment.

`coupling_grid_size` is a ledger value and it says six; three points are
recorded. Until this theorem existed it was a recorded number nothing read,
while the prose strength of CP2 ("no positive weight in the full *six-point*
grid") rested on it.

FALSIFIER: a ledger in which the recorded grid size is three. The caveat on CP1
— that the unrecorded alphas could still bump — would then be describing a gap
that does not exist, and CP1 could be read as a claim about the whole sweep.
The caveat is load-bearing precisely because this inequality holds.
-/
theorem couplingRecorded_covers_part_of_the_grid :
    couplingRecorded.length < couplingGridSize := by decide

/--
**CP6.** The best positive weight lies strictly between the two recorded
endpoints, so the middle entry of `couplingRecorded` belongs where it is placed.

`generate_lean_ledger.py` refuses to emit the list unless this holds; CP6 checks
the shipped result rather than the guard, and is the only thing in the tree that
reads `couplingBestAlpha`.

FALSIFIER: a recorded `coupling_best_alpha` of 0 or 1. At 0 the "best positive
weight" would be the anchor compared with itself; at 1 the list would carry two
entries for one weight. Either makes the alpha-ordering of `couplingRecorded` —
which is the whole content of CP1 — meaningless.
-/
theorem couplingBestAlpha_is_between_the_recorded_endpoints :
    lit 0 1 < couplingBestAlpha ∧ couplingBestAlpha < lit 1 1 := by decide

/--
**CP7.** The two channels' anchors and the shipped Stage 2 model are one number.

Three labels written by three recorders: `lexical_anchor_accuracy`,
`coupling_alpha0_accuracy`, `stage2_g2p_accuracy`. Both channel recorders guard
this in Python (`record_coupling_claims.py` Guard 3 refuses to record when the
anchor drifts from the ledger's Stage 2 accuracy); this checks the stored
result rather than the guard.

FALSIFIER: any of the three drifting from the others. Every margin in both
channel tables is measured against the anchor, so a drift would silently
re-base two independent experiments on two different models.
-/
theorem channel_anchors_agree_with_stage2 :
    lexicalAnchorAccuracy = stage2G2pModel ∧ couplingAlpha0Accuracy = stage2G2pModel := by
  decide

/-! ## Stage gates — every audited row, classified in the kernel

`scripts/audit_gates.py` computes a verdict per row in Python. These theorems
recompute it from the recorded triple through `classify`, which is transcribed
from `classify_gate` in `src/ailm/gates.py`.

The direction on each row mirrors the audit's `lower_is_better` flag, and the
threshold is the baseline because that is what the audit compares against.
Both are stated explicitly rather than inferred.

### READ THIS BEFORE CITING GT1-GT6: they are one-sided

`audit_gates.py` sets `threshold := baseline` on every audited row, so
`LedgerData.lean` emits each `…Threshold` as a definitional *alias* of its
`…Baseline`. By `bracketed_iff` (G4), a bracketed verdict means "the baseline
fails the bar **and** the ceiling clears it" — but with the bar definitionally
equal to the baseline, the first conjunct is `passes_irrefl` (G1). It is a
tautology, for every population and every direction, and no re-measurement can
ever move a baseline across a threshold that moves with it.

`bracketed_at_baseline_bar_iff` (G9) is that reduction, machine-checked: at
`t = b`, a bracketed verdict is *equivalent* to `passes d cv b`. So the entire
measured content of GT1-GT6 is **one order comparison each**. GT1-GT5 assert
that the recorded comparator strictly beats the recorded baseline; GT6 asserts
the opposite and therefore classifies `aboveReference`. These are real,
falsifiable facts about recorded data. They are not two independent hurdles,
and a reader who took a bracketed verdict to mean that would be reading in more
than is here.

The comparator *kind* on each row is structural, like the direction: it mirrors
the `ceiling_type` on the matching `audit_gates.py` row. GT1 and GT2 carry
model-class bounds (evaluation-population entropy floors), so they classify
`bracketedBound`; GT3-GT5 carry a smoothing reference or a privileged oracle
that beats the baseline, so they classify `bracketedReference`. GT6 carries a
privileged oracle that does not beat the baseline and therefore classifies
`aboveReference` — a placement claim whose comparator could in principle be
exceeded.

Two things keep this from being decoration anyway. The comparison that remains
is genuinely falsifiable by re-measurement (a ceiling that stopped beating its
baseline breaks the proof). And NC5-NC10 below drive the *same* recorded
Stage 1 population to five of the six verdicts by moving the bar off the
baseline and flipping the declared kind, which is what shows `classify` is not
constant on real data — a demonstration GT1-GT6 cannot make on their own
precisely because of the aliasing. -/

/-- **GT1.** Stage 1 (letter perplexity, lower is better) is BRACKETED_BOUND:
its comparator is the evaluation passage's own order-1 entropy floor, a
model-class bound. -/
theorem stage1_bracketed :
    classify Direction.lowerIsBetter stage1Baseline
      (some (stage1Ceiling, ComparatorKind.modelClassBound)) stage1Threshold
      = Verdict.bracketedBound := by decide

/-- **GT2.** Stage 2A (phoneme perplexity, lower is better) is BRACKETED.

Population note: this triple is the full-dictionary, all-position one, including
the independently reset start of every word. `audit_gates.py` consumes the same
sufficient statistics and exact denominator. -/
theorem stage2Phoneme_bracketed :
    classify Direction.lowerIsBetter stage2PhonemeBaseline
      (some (stage2PhonemeCeiling, ComparatorKind.modelClassBound))
      stage2PhonemeThreshold = Verdict.bracketedBound := by decide

/-- **GT3.** Stage 2B (G2P accuracy, higher is better) is BRACKETED.

Population note: the full held-out test set, where `audit_gates.py` uses a
strided subsample. Same verdict, different position set. -/
theorem stage2G2p_bracketed :
    classify Direction.higherIsBetter stage2G2pBaseline
      (some (stage2G2pCeiling, ComparatorKind.referenceModel))
      stage2G2pThreshold = Verdict.bracketedReference := by decide

/-- **GT4.** Stage 3 (word per-emission perplexity, lower is better) is
BRACKETED_REFERENCE: its comparator is the held-out-oracle word model, a
privileged-information reference. -/
theorem stage3Word_bracketed :
    classify Direction.lowerIsBetter stage3WordBaseline
      (some (stage3WordCeiling, ComparatorKind.privilegedOracle))
      stage3WordThreshold = Verdict.bracketedReference := by decide

/-- **GT5.** Stage 4 on the derived population (agreement, higher is better) is
BRACKETED_REFERENCE: its comparator is forward-backward smoothing, a reference
the shipped filtering agent measurably exceeds. -/
theorem stage4Derived_bracketed :
    classify Direction.higherIsBetter stage4DerivedBaseline
      (some (stage4DerivedCeiling, ComparatorKind.referenceModel))
      stage4DerivedThreshold = Verdict.bracketedReference := by decide

/-- **GT6.** Stage 5 (clause perplexity, lower is better) is ABOVE_REFERENCE:
neither the baseline-tied bar nor its privileged type-oracle comparator clears
the strict gate. The comparator is exceedable, so this is not an impossibility
claim. See GT12. -/
theorem stage5Clause_above_reference :
    classify Direction.lowerIsBetter stage5ClauseBaseline
      (some (stage5ClauseCeiling, ComparatorKind.privilegedOracle))
      stage5ClauseThreshold = Verdict.aboveReference := by decide

/-! ### Outcomes: five stages clear their bar and one does not

Placement (is the gate meaningful?) and outcome (did the model beat it?) are
deliberately separate in `audit_gates.py`, and they are separate here. GT7-GT11
are the PASS rows; GT12 is the non-passing Stage 5 row.

Together these are what stops the block above from being green by construction:
the same `passes` relation, applied to the same kind of recorded triple, comes
out false exactly once — on the row the ledger and `docs/decisions.md` entry 83
record as non-passing with unresolved gain. -/

/-- **GT7.** The Stage 1 letter bigram beats its unigram bar. -/
theorem stage1_model_passes :
    passes Direction.lowerIsBetter stage1Model stage1Threshold := by decide

/-- **GT8.** The Stage 2A phoneme bigram beats its unigram bar. -/
theorem stage2Phoneme_model_passes :
    passes Direction.lowerIsBetter stage2PhonemeModel stage2PhonemeThreshold := by decide

/-- **GT9.** The Stage 2B G2P agent beats the per-letter argmax bar. -/
theorem stage2G2p_model_passes :
    passes Direction.higherIsBetter stage2G2pModel stage2G2pThreshold := by decide

/-- **GT10.** The Stage 3 word model beats the letter-bigram bar. -/
theorem stage3Word_model_passes :
    passes Direction.lowerIsBetter stage3WordModel stage3WordThreshold := by decide

/-- **GT11.** The Stage 4 case agent beats the context-free bar on the derived
population. -/
theorem stage4Derived_model_passes :
    passes Direction.higherIsBetter stage4DerivedModel stage4DerivedThreshold := by decide

/--
**GT12.** The Stage 5 mixture does **not** beat the type-free bar.

Manuscript: non-passing with unresolved privileged-label gain
(`docs/decisions.md` entry 83). `audit_gates.py` treats a *pass* on this row or
a resolved paired interval as a changed finding, not a success.

FALSIFIER: a recorded `stage5_mixture_perplexity` below
`stage5_typefree_perplexity`. If a re-measurement ever produced one, this proof
would stop compiling — which is the correct alarm, because the recorded finding
would have changed and the manuscript's Stage 5 status would need rewriting.

This is also the theorem that proves GT7-GT11 are not vacuous: `passes` is not
a relation that holds of every recorded pair.
-/
theorem stage5Clause_model_fails :
    ¬ passes Direction.lowerIsBetter stage5ClauseModel stage5ClauseThreshold := by decide

/-! ## The declared gates — the defects this project shipped and diagnosed

The rows above use the *repaired* thresholds. These use the thresholds
originally declared, and they are where the formal layer earns its keep: the
same declared bar produces two different failure modes on two recorded
populations, and a third declared bar is vacuous. -/

/--
**DG1.** The Stage 4 declared bar (>=60%) is ABOVE_REFERENCE on the derived
population: neither the context-free baseline nor the smoothing comparator
clears it — and the comparator is a *reference*, so no impossibility claim
follows.

This theorem used to say UNREACHABLE, and the retype is the external audit's
central correction applied to this project's own record. The smoothing decode
is a measured comparator the shipped agent exceeds on this very population
(see G6's discussion), so "nothing at this model order can reach 60%" was
never licensed by it; what is licensed is that the declared bar sits above
every comparator measured here. `unreachable_requires_bound` (G10) is the
general fact; this row is where it bites recorded data.

Manuscript: `docs/decisions.md` entry 16 — the declared gate "fails in BOTH
directions depending on the population".

FALSIFIER: a derived-population comparator strictly above 60. Not "at or
above" — `passes` is strict, so a comparator sitting exactly on the bar still
fails to clear it, and the verdict stays ABOVE_REFERENCE. With a comparator
above 60 the bar would merely be hard rather than above every measured
reference, and the diagnosis would be wrong about which defect it was.
-/
theorem stage4Derived_declared_gate_above_reference :
    classify Direction.higherIsBetter stage4DerivedBaseline
      (some (stage4DerivedCeiling, ComparatorKind.referenceModel))
      stage4DeclaredGate = Verdict.aboveReference := by decide

/--
**DG2.** The *same* declared bar is VACUOUS on the gold population: the trivial
baseline already clears it.

FALSIFIER: a gold-population baseline at or below 60 — including exactly 60,
since a baseline sitting on the bar does not strictly clear it and the verdict
would move off VACUOUS.

Together with DG1 this is the whole of decision 16 — one threshold, two
populations, two opposite defects — and it is now checked rather than narrated.
Neither theorem alone says it: DG1 without DG2 is a hard gate, DG2 without DG1
is a soft one.
-/
theorem stage4Gold_declared_gate_vacuous :
    classify Direction.higherIsBetter stage4GoldBaseline
      (some (stage4GoldCeiling, ComparatorKind.referenceModel))
      stage4DeclaredGate = Verdict.vacuous := by decide

/--
**DG3.** The gold population is degenerate: its baseline, its ceiling and the
shipped agent are one number.

FALSIFIER: any of the three differing. This is the measurement that makes DG4
apply, and it is why the gold population was abandoned rather than re-tuned.
-/
theorem stage4Gold_is_degenerate :
    stage4GoldBaseline = stage4GoldCeiling ∧ stage4GoldModel = stage4GoldBaseline := by
  decide

/--
**DG4.** Therefore **no** threshold whatsoever is sound on the gold population.

This is G7 (`degenerate_no_gate`) instantiated on recorded data, and the `∀ t`
is the content. A per-threshold statement would only say that 60% was a bad
choice; this says the population supports no gate at all, so no amount of
threshold tuning could have rescued it.

FALSIFIER: a single rational `t` at which the gold triple classifies as
either bracketed verdict. G7 shows there is none, and DG3 supplies the
degeneracy from the ledger — so the falsifier would have to be a change in the
*measurements*, not in the choice of bar.
-/
theorem stage4Gold_no_threshold_is_sound (t : Rat) :
    (classify Direction.higherIsBetter stage4GoldBaseline
      (some (stage4GoldCeiling, ComparatorKind.referenceModel)) t).isBracketed
      = false := by
  rw [← stage4Gold_is_degenerate.1]
  exact degenerate_no_gate _ _ _ _

/--
**DG5.** The Stage 3 declared bar (perplexity < 500) is VACUOUS: the
letter-bigram baseline already clears it.

Manuscript: `docs/decisions.md` entry 14 — declared "by a wide margin" vacuous,
against a corpus this project does not even have.

FALSIFIER: a Stage 3 baseline perplexity at or above 500. The gate would then
have been merely mis-specified about its corpus rather than incapable of
failing.
-/
theorem stage3Word_declared_gate_vacuous :
    classify Direction.lowerIsBetter stage3WordBaseline
      (some (stage3WordCeiling, ComparatorKind.privilegedOracle))
      stage3DeclaredGate = Verdict.vacuous := by decide

/-! ### Stage 1: unreachable for every model at this order, not just for ours

The Stage 1 defect is the opposite of Stage 3's: the declared bar sat above the
information-theoretic ceiling. The manuscript states this universally — "No
letter bigram model, not a better smoothed one, not a better implemented one,
could have passed it" — and a theorem about the one model that was built would
not say that. So the theorem below quantifies over every model bounded by the
recorded ceiling. -/

/--
**DG6.** The recorded order-1 ceiling does not clear the declared 2-nat bar.

The measured fact the universal claim rests on.

FALSIFIER: a recorded `stage1_max_bigram_gain_nats` above 2. The gate would
then have been achievable and Stage 1's diagnosis would be wrong.
-/
theorem stage1_ceiling_misses_declared_gate :
    ¬ passes Direction.higherIsBetter stage1MaxBigramGainNats stage1DeclaredGateNats := by
  decide

/--
**DG7.** Every model whose gain is bounded by the recorded order-1 ceiling is
judged UNREACHABLE against the declared bar.

The `∀ g` is the manuscript's claim. `hg` says the model does not beat the
information-theoretic ceiling, which is not an assumption about *this* model —
it holds of every order-1 model by the definition of the ceiling.

A note on the shape, since `classify`'s first argument is named for the trivial
baseline everywhere else in this file: here it is universally quantified, so it
ranges over *every* candidate gain at this order — the trivial baseline (which
gains 0 nats over uniform, and 0 is bounded by the ceiling), the shipped model,
and every better-implemented model anyone might build. That is precisely the
scope the manuscript's sentence claims, and it is why the statement is a `∀`
rather than a fact about the one model that was measured.

FALSIFIER: a gain `g` bounded by the ceiling that nonetheless classifies
BRACKETED or VACUOUS against the 2-nat bar. Since the ceiling itself misses the
bar (DG6), no such `g` exists; a re-measurement raising the ceiling above 2
nats would break DG6 first.

NON-VACUITY: `hg` is inhabited by the shipped model's own recorded gain — see
`stage1_shipped_model_is_bounded_by_the_ceiling` below. Without that witness
this would be a conditional theorem about a possibly-empty class, which is the
defect this project's Lean doctrine names first.
-/
theorem stage1_declared_gate_unreachable_for_every_bounded_model (g : Rat)
    (hg : ¬ passes Direction.higherIsBetter g stage1MaxBigramGainNats) :
    classify Direction.higherIsBetter g
      (some (stage1MaxBigramGainNats, ComparatorKind.modelClassBound))
      stage1DeclaredGateNats = Verdict.unreachable := by
  refine unreachable_of_bound_fails rfl ?_ stage1_ceiling_misses_declared_gate
  simp only [passes] at hg ⊢
  have hceiling : ¬ (stage1DeclaredGateNats < stage1MaxBigramGainNats) := by
    have := stage1_ceiling_misses_declared_gate
    simpa only [passes] using this
  grind

/--
**DG8.** NON-VACUITY WITNESS for DG7: the shipped letter bigram's recorded gain
is bounded by the recorded ceiling.

FALSIFIER: a recorded `stage1_gain_over_uniform_nats` above
`stage1_max_bigram_gain_nats` — a model beating its own information-theoretic
bound, which would mean one of the two numbers is measured wrong. The recorder
computes `stage1_bigram_signal_captured_pct` as the ratio of exactly these two
values, so a violation would also make that percentage exceed 100.
-/
theorem stage1_shipped_model_is_bounded_by_the_ceiling :
    ¬ passes Direction.higherIsBetter stage1GainOverUniformNats stage1MaxBigramGainNats := by
  decide

/-- DG7 therefore fires on the shipped model, non-vacuously. -/
theorem stage1_shipped_model_was_unreachable :
    classify Direction.higherIsBetter stage1GainOverUniformNats
      (some (stage1MaxBigramGainNats, ComparatorKind.modelClassBound))
      stage1DeclaredGateNats = Verdict.unreachable :=
  stage1_declared_gate_unreachable_for_every_bounded_model _
    stage1_shipped_model_is_bounded_by_the_ceiling

/-! ## Negative controls

Everything above holds. A block of theorems that all come out the way the
manuscript wants is exactly the shape of a check that cannot fail, so this
section exhibits the failures.

The controls are built by *mutating recorded data* rather than by hand-typing
plausible-looking numbers: a hand-typed control is only as adversarial as
whoever typed it, and hand-typed numbers are what this whole layer exists to
eliminate. Where a genuinely synthetic constant is needed it is an obvious one
(`lit 99 1` as an impossible accuracy, `lit 100 1` as an arbitrary bar), never
something that could be mistaken for a measurement. -/

/-- A manufactured sweep: the recorded lexical tail with an impossible 99%
anchor bolted on. Not a measurement, and not shaped like one. -/
def manufacturedAnchorWins : Scores := lit 99 1 :: lexicalSweep.drop 1

/--
**NC1.** The manufactured sweep has **no** interior optimum.

POSITIVE CONTROL for LX1. `InteriorOptimum` is a decidable predicate; if the
decision procedure were broken in the "always true" direction, LX1 would still
compile and would mean nothing. Here the same procedure returns false.

FALSIFIER: this theorem failing would mean `InteriorOptimum` accepts a sweep
whose argmax is at the anchor — i.e. the band assignment in
`fig:channel_comparison` cannot distinguish the two channels at all.
-/
theorem manufacturedAnchorWins_not_interior : ¬ InteriorOptimum manufacturedAnchorWins := by
  decide

/-- **NC2.** ... and it lands in the other band instead: no weight beats its
anchor. So the two bands are exhibited firing on two inputs that differ in one
entry. -/
theorem manufacturedAnchorWins_noInteriorGain : NoInteriorGain manufacturedAnchorWins := by
  decide

/-- The recorded sweep windowed to weights at and above the peak. Every value is
real; the window is the mutation. -/
def lexicalSweepFromPeak : Scores := lexicalSweep.drop lexicalPeakIndex

/--
**NC3.** Windowing the recorded sweep to weights >= the peak destroys the
interior optimum.

The diagnosis is a property of the whole sweep, not of any suffix of it: on the
window the argmax has become the anchor, and the channel would be classified
redundant. This is a real methodological hazard — a sweep truncated at the
weight that looked best is a sweep whose shape verdict is an artifact of the
truncation — and it is exhibited here on the actual recorded numbers.

FALSIFIER: `InteriorOptimum` holding on the window too, which would mean the
predicate is insensitive to where the grid starts.
-/
theorem lexicalSweepFromPeak_not_interior : ¬ InteriorOptimum lexicalSweepFromPeak := by
  decide

/-- **NC4.** The windowed sweep classifies as the *redundant* shape — the
opposite band from the full recorded sweep (LX1/LX5). -/
theorem lexicalSweepFromPeak_noInteriorGain : NoInteriorGain lexicalSweepFromPeak := by
  decide

/-! ### The classifier is not constant on recorded data

GT1-GT5 return bracketed verdicts while GT6 returns ABOVE_REFERENCE. Below, the
*same* recorded Stage 1 baseline and ceiling produce five of the six verdicts
as the threshold, the ceiling-availability, and the declared comparator kind
vary — so the theorems above distinguish possible worlds. -/

/-- **NC5.** Real Stage 1 triple, bar moved to the ceiling itself: UNREACHABLE.
Equality fails in both directions (G1), so a bar sitting exactly on the ceiling
is not clearable. -/
theorem stage1_bar_at_ceiling_is_unreachable :
    classify Direction.lowerIsBetter stage1Baseline
      (some (stage1Ceiling, ComparatorKind.modelClassBound)) stage1Ceiling
      = Verdict.unreachable := by decide

/-- **NC6.** Real Stage 1 triple, bar moved to an arbitrary loose perplexity of
100: VACUOUS, because the trivial baseline already clears it. This is the Stage
3 defect reproduced deliberately on Stage 1's numbers. -/
theorem stage1_loose_bar_is_vacuous :
    classify Direction.lowerIsBetter stage1Baseline
      (some (stage1Ceiling, ComparatorKind.modelClassBound)) (lit 100 1)
      = Verdict.vacuous := by decide

/-- **NC7.** Real Stage 1 triple with the ceiling withheld: UNAUDITED, not a
guess. This is the branch that keeps an unmeasured ceiling from being silently
assumed reachable. -/
theorem stage1_without_ceiling_is_unaudited :
    classify Direction.lowerIsBetter stage1Baseline none stage1Threshold
      = Verdict.unaudited := by decide

/-- **NC8.** Five of the six verdicts, from one recorded population. Stated as
a conjunction so the non-constancy of `classify` on real data is a single
citable fact. (The sixth, ABOVE_REFERENCE, fires on the same numbers in NC10.) -/
theorem stage1_population_reaches_every_verdict :
    classify Direction.lowerIsBetter stage1Baseline
        (some (stage1Ceiling, ComparatorKind.modelClassBound)) stage1Threshold
        = Verdict.bracketedBound
      ∧ classify Direction.lowerIsBetter stage1Baseline
        (some (stage1Ceiling, ComparatorKind.referenceModel)) stage1Threshold
        = Verdict.bracketedReference
      ∧ classify Direction.lowerIsBetter stage1Baseline
        (some (stage1Ceiling, ComparatorKind.modelClassBound)) stage1Ceiling
        = Verdict.unreachable
      ∧ classify Direction.lowerIsBetter stage1Baseline
        (some (stage1Ceiling, ComparatorKind.modelClassBound)) (lit 100 1)
        = Verdict.vacuous
      ∧ classify Direction.lowerIsBetter stage1Baseline none stage1Threshold
        = Verdict.unaudited := by decide

/--
**NC9.** Direction is load-bearing on recorded data.

Read with the wrong direction, the Stage 1 row inverts: the model that beats
its bar (GT7) fails it, and the ceiling that clears the bar does not.

FALSIFIER: this failing would mean the two directions agree on the recorded
Stage 1 numbers, and the hand-written `Direction` annotations above would be
unfalsifiable decoration. They are not: getting one wrong flips the verdict.
-/
theorem stage1_direction_is_load_bearing :
    ¬ passes Direction.higherIsBetter stage1Model stage1Threshold
      ∧ classify Direction.higherIsBetter stage1Baseline
        (some (stage1Ceiling, ComparatorKind.modelClassBound)) stage1Threshold
        = Verdict.unreachable := by decide

/--
**NC10.** The declared comparator kind is load-bearing on recorded data.

The same recorded Stage 1 triple, with the bar at the ceiling: declared a
model-class bound it classifies UNREACHABLE (NC5); declared a reference it
classifies ABOVE_REFERENCE. Nothing numeric changes between the two — only the
epistemic status of the comparator, which is exactly the distinction the
external audit found the untyped taxonomy erasing.

FALSIFIER: the two classifications agreeing. That would mean the kind
annotation is unfalsifiable decoration, and `unreachable_requires_bound` (G10)
would be a theorem about a distinction the classifier does not draw.
-/
theorem stage1_comparator_kind_is_load_bearing :
    classify Direction.lowerIsBetter stage1Baseline
        (some (stage1Ceiling, ComparatorKind.referenceModel)) stage1Ceiling
        = Verdict.aboveReference
      ∧ classify Direction.lowerIsBetter stage1Baseline
        (some (stage1Ceiling, ComparatorKind.modelClassBound)) stage1Ceiling
        = Verdict.unreachable := by decide

end AILM
