/-
  AILM.LedgerData — GENERATED FILE. DO NOT EDIT BY HAND.

  Written by `scripts/generate_lean_ledger.py` from `data/claims.json`.
  Regenerate with:

      .venv/bin/python scripts/generate_lean_ledger.py

  `tests/test_formal_isomorphism.py` requires a fresh run to reproduce this
  file byte for byte, so a hand edit here cannot survive the suite, and neither
  can a ledger change that was never regenerated.

  ## Every number below is a recorded measurement, with one marked exception

  Each `def` names the ledger label it came from. Nothing here is typed by
  hand, rounded, or adjusted.

  The exception is `lexicalGrid`: the mixing weights the sweep was *run at*,
  fixed in advance rather than measured. It is structure, in the same sense the
  `Direction` annotations in `Instances.lean` are structure, and every one of
  its entries says `[structural]` in place. It is emitted so the kernel can
  check `lexicalPeakIndex` against the recorded `lexical_best_alpha` instead of
  trusting the Python that derived it.

  The synthetic shape witnesses in `GateVerdict.lean`, `SweepShape.lean`,
  `Filtering.lean` and `Bonferroni.lean` are a third thing again: invented
  numbers chosen to exhibit a shape. Do not confuse the three — this is the only
  module in the tree whose literals describe the world.

  No prose in this file may carry a measurement. A number written into a comment
  here resolves to no ledger label, so byte-identity regeneration preserves it
  forever: it would go stale on the next re-measurement while every proof below
  it stayed green. Scope notes name populations, never values.

  ## Exact conversion policy

  `data/claims.json` is parsed with `parse_float=str`, so the source is the
  literal decimal text in the file rather than a re-rounded binary float. That
  text becomes an exact `Fraction` (an invented 68.34 -> 6834/100), reduced to
  lowest terms (3417/50) and emitted as `lit n d`. `lit` is built on `Rat.mk'`,
  whose `Nat.Coprime` field is discharged by `decide` at elaboration time — so
  if this script ever emitted an unreduced fraction the file would fail to
  compile rather than quietly hold a wrong number. No float arithmetic, no
  rounding, and no tolerance anywhere in the conversion.

  ## What is deliberately NOT here

  * Metric direction (higher/lower is better) is structural, not measured, so
    it is hand-written in `Instances.lean` where it mirrors
    `scripts/audit_gates.py`.
  * The Stage 6 and generation gate rows. `audit_gates.py` audits both, but
    neither typed comparator has a ledger label (Stage 6's is fitted on test,
    generation's is the analytic 0.0). A triple is emitted only when all three
    components exist; inventing the third would be the fabrication this layer
    exists to make impossible.
  * The unrecorded interior of the coupling grid — see the scope note on
    `couplingRecorded`.
-/
import AILM.Core

namespace AILM

/-! ## Lexical channel — `sec:hierarchy`, the novel-message sweep -/

/-- the uncoupled alpha=0 anchor (ledger: `lexical_anchor_accuracy`). -/
def lexicalAnchorAccuracy : Rat := lit 70801 1000

/-- best accuracy over the grid (ledger: `lexical_best_accuracy`). -/
def lexicalBestAccuracy : Rat := lit 37067 500

/-- the weight that achieved it (ledger: `lexical_best_alpha`). -/
def lexicalBestAlpha : Rat := lit 7 10

/-- The recorder's own interior-optimum flag, `1` for true. `Instances.lean`
re-decides the shape from `lexicalSweep` and requires the two to agree, so
this is a cross-check on the recorder rather than an input to any proof
(ledger: `lexical_optimum_is_interior`). -/
def lexicalOptimumIsInteriorFlag : Nat := 1

/-- The prospectively specified lexical mixing-weight grid.

STRUCTURAL, NOT MEASURED — the only such list in this file. These are the
weights the sweep was run at, fixed in advance by
`scripts/record_lexical_channel_claims.py`; no recorder measured them, so no
entry names a ledger label and every entry is marked `[structural]`.

It is here so `lexicalPeakIndex` stops being a number Lean takes on trust:
`Instances.lean` decides that the index really selects `lexical_best_alpha`. -/
def lexicalGrid : List Rat :=
  [ lit 0 1   -- alpha 0.0  [structural]
  , lit 1 10  -- alpha 0.1  [structural]
  , lit 1 5   -- alpha 0.2  [structural]
  , lit 3 10  -- alpha 0.3  [structural]
  , lit 1 2   -- alpha 0.5  [structural]
  , lit 7 10  -- alpha 0.7  [structural]
  , lit 9 10  -- alpha 0.9  [structural]
  ]

/-- Index of `lexical_best_alpha` within `lexicalGrid`. Derived here from the
ledger and the grid order, then re-checked in the kernel: `Instances.lean`
decides both that the sweep peaks at this index and that the grid holds the
recorded best alpha there. -/
def lexicalPeakIndex : Nat := 5

/-- The lexical mixing-weight sweep, index 0 = the uncoupled anchor. Every
entry is a recorded accuracy; the grid order is the one
`scripts/record_lexical_channel_claims.py` fixed in advance. -/
def lexicalSweep : List Rat :=
  [ lit 70801 1000  -- alpha 0.0  (lexical_sweep_alpha_0_0_accuracy)
  , lit 9088 125    -- alpha 0.1  (lexical_sweep_alpha_0_1_accuracy)
  , lit 73319 1000  -- alpha 0.2  (lexical_sweep_alpha_0_2_accuracy)
  , lit 9206 125    -- alpha 0.3  (lexical_sweep_alpha_0_3_accuracy)
  , lit 73951 1000  -- alpha 0.5  (lexical_sweep_alpha_0_5_accuracy)
  , lit 37067 500   -- alpha 0.7  (lexical_sweep_alpha_0_7_accuracy)
  , lit 74059 1000  -- alpha 0.9  (lexical_sweep_alpha_0_9_accuracy)
  ]

/-! ## Coupling channel — `sec:hierarchy`, the redundant-message sweep -/

/-- the uncoupled alpha=0 anchor (ledger: `coupling_alpha0_accuracy`). -/
def couplingAlpha0Accuracy : Rat := lit 70801 1000

/-- the fully-coupled endpoint (ledger: `coupling_alpha1_accuracy`). -/
def couplingAlpha1Accuracy : Rat := lit 8461 500

/-- MAXIMUM over every alpha > 0 in the grid (ledger: `coupling_best_accuracy`). -/
def couplingBestAccuracy : Rat := lit 705371 10000

/-- the weight that achieved it (ledger: `coupling_best_alpha`). -/
def couplingBestAlpha : Rat := lit 1 10

/-- Size of the coupling alpha grid (ledger: `coupling_grid_size`). Only three
of these points have recorded accuracies. -/
def couplingGridSize : Nat := 6

/-- The recorded coupling points, ordered by mixing weight.

SCOPE — read this before citing the theorems about it:

Only three points of the six-point coupling grid are recorded: the alpha=0
anchor, the BEST positive alpha, and alpha=1. The middle entry is the grid
MAXIMUM over alpha > 0 (record_coupling_claims.py: best_alpha =
max(positives, key=...)), so 'no entry beats the anchor' on this
three-point list is exactly 'no positive weight in the full six-point grid
beats the anchor' -- a stronger statement than its length suggests. What
it is NOT is a claim about the SHAPE between the points: alphas 0.25, 0.5
and 0.75 have no recorded values, so no monotonicity claim about the full
sweep can be read off this list. -/
def couplingRecorded : List Rat :=
  [ lit 70801 1000    -- alpha 0 (anchor)  (coupling_alpha0_accuracy)
  , lit 705371 10000  -- alpha 1/10 (best over alpha > 0)  (coupling_best_accuracy)
  , lit 8461 500      -- alpha 1 (fully coupled)  (coupling_alpha1_accuracy)
  ]

/-! ## Stage gate triples — the rows `scripts/audit_gates.py` classifies

Each row is (baseline, typed comparator, model). The audit sets the THRESHOLD to the
baseline for every row, so the threshold is emitted as an alias rather than
as a second literal pretending to be its own measurement. The retained
`*Ceiling` Lean identifiers are historical API names: their values are typed
comparators and are not necessarily bounds. -/

/-! ### `stage1_letter`

held-out Brown text; same population as audit_gates.py -/

/-- `stage1_letter` trivial baseline (ledger: `stage1_unigram_baseline`). -/
def stage1Baseline : Rat := lit 99779 5000

/-- `stage1_letter` typed comparator (legacy `Ceiling` identifier) (ledger: `stage1_bigram_floor_perplexity`). -/
def stage1Ceiling : Rat := lit 53753 5000

/-- `stage1_letter` shipped model (ledger: `stage1_letter_perplexity`). -/
def stage1Model : Rat := lit 23843 2000

/-- `stage1_letter` threshold. `audit_gates.py` sets this row's
threshold to its baseline, so this is an alias, not a new number. -/
def stage1Threshold : Rat := stage1Baseline

/-! ### `stage2_phoneme`

FULL CMUdict, closed form; same population as audit_gates.py -/

/-- `stage2_phoneme` trivial baseline (ledger: `stage2_phoneme_unigram_baseline`). -/
def stage2PhonemeBaseline : Rat := lit 34493 1250

/-- `stage2_phoneme` typed comparator (legacy `Ceiling` identifier) (ledger: `stage2_phoneme_all_position_floor_perplexity`). -/
def stage2PhonemeCeiling : Rat := lit 158313 10000

/-- `stage2_phoneme` shipped model (ledger: `stage2_phoneme_perplexity`). -/
def stage2PhonemeModel : Rat := lit 33489 2000

/-- `stage2_phoneme` threshold. `audit_gates.py` sets this row's
threshold to its baseline, so this is an alias, not a new number. -/
def stage2PhonemeThreshold : Rat := stage2PhonemeBaseline

/-! ### `stage2_g2p`

FULL held-out G2P test set. audit_gates.py scores this row on an
evenly-spaced strided subsample instead; same verdict, different position
set, and therefore different numbers than the ones below -/

/-- `stage2_g2p` trivial baseline (ledger: `stage2_g2p_baseline_accuracy`). -/
def stage2G2pBaseline : Rat := lit 14051 200

/-- `stage2_g2p` typed comparator (legacy `Ceiling` identifier) (ledger: `stage2_g2p_smoothing_ceiling`). -/
def stage2G2pCeiling : Rat := lit 37821 500

/-- `stage2_g2p` shipped model (ledger: `stage2_g2p_accuracy`). -/
def stage2G2pModel : Rat := lit 70801 1000

/-- `stage2_g2p` threshold. `audit_gates.py` sets this row's
threshold to its baseline, so this is an alias, not a new number. -/
def stage2G2pThreshold : Rat := stage2G2pBaseline

/-! ### `stage3_word`

in-vocabulary held-out Brown; same population as audit_gates.py -/

/-- `stage3_word` trivial baseline (ledger: `stage3_bigram_baseline`). -/
def stage3WordBaseline : Rat := lit 90547 10000

/-- `stage3_word` typed comparator (legacy `Ceiling` identifier) (ledger: `stage3_ceiling_perplexity`). -/
def stage3WordCeiling : Rat := lit 2028 625

/-- `stage3_word` shipped model (ledger: `stage3_word_perplexity`). -/
def stage3WordModel : Rat := lit 32713 10000

/-- `stage3_word` threshold. `audit_gates.py` sets this row's
threshold to its baseline, so this is an alias, not a new number. -/
def stage3WordThreshold : Rat := stage3WordBaseline

/-! ### `stage4_case`

derived case oracle, held-out UD EWT; same population as audit_gates.py,
which prints these same three measurements at full precision where the
ledger stores them rounded. READ THIS BEFORE CITING THE COMPARATOR: on
this row the shipped model scores ABOVE the recorded forward-backward
SMOOTHING reference. It is a reference comparator, not a bound over every
model in the class. Nothing in this tree assumes otherwise -- G6's
dominance hypothesis is about the BASELINE, which does sit below it -- but
the manuscript sentence that once justified that hypothesis 'by
construction' was wrong on exactly this row -/

/-- `stage4_case` trivial baseline (ledger: `stage4_derived_baseline`). -/
def stage4DerivedBaseline : Rat := lit 4291 100

/-- `stage4_case` typed comparator (legacy `Ceiling` identifier) (ledger: `stage4_derived_smoothing`). -/
def stage4DerivedCeiling : Rat := lit 2479 50

/-- `stage4_case` shipped model (ledger: `stage4_derived_agreement`). -/
def stage4DerivedModel : Rat := lit 1268 25

/-- `stage4_case` threshold. `audit_gates.py` sets this row's
threshold to its baseline, so this is an alias, not a new number. -/
def stage4DerivedThreshold : Rat := stage4DerivedBaseline

/-! ### `stage5_clause`

held-out clause corpus; same population as audit_gates.py -/

/-- `stage5_clause` trivial baseline (ledger: `stage5_typefree_perplexity`). -/
def stage5ClauseBaseline : Rat := lit 43741 10000

/-- `stage5_clause` typed comparator (legacy `Ceiling` identifier) (ledger: `stage5_type_oracle_perplexity`). -/
def stage5ClauseCeiling : Rat := lit 43819 10000

/-- `stage5_clause` shipped model (ledger: `stage5_mixture_perplexity`). -/
def stage5ClauseModel : Rat := lit 21929 5000

/-- `stage5_clause` threshold. `audit_gates.py` sets this row's
threshold to its baseline, so this is an alias, not a new number. -/
def stage5ClauseThreshold : Rat := stage5ClauseBaseline

/-! ## Declared gates and information-theoretic bounds

The historical thresholds Stages 1, 3 and 4 originally declared, plus the
bounds they are judged against. `stage4DeclaredGate` is the interesting one:
the same bar classifies differently on the two Stage 4 populations, and both
populations are recorded. -/

/-- the >=2 nats Stage 1 originally declared (ledger: `stage1_declared_gate_nats`). -/
def stage1DeclaredGateNats : Rat := lit 2 1

/-- what the shipped letter bigram achieved (ledger: `stage1_gain_over_uniform_nats`). -/
def stage1GainOverUniformNats : Rat := lit 9229 10000

/-- the order-1 ceiling: no bigram model beats this (ledger: `stage1_max_bigram_gain_nats`). -/
def stage1MaxBigramGainNats : Rat := lit 5131 5000

/-- the perplexity < 500 Stage 3 originally declared (ledger: `stage3_declared_gate`). -/
def stage3DeclaredGate : Rat := lit 500 1

/-- the >=60% Stage 4 originally declared (ledger: `stage4_declared_gate`). -/
def stage4DeclaredGate : Rat := lit 60 1

/-- gold-morphology population: context-free argmax (ledger: `stage4_gold_baseline`). -/
def stage4GoldBaseline : Rat := lit 319 5

/-- gold-morphology population: forward-backward smoothing (ledger: `stage4_gold_smoothing`). -/
def stage4GoldCeiling : Rat := lit 319 5

/-- gold-morphology population: the shipped agent (ledger: `stage4_gold_agreement`). -/
def stage4GoldModel : Rat := lit 319 5

end AILM
