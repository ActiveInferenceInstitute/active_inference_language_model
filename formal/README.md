# `formal/` — the AILM Lean 4 layer

Machine-checked statements of the parts of AILM's argument that are actually
mathematical: the gate taxonomy, the sweep-shape diagnostic, the filtering
recursion, the Bonferroni arithmetic, and the Holm step-down comparison that
justifies keeping it.

This layer is **small on purpose**. Its value is not breadth; it is that a
handful of load-bearing claims stop being prose that a careful reader has to
re-derive, and become statements a kernel rejects if they are wrong. The list of
things deliberately *not* proved here is as much a part of the deliverable as
the list of things proved, and it is below.

## Build

The toolchain is `elan`-managed. `elan` reads `lean-toolchain` in this directory
and fetches the pinned release the first time; after that every command below is
offline.

```bash
# Run from the repository root.
(cd formal && lake build && ./Scripts/axiom_report.sh)
.venv/bin/python -m pytest tests/test_formal_isomorphism.py::test_the_negative_control_does_not_compile -v --no-cov
```

The ledger bridge is driven from the repository root:

```bash
.venv/bin/python scripts/generate_lean_ledger.py          # writes AILM/LedgerData.lean
.venv/bin/python scripts/generate_lean_ledger.py --check  # fails if it is stale
.venv/bin/python scripts/record_formal_claims.py          # records formal_* claims
.venv/bin/python -m pytest tests/test_formal_isomorphism.py
```

`tests/test_formal_isomorphism.py` skips its Lean-dependent checks cleanly when
`lake` is not on `PATH`, naming the pinned toolchain in the skip reason. Its
ledger-binding checks need no toolchain and always run, so a machine without
`elan` still cannot desynchronise the Lean from the measurements.

The toolchain is pinned in `lean-toolchain` to `leanprover/lean4:v4.35.0`, and
`lakefile.toml` has **no `[[require]]` stanza**: there are no dependencies, not
even Mathlib. `lake build` is therefore offline by construction — no network, no
cache download, no version resolution. Everything is proved against plain Lean 4
core (`List`, `Rat`, `Decidable`).

## Why every computed fact is an order comparison

The no-Mathlib choice has one consequence that shapes the whole design, and it
was measured rather than assumed:

**`Rat` arithmetic does not kernel-reduce.** `decide` fails on `(1:Rat)+1 = 2`
and on `[1/2, 1/2].sum = 1`; reduction gets stuck at `Rat.add`, because
`Rat.normalize` is opaque to the kernel. **`Rat` order comparisons do reduce**
on `Rat.mk'` literals — `a < b`, `a ≤ b`, `a ≠ b` all close by `decide`.

So:

* every fact closed by `decide` in this tree is a **pure order comparison**;
* arithmetic facts are closed by `grind`, which reasons about `Rat` numerals
  symbolically without asking the kernel to evaluate them;
* `AILM.lit` (in `AILM/Core.lean`) is the only literal constructor, and it emits
  `Rat.mk'` directly so its literals *are* `decide`-reducible. `Rat.mk'` carries
  a `Nat.Coprime` field, so a literal must be supplied already reduced —
  `lit 2 8` is a compile error.

`native_decide` is used **nowhere**. It would add the `Lean.ofReduceBool`
trusted-code axiom, trading kernel checking for compiler trust; nothing here
needs it, because no `decide` in the tree is expensive.

## Axiom policy

`axiom_report.txt` is machine output — `#print axioms` for all
claim:formal_theorem_count named theorems plus the six `Decidable` instances
that every `decide` in the tree routes through, claim:formal_axiom_report_verdicts
verdicts in all. Those are ledger tags, not digits: they were hand-typed here
until 2026-08-05, correct on the day they were written and with no way to stay
correct. Regenerate the report with `./Scripts/axiom_report.sh`, which refuses to
write one containing zero verdicts (a report that checks nothing reports clean).

Coverage is enforced rather than trusted: `scripts/record_formal_claims.py`
refuses to record anything unless the report's entries and the declarations in
`AILM/*.lean` are the **same set**, checked in both directions. A report
covering part of the tree reports clean for the rest, and an enumeration only
constrains what someone remembered to enumerate.

Allowed axioms: **`propext`, `Classical.choice`, `Quot.sound`, `funext`.**

`Classical.choice` is on that list for a measured reason. Lean **core's own**
`Rat.add_comm` reports `[propext, Classical.choice, Quot.sound]`. A policy
demanding only `propext / Quot.sound / funext` would be unsatisfiable against
core `Rat` and would fail for reasons unrelated to anything proved here. Do not
"tighten" it without checking that first.

Observed in the current report: only `propext`, `Classical.choice` and
`Quot.sound` appear; `funext` is not needed. Notably **absent**:

* `sorryAx` — an incomplete proof. Lean emits a *warning*, not an error, for an
  unfinished proof, so a zero exit code alone would not catch one; the axiom
  report is the check that does.
* `Lean.ofReduceBool` — the `native_decide` trusted-code axiom.

## The negative control

`Controls/MustFail.lean` **must not compile**. It sits outside the `AILM/`
library root, so `lake build` never sees it. The pytest control invokes it
directly, requires a nonzero exit status, and checks the rejection text for every
expected refutation plus the ledger-backed diagnostic. A raw shell negation is
not the control: it would also accept a typo, missing import, wrong path, or
broken toolchain. The file asserts four false statements, each false for a
different reason:

1. equality passing a strict gate;
2. a degenerate population classifying as either bracketed verdict;
3. a sweep in both shape classes at once;
4. **`InteriorOptimum couplingRecorded`** — false on the *recorded* coupling
   accuracies.

The fourth is what covers the ledger bridge. The first three are refuted on
synthetic witnesses, so on their own they show only that `decide` can reject
invented numbers. Without a control refuted on real ledger literals, every
`by decide` in `AILM/Instances.lean` would be an assertion that the kernel
evaluated something.

Without this file, "`lake build` succeeded" is an assertion nobody has made
fail. It is the Lean-side analogue of `verify_negative_control_fails` in
`src/ailm/control.py`.

## What is proved

| Module | Manuscript object | Theorems |
|---|---|---|
| `AILM/GateVerdict.lean` | `eq:gate_satisfies`, comparator typing, `eq:gate_verdict` | claim:formal_module_gate_verdict_theorem_count |
| `AILM/SweepShape.lean` | `sec:hierarchy` shape classes | claim:formal_module_sweep_shape_theorem_count |
| `AILM/Filtering.lean` | `eq:filtering` (both halves) | claim:formal_module_filtering_theorem_count |
| `AILM/Bonferroni.lean` | `sec:multiplicity` union bound | claim:formal_module_bonferroni_theorem_count |
| `AILM/Holm.lean` | the Holm paragraph of `98_supplement_statistics.md` | claim:formal_module_holm_theorem_count |
| `AILM/PairedDifference.lean` | `eq:paired_difference` of `98_supplement_statistics.md` | claim:formal_module_paired_difference_theorem_count |
| `AILM/LedgerData.lean` | GENERATED — recorded values, no theorems | claim:formal_module_ledger_data_theorem_count |
| `AILM/Instances.lean` | the recorded values, run through the theory | claim:formal_instance_theorem_count |

`AILM/Holm.lean` proves that Holm's per-rank bar `α/(m-i+1)` is never stricter
than Bonferroni's `α/m` at a valid rank, that the bar is nondecreasing in the
rank, that the two coincide at rank 1, and that on a *sorted* family every
member the fixed bar rejects the step-down walk rejects too, with a synthetic
family on which the walk rejects strictly more. That witness is the "and the
converse can fail" half of the manuscript sentence. What it does **not** prove is
familywise error control, exactly as `AILM/Bonferroni.lean` does not: there is
no probability measure in this tree, so "Holm controls the FWER under arbitrary
dependence" remains assumed. The scope list below carries the rest.

`AILM/PairedDifference.lean` proves that the manuscript's two forms of
`eq:paired_difference` agree — the share-weighted `(2s − 1)(b + c)/n`, with
`s = c/(b + c)`, equals the difference form `(c − b)/n` — and pins the sign and
the zero behavior of `δ`. What it does **not** prove is statistics: there is no
probability space in this tree, so it says nothing about the sampling
distribution of `δ`, the coverage of the bootstrap interval
`paired_difference_interval` builds, or the validity of the McNemar test the
discordant counts feed. The scope list below carries the rest.

Every per-module total includes its named witness theorems. Anonymous `example`
witnesses and named `instance` declarations are excluded from these theorem
counts; anonymous examples appear only in claim:formal_example_count. The table
and whole-tree totals are ledger tags written by `scripts/record_formal_claims.py`
from the shared declaration scan after a real build and a fresh, complete axiom
report. The report driver is generated from that scan with
`--generate-axiom-script`, and `--check-axiom-script` checks freshness without
writing or recording. Generation alone certifies no proof:
claim:formal_theorem_count named theorems in all, of which
claim:formal_theory_theorem_count are theory and
claim:formal_instance_theorem_count are proved about recorded values, plus
claim:formal_example_count witness examples.

Every theorem carries a doc comment naming its manuscript equation and stating
**what would falsify it** — which possible world it excludes. A theorem that
merely restates a definition is decoration; the falsifier line is where that
gets checked by a reader.

Every conditional theorem carries a **satisfiability witness**: an instance
proving its hypothesis is inhabited, *and* an instance firing the theorem on it.
A conditional theorem with an unsatisfiable hypothesis compiles clean and proves
nothing, and that is a defect this project has shipped before. Where a
hypothesis is genuinely load-bearing there is also a **negative control** showing
the conclusion fails without it — `Filtering.zeroMassCounterexample`,
`Filtering.raggedColumnLosesMass`, `Filtering.leakyColumnLosesMass`,
`Filtering.missingColumnLosesMass`, `Bonferroni.scaleHypothesisIsLoadBearing`,
`Bonferroni.zeroFamilyThresholdCollapses`, `Holm.sortednessIsLoadBearing`,
`Holm.holm_rank_one_hypothesis_is_load_bearing`,
`Holm.holm_rank_positivity_is_load_bearing`,
`Holm.inversionPositivityIsLoadBearing`,
`PairedDifference.pairedDelta_pos_drops_imbalance`,
`PairedDifference.pairedDelta_pos_drops_nonempty`,
`PairedDifference.pairedDelta_neg_drops_imbalance`,
`PairedDifference.pairedDelta_neg_drops_nonempty`.

The three prediction-step controls are stated as one-fault inputs rather than as
bare counterexamples: each drops exactly one hypothesis of `predict_isDistribution`
and a companion theorem — `raggedTransition_meets_the_other_hypotheses` and its
two siblings — proves the remaining hypotheses hold on that same input. Without
the companion, a counterexample that violated three hypotheses at once would
read as evidence for whichever one it was filed under.

That sentence was false when it was first written, which is worth recording
because it is the exact failure mode the rule names. `bonferroni_union_bound_div`
(B3) — the *divided* form the manuscript actually writes — had no witness and no
instantiation anywhere in the tree: its hypothesis is stated against
`bonferroniThreshold α n`, while `witnessFamily_bounded` bounded the family
against the hand-supplied literal `1/1100`, so the two never met.
`witnessFamily_bounded_by_corrected_threshold` and `witnessFamily_bound_div` are
the missing pair. A README asserting a universal is not a check that the
universal holds; the per-theorem witness blocks are.

Every decidable predicate is exhibited **both satisfiable and refutable**, so no
`Decidable` instance in the tree is deciding a constant. Same caveat, same
history: `satisfies` had no refutation witness and `PeakAt` had no failing
instance until 2026-08-05, so two of the six `Decidable` instances were being
exercised in one direction only.

## The isomorphism

Each row is one claim held in four places at once: a Lean declaration, the
manuscript object it transcribes, the Python that computes the same thing on real
data, and the test that exercises the Python. The point of the table is that all
four are named — a Lean theorem whose Python counterpart cannot be pointed at is
proving something about a definition this project does not run, and that is the
failure mode the table makes visible rather than the one it hides.

| Lean declaration | Manuscript object | Python source of truth | Exercising test |
|---|---|---|---|
| `GateVerdict.passes`, `GateVerdict.satisfies`, `passes_iff_satisfies_and_ne`, `passes_irrefl`, `satisfies_of_passes` | `eq:gate_satisfies` — strict audit `≻` vs inclusive declared `⪰` | `gate_passes` in `src/ailm/gates.py` | `tests/test_gates.py` |
| `GateVerdict.classify`, `bracketed_iff`, `vacuous_iff`, `unreachable_of_bound_fails`, `above_reference_of_reference_fails`, `unreachable_requires_bound`, `above_reference_requires_exceedable`, `no_co_fire`, `degenerate_no_gate` | `eq:gate_verdict`, the comparator typing, and the gate-soundness definition in `03a_gate_design.md` | `classify_gate` in `src/ailm/gates.py`, the single classifier imported by both `scripts/audit_gates.py` and `src/ailm/figures/gates.py` | `tests/test_gates.py`; end-to-end by `scripts/audit_gates.py` |
| `SweepShape.InteriorOptimum`, `NoInteriorGain`, `MonotoneDecline`, `interior_excludes_no_gain`, `interior_excludes_monotone`, `monotone_decline_no_interior_gain` | the pre-registered shape classes of `sec:hierarchy` (decisions 18–19) | `sweep_lexical_alpha` in `src/ailm/levels/lexical_channel.py`; `sweep_alpha` in `src/ailm/levels/coupling.py` | `tests/test_lexical_channel.py`, `tests/test_coupling.py` |
| `Filtering.normalize`, `sum_div`, `normalize_sum`, `normalize_nonneg`, `normalize_isDistribution` | the correction half of `eq:filtering`; `eq:forward_recursion` | the `unnormalized / norm` lines of `forward_filter` in `src/ailm/verification/oracle.py` | `tests/test_oracle.py` |
| `Filtering.predict`, `predict_length`, `predict_sum`, `predict_nonneg`, `predict_isDistribution` | the prediction half of `eq:filtering` — `ρ_t = B q_{t-1}` | the `transition @ pred` line of `forward_filter`, whose `transition[i, j] = P(next = i \| cur = j)` is the column-major convention `Transition` transcribes | `tests/test_oracle.py` |
| `Bonferroni.bonferroniThreshold`, `sum_le_length_mul`, `bonferroni_union_bound`, `bonferroni_union_bound_div`, `bonferroni_threshold_scales` | the multiplicity correction behind `tbl:multiplicity_family` | `src/ailm/statistics.py`; the family is assembled and recorded by `scripts/record_statistical_claims.py` | `tests/test_statistics.py` |
| `Holm.holmThreshold`, `bonferroni_le_holm_threshold`, `holm_threshold_monotone_in_rank`, `holm_rank_one_is_bonferroni`, `stepDownCount`, `holm_rejects_every_bonferroni_rejection` | the Holm paragraph of `98_supplement_statistics.md`, "uniformly at least as powerful … and the converse can fail" | `holm_step_down` in `src/ailm/statistics.py`, whose rank-`i` bar is `alpha / (size - index)` at `i = index + 1` | `tests/test_statistics.py`; the witness counts are re-run through the real procedure by `tests/test_formal_isomorphism.py::test_holm_witness_agrees_with_the_executable_step_down` |
| `PairedDifference.pairedDelta`, `discordantShare`, `pairedDelta_discordant_form`, `pairedDelta_pos`, `pairedDelta_neg`, `pairedDelta_zero` | `eq:paired_difference` of `98_supplement_statistics.md`, "discordant win share mapped back to a paired difference" | `paired_difference_interval` in `src/ailm/statistics.py` (re-exported by `src/ailm/benchmarks/scoring.py`); the derived-bound convention is `paired_difference_bound_from_counts` in `src/ailm/manuscript_variables/_stats.py` | `tests/test_statistics.py::TestDerivedPairedBound` |
| `LedgerData.*` (generated; no theorems) | the recorded values themselves | `scripts/generate_lean_ledger.py`, reading `data/claims.json` | `tests/test_formal_isomorphism.py` — byte-identity plus an independent re-derivation of every literal |
| `Instances.*` (the claim:formal_instance_theorem_count ledger-facing theorems) | `tbl:gate_audit`, `tbl:unaudited_gates`, `tbl:lexical_channel` | the stage recorders, re-derived independently by `scripts/audit_gates.py` | `tests/test_formal_isomorphism.py` (build + axioms); `scripts/record_formal_claims.py` |
| `Controls/MustFail.lean` (must **not** compile) | — | `verify_negative_control_fails` in `src/ailm/control.py`, the same idea one layer down | `tests/test_formal_isomorphism.py::test_the_negative_control_does_not_compile` |

Two rows deserve a caveat a reader should carry away with them. The
`SweepShape` row formalises the *logical relationship* between the two
pre-registered shape predicates; the Python computes the sweeps that those
predicates are then evaluated on. Lean says the classes are mutually exclusive,
not that either class was correctly measured. And the `Bonferroni` row is
arithmetic only — see the scope list below before citing it.

## Two kinds of number, and never a third

Exactly one module holds measured values: **`AILM/LedgerData.lean`**, which is
generated from `data/claims.json`. Every literal in the five theory modules is a
**synthetic witness** chosen to have the right *shape* — a finite sweep with an
interior peak, a finite p-value family, an unnormalized evidence vector — and
every witness block says so in place. The current measured family size and
threshold come through generated `LedgerData`; prose must not copy them into
this theory-layer description.

The distinction is not decorative. A synthetic witness proves a theorem is
non-vacuous; a ledger value proves something about the world. Citing one as the
other is the whole failure mode this layer exists to prevent.

There is no third kind. **No ledger value is ever hand-typed into Lean.**

### How the bridge is held together

`scripts/generate_lean_ledger.py` is the only writer of `LedgerData.lean`, and
`tests/test_formal_isomorphism.py` requires a fresh run to reproduce the
committed file byte for byte. That single check closes both directions:

* a **hand edit** cannot survive — the file would no longer match a fresh run,
  and the numbers in it would be unmeasured;
* a **stale file** cannot survive either — if a recorder moves a value and
  nobody regenerates, the committed Lean keeps proving things about the old
  number while the build stays green.

Byte identity alone would only prove the generator is deterministic, so a second
and independent oracle re-derives every emitted literal from the ledger and
compares: a generator that consistently truncated a decimal would pass the
identity check forever and fail that one on the first value.

### The exact conversion policy

`data/claims.json` is parsed with `parse_float=str`, so the source is the
literal decimal **text** in the file rather than a re-rounded binary float. That
text becomes an exact `Fraction` (`68.34` → `6834/100`), is reduced to lowest
terms (`3417/50`), and is emitted as `lit n d`. Because `lit` is built on
`Rat.mk'` with its `Nat.Coprime` field discharged by `decide`, an unreduced
fraction is a **compile error** rather than a silently wrong number — the kernel
checks that the gcd reduction actually happened. No float arithmetic, no
rounding, and no tolerance anywhere in the conversion.

### What the ledger instances claim, and what they do not

They do **not** claim any measurement is correct. Lean cannot check that the
Stage 3 baseline was computed over the right positions; the recorders' own
guards do that. What is checked is that the recorded numbers have the logical
properties the manuscript and `scripts/audit_gates.py` assert of them — the
lexical optimum really is interior, the gold Stage 4 population really supports
no sound threshold at *any* bar, the declared claim:stage4_declared_gate%
threshold really does fail in two
opposite directions on the two recorded populations.

Two scope notes travel with the results and should travel with any citation:

* **The coupling channel has three of six grid points recorded.** The honest
  predicate there is `NoInteriorGain`, which the recorded points support at full
  grid strength because the middle entry is the maximum over every positive
  alpha. `couplingRecorded_declines_across_recorded_points` is about those three
  points only — the unrecorded alphas 0.25, 0.5 and 0.75 rule nothing in or out.
* **One gate row describes a different population than the audit's own run.**
  The Stage 2B triple is the full-population ledger value, while
  `audit_gates.py` scores that row on a strided subsample. Stage 2A now uses the
  exact full-dictionary sufficient statistics in both places, including every
  independently reset word start. The Stage 2B verdict agrees across its two
  populations; the numbers are not the same numbers, and the generated block
  says so.

* **The six audited stage-gate rows are one-sided, and a bracketed verdict means less
  there than it looks.** `scripts/audit_gates.py` sets `threshold := baseline`
  on every audited row, so the generated `…Threshold` is a definitional alias of
  the `…Baseline`. Under that convention the vacuity half of a bracketed verdict — "the
  baseline fails the bar" — is discharged by `passes_irrefl` unconditionally,
  and `bracketed_at_baseline_bar_iff` (G9) proves the reduction: at `t = b`,
  a bracketed verdict is *equivalent* to "the ceiling strictly beats the baseline". So
  `stage1_bracketed` and its five siblings each assert exactly one order
  comparison. That comparison is a real measured fact; two independent hurdles
  cleared is not what is being claimed.

* **"Ceiling" names a measured comparator, not a bound over the model class.**
  The Stage 4 derived row's shipped agent scores *above* its recorded
  `stage4_derived_smoothing` ceiling. Nothing in this tree assumes otherwise —
  `no_co_fire`'s dominance hypothesis is about the baseline, which does sit
  below its ceiling on every recorded row — but prose that justified that
  hypothesis "by construction, since the ceiling bounds every model in the
  class" was wrong on exactly this row, and has been corrected in
  `docs/manuscript/03a_gate_design.md` and in `AILM/GateVerdict.lean`.

Gate theorems are falsifiable by a **re-measurement**: if a recorder remeasures
and a value moves across a threshold, regeneration changes `LedgerData.lean` and
the proof stops compiling. That is the intended alarm, not a maintenance burden.
For the six audited rows read that precisely: what can move is the *ceiling*
relative to its baseline. A baseline can never move across its own threshold,
because the threshold moves with it — which is the one-sidedness above, stated
as its consequence for the alarm.

## Deliberately NOT proved

Naming these is load-bearing. A formal layer that stays quiet about its
boundaries invites the reader to assume it covers more than it does.

* **Familywise error control.** `AILM/Bonferroni.lean` proves *arithmetic*:
  if each p-value is at most `t` and `n·t ≤ α`, the total is at most `α`. There
  is no probability measure here, no event algebra, no measure-theoretic union
  bound. The step from that arithmetic to "the probability of at least one false
  rejection is at most α" is Boole's inequality applied to the rejection events,
  and it is **assumed, not proved**. Citing this module as establishing FWER
  control would be exactly the necessary-is-not-sufficient defect
  `sec:gate_design` diagnoses elsewhere in this project.
* **Familywise error control for Holm, and Holm's stopping rule.**
  `AILM/Holm.lean` inherits the exclusion above unchanged and adds one of its
  own. It proves that Holm's per-rank bar dominates Bonferroni's and that the
  step-down walk therefore rejects at least as many members of a sorted family.
  It does **not** prove that Holm controls the FWER. That is Holm (1979),
  resting on the same event algebra this tree does not have. Nor does it prove
  that halting at the first failure is the *correct* stopping rule. The
  halt is a **definition** here (`stepDownCount` returns `0` on the first
  failure); its justification is a closed-testing argument, and it is assumed.
  Read "uniformly at least as powerful" in this module as a statement about
  rejection counts on a given family, which is the sense the manuscript sentence
  uses, and not as a statement about power functions under any distribution.
* **That the Lean `stepDownCount` and the Python `holm_step_down` compute the
  same rejection set.** Same gap as the `predict` row below, same reason: the
  Python sorts and divides floats, the Lean counts over exact `List Rat`, and
  nothing bridges the two. The witness family is the closest thing to a bridge:
  `tests/test_formal_isomorphism.py` re-runs the real procedure on the same four
  p-values and requires the same two counts. Agreement on one family is a check,
  not a proof of agreement in general.
* **Perplexity (`eq:perplexity`) and the order-1 floor
  (`eq:conditional_entropy`).** Both need `exp`/`log`. `Rat` has no
  transcendental functions and there is no `Real` without Mathlib. A Lean
  "theorem about perplexity" here would quantify over an uninterpreted symbol
  and prove nothing about the measured quantity.
* **That the Lean `predict` and the NumPy `transition @ pred` compute the same
  vector.** `AILM/Filtering.lean` proves that the prediction step maps the
  simplex into itself; it does not prove that `oracle.forward_filter` implements
  *that* step. Nothing bridges exact `Rat` list arithmetic to float BLAS, and
  the honest reading of the isomorphism row is "the same object, stated twice"
  rather than "the same computation, checked once". `tests/test_oracle.py` is
  what exercises the Python.
* **That interior-optimum shape implies novelty.** `AILM/SweepShape.lean` pins
  the logical relationship between the two pre-registered shape predicates and
  proves they are mutually exclusive. It does **not** prove the empirical
  working hypothesis of `sec:hierarchy` — that shape diagnoses novelty. That
  claim rests on two measured channels of each kind, in one language, and
  dressing an inductive conjecture in theorem syntax would be decoration with a
  proof attached.
* **`eq:vfe`, `eq:efe`, `eq:case_free_energy`, `eq:mcnemar`,
  `eq:decoder_equivalence`, `eq:generative`**, Clopper–Pearson intervals, and
  exact binomial power. Float or measure-theoretic; not Lean objects at this
  scope.
* **`gate_margin`.** `margin = x - t` is a `Rat` subtraction, unusable in any
  `decide`d fact, and "margin is positive exactly when `passes`" restates the
  definition. Omitted as decoration rather than descoped for difficulty.
* **That any recorded number is correct.** `AILM/Instances.lean` proves the
  recorded values have the asserted logical properties. Whether they were
  measured over the right positions, with the right split, from the right
  corpus, is the recorders' guards and the stage-gate tests — not Lean's. No
  theorem in this tree should ever be cited as evidence that a measurement is
  right, only that it is the shape the argument claims.
* **The full coupling sweep shape.** Three of its six grid points are recorded;
  see the scope note above.

### Why this list is scoping rather than laziness

Each exclusion above falls into exactly one of three kinds, and the kind is
stated in the entry rather than left for the reader to guess:

1. **Not a Lean object at this scope.** Perplexity, the order-1 floor,
   Clopper–Pearson, exact binomial power, familywise error control for either
   correction, Holm's stopping rule, the sampling distribution and interval
   coverage of the paired difference, and the float-level agreement between
   `predict` and `transition @ pred` or between `stepDownCount` and
   `holm_step_down`. These need `exp`/`log`, a measure, or IEEE
   arithmetic, and there is no `Real` without Mathlib. A theorem quantifying over
   an uninterpreted `perplexity` symbol would compile, would look like coverage
   in this README's table, and would constrain nothing about the measured
   quantity. Proving less is the honest move; proving a lookalike is worse than
   the gap.
2. **Empirical, and formalising it would launder a conjecture.** That
   interior-optimum shape diagnoses novelty is the project's *working
   hypothesis*, resting on measured channels in one language. It is exactly the
   kind of claim that gains unearned authority from theorem syntax, so it stays
   out. `SweepShape.lean` proves what is actually a theorem — that the two shape
   classes are mutually exclusive — and stops there.
3. **Decoration if proved.** `gate_margin`, and the general class of statements
   that restate a definition. A theorem that excludes no possible world is a
   line in a count, and the count is one of the numbers this project records.
   Omitting them keeps `formal_theorem_count` a measure of something.

No exclusion on the list is *deferred* — provable in core, merely unproved. One
was, and the distinction is why it is worth keeping the two apart: an unproved
thing that could be proved and an unproved thing that should not be are
different facts about the layer, and a scope list that blurs them is how a
formal layer starts implying coverage it does not have.

### What paying off the one deferral cost

The prediction half of `eq:filtering` sat on this list as "provable in core, but
it needs a sum-exchange lemma core does not provide". That reading was half
right. Core supplies no Fubini-over-lists, and it is not needed: the exchange is
an artifact of representing `B` as rows. Stored **column-major** — as
`oracle.forward_filter` already stores it, since `transition[i, j]` is
`P(next = i | cur = j)` — the product `B q` is a linear combination of columns,
the outer sum is a fold, and the exchange decomposes into two one-line
inductions: pull a scalar out of a sum (`smul_sum`), and split a pointwise sum
in two (`vadd_sum`).

The cost was twelve theory theorems and twelve witnesses, no new axioms, no new
dependency, and no `native_decide`. Three of the twelve exist only to track
widths, and every result about mass carries a width hypothesis, because
`List.zipWith` truncates rather than padding: a column narrower than the state
space silently drops mass. That bookkeeping is not overhead — it is the entire
content of `raggedColumnLosesMass`, a defect a matrix type would have made
unrepresentable and a `List Rat` model has to exclude by hand.

The general lesson, for the next entry someone is tempted to file as deferred:
"needs a lemma the library lacks" can be a claim about the chosen representation
rather than about the library, and it is worth testing which before recording
the deferral.

## Scanning this tree for incomplete proofs

The canonical scan is a test, so an absent tree or a pattern that matches
nothing cannot masquerade as a clean proof tree. Run it from the project root:

```bash
.venv/bin/python -m pytest tests/test_formal_isomorphism.py::test_no_incomplete_proofs tests/test_formal_isomorphism.py::test_the_placeholder_scan_covers_a_non_empty_file_set -v --no-cov
```

Three notes for whoever automates this:

1. Scan **`AILM/`**, not the whole directory. `Scripts/AxiomReport.lean`
   documents the axiom policy and therefore mentions `sorryAx` by name in a
   comment; a whole-tree scan false-positives on the very file that documents
   the gate.
2. Anchor the pattern to **word boundaries**. The bare-substring form the
   convention used first is what the naming constraint below is about.
3. A lexical scan is necessary but not sufficient. An unfinished proof is a
   *warning*, so `lake build` still exits `0`. The binding check is the axiom
   report: `sorryAx` in any verdict line means an unfinished proof regardless of
   the exit code. Run both — `scripts/record_formal_claims.py` does.

### The naming constraint

**No identifier or comment in `AILM/` may contain either placeholder tactic's
name as a word.**

This was earned. A theorem in `AILM/Instances.lean` was originally named
`stage4Gold_admits_no_sound_threshold`, and the bare-substring scan flagged
`admits` — a false positive that reads exactly like an incomplete proof. Both
sides were fixed: the scan is now word-boundary anchored, and the theorem was
renamed to `stage4Gold_no_threshold_is_sound`.

The scan deliberately covers comments as well as code. A scanner that parsed
comments out would be one more component capable of being wrong in the direction
of reporting clean, and the cost of keeping it dumb is only that this paragraph
lives here — in a file the scan does not read — rather than beside the theorem.

### The second constraint: keep declaration keywords off the left margin of prose

`ailm.formal_inventory`, shared by the driver generator and recorder, finds
declarations with column-anchored
patterns (`^theorem NAME`, `^instance NAME`), for the same reason: an unanchored
scan matches the word inside prose. The cost is symmetric — a doc-comment line
that happens to wrap so that "theorem existed, ..." lands at column zero reads as
a declaration named `existed`, and the axiom-report coverage check then refuses
because that "declaration" has no report entry.

This is the *safe* direction — it fails closed, loudly, at record time — and it
was earned the same way as the constraint above: a comment written while fixing
these very findings tripped it. Keep declaration keywords off the left margin of
prose. Also note what the scanner now does with forms it was not built to read:
an indented declaration, or one behind an `@[...]` attribute, is a **refusal**
rather than a silent miss, because a declaration the scanner cannot see is one
the axiom report is never required to cover.
