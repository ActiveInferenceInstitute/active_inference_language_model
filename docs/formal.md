# The formal layer

Selected mathematical properties used by AILM are machine-checked in Lean 4 under
[`../formal/`](../formal). This file is the docs-tree entry point: what is
formalized, how it binds to the claims ledger, the commands that regenerate and
re-record it, and — at least as important — what is deliberately outside its
scope.

[`../formal/README.md`](../formal/README.md) is the in-tree companion and stays
authoritative on the build mechanics: why every computed fact in the tree is an
order comparison rather than an arithmetic one, why `Rat` does not kernel-reduce
without Mathlib, the placeholder-scan naming constraint, and the per-theorem
falsifier convention. Where this file and that one disagree about how to run
something, that one wins — it lives next to the code it describes.

The Lean results do not verify the connected runtime's full execution schedule,
soft-observation message semantics, learned arrays or GUM performance. The
[connected contract](design/connected_runtime.md#connections-and-numerical-meaning)
uses independent numerical references for conditional updates; those tests are
separate from the theorems enumerated here. Neither layer proves a globally
exact joint posterior or general language competence.

## Why there is one at all

The project's own doctrine is that a green suite means whatever its weakest gate
means. Most of AILM's verification apparatus is empirical: recorders, controls,
oracles, negative controls that must fail. A few claims are not empirical at all
— they are statements about definitions, and a careful reader currently has to
re-derive them from prose to know they hold.

Those are the ones in `formal/`. The gate taxonomy either does or does not
partition its cases; the two pre-registered sweep-shape classes either are or are
not mutually exclusive; a Bayes correction either does or does not land on the
simplex; `n · (α/n)` either does or does not equal `α`; one correction's per-test
bar either does or does not dominate another's at every rank. Each is a place
where "the author says so" can become "the kernel rejects it otherwise".

The layer is small on purpose. Breadth is not the deliverable — the list of what
is *not* proved (below) is as much of it as the list of what is.

## What is formalized

| Module | Manuscript object | Content |
|---|---|---|
| [`AILM/GateVerdict.lean`](../formal/AILM/GateVerdict.lean) | `eq:gate_satisfies`, `eq:gate_verdict` | The strict audit relation vs the inclusive declared relation, and the comparator-typed VACUOUS / UNREACHABLE / ABOVE_REFERENCE / UNAUDITED / BRACKETED_BOUND / BRACKETED_REFERENCE classification, with `unreachable_requires_bound` proving impossibility verdicts need a bound-typed comparator |
| [`AILM/SweepShape.lean`](../formal/AILM/SweepShape.lean) | the shape classes of `sec:hierarchy` | `InteriorOptimum`, `NoInteriorGain`, `MonotoneDecline` and their exclusion relations |
| [`AILM/Filtering.lean`](../formal/AILM/Filtering.lean) | `eq:filtering` (both halves), `eq:forward_recursion` | Correction: normalizing an unnormalized evidence vector yields a distribution — under a hypothesis that is shown to be load-bearing. Prediction: a column-stochastic transition maps the simplex into itself, so the predicted prior is again a distribution |
| [`AILM/Bonferroni.lean`](../formal/AILM/Bonferroni.lean) | the correction behind `tbl:multiplicity_family` | The union-bound *arithmetic*, and only the arithmetic |
| [`AILM/Holm.lean`](../formal/AILM/Holm.lean) | the Holm paragraph of `98_supplement_statistics.md` | Holm's rank-`i` bar `α/(m-i+1)` is never stricter than Bonferroni's `α/m`, is nondecreasing in the rank, and coincides with it at rank 1; on a sorted family the step-down walk rejects every member the fixed bar rejects, and on a synthetic witness it rejects strictly more. Rejection-count arithmetic, and only that |
| [`AILM/LedgerData.lean`](../formal/AILM/LedgerData.lean) | the recorded values | GENERATED from `data/claims.json`; no theorems, only literals |
| [`AILM/Instances.lean`](../formal/AILM/Instances.lean) | `tbl:gate_audit`, `tbl:unaudited_gates`, `tbl:lexical_channel` | The recorded values run through the theory above |
| [`Controls/MustFail.lean`](../formal/Controls/MustFail.lean) | — | Four false statements that **must not compile**, one of them false on *recorded* data |

Recorded totals, from a real build: claim:formal_theorem_count named theorems and
claim:formal_example_count witness examples, claim:formal_axiom_report_verdicts
verdicts in the axiom report, claim:formal_native_decide_uses uses of
`native_decide`. These are ledger entries written by
[`../scripts/record_formal_claims.py`](../scripts/record_formal_claims.py), not
counts typed into this table.

## The isomorphism

Each row is one claim held in four places: a Lean declaration, the manuscript
object it transcribes, the Python that computes the same thing on real data, and
the test that exercises the Python. The value of naming all four is that a Lean
theorem with no Python counterpart is proving something about a definition the
project does not run — and this table is where that would be visible.

| Lean declaration | Manuscript object | Python source of truth | Exercising test |
|---|---|---|---|
| `GateVerdict.passes`, `satisfies`, `passes_iff_satisfies_and_ne` | `eq:gate_satisfies` | `gate_passes` — [`gates.py`](../src/ailm/gates.py) | [`test_gates.py`](../tests/test_gates.py) |
| `GateVerdict.classify`, `bracketed_iff`, `vacuous_iff`, `unreachable_of_bound_fails`, `above_reference_of_reference_fails`, `unreachable_requires_bound`, `above_reference_requires_exceedable`, `no_co_fire`, `degenerate_no_gate` | `eq:gate_verdict`, comparator typing, gate-soundness definition | `classify_gate` — [`gates.py`](../src/ailm/gates.py), imported by both [`audit_gates.py`](../scripts/audit_gates.py) and [`figures/gates.py`](../src/ailm/figures/gates.py) | [`test_gates.py`](../tests/test_gates.py) |
| `SweepShape.InteriorOptimum`, `NoInteriorGain`, `MonotoneDecline`, `interior_excludes_no_gain`, `interior_excludes_monotone` | `sec:hierarchy` shape classes (decisions 18–19) | `sweep_lexical_alpha` — [`lexical_channel.py`](../src/ailm/levels/lexical_channel.py); `sweep_alpha` — [`coupling.py`](../src/ailm/levels/coupling.py) | [`test_lexical_channel.py`](../tests/test_lexical_channel.py), [`test_coupling.py`](../tests/test_coupling.py) |
| `Filtering.normalize`, `sum_div`, `normalize_sum`, `normalize_nonneg`, `normalize_isDistribution` | `eq:filtering` correction half, `eq:forward_recursion` | `forward_filter` — [`oracle.py`](../src/ailm/verification/oracle.py), the `emission[obs_t, :] * pred_t / Z_t` step | [`test_oracle.py`](../tests/test_oracle.py) |
| `Filtering.predict`, `predict_length`, `predict_sum`, `predict_nonneg`, `predict_isDistribution` | `eq:filtering` prediction half, `eq:forward_recursion` | `forward_filter` — [`oracle.py`](../src/ailm/verification/oracle.py), the `transition @ pred` step over a column-stochastic `transition` | [`test_oracle.py`](../tests/test_oracle.py) |
| `Bonferroni.bonferroniThreshold`, `sum_le_length_mul`, `bonferroni_union_bound`, `bonferroni_union_bound_div`, `bonferroni_threshold_scales` | `tbl:multiplicity_family` | [`statistics.py`](../src/ailm/statistics.py); family assembled by `record_statistical_claims.py` | [`test_statistics.py`](../tests/test_statistics.py) |
| `Holm.holmThreshold`, `bonferroni_le_holm_threshold`, `holm_threshold_monotone_in_rank`, `holm_rank_one_is_bonferroni`, `stepDownCount`, `holm_rejects_every_bonferroni_rejection` | the Holm paragraph of `98_supplement_statistics.md` | `holm_step_down` in [`statistics.py`](../src/ailm/statistics.py) | [`test_statistics.py`](../tests/test_statistics.py); the witness counts are re-run through the real procedure by [`test_formal_isomorphism.py`](../tests/test_formal_isomorphism.py) |
| `LedgerData.*` | the recorded values | GENERATED by [`generate_lean_ledger.py`](../scripts/generate_lean_ledger.py) from `data/claims.json` | [`test_formal_isomorphism.py`](../tests/test_formal_isomorphism.py) |
| `Instances.*` | `tbl:gate_audit`, `tbl:unaudited_gates`, `tbl:lexical_channel` | the stage recorders; re-derived independently by [`audit_gates.py`](../scripts/audit_gates.py) | [`test_formal_isomorphism.py`](../tests/test_formal_isomorphism.py) |
| `Controls/MustFail.lean` | — | `verify_negative_control_fails` — [`control.py`](../src/ailm/control.py) | `test_the_negative_control_does_not_compile` |

Read the `SweepShape` row carefully. Lean proves the two pre-registered classes
are mutually exclusive. It does **not** prove either class was correctly measured,
and it does not prove the empirical working hypothesis that shape diagnoses
novelty. See the scope list.

## How to build it

The toolchain is `elan`-managed and pinned in
[`../formal/lean-toolchain`](../formal/lean-toolchain). `lakefile.toml` declares
**no dependencies** — not even Mathlib — so there is no version resolution, no
cache download, and `lake build` runs offline. Everything is proved against plain
Lean 4 core (`List`, `Rat`, `Decidable`).

```bash
# Run from the repository root.
(cd formal && lake build && ./Scripts/axiom_report.sh)
.venv/bin/python -m pytest tests/test_formal_isomorphism.py::test_the_negative_control_does_not_compile -v --no-cov
.venv/bin/python -m pytest tests/test_formal_isomorphism.py::test_no_incomplete_proofs tests/test_formal_isomorphism.py::test_the_placeholder_scan_covers_a_non_empty_file_set -v --no-cov
```

The mutation-control test checks the rejection text as well as the exit status,
so an unrelated path, import, parse, or toolchain error cannot satisfy it. The
placeholder tests scan a non-empty recursive `formal/AILM/` inventory and anchor
the pattern to word boundaries on purpose;
[`../formal/README.md`](../formal/README.md) explains both, and why a lexical scan
is necessary but not sufficient.

## Regeneration and recording

Run from the project root. The order matters: the generator reads the ledger, and
the recorder reads a real build.

```bash
# Run from the current AILM repository root.

# 1. Ledger -> Lean. The ONLY writer of formal/AILM/LedgerData.lean.
.venv/bin/python scripts/generate_lean_ledger.py
.venv/bin/python scripts/generate_lean_ledger.py --check   # non-zero if stale

# 2. Build -> ledger. Writes the formal_* claims from a real lake build
#    and the regenerated axiom report; refuses to record if the report's
#    entries and the declarations in AILM/*.lean are not the same set.
.venv/bin/python scripts/record_formal_claims.py

# 3. The binding tests. Ledger checks always run; Lean checks skip cleanly
#    (naming the pinned toolchain) when lake is absent.
.venv/bin/python -m pytest tests/test_formal_isomorphism.py -q --no-cov

# 4. Ledger -> manuscript tokens, as for every other number.
.venv/bin/python scripts/z_generate_manuscript_variables.py
```

A change to any recorded value that the Lean layer reads means step 1 *and*
step 2, in that order. Step 1 alone leaves `formal_theorem_count` and its
siblings describing the previous build; step 2 alone leaves the proofs about
stale numbers.

## The honesty doctrine, and what each rule cost

Every rule below was earned by a defect this project or its siblings shipped.
They are enforced by [`../tests/test_formal_isomorphism.py`](../tests/test_formal_isomorphism.py)
and by `record_formal_claims.py`, not by convention.

- **Zero `sorry` / `admit`, checked twice.** A word-boundary grep *and* the axiom
  report. Lean emits a *warning*, not an error, for an unfinished proof, so
  `lake build` exits `0` with one in the tree; `sorryAx` in a verdict line is the
  check that actually binds. The recorded verdict is `claim:formal_sorry_free`.
- **No vacuous conditionals.** A theorem whose hypothesis cannot be satisfied
  compiles clean and proves nothing. Every conditional theorem here carries a
  satisfiability witness — an instance proving the hypothesis is inhabited, and
  an instance firing the theorem on it — and where a hypothesis is load-bearing
  there is also a negative control showing the conclusion fails without it
  (`Filtering.zeroMassCounterexample`,
  `Bonferroni.scaleHypothesisIsLoadBearing`,
  `Bonferroni.zeroFamilyThresholdCollapses`, `Holm.sortednessIsLoadBearing`,
  `Holm.holm_rank_positivity_is_load_bearing`).
  This sentence was false when first written, and the way it was false is the
  rule's own subject: `bonferroni_union_bound_div` stated its hypothesis against
  `bonferroniThreshold α n` while the witness family was bounded against a
  literal, so the two never met and the divided form of the correction — the form
  the manuscript writes — was never instantiated. A doc asserting a universal is
  not a check that the universal holds.
- **`decide` over `native_decide`.** `native_decide` adds the
  `Lean.ofReduceBool` trusted-code axiom, trading kernel checking for compiler
  trust. Recorded uses: `claim:formal_native_decide_uses`.
- **`#print axioms` on every named theorem, recorded.** The allowed set is
  `propext`, `Quot.sound`, `funext`, `Classical.choice`. The number of axioms
  reached beyond the core three is `claim:formal_axioms_beyond_core`, and the one
  reached is
  `Classical.choice`, which is on the list for a *measured* reason — Lean core's
  own `Rat.add_comm` reports it, so a stricter policy would be unsatisfiable
  against core `Rat` and would fail for reasons unrelated to anything proved
  here. Do not tighten it without checking that first.
- **Every theorem must exclude a possible world.** Each carries a doc comment
  naming its manuscript object and stating what would falsify it. A theorem that
  restates a definition is decoration, and decoration inflates a recorded count.
- **A negative control that must not compile.** `Controls/MustFail.lean` sits
  outside the library root and asserts four false statements, each false for a
  different reason so the control cannot pass on one unrelated error. The fourth
  is refuted on *recorded* ledger values — without that one, every `by decide` in
  `Instances.lean` would be an assertion that the kernel evaluated something.

## The ledger bridge

Exactly one module holds measured values —
[`AILM/LedgerData.lean`](../formal/AILM/LedgerData.lean) — and it is generated.
Every literal in the five theory modules is a *synthetic witness* chosen for its
shape, and every witness block says so in place. There is no third kind of
number, and **no ledger value is ever hand-typed into Lean.**

`tests/test_formal_isomorphism.py` requires a fresh generation to reproduce the
committed file byte for byte. That one check closes both directions: a hand edit
no longer matches a fresh run, and a stale file no longer matches a moved ledger.
Byte identity alone would only prove the generator is deterministic, so a second,
independently written oracle re-derives every emitted literal from
`data/claims.json` and compares — a generator that consistently truncated a
decimal would pass the identity check forever and fail that one on the first
value.

The conversion is exact by construction: `data/claims.json` is parsed with
`parse_float=str`, so the source is the decimal *text*, which becomes a
`Fraction`, is reduced, and is emitted as `lit n d`. `lit`
([`AILM/Core.lean`](../formal/AILM/Core.lean), the tree's only literal
constructor) is built on `Rat.mk'`
with a `Nat.Coprime` field discharged by `decide`, so an unreduced fraction is a
**compile error** rather than a silently wrong number. No floats, no rounding, no
tolerance anywhere in the chain.

The consequence is the falsifier: a re-measurement that moved a sweep's argmax
off its interior point, or moved a baseline or ceiling across a declared
threshold, changes the generated file and the corresponding proof stops
compiling. That is the intended alarm, not a maintenance burden.

### What a bracketed verdict claims on the six audited rows, and what it does not

Read this before citing `stage1_bracketed` or any of its five siblings.
`scripts/audit_gates.py` sets each audited row's threshold *to* its measured
baseline, so the generated `…Threshold` is a definitional alias of the
`…Baseline`. `bracketed_iff` says a bracketed verdict means the baseline fails the bar
and the ceiling clears it — but with the bar equal to the baseline, the first
half is `passes_irrefl`, true for every population and either direction. It is a
tautology, not a measurement.

`bracketed_at_baseline_bar_iff` proves the reduction: at `t = b`, a bracketed verdict is
*equivalent* to "the ceiling strictly beats the baseline". So each of those six
theorems asserts exactly one order comparison on recorded data. That comparison
is real and falsifiable — a re-measurement that moved a ceiling below its
baseline breaks the proof — and it is all they assert. A baseline can never move
across a threshold defined to equal it, so the alarm above fires on ceilings
only for these rows.

Two things keep them from being decoration anyway: the surviving comparison is a
genuine measured fact, and `NC5`–`NC10` drive the same recorded Stage 1
population to five of the six verdicts by moving the bar *off* the baseline and
flipping the declared comparator kind — which is
how `classify` is shown non-constant on real data, a demonstration the six
aliased rows cannot make themselves.

Relatedly: "ceiling" in this project names a *measured comparator*, not a bound
over the model class. The Stage 4 derived row's shipped agent scores above its
recorded smoothing ceiling. No theorem here assumes otherwise; prose that once
did has been corrected.

## Deliberately not formalized

Naming these is load-bearing. A formal layer that stays quiet about its
boundaries invites a reader to assume it covers more than it does. The full list
with per-item reasoning is in [`../formal/README.md`](../formal/README.md); the
summary and the *kinds* are here.

- **Familywise error control.** `Bonferroni.lean` proves arithmetic: if each
  p-value is at most `t` and `n·t ≤ α`, the total is at most `α`. There is no
  probability measure, no event algebra, no measure-theoretic union bound. The
  step to "the probability of at least one false rejection is at most α" is
  Boole's inequality applied to the rejection events, and it is **assumed**.
  Citing this module as establishing FWER control would be exactly the
  necessary-is-not-sufficient defect [`gate_design.md`](gate_design.md) diagnoses
  elsewhere.
- **Familywise error control for Holm, and Holm's stopping rule.** `Holm.lean`
  inherits the exclusion above and adds one of its own. It proves that Holm's
  per-rank bar dominates Bonferroni's, so the step-down walk rejects at least as
  many members of a sorted family. It does **not** prove that Holm controls the
  FWER, and it does not prove that halting at the first failure is the correct
  stopping rule. The halt is a definition here, and its justification is a
  closed-testing argument this tree has no measure to state. Read "uniformly at
  least as powerful" as a statement about rejection counts on a given family,
  the sense the manuscript sentence uses, not about power functions.
- **That `stepDownCount` and `holm_step_down` compute the same rejection set.**
  The Python sorts and divides floats; the Lean counts over exact `List Rat`.
  The witness family is checked through both, which is a check on one family
  rather than a proof of agreement in general.
The list once carried a fourth kind, *deferred*: the prediction half of
`eq:filtering`, held back for a sum-exchange lemma core does not provide. It is
now proved, and the reason it was ever hard is worth keeping. The exchange was an
artifact of representing the transition as rows; stored column-major, as
`oracle.forward_filter` already stores it, `B q` is a linear combination of
columns, the outer sum is a fold, and the lemma decomposes into two one-line
inductions. The blocker was a representation, not a missing primitive — which is
the failure mode to suspect first the next time this list grows a deferral.

- **Perplexity and the order-1 floor.** Both need `exp`/`log`. `Rat` has no
  transcendental functions and there is no `Real` without Mathlib.
- **That interior-optimum shape implies novelty.** The empirical working
  hypothesis of `sec:hierarchy`, resting on measured channels in one language.
- **`eq:vfe`, `eq:efe`, `eq:case_free_energy`, `eq:mcnemar`,
  `eq:decoder_equivalence`, `eq:generative`**, Clopper–Pearson intervals, exact
  binomial power. Float or measure-theoretic.
- **`gate_margin`.** "Margin is positive exactly when `passes`" restates the
  definition. Omitted as decoration, not descoped for difficulty.
- **That any recorded number is correct.** Lean checks the recorded values have
  the logical properties the manuscript and `audit_gates.py` assert of them. It
  cannot check they were measured over the right positions, with the right split,
  from the right corpus — that is the recorders' guards and the stage gates.

### Scoping, not laziness

Every exclusion falls into one of three kinds, and the entry says which:

1. **Not a Lean object at this scope** — needs `Real` or a measure. A theorem
   quantifying over an uninterpreted `perplexity` symbol would compile, would
   look like coverage in the table above, and would constrain nothing. Proving
   less is honest; proving a lookalike is worse than the gap.
2. **Empirical, and formalising it would launder a conjecture** — the novelty
   hypothesis. It is precisely the kind of claim that gains unearned authority
   from theorem syntax.
3. **Decoration if proved** — `gate_margin` and the general class of statements
   that exclude no possible world. Each would be a line in
   `claim:formal_theorem_count`, and that count is meant to measure something.

No current item is *deferred* rather than descoped. One historically was—the
filtering prediction proof described above—and paying it off is why the
distinction remains visible: an unproved thing that could be proved and an
unproved thing that should not be are different facts about this layer. A scope
list that blurs them is how a formal layer starts implying coverage it does not
have.

## See also

- [`../formal/README.md`](../formal/README.md) — the in-tree reference: `Rat`
  reduction behavior, the axiom policy in full, the placeholder-scan naming
  constraint, the per-theorem falsifier convention
- [`gate_design.md`](gate_design.md) — the gate taxonomy the Lean transcribes
- [`testing_philosophy.md`](testing_philosophy.md) — what a green suite is
  allowed to mean, which is the same argument one layer down
- [`formalism_map.md`](reference/formalism_map.md) — every labeled equation mapped to the
  code that computes it and the test that exercises it
- [`../manuscript/07_reproducibility.md`](manuscript/07_reproducibility.md) —
  the manuscript's account of this layer, written against ledger tokens
