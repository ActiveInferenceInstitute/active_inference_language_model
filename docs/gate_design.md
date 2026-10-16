# Gate Design

How to specify a level gate that can actually fail, and that a correct
implementation can actually pass.

This project has shipped **three** defects in this area, in three different
directions: a gate nothing could pass, a gate anything could pass, and an
evaluation set that made a thin result look comfortable. All three were caught
after the fact. All implemented quantitative gates are now audited; this
document is the reusable contract for changing one or adding another.

- [What a gate is](#what-a-gate-is)
- [The typed gate-verdict taxonomy](#the-typed-gate-verdict-taxonomy)
- [Defect 1 — Stage 1 was UNREACHABLE](#defect-1--stage-1-was-unreachable)
- [How to compute an information ceiling](#how-to-compute-an-information-ceiling)
- [Defect 2 — Stage 2 was VACUOUS](#defect-2--stage-2-was-vacuous)
- [Uniform is a crippled comparator](#uniform-is-a-crippled-comparator)
- [Beating the baseline is necessary, not sufficient](#beating-the-baseline-is-necessary-not-sufficient)
- [Defect 3 — reproducible is not representative](#defect-3--reproducible-is-not-representative)
- [A thin margin needs a paired test](#a-thin-margin-needs-a-paired-test)
- [When the full set is too slow: oracle equivalence](#when-the-full-set-is-too-slow-oracle-equivalence)
- [Using scripts/audit_gates.py](#using-scriptsaudit_gatespy)
- [Pre-build checklist](#pre-build-checklist)

---

## What a gate is

A gate is a falsifiable numeric claim about one level of the hierarchy, stated
before that level is built, with three parts:

| Part | Meaning | Failure if absent |
|---|---|---|
| **Metric** | What is measured, over which evaluation set, with which direction (lower/higher is better) | "accuracy" without an eval set is unfalsifiable |
| **Comparator** | The score a model that does *not* do the thing being claimed would get | Without it, a number is not evidence of anything |
| **Threshold** | The value that separates pass from fail | Without it, every result is a pass in retrospect |

A gate with all three parts can still be worthless. The two remaining questions
are whether the comparator already clears the threshold, and whether anything
can. Those are the two failure modes below, and they are opposites.

---

## The typed gate-verdict taxonomy

The complete enum is `VACUOUS`, `UNREACHABLE`, `ABOVE_REFERENCE`, `UNAUDITED`,
`BRACKETED_BOUND`, and `BRACKETED_REFERENCE`.

The classification is mechanical once the trivial baseline and the achievable
ceiling have been measured. The single implementation is
[`../src/ailm/gates.py`](../src/ailm/gates.py); the audit script and figures
delegate to it:

Every ceiling carries a declared `ComparatorType` — `model_class_bound`,
`analytic_bound`, `privileged_oracle`, or `reference_model` — and the verdict
depends on it, because only a bound licenses an impossibility claim:

| Verdict | Condition | What it means |
|---|---|---|
| **VACUOUS** | The trivial baseline already satisfies the threshold | A model doing nothing hierarchical passes. The gate constrains nothing. |
| **UNREACHABLE** | A *bound-typed* ceiling cannot satisfy the threshold | No correct implementation can pass **on the population the bound was measured on**. The gate blocks the stage forever. |
| **ABOVE_REFERENCE** | An *exceedable* ceiling (reference or oracle) cannot satisfy the threshold | The bar sits above that comparator; another model in the class may still exceed it, so no impossibility claim follows. |
| **BRACKETED_BOUND** | Baseline fails, a bound-typed ceiling clears | The gate discriminates, and the passing region is capped by a genuine bound. |
| **BRACKETED_REFERENCE** | Baseline fails, an exceedable ceiling clears | The gate discriminates; the comparator could itself be exceeded. |
| **UNAUDITED** | The baseline fails, but no ceiling is available | Reachability is unknown; missing evidence is not treated as a favorable verdict. |

The two measurements that decide this — baseline and ceiling — are the whole
technique. Neither is optional, and measuring only one gives you exactly one of
the two defects this project shipped.

> **A gate is bracketed only when the trivial baseline FAILS it and the measured
> ceiling CLEARS it — by-bound or by-reference according to the comparator's
> declared type.** Everything below is the elaboration of that sentence.

The visual version of this taxonomy is
`{#fig:gate_taxonomy}` (`output/figures/gate_taxonomy.png`).

---

## Defect 1 — Stage 1 was UNREACHABLE

The Stage 1 gate required a gain of **claim:stage1_declared_gate_nats nats over uniform**.

That target is information-theoretically unreachable for *any* letter bigram
model, correct or not — on both populations, which took two separate bounds to
establish. On the training distribution, its conditional entropy caps the
available gain over uniform at claim:stage1_max_bigram_gain_nats nats. On the
held-out passage the gate actually scores, the passage's transition conditional
entropy plus the fixed uniform-start loss, averaged over all scored positions,
caps it at claim:stage1_heldout_max_gain_nats nats. The first bound does not
imply the second — a training conditional entropy is not a bound on held-out
cross-entropy — and for a while this document asserted the conclusion from the
first alone. A transitions-only held-out entropy is not the all-position metric
either: `compute_perplexity` also scores the first character, and the recorder's
bound includes that term. Continuing with the training figure:
the demanded gain is roughly double the ceiling (the exact ratio is the derived
manuscript token `{{STAGE1_CEILING_SHORTFALL_RATIO}}`; it is computed in
[`../src/ailm/manuscript_variables/`](../src/ailm/manuscript_variables/__init__.py), not
typed here). The shipped agent realizes claim:stage1_gain_over_uniform_nats nats,
i.e. claim:stage1_bigram_signal_captured_pct% of what the model order permits.

So the agent was near-optimal for its order and still failing the gate. Two
things that could have happened, and one that nearly did:

1. Conclude the implementation is broken and rewrite working code.
2. Quietly lower the threshold until the current number passes — which converts
   the gate into a description of whatever was built.
3. Measure the ceiling, discover the target was impossible, and restate the gate
   against a comparator that is both reachable and diagnostic.

Only (3) is honest, and it requires the ceiling *number*, not an intuition. The
restated gate is "beats the context-free unigram baseline"
(claim:stage1_unigram_baseline versus claim:stage1_letter_perplexity), which a
model cannot clear without using the previous letter.

**The tell for UNREACHABLE:** a threshold expressed as a round constant borrowed
from somewhere else (claim:stage1_declared_gate_nats nats,
claim:stage3_declared_gate perplexity, or claim:stage4_declared_gate%) with no derivation
showing it is inside the achievable range.

A gain of claim:stage1_declared_gate_nats nats is a *trigram-or-higher* target. Do not carry a threshold across
model orders.

---

## How to compute an information ceiling

For an unrestricted order-one conditional table evaluated on the same empirical
transition distribution used to compute the bound, its minimum cross-entropy
is that distribution’s conditional entropy:

$$
H(X_t \mid X_{t-1}) = -\sum_{j} P(x_{t-1} = j) \sum_{i} P(x_t = i \mid x_{t-1} = j) \log P(x_t = i \mid x_{t-1} = j)
$$

and the corresponding perplexity floor is $\exp\big(H(X_t \mid X_{t-1})\big)$.
The implementation is `_conditional_entropy` in
[`../scripts/audit_gates.py`](../scripts/audit_gates.py): build the joint from
the raw `"prev,next"` bigram counts, normalize it, marginalize for
$P(x_{t-1})$, and accumulate per-column.

Three details that decide whether the number you get is the number you want:

- **Smoothed or unsmoothed.** The floor is conventionally the *unsmoothed* MLE
  table. Add-1 smoothing raises it. For phonemes the two differ in the third
  decimal place, which is enough to make an equality assertion fail on correct
  code. The module docstring of
  [`../src/ailm/levels/phoneme.py`](../src/ailm/levels/phoneme.py) records both.
- **Which positions count.** The floor over genuine *transitions* excludes
  sequence-initial positions, which have no predecessor. A language-model
  perplexity that scores every position pays for position 0 under the start
  prior `D`, so it is legitimately *above* a transitions-only floor. Comparing
  the two is a category error: a gate asserting
  `compute_phoneme_perplexity(...) ≈ floor` can never pass. Stage 2's numbers
  are claim:stage2_phoneme_perplexity (all positions, per-sequence reset) and
  claim:stage2_phoneme_floor_perplexity (transitions only, unsmoothed) — both
  correct, not interchangeable. The Stage 2A gate instead uses
  claim:stage2_phoneme_all_position_floor_perplexity, computed by
  `full_corpus_order1_floor_perplexity` from the maximum-likelihood start and
  transition distributions over the identical full-dictionary denominator.
- **Float width.** JAX defaults to float32, so analytic anchors land within
  roughly 1e-7 relative. Assert with a tolerance; never for equality.

For a classification metric the comparator is not an entropy but the best decode
available to a strictly more powerful inference procedure — a *reference*, not a
bound: it can be, and at Stage 4 is, exceeded by the shipped model. For Stage 2's
letter-to-phone level (position-aligned; `g2p` in code) that reference is forward–backward smoothing
(claim:stage2_g2p_smoothing_ceiling%) and Viterbi
(claim:stage2_g2p_viterbi_ceiling%), both of which see the whole word, against a
filtering agent that sees only left context. The claim labels say `ceiling`
because they are identifiers the ledger resolves and renaming them would break
the single-writer discipline; read them as references. Recording the comparator
makes the deliberately unclaimed headroom visible instead of hidden.

---

## Defect 2 — Stage 2 was VACUOUS

The Stage 2 gate copied Stage 4's **claim:stage4_declared_gate% threshold** and
named the English EWT oracle. Wrong twice:

1. **Wrong corpus.** UD English EWT is a syntactic treebank; its CoNLL-U fields
   are `lemma`/`upos`/`feats`/`deprel`, with no phonemic annotation of any kind.
   The line was a copy-paste of Stage 4's case gate. A phoneme task cannot be
   scored on it at all.
2. **Vacuous threshold.** A context-free per-letter lookup table — no transition
   matrix, no pymdp, no inference — scores
   claim:stage2_g2p_baseline_accuracy% on held-out CMUdict. The declared gate
   would have been passed by a model doing nothing hierarchical.

Note the direction. Stage 1's threshold was too hard for anything; Stage 2's was
too easy for everything. **Both were found by the same procedure** — measure the
trivial baseline and the achievable ceiling before accepting the number — which
is why that procedure is now a script rather than a habit.

**The tell for VACUOUS:** nobody has written down what a model that does *not*
implement the claimed mechanism would score. If the answer is unknown, the gate's
status is unknown.

**The tell for wrong-corpus:** the gate names a dataset whose field list has
never been checked against the metric. Check the fields, not the reputation.

---

## Uniform is a crippled comparator

Uniform is the comparator that feels rigorous and proves nothing, because a
model can beat uniform while using no context whatsoever.

The Stage 2 phoneme ladder, in order:

| Comparator | Uses context? | Value |
|---|---|---|
| Uniform over the phone inventory | no | claim:stage2_phoneme_uniform_baseline |
| Context-free unigram phone frequencies | no | claim:stage2_phoneme_unigram_baseline |
| **Phoneme bigram agent** | yes, one step | **claim:stage2_phoneme_perplexity** |
| Exact all-position order-1 MLE floor | model-class bound | claim:stage2_phoneme_all_position_floor_perplexity |
| Transitions-only conditional entropy | diagnostic | claim:stage2_phoneme_floor_perplexity |

And for G2P, where the spread is starker: a uniform-random phone guess sits near
1/`N_PHONES`, and even a majority-phone constant predictor reaches about a tenth
of the positions — while `per_letter_argmax_baseline` reaches
claim:stage2_g2p_baseline_accuracy%. Gating against uniform here would have been
a gate against the reciprocal of the phone-inventory size.

**Rule.** The comparator must be the *strongest model that lacks the specific
mechanism the level claims to add.* For a sequence level, that is the
context-free unigram model. For a level that infers a hidden alphabet from an
observed one, it is the context-free per-observation lookup table. Uniform belongs in the report as a scale marker
(see `{#fig:perplexity_ladder}`), never as the bar.

The corollary bites in the other direction too: do not choose a comparator so
weak that beating it is uninformative, and do not choose one so strong that it
already implements the mechanism (then the gate measures implementation quality,
not presence of mechanism).

---

## Beating the baseline is necessary, not sufficient

This is the Level 2 lesson, and it is the reason the taxonomy above is a floor
rather than a ceiling on gate design.

Stage 1's wiring bug — the learned bigram matrix in the emission `A` with an
identity `B` — is *anti-informative* rather than loudly broken. Observing symbol
`c` yields a posterior over the symbol that **preceded** `c`, and an identity `B`
carries that backward-looking belief forward as the next-symbol prediction. It
measured claim:stage1_anti_informative_perplexity against
claim:stage1_letter_perplexity for correct wiring, and on a small in-sample
corpus scored *worse than uniform* while still slipping under a naive `< 28.0`
threshold. pymdp cannot catch this class of error: a bigram matrix is
column-stochastic and satisfies `_validate()` in either slot.

At Level 2 there are two learned matrices and therefore twice the room. Three
mis-wirings were built and measured against the context-free bar (figures and the
exact per-variant numbers are in the
[`../src/ailm/levels/phoneme.py`](../src/ailm/levels/phoneme.py) module
docstring, derived rather than retyped here):

| Mis-wiring | Caught by "beats the baseline"? |
|---|---|
| Time-reversed `B` (`P(prev \| cur)`) | yes — drops below the bar |
| Uniform, information-free `A` | yes — collapses |
| `A` normalized over phones instead of letters, then renormalized so pymdp still validates it | **no** — clears the bar while being wrong |

The third variant is the important one. It beats the context-free baseline. Any
gate whose only guard is `assert model > baseline` accepts it.

What actually pins the emission down is two things the baseline comparison cannot
express:

1. **A structural invariant asserted on the right axis.** `A[0, :, p]` must sum
   to 1 *before* any renormalization, and column-stochasticity must be asserted
   along the specific axis — a transposed matrix still sums to 1 along *some*
   axis, so `assert np.allclose(M.sum(axis=...), 1)` with the wrong axis is a
   green check on broken code. Both assertions live in
   [`../tests/test_stage2_gate.py`](../tests/test_stage2_gate.py)
   (`test_stage2_g2p_emission_orientation_is_pinned`).
2. **Numerical agreement with an independent closed-form oracle.** See below.

**Rule.** Every gate needs at least one guard that is not a comparison against a
baseline: an invariant, an oracle agreement, or an explicit mis-wiring that must
fail. Enumerate the mis-wirings the level makes available and test each one —
finding that one of them survives your guard set is the point of the exercise.

---

## Defect 3 — reproducible is not representative

The G2P evaluation originally subsampled with `pairs[:limit]`, on the reasoning
that a deterministic prefix is reproducible. It is. It is also biased:
`g2p_pairs` is alphabetically sorted, so a prefix is **only a/b/c words**.

The prefix reported a margin several times the full-set
claim:stage2_g2p_margin_pts pts — a comfortable-looking pass built entirely out
of the slice. The exact prefix figure and the overstatement factor are derived in
[`../scripts/record_stage2_claims.py`](../scripts/record_stage2_claims.py) and
retained in the dated [development history](development_history.md); they are not retyped here,
because a number that appears in two places drifts in one of them.

The fix was to subsample by **even spacing** (a stride), which spans all 26
initials at the same cost. `_iter_pairs` in
[`../src/ailm/levels/phoneme.py`](../src/ailm/levels/phoneme.py) does that now.

Then the second, sharper problem appeared. A *representative* stride of the same
size is **underpowered**: over claim:stage2_g2p_subsample_positions positions the
paired test gives p = claim:stage2_g2p_subsample_mcnemar_p, an order of magnitude
weaker than the full-set p = claim:stage2_g2p_mcnemar_p over
claim:stage2_g2p_eval_positions positions. The biased prefix had looked
significant *for the wrong reason*: its inflated margin bought the p-value that
its sample size could not.

The same bias recurred on Part A. Measured on the length-matched G2P subset, the
phoneme perplexity is claim:stage2_phoneme_subset_perplexity against
claim:stage2_phoneme_perplexity over the full dictionary
(claim:stage2_phoneme_eval_positions positions), so Part A is now computed in
closed form over every position rather than on whatever subset happened to be at
hand.

The executable cross-stage audit also keeps its Part A comparators on one
population. The model, unigram bar, and empirical order-one floor all consume
the full-dictionary sufficient statistics and the same
denominator of claim:stage2_phoneme_eval_positions positions. The floor includes
each word's reset under `D` as well as its within-word transitions under `B`.
This is explicitly an in-sample dictionary protocol, not held-out
generalization. A transitions-only entropy remains a useful optimistic
reachability quantity, but it is not a bound on an all-position score and is
therefore not used as that audit row's ceiling.

**Rules.**

- Reproducibility and representativeness are independent properties. A seed or a
  sorted prefix buys the first and says nothing about the second.
- Before trusting a subsample, ask what the data is *sorted by*. If it is sorted
  by anything correlated with the metric, a prefix is a stratified sample of one
  stratum.
- State the eval-set size next to every reported number, and record it as a
  claim (claim:stage2_g2p_eval_positions,
  claim:stage2_phoneme_eval_positions). A margin without an *n* is not a result.
- If subsampling changes the verdict, the subsample is not an optimization, it
  is the finding. `{#fig:subsample_bias}` plots prefix versus stride versus full.

---

## A thin margin needs a paired test

Stage 2's G2P margin is claim:stage2_g2p_margin_pts pts
(claim:stage2_g2p_accuracy% against claim:stage2_g2p_baseline_accuracy%). That is
thin **by construction**, not by weakness: on a transparent-orthography subset
the letter alone is nearly sufficient, which is exactly why the context-free
baseline is strong (see [`design/stage2_g2p.md`](design/stage2_g2p.md)).

A bare `assert model_accuracy > baseline_accuracy` on half a point is not
evidence. It is within the range a different tie-break rule or smoothing constant
could produce. Comparing two *aggregate* accuracies throws away the pairing that
makes the comparison powerful.

The gate therefore pairs the two decoders position-for-position
(`decode_positions` / `baseline_positions`) and runs **McNemar** on the
discordant pairs: claim:stage2_g2p_mcnemar_b positions where only filtering is
right, claim:stage2_g2p_mcnemar_c where only the baseline is,
p = claim:stage2_g2p_mcnemar_p.

Two controls make the test itself falsifiable:

- **A no-effect control.** Run the same procedure on two *identical* predictors
  and require a non-significant result
  (`test_stage2_mcnemar_can_return_no_effect`). Without it, a significance test
  that always returns "significant" would read as a pass.
- **Agreement between the public accuracy API and the paired positions**
  (`test_stage2_g2p_public_accuracy_api_matches_paired_positions`), so the
  headline number and the tested number cannot diverge.

**Rule.** Decide the significance machinery *when specifying the gate*, from the
expected margin and the eval-set size — not after seeing a disappointing
inequality. If the honest margin is small, say so in the gate, and gate on the
paired statistic.

---

## When the full set is too slow: oracle equivalence

An eager per-position loop can make a full-corpus evaluation expensive. That
is an execution characteristic, not a reason to change the registered population.
The connected runtime now caches compiled calls to the actual pymdp methods;
its full GUM recorder uses that path rather than substituting a NumPy scorer.

The tempting responses are both wrong: shrink the eval set (defect 3) or report
the fast approximation as if it were the agent.

The pattern that is honest:

1. Implement an **independent** closed-form oracle —
   [`../src/ailm/verification/oracle.py`](../src/ailm/verification/oracle.py):
   `forward_filter`, `forward_backward`, `viterbi`, `sequence_perplexity`,
   `markov_chain_perplexity`. Pure numpy. **It must never import pymdp or jax**,
   because an auditor that shares code with the thing it audits inherits the
   flaw it audits for.
2. Check the two decoders on the declared bounded subset — position-level
   agreement, count claim:stage2_decoder_disagreements disagreements.
3. Use the fast oracle for the full-set statistics.
4. **Refuse to record** if the disagreement count is ever nonzero, since the
   full-set figures would then no longer describe the pymdp agent.
   `verify_decoder_equivalence` in
   [`../scripts/record_stage2_claims.py`](../scripts/record_stage2_claims.py)
   enforces step 4.

Step 4 makes the substitution conditional on observed agreement. It does not
turn a bounded check into a proof for every input, and equal decoded labels do
not imply equal predictive distributions. Report the checked population and
quantity; compare probabilities directly when the claim concerns log loss.

The same oracle is what caught the Stage 1 wiring bug, as a throwaway scratch
script that disagreed materially with pymdp's perplexity. It
is now a tested module precisely so every level can be checked the same way
instead of the instrument being rebuilt ad hoc.

---

## Using `scripts/audit_gates.py`

```bash
cd "$AILM_REPO_ROOT"
.venv/bin/python scripts/audit_gates.py
.venv/bin/python scripts/audit_gates.py --limit 300   # faster, noisier
```

Per stage it prints the baseline, the model, the ceiling, whether the model sits
on the right side of the bar (with the margin), the verdict, and the original
declared gate with its historical verdict. The L7 speech boundary is listed
separately because it is an interface contract rather than a numeric model gate.

Exit status is nonzero if an ordinary audited gate is not BRACKETED or if a positive
finding’s model fails its bar. The canonical Stage 5 row is the widened
observation arm promoted by decision 114. The retained case-only ablation is
checked separately against its declared non-passing outcome: its placement,
reference-gain intervals and planted-effect control must remain consistent.
A clean audit therefore does not mean every retained experimental arm improved.
The audit writes distinct canonical and ablation reports when `--json` is used.

Two caveats that are printed in the output and are worth restating:

- `--limit` bounds the G2P evaluation. A bounded subsample is enough to
  *classify* the gate (baseline versus ceiling versus threshold) but **not** to
  establish significance. The gate statistic lives in
  [`../scripts/record_stage2_claims.py`](../scripts/record_stage2_claims.py),
  which runs the full set.
- The subsample values printed by the audit will not equal the recorded claims,
  and should not be quoted as results. The claims ledger
  ([`../data/claims.json`](../data/claims.json)) is the source of record; the
  audit is a classification instrument.

**Adding a stage to the audit.** Write an `audit_stageN()` returning the dict
shape the existing ones return (`gate_id`, `baseline`, `baseline_name`, `model`, `ceiling`,
`ceiling_name`, `ceiling_type`, `threshold`, `threshold_name`, `lower_is_better`,
`sesoi_margin`, plus `original`
and `original_verdict` for the historical record), add it to the `audits` list in
`main()`, add its stable identity to `GateId`/`AUDITED_GATE_IDS` and the registered SESOI
contract, and register a
representative negative control in `tests/test_control_meta_validation.py`.
Inventory equality is tested, so an unregistered gate fails closed.

---

## Pre-build checklist

Work through this list *before* writing a new level or changing a metric. After
the result exists, the temptation to tune the threshold to it is hard to resist,
and a threshold fitted to a result is not a gate.

For the stage you are about to build:

- [ ] **Name the metric, the eval set, and the direction.** Write down the eval
      set's size and how it was selected. Check what it is sorted by.
- [ ] **Check the corpus actually carries the annotation the metric needs.** List
      the fields. This is the Stage 2 wrong-corpus check, and it costs one
      minute.
- [ ] **Measure the trivial baseline** — the strongest model lacking the
      mechanism the level claims. If it clears the declared threshold, the gate
      is VACUOUS; restate it against the baseline.
- [ ] **Measure the achievable ceiling** — conditional entropy for a sequence
      metric, or a strictly more powerful decode for a classification metric.
      Record its comparator type: only a bound that cannot clear the threshold
      licenses UNREACHABLE; an exceedable reference gives ABOVE_REFERENCE.
      Record any correction and *why* in [`../CHANGELOG.md`](../CHANGELOG.md).
- [ ] **Estimate the expected margin.** If it is thin, specify the paired
      statistic and its no-effect control now.
- [ ] **Enumerate the mis-wirings this level makes available** (which matrices
      are learned? which could be transposed, mis-normalized, or time-reversed?)
      and write one negative control per mis-wiring. Confirm each fails, and note
      any that your guard set does *not* catch.
- [ ] **Wire the oracle cross-check** before trusting any number. Extend
      [`../src/ailm/verification/oracle.py`](../src/ailm/verification/oracle.py)
      rather than writing a scratch script.
- [ ] **Add `audit_stageN()`** to
      [`../scripts/audit_gates.py`](../scripts/audit_gates.py) and remove the
      stage from `UNAUDITED_STAGES`.
- [ ] **Record every number as a claim** and reference it as `claim:<label>` in
      prose (see [`syntax_guide.md`](syntax_guide.md)).

The following are historical declarations and their failure modes, retained as
lessons rather than current specifications. The [hierarchy](hierarchy.md) and
[decision chronology](decisions.md) govern the repaired gates.

Stage-specific warnings already known:

| Stage | Declared gate | Known hazard |
|---|---|---|
| **3** words | perplexity < claim:stage3_declared_gate on PTB-1k | A round borrowed constant with no derivation — the Stage 1 signature. Measure the trie-constrained ceiling and a context-free word-unigram baseline before accepting that threshold. Also decide whether the G2P alignment restriction carries forward, and restate what is being measured if so. |
| **4** cases | ≥claim:stage4_declared_gate% vs the UD 8-case oracle | The oracle is a **research claim, not ground truth**: only NOM/ACC/GEN are morphologically marked in English; DAT/INS/LOC/ABL are derived from `deprel`. Agreement with it is *agreement*, not accuracy, and must not be reported as the latter. Measure what a majority-class and a deprel-only predictor score — the same threshold has the shape of Stage 2's vacuous gate. |
| **5** clauses | free energy tracks clause length | "Tracks" is not a threshold. Specify the statistic (a correlation? a monotonicity sweep?), its eval set, and what a model with shuffled clause structure scores. A monotonicity claim needs an exhaustive sweep, not a few examples. |
| **6** narrative | `narrative C vector constrains clause priors` (legacy declaration; the shipped seam intervenes on L5's `D`) | A mechanism assertion, not a measurement. Turn it into a differential on the receiver's prior and keep that structural intervention separate from the registered recognition estimand. If nothing changes, the coupling is decorative. |
| **7** speech | Whisper → phoneme → TTS round trip | A round trip that "works" is a smoke test. Define the error metric on the recovered phoneme sequence and the comparator (what does the round trip score with the pymdp level removed?). |

---

## Cross-references

- [`testing_philosophy.md`](testing_philosophy.md) — controls, `GateVacuousError`, the oracle pattern
- [`design/architecture.md`](design/architecture.md) — the level hierarchy and each level's matrices
- [`design/stage2_g2p.md`](design/stage2_g2p.md) — why the G2P margin is thin by construction
- [`syntax_guide.md`](syntax_guide.md) — recording and referencing measured numbers
- [`../scripts/audit_gates.py`](../scripts/audit_gates.py) — the re-runnable classification
- [`../scripts/record_stage2_claims.py`](../scripts/record_stage2_claims.py) — the full-set gate statistic and the equivalence proof
- [`../AGENTS.md`](../AGENTS.md) — the stage gate table and both gate restatements
