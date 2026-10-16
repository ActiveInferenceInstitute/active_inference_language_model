# FAQ

Design questions, answered without flattering the project. For failures rather
than questions, see [`troubleshooting.md`](../troubleshooting.md).

---

## Why pymdp instead of a transformer?

Because the claim being tested is not "can this generate fluent text" — a
transformer answers that, and answers it far better. The claim is that a language
system can be built from named factors and explicit stochastic arrays, with no
learned embeddings or neural parameters.

That provides parameter-level explicitness and structural traceability, not a
blanket guarantee of mechanistic interpretability. The constraint is still hard:
real pymdp discrete state-space builders, explicit `A`/`B`/`C`/`D` arrays, and no
neural component inside AILM. The historical standalone L3-L6 scores use exact or NumPy paths
whose relationship to the live agents is stated level by level; an Agent existing
beside a scorer is not called live inference. The separate connected runtime
calls real pymdp inference and policy methods on its event schedule; see
[the connected contract](../design/connected_runtime.md).

The constraint pays for itself in a way that is easy to miss. When the letter
level scored claim:stage1_anti_informative_perplexity instead of
claim:stage1_letter_perplexity, the cause was locatable in one sentence: the
bigram matrix was in the emission `A` instead of the transition `B`. There is no
equivalent sentence for a transformer that underperforms. This local traceability
is a debugging instrument; it does not license a system-wide interpretability
claim.

## Is it competitive with a real language model?

The evidence does not establish competitiveness with a general-purpose language
model. The external probes are a reference frame with unequal training exposure,
fitting opportunity and interfaces, not a controlled leaderboard.

Concretely: the letter level reaches claim:stage1_letter_bits bits per character,
which is consistent with published order-1 English letter entropy and roughly
what a bigram model has always achieved. The letter-to-phone level
(position-aligned; `g2p` in code) beats a context-free
lookup table by claim:stage2_g2p_margin_pts percentage points, on a
subset of CMUdict chosen for tractability. L3-L6 have recorded gate outcomes;
L7 is an interface contract. Details are in
[`../docs/decisions.md`](../decisions.md).

What it *is* for:

1. **An explicit discrete-model testbed.** Each reported result has a named
   scorer and parameter source. That makes it a usable object for studying
   proposed composition seams, which
   is hard to do inside a system where the mechanism is distributed across
   billions of parameters.
2. **A testbed containing a source-bounded CEREBRUM case scaffold.** The source's
   structurally distinct case formulations are recorded; AILM implements only a
   project-owned NOM/ACC scalar approximation, which is tested but inert in the
   shipped case decoder. Neither
   CEREBRUM nor its categorical companion provides a generation mechanism — both
   describe comprehension. AILM implements a narrower L3→L1g generator: policies
   are next-letter commitments scored by expected free energy G(π), L3 supplies
   lexical `C`, and the Brown bigram supplies `E`. L6→L5 is a separate model-output
   prior intervention, while the typed word-prior arm uses a privileged
   punctuation-type oracle rather than L5 output. Neither is an earlier stage of
   that generator. See
   [`design/generation_problem.md`](../design/generation_problem.md).
3. **A worked example of verification discipline.** Arguably the most portable
   output. Three defects shipped here — an unreachable gate, a vacuous gate, and
   a biased evaluation subsample — and the apparatus built to catch them
   (the control harness, the implementation-independent reference decoder — the
   oracle — the claims ledger,
   `scripts/audit_gates.py`) transfers to any project that reports numbers. See
   [`gate_design.md`](../gate_design.md).

Anyone reading a perplexity or accuracy figure here as a competitive result has
been misled, and the prose is at fault. See
[`style_guide.md`](../method/style_guide.md#honesty-rules).

## Can I run the connected model on my own text?

Yes. Follow the [connected quickstart](quickstart.md#1b-run-the-connected-model).
Raw mode normalizes ASCII letter runs and uses punctuation only for sentence
boundaries. Structured mode can supply UPOS known before the token’s spelling;
it is a different observation condition, not interchangeable with raw evaluation.

The trace records predictions before observations and the following live-agent
posteriors. Its independent oracle checks each conditional update, not a global
Bayesian posterior over the entire hierarchy. Disabling all named channels keeps
the shared raw lexical UPOS encoder; it does not make every computation independent.
Out-of-vocabulary words remain in the denominator and make full word loss
infinite. Generation selects a bounded next word with real policy inference;
it is not evidence of fluent conversation or an acoustic speech system.

## Why only claim:stage2_g2p_length_matched_pct% of CMUdict?

Because the rest requires an alignment that would have to be invented.

CMUdict gives `WORD → phone sequence`, and the two are usually different lengths:
`CAT → K AE T` is 1:1, `THROUGH → TH R UW` is 7:3, `X → EH K S` is 1:3. To learn
`P(letter | phoneme)` you must know *which* letter emitted *which* phoneme. Equal lengths permit a deterministic position pairing, but do not establish
independently annotated linguistic alignment. Unequal lengths require an
additional alignment model, commonly fitted with EM.

A guessed alignment is fabricated training data. An EM alignment would be
defensible — with its own convergence diagnostics and a held-out likelihood
check — but that is a research contribution in its own right and cannot be
smuggled in as a preprocessing step for a level gate. Restricting to
length-matched pairs makes the position-pairing rule inspectable; it does not
validate those pairs as ground-truth grapheme–phoneme alignment. See the
[Stage 2 alignment boundary](../design/stage2_g2p.md).

**The cost, stated plainly.** The subset is claim:stage2_g2p_train_pairs train
plus claim:stage2_g2p_test_pairs test pairs, and it is **not a random sample of
English**. It is biased toward short words, transparent orthography (`cat`,
`stop`, `plant` rather than `through`, `knight`), and words without silent
letters, digraphs, or vowel clusters. Every accuracy figure at this level is a
claim about the transparent-orthography subset, which is an easier problem.
Comparison with published G2P systems would be invalid without matching the
subset.

Stage 3 uses a letter-bigram bar on identical positions instead; the same
length-matched alignment issue recurs for any EM-gated extension. Full reasoning in [`design/stage2_g2p.md`](../design/stage2_g2p.md).

## Why is the G2P margin so small?

Because on a transparent-orthography subset, the letter alone is *nearly
sufficient* — which is exactly why the context-free baseline is already strong at
claim:stage2_g2p_baseline_accuracy%. The headroom for phoneme context is small to
begin with.

Forward filtering captures part of it, reaching claim:stage2_g2p_accuracy% on
the full-set NumPy path after a bounded zero-disagreement check against the live
pymdp Agent. It cannot capture all of it, because `infer_states` is a **filtering** step: the
posterior at position *t* conditions on observations up to *t* and never revisits
them. Procedures that see the whole word do better —
claim:stage2_g2p_smoothing_ceiling% for forward–backward,
claim:stage2_g2p_viterbi_ceiling% for Viterbi.

A thin margin is not a weak result, but it *is* a result that a bare inequality
cannot establish. claim:stage2_g2p_margin_pts points is within the range a
different tie-break rule or smoothing constant could produce. So the gate is a
**paired McNemar test** over claim:stage2_g2p_eval_positions positions
(claim:stage2_g2p_mcnemar_b where only filtering is right,
claim:stage2_g2p_mcnemar_c where only the baseline is,
p = claim:stage2_g2p_mcnemar_p), with a no-effect control requiring the same
procedure to come back non-significant on two identical predictors.

The margin being small is why the statistics are careful, not a reason to shop
for a friendlier evaluation set. That shopping is exactly what produced the third
defect: see
[`gate_design.md`](../gate_design.md#defect-3--reproducible-is-not-representative).

## Why filtering rather than smoothing, if smoothing scores better?

Because smoothing describes a computation the architecture does not perform
online.

The shipped filtering paths condition on observations available at their event
clock. This does not imply that L3 consumes a streamed L2 posterior: the connected
runtime’s named lexical channel runs from word predictions to phone priors.
Its [schedule](../design/connected_runtime.md) names each direction and reset.
Reporting a smoothed number as that online path would describe a different system.

The smoothing and Viterbi figures are recorded as the achievable **ceiling**
precisely so the unclaimed headroom is visible rather than hidden. If a future
stage needs those points, the honest move is to state that it uses whole-word
smoothing and gate it as a separate, offline capability.

## Why is the gate against the unigram baseline rather than uniform?

Because uniform is a crippled comparator (see
[`gate_design.md#uniform-is-a-crippled-comparator`](../gate_design.md#uniform-is-a-crippled-comparator)):
a model can beat uniform while using no context at all. The uniform figures
(claim:stage1_uniform_baseline, claim:stage2_phoneme_uniform_baseline) are scale
markers, not bars.

The comparator has to be the strongest model that lacks the specific mechanism the
level claims to add. For a sequence level that is the context-free unigram model
(claim:stage1_unigram_baseline, claim:stage2_phoneme_unigram_baseline); for the
hierarchical G2P coupling it is the context-free per-letter lookup table. Clearing
those requires actually using context.

There is a second reason, learned the hard way. The original Stage 1 gate was
phrased as claim:stage1_declared_gate_nats nats over uniform, and that framing hid the fact
that the total gain available over uniform at order 1 is only
claim:stage1_max_bigram_gain_nats nats. Gating against a comparator whose distance
you have not measured is how you write an unreachable threshold.

## Why is the phoneme perplexity above the information floor?

It is above the historical transitions-only diagnostic because they measure
different position sets. The actual gate now uses a comparable all-position
bound.

- claim:stage2_phoneme_perplexity scores **every** position, with the prior reset
  to `D` at each word boundary. Position 0 has no predecessor, so it is paid for
  under the unigram prior. This is the honest language-model number.
- claim:stage2_phoneme_floor_perplexity is the conditional-entropy floor over
  **genuine transitions only**, word-initial positions excluded, computed from
  the unsmoothed MLE table.
- claim:stage2_phoneme_all_position_floor_perplexity combines the unsmoothed
  maximum-likelihood start distribution and transition columns over the same
  full-dictionary positions as the model. This is the gate's model-class bound.

A gate asserting that the model should equal the transitions-only diagnostic
can never pass. Add-1
smoothing shifts the floor again, in the third decimal. Not resetting at word
boundaries — treating the dictionary as one continuous phone stream — measures
*worse* than paying the unigram price, which is why the per-sequence reset is
correct rather than merely convenient. All three numbers and the reasoning are in
the [`../src/ailm/levels/phoneme.py`](../../src/ailm/levels/phoneme.py) module
docstring.

## Can I skip a stage?

Prospectively, no: Stage N+1 should not start until Stage N's gate has been
audited and passed. The historical record does contain exceptions. The original
Stage 1-4 gates were repaired after their levels had been built or measured, so
the current controls are retroactive re-certification rather than evidence that
the build proceeded sequentially.

The reason is empirical rather than procedural. Stage 1's wiring bug was
*inherited from a handoff spec* and was invisible to pymdp's validator. Had Stage
2 been built on top of it, the phoneme level would have been trained against a
letter level that predicted the previous letter — and the phoneme numbers would
have looked plausible. Each gate is the thing that stops a quiet defect from
becoming a foundation.

The canonical widened-observation Stage 5 arm passes its development-session
gate and has no held-back replication; the retained case-only ablation remains
non-passing. Stage 6's document-clustered primary interval spans zero on its
development population and excludes zero on the separately registered held-back
population at the declared, but not family-corrected, coverage. Stage 7 remains a
speech boundary contract. The [governing hierarchy](../hierarchy.md) identifies
each scorer and the separate connected runtime; these outcomes are not evidence
of fluent or speech-ready language generation.

Audit before building additional levels; the checklist is in
[`gate_design.md`](../gate_design.md#pre-build-checklist).

## Why is the case oracle "agreement" and not "accuracy"?

Because the reference is itself a research claim. English is morphologically
impoverished: of the claim:stage0_case_roles linguistic case roles, only
NOM/ACC/GEN carry morphological marking in UD English EWT. DAT/INS/LOC/ABL are
derived from `deprel` (`iobj`→DAT, `obl:with`→INS, `obl`→LOC, `obl:from`→ABL, and
so on) — a principled heuristic, not ground truth.

Matching a heuristic is *agreement with that heuristic*. Calling it accuracy
implies a gold standard that does not exist, and Stage 4 prose must not. The
mapping is documented in [`provenance/sources.md`](../provenance/sources.md).

## Where does every number come from?

[`../data/claims.json`](../../data/claims.json), recorded by the
`scripts/record_stage*_claims.py` runs, each entry carrying the value, the
`run_id`, and the `source` that produced it. Nothing quantitative is hand-typed:
the manuscript resolves `{{TOKEN}}`s from the ledger and fails the render on a
missing claim, and `docs/`, `README.md`, `AGENTS.md`, `CHANGELOG.md` and `src/`
docstrings carry `claim:<label>` tags that a test verifies against the same file.

To audit:

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root
.venv/bin/python -m pytest tests/test_claims_discipline.py -q
.venv/bin/python -m json.tool data/claims.json
```

Mechanics in [`syntax_guide.md`](../syntax_guide.md).

## Is the claims ledger not overkill for a research prototype?

It was written *after* the failure it prevents, which is the usual reason such
things exist.

`ClaimsLedger.scan_prose` shipped in Stage 0 with a full set of passing tests —
all of which ran against `tmp_path` fixtures. Meanwhile
claim:phoneme_count_39 sat **unbacked in the real README, AGENTS.md, and the
provenance doc** for an entire stage: the protocol was documented, tested in
miniature, and never once applied to the files a reader would take as factual.

A gate that runs only on synthetic input constrains nothing. The ledger is cheap;
the class of error it catches is one where every test stays green.

## Why does `src/ailm/verification/oracle.py` refuse to import pymdp?

Because its entire value is independence. If the auditor shared code with the
thing it audits, it would inherit the flaw it audits for — and the flaw it exists
to catch is precisely one that pymdp's own validator cannot see, since a bigram
matrix is column-stochastic in either the emission or the transition slot.

The oracle is pure numpy, tested against closed forms rather than against pymdp: a
uniform transition must give exactly the inventory size, a deterministic chain
exactly 1. Anchoring it on pymdp would let both be self-consistently wrong. Its
throwaway ancestor is what caught the Stage 1 bug; it is now a tested module so
every level can be checked the same way.

## What does `{{META_CLAIM_COUNT}}` mean if the ledger keeps growing?

That the count is generated, not asserted. It is a build-provenance token read
from the ledger at render time, alongside the artifact checksums
(`{{META_BROWN_COUNTS_SHA256_SHORT}}`, `{{META_CMUDICT_COUNTS_SHA256_SHORT}}`),
so a reader can tell which ledger and which corpus bytes a given rendering
describes. Nothing in prose should ever state a claim count as a literal — the
number changes with every stage.

---

## See also

- [`gate_design.md`](../gate_design.md) — the taxonomy and the three defects
- [`testing_philosophy.md`](../testing_philosophy.md) — controls and oracle cross-checks
- [`design/stage2_g2p.md`](../design/stage2_g2p.md) — the alignment decision in full
- [`design/architecture.md`](../design/architecture.md) — the level hierarchy
- [`../README.md`](../../README.md) — scope statement and stage roadmap
