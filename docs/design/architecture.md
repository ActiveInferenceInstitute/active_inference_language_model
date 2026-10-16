# AILM Architecture — implemented models and design rationale

> **Scope.** This file records the design rationale and the corrected shipped
> structure. The compact, governing model/scorer table is
> [`docs/hierarchy.md`](../hierarchy.md); it wins on any disagreement about the
> shipped level structure. Historical plans are
> labeled as such; `ISA.md` is not a current-status source.

The standalone stage scorers and the [connected runtime](connected_runtime.md)
use the same declared component models with different execution contracts.
Historical intervention results below remain evidence about their original
seams; the connected raw-text comparison is reported separately under
[decision 132](../decisions.md#132--implemented-a-conditional-connected-runtime-and-a-locally-registered-gum-comparison-2026-10-05).

## Hierarchy

AILM specifies six implemented levels of pymdp agents. Each level is a proper
discrete state-space active inference agent with its own A, B, C, D matrices;
Stage 7 is an interface-only speech boundary rather than another generative level.

```
Level 6 (L6): Narrative
Level 5 (L5): Clause
Level 4 (L4): Case structure
Level 3 (L3): Word identity × within-word position
Level 2 (L2): Phoneme    — L2A phone sequence, L2B grapheme-to-phoneme
Level 1 (L1): Letter
```

The level list alone declares no execution schedule. In standalone decoding,
L2B and L1 share an alphabet (`A = P(letter | phone)`) without exchanging beliefs.
The connected runtime explicitly adds timed prior and observation messages;
[its contract](connected_runtime.md#connections-and-numerical-meaning) owns those
insertion sites. Historical receiver interventions have their own sender,
receiver and recorded outcome and must not be substituted for that schedule.

The reported receiver interventions are measured on their real source → receiver
pairs. Four sources are model outputs; the typed word-prior arm is a privileged
reference:

| channel | direction | measured outcome | entry |
|---|---|---|---|
| L1 → L2B coupling | bottom-up | **harms** significantly — the receiver already observes the letter | 9 |
| L3 → L2B lexical | top-down | **helps** at the selected predictive insertion site and interior weight | 19 |
| L3 → L1g generation | top-down | **works** — the lexicon supplies the generator's `C` | 17 |
| L6 → L5 prior intervention | top-down | moves the receiver's joint `D`; evaluated independently of the other seams | 25–26 |
| punctuation-type oracle → L3 word prior | privileged reference intervention | **helps after correction** on aggregate perplexity and token direction — the predicted harm did not reproduce; no L5 posterior is used | 27, 88 |

Entry 18's novel-information criterion is an evidence-bounded working
hypothesis, not an architectural law. Entry 27's outcome qualified its first
clause per the stated withdrawal condition. The L3→L1g and L6→L5 model-output
seams and the oracle→L3 intervention were not executed as one cascade.

## Why Levels, Not One Big Agent

A hypothetical Cartesian product of every linguistic inventory would multiply
unrelated state spaces and require a likelihood over the whole joint. AILM does
not build that object. Each level owns only its stated factors, and L2B connects
letters and phones inside one emission rather than through a global joint agent.
Exact current memory and operation counts are generated in
[`../../manuscript/98_supplement_complexity.md`](../manuscript/98_supplement_complexity.md),
not duplicated here.

## A_dependencies Factorization

Within each level, `A_dependencies` further factorizes the likelihood:

At L3, the letter modality depends on both word identity and within-word
position, declared as `A_dependencies=[[0, 1]]`. The two transition factors stay
separate: word identity persists and position advances. This is factorization of
one level's model, not evidence of a connected cross-level hierarchy.

This is verified in `tests/test_pymdp_smoke.py::TestADependenciesShape`.

## Level 1: Letters

**State space:** 30 discrete letter states (a-z + space + punctuation)
**A matrix:** 30×30 **identity** emission — observing a letter reveals the letter state
**B matrix:** 30×30 bigram transition, `B[0,i,j,0] = P(next=i | current=j)`, learned from Brown
**C vector:** zeros in the comprehension agent
**D vector:** uniform over letters

The controllable L1g generation agent is separate: L3 supplies its lexical `C`
and the Brown bigram supplies its policy habit `E`. L2 does not set L1's `C`.

The bigram statistics live in the **transition**, not the emission. The inverse
wiring (bigram in `A`, identity `B`) is anti-informative rather than loudly
broken — it yields a posterior over the *preceding* letter and carries that
backward-looking belief forward as the next-letter prediction. Measured, it
scores claim:stage1_anti_informative_perplexity on held-out prose and *worse
than uniform* on a small in-sample corpus, while still slipping under a naive
`< 28.0` threshold. `tests/test_stage1_gate.py` pins the orientation and keeps a
live negative control against it.

### Stage 1 gate (as implemented)

The gate is **beating the context-free unigram baseline**, not beating uniform.
Uniform (30.0) is a crippled comparator: a model can clear it while carrying no
context at all.

| Comparator | Perplexity | Nats/letter |
|---|---|---|
| Uniform (30 symbols) | claim:stage1_uniform_baseline | 3.4012 |
| Unigram letter frequencies (context-free) | claim:stage1_unigram_baseline | — |
| **Bigram agent (this level)** | **claim:stage1_letter_perplexity** | **claim:stage1_letter_nats** |
| Order-1 information floor `H(X_t\|X_{t-1})` | claim:stage1_bigram_floor_perplexity | claim:stage1_bigram_floor_nats |

The agent captures claim:stage1_bigram_signal_captured_pct% of the information
available at this order, landing claim:stage1_letter_bits bits/char — consistent
with published order-1 English letter entropy.

> **Correction to the original gate spec.** Earlier drafts of this document and
> the Stage 1 handoff demanded a gain of claim:stage1_declared_gate_nats nats
> above uniform. That target is
> **information-theoretically unreachable for any letter bigram model**: the
> conditional entropy of the Brown bigram distribution itself caps the
> achievable gain at claim:stage1_max_bigram_gain_nats nats. The measured gain
> is claim:stage1_gain_over_uniform_nats nats — i.e. near-optimal for this
> order, not a shortfall. A claim:stage1_declared_gate_nats nat gain would require a higher-order model
> (trigram or beyond). The gate was restated as the unigram-baseline comparison,
> which is both achievable and actually diagnostic of bigram learning.

## Level 2: Phonemes

Two models ship at this level, and only the second is described below. Part A is
the phoneme *sequence* model — the Level 1 construction one inventory down, with an
identity `A` and a CMUdict phone bigram in `B`. Part B is the grapheme-to-phoneme
model:

**State space:** claim:phoneme_count_39 ARPAbet phonemes (from the CMUdict inventory, stress stripped)
**A matrix:** `P(letter | phoneme)` — the emission, learned from 1:1 aligned CMUdict pairs
**B matrix:** `P(next phoneme | current phoneme)` — phoneme bigram transition
**C vector:** zeros over letter observations
**D vector:** learned first-phone distribution

The separate L2A phone-sequence agent has identity `A`, the phone bigram in `B`,
zero `C`, and the learned phone unigram in `D`. Its canonical full-dictionary
perplexity is computed exactly from initial-phone and bigram sufficient
statistics; a bounded live-pymdp walk is required to agree with the independent
NumPy chain before that full score is trusted.

As at Level 1, the learned sequence dynamics live in the **transition `B`**, and
`A` is the emission. L2B is the first model where `A` is *not* the identity:
letters are the observations, phonemes the hidden states, so `A` is the genuine
grapheme–phoneme likelihood and the level performs real inference rather than
direct state read-off.

### Stage 2 gate (as implemented)

Two parts. Both gate against a **context-free** comparator, never uniform.

**(A) Phoneme sequence perplexity**

| Comparator | Perplexity |
|---|---|
| Uniform over 39 phones | claim:stage2_phoneme_uniform_baseline |
| Context-free unigram phone frequencies | claim:stage2_phoneme_unigram_baseline |
| **Phoneme bigram agent** | **claim:stage2_phoneme_perplexity** |
| Exact all-position order-1 MLE floor | claim:stage2_phoneme_all_position_floor_perplexity |
| Transitions-only diagnostic `H(P_t\|P_{t-1})` | claim:stage2_phoneme_floor_perplexity |

**(B) Grapheme-to-phoneme accuracy** — the first genuinely hierarchical claim,
letters observed and phonemes inferred:

| Method | Position accuracy |
|---|---|
| Context-free per-letter `argmax P(phone\|letter)` | claim:stage2_g2p_baseline_accuracy |
| **Full-set NumPy forward filter after a bounded zero-disagreement check against the pymdp Agent** | **claim:stage2_g2p_accuracy** |

The canonical Part B statistic is evaluated by the NumPy filter over the full
held-out set. The recorder first requires the live pymdp decoder to make exactly
the same position-level decisions on its bounded verification subset; that
agreement licenses attribution to the same fitted model without pretending the
full-set loop called pymdp.

The letter-to-phone (position-aligned; `g2p` in code) margin is small by construction (see
[`stage2_g2p.md`](stage2_g2p.md)), so it is defended with a **paired McNemar
test** (p = claim:stage2_g2p_mcnemar_p over
claim:stage2_g2p_eval_positions positions) rather than by a bare inequality.

> **Correction to the original gate spec.** Earlier drafts specified letter-to-phoneme
> accuracy at the copied Stage 4 threshold of claim:stage4_declared_gate% on the
> UD EWT oracle. That was wrong twice over. First, UD EWT is
> a *syntactic* treebank — its CoNLL-U fields are `lemma`/`upos`/`feats`/`deprel`,
> with **no phonemic annotation of any kind**, so it cannot score a phoneme task;
> the line was a copy of Stage 4's case gate. Second, that threshold is
> **vacuous**: a context-free per-letter lookup table — no transition matrix, no
> pymdp, no active inference — scores claim:stage2_g2p_baseline_accuracy on
> held-out CMUdict, comfortably clearing it. A model could pass the stated gate
> while doing nothing hierarchical at all. The gate is therefore stated against
> that lookup table, which is the weakest comparator that still requires using
> phoneme context to beat.
>
> Note the failure mode is the *opposite* of Stage 1's: Stage 1's target was
> unreachable, Stage 2's was too easy. Both were caught the same way — by
> measuring the ceiling and the trivial baseline before accepting the threshold.

## Component models and historical stage scores

### L3 word

- Hidden factors: word identity × within-word position.
- Observation: lowercase letter or END.
- `A`: deterministic spelling likelihood depending on both factors.
- `B`: identity over word identity and a forward shift over position.
- `C`: zero.
- `D`: learned word unigram and an opening-position point mass.
- Score: exact path probability. A sampled live-pymdp walk pins the identity, but
  the full headline score does not call pymdp inference.

### L4 syntactic case role

- Hidden factor: one of the UD-derived syntactic case roles.
- Observation: UPOS only; the model is structurally forbidden from reading the
  `deprel` and morphology fields used to construct evaluation labels.
- `A`: learned `P(UPOS | case)`; `B`: learned case-role bigram; `C`: zero;
  `D`: learned first-case distribution.
- Score: NumPy forward filtering. The recorded comparison requires live pymdp
  inference to reproduce the position-level decoder at every held-out
  position, argmax and posterior alike (decision 94).
- The source-faithful CEREBRUM formulation metadata and AILM-owned NOM/ACC
  scalar approximation remain tested but inert; neither is the case decoder.

### L5 clause

- Canonical hidden factor: joint sentence type × widened symbol; observations
  are case roles plus `VERB` and `AUX` in surface order.
- `A`: deterministic symbol-component read-out; `B`: type-persistent blocks of
  learned widened-symbol chains; `C`: zero;
  `D`: type prior × first-symbol-given-type.
- The case-only model remains a separately identified ablation. It uses the
  same joint-state pattern with case roles as its smaller symbol alphabet.
- Score: closed-form mixture evidence. The live Agent stores the mapped arrays,
  and its recursion is checked against the NumPy joint filter on the full
  held-out set at the posterior level (argmax and joint-posterior tolerance,
  decision 94); that is posterior agreement, not score equivalence for the
  headline mixture estimand.
- Canonical outcome: the widened gate passes as a development-session
  specification, with no held-back replication. Its paired privileged-label gain
  resolves away from zero. The retained case-only ablation remains non-passing
  with unresolved gain, including its held-back replication. [Decision 114](../decisions.md#114--the-widened-observation-clause-model-is-the-canonical-level-5-the-case-only-model-is-the-retained-ablation-2026-09-06)
  records the promotion and preserves both limits.
- The type-label arm is a privileged-information sensitivity reference, not a
  hard information bound or an equivalence result.

### L6 document position

- Hidden factor: document-position tercile.
- Observation: sentence type.
- `A`: learned `P(type | tercile)`; `B`: learned left-to-right tercile
  transition; `C`: zero; `D`: opening-tercile point mass.
- Registered score: observed-position direct conditional log loss; it selects the
  known tercile column and does not call live inference. A separate descriptive
  sequential predictive score resets `D` per document and runs
  `infer_states`/`update_empirical_prior` through the real Agent. The two are
  different estimands, not an equivalence claim.
- Development outcome: positive point estimate with a document-clustered
  primary interval spanning zero. The later held-back document-level interval
  excludes zero at its declared coverage but spans zero at family-corrected
  coverage; that population had already been used, so the result is descriptive.
  The sentence-level sign test is a different estimand. The full qualifications
  are in the [governing hierarchy](../hierarchy.md).

L7 is the ARPAbet serialization/parsing boundary described in
[`speech_boundary.md`](speech_boundary.md), not another generative level.

## Cross-level direction

L2B uses the letter alphabet directly as its observation space; L1 and L2B do
not exchange beliefs in the default decoder. The measured L1→L2B message harms,
while the selected L3→L2B predictive lexical message helps. The top-down L6→L5
and L3→L1g model-output interventions and the privileged oracle→L3 arm are
independently scored experiments. The newer connected runtime executes its own
explicit messages and has a separate evaluation; it does not inherit those
intervention outcomes. `update_empirical_prior` is always within one agent when
it is called.

## Scale accounting

The builders and generated complexity supplement own the current shapes, byte
counts, and operation counts. This design file intentionally does not duplicate
those changing values. The stable result is structural: L3 uses a factored
word×position likelihood, and no implementation constructs a global joint state
over all linguistic levels.
