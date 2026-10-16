# How the GNN-specified levels compose, and how they do inference

The canonical [GeneralizedNotationNotation](../gnn/README.md) specs under
`gnn/models/` cover L1–L6, including separate phone-sequence and letter-to-phone
models, the canonical widened-observation clause model and its retained case-only
ablation, plus the L1g letter-generation variant. Their inventory is bound to live
`pymdp.agent.Agent` objects by `SPEC_NAMES` in
[`../tests/test_gnn_conformance.py`](../tests/test_gnn_conformance.py). Its
default-deny directory check requires the covered set to equal the canonical
specs on disk, and [`../tests/test_spec_agent.py`](../tests/test_spec_agent.py)
separately requires every spec-built Agent to be bit-identical to its
builder-built counterpart.

The document explains what those specs say, how a spec becomes a running agent,
what one step of inference actually computes, and — the part that is easiest to
get wrong — in exactly what sense the levels *compose*.

## Read this first: recorded interventions and connected execution

The original stage scorers and recorded intervention experiments retain their
own populations and estimands. The connected development runtime additionally
executes a causal schedule of messages through GNN-built agents. Its observation
access, clocks, resets, train-only kernels and conditional reference are specified
in the [connected runtime contract](design/connected_runtime.md), with the
[governing hierarchy](hierarchy.md#connected-development-runtime) stating the
scientific boundary. This is local conditional inference with explicit timing,
not a single global joint posterior. The [locally frozen GUM comparison](manuscript/04_results.md#sec:connected_gum)
provides separate population-specific character-loss evidence; it does not
establish general performance, finite full-stream word loss or generation quality.

The historical measured intervention inventory is owned by
`ailm.channel_registry`: L1 → L2B coupling harms, L3 → L2B lexical messaging
helps, L3 → L1g generation works, L6 → L5 moves a joint prior, and the privileged
punctuation-type-oracle → L3 intervention helps on its recorded directional
summaries. The case-word bridge additionally has exploratory prior-only sweep
evidence; its privileged-oracle diagnostic remains structural. These measurements
were independent experiments. They do not measure the new connected runtime,
and the privileged oracle arm is not an L5 posterior channel.

`update_empirical_prior` is not one of them. It is used **only** for within-level
temporal recursion: it takes a level's own posterior at position `t` and produces
that same level's prior for position `t+1`. It never crosses a level boundary.

A real Level 1 to Level 2 soft-observation channel *was* implemented
([`../src/ailm/levels/coupling.py`](../src/ailm/levels/coupling.py)) and measured
**harmful** — a margin of claim:coupling_margin_pts points, McNemar
p = claim:coupling_mcnemar_p, recorded support flag
claim:coupling_supported. It ships gated off. The cause is structural rather than
a wiring defect: at position `t` the position-aligned letter-to-phone (G2P)
level *already observes* letter `t`,
while the L1 predictive is a function of letters `0..t-1` only, so mixing the L1
message in substitutes weak evidence for a certainty. The full measurement and
its controls are [`decisions.md` entry 9](decisions.md#9--inter-level-soft-observation-coupling-is-a-recorded-negative-result).

What genuinely composes is set out in [§4](#4-how-the-levels-compose). Shared
alphabets and shared spec grammar are static relationships; the historical
interventions are separately measured experiments. The connected runtime adds an
explicit execution schedule rather than deriving one from these declarations.

---

## 1. What a GNN spec is

A GNN spec is an Active Inference generative model written as structured
Markdown: a fixed sequence of `##` sections declaring the state space, the
connections between variables, the parameterization, the time structure, and an
ontology block naming the semantic *role* of every matrix. AILM accepts exactly
the canonical thirteen sections, in canonical order; anything else is a parse
error rather than an extension, because a misspelled `## ModelParamaters` would
otherwise silently take the dimensions with it. The enforced subset and its
limits are [`../gnn/SPEC.md`](../gnn/SPEC.md).

Five sections carry load. The excerpts below are the real ones from
[`../gnn/models/L2b_grapheme_to_phoneme.gnn.md`](../gnn/models/L2b_grapheme_to_phoneme.gnn.md), the L2B grapheme-to-phoneme level.

### `StateSpaceBlock` — every variable and its dimensions

```
A[30,39,type=float]      # Recognition matrix, P(letter | phone) — 30 obs by 39 states
B[39,39,1,type=float]    # Transition, P(next phone | current phone), 1 control
C[30,1,type=float]       # Log-preferences over observed LETTERS, flat here
D[39,1,type=float]       # Prior over the first phone of a word
```

Dimensions must be positive integer literals. Symbolic dimensions are rejected:
upstream defaults an unparseable dimension to `1`, and a checker cannot bind that
to a real matrix.

The asymmetry here is the whole content of the level. `A` is 30 by 39 — the
observed alphabet (letters) and the hidden alphabet (claim:phoneme_count_39
ARPAbet phones) are *different alphabets of different sizes*, so unlike the other
two levels the emission cannot be confused with a transition by shape alone.

### `Connections` — which variables touch which

```
D>s_t
s_t-A
A-o_t
s_t-B
B>s_prime
C-o_t
```

`>` is directed, `-` undirected. Both endpoints must be declared in
`StateSpaceBlock`. Conformance checks a nonempty graph for orphaned variables,
prior direction, policy-to-transition consistency and emission state-axis
neighbors. The ontology still pins each matrix into its semantic slot. These
checks do not execute messages or establish a joint posterior.

### `InitialParameterization` — structural form, or provenance

```
A=Learned(generator=scripts/generate_phoneme_counts.py, artifact=vendor/cmudict_counts.json, key=g2p_pairs, split=train, alignment=length-matched-1to1, estimator=add-1-Laplace, normalization=column-stochastic-over-letters)

C=Zeros(30)
```

Only four forms are accepted: `Identity(n)`, `Uniform(n)`, `Zeros(n)` and
`Learned(k=v, ...)`. Learned entries are **not inlined**. A 30-by-39 probability
table is a derived artifact reproduced byte-identically by its generator
([`../scripts/generate_phoneme_counts.py`](../scripts/generate_phoneme_counts.py)
from `vendor/cmudict_counts.json`); a copy inside the spec would be a second
source of truth, and the copy is the one that drifts.

The structural forms are checked against live arrays, including finite values.
Declared normalization and unqualified Laplace positivity also constrain learned
arrays. These checks reject malformed distributions and same-shape structural
swaps; they do not independently reconstruct fitted probabilities.

### `ActInfOntologyAnnotation` — the role of every slot

```
A=RecognitionMatrix
B=TransitionMatrix
C=LogPreferenceVector
D=Prior
s_t=HiddenState
s_prime=NextHiddenState
o_t=Observation
t=Time
```

This is the guard, and the reason AILM carries specs at all. The annotation must
be in bijection with the declared variables: nothing declared may be
un-annotated, nothing annotated may be undeclared. Exactly one variable may carry
each of `RecognitionMatrix`, `TransitionMatrix` and `Prior` — two variables
claiming a role pin nothing, so ambiguity is itself a violation.

Declaring `B=TransitionMatrix` is what turns "the learned bigram is in the wrong
slot" from a slightly different number into a named violation. Both slots accept
a column-stochastic matrix, so pymdp's own validation is indifferent between
them; see [`decisions.md` entry 1](decisions.md#1--learned-sequence-dynamics-go-in-b-the-emission-is-a).

### `ModelParameters` — the sizes both sides must agree on

```
num_hidden_states: 39
num_obs: 30
num_factors: 1
num_modalities: 1
```

These are checked twice: for internal agreement with the spec's own
`StateSpaceBlock` (the `RecognitionMatrix` must be declared `num_obs` by
`num_hidden_states`, the `TransitionMatrix` `num_hidden_states` square, the
`Prior` a `num_hidden_states` vector), and against the live agent.

---

## 2. How a spec becomes a running agent

AILM does not use upstream GNN's multi-backend generation pipeline. It keeps two
local paths with different jobs. The hand-written level builder constructs the
Agent used by the recorded measurements, while
`ailm.spec_agent.build_agent_from_spec` constructs an Agent from the declaration
through a default-deny registry of permitted `Learned(...)` provenances. Both
paths call the same low-level matrix builders, and
[`../tests/test_spec_agent.py`](../tests/test_spec_agent.py) requires their slot
arrays to match in shape, dtype and bytes before the spec-built path can stand in
for the historical builder.

The independent guard remains [`../src/ailm/gnn/`](../src/ailm/gnn): it is
stdlib plus NumPy and never imports the construction adapter, pymdp, JAX, or the
upstream pipeline. It parses the declaration and checks the already-constructed
live object. Keeping construction in `ailm.spec_agent` and conformance in
`ailm.gnn` prevents the checker from inheriting the assembly logic it audits.

The path, for L2B:

| Step | Function | Produces |
|---|---|---|
| 1 | `ailm.gnn.load_spec("L2b_grapheme_to_phoneme")` → `parse_gnn_spec` | a `GnnSpec` |
| 2 | `ailm.levels.phoneme.build_g2p_emission` | `A`, shape `(1, 30, 39)` |
| 3 | `ailm.levels.phoneme.build_g2p_transition` | `B`, shape `(1, 39, 39, 1)` |
| 4 | `ailm.levels.phoneme.build_g2p_start` | `D`, shape `(1, 39)` |
| 5 | `ailm.levels.phoneme.build_g2p_agent` | a `pymdp.agent.Agent`, with a flat `C` of 30 zeros |
| 6 | `ailm.gnn.describe_shapes(agent)` | shapes **and arrays** read off the live object |
| 7 | `ailm.gnn.check_conformance(spec, **described)` | every violation, as a list |
| 8 | `ailm.gnn.assert_conforms(spec, **described)` | raises `GnnConformanceError` with the complete list |

The same describe-and-check path covers the remaining registered specs, including L3's
two hidden factors, L5's canonical joint type × widened-symbol state and retained
case-only ablation, L6's document-position state,
and L1g's action-indexed transition and policy prior. The checked level Agent is
always built from the real vendored artifact where it has learned parameters.

Read the slot off the agent builder, not off a function name:
`build_letter_transition` returns the learned Brown bigram matrix, and
`build_letter_agent` installs it in **`B`** while supplying an identity `A`.

Step 6 is what makes this a guard rather than a comment. `describe_shapes` reads
the public `A`/`B`/`C`/`D` lists off the object pymdp actually constructed, and
returns the arrays as well as their shapes — the structural-form checks
(`Identity(n)`, `Uniform(n)`, `Zeros(n)`) cannot be decided from a shape, and at
the letter level the correct and the historically swapped wirings have identical
shapes.

[`../tests/test_gnn_conformance.py`](../tests/test_gnn_conformance.py) runs this
for every registered spec against agents built for real from the vendored artifacts.
Its deliberate negative controls include an A/B-swapped letter agent, an inverted
39-by-30 G2P emission, contradictory declarations, mutated dimensions and the
per-level mutation specs; each must be **rejected**, with the test asserting on
the reported violation rather than accepting an unrelated exception.

[`../tests/test_spec_agent.py`](../tests/test_spec_agent.py) then exercises the
other direction for every registered spec: build from the spec, compare byte-for-byte with
the canonical builder, and refuse an unregistered generator, an altered
provenance argument, or a registered provenance attached to the wrong ontology
role. Conformance establishes declared structure; bit identity establishes that
the two construction paths currently denote the same fitted model. Neither test
by itself validates a reported perplexity or accuracy.

`fig:gnn_pipeline` ([`../src/ailm/figures/composition.py`](../src/ailm/figures/composition.py))
draws the builder-and-conformance path, with the closing arrow drawn back to the
spec; the newer spec-construction path is documented above rather than added to
that panel. Its declared
dimensions are read out of `gnn/models/L2b_grapheme_to_phoneme.gnn.md` at generation time, so the panel
cannot show a declaration the file does not carry.

> One rule governs step 7 failing: **never edit a spec to make a conformance test
> pass.** Either the code changed intentionally, in which case the spec is
> updated deliberately, or the code changed by accident, in which case the spec
> just caught it. [`../gnn/AGENTS.md`](../gnn/AGENTS.md) is the contract.

---

## 3. Inference within a level, one symbol at a time

Every comprehension Agent admits the same four-part filtering recursion, but
not every recorded headline score executes it through pymdp. Below is one live
L2B step. L1's headline uses the live loop; L2A's full-dictionary score instead
collapses the same chain to exact sufficient statistics; L2B's full held-out
score uses an independent NumPy forward filter after a bounded zero-disagreement
check against the live Agent. L3 and L5 use exact scorers (L5's live recursion
is separately checked against the NumPy joint filter on the full held-out set
at the posterior level), L4 uses a NumPy forward filter that the live Agent
reproduces at every held-out position, and L6's registered
observed-position score indexes its fitted `A` table directly. L6 additionally
reports a separate descriptive sequential score through its live Agent. These
paths evaluate the declared models, but only the paths explicitly named live
call pymdp inference for their reported score.

Let `π_t` be the empirical prior — the predictive belief over the hidden state at
position `t`, given observations `1..t-1`.

**Step 0 — the prior at the start of a word.** `π_1 = D`. At L2B, `D` is the
distribution over the first phone of a training word; at L2A it is the phone
unigram; at L1 it is uniform. L2A and L2B reset to `D` at the start of **every**
word, because their corpus is a word list and there is no bigram across the
boundary. L1 does not reset: its corpus is running text, where the transition
across a space is a real letter bigram (space is one of the thirty symbols). See
[`decisions.md` entry 7](decisions.md#7--level-1-does-not-reset-the-prior-per-word-l2a-and-l2b-do).

**Step 1 — the predictive read-off. `A` enters.**

```
P(o_t | π_t) = Σ_j A[0, o_t, j] · π_t[j]
```

This is the one-step-ahead probability the model assigns to the symbol it is
about to see. It is read from the prior *before* the update, so the quantity
scored is a genuine prediction rather than a posterior fit — the read-off taps
the prior without consuming it.

At L1 this is where the live headline perplexity accumulates:
`exp(−mean_t log P(o_t | π_t))`, implemented by
`ailm.levels.letter.compute_perplexity`. The bounded L2A live walk uses the same
read-off to check the independent NumPy chain; the canonical full-dictionary
number is the exact initial-phone and bigram sufficient-statistic score. Under
the identity emission both reduce to `π_t[o_t]`.

L2B is scored differently, because its observation and its state are different
alphabets and a per-letter perplexity would not be the quantity of interest.
Its metric is position accuracy of the decoded phone against the reference
pronunciation. The full-set decoder is the NumPy forward filter; a bounded live
Agent walk must make zero different position decisions before that result is
attributed to the same declared model. The comparator is the context-free
per-letter argmax bar on the identical position set.

**Step 2 — the posterior. `A` enters again.**

```
q(s_t) ∝ A[0, o_t, :] ⊙ π_t          agent.infer_states([o_t], π_t)
```

At L1 and L2A, `A` is the identity and this collapses onto the observed symbol.
At L2B it does real work: the observation is a letter, the state is a phone, and
`A[0, letter, phone] = P(letter | phone)` reweights the predictive over phones by
how well each phone explains the glyph. The decoded phone is `argmax_p q(s_t)[p]`
(`ailm.levels.phoneme.decode_phonemes`).

**Step 3 — the next prior. `B` enters.**

```
π_{t+1} = B[0, :, :, a] · q(s_t)      agent.update_empirical_prior(a, q)
```

`a` is the single trivial action in a comprehension Agent. Within this live
filtering loop, this is the place `B` is consulted and
`update_empirical_prior` is called. It advances the same level by one position.
It does not communicate with any other level.

**Which matrix enters where — and why that is the invariant**

| Step | Call | Matrix consulted |
|---|---|---|
| predictive read-off | (arithmetic on `π_t`) | `A` |
| posterior | `agent.infer_states` | `A` |
| next prior | `agent.update_empirical_prior` | `B` |

Corpus-learned sequence statistics belong in `B`, because `B` is the slot that
advances time. Installing the learned bigram as `A` with an identity `B` is
column-stochastic in both slots and passes pymdp's `_validate()`, but observing
symbol `c` then yields a posterior over the symbol that *preceded* `c`, and the
identity `B` carries that backward-looking belief forward as the next-symbol
prediction. The model becomes anti-informative rather than weak: at the letter
level it scores claim:stage1_anti_informative_perplexity against
claim:stage1_letter_perplexity for the shipped wiring, with the context-free
unigram baseline at claim:stage1_unigram_baseline and uniform at
claim:stage1_uniform_baseline.

Decoding is **filtering, not smoothing**: position `t` conditions on symbols
`1..t` only, never on what follows. Smoothing and Viterbi use the whole word and
are reported as privileged headroom references, not as what this model does and
not as hard bounds on a same-information model
([`decisions.md` entry 8](decisions.md#8--filtering-not-smoothing)).

---

## 4. How the levels compose

Three senses are kept separate: shared alphabets, shared specification
machinery, and measured interventions. The first two are static composition.
The third describes the original independent experiments. The newer connected
runtime has its own explicit schedule and evidence boundary above.

### 4.1 Representational composition — a shared alphabet

**L2B's observation alphabet *is* L1's state alphabet.**

| | L1 letter | L2A phoneme | L2B grapheme-to-phoneme |
|---|---|---|---|
| hidden factor | letter, 30 symbols | phone, claim:phoneme_count_39 ARPAbet | phone, claim:phoneme_count_39 ARPAbet |
| observation modality | letter, 30 symbols | phone, claim:phoneme_count_39 ARPAbet | **letter, 30 symbols** |
| `A` | identity | identity | learned, 30 by 39 |
| `B` | Brown letter bigram | CMUdict phone bigram | phone bigram, TRAIN split only |

`A[0, letter, phone] = P(letter | phone)` maps the 39-phone state space onto the
30-letter observation space. That 30-symbol observation space is exactly L1's
hidden state space, and L2B's 39-phone state space is exactly L2A's. The levels
meet in shared alphabets.

This is a **type identity fixed at build time, not a value computed at run
time**. No belief, distribution or message travels along it. Two agents happening
to index the same symbol set is a shared vocabulary, in the way that two
functions can share an argument type without ever calling each other.

`fig:composition` draws the shared alphabet as a plain connector with an
identity glyph and deliberately no arrowhead: an arrowhead is the visual idiom
for a value flowing, and nothing flows.

### 4.2 Structural composition — one grammar, one parser, one checker

All registered specs share the same section grammar and core ontology vocabulary
(`RecognitionMatrix` / `TransitionMatrix` / `Prior`). Single-factor models use
`A`, `B`, `C`, `D`, `s_t`, `s_prime`, `o_t` and `t`; L3 uses suffixed `B_f<N>`
and `D_f<N>` declarations for its two factors, and L1g additionally declares
the policy prior `E`. One parser reads every registered model file and one checker binds each to
its live Agent without a per-level exception list.

This is a real, usable property today, and it is what
[`../tests/test_gnn_conformance.py`](../tests/test_gnn_conformance.py) exploits:
the same `assert_conforms(spec, **describe_shapes(agent))` call covers every
level. `TestUpstreamInteroperability` additionally parses every registered file with
the real upstream Markdown parser and requires both readers to extract the same
model, so "expressed in GNN" is demonstrated rather than asserted.

### 4.3 Measured intervention seams are independent

Five source→receiver experiments are implemented and measured; the final source
is a privileged oracle rather than a model level:

| Seam | Receiver change | Recorded outcome |
|---|---|---|
| L1 → L2B | mixes the L1 letter predictive into L2B evidence | harms; ships at the uncoupled anchor |
| L3 → L2B | mixes a lexical-prefix phone distribution into L2B prediction | helps at the selected predictive insertion site |
| L3 → L1g | installs a lexicon-derived next-letter preference in L1g `C` | generation works under policy selection |
| L6 → L5 | blends `P(type \| position tercile)` into the clause Agent's joint `D` | moves the prior with exact zero-weight and scrambled controls |
| punctuation-type oracle → L3 | blends train-estimated `P(word \| punctuation-defined type)` into L3 `D[0]` | helpful on aggregate perplexity and token direction after correction; predicted harm did not reproduce; no L5 posterior is used |

The L6 model also has two recognition readouts that must not be conflated with
each other or with the intervention. The registered observed-position score
indexes the fitted `A` column using the known sentence-position tercile and has
a positive point estimate of claim:stage6_logloss_improvement_nats nats. Its
document-clustered primary interval
[claim:cluster_stage6_logloss_ci_low, claim:cluster_stage6_logloss_ci_high]
spans zero, so it is imprecise and non-confirmatory; no target-effect power
analysis is recorded. The sentence sign result
is secondary: reversing the tercile columns makes mean loss worse while still
yielding a nominally favorable sign test, so the sign statistic passes a
falsifier of the registered mean-loss claim. A separate descriptive score runs
`infer_states` and `update_empirical_prior` through the live L6 Agent, recording
claim:stage6_live_filter_logloss nats and
claim:stage6_live_filter_improvement_nats nats of improvement. That sequential
latent-state score is a different estimand, not a validation bridge for the
observed-position lookup.

The last row does not pass a Clause Agent posterior: its source is a privileged
punctuation-derived reference type paired with a statistic estimated from typed
training sentences. The L6 → L5 row likewise does not send
L6 `C`; it reads the type distribution from L6 `A` and constructs a new prior
for the receiving L5 Agent. Stage 5's privileged-label arm is a sensitivity
reference for that receiver, not a ceiling or hard bound on what the
intervention could deliver under misspecification.

L3 → L1g and L6 → L5 model-output seams and the oracle → L3 intervention were
executed on separate fixtures and scored separately. Their results do not
establish a cascade effect. The connected runtime now executes its own declared
messages, using inferred clause beliefs rather than the historical privileged
oracle, and requires a separate performance evaluation.

### 4.4 The channel that was built, and what it cost

The obvious fourth sense — an actual message from L1 into L2B — was implemented
and measured. It is a recorded negative result, not an omission.

`ailm.levels.coupling.couple_letter_to_phoneme` mixes the L1 predictive over
letters with the observed glyph by weight `alpha` and marginalizes the result
through the G2P emission, producing a soft observation likelihood. The module is
kept, gated, and **not enabled**: `alpha` defaults to `0`, which is provably the
uncoupled decode.

Over claim:coupling_eval_positions held-out positions with a grid of
claim:coupling_grid_size alphas, the best positive alpha is
claim:coupling_best_alpha, reaching claim:coupling_best_accuracy% against
claim:coupling_alpha0_accuracy% for the uncoupled anchor — a margin of
claim:coupling_margin_pts points, i.e. a loss, significant in the direction of
harm at paired-McNemar p = claim:coupling_mcnemar_p
(claim:coupling_mcnemar_b versus claim:coupling_mcnemar_c discordant positions).
Accuracy falls monotonically as `alpha` rises. The recorded support flag is
claim:coupling_supported.

It is a measurement, not a wiring bug. The channel carries real information: at
`alpha = 1`, where the glyph is discarded entirely, the real message scores
claim:coupling_alpha1_accuracy% against
claim:coupling_alpha1_scrambled_accuracy% for the reversed-axis control. The
message simply is not *new* information — it assigns mean probability
claim:coupling_message_mean_prob to the letter actually observed and its argmax
picks that letter claim:coupling_message_argmax_pct% of the time. Spending
observation weight on it substitutes weak evidence for a certainty, because at
position `t` the G2P agent already observes letter `t` while the L1 predictive is
a function of letters `0..t-1` only.

The null carries the same controls a positive result would: the `alpha = 0`
identity agrees with the implementation-independent reference decoder (the
oracle) at
claim:coupling_alpha0_identity_diffs differences and with the pymdp agent at
claim:coupling_pymdp_disagreements disagreements; the anchor is proven
non-vacuous; a scrambled-message control fixes entropy and removes only meaning;
and `validate_alpha_grid` raises `GateVacuousError` unless the grid contains
`0.0` plus at least three distinct positive values, so "no alpha beat the
baseline" cannot come from a sweep that collapsed to a point.
[`../tests/test_coupling.py`](../tests/test_coupling.py) is where those live;
[`decisions.md` entry 9](decisions.md#9--inter-level-soft-observation-coupling-is-a-recorded-negative-result)
is the record.

**A coupling form that adds an *independent* channel is not what was measured
here.** If one is ever built, it needs its own measurement.

---

## 5. Boundaries on what the Agents execute

Naming the boundaries is part of the specification, but they differ between
comprehension, generation, and separate experiments.

- **Comprehension does not select policies.** The L1–L6 comprehension Agents
  have flat `C` arrays and one trivial action. L1g is the explicit exception:
  its action-indexed transition, lexical `C`, policy habit `E`, and closed-form
  policy posterior (`efe_policy_posterior`, verified against
  `agent.infer_policies` with the agreement recorded) implement the measured
  generation seam.
- **Expected free energy is used at L1g, not smuggled into recognition.** The
  comprehension scores are filtering, exact, or NumPy likelihood calculations
  with no preference-driven action choice. The G2P `C` is indexed by observed
  letters, not hidden phones, and remains flat.
- **Standalone level scorers retain their original paths.**
  `update_empirical_prior` is within-level only. The channels in §4.3 were
  measured one at a time; connected execution is selected explicitly through
  `ConnectedRuntime`.
- **No comprehension headline uses a backward pass.** Smoothing and Viterbi are
  whole-word reference arms at L2B, not the shipped causal decoder. Some later
  headline scores use exact or NumPy forward computations rather than the live
  Agent, as §3 states.
- **Ordinary decoding uses fixed fitted arrays.** `A`, `B` and `D` are assembled
  from vendored training artifacts before scoring. A separate drift experiment
  exercises `infer_parameters`; it is not the default decode path and must not
  be generalized into online learning at every level.
- **Stage 7 is not another GNN Agent.** It is a tested speech serialization and
  boundary contract. The modeled linguistic stack ends at L6.
- **Connected execution is conditional and bounded.** Its live next-word
  policy selection is a lexical continuation. The historical independent
  L3 → L1g, L6 → L5 and privileged oracle → L3 effects are not measurements
  of this runtime or evidence of general semantic document generation.

Conformance pins structure, declared roles, supported control and inference
settings, and declared distribution constraints. It does not establish that a
fitted conditional distribution estimates the intended population. A
spec-conformant agent can still hold a badly estimated matrix. That is what the gates, the independent
numpy oracle and the context-free comparators are for
([`gate_design.md`](gate_design.md), [`testing_philosophy.md`](testing_philosophy.md)).

---

## 6. Reading the figures

The historical spec and independent-component figures are generated by
[`../src/ailm/figures/composition.py`](../src/ailm/figures/composition.py) and
checked by [`../tests/test_figures_composition.py`](../tests/test_figures_composition.py).

| Label | What it shows |
|---|---|
| `fig:gnn_pipeline` | Spec sections → builder functions → the constructed `pymdp.agent.Agent`, with conformance as a closing arrow back to the spec. Each matrix box names the sections that pin its size and its role. Declared dimensions are read from `gnn/models/L2b_grapheme_to_phoneme.gnn.md` at generation time |
| `fig:composition` | The three Part-A models side by side; the shared-alphabet identity between L1's state space and L2B's observation space, drawn without an arrowhead; the crossed-out default inter-agent channel and the separately measured harmful coupling result |

The separate `fig:connected_core`, generated by
[`ailm.figures.connected`](../src/ailm/figures/connected.py), depicts the current
runtime channels from the typed channel contract and the report-bound GUM paired
character-loss result. Its encoder annotation discloses what both arms retain;
the aggregate gain is not assigned to individual arrows. The figure reader
refuses incomplete execution bindings, changed document populations and
inconsistent claims. Its controls are in
[`test_figures_connected.py`](../tests/test_figures_connected.py).

Beyond the project-wide layout and legibility gates
([`../tests/test_figure_layout.py`](../tests/test_figure_layout.py)),
`fig:composition` carries a **content** gate: the rendered text is scanned for
the vocabulary of inter-agent messaging, and every occurrence must be a denial.
A positive control plants an offending line on the real figure and requires the
scan to catch it. That gate is lexical, so it is necessary and not sufficient —
it cannot judge a drawing — which is why the connector's lack of an arrowhead is
argued at its call site rather than left to the reader.

---

## Read next

- [`../gnn/README.md`](../gnn/README.md) — what GNN is and why AILM carries specs
- [`../gnn/SPEC.md`](../gnn/SPEC.md) — exactly what the checker enforces, and exactly what it does not
- [`../gnn/AGENTS.md`](../gnn/AGENTS.md) — the contract; read before editing any spec
- [`decisions.md`](decisions.md) — every decision behind the numbers cited here
- [`design/architecture.md`](design/architecture.md) — the scientific architecture: per-level `A`/`B`/`C`/`D`, why words are paths, the scale arithmetic
- [`architecture.md`](architecture.md) — the code architecture: what may import what
- [`output_inventory.md`](reference/output_inventory.md) — every generated artifact, with its label and generator
