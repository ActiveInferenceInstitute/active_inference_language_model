# The AILM Level Hierarchy — the governing description

This file is the single authoritative description of AILM's level structure: what
each level *is*, what relationship it has to the level below it, and what is
**not** hierarchical here.

**Precedence.** Where any other file disagrees with this one about the level
relationships, this file governs and the other file is a defect to be fixed. It is
written against the shipped code (`src/ailm/levels/`, `src/ailm/figures/architecture.py`)
and the recorded ledger (`data/claims.json`), not against the design intent.

Scope note: `docs/design/architecture.md` records the rationale behind this
corrected shipped structure and must agree with the compact table here. `ISA.md`
and the handoff records are historical plans, not current-status authorities. See
[Conflicting descriptions](#description-audit) at the end for the
audit history.

---

## Choose the model path

| Path | Execution and evidence |
| --- | --- |
| [Connected runtime](#connected-development-runtime) | Raw or explicitly UPOS-conditioned input, timed cross-level messages, local conditional references and a separate GUM comparison |
| [Standalone level scores](#what-is-shipped-level-by-level) | Historical stage estimands, including exact/NumPy scorers and their stated live-agent agreement checks |
| [Historical intervention seams](#the-relationship-between-levels--stated-precisely) | Individually measured effects; they are not substitutes for the connected evaluation |
| [Generation](design/generation_problem.md) | The measured standalone L3→L1g law and the bounded connected next-word policy path remain distinct |
| [Browser inspection](../src/ailm/dashboard/README.md) and [speech boundary](design/speech_boundary.md) | Separate demonstrator and symbol interface, not new connected-model measurements |

## Current models and evaluation scopes

The historical scores below remain tied to their original independent evaluations.
An additional connected development runtime now calls the GNN-built agents on an
explicit multirate schedule; its contract is summarized under
[Connected development runtime](#connected-development-runtime).

Live pymdp builders exist for L1, L2A, L2B, L3, L4, L5, and L6. Stage 0 uses a
synthetic smoke Agent to verify the installed API, and Stage 7 is a
speech-boundary interface. The linguistic models' A/B/C/D structures are explicit, but their
headline scores do not all run through pymdp inference: L1 does; L2A's
full-dictionary score is an exact sufficient-statistic calculation and L2B's
full-set score is a NumPy forward filter, each licensed by a bounded
live-pymdp/NumPy agreement check; L3 uses an exact path identity; L4 uses a
NumPy forward filter that the live-pymdp recursion reproduces at every held-out
position, with the agreement recorded; L5 uses a closed-form mixture, with a
recorded live-pymdp posterior-agreement check (argmax and joint-posterior
tolerance) on the full held-out set that is not score equivalence for the
headline estimand; and L6's registered observed-position
score indexes the fitted conditional table directly. L6 also has a separate,
descriptive sequential score through the live Agent; it is a different estimand,
not an inference-equivalence bridge for the registered score. L5's canonical
gate is the widened-observation clause arm that decision 114 promoted: it was
registered before its artifact existed, it beats its refitted bar, and its
paired privileged-label gain resolves away from zero. Both of its limits belong
with that verdict. Registration and outcome share one development session, so
its labels are descriptive, and it has no held-back replication because the
`ud_ewt_test` population was already spent under decision 111. The case-only
arm it displaced is retained as an ablation and keeps its record: its gate is
non-passing, its paired privileged-label gain is unresolved, and it stays
non-passing on the held-back UD EWT test split, where the mixture and the
always-majority baseline agree on every one of
claim:confirmatory_stage5_sentences typed sentences, leaving no discordant pair
to test. L6 has a preregistered positive point estimate whose
document-clustered primary interval spans zero on the development population.
Its registered observed-position sign test crosses the frozen family threshold
on the held-back split (claim:confirmatory_stage6_sign_test_favored positive
against claim:confirmatory_stage6_sign_test_opposed negative,
p = claim:confirmatory_stage6_sign_test_p), which is the sentence-level
estimand and not the document-level one. The document-level estimand was
registered separately under decision
claim:confirmatory_stage6_clustered_decision and carried to that same split,
where the interval
[claim:confirmatory_stage6_clustered_ci_low,
claim:confirmatory_stage6_clustered_ci_high] excludes zero over
claim:confirmatory_stage6_clustered_documents documents at
claim:confirmatory_stage6_clustered_confidence_pct percent coverage. Three
limits are recorded with it and are not optional when quoting it: the
development-population interval is not retracted and still spans zero; the same
held-back bootstrap read at the family-corrected
claim:confirmatory_stage6_clustered_corrected_confidence_pct percent spans zero
as well; and the held-back population's own resolution,
claim:confirmatory_stage6_clustered_min_detectable_effect_nats nats, is coarser
than the development-population effect of
claim:confirmatory_stage6_clustered_control_planted_shift_nats nats, so a null
there would have been uninterpretable and the detection is of a larger effect
rather than a demonstration that this effect size generalizes. That arm is
descriptive, not confirmatory: the population was already spent when it was
registered.

Four of the sixteen frozen family members have been rescored once against data
nothing here had scored: the whole of `en_ewt-ud-test.conllu`, registered by
decision 111 and vendored through the identical generators with no stride and
no re-fitting. Two of the four cross the frozen threshold in the recorded
direction (the Stage 4 derived case comparison and the Stage 6 sign test), and
two are uncomputable on that population because the model and its comparator
produce no discordant pair at all (the Stage 4 gold comparison and the Stage 5
classification). The remaining twelve members are unchanged
development-population inference. The population is spent: the sentinel
`confirmatory_run_completed_ud_ewt_test` makes a second pass refuse.

The cross-level interventions have been measured individually, with mixed outcomes:
L1→L2B soft-observation coupling harms; the selected L3→L2B predictive lexical
channel helps; L3→L1g supplies the lexical preference used by generation;
L6→L5 changes the clause joint prior; and a privileged punctuation-type
oracle→L3 changes the word prior with a helpful result on both directional
summaries whose predicted harm did not reproduce. The oracle arm does not use
an L5 posterior. The last three are independent
experiments. Their recorded effects do not measure the newer connected schedule.

The typed registry also carries two structural case-word seams into L3's word
prior: the case-word bridge, which marginalizes the case level's own per-token
role posterior into a train-only-fitted message (`ailm.case_word_channel`), and
its privileged-oracle diagnostic. The bridge now has a scored estimand: its
influence sweep on the EWT-dev development population is recorded under
decision 121 (`scripts/record_case_word_influence_claims.py`) — an exploratory,
prior-only measurement, reported in the results section, never confirmatory.
The privileged-oracle diagnostic remains registered without a scored
estimand — its evaluation (`scripts/evaluate_case_word_channel.py`) is
structural and records nothing — so it stays a seam, not a measured
intervention, and no performance reading exists for it.

---

## What is shipped, level by level

Each L1-L6 builder returns a live `pymdp.agent.Agent`. That establishes the model
object and its arrays; it does not by itself establish that a reported score used
`Agent.infer_states`. The final column makes that distinction explicit.

| Level | Hidden factor(s) | Observation | `A` | `B` | `C` | `D` | Headline score path |
|---|---|---|---|---|---|---|---|
| **L1 letter** | current letter | current letter | identity | Brown letter bigram | zero | uniform letter prior | live pymdp filtering |
| **L2A phone sequence** | current phone | current phone | identity | CMUdict phone bigram | zero | learned phone unigram | exact full-dictionary sufficient-statistic score; bounded live-pymdp/NumPy agreement check |
| **L2B G2P** | current phone | current letter | learned `P(letter \| phone)` | train-only phone bigram | zero | learned first-phone distribution | full-set NumPy forward filter; bounded live-pymdp zero-disagreement check |
| **L3 word** | word identity × within-word position | letter or END | deterministic spelling, dependent on both factors | identity over word × forward position shift | zero | word unigram × opening-position point mass | exact closed-form path score; sampled live-pymdp identity check |
| **L4 case** | syntactic case role | UPOS | learned `P(UPOS \| case)` | learned case-role bigram | zero | learned first-case distribution | NumPy forward filter; exhaustive position-level live-pymdp comparison over the held-out test set, agreement recorded; rescored once on the held-back test split under decision 111, where the paired comparison crosses the frozen threshold on the derived population and yields no discordant pair on the gold population |
| **L5 clause** | joint sentence type × widened symbol | widened symbol: a case role, or `VERB`, or `AUX` | deterministic read-out of the symbol component | learned type-specific widened-symbol chains in persistent-type blocks | zero | type prior × first-symbol-given-type | exact closed-form mixture; live-pymdp posterior-agreement check (argmax and joint-posterior tolerance) on the full held-out set, not score equivalence for the headline estimand; gate registered before the artifact existed and passing, with no held-back replication because the population is spent. The retained case-only ablation observes case roles alone, is non-passing with its gain unresolved, and was rescored once on the held-back test split under decision 111, where that non-passing result reproduces with zero discordant pairs |
| **L6 document position** | document-position tercile | sentence type | learned `P(type \| tercile)` | learned left-to-right tercile transition | zero | opening-tercile point mass | registered observed-position direct conditional log loss; separate descriptive live-pymdp sequential predictive loss; different estimands; the registered observed-position score was rescored once on the held-back test split under decision 111 and crosses the frozen threshold there; the document-level primary estimand was registered separately and excludes zero on that split while spanning zero on the development population |

L0 is an infrastructure smoke agent, not a linguistic level. L1g is the separate
controllable letter agent in `src/ailm/generation.py`: identity `A`, action-indexed
deterministic `B`, zero `C` until each lexical preference is installed, uniform
`D`, and a Brown-bigram policy habit `E`.

Generation by policy selection is built and measured at the L3→L1g seam.
The policy posterior over next-letter commitments under expected free energy is
computed in closed form (`efe_policy_posterior`) and verified against
`agent.infer_policies`, with the worst deviation recorded as
claim:generation_policy_posterior_pymdp_max_abs_dev over
claim:generation_policy_posterior_pymdp_states_checked (letter, prefix) states;
the L3 lexicon supplies `C` and the Brown bigram supplies `E`. Neither L6 nor L5
participates in that run. Results are reported over claim:generation_seeds seeds.

### Prior-reset behavior

- **L0** has no linguistic stream or reset policy; its synthetic Agent exists
  only for API smoke tests.
- **L1** does not reset per word. Space is a symbol inside the letter chain.
- **L2A** resets to `D` at every word boundary (phone sequences restart).
- **L2B** resets to `D` at every word (first-phone distribution).
- **L3** starts each scored word from its word-frequency prior and the opening
  position point mass.
- **L4-L5** start each scored sentence from their learned `D` distributions.
- **L6** starts each document from its opening-tercile point mass.

---

## The relationship between levels — stated precisely

### L2B ← L1: cross-representation inference, inside ONE agent

This is the defensible meaning of "hierarchical" in AILM, and it is worth stating
exactly:

> L2B's hidden factor is drawn from the phone inventory and its observation modality
> is drawn from the letter alphabet. `A` is therefore a genuine emission of
> claim:stage0_letter_alphabet_size rows over claim:phoneme_count_39 columns, linking two *different* alphabets of
> *different* sizes. Letters are observed; phones are inferred; the posterior is
> over a variable that never appears in the data.

What makes this a real hierarchical relationship:

1. The observation alphabet is the level below's *state* alphabet. L2B consumes the
   representation L1 is defined over.
2. `A` is not square, so a transposed emission is a dimension violation.
3. The inference is falsifiable against a context-free comparator:
   claim:stage2_g2p_accuracy% vs claim:stage2_g2p_baseline_accuracy%, paired
   McNemar p = claim:stage2_g2p_mcnemar_p.

What this relationship is **not**: it is not two agents exchanging beliefs. L1 and
L2B's standalone scorer reuses L1's alphabet without running L1. The connected
runtime separately executes both under its explicit channel schedule.

### L3 → L2B lexical channel (helps)

The selected lexical channel changes L2B's predictive phone distribution by
narrowing which phones are consistent with the current word's phonological form.
Its recorded marginal effect is claim:lexical_best_margin_pts points at
p = claim:lexical_best_mcnemar_p (ISC-41). Separate factorial experiments also
tested observation-likelihood insertion; do not conflate those cells with the
selected predictive channel.

### The factorial, the rule, and what each settled

Three later measurements changed how the two results above should be read, and
each corrected the one before it. They are worth following in order, because the
sequence is the argument.

**The factorial** (`src/ailm/channels.py`, decision 46) crossed message *source*
with insertion *site*. Until it ran, the L1 and L3 channels differed in both at
once, so the gap between them attributed to neither. Completing the 2×2 showed
the letter message harming significantly at the evidence site and the lexical
message resolving neither way there, while the same lexical message helps by
claim:lexical_best_margin_pts points at the predictive site. Same message, same
receiver, a gain at one site and nothing detectable at the other, and only the
insertion point differing. (Before the backoff correction of decision 110 the
lexical evidence-site cell recorded a significant cost; the corrected cell is
a null.)

**The rule** (decision 47) then showed that site effect was itself an artifact.
All four cells shared linear mixing. Under geometric pooling — a product of
experts, and what `pymdp` uses to combine modalities — the two sites are
provably the same computation, measured at a maximum absolute gap of
claim:geometric_site_max_abs_gap with claim:geometric_site_argmax_disagreements
argmax disagreements anywhere. A factorial cannot see the factor every one of its cells holds fixed.

**The control** settled it. Under geometric pooling a uniform message is inert by
construction, so dilution is structurally excluded and every remaining effect is
content. The letter message *still* harms, monotonically worse the more it is
weighted. That vindicates the original reading below: the channel carries nothing
L2B does not already observe. It took two corrections and a third rule to
establish on evidence what had been a plausible story.

**Fitting** (decision 49) closed the last gap. Four numbers governing the lexical
gain were hand-set; selecting them on an inner split of TRAIN, with the held-out
set scored once after selection closed, buys claim:fitted_gain_pts points at
p = claim:fitted_paired_p and puts the level at claim:fitted_accuracy%, which is
claim:fitted_margin_over_anchor_pts above the uncoupled decode.

### L1 → L2B coupling (measured, harmful)

`src/ailm/levels/coupling.py` implements L1's predictive mixed into L2B's
observation likelihood. Result: **loss** of claim:coupling_margin_pts points at
p = claim:coupling_mcnemar_p — the channel carries nothing L2B does not already
observe. See [the coupling section](#the-l1l2b-cross-level-message-was-built-measured-and-rejected).

### Punctuation-type oracle → L3 word-prior intervention (helpful after correction)

`src/ailm/levels/clause_channel.py` mixes a clause-derived word-distribution
statistic, estimated on the train split, into L3's word prior `D[0]` at weight
`alpha`. Its type is the privileged punctuation-defined reference, not an L5
posterior. The intervention was pre-registered as a **harm** (decision 27): L3 already
observes the words it recognizes, so a clause-derived statement about them should
be redundant. Measured over claim:clause_word_eval_emissions held-out emissions,
aggregate perplexity changes in the *opposite* direction — by
claim:clause_word_best_delta_ppl at the high tested endpoint
claim:clause_word_best_alpha, where the prediction called for monotone harm.
Every tested positive step improves; the sweep therefore does not locate an
optimum.
More individual tokens improve than worsen, so the two recorded directional
summaries agree. A whole-document bootstrap puts the mean log gain in
[claim:clause_word_cluster_loggain_ci_low,
claim:clause_word_cluster_loggain_ci_high] nats per emission, and the
assumption-conditional sign-flip diagnostic reaches its declared resolution
floor claim:clause_word_cluster_loggain_p_floor. The real message also beats a
type-free bridge marginal (claim:clause_word_typefree_ppl_at_best) and a swapped
type-label control (claim:clause_word_swapped_type_ppl_at_best). The corrected
result falsifies the oracle arm's predicted harm without becoming a fresh
confirmatory population or establishing a general channel law. Recorded by
`scripts/record_clause_channel_claims.py`; guarded by
`tests/test_clause_word_channel.py`.

### Two senses of "channel" in this codebase

"Channel" is used in two distinct senses and the two are not interchangeable.

**Measured receiver interventions:** every source-to-receiver experiment whose
effect was measured. The typed registry distinguishes model outputs from the
privileged punctuation-type oracle rather than counting both as model senders.
They are named in the one-sentence answer above:
L1→L2B (coupling, harmful), L3→L2B (predictive lexical message, helpful),
L3→L1g (generation preference), L6→L5 (clause-prior intervention), and the
punctuation-type oracle→L3 word-prior intervention (helpful after the corrected
Brown partition).

**Top-down intervention inventory:** L3→L1g writes a lexical preference to
L1g's `C`; L6→L5 blends a phase-conditioned type distribution into L5's joint
`D`; the privileged oracle arm blends a typed word distribution into L3's
`D[0]`; and the two case-word seams blend a case-posterior-derived (bridge) or
privileged-oracle-derived (diagnostic) word message into that same `D[0]`. Only
the first is a `C`-preference channel; the first three — generation, L6→L5, and
the case-word bridge — use a model-level sender, while the oracle arm and the
diagnostic are privileged references. The measured arms were run independently
and must not be drawn or described as a connected cascade; the case-word
bridge's influence is measured as an exploratory development-population sweep
(decision 121), and the privileged-oracle diagnostic remains the one
registered structural inlet whose evaluation records nothing.

---

## What is NOT hierarchical here

### `update_empirical_prior` does not couple anything

`update_empirical_prior` is a **within-level temporal** operation everywhere it
appears in `src/`. The L1 and L2 scorers use it after `infer_states`; bounded L3
and L4 checks also use it when comparing a live agent with their exact/NumPy
paths. The L3 headline score, L4 full score, L5 closed-form mixture, and L6 table
score do not call it. Where it is called, it pushes the same agent's posterior
through that same agent's `B`; it is not a message bus.

### The L1→L2B cross-level message was built, measured, and rejected

`src/ailm/levels/coupling.py` implements the smallest honest version of an L1→L2B
message: the L1 predictive over letter *t* mixed with the observed one-hot at
weight `alpha`, marginalized through L2B's emission to form a **soft** observation
likelihood over phones. `alpha = 0` is the shipped uncoupled decode.

The result is a null, recorded by `scripts/record_coupling_claims.py` over
claim:coupling_eval_positions held-out positions — the channel harms.

That harm is measured on argmax accuracy, and accuracy is not a proper scoring
rule. `scripts/record_channel_predictive_loss_claims.py` re-scores the same cell
as the receiver's own proper predictive loss over
claim:channel_proper_loss_positions positions: the anchor costs
claim:channel_proper_loss_anchor_nats nats per position and no engaged weight
improves on it, so **sign and shape agree between the two axes** (decision 59).
An oracle-sender arm gives the reason the harm is structural rather than
incidental — a perfect message about the letter moves the loss by
claim:channel_proper_loss_oracle_spread nats, because the receiver already
observes the letter, so its sensitivity to the message variable is
claim:channel_proper_loss_sensitivity_nats and the realized effect is bounded
above by zero.

### Historical cascade boundary

Comprehension-level `C` arrays are zero. The recorded generation experiment uses
an L3-derived lexical preference in L1g's `C`; the narrative intervention moves
the clause `D` prior; and the privileged oracle intervention moves the word
`D[0]` prior. These were separate measured seams (decisions 17, 25-26, 27).
Their recorded effects cannot be combined into a measured cascade result.

### Connected development runtime

`ailm.levels.connected.ConnectedRuntime` now composes GNN-built letter, phone,
word, case, widened-clause and document-position agents. The train-only bridge
kernels are assembled by `ailm.levels.connected_builder.load_connected_model`.
Raw text supplies normalized ASCII lexical runs and sentence boundaries;
structured `Token` inputs may additionally supply UPOS before spelling. Derived
case, dependency, morphology, phoneme and sentence-type labels have no input
slot. Raw-mode UPOS is inferred from the completed lexical posterior with a
training-distribution backoff for unsupported words.

The schedule carries document-position predictions into clause priors, clause
predictions into case priors, and case beliefs into word priors before receiving
the current spelling. Prefix-conditioned word beliefs then affect phone and
letter predictions. Completed-token case evidence updates the widened clause
agent; completed-sentence type beliefs update document position for the next
sentence. Each call starts a new document; word and sentence boundaries reset
the level states specified in the [runtime contract](design/connected_runtime.md).
The named-channel-disabled comparator retains the same observed spelling and
train-fitted postword UPOS encoder; it disables the named connections rather
than isolating every representation. OOV tokens remain in the character stream
and give infinite full-stream closed-vocabulary word loss. Structured external
UPOS changes the information available and is not the GUM raw-text task.
`generate_next_word` uses the resulting next-word prior to drive live L1g
`infer_policies` and `sample_action` calls for a bounded lexical continuation.

This is an executable conditional composition with explicit ablation and
scrambled-message controls. Its exact-reference checks concern each declared
conditional update, not a globally exact joint posterior. The channel strength
is an engineering setting. The locally frozen GUM comparison reports
claim:connected_gum_document_mean_gain nats of equal-document normalized-character
gain, with interval [claim:connected_gum_document_ci_low,
claim:connected_gum_document_ci_high]. Its full vocabulary/OOV accounting and
local prospective scope are in the [results](manuscript/04_results.md#sec:connected_gum). It does not change the estimands or qualified verdicts of
the historical stage gates. The code and controls are in `tests/test_connected.py`.

---

## Description audit

Audited 2026-07-31 against the shipped code. Each row is a description that once
disagreed with this file, and its current state.

| File | Was | Now |
|------|-----|-----|
| `docs/manuscript/00_abstract.md` | "Two levels are built" | **FIXED** — now enumerates the modeled levels and separates their score paths and qualified outcomes |
| `docs/manuscript/08_conclusion.md` | "Two levels ... three of its five levels do not exist" | **FIXED** — now enumerates the built levels and qualifies the later-stage outcomes |
| `docs/manuscript/03_methods.md` | Architecture table shows only L1-L2 | **FIXED 2026-08-04** — `tbl:level_stack` now extends through L6 (03_methods.md:30-31) |
| `AGENTS.md` | Stage table shows Stages 3-6 "NOT STARTED" | **UPDATED 2026-07-31** — all stages now reflect built state |
| `docs/hierarchy.md` (this file) | "Three pymdp agents exist"; "L3 and above: nothing" | **UPDATED 2026-07-31** — full 8-column table, generation, channels |

Terminology, so the collision stops recurring:

- **cross-representation inference** — what L2B does. Correct, measured, shipped.
- **coupling** — the L1→L2B soft-observation message in `levels/coupling.py`. Built,
  measured, harmful.
- **lexical channel** — L3→L2B phone-observation constraint. Prospectively specified in-session, helps.
- **typed word-prior intervention** — privileged punctuation-type oracle→L3
  message into the word prior in `levels/clause_channel.py`. Pre-registered as
  harmful, corrected measurement helpful on aggregate perplexity and token
  direction; it is not an L5-output channel.
- **generation seam** — L3→L1g lexical preferences in a separate controllable
  letter agent; L5 and L6 are absent from that run.
- **document-position → clause intervention** — L6→L5 message into the clause
  joint `D`; measured independently of generation and the oracle→word arm.
- **general-purpose belief/message passing between all levels** — not claimed;
  the connected runtime executes only its declared conditional schedule
  in AILM; the measured coupling, lexical, generation, narrative→clause, and
  oracle→word interventions are explicit typed experiments rather than a shared
  message bus.

---

## Cross-links

| Read this for | File |
|---|---|
| **The level relationships** | **this file — governing** |
| The coupling experiment | `src/ailm/levels/coupling.py` docstring |
| The lexical channel | `src/ailm/levels/lexical_channel.py`, `scripts/record_lexical_channel_claims.py` |
| The punctuation-type-oracle → word-prior intervention | `src/ailm/levels/clause_channel.py`, `scripts/record_clause_channel_claims.py` |
| Generation | `src/ailm/generation.py`, `scripts/record_generation_claims.py` |
| Per-level GNN specs | `gnn/models/L*.gnn.md` |
| Why sequence statistics belong in `B` | `src/ailm/levels/AGENTS.md`, `docs/manuscript/03_methods.md` |
| Design rationale aligned to the shipped six-level structure | `docs/design/architecture.md` |
| Stage gate audit classifications | `scripts/audit_gates.py`, `docs/manuscript/03_methods.md` (taxonomy), `docs/manuscript/04_results.md` (outcomes) |
| Module ownership and imports | `docs/architecture.md` |
