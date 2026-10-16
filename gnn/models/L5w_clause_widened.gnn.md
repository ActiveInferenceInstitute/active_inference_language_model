# AILM L5w — Widened-Observation Clause Level

# GNN Version: 1.0

# Single-factor, single-modality pymdp agent over the joint (type × widened symbol) space

## GNNSection

AILMClauseLevelWidened

## GNNVersionAndFlags

GNN v1

## ModelName

AILM Widened Clause Level

## ModelAnnotation

The widened-observation arm of Level 5 (docs/decisions.md entry 112): a
two-component mixture of Markov chains over the WIDENED alphabet, embedded as a
single hidden factor over the joint (sentence type × widened symbol) space —
2 types × 10 symbols = 20 states, type-major order.

This separately recorded arm is canonical Level 5 under decision 114. The
L5_clause spec remains the retained case-only ablation: it has its own artifact,
recorder, and consistency guard in scripts/audit_gates.py::audit_stage5.

- Hidden factor: the joint (type, symbol) state, 20 states. The type component
  persists for a whole sentence; the symbol component changes along it.
- Observation modality: the current widened symbol, 10 observations — the 8
  derived case roles plus one distinguished symbol for each VERB and each AUX
  token in surface order. The two added symbols are what let the level see
  subject-auxiliary inversion, which the case-only projection cannot represent
  at all.
- A is deterministic: the observed symbol IS the symbol component of the joint
  state, for either type.
- B is block-diagonal: within each type, that type's add-k widened bigram; ZERO
  cross-type mass. A sentence does not change type mid-way.
- D is P(type) ⊗ P(first symbol | type), the only place the type prior enters.
- C is flat. This level expresses no preferences and selects no actions.

What the measurement found. The widened arm's mixture scores
claim:stage5w_mixture_perplexity against the type-free bar refitted on the
identical widened stream, claim:stage5w_typefree_perplexity. Its privileged
type-label reference scores claim:stage5w_type_oracle_perplexity, a gain of
claim:stage5w_oracle_gain_nats_per_sentence nats per sentence whose
claim:stage5w_oracle_gain_ci_confidence_pct% document-clustered interval runs
from claim:stage5w_oracle_gain_sentence_ci_low to
claim:stage5w_oracle_gain_sentence_ci_high. That reference is a
receiver-sensitivity probe, not a ceiling or a hard information bound under
model misspecification. Classification reaches claim:stage5w_auc against the
length-only comparator's claim:stage5w_auc_length_control and the
first-symbol-is-AUX comparator's claim:stage5w_auc_first_aux_control. The
identical harness on a synthetic corpus whose types genuinely differ classifies
at claim:stage5w_control_classification_pct%; that detectability control
establishes a live instrument for a large planted effect and establishes
nothing about absence or negligibility on English.

Every number above is recorded `descriptive`. Entry 112 is a
development-session specification, not a confirmatory registration, and this
spec describes constructed machinery rather than a confirmed effect.

Disjointness, strengthened rather than dropped. The type label's source is
sentence-final punctuation. PUNCT tokens emit nothing, and a VERB or AUX token
contributes its UPOS only — never its lemma, its feats, or its surface form. A
verb's form is the richest correlate of interrogative mood in this corpus, so
admitting it would make the widening a label leak wearing an alphabet's
clothes. Both properties are asserted per sentence at artifact build time by
ailm.levels.widened.assert_widened_disjointness, which recomputes rather than
trusts (decision 16, one level up, at the wider alphabet).

## StateSpaceBlock

# Generative model parameters

A[10,20,type=float]      # Recognition: P(widened symbol | joint state) — deterministic read-off
B[20,20,1,type=float]    # Transition: block-diagonal per-type widened bigram, 1 control
C[10,1,type=float]       # Log-preferences over observed symbols, flat here
D[20,1,type=float]       # Prior: P(type) ⊗ P(first symbol | type)

# Hidden states

s_t[20,1,type=float]     # Belief over the joint (type, symbol) state at position t
s_prime[20,1,type=float] # Belief over the next joint state

# Observation

o_t[10,1,type=int]       # Observed widened-symbol index at position t

# Time

t[1,type=int]            # Discrete position within the sentence

## Connections

D>s_t
s_t-A
A-o_t
s_t-B
B>s_prime
C-o_t

## InitialParameterization

# A is the structural symbol read-off: state (z, c) emits symbol c with
# certainty, where z is sentence type. The symbol t is reserved for discrete time.

A=Learned(generator=src/ailm/levels/clause.py, artifact=build_joint_emission, estimator=structural-case-readoff, normalization=column-stochastic-over-symbols, n_symbols=10)

# B carries the learned per-type widened bigrams on its diagonal blocks and
# exact zeros off them. The entries are derived from the vendored artifact by
# the named generator; the block structure is the level's structural claim.

B=Learned(generator=scripts/generate_clause_widened_counts.py, artifact=data/widened/ud_ewt_clause_widened_counts.json, estimator=add-k-Laplace-per-type-block, normalization=column-stochastic-block-diagonal)

# C is flat — no preference over observations at this level.

C=Zeros(10)

# D is the learned type prior times each type's learned start distribution.

D=Learned(generator=scripts/generate_clause_widened_counts.py, artifact=data/widened/ud_ewt_clause_widened_counts.json, estimator=type-prior-times-start, normalization=sums-to-one-over-joint)

## Equations

# Mixture evidence of a widened sequence, reported by the exact closed-form
# scorer. The live Agent stores the same mapped arrays; tests check that mapping
# and an independent manual array walk, not live-pymdp equivalence for the
# headline score.

P(o_{1:T}) = sum_z P(z) * P(o_{1:T} | z)

# Type posterior: the block marginal of the joint filtered posterior

P(z | o_{1:T}) = sum_c q_T[(z, c)]

# Per-position perplexity over the held-out typed sentences

PP = exp(-(1/N) * sum ln P(o_{1:T}))

## Time

Time=t
Dynamic
Discrete

# The prior is reset to D at every sentence boundary: the unit of this level
# is the sentence, matching the shipped clause level's per-sentence reset.

## ActInfOntologyAnnotation

A=RecognitionMatrix
B=TransitionMatrix
C=LogPreferenceVector
D=Prior
s_t=HiddenState
s_prime=NextHiddenState
o_t=Observation
t=Time

## ModelParameters

num_hidden_states: 20
num_obs: 10
num_factors: 1
num_modalities: 1

## Footer

AILM Widened Clause Level, the L5w arm of the AILM hierarchy. Implemented by
src/ailm/levels/widened.py (build_widened_agent, load_widened_clause_artifact)
over the shipped builders in src/ailm/levels/clause.py (build_type_models,
build_joint_transition, build_joint_start, build_joint_emission), which take the
observation alphabet as a parameter so this arm runs the level's own estimators
rather than a parallel copy. Conformance to this spec is checked against a LIVE
pymdp agent by the default-deny suite in tests/test_gnn_conformance.py and
locally by tests/test_stage5_widened.py; the mutations under gnn/mutations/ are
negative controls. The arm's outcome is recorded by
scripts/record_stage5_widened_claims.py and its gate is classified by
scripts/audit_gates.py::audit_stage5, which since decision 114 is the canonical
Stage 5 row. It is separate from and never alters audit_stage5_case_only, the
retained case-only ablation.

## Signature

Unsigned. AILM does not sign its GNN specs; provenance is the git history of
this file plus the checksums recorded in docs/provenance/.
