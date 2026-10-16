# AILM L5 — Clause Level MUTATION: identity joint transition

# GNN Version: 1.0

# Single-factor, single-modality pymdp agent over the joint (type × case) space

## GNNSection

AILMClauseLevelIdentityTransition

## GNNVersionAndFlags

GNN v1

## ModelName

AILM Clause Level

## ModelAnnotation

Level 5 of the AILM hierarchy: a two-component mixture of case-role Markov
chains, embedded as a single hidden factor over the joint (sentence type ×
case role) space — 2 types × 8 cases = 16 states, type-major order.

- Hidden factor: the joint (type, case) state, 16 states. The type component
  persists for a whole sentence; the case component changes along it.
- Observation modality: the current case role, 8 observations. This is the same
  alphabet Level 4 uses as a hidden state, but the scored observations are
  UD-derived corpus labels, not posteriors emitted by a running Level 4 Agent.
- A is deterministic: the observed role IS the case component of the joint
  state, for either type.
- B is deliberately replaced by the 16-state identity. It preserves the joint
  alphabet but removes every learned within-type case transition; this is the
  mutation that the live-agent conformance check must reject.
- D is P(type) ⊗ P(first case | type), the only place the type prior enters.
- C is flat. This level expresses no preferences and selects no actions.

What the measurement found. Stage 5 is non-passing and its paired
privileged-label gain remains unresolved (docs/decisions.md entry 83). The
privileged type-label arm selects the fitted chain with a privileged
reference label. Its claim:stage5_oracle_gain_ci_confidence_pct% sentence-paired
gain interval runs from claim:stage5_oracle_gain_sentence_ci_low to
claim:stage5_oracle_gain_sentence_ci_high nats per sentence; no equivalence
margin was registered. It is a receiver-sensitivity reference, not a ceiling or
hard information bound under model misspecification. The
mixture scores claim:stage5_mixture_perplexity against the bar's
claim:stage5_typefree_perplexity (worse), and MAP classification recovers
claim:stage5_interrogative_recall_pct% of held-out interrogatives against a
claim:stage5_majority_baseline_pct% majority baseline. The identical harness
on a synthetic corpus whose types genuinely differ classifies at
claim:stage5_control_classification_pct%. That detectability control establishes
a live instrument for a large planted effect; it does not establish absence or
negligibility on English. This spec describes the constructed machinery; the
non-passing/unresolved status describes its measured value on this split.

Disjointness. The type label's source is sentence-final punctuation; the
observations are derived case roles; PUNCT tokens carry no case role, so the
label source cannot appear in the observation stream (decision 16, one level
up).

## StateSpaceBlock

# Generative model parameters

A[8,16,type=float]       # Recognition: P(case role | joint state) — deterministic read-off
B[16,16,1,type=float]    # Transition: block-diagonal per-type case bigram, 1 control
C[8,1,type=float]        # Log-preferences over observed roles, flat here
D[16,1,type=float]       # Prior: P(type) ⊗ P(first case | type)

# Hidden states

s_t[16,1,type=float]     # Belief over the joint (type, case) state at position t
s_prime[16,1,type=float] # Belief over the next joint state

# Observation

o_t[8,1,type=int]        # Observed case-role index at position t

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

# A is the structural case read-off: state (z, c) emits role c with certainty,
# where z is sentence type. The symbol t is reserved for discrete time.

A=Learned(generator=src/ailm/levels/clause.py, artifact=build_joint_emission, estimator=structural-case-readoff, normalization=column-stochastic-over-roles)

# B carries the learned per-type case bigrams on its diagonal blocks and
# exact zeros off them. The entries are derived from the vendored artifact by
# the named generator; the block structure is the level's structural claim.

B=Identity(16)

# C is flat — no preference over observations at this level.

C=Zeros(8)

# D is the learned type prior times each type's learned start distribution.

D=Learned(generator=scripts/generate_clause_counts.py, artifact=vendor/ud_ewt_clause_counts.json, estimator=type-prior-times-start, normalization=sums-to-one-over-joint)

## Equations

# Mixture evidence of a case sequence, reported by the exact closed-form scorer.
# The live Agent stores mapped arrays; conformance tests check those arrays. This
# is not live-pymdp equivalence for the headline Stage 5 score. The identity
# replacement is a negative control, not an estimator.

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
# is the sentence, matching L3's per-word and L4's per-sentence resets.

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

num_hidden_states: 16
num_obs: 8
num_factors: 1
num_modalities: 1

## Footer

AILM Clause Level, L5 of the AILM hierarchy. Implemented by
src/ailm/levels/clause.py (build_clause_agent, build_type_models,
build_joint_transition, build_joint_start, build_joint_emission). Conformance
to this mutation is checked against a LIVE pymdp agent by the default-deny
suite in tests/test_gnn_conformance.py and locally by tests/test_stage5_gate.py.
Stage 5 is non-passing and its paired privileged-label gain is unresolved — see
docs/decisions.md entry 83 — and scripts/audit_gates.py::audit_stage5 enforces
that joint status rather than demanding a metric pass.

## Signature

Unsigned. AILM does not sign its GNN specs; provenance is the git history of
this file plus the checksums recorded in docs/provenance/.
