# AILM L6 — Narrative Level

# GNN Version: 1.0

# Single-factor, single-modality pymdp agent over document-position terciles

## GNNSection

AILMNarrativeLevel

## GNNVersionAndFlags

GNN v1

## ModelName

AILM Narrative Level

## ModelAnnotation

Level 6 of the AILM hierarchy: document-position terciles over sentence types.

- Hidden factor: the document-position tercile — the opening, middle or closing
  third of a document by sentence index — 3 states. The code, the figure panels
  and the claim labels all name it `phase`; that name is the implementation's,
  and the construct it denotes is the tercile. Nothing in this corpus licenses
  reading it as a narrative phase. Every parameter is a count and nothing is
  fitted by EM.
- Observation modality: the sentence type, 2 observations. This is the same
  alphabet as Level 5's type component, but the scored sequence contains
  punctuation-defined corpus labels, not posteriors emitted by a running Level
  5 Agent.
- A is the learned tercile-conditional type distribution. The directional
  prediction and intervention anchors were committed before the held-out
  measurement (docs/decisions.md entries 25–26). The registered
  **observed-position** score indexes A by the known position tercile and has a
  positive point estimate of claim:stage6_logloss_improvement_nats nats over
  the unigram bar; it does not run pymdp filtering through B and D. At the
  registered document sampling unit, the interval
  [claim:cluster_stage6_logloss_ci_low, claim:cluster_stage6_logloss_ci_high]
  spans zero, so the primary result is imprecise and non-confirmatory. No
  target-effect power analysis is recorded. The
  sentence sign test (p = claim:stage6_sign_test_p) is secondary: reversing A's
  tercile columns makes mean log loss worse while still producing a nominally
  favorable sign result, so that statistic cannot confirm the mean-loss claim.
- A separate descriptive sequential score does run the live Agent, resetting D
  per document and using infer_states/update_empirical_prior through B. It
  records claim:stage6_live_filter_logloss nats and an improvement of
  claim:stage6_live_filter_improvement_nats nats (sign p =
  claim:stage6_live_filter_sign_p). This is a different estimand, not an
  inference-equivalence bridge for the registered score.
- B is the tercile transition — left-to-right by construction. In
  B[next, current] orientation every non-zero entry sits ON or BELOW the
  diagonal; the strictly upper triangle is exactly zero
  (claim:stage6_transition_above_diagonal), asserted rather than trusted
  because this project's first draft stated the wrong triangle (entry 26).
- D is a point mass on the opening phase: tercile(0, n) = 0 for every n.
- C is flat at construction. The independently measured L6→L5 intervention
  reads P(type | phase) from A and blends it into the receiving clause Agent's
  joint D prior, with an exact alpha = 0 identity anchor and a phase-permuted
  scramble control. It does not modify or transmit L6's C, and it is not
  connected to the separate privileged punctuation-type-oracle→L3 or
  L3→L1g runs.

## StateSpaceBlock

# Generative model parameters

A[2,3,type=float]        # Recognition: P(sentence type | phase)
B[3,3,1,type=float]      # Transition: tercile advance, upper triangle zero, 1 control
C[2,1,type=float]        # Log-preferences over types, flat and unused by the D intervention
D[3,1,type=float]        # Prior: point mass on the opening phase

# Hidden states

s_t[3,1,type=float]      # Belief over the phase at sentence t
s_prime[3,1,type=float]  # Belief over the next phase

# Observation

o_t[2,1,type=int]        # Observed sentence-type index at sentence t

# Time

t[1,type=int]            # Discrete sentence position within the document

## Connections

D>s_t
s_t-A
A-o_t
s_t-B
B>s_prime
C-o_t

## InitialParameterization

# A carries the learned phase-conditional type distribution.

A=Learned(generator=scripts/generate_narrative_counts.py, artifact=vendor/ud_ewt_narrative_counts.json, estimator=add-k-Laplace-per-phase, normalization=column-stochastic-over-types)

# B is the tercile transition counted from consecutive-sentence pairs.

B=Learned(generator=scripts/generate_narrative_counts.py, artifact=vendor/ud_ewt_narrative_counts.json, estimator=tercile-pair-counts, normalization=column-stochastic-left-to-right)

# C is flat. The separate L6-to-L5 intervention changes the receiver's D; it
# does not install or transmit this preference vector.

C=Zeros(2)

# D is the structural point mass on the opening phase.

D=Learned(generator=src/ailm/levels/narrative.py, artifact=build_narrative_agent, estimator=structural-point-mass-opening, normalization=sums-to-one-over-phases)

## Equations

# Phase-conditional type likelihood, the recognition content

P(type_t | phase_t) = A[type_t, phase_t]

# The independently measured prior intervention into the clause level

P'_clause(type; alpha, phase) = (1 - alpha) * P_clause(type) + alpha * A[type, phase]
D'_clause[type, case] = P'_clause(type; alpha, phase) * P_clause(first case | type)

# Registered observed-position score, per sentence; direct A lookup, not filtering

L = -(1/N) * sum ln A[type_t, tercile(t, T)]

## Time

Time=t
Dynamic
Discrete

# The prior is reset to D at every document boundary; the unit of this level
# is the document.

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

num_hidden_states: 3
num_obs: 2
num_factors: 1
num_modalities: 1

## Footer

AILM Narrative Level, L6 of the AILM hierarchy. Implemented by
src/ailm/levels/narrative.py (build_narrative_agent, build_phase_models,
build_phase_transition, clause_prior_with_narrative). Conformance to this
spec is checked against a LIVE pymdp agent by the default-deny suite in
tests/test_gnn_conformance.py and locally by tests/test_stage6_gate.py; the
mutation under gnn/mutations/ is its negative control. The registered
observed-position score, the non-confirmatory document-clustered primary
analysis, the distinct descriptive live-Agent score, and the independent
L6→L5 D-prior intervention are recorded separately; chronology and outcome are
in docs/decisions.md entries 25–26 and their audit correction.

## Signature

Unsigned. AILM does not sign its GNN specs; provenance is the git history of
this file plus the checksums recorded in docs/provenance/.
