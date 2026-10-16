# AILM L4 — Case Level

# GNN Version: 1.0

# Single-factor, single-modality pymdp agent over 8 heuristic syntactic case
# roles — NOT morphological case

## GNNSection

AILMCaseLevel

## GNNVersionAndFlags

GNN v1

## ModelName

AILM Case Level

## ModelAnnotation

Level 4 of the AILM hierarchy: a case-role sequence model over the tokens of a
sentence.

- Hidden factor: the token's case role, 8 states (NOM, ACC, GEN, DAT, INS, LOC,
  ABL, VOC).
- Observation modality: the token's UPOS tag. The alphabet is sized by the
  population: claim:stage4_upos_alphabet_size tags on the derived population,
  which is the one this spec describes and the one the gate is stated over.
- A is a REAL learned emission, P(UPOS | case), over a different alphabet from
  the hidden one — structurally this level is L2B one level up.
- B is the case-role transition along the sentence, P(case_t | case_{t-1}). This
  is the level's claim: that case roles follow a learnable order.
- D is the case distribution at a sentence's first labeled token.
- C is flat. This level expresses no preferences and selects no actions.

What the model is NOT allowed to observe. The observation is the UPOS tag and
nothing else. Neither `deprel` nor `feats[Case]` reaches the model, because the
evaluation labels are DERIVED from those fields — a model observing them would
be reading its own answer key, reaching near-perfect agreement while measuring
nothing. This trap is more dangerous than any mis-wiring below it: every earlier
one made a model visibly worse, whereas this one makes it excellent, so there is
no numeric signature to notice and the guard has to be structural. See
docs/decisions.md entry 15.

Agreement, not accuracy. Only NOM/ACC/GEN are morphologically marked in English;
claim:stage4_cases_without_morphology of the eight roles have no morphological
ground truth in English EWT at all, so their labels come from a documented
syntactic heuristic. That heuristic agrees with real morphology on
claim:stage4_heuristic_gold_agreement_pct% of the tokens carrying both — which in
this corpus means pronouns, in the claim:stage4_gold_cases_present marked roles.
It checks the heuristic exactly where the corpus can check it. It is NOT a bound
on the claim:stage4_cases_without_morphology roles that have no morphological
ground truth here, and therefore not a bound on the fidelity of the derived
labeling as a whole. In particular, that overlap contains
claim:stage4_heuristic_gold_adposition_token_tokens token-level adposition-route
examples; decision 84 records the route's separate ADP/head audit. The implementation's scoring
function is named `case_agreement` rather than `case_accuracy` so that
overstating it takes deliberate effort.

Gate. Stage 4 is gated on beating the context-free argmax P(case | UPOS) bar on
the derived population (claim:stage4_derived_baseline% against
claim:stage4_derived_agreement%), defended by a paired McNemar test. The
declared claim:stage4_declared_gate% threshold against the UD case oracle has
two population-dependent defects:
it is VACUOUS on the gold population, where a constant NOM predictor scores
claim:stage4_gold_majority%, and ABOVE_REFERENCE on the derived population,
where every measured, exceedable comparator falls below it. Which defect you get depends only
on which population the threshold is scored against, which is why a threshold
without a named population is not a threshold.

No gate is stated on the gold population. After a held-out split it carries
claim:stage4_gold_test_tokens tokens, and the model and the context-free
baseline make IDENTICAL predictions on every one of them —
claim:stage4_gold_mcnemar_b and claim:stage4_gold_mcnemar_c discordant pairs.
Every gold-morphology token in English EWT is a PRON, so the gold observation
alphabet has size claim:stage4_gold_upos_alphabet_size. That removes state
information from the emission but not the learned transition. The committed
split's identical predictions are therefore a measured empirical null, not a
universal consequence of a one-tag observation alphabet.

Learned entries are a derived artifact, not part of this specification. The
counts are reproduced byte-identically from the vendored artifact by the
generator named in InitialParameterization.

## StateSpaceBlock

# Generative model parameters

A[12,8,type=float]       # Recognition matrix, P(UPOS tag | case role)
B[8,8,1,type=float]      # Transition, P(case_t | case_{t-1}), 1 control
C[12,1,type=float]       # Log-preferences over observed UPOS tags, flat here
D[8,1,type=float]        # Prior over the case role of the first labeled token

# Hidden states

s_t[8,1,type=float]      # Belief over the current token's case role
s_prime[8,1,type=float]  # Belief over the next token's case role

# Observation

o_t[12,1,type=int]       # Observed UPOS tag index at position t

# Time

t[1,type=int]            # Discrete token position within the sentence

## Connections

D>s_t
s_t-A
A-o_t
s_t-B
B>s_prime
C-o_t

## InitialParameterization

# A is the learned P(UPOS | case) emission, normalized over the OBSERVATION axis
# for each case. The transpose is a different model that pymdp would accept, so
# the axis is named here and asserted in the tests rather than assumed.

A=Learned(generator=scripts/generate_case_counts.py, artifact=vendor/ud_ewt_case_counts.json, estimator=add-k-Laplace, normalization=column-stochastic-over-upos)

# B holds the learned case-role transition. This is where the level's sequence
# claim lives — in the TRANSITION, not the emission, as at every level below.

B=Learned(generator=scripts/generate_case_counts.py, artifact=vendor/ud_ewt_case_counts.json, estimator=add-k-Laplace, normalization=column-stochastic-over-next-case)

# C is flat — no preference over observations at this level.

C=Zeros(12)

# D is the learned distribution over the first labeled token's case role.

D=Learned(generator=scripts/generate_case_counts.py, artifact=vendor/ud_ewt_case_counts.json, estimator=add-k-Laplace, normalization=sums-to-one-over-cases)

## Equations

# Predictive belief over the current case role, the posterior pushed through B

pred_t = B * q_{t-1}

# Posterior over the case role after observing the UPOS tag

q_t = normalize(A[o_t, ] * pred_t)

# Reported score over T held-out positions. AGREEMENT with the derived heuristic
# labeling,
# not accuracy: on the derived population the reference is a documented
# heuristic, not ground truth.

AGR = (1/T) * sum_t [ argmax_c q_t[c] == label_t ]

## Time

Time=t
Dynamic
Discrete

# The prior is reset to D at every sentence boundary. The unit of this level is
# the sentence: the first labeled token of a sentence has no predecessor whose
# case role could predict it, and a transition across a sentence boundary is not
# a case-role transition.

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

num_hidden_states: 8
num_obs: 12
num_factors: 1
num_modalities: 1

## Footer

AILM Case Level, L4 of the AILM hierarchy. Implemented by
src/ailm/levels/case.py (build_case_agent, build_case_emission,
build_case_transition, build_case_start). Conformance to this spec is checked
against a LIVE pymdp agent by tests/test_gnn_conformance.py. The mutations under
gnn/mutations/ are the negative controls for this model; each must FAIL
conformance or the gate that admits it.

## Signature

Unsigned. AILM does not sign its GNN specs; provenance is the git history of
this file plus the checksums recorded in docs/provenance/.
