# AILM L2A — Phoneme Sequence Level

# GNN Version: 1.0

# Single-factor, single-modality pymdp agent over 39 canonical ARPAbet phones

## GNNSection

AILMPhonemeSequenceLevel

## GNNVersionAndFlags

GNN v1

## ModelName

AILM Phoneme Sequence Level

## ModelAnnotation

Part A of level 2: the letter-level model one level down, over the
claim:phoneme_count_39 canonical ARPAbet phones with stress digits stripped.

- Hidden factor: the current phone, 39 states.
- Observation modality: the current phone, 39 observations.
- A is the identity. A phone sequence is observed directly here; the genuine
  letter/phone emission lives in the sibling G2P spec (gnn/ailm_g2p.md).
- B is the CMUdict phone bigram transition, B[next, current].
- D is the phone unigram distribution, learned rather than uniform: it is the
  price paid for the first phone of every word, which has no predecessor.
- C is flat. No preferences, no action selection.

Where the learned statistics live. Sequence dynamics go in the TRANSITION; the
emission is A. The same trap the letter level documents is available here in
identical form, and with identical shapes: a bigram table in A with an identity B
yields a posterior over the PRECEDING phone. Shapes cannot distinguish the two
wirings, so the ActInfOntologyAnnotation and InitialParameterization blocks below
are what pin the orientation.

Gate. Stage 2 Part A is gated on beating the CONTEXT-FREE phone unigram bar
(claim:stage2_phoneme_unigram_baseline); the agent reaches
claim:stage2_phoneme_perplexity. Uniform over 39 phones
(claim:stage2_phoneme_uniform_baseline) is a crippled comparator and is not the
gate. The order-1 conditional-entropy floor is
claim:stage2_phoneme_floor_perplexity, so the total headroom available over
uniform at this order is claim:stage2_phoneme_max_gain_nats nats — under one nat.
Any gate phrased as "N nats below uniform" for N of 2 or more is unreachable at
this level regardless of implementation quality.

Learned entries are a derived artifact, not part of this specification. The
bigram table and the unigram prior are reproduced byte-identically from the
vendored counts by their generator; InitialParameterization names the generator
and the artifact rather than inlining the matrices.

## StateSpaceBlock

# Generative model parameters

A[39,39,type=float]      # Recognition matrix, P(observed phone | phone state)
B[39,39,1,type=float]    # Transition, P(next phone | current phone), 1 control
C[39,1,type=float]       # Log-preferences over observed phones, flat here
D[39,1,type=float]       # Prior over the first phone of a word

# Hidden states

s_t[39,1,type=float]     # Belief over the current phone at time t
s_prime[39,1,type=float] # Belief over the next phone

# Observation

o_t[39,1,type=int]       # Observed phone index at time t

# Time

t[1,type=int]            # Discrete phone position within a word

## Connections

D>s_t
s_t-A
A-o_t
s_t-B
B>s_prime
C-o_t

## InitialParameterization

# A is the identity: the observed phone IS the hidden phone. Structural, not
# learned.

A=Identity(39)

# B holds the learned CMUdict phone-bigram statistics, regenerated
# byte-identically by the generator named here. Deliberately not inlined.

B=Learned(generator=scripts/generate_phoneme_counts.py, artifact=vendor/cmudict_counts.json, key=phone_bigrams, estimator=add-1-Laplace, normalization=column-stochastic-over-next-phone)

# C is flat — no preference over observations at this level.

C=Zeros(39)

# D is the learned phone unigram distribution, NOT uniform. Position 0 of every
# word has no predecessor, so it is scored under this prior.

D=Learned(generator=scripts/generate_phoneme_counts.py, artifact=vendor/cmudict_counts.json, key=phone_unigrams, estimator=add-1-Laplace)

## Equations

# Predictive belief over the current phone, the posterior pushed through B

pred_t = B * q_{t-1}

# Predictive likelihood of the phone actually observed

P(o_t) = sum_j A[o_t, j] * pred_t[j]

# Posterior over the current phone after observing it

q_t = normalize(A[o_t, ] * pred_t)

# Per-phone perplexity, accumulated over all positions of all sequences

PP = exp(-(1/N) * sum_w sum_t ln P(o_wt))

## Time

Time=t
Dynamic
Discrete

# The prior is reset to D at the start of EVERY word: a word boundary is not a
# phone transition. Measured, treating the dictionary as one continuous phone
# stream instead scores worse than paying the unigram price at each boundary, so
# the reset is correct rather than merely convenient.

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

num_hidden_states: 39
num_obs: 39
num_factors: 1
num_modalities: 1

## Footer

AILM Phoneme Sequence Level, L2 Part A. Implemented by
src/ailm/levels/phoneme.py (build_phone_transition, build_phone_unigram,
build_phoneme_agent). Conformance to this spec is checked against a LIVE pymdp
agent by tests/test_gnn_conformance.py.

## Signature

Unsigned. AILM does not sign its GNN specs; provenance is the git history of
this file plus the checksums recorded in docs/provenance/.
