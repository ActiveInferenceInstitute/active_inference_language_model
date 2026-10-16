# AILM L2B — Position-Aligned Letter-to-Phone Level

# GNN Version: 1.0

# 30 letter observations over 39 hidden phone states — a genuine emission

## GNNSection

AILMLetterToPhoneLevel

## GNNVersionAndFlags

GNN v1

## ModelName

AILM Letter To Phone Level

## ModelAnnotation

Part B of level 2, and the first AILM level whose recognition matrix is a REAL
emission rather than the identity. This is the crux of the level and the reason
it exists:

- Hidden factor: the current PHONE, 39 states (the canonical ARPAbet inventory,
  claim:phoneme_count_39 phones, stress stripped).
- Observation modality: the current LETTER, 30 observations (a-z, space, period,
  apostrophe, hyphen).
- A is 30 by 39. A[letter, phone] = P(letter | phone). The observed alphabet and
  the hidden alphabet are DIFFERENT alphabets of different sizes, so unlike every
  other AILM level the recognition matrix is not square and cannot be confused
  with a transition by shape alone. It is also the whole of what joins this level
  to the one below: the observed alphabet IS level 1's hidden alphabet. That is a
  type identity fixed at build time, not a message — nothing flows along it.
- B is 39 by 39: the phone bigram transition, trained on the TRAIN split ONLY.
  The full-corpus phone bigrams in the vendored artifact cover the held-out test
  words as well, so using them would leak test statistics into the decoder.
- D is the distribution over the FIRST phone of each training word — not the
  overall phone unigram, and not uniform.
- C is flat, and is indexed by the 30 letters (the observation modality), not by
  the 39 phones.
- The task is equal-length position-aligned letter-to-phone decoding, NOT
  grapheme-to-phoneme conversion. A is estimated over letter/phone pairs that
  share an index in a word whose letter and phone counts happen to coincide, and
  that index pairing is a stipulation of the evaluation rather than a recovered
  alignment. `use` = `Y UW S` is counted as u->Y, s->UW, e->S, and all three are
  wrong. A is therefore a statistical relation between co-indexed symbols, and no
  figure derived from it is evidence of a learned phonographic relationship. The
  filename and the `g2p` identifiers throughout the implementation predate the
  correct name and are kept because tests and the ledger resolve them as keys.

Read the ontology block below as the load-bearing part of this file. A is
declared RecognitionMatrix and B TransitionMatrix, and ModelParameters fixes
num_obs at 30 against num_hidden_states at 39. Those three declarations together
reject the inverted model — phones observed, letters hidden — which is a
constructible, pymdp-valid agent that is simply the wrong direction of the map.

Why a shape check is not enough elsewhere but bites here. On the letter and
phoneme-sequence levels the correct and the swapped wirings have identical
shapes, so those specs rely on the structural forms in
InitialParameterization. Here the asymmetry of 30 by 39 makes the transposed
emission a dimension violation as well, which is why this level is the easiest of
the three to pin and the hardest to get quietly wrong.

Gate. Stage 2 Part B is gated on beating the per-letter argmax bar — a
context-free lookup table with no transition matrix and no inference at all,
which reaches claim:stage2_g2p_baseline_accuracy percent on the held-out split.
Forward filtering reaches claim:stage2_g2p_accuracy percent, a margin of
claim:stage2_g2p_margin_pts points over claim:stage2_g2p_eval_positions positions.
A margin that thin cannot be established by comparing two aggregate accuracies,
so the gate pairs the two decoders position-for-position and runs McNemar on the
discordant pairs (p = claim:stage2_g2p_mcnemar_p).

Beating that bar is NECESSARY BUT NOT SUFFICIENT, and this is the single most
important caveat attached to this level. A mis-normalized emission — normalized
over phones instead of letters, then renormalized so pymdp still validates it —
measured ABOVE the context-free baseline on a held-out subsample. It cleared the
bar while being wrong. The measured figures for that and two other mis-wirings
are recorded in the module docstring of src/ailm/levels/phoneme.py, which is the
single place they are reported. That is precisely the gap this GNN spec closes:
the ontology names the slot, and the declared 30-by-39 orientation is checked
against the live agent rather than inferred from a score.

Learned entries are a derived artifact, not part of this specification. The
emission, the transition and the start prior are all reproduced byte-identically
from the vendored G2P pairs by their generator, with the train/test split defined
by index stride inside the artifact; InitialParameterization names the generator,
the artifact and the split rather than inlining any matrix.

## StateSpaceBlock

# Generative model parameters

A[30,39,type=float]      # Recognition matrix, P(letter | phone) — 30 obs by 39 states
B[39,39,1,type=float]    # Transition, P(next phone | current phone), 1 control
C[30,1,type=float]       # Log-preferences over observed LETTERS, flat here
D[39,1,type=float]       # Prior over the first phone of a word

# Hidden states

s_t[39,1,type=float]     # Belief over the current phone at time t
s_prime[39,1,type=float] # Belief over the next phone

# Observation

o_t[30,1,type=int]       # Observed letter index at time t

# Time

t[1,type=int]            # Discrete letter position within a word

## Connections

D>s_t
s_t-A
A-o_t
s_t-B
B>s_prime
C-o_t

## InitialParameterization

# A is a genuine learned emission over a DIFFERENT alphabet than the state
# space: 30 letters observed, 39 phones hidden. Normalized over LETTERS, so
# A[, p] sums to 1 for every phone p. Normalizing over phones instead and then
# renormalizing produces a matrix pymdp still accepts and that still beats the
# context-free baseline, which is why this orientation is declared here and
# checked, not assumed.

A=Learned(generator=scripts/generate_phoneme_counts.py, artifact=vendor/cmudict_counts.json, key=g2p_pairs, split=train, alignment=length-matched-1to1, estimator=add-1-Laplace, normalization=column-stochastic-over-letters)

# B is the phone bigram transition computed from the TRAIN split only, never
# from the artifact's full-corpus phone_bigrams, which would leak the held-out
# test words into the decoder.

B=Learned(generator=scripts/generate_phoneme_counts.py, artifact=vendor/cmudict_counts.json, key=g2p_pairs, split=train, estimator=add-1-Laplace, normalization=column-stochastic-over-next-phone)

# C is flat — no preference over observed letters at this level.

C=Zeros(30)

# D is the distribution over the first phone of each TRAINING word. Not the
# overall phone unigram and not uniform.

D=Learned(generator=scripts/generate_phoneme_counts.py, artifact=vendor/cmudict_counts.json, key=g2p_pairs, split=train, statistic=first-phone-of-word, estimator=add-1-Laplace)

## Equations

# Predictive belief over the current phone, the previous posterior through B

pred_t = B * q_{t-1}

# Posterior over the hidden phone given the observed letter — filtering, so
# position t sees letters 1 through t only and never the letters after it

q_t = normalize(A[o_t, ] * pred_t)

# Decoded phone at position t

phone_t = argmax_p q_t[p]

# Position-level accuracy against the reference pronunciation

ACC = (1/N) * sum_w sum_t indicator(phone_wt == reference_wt)

## Time

Time=t
Dynamic
Discrete

# The prior is reset to D at the start of EVERY word, and decoding is FILTERING
# rather than smoothing: position t conditions on letters 1 through t only.
# Smoothing and Viterbi both score higher on this task and are documented as
# reachable ceilings, not as what this model does.

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
num_obs: 30
num_factors: 1
num_modalities: 1
combination_rule: linear

## Footer

AILM Position-Aligned Letter-to-Phone Level, L2 Part B. Implemented by
src/ailm/levels/phoneme.py (build_g2p_emission, build_g2p_transition,
build_g2p_start, build_g2p_agent). Conformance to this spec is checked against a
LIVE pymdp agent by tests/test_gnn_conformance.py, which carries the positive
control proving an inverted 39-by-30 emission is rejected.

## Signature

Unsigned. AILM does not sign its GNN specs; provenance is the git history of
this file plus the checksums recorded in docs/provenance/.
