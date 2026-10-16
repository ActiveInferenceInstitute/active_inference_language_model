# AILM L3 — Word Level — MUTATION @transposed_factors

# GNN Version: 1.0

# Two-factor, single-modality pymdp agent over a bounded lexicon

## GNNSection

AILMWordLevel

## GNNVersionAndFlags

GNN v1

## ModelName

AILM Word Level

## ModelAnnotation

DELIBERATE MUTATION of L3_word. It exists to be REJECTED.

Declares the emission's two state axes in the wrong order. Both orderings
have the same element count, so only a factor-order check catches it.

A mutation that passes conformance means the check does not constrain what it
claims to. tests/test_gnn_conformance.py asserts every mutation here fails.

Level 3 of the AILM hierarchy: a closed-vocabulary word model whose hidden state
factorizes into word identity and position within the word.

- Hidden factor 0: word identity, claim:stage3_vocab_size states.
- Hidden factor 1: position within the word, one state per letter slot plus a
  terminal slot.
- Observation modality: the current letter, 27 observations (a-z plus an
  end-of-word symbol).
- A depends on BOTH factors: A[letter, word, position] is 1 exactly when that
  letter is the one word w carries at that position, or the end symbol once the
  position is past the word's last letter. It is deterministic, and it is the
  only place the spelling of the lexicon is written down.
- B_f0 is the IDENTITY over word identity. A word does not turn into another
  word part-way through being spelled.
- B_f1 is the +1 shift over position, absorbing at the terminal slot.
- D_f0 carries the learned Brown word unigram frequencies.
- D_f1 is a point mass on position 0.
- C is flat. This level expresses no preferences and selects no actions.

Where the learned statistics live, and why that differs from L1 and L2. At every
level below this one the learned sequence dynamics parameterize the TRANSITION.
Here both transitions are structural constants and the single learned quantity is
the PRIOR D_f0. That is the level's whole content — a lexicon with frequencies —
and it relocates the mis-wiring risk accordingly: a wrong D_f0 at this level is
what a wrong B was at Level 1.

The relocation matters because the accuracy bar does not catch it. Three of the
five mis-wirings enumerated for this level still beat the letter-bigram bar,
including one that inverts the frequency ranking so the rarest words are
predicted most confidently: it scores claim:stage3_miswire_reversed_prior against
a bar of claim:stage3_bigram_baseline. What pins the level down is the
closed-form path identity — the probability the model assigns to a word's full
emission path must equal that word's unigram probability, to
claim:stage3_path_identity_max_abs_err — together with the per-axis structural
declarations below. See docs/decisions.md entries 13 and 14.

Why this is not a trie over states. The design document specifies words as paths
through a letter-phoneme trie. A dense pymdp transition over trie nodes is
claim:stage3_trie_dense_gb GB at the G2P vocabulary, and pymdp has no sparse
path. The trie is not abandoned: after observing a prefix, the posterior over
factor 0 is supported exactly on the vocabulary words carrying that prefix, which
IS the trie node. This factorization is the same model in cheaper coordinates,
and the correspondence is checked against a live agent rather than asserted.

Closed vocabulary. The model assigns NO probability to a word outside its
lexicon — not a small one. claim:stage3_coverage_pct% of held-out alphabetic
Brown tokens are in vocabulary, and every Stage 3 number describes that
sublanguage.

Gate. Stage 3 is gated on beating a letter-bigram model over the identical
alphabet, trained on the identical tokens and scored on the identical positions
(claim:stage3_bigram_baseline against claim:stage3_word_perplexity). The declared
"perplexity < 500" target was VACUOUS: the bar itself clears it by a factor of
claim:stage3_declared_gate_slack.

Learned entries are a derived artifact, not part of this specification. The word
frequencies are reproduced byte-identically from the vendored artifact by the
generator named in InitialParameterization.

## StateSpaceBlock

# Generative model parameters

A[27,21,2000,type=float]   # Recognition, factors declared in the wrong order
B_f0[2000,2000,1,type=float] # Transition over word identity: the identity map
B_f1[21,21,1,type=float]   # Transition over position: the +1 shift
C[27,1,type=float]         # Log-preferences over observed letters, flat here
D_f0[2000,1,type=float]    # Prior over word identity: learned unigram frequencies
D_f1[21,1,type=float]      # Prior over position: point mass on 0

# Hidden states

s_word[2000,1,type=float]  # Belief over word identity at time t
s_pos[21,1,type=float]     # Belief over position within the word at time t

# Observation

o_t[27,1,type=int]         # Observed letter index at time t, 26 is end-of-word

# Time

t[1,type=int]              # Discrete position within the word

## Connections

D_f0>s_word
D_f1>s_pos
s_word-A
s_pos-A
A-o_t
s_word-B_f0
s_pos-B_f1
C-o_t

## InitialParameterization

# A is deterministic and depends on both factors. It is not "learned" in the
# estimator sense — it is the spelling of the lexicon, read off the vocabulary
# without smoothing or counting — but it IS derived from the vendored artifact,
# so the generator is named here rather than the table inlined.

A=Learned(generator=scripts/generate_word_counts.py, artifact=vendor/brown_word_counts.json, estimator=deterministic-spelling, normalization=column-stochastic-over-observations)

# B_f0 is the identity: a word does not become a different word mid-spelling.
# This is a structural claim, and any other matrix here breaks the path identity
# that pins the level.

B_f0=Identity(2000)

# B_f1 is the +1 shift, absorbing at the terminal slot. Declared Learned rather
# than Identity because it is emphatically NOT the identity: the identity here
# would freeze the model at position 0, emitting the first letter forever, and
# the transpose is the time-reversed model that spells words backwards.

B_f1=Learned(generator=src/ailm/levels/word.py, artifact=build_position_transition, estimator=structural-shift, normalization=column-stochastic-over-next-position)

# C is flat — no preference over observations at this level.

C=Zeros(27)

# D_f0 holds the learned Brown word unigram frequencies. This is the ONLY
# learned probability in the level, and the one the accuracy bar fails to pin.

D_f0=Learned(generator=scripts/generate_word_counts.py, artifact=vendor/brown_word_counts.json, estimator=add-k-Laplace, normalization=sums-to-one-over-vocabulary)

# D_f1 is a point mass on position 0 — every word starts at its first letter.

D_f1=Learned(generator=src/ailm/levels/word.py, artifact=build_word_agent, estimator=structural-point-mass, normalization=sums-to-one-over-positions)

## Equations

# Predictive belief over each factor, the posteriors pushed through their own B

pred_word_t = B_f0 * q_word_{t-1}
pred_pos_t = B_f1 * q_pos_{t-1}

# Predictive likelihood of the letter actually observed, over the joint

P(o_t) = sum_w sum_p A[o_t, w, p] * pred_word_t[w] * pred_pos_t[p]

# Posterior over word identity after observing the letter

q_word_t = normalize(sum_p A[o_t, , p] * pred_word_t * pred_pos_t[p])

# Path probability of a whole word, which collapses because every factor except
# D_f0 is deterministic. This identity is what pins the level.

P(path of w) = D_f0[w]

# Per-emission perplexity over T scored emissions, where an emission is a letter
# or the end-of-word symbol

PP = exp(-(1/T) * sum_t ln P(o_t))

## Time

Time=t
Dynamic
Discrete

# The prior is reset to D at every word boundary. The unit of this level is the
# word, and the first letter of a word has no predecessor within the model: the
# position factor returns to 0 and the word factor returns to the unigram
# distribution. This matches L2's per-word reset and differs from L1, which
# models running text as one stream.

## ActInfOntologyAnnotation

A=RecognitionMatrix
B_f0=TransitionMatrix
B_f1=TransitionMatrix
C=LogPreferenceVector
D_f0=Prior
D_f1=Prior
s_word=HiddenState
s_pos=HiddenState
o_t=Observation
t=Time

## ModelParameters

num_hidden_states_f0: 2000
num_hidden_states_f1: 21
num_obs: 27
num_factors: 2
num_modalities: 1

## Footer

AILM Word Level, L3 of the AILM hierarchy. Implemented by
src/ailm/levels/word.py (build_word_agent, build_emission, build_word_prior,
build_word_transition, build_position_transition). Conformance to this spec is
checked against a LIVE pymdp agent by tests/test_gnn_conformance.py. The
mutations under gnn/mutations/ are the negative controls for this model; each
must FAIL conformance or the gate that admits it.

## Signature

Unsigned. AILM does not sign its GNN specs; provenance is the git history of
this file plus the checksums recorded in docs/provenance/.
