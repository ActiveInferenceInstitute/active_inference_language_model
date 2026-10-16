# AILM L1g — Letter Level, Generative — MUTATION @habit_as_transition

# GNN Version: 1.0

# Single-factor pymdp agent with a CONTROLLABLE transition, for generation

## GNNSection

AILMLetterGeneration

## GNNVersionAndFlags

GNN v1

## ModelName

AILM Letter Generation

## ModelAnnotation

DELIBERATE MUTATION of L1g_letter_generation. It exists to be REJECTED.

Declares the controllable transition to be the identity, i.e. every action
leaves the letter unchanged. The agent could then never emit anything it had
not already emitted.

A mutation that passes conformance means the check does not constrain what it
claims to. tests/test_generation.py asserts every mutation here fails.

Level 1 of the AILM hierarchy in its GENERATIVE configuration. Same state space
as L1_letter, different control structure, and the only AILM model with a real
policy space.

- Hidden factor: the current letter, 30 states.
- Observation modality: the current letter, 30 observations.
- A is the identity, as in L1_letter.
- B is CONTROLLABLE with 30 actions: B[next, current, k] = 1 iff next == k.
  Action k means "emit letter k", so choosing an action IS choosing the next
  symbol. This is what makes policy selection generation rather than a metaphor
  for it, and it is the difference from L1_letter, whose single trivial action
  left the policy machinery unexercised.
- E, the prior over policies, is the learned Brown bigram column for the current
  letter. Under a controllable B the corpus statistics are the agent's HABIT —
  what it would do absent any goal — rather than its dynamics.
- C, log-preferences over letter observations, is supplied TOP-DOWN by the word
  level from the lexicon posterior given the prefix emitted so far. This is the
  first place in AILM where one level sets another's C, and the first non-flat C
  anywhere in the project.
- D is uniform.

Where the corpus statistics go, and why it differs from L1_letter. At every
recognition level the learned statistics parameterize the TRANSITION. Here the
transition is a structural constant — a deterministic consequence of the chosen
action — and the same bigram table appears as E instead. The table is identical;
its role is not. A spec that annotated it TransitionMatrix here would describe a
model in which the agent cannot choose what to emit.

The gate is NOT membership. The obvious metric — the fraction of generated tokens
that are real words — is green by construction: the preference floors any letter
leaving the lexicon, so membership is claim:generation_best_membership_pct% at
the operating point AND claim:generation_degenerate_membership_pct% at the
degenerate precision that emits a single word forever. A metric that cannot fail
is not a gate. The gate is total-variation distance to the true word
distribution (claim:generation_best_tv against claim:generation_baseline_tv for
ancestral sampling), paired with a diversity floor.

The optimum is INTERIOR. Distance to the true distribution is U-shaped in the
policy precision gamma: at claim:generation_low_gamma the goal is too weak and
the agent reverts toward the habit (claim:generation_low_gamma_tv), and at
claim:generation_degenerate_gamma it collapses onto
claim:generation_degenerate_distinct distinct word(s)
(claim:generation_degenerate_tv). The shipped operating point is
claim:generation_best_gamma. No recognition gate in this project fails in two
opposite directions; this one does.

Learned entries are a derived artifact, not part of this specification.

## StateSpaceBlock

# Generative model parameters

A[30,30,type=float]      # Recognition matrix, identity — the letter is observed
B[30,30,30,type=float]   # Controllable transition, 30 actions: emit letter k
C[30,1,type=float]       # Log-preferences over letters, set by the word level
D[30,1,type=float]       # Prior over the first letter
E[30,1,type=float]       # Policy prior: the learned bigram column, the habit

# Hidden states

s_t[30,1,type=float]     # Belief over the current letter at time t
s_prime[30,1,type=float] # Belief over the next letter

# Observation

o_t[30,1,type=int]       # Observed letter index at time t

# Policy

pi[30,1,type=float]      # Posterior over actions, softmax(gamma * neg_efe + ln E)

# Time

t[1,type=int]            # Discrete letter position

## Connections

D>s_t
s_t-A
A-o_t
s_t-B
B>s_prime
C-o_t
E-pi
pi>B

## InitialParameterization

# A is the identity: the observed letter IS the hidden letter.

A=Identity(30)

# B is a structural constant here, NOT the learned bigram: action k emits letter
# k with certainty. The learned table has moved to E.

B=Identity(30)

# C is set per step by the word level and is therefore NOT a fixed artifact. It
# is declared Zeros because that is its value at construction; the top-down
# channel installs the real preference at each generation step.

C=Zeros(30)

# D is uniform — no information about the first letter.

D=Uniform(30)

# E carries the learned Brown bigram column as the policy prior — the habit.

E=Learned(generator=scripts/generate_counts.py, artifact=vendor/brown_counts.json, estimator=add-1-Laplace, normalization=column-stochastic-over-next-letter)

## Equations

# Top-down preference: marginalize the word posterior over the next letter

C[k] = ln sum_{w in trie(prefix)} D_word[w] * [ spelling(w)[|prefix|] == k ]

# Expected free energy of each action. With a deterministic controllable B and an
# identity A, the epistemic term vanishes: taking action k yields observation k
# with certainty, so nothing remains to be learned about the state.

G(k) = -C[k]

# Policy posterior, combining goal and habit

q(pi = k) = softmax(gamma * (-G(k)) + ln E[k])

# Reported score: total-variation distance between the generated word
# distribution and the lexicon's, all non-words in one bucket

TV = (1/2) * sum_w | p_generated(w) - p_lexicon(w) |

## Time

Time=t
Dynamic
Discrete

# The preference is recomputed at every step from the prefix emitted so far, and
# the prefix resets at each word boundary — the space symbol. The habit column is
# re-selected each step from the letter just emitted.

## ActInfOntologyAnnotation

A=RecognitionMatrix
B=TransitionMatrix
C=LogPreferenceVector
D=Prior
E=PolicyPrior
s_t=HiddenState
s_prime=NextHiddenState
o_t=Observation
pi=PolicyVector
t=Time

## ModelParameters

num_hidden_states: 30
num_obs: 30
num_factors: 1
num_modalities: 1
num_controls: 30

## Footer

AILM Letter Generation, the generative configuration of L1. Implemented by
src/ailm/generation.py (build_generative_letter_agent, lexicon_preference,
efe_policy_posterior, generate_by_policy_selection). The closed-form policy
posterior is checked against pymdp's own infer_policies by
tests/test_generation.py, which also carries the identity anchor proving a flat
C reproduces ancestral sampling character-for-character.

## Signature

Unsigned. AILM does not sign its GNN specs; provenance is the git history of
this file plus the checksums recorded in docs/provenance/.
