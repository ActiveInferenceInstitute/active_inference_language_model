# The Generation Problem

## What the Papers Provide (and Don't)

Both source deposits are **comprehension** papers. Their identities require the
same two-layer reading used throughout AILM: the Zenodo record calls CEREBRUM
version 1.4, while its exact deposited PDF prints version 1.2; the other Zenodo
record is titled *Compositional Approaches to Linguistic Case for Cognitive
Modeling* version `v1`, while its exact deposited PDF prints *Cognitive Diagrams:
Reviewing Categorical Accounts of Linguistic Case* and no version. Exact record
and file identities are pinned in `data/reference_sources.json`. The papers
describe how a model assigns case roles to observed inputs, how belief updating
propagates through a case-structured grammar, and how the enriched categorical
structure constrains interpretation.

Neither paper provides a generation mechanism. There is no description of how
CEREBRUM or its categorical companion would *produce* a sentence from a goal state.

AILM's implemented contribution is deliberately narrow: L3 lexical preferences
drive a controllable L1g letter agent. The standalone generation experiment and
the [connected next-word API](connected_runtime.md#active-generation-and-invocation)
use that mechanism under different prior and execution contracts; their evidence
must be kept separate.

## The measured standalone policy generator

In standard active inference, an agent selects a policy π (a sequence of actions)
by minimizing the expected free energy G(π):

```
G(π) = sum_t [ KL[q(o_t|π) || p(o_t|C)] + E_q[H[p(o_t|s_t,π)]] ]
```

where `C` encodes the agent's preferences over observations.

AILM's shipped generator interprets **generation** as policy selection at one
level:

- **Policies/actions** are next-letter commitments in the controllable L1g agent.
- **Preferences (`C`)** are the L3 lexicon's distribution over the next letter,
  conditional on the emitted prefix.
- **Policy habits (`E`)** are Brown-bigram next-letter probabilities.
- The closed-form policy posterior (`efe_policy_posterior`, verified against
  `agent.infer_policies` with the agreement recorded) combines the lexical
  preference and habit; the selected action deterministically sets the next
  letter through L1g's action-indexed `B`.

## Exact distributional scoring

The generation gate does not estimate its comparator from a finite sample. A
word's ancestral probability is the product of the Brown-transition entries on
its path from the boundary space through its letters and back to space. The
policy generator has the same finite-path calculation with each Brown column
replaced by the prefix-conditioned policy posterior. A path reaching the forced
word-length horizon is emitted without an additional terminator, exactly as in
the sampler.

Both calculations enumerate the shipped finite lexicon and return one additional
out-of-lexicon remainder. They also condition out an immediate boundary-space
draw, because `GenerationResult` retains non-empty word tokens rather than
requested attempts. Thus the baseline, policy model, and improvement use the
same probability space. Seeded samples remain useful as implementation controls
and for realized-run diversity, but sampled TV values are diagnostics rather
than gate values.

## Historical seams and connected next-word generation

The historical measurements below evaluated separate interventions. They do not
supply a joint result for the newer connected runtime:

```text
L6 document position → L5 clause D     independent prior intervention
punctuation-type oracle → L3 word D[0] independent prior intervention
L3 lexicon           → L1g letter C   the implemented generator
```

The first seam blends `P(type | tercile)` into L5's joint initial prior. The
second uses held-out punctuation type directly to select a typed word
distribution and blends that privileged-oracle message into L3's word prior; it
does not consume an L5 posterior. The third is the only seam in the generation
run. Their published effects remain attached to those individual experiments.

`ConnectedRuntime.generate_next_word` now starts a new sentence from the inferred
document-phase prediction, forms clause/case-conditioned word priors and calls
live L1g policy inference and action selection while L3 consumes generated letters.
It stops at a word boundary or an explicit length limit. The returned termination
flag must be read; the limit is not evidence that a complete word was generated.
This is a bounded lexical continuation, not autonomous semantic document planning.
The [runtime contract](connected_runtime.md) owns its observation and timing rules;
its GUM character-prediction comparison is not a generation-quality benchmark.

## Why This Works Without Embeddings

In standard language models, generation requires a learned distribution over
the vocabulary. In AILM:
- The B matrix at each level encodes transition probabilities between discrete states
- The C vector encodes log-preferences over observations
- Policy selection via G(π) produces a distribution over next-symbols
- The selected symbol becomes the next observation, updating beliefs via A

No continuous embedding space is required. Everything is discrete probability
distributions and stochastic maps.

## What This Means for Traceability

The measured standalone generator exposes the arrays entering each next-letter choice:
the current lexical `C`, Brown-bigram habit `E`, action-indexed transition `B`,
and resulting policy posterior. That is parameter-level traceability for this
mechanism. It is not evidence that clause, case, phoneme, and narrative causes
were involved: they are absent from the generation run.

## Status

L3→L1g policy selection is implemented and measured. L6→L5 is implemented and
measured as a model-output `D`-prior intervention. The separate typed word-prior
experiment uses a privileged punctuation-type oracle as its source, not L5
output. The connected conditional next-word path is shipped separately, with
live-policy controls described in [connected_runtime.md](connected_runtime.md).
Neither the historical generation result nor the connected character-loss result
establishes sentence- or document-generation quality.
