# Connected causal runtime

The connected runtime was implemented as a multirate composition of the shipped
GNN-specified pymdp agents. The implementation is
[`ConnectedRuntime`](../../src/ailm/levels/connected.py), its typed input and trace
contract is in [`connected_types.py`](../../src/ailm/levels/connected_types.py), and
[`load_connected_model`](../../src/ailm/levels/connected_builder.py) assembles the
agents and train-only connection kernels. The existing stage measurements remain
the independent experiments described in the [governing hierarchy](../hierarchy.md).
No stage score or claim-ledger value is reassigned to this new runtime.

The rejected alternative was to concatenate the existing intervention scores and
describe them as an integrated model. Those scores do not specify a joint
observation stream, clocks or feedback. A monolithic exact joint model was also
not adopted: the present component transitions and posterior-message operations
do not establish such a joint model. The cost is explicit: this runtime performs
causal modular conditional filtering, **not globally exact Bayesian inference**.
Its parameter settings are engineering choices, without a held-back optimization
or generalization claim.

## Available observations

Raw text becomes lowercase ASCII letter runs. Punctuation closes a sentence but
its identity is withheld; apostrophes, hyphens, digits and non-ASCII characters
delimit lexical runs. The scored character stream consists of each retained run
followed by a space. It is therefore a normalized lexical stream, not the original
byte stream or a Unicode language model.

Structured input supplies a sequence of sentences containing `Token(form, upos)`.
The form uses the same normalized letter alphabet. Optional UPOS is explicitly an
external observation available **before** the current word. This conditional task
must not be compared with raw-text prediction as if the information were equal.
The token API has no slot for dependency labels, morphological case, gold phones
or sentence type; the JSON interface rejects extra token fields.

Raw mode does not look up the current word's tag before predicting it. After the
word is observed, its live lexical posterior is mapped through a train-fitted
UPOS kernel. That distribution is an uncertain lexical observation encoding,
not annotated syntax or a gold label. The fixed encoder remains present in the
channel-disconnected arm, which is consequently a **channel ablation**, not a
claim that every agent is isolated from the input representation.

## Clocks, resets and order

| Component | Clock | Reset | Inputs available at update |
| --- | --- | --- | --- |
| L6 phase | completed sentence | document | inferred L5 type distribution |
| L5 widened clause | completed token | sentence | L4 case posterior and observed/inferred UPOS |
| L4 case | token | sentence | previous-state prediction, L5 prediction, optional current external UPOS; raw UPOS only after spelling |
| L3 word | character, including END | word | case-derived initial word prior and observed prefix |
| L2B phone | character | word | previous phone prediction, L3 prefix predictive, current letter |
| L1 letter | normalized character and token space | document | previous-letter prediction, L3 prefix predictive, current character |

At sentence opening, L6's predictive type distribution modifies L5's type prior,
preserving L5's conditional first-symbol distributions. Before a token, L5's
predicted case-symbol marginal modifies the temporal L4 prior. In structured
mode L4 consumes the supplied UPOS now; in raw mode its preword belief contains
only the previous observations. The resulting role distribution modifies L3's
initial word prior. This distribution is retained before any current letters
and supplies the proper word-probability score.

Before each current letter, L3's prefix belief supplies the predictive letter
and phone messages. The current letter then updates the live L1, L2B and L3
agents. L3's position transition advances only after that observation. The END
observation closes the word. Raw mode now obtains its lexical UPOS distribution
and updates L4. The widened case/VERB/AUX distribution updates L5, whose temporal
transition predicts the next token. On sentence completion, the L5 type
distribution updates L6 and its temporal transition predicts the next sentence.
Each call stores all mutable beliefs locally, so processing another document
cannot retain a preceding document's state.

`update_empirical_prior` performs only these within-agent temporal transitions.
Cross-agent messages are computed and inserted explicitly by `ConnectedRuntime`.
No future word, future document length, current punctuation-type label or gold
case label enters this schedule.

## Connections and numerical meaning

| Channel | Sender | Receiver and insertion |
| --- | --- | --- |
| `phase_clause` | predictive L6 type law | L5 joint initial prior |
| `clause_case` | L5 predictive case-symbol marginal | L4 predictive prior |
| `case_word` | current available L4 role belief | L3 word initial prior |
| `word_phone` | live L3 prefix belief through pronunciation kernel | L2B predictive prior |
| `word_letter` | live L3 next-symbol predictive | L1 predictive prior |
| `case_clause` | L4 posterior, with UPOS verb/auxiliary mass | L5 categorical observation |
| `clause_phase` | completed-sentence L5 type posterior | L6 categorical observation |

Prior messages use the declared convex mixture
`p_receiver = (1 - strength) p_local + strength message`. The scalar is fixed
before inference and is not selected using evaluation input. Its admissible
range retains positive local support. Each named channel can be disabled or
shuffled independently. Shuffling reverses the receiver message coordinates and
preserves its entropy while corrupting its interpretation.

Disabling an observation channel means **missing evidence**: the posterior is
the prior, followed by the normal temporal transition. Uniform categorical
evidence is not substituted because it is informative under the expected-log
likelihood convention. The event trace represents missing evidence with an
all-zero observation vector. A zero connection strength disconnects all named
channels.

For categorical observation `o` and emission `A`, pymdp uses likelihood
`exp(sum_o o * log(A[o, state]))`, with its numerical floor at structural zeros.
This is a geometric, virtual-evidence update. In particular, transferring an
L4 or L5 posterior is **not** exact marginalization of a globally specified
generative model. The independent test oracle enumerates the conditional joint
over each receiving agent's state factors using the event's incoming prior and
this same declared observation semantics. It compares the live posterior
marginals and pre-observation predictions, including the L3 word-position joint.
That licenses conditional-update agreement only.

## Training, coverage and output

The default loader constructs agents from the existing per-level GNN files via
`build_agent_from_spec`; its allowlisted resolvers call the shipped matrix
builders. The composition schedule is enforced by Python and its tests, not by
the single-agent GNN conformance checker.

The case-word kernel uses the existing train-route counts, falling back to the
lexical unigram for unsupported roles. The raw UPOS encoder uses only the
case-word artifact's training rows, with additive smoothing. The word-phone
kernel uses only the existing G2P training partition; unpronounced words and
positions beyond known pronunciations use the phone start distribution. Those
fallbacks are fixed model definitions and do not add words from evaluation data.

The bounded lexicon assigns an OOV word zero probability. `DocumentResult`
retains the token and reports infinite full-document word loss. Character and
phone inference continue; after the prefix leaves lexical support, the lexical
channels stop contributing. The raw UPOS encoder uses its training marginal for
an unrecognized word. Reports of in-vocabulary loss must disclose coverage and
must not conceal the infinite full-stream closed-vocabulary loss.

`DocumentResult.events` records the clock, incoming priors, observation,
pre-observation prediction and returned live posterior. Word scores and
letter scores are distinct estimands. Phone posteriors are position-aligned
G2P outputs, not a pronunciation-length or speech model.

## Recorded connected evaluation

The locally frozen GUM comparison is recorded separately from the implementation
controls and historical stage gates. Its [protocol and manifest](../provenance/sources.md#ud-english-gum-connected-evaluation-input)
fix raw-text observations, development-document exclusions, channel strength,
paired estimand and document bootstrap. The [results](../manuscript/04_results.md#sec:connected_gum)
report the existing character-loss gain, vocabulary coverage and infinite
full-stream closed-vocabulary word loss. This is a development-session
prospective comparison, not external preregistration or independent replication.

The comparator disables every named channel while retaining the same spelling
stream and train-fitted lexical UPOS encoder. Its paired gain therefore concerns
that aggregate intervention; it cannot attribute a benefit to an individual
channel. Structured external UPOS is not used in that raw-text comparison.
`connected_provenance.canonical_execution_binding` binds the complete execution
inputs, while the recorder and figure reader also authenticate the frozen
population and report. A changed model, observation mode or population requires
new evaluation; updating a digest alone does not create that evidence.

## Active generation and invocation

`generate_next_word(result)` uses the completed document's phase prediction to
open a fresh sentence and derive a connected word prior. The controllable L1g
agent receives the live L3 next-letter distribution as log preferences and the
Brown bigram as policy habit. It calls `infer_policies` and `sample_action`, then
feeds the selected letter into live L3 inference before selecting again. Space
terminates the word. The returned `terminated` flag distinguishes termination
from the explicit length bound. This is next-sentence, one-word conditional
generation, not autonomous document planning. The lexical policy preference is
the generation mechanism; `word_letter` controls the recognition prior inlet.

```python
from ailm.levels.connected import ConnectedRuntime
from ailm.levels.connected_builder import load_connected_model

runtime = ConnectedRuntime(load_connected_model())
result = runtime.infer_text("The man. A man?")
generated = runtime.generate_next_word(result)
```

```bash
.venv/bin/python scripts/run_connected_ailm.py --text "The man." --generate
.venv/bin/python scripts/run_connected_ailm.py --structured document.json --trace
.venv/bin/python -m pytest tests/test_connected.py -q --no-cov
```

The structured file is a JSON array of sentences, each an array of objects with
`form` and optional `upos`. CLI output is strict JSON: infinite word loss is
represented by `null` together with `word_loss_status: "infinite_oov"`.

## Falsification and limits

[`test_connected.py`](../../tests/test_connected.py) exercises live conditional
inference against independent joint enumeration, a corrupted-posterior negative
control, channel reversal and disconnection, document/sentence/word resets,
pre-observation scoring, withheld punctuation-type and label fields, OOV
continuation, and live policy selection against its independent exact action law.
These are implementation and development controls, not the corpus comparison
itself. The separately recorded GUM evaluation above provides population-specific
evidence under its frozen contract. A further comparison must specify identities,
observation mode, populations, ablations, coverage and document-level uncertainty
before scoring, with harmful or null outcomes retained. Conditional references
and the current GUM result do not establish a globally exact posterior, general
semantic generation or an end-to-end speech system.

## Frozen source-comment boundary

The original measured source inventory includes `levels/widened.py` module
comments written before decision 114. Their references to a shipped case-only
arm describe that historical source epoch, not the current canonical model.
The canonical widened arm and retained case-only ablation are distinguished
by this runtime contract, [the hierarchy](../hierarchy.md), and
[the notation reference](../reference/notation.md). Preserve the measured
module’s exact bytes when reproducing its bound report; editing even a
docstring invalidates that source binding. This is retained provenance, not
recommended current method guidance.
