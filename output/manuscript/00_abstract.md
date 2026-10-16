# Abstract {#sec:abstract}

Language offers a testbed for cognitive models that connect uncertain perception,
structured representations and goal-directed action. AILM (Active Inference
Language Model) explores this connection through explicit discrete generative
models implemented in `pymdp`. Its six linguistic levels
represent letters, phonemes, words, case roles, sentence types and document
position. Conditional probabilities, preferences, habits and policy precision
have declared meanings and inspectable parameter sources.

AILM provides a bounded alternative to learned embedding representations for
studying specified linguistic mechanisms, and a potential complement to the
broader predictive capabilities of large language models. Its generative core
uses no transformer, neural network or learned embedding. Generalized Notation
Notation (GNN) specifies the models; `pymdp` executes their inference and policy
updates. A connected runtime exchanges conditional messages and supports
next-word policy selection. This composition does not establish a globally
exact posterior, and the letter generator's expected-free-energy rule reduces
to a preference-weighted softmax over the habit. These explicit mechanisms are
candidates for cognitive investigation, not evidence of human-like understanding.

On a locally frozen Georgetown University Multilayer (GUM) corpus population,
the connected comparison indicates
lower integrated loss than disabling the named channels while retaining
the same train-fitted observation encoder. This result concerns normalized
letters and token spaces; full-stream closed-vocabulary word loss remains
infinite because out-of-vocabulary tokens receive zero word probability. The comparison is local evidence for a fixed
composition, not external replication or unrestricted language competence.

Separate component and intervention studies show that linguistic messages can
help or harm a receiver; different insertion sites constrain their attribution.
The accompanying gate-audit method compares thresholds with trivial baselines,
typed comparators and deliberately broken variants, and binds numerical results
to recorded runs through a claims ledger. Historical gates were largely repaired
after construction, and held-back findings retain their stated task and
uncertainty limits. The contribution is an inspectable platform for testing
linguistic inference mechanisms together with an audit method that makes their
failures visible. External neural models serve as supplemental reference probes;
integration with them remains a proposal.

The planned release repository is
<https://github.com/ActiveInferenceInstitute/Active_Inference_Language_Model>.
