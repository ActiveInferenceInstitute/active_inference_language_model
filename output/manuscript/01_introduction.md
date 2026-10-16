# Language and Cognition Through Explicit Active Inference {#sec:introduction}

## Language as inference over structured representations

Language processing presents a cognitive modeling problem: observations arrive
sequentially, their interpretation is uncertain, and expectations at different
linguistic scales can change what a listener or reader infers. Active inference
offers a framework for relating these processes through generative models,
belief updating and policy selection [@friston2017process; @dacosta2020active;
@parr2022active]. Applications to linguistic communication and active listening
already demonstrate how these ideas can organize models of speaking and
listening [@friston2020linguistic; @friston2021listening]. The Active Inference
Language Model (AILM) asks a narrower,
executable question: what can be represented, composed and tested when the
linguistic states and the probabilistic roles of the parameters are explicitly
declared?

The motivation is mechanistic as well as predictive. A cognitive model should
make it possible to specify which expectations enter a perceptual update, which
preferences enter an action choice, and what changes when a message between
representations is removed. AILM supplies a bounded platform for asking those
questions on corpus-derived tasks. Its cognitive vocabulary motivates the
modeling choices; the measurements evaluate prediction and operational label
agreement. They do not validate a model of human reading, embodied meaning or
neural implementation. The distinction matters because active inference's
account of cognition includes purposive interaction with the world, not just
prediction of text [@pezzulo2024generating].

Expectation-based psycholinguistics provides a complementary starting point.
@hale2001probabilistic and @levy2008expectation relate incremental processing
difficulty to the surprisal of incoming language under a probabilistic model.
That tradition supplies a reason to examine predictive distributions at the
point where an observation arrives, rather than only aggregate text scores.
Active inference additionally distinguishes belief updating from policy
selection. These are related modeling questions, but neither a good predictive
score nor a free-energy formulation establishes that a particular model
reproduces human processing. The proposed behavioral and physiological tests
in [@sec:future] would evaluate that bridge directly.

## Explicit linguistic states and their cognitive limits

AILM builds along linguistic distinctions, then makes an explicit modeling
choice about how to discretize each one. Orthography, phonology, the lexicon,
syntactic case role, clause type and position in a discourse supply the
vocabulary for those choices. The case role is a heuristic class derived from
Universal Dependencies (UD) annotation [@marneffe2021universal; @nivre2020ud].
It borrows its vocabulary from the Case-Enabled Reasoning Engine with Bayesian
Representations for Unified Modeling (CEREBRUM) [@friedman2024cerebrum]; it is
not grammatical case in UD's morphosyntactic sense. Case grammar's semantic
relations, morphological case and a model's assigned functional roles must be
distinguished [@fillmore1968case; @blake2001case]. The resulting states are operational definitions,
not an assertion that this particular inventory is uniquely licensed by
linguistic theory. The object is therefore a stack
of agents implemented in `pymdp`, a Python library for discrete-state active
inference [@heins2022pymdp], whose states are enumerated in the vocabulary of the field:
letters, phones, words, case roles, sentence types, document-position terciles.
Each builder exposes named roles: `A` is an emission, `B` is a transition, `C`
is a log-preference over observations, and `D` is a prior over initial states.
That semantic declaration and the parameter source are executable conformance
properties; they are not a claim that every matrix was historically specified
before any estimate was inspected. Each entry arrives one of two ways. It is
fixed by the design — an identity emission, a positional shift, a flat
preference — or it is a smoothed relative frequency, a count of a named event
over a count of its condition ([@sec:architecture] names which is which at every
level).

These divisions are modeling hypotheses, not a consensus architecture of
language. Stratificational linguistics examines relations across linguistic
levels [@lamb1966outline], while constructionist cognitive linguistics examines
learned generalizations linking linguistic form and function
[@goldberg2006constructions]. AILM's named levels and message interfaces make
some such distinctions experimentally accessible, but its present states do
not implement a construction grammar or an embodied account of meaning.

Different corpora support different evaluations: Brown supplies letter and
word statistics [@francis1979brown]; the Carnegie Mellon University Pronouncing
Dictionary (CMUdict) supplies pronunciation entries [@cmudict]; the English Web
Treebank (EWT) supplies universal part-of-speech (UPOS) and dependency annotation
[@silveira2014gold]; and the
Georgetown University Multilayer (GUM) corpus supplies the text population for
the connected comparison [@zeldes2017gum]. Corpus annotation, derived task
labels and the model's observations remain distinct. Their availability does
not establish that every linguistic layer of a corpus was used or evaluated.

That declaration does not live only in a builder's variable names. Every level
is additionally written down in Generalized Notation Notation (GNN), the
Active Inference Institute's text-based specification language for generative
models [@smekal2023gnn]: a `StateSpaceBlock` fixes each hidden factor and
observation modality and its size, and an `ActInfOntologyAnnotation` names
which of `A`, `B`, `C`, `D` a given array is supposed to be. A conformance
checker, `ailm.gnn.assert_conforms`, then reads a level's declaration and its
live `pymdp.Agent` side by side, and `ailm.spec_agent.build_agent_from_spec`
goes further and constructs the agent from the declaration and its registered
parameter artifacts, so independently authored construction paths can be
compared bit for bit. Declaring the role
of every slot this way matters because `pymdp`'s own validator has nothing to
say about it: a column-stochastic matrix is equally well-formed sitting in `A`
or in `B`, so a matrix placed in the wrong slot passes every check the library
runs. Mutation controls plant exactly that swap against a GNN declaration and
require the conformance check to fail, which is what turns the claim into
something falsifiable rather than aspirational ([@sec:architecture]): a matrix
in the wrong slot stops being a slightly different number and becomes a named
violation.

What that buys is structural transparency, and it is worth being exact about
where it stops. Asking which entries produced a given prediction is answered by
naming a column of `B` and a row of `A` and reading the numbers, with no
recovery step in between. The probability and preference arrays have declared roles and axes; policy
habits and precision are explicit settings as well. This applies to generation
as well as recognition. It is not the same as saying that
behavior reduces to those entries: behavior over a trajectory additionally
depends on the recursion that compounds them, on preprocessing, on the coupling
rule where a level receives a message, on the numerical floor, and, in
generation, on the policy computation. Explicitness moves interpretability from
recovering a latent feature to validating a model and its implementation. It
does not remove the validation.

Nor does it follow that the model explains itself. A named matrix does not say why that
state ontology was chosen over another, how its entries compound along a
trajectory, what smoothing did to the rare events, or whether a label such as
*case role* or *document phase* validly measures the construct it borrows its
name from — [@sec:limitations] takes up the last of those, and the transposed
emission of [@sec:verification], which every structural check in the library
accepted, is the standing demonstration that a fully named model is not thereby
a self-checking one. Explicitness moves the interpretability question from
recovery to validation. It does not close it. Accordingly, the conformance tests
validate declared array roles and the inference checks validate specified
recursions; neither establishes that a case inventory or document-position
partition captures a linguistic or cognitive construct. Those require evidence
beyond agreement with the labels used to define the task.

What this costs is capability, and the cost is paid at every level. Discrete
state-space active inference [@friston2017process; @dacosta2020active; @parr2022active],
realized in `pymdp` [@heins2022pymdp], purchases transparency by committing to small,
enumerated inventories and low-order dynamics. A phoneme inventory of
39 is a real inventory; a first-order chain over it is not a
real phonology. AILM models letters, phonemes, words as factored word-and-position
states, case roles, sentence types, and document-position terciles across
six built levels. Throughout, a level is named three
interchangeable ways — its ordinal position (Level 4), the short code used in
channel and score-path notation (L4), and, where a gate applies, the Stage
number that audits it (Stage 2 audits L2B) — and [@tbl:level_stack] gives the
full correspondence.
The letter model uses first-order Markov transitions, a restricted source of
context within the tradition of statistical language prediction
[@shannon1948; @shannon1951]. This manuscript makes no claim of competitiveness
with neural language models, and the comparison is not one AILM is designed to
win. The claim is that a system built this way can be audited to the level of
individual matrix entries, and that auditing it that way turns up defects that a
benchmark number hides.

## A bounded alternative and potential complement to large language models

Large language models (LLMs) based on transformers learn distributed representations that support
broad text prediction and generation [@vaswani2017attention; @brown2020language].
AILM makes a different representational commitment: a state denotes a declared
linguistic category, and each probability array relates named observations and
states. This makes it an alternative for experiments whose question concerns a
specified mechanism, such as the effect of a lexical expectation on phoneme
decoding. The restriction also limits its vocabulary, dynamics and expressive
capacity; an explicit inventory alone provides no general-purpose semantic
representation.

The contrast concerns how model roles are established. Standard next-token
training does not require individual neural parameters to correspond to
specified linguistic categories. Mechanistic interpretability therefore studies
features and circuits recovered from trained networks [@olah2020zoom;
@elhage2021framework]. AILM instead offers *designed semantic addressability*:
state labels, parameter axes and probability roles are assigned at construction
and checked against the implementation. Neither approach by itself proves that
people can understand the model or that its categories explain cognition
[@lipton2018mythos; @doshivelez2017rigorous]. Here, "semantic" refers to declared
model roles, not a demonstrated theory of meaning.

Complementarity is a research direction with existing precedents. A reading
model by @donnarumma2025reading combines LLM text predictions with active
inference for directing eye movements toward informative text. Such a division
of labor suggests that learned predictors and explicit belief-and-policy models
can address different parts of a cognitive task. For AILM, coupling an external
predictor to a declared observation or prior interface would require its own
specification, controls and evaluation ([@sec:future]). That integration is not
implemented or demonstrated here. GNN is the specification layer, `pymdp` is
the computational core, and external LLMs appear as separately evaluated
supplemental probes ([@sec:external_benchmarks]). Their performance does not
establish the behavior of the AILM hierarchy.

## Belief updating, action selection and the scope of each result

Three things travel together under the name active inference, and this
manuscript keeps them apart. The first is Bayesian filtering under a discrete
generative model — the predictive prior pushed through `B`, corrected by the
observation through `A` ([@eq:filtering]). The second is the variational reading
of that update, under which the same recursion minimizes a free energy
functional ([@eq:vfe]). The third is expected free energy, which scores policies
so that an agent can act ([@eq:efe]).

Every comprehension-level Agent built here runs with a flat `C` and a single
action; the generation Agent is a separate controlled variant. Nothing is therefore selected and no expected free energy is computed.
The historical stage score paths differ: only L1's headline calls live
pymdp filtering, and each later level's exact or NumPy score path, with the
live-Agent check that licenses it, is inventoried in [@sec:verification] and
[@tbl:level_stack]. The later connected comparison in [@sec:connected_gum]
executes the GNN-built agents live under the causal schedule of [@sec:hierarchy],
with its own normalized-character estimand and explicit vocabulary coverage.
For the letter level and the bounded letter-to-phone check, the live recursion
is the forward algorithm of a hidden Markov model (HMM) [@rabiner1989tutorial].
That equivalence is not a concession; it is what makes the
verification in [@sec:verification] possible, since an independent plain-numpy
HMM can only be an oracle for a decoder that is one. [@sec:related] states the
correspondence in full.

The third layer is exercised by the letter-generation mechanism, both in its
historical standalone experiment and in connected next-word continuation. It makes the letter
transition controllable — one action per letter, so choosing an action *is*
choosing the next symbol — and takes its preference vector from the word level,
which is the only non-flat `C` in the system. The emitted letter is then a
compromise between the corpus habit in the policy prior and the lexicon's goal
in `C`, selected by minimizing expected free energy ([@sec:generation]). Results
in this manuscript are attributable accordingly. The six levels of
[@sec:letter_level] through [@sec:narrative_level], and
six of the seven cross-level
channels of [@sec:hierarchy] — the likelihood-reweighting channel and the
prior/predictive-message channels — are historical interventions on the filtering components
([@sec:related]), not a claim of HMM equivalence for the complete connected schedule. The remaining channel, the lexicon into letter
generation, and the generation results it produces are the part that subset does
not cover.

## Contributions: explicit models, conditional composition and falsifiable evaluation

The contributions below span the level stack itself, each level's GNN
declaration and the conformance check that binds it to the live agent, the
gate-audit-and-claims-ledger method that binds every reported number to a
recorded run, and the measured cross-level interventions between
representations, followed by the separately evaluated connected runtime. Comparisons against external
LLM probes (Ollama, OpenRouter) are an accessory reference frame confined to a
supplement ([@sec:external_benchmarks]); no result in the body depends on
them.

**A connected conditional runtime, evaluated separately from its component history.**
The current runtime builds its agents from GNN specifications and registered
artifacts, schedules inter-level messages, and compares the complete named-channel
configuration with disabling those channels on a locally frozen GUM population
([@sec:connected_gum]). Both arms retain the same observation encoder. The
comparison reports normalized-character predictive loss and full vocabulary
coverage; its closed-vocabulary whole-word loss remains explicit even when
out-of-vocabulary input makes it infinite. It tests that fixed composition on
that population, not global posterior exactness, individual channel benefits,
semantic comprehension, or independent replication.

**A language model with explicit conditional probabilities, preferences, habits
and policy precision, declared in GNN and checked against what got built.** The
letter level (`src/ailm/levels/letter.py`) and the phoneme level
(`src/ailm/levels/phoneme.py`) are `pymdp` `Agent` instances whose sequence
statistics live in the transition `B` and whose emissions live in `A`; those
matrices and the filtering recursion over them define those standalone components. Every
level's matrices, factors, and slot roles are additionally declared in a GNN
file under `gnn/models/` and checked against the live agent by
`ailm.gnn.assert_conforms`, so the specification and the code are two sources
kept in agreement rather than one source with a description bolted on
([@sec:architecture]). The
letter and phoneme models improve on their context-free unigram baselines;
the letter-to-phone decoder is the
first level in the system whose emission is not the identity — letters observed,
phones inferred through a learned `A`. Details in [@sec:letter_level] and
[@sec:phoneme_level]. The remaining built levels each have a recorded gate outcome: words
as a factored model whose posterior support induces a trie ([@sec:word_level]),
case-role assignment ([@sec:case_level]), sentence type over a widened symbol
stream ([@sec:clause_level], passing on a bar registered before the artifact
existed, with no held-back replication available; the case-only arm it
displaced is retained as a non-passing ablation), and document-position terciles
([@sec:narrative_level]). The builders are live agents, but the later headline
scores use the exact/NumPy paths just stated. The six levels have separate state representations. In the
historical stage evaluations, each is scored as a separate probabilistic
component. The subsequent connected runtime coordinates them through conditional
messages; it does not turn them into a six-level exact joint
posterior ([@sec:limitations]).

**A stack of six levels, seven receiver interventions, and a criterion still under test.** The six levels make
a stack; whether they make a hierarchy depends on what passes between them.
The historical standalone decoders exchange no beliefs, whereas the connected
runtime now executes the causal schedule described in [@sec:hierarchy]. Those two words are used in the
narrow senses [@sec:hierarchy] fixes, alongside *composition* and *coupling*,
and the historical component results alone do not establish the strongest of
the four. The 7
receiver interventions were therefore built on top of that composition, and
those that have run were measured one at a time; their
signs disagree. The letter predictive into the letter-to-phone decoder
(L1→L2B) *harms* it, a
corrected-threshold survivor. A lexical prefix message into that same receiver (L3→L2B)
helps, also a corrected-threshold survivor. The magnitudes and uncertainty are
reported in [@sec:hierarchy]. The lexicon
into letter generation (L3→L1g)
supplies the first non-flat preference vector in the project and carries a
constraint no bigram can represent ([@sec:generation]). The
document-position-to-clause intervention (L6→L5; `narrative` in code)
demonstrably moves the receiver's joint `D` prior against a tercile-scrambled
control, but no held-out clause score under that blended prior is reported, so
it is one of the three prior-only results among the
seven ([@sec:narrative_level]). The receiver's
privileged-label reference indicates low sensitivity in the retained case-only
arm, without constituting a hard bound. The separate
privileged punctuation-type-oracle→L3 word-prior intervention was predicted to
harm and did not (it does not use an L5 posterior):
aggregate perplexity and the token-direction tally both favor the message under
the corrected document partition, a helpful result that falsifies the predicted
harm ([@sec:clause_word_channel]). What organizes results with disagreeing signs is
the novel-information criterion — a message may help when it is novel to the
receiver *and* the receiver's likelihood is sensitive to it. The channel results
and one privileged-label sensitivity reference are the whole of its evidence
base; it is labeled a working hypothesis throughout. L3→L1g, L6→L5, and the
oracle→L3 intervention were measured independently, not as a connected cascade.

**Action selection with an explicit account of what drives it.** In the letter
generator, deterministic controllable transitions make the policy-conditioned
state predictive a point mass, so the information-gain term vanishes. The identity
emission then indexes preferences by the emitted symbol, yielding
$\sigma(\gamma \mathbf{C} + \ln \mathbf{E})$: preference and corpus habit
jointly determine selection ([@eq:policy_reduction], [@sec:generation]). The
derivation and contrasting stochastic-transition, flat-likelihood and
action-independent controls identify which assumptions cause the reduction.
A separate budgeted-attention experiment exercises information seeking in
letter-to-phone decoding, but loses to an exhaustively optimal fixed observation
schedule ([@sec:limitations]). These results distinguish implemented policy
mechanisms from a generic appeal to active inference; they do not demonstrate
an advantage for adaptive attention or embodied agency.

**Mis-specified gates, diagnosed and re-derived.** The original Stage 1--4 gates
were repaired after their levels had already been built or measured, and later
stage declarations were re-scoped during audit. Two early gates fail in opposite
directions and are worked through here. The Stage 1 gate
demanded a gain over uniform larger than the first-order information the
training distribution contains. The Stage 2 gate
demanded an accuracy that a context-free lookup table already exceeds,
and specified it against a treebank with no
phonemic annotation. One gate was out of reach of its own model class, the other
free, and both were accepted without a baseline or a comparator having been
measured. `scripts/audit_gates.py` reports every stage as VACUOUS, UNREACHABLE,
ABOVE_REFERENCE, BRACKETED_BOUND, BRACKETED_REFERENCE, or UNAUDITED from its
measured trivial baseline and its typed comparator, so the check is an artifact
rather than an act of attention.
[@sec:gate_design] develops the classification, states which population each
baseline and ceiling was measured on, and tabulates every stage's verdict.

**A comparator stronger than the model it was meant to floor.** The gate audit's
sharpest single finding is mechanistic rather than procedural: a left-context
lookup table, fitted on
training data alone with no latent state and no inference at decode time,
outperforms the shipped letter-to-phone filter, so clearing the weaker
context-free bar did not show
the latent-state machinery doing the work ([@sec:comparator_strength]).

**Verification apparatus, treated as a result.** `pymdp` validates that
matrices are normalized, which is not the same as validating that they are in
the right slot — a bigram matrix is column-stochastic whether it sits in `A` or
in `B`. AILM therefore verifies numerically rather than structurally, and the
apparatus is sized to the claim surface it defends. A closed-form HMM oracle in
plain numpy (`src/ailm/verification/oracle.py`), forbidden by test from
importing `pymdp`, decodes identically to the shipped decoder across a bounded
equivalence subset;
full-set attribution to pymdp is an extrapolation from those checked positions
([@sec:limitations]). A control harness (`src/ailm/control.py`) requires a
positive *and* a negative control per gate and raises `GateVacuousError` when
they fail to separate. Multiplicity accounting reports which members of the
declared significance family survive fixed Bonferroni and Holm step-down
correction ([@sec:multiplicity]). A
claims ledger (`src/ailm/claims.py`) binds every quantitative statement in this
manuscript to a recorded run. None of
these instruments was designed in advance. [@sec:verification] pairs each with
the shipped defect that motivated it.

## Testing linguistic mechanisms with baselines and broken controls

AILM's prospective method prescribes building one level at a time and opening a
level only after the one below it has passed a falsifiable gate. The historical
build did not consistently follow that prescription, as the chronology below
states. Three requirements on the gate itself are:

1. **Use an informative trivial baseline.** For a claim that a letter or phone
   model uses context, marginal symbol frequencies provide the appropriate
   context-free baseline. Beating uniform alone can reflect only frequency
   knowledge. Other claims need their own named baseline: a word model is
   compared with a letter bigram, and a clause mixture with a type-free chain.
   The baseline must remove the mechanism the gate purports to test.
2. **The gate must be bracketed before it is accepted.** Measure what the
   trivial baseline scores and what the model order can achieve. If the baseline
   already satisfies the threshold, the gate is vacuous; if the ceiling cannot
   reach it, the gate is unreachable. A threshold strictly between those scores
   separates those comparators; it still needs mechanism-specific controls.
3. **The gate needs a relevant negative control that actually fails.** A deliberately
   broken variant is run against the same threshold and must not pass. Stage 1's
   mis-wired agent and Stage 2's transposed-transition and corrupted-emission
   controls are kept in the test suite for exactly this purpose.

The third requirement is what makes the first two enforceable, and the second is
exactly where the Stage 1 and Stage 2 gates above failed: neither had its
ceiling or its baseline measured before it was accepted.

## Reading the evidence: development findings and held-back tests

Numbered *decisions* cited throughout are entries in the repository's
chronological decisions log, `docs/decisions.md`, released with the code; each
records a design choice or a measured outcome, including every null.
Quantitative statements in this manuscript use the following strength labels,
and the prose holds each result to its label. *Corrected-threshold survivor*: the effect
crosses the fixed Bonferroni per-test threshold for the declared family of
16 significance tests at $\alpha =$
0.01; 7 tests do, including
the helpful lexical channel and the harmful coupling channel.
*Step-down survivor*: the effect crosses its Holm threshold at the same
familywise level and under the same dependence assumptions, but not the fixed
one; none does ([@sec:multiplicity]).
*Nominally significant*: the test whose prediction was timestamped before its
measurement crosses the uncorrected familywise level but neither corrected
threshold. None of these three labels means *confirmed*. Each names a threshold
crossed on a population that had already contributed to building the family —
each is a retrospective-family threshold survivor on the reused development
population — and a familywise correction bounds a fixed family's Type-I rate
rather than restoring independence to a reused split ([@sec:multiplicity]).
*Non-passing, unresolved*: the shipped comparison does not clear its bar and
the paired gain interval spans zero. It is reported with uncertainty on every
intervalled axis and with a control showing that the same machinery recovers a
deliberately larger planted effect; that control does not prove power for the
real effect at the shipped operating point. The retained case-only clause
ablation ([@sec:clause_level]) holds this status. *Passing on a
development-session specification*: the bar was registered before the artifact
that answers it existed, so it could have failed and did not, but registration
and outcome share one session and no held-back partition remains on which to
replicate it. It licenses less than a confirmatory result and more than a
restated bar, and it is reported with both halves of that sentence every time.
The canonical clause level ([@sec:clause_widened]) holds this status. *Working
hypothesis*:
the claim organizes the design but is bounded by the evidence that produced it;
the novel-information criterion for cross-level channels (decisions 18--19)
is stated over the measured channels and the privileged-label
receiver-sensitivity reference that supports it, never as a universal rule. These labels delimit the intended interpretation; they do not replace checking
the population, comparator and uncertainty. The claims ledger ([@sec:verification]) makes every label auditable against a
recorded run.

A qualification governs every label and belongs here rather than in the
limitations. The original Stage 1--4 gates were restated after their levels were
built or measured because the thresholds were vacuous, unreachable, or named
the wrong population; later stage declarations were re-scoped during audit
([@tbl:gate_audit]). Stage 6's directional prediction was committed before its
held-out measurement, but the development population and analysis family were
reused, and the document-clustered primary analysis is non-confirmatory. The
method therefore includes retroactive re-certification and does not establish a
prospective sequential build. Converting any label into ordinary confirmation
requires a frozen rerun on a population untouched by development. That rerun
was pre-registered, executed to its first data touch, and retired by
exhaustion for the `external_lexicon` members (decision
91); the `ud_ewt_test` members were then scored exactly once against the held-back EWT
test split under decision 111
([@sec:confirmatory]): two clear the frozen per-test bar in their recorded direction, two yield no
discordant pair, and the other twelve members of the family remain
development-population inference.

## Relation to prior language models in active inference

Active inference has reached language before: @friston2020linguistic simulate
generation and comprehension in a deep discrete model on a constructed Twenty
Questions task; the narrower contribution here is to fit such models to
natural corpora and score them on held-out text. To our knowledge, following
searches of arXiv, OpenAlex, and Semantic Scholar through
August 2026 whose full record is published with this work, no
earlier system satisfies the joint corpus-fit and held-out-metric criterion;
the broader index searches were incomplete (rate-limited), so this is a bounded
no-counterexample report and the manuscript makes no priority claim.

## Organization of the manuscript

[@sec:background] fixes notation: variational and expected free energy, the
discrete generative model, the filtering recursion, perplexity, and the
empirical conditional-entropy optimum, followed by the CEREBRUM case inventory
[@friedman2024cerebrum; @friedman2024ccd] that the built Level 4 uses
([@sec:case_level]). [@sec:architecture] gives the level stack and states the
rule that sequence dynamics belong in `B`, and [@sec:gate_design] formalizes
threshold placement. [@sec:letter_level] through
[@sec:narrative_level] report the six built levels; [@sec:generation] reports
generation by policy selection and [@sec:hierarchy] the measured cross-level
channels. [@sec:verification] covers the oracle, the control harness, and the
claims ledger; [@sec:results] collects the measurements; [@sec:limitations]
states what the length-matched subset, the order-1 dynamics, and the
historical standalone decoders and the conditional connected schedule rule out; [@sec:related]
positions the work; and [@sec:reproducibility] gives the commands. The
supplements follow the conclusion: [@sec:audit_trails] carries the gate-audit
trails and the confirmatory replication, [@sec:channel_factorial] the channel
factorial, [@sec:complexity] reports
complexity, [@sec:external_benchmarks] scores the registered zero-shot external
models and an AILM comparator on letter, phoneme, and position-aligned
letter-to-phone next-symbol tasks, [@sec:lean] presents the machine-checked
Lean layer, and [@sec:statistics] supplies statistical
details. The external benchmark uses identical units and a bounded response
contract, but its prose fixture and phoneme comparator are its own rather than
the body's. It is therefore a reference frame rather than a competitiveness
claim, and no result in the body depends on it. [@sec:future] gathers the next
steps this record opens rather than closes — including where an explicit
system built this way and a learned one might call each other — as proposals
to be built and measured, never as results reported here.
