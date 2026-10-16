# Related Work: Active Inference, Finite-State Language, and Evaluation {#sec:related}

AILM's nearest neighbors are not transformers. They are the explicit
probabilistic finite-state models — hidden Markov models, their hierarchical
extensions, and weighted finite-state transducers — that carried speech and
pronunciation modeling for two decades and that already had the properties this
project claims for itself: named states, parameters that are conditional
probabilities, and components that compose. That tradition is the comparison
class, and this section places AILM inside it immediately after stating the
active-inference formulation it builds on. Four further literatures are placed:
active inference as a process theory, which
supplies the engine; letter-to-phone conversion, which supplies Level 2's task
and the alignment problem it declines; mechanistic interpretability, which
supplies the motivation; and discourse structure, which already owns a term
Level 6 declines to use for what it measures. Three more are borrowed from
rather than contributed to — the statistics of paired comparison, the
methodology of evaluating prompted language models, and the reproducibility,
testing, and machine-checked-proof tooling used to audit the rest — and the
subsection before the last says what is taken from each and what this
manuscript declines to claim on the strength of it.

## Discrete Active Inference and the Scope of the pymdp Implementation

The free energy principle casts perception, learning, and action as minimization of
a variational bound on surprise [@friston2010free], and active inference turns that
into a process theory in which agents select policies by expected free energy
[@friston2017process]. The discrete state-space formulation — categorical
distributions over hidden states, likelihood matrix `A`, transition matrix `B`,
preference vector `C`, initial-state prior `D` — is synthesized by
@dacosta2020active and given textbook treatment by @parr2022active. That is the
formulation used for the component agents. AILM additionally specifies their
state ontologies, counted parameter sources and inter-level message schedule;
the connected schedule is not itself derived here as exact inference in a
single joint generative model.

`pymdp` [@heins2022pymdp] is an open-source implementation of that formulation and the one used here, and
this project uses it as the implementation of its discrete agents. Historical
exact and NumPy scorers, and the checks linking them to live agents, are
identified separately in [@tbl:level_score_paths]. The
consequence is a hard constraint the rest of the design has to satisfy: whatever a
level does, it must be expressible as `A`, `B`, `C`, `D` over a discrete factor,
because there is no escape hatch. Where pymdp's numerical behavior matters it is
noted explicitly — version 1.0.4 runs on JAX [@jax2018] with float32 defaults, so
analytic anchors are compared with tolerance rather than for equality.

### Prior Language Simulations and the Bounded Natural-Corpus Search

Active inference has been applied to language before, and by the authors of the
formalism. @friston2020linguistic build a deep discrete generative model of
linguistic communication and simulate both generation and comprehension:
synthetic agents take the roles of questioner and answerer in a game of Twenty
Questions, using one shared model, with belief updating by marginal message
passing. The paper's results are architectural and neurophysiological — it shows
which factorial structure the combinatorics of language demands, and that
theta-gamma coupling emerges from belief updating while listening.

That work is the prior art for the sentence "a language model built entirely in
discrete active inference", and it predates this one by six years. Any claim of
priority here has to be narrower than that sentence, and the narrower claim is
about the evaluation rather than the architecture. @friston2020linguistic is
fitted to no corpus: its lexicon and task are constructed for the simulation, and
it reports emergent dynamics rather than a score on held-out text. AILM is fitted
to Brown, CMUdict and UD English EWT, and reports held-out perplexity, position
accuracy and agreement against measured baselines and comparators at every level.

Five further systems sit near the same boundary, and each stops short of it on
a different side. @friston2017deep simulate reading itself in a deep temporal
hierarchy — a discrete model whose slow states generate sentences and whose
fast states generate words, with epistemic sampling of the text driving where
the model looks next — but the simulation runs on a constructed lexicon and
grammar: no parameter is estimated from a natural corpus, and no held-out
language-modeling metric is reported. @friston2021listening move from text to
speech, combining discrete lexical states with a continuous acoustic model.
Candidate word boundaries are selected by minimizing variational free energy
over already sampled acoustic segments. The authors distinguish this selection
from minimizing expected free energy when planning future observations
[@friston2021listening, sec. 5.1]. Their simulated recognition and speech
repetition establish face validity, without reporting a held-out
natural-corpus language-modeling score. @donnarumma2025reading come closest on the linguistic side and
stop shortest on the fitting side: their hierarchical active inference model of
reading and dyslexia takes its textual predictions from an external large
language model, so the linguistic distributions are supplied by a trained
transformer rather than estimated in the discrete generative model itself, and
what is evaluated is the plausibility of the simulated eye movements rather
than a language-modeling metric.

Two newer robot-language systems widen the neighborhood without meeting the same
criterion. @vijayaraghavan2024compositionality combine a neural predictive-coding
architecture with active inference and test compositional generalization in
constructed language--action simulations. @tinker2025curiosity amortize active
inference with Q-learning while robots acquire imperative sentence--action
associations through self-exploration. Both are relevant demonstrations of
language and action learning; neither is a discrete active-inference language
model whose own linguistic probabilities are fitted to a natural corpus and
scored by a held-out language-modeling metric.

The bounded statement this project can defend is therefore: **the documented
search found no earlier system meeting every listed facet — discrete active
inference, natural-corpus fitting, and a held-out quantitative language-modeling
metric.** That is a report about the search, not a claim of priority. Any earlier
system satisfying every listed facet is a counterexample. The unqualified claim
that active inference had not reached language is already false because of
@friston2020linguistic.

The scoping was checked by a targeted search rather than assumed, and its
machine-readable record is `data/literature_search.json`. The arXiv API query was
complete through August 2026. The wider OpenAlex title search
was interrupted by rate limiting before every cursor page was retrieved, and
Semantic Scholar returned Hypertext Transfer Protocol (HTTP) status `429`
before returning any result. The record
preserves those failures rather than converting them into an empty result set.
Its exact query strings cover "active inference" jointly with language,
linguistics, speech, and reading, plus the narrower corpus and held-out
perplexity formulations. The inclusion criterion requires a discrete
active-inference system whose own linguistic parameters are estimated from
natural-language corpora *and* which reports a held-out quantitative
language-modeling metric. The screened candidates include the works above —
@friston2020linguistic, @friston2017deep, @friston2021listening,
@donnarumma2025reading, @vijayaraghavan2024compositionality, and
@tinker2025curiosity — and none satisfies all facets. Because the broader search
is incomplete, the record explicitly forbids a priority claim.

What AILM adds on this axis is therefore not that the formalism reaches language,
which was known, but a measurement of what the formalism contributes once it
does. [@sec:generation] gives the sharpest form of that: under the construction
these levels use, the expected free energy's information-gain term is identically
zero, so policy selection reduces analytically to a preference-weighted softmax
over the habit. The reduction follows from the model's construction and does
not require corpus fitting. The corpus-fitted experiments and controls measure
the behavior of that reduced policy rule on the declared tasks.

## Hidden Markov Models and Weighted Finite-State Composition

With their conditional inputs fixed, AILM's standalone filtering components
have the finite-state Markov structure described below. This correspondence is
not a global HMM claim about the connected schedule. The letter level is an HMM with an identity emission; the G2P
(position-aligned letter-to-phone) level is an
HMM with a learned emission and a learned transition, decoded by forward filtering.
The classical treatment of that model class — the forward recursion, forward–backward
smoothing, Baum–Welch training — is @rabiner1989tutorial, and MAP path decoding is
@viterbi1967error. The oracle in `src/ailm/verification/oracle.py` implements
forward filtering, forward–backward smoothing, and Viterbi decoding. It does
not implement Baum–Welch training: the corpus-derived level parameters are
estimated from counted training artifacts. The independent inference recursions provide the check on
pymdp, with numerical agreement assessed under the declared tolerances.

Multiscale discrete Markov structure has a direct predecessor, and it is older
than this project's framing suggests. @fine1998hierarchical introduce the
hierarchical hidden Markov model, in which a state may itself be a sub-model,
and demonstrate it on natural English text with levels of the hierarchy
corresponding to structure at different length scales in that text. Two
differences from AILM are worth naming precisely, and only one of them favors
this work. An HHMM is a single automaton: inference runs over the whole nested
structure at once, and the paper's technical contribution is an extension of
Baum–Welch that *estimates* the hierarchy from unlabeled data. AILM's levels
are separately parameterized agents, fitted separately from labeled corpus
statistics. The original stage evaluations and intervention measurements were
separate; the subsequent connected runtime executes the conditional schedule in
[@sec:hierarchy]. It does not estimate a single globally exact joint hierarchy,
so its locally evaluated composition in [@sec:connected_gum] does not establish
the HHMM's stronger joint-inference interpretation. What its levels
have and an induced HHMM's levels do not is a declared meaning per state, which
is what lets a channel between two of them be stated as a claim about symbols
rather than about indices.

The second comparator is the one that bounds this project's cost claims.
Weighted finite-state transducers [@mohri2002weighted] are the mature framework
for exactly the composition AILM performs by hand: a lexicon, a pronunciation
mapping, and a transition system are separate weighted machines, joined by a
composition procedure with state pairs and an optional lazy implementation
that constructs them as needed. Determinization and minimization can reduce
redundancy where their conditions hold; they do not guarantee that composition
avoids a product state space, and not every weighted automaton is determinizable
[@mohri2002weighted]. That matters twice here. It is the
architecture against which an explicit lexicon-to-pronunciation cascade is
normally measured, so Levels 1 through 3 sit on its home ground rather than on a
transformer's. And it is the counterexample to reading [@sec:complexity]'s
dense-representation figures — the 24.6 GB trie transition
above all — as the computational price of being explicit. That section already
attributes the cost to pymdp's dense arrays and their absent sparse path through `Agent`; the
transducer literature supplies explicit, composable graph representations whose
storage need not be a dense transition matrix over every state pair. This is an
alternative representation, not a bound guaranteeing a small composed graph.

A third member of the same explicit-parameter tradition sits one rung above
AILM on the Chomsky hierarchy rather than beside it. Probabilistic context-free
grammars assign probabilities to context-free production rules and estimate
them from a corpus with the inside-outside algorithm, itself a context-free
generalization of the forward–backward recursion used here [@lari1990estimation].
That step buys unbounded recursive structure — center-embedding chief among
it — that no finite-state model, including every level built in this project,
can generate. AILM does not take it: each isolated component has first-order transitions over its declared state
space and filtering rather than constituent parsing, a property
this section returns to below when Reich's finiteness thesis is weighed against
Chomsky's objection to it. A PCFG is the instrument for constituent structure
that AILM's finite-state levels decline to represent, and the corpus-fitted
comparators chosen here are the other finite-state members of the same
explicit-parameter tradition for exactly that reason.

The HMM correspondence is worth stating rather than obscuring, and it cuts two
ways. The Level 1 and Level 2 results use classical sequence-model recursions;
their numerical reproduction requires the same fitted arrays, priors,
decoding conventions, and tolerances. The measured addition beyond passive HMM
inference is the use of preference vectors `C` to make generation a
policy-selection problem. Generation by policy selection is built and measured
([@sec:generation]). The case scaffold records the source's structurally distinct
formulations and implements only AILM's NOM/ACC scalar approximation; no shipped
decoder calls it ([@sec:case_level]). A further candidate
has been struck off rather than left pending — the first inter-level message
channel was built and measured harmful ([@sec:limitations]) while a second,
carrying information its receiver lacked, helped ([@sec:hierarchy]), so the historical alphabet-sharing construction and its separately measured
interventions must be distinguished from the current conditional runtime.
Component HMM equivalences support the local inference checks; they do not
establish a globally exact HMM for coupled execution. Generation by policy
selection ([@sec:generation]) is evaluated separately, and the case-conditioned
scaffold remains specified rather than shown.

## N-Gram Estimation and Letter Entropy as Information-Theoretic Context

The information-theoretic frame the gates in [@sec:gate_design] depend on is
Shannon's, cited here as historical context rather than as a comparison
condition: entropy as the bound on compressibility [@shannon1948], and the
entropy of printed English measured by human prediction experiments
[@shannon1951]. No AILM number reproduces or tests those measurements. The
letter-level ceiling in [@tbl:letter_information] is the conditional entropy $H(X_t
\mid X_{t-1})$ of the training distribution, which is the order-1 instance of
exactly that quantity. Later work tightened the bounds with gambling estimates
[@cover1978gambling] and with large-corpus models [@brown1992entropy], and the
smoothing literature [@chen1999empirical] is where the add-1 smoothing used here
would be replaced by something defensible if the transition tables mattered as
language models rather than as generative-model parameters. The classical
pairing of an n-gram language model with an HMM acoustic model that this
lineage feeds into is set out at length in [@jelinek1997statistical]; AILM's
letter- and phoneme-level HMMs sit in that same discrete-state, count-based
tradition, cited here for the tradition rather than as a system this work
extends.

One further quantity from the same lineage governs [@sec:hierarchy] rather than
the gates. What a message from one level can be worth to another is, under
logarithmic loss and an optimal predictor, the conditional mutual information
between the message and the receiver's target given what the receiver already
holds; @jiao2015logarithmic derive that correspondence from the benefit of side
information and use it to justify logarithmic loss itself. That theorem is the
formal content of the *potential value* half of the working hypothesis stated in
[@sec:hierarchy]. Its *realized value* half has no theorem behind it, because
realized value depends on the receiver's misspecification and on the rule by
which a message is mixed in, neither of which the optimal-predictor result
constrains. That asymmetry is why the hypothesis is stated as two clauses rather
than as a biconditional, and the citation is what marks which clause is
borrowed.

Two points of positioning. First, the perplexities [@sec:results] reports for Levels 1 and 2
are order-1 character and phone perplexities, and no perplexity reported there,
including the Level 3 per-emission letter perplexities, is comparable to
word-level perplexities on standard benchmarks such as the Penn Treebank
[@marcus1993penn], the corpus the original Stage 3 gate named before the built
Stage 3 was measured on Brown instead ([@sec:word_level]). Second,
the corpora themselves are the standard ones — Brown [@francis1979brown], accessed
through NLTK [@bird2009nltk] — so the letter statistics should be reproducible by
anyone, and are vendored as frozen counts precisely so that they are
([@sec:reproducibility]). Reproducibility of that kind is necessary and not
sufficient: a checksum fixes which bytes were used and says nothing about whose
English they are. The data-statement practice @bender2018data propose asks for
the rest — language variety, speaker and curation demographics, the selection
rule, and the population a result may be generalized to.
`docs/reference/data_statement.md`, described in [@sec:reproducibility], collects these
boundaries for the English corpora and fixture, the exploratory Czech case arm,
and the stipulated controls. Their distinct populations and fitting histories
limit different results; sharing checksummed artifacts does not make them
representative of language as a whole.

## Grapheme-to-Phoneme Conversion and the Cost of Stipulated Alignment

Grapheme-to-phoneme conversion has a long dedicated literature, and its central
technical problem is the one AILM declines. CMUdict [@cmudict] entries are not
length-matched, so learning $P(\text{letter} \mid \text{phone})$ requires knowing
which letters emitted which phones. The field treats that correspondence as a
latent structure to be inferred rather than as a preprocessing detail:
@jiampojamarn2007many infer many-to-many alignments between letter and phone
substrings before training, and joint-sequence models [@bisani2008joint] promote
the aligned unit itself to the modeling primitive. Neural sequence-to-sequence
models later superseded both on standard splits [@yao2015sequence] by dispensing
with an explicit alignment variable altogether.

Set against that literature, equal letter and phone counts do not make the
correspondence unambiguous — they make the *index pairing* unique, which is a
different and much weaker property. English orthography contains digraphs,
silent letters, and context-dependent graphemes that do not respect single
character boundaries, so a word can carry as many phones as letters and still
require a non-positional alignment. [@sec:phoneme_level] scopes the level to the
pairing as a stipulation and names the task position-aligned letter-to-phone
decoding for this reason. The consequence worth stating here is stronger than
the sampling bias that section already records. On a length-matched word whose
true alignment is not positional, the fitted emission absorbs a wrong local
correspondence, so `A` is a statistical relation between co-indexed symbols and
not a learned grapheme-to-phoneme mapping. That is what makes the accuracy
incomparable with any published G2P figure, and no such comparison is offered
([@sec:limitations]).

What the Level 2 result is *for* is different from what a G2P system is for: it
is the first place in this architecture where the emission matrix is not the
identity, so it is the first place where the posterior is over a variable that
was never observed rather than a read-off of the observation. It is scored
against a context-free lookup table on the same subset rather than against the
field.

## Parameter Decomposability and the Missing Human Evaluation

The mechanistic interpretability works considered here recover mechanism from
trained networks after the fact: circuits found by feature visualization and ablation
[@olah2020zoom], and the mathematical decomposition of transformer components
[@elhage2021framework]. @rudin2019stop argues the harder line that for
consequential decisions one should build models that are interpretable by
construction instead of explaining opaque ones afterwards.

AILM is an instance of that harder line taken to its limit for a language
system, and the literature has a name for the specific property it buys.
@lipton2018mythos separates transparency — simulatability, decomposability,
algorithmic transparency — from post-hoc explanation, and argues that
"interpretable" labels a family of unlike properties rather than one. What AILM
claims is parameter-level decomposability, and nothing wider: conditional
probability entries refer to named symbols, while preferences, policy habits
and precision have separately declared roles. `B[0, i, j, 0]` *is* $P(\text{next} = i \mid
\text{current} = j)$, and a prediction can be traced to specific matrix entries
through the stated recursion. The slot semantics need not be recovered post hoc;
the validity of the state ontology and the explanation of a whole trajectory
remain open questions.

This structural claim does not establish explanation quality.
@doshivelez2017rigorous distinguish application-grounded evaluation,
human-grounded evaluation and functional proxies. AILM's named axes and
conformance checks support a proposed proxy for auditability; no human study
validates that proxy, and no downstream task is shown to improve because the
matrices can be inspected. A future evaluation could ask a specified group of
readers to localize planted wiring defects or predict the effect of a stated
intervention, with a matched alternative interface and a separately defined
success criterion. That is a proposed study, not evidence supplied by the
current tests. The low-order and closed-vocabulary limits in [@sec:limitations]
characterize this implementation; they do not estimate a universal capability
cost of interpretable modeling. The finding that
pymdp cannot detect a transposed emission ([@sec:verification]) is the sharpest
available illustration of the gap between the two notions: a fully named, fully
inspectable model is not a self-checking one. Decomposable structure made the
wiring defect *explicable* once found; it did nothing to find it.

## Transformer Representations, Tool Use, and Embodied Meaning

The object the previous section's interpretability literature works to recover
mechanism from is, in current practice, almost always a transformer
[@vaswani2017attention]: self-attention over a continuous embedding, stacked
into residual blocks whose weights are learned by gradient descent and whose
units carry no declared meaning until a separate research program goes looking
for one. AILM's tables are the reverse commitment. `B[0, i, j, 0]` is
$P(\text{next} = i \mid \text{current} = j)$ by construction
([@sec:architecture]), not a quantity a probe recovers from a trained
network; there is no embedding matrix anywhere in the six levels for an
attention pattern to be computed over. The two architectures are not
positioned on a shared capability axis in this manuscript
([@sec:external_benchmarks]) — the contrast is representational, not
competitive, and this section's earlier subsection already gives the sharper
form of it: circuit-finding and transformer-component decomposition
[@olah2020zoom; @elhage2021framework] recover after training what a GNN
declaration and its conformance check fix before it. AILM does not make
transformer interpretability easier or harder; it removes the recovery step
for the family of models it can express, at the capability cost
[@sec:limitations] already states. A forward-looking account of what a
declared, GNN-checked notation and a trained transformer's representations
could still learn from each other, rather than what either already does, is
[@sec:future].

Neighboring literatures compose language models with an external symbolic or
probabilistic layer rather than reading one out of the weights after the
fact, and they bear on what AILM's own declared tables could still become.
@wong2023word translate natural language into a probabilistic language of
thought — a program in a structured generative model, synthesized by an LLM
and then executed and scored by the probabilistic system rather than by the
LLM itself — which is the same division of labor AILM's own case scaffold
gestures at without building: an external system proposes structure, a
declared generative model is the one that computes with it
([@sec:case_level]). Tool-use prompting composes a language model with
external callables at inference time rather than at training time:
@schick2023toolformer teach a model to emit calls to external tools during
generation, and @yao2023react interleave a model's reasoning traces with
actions against an external environment and the observations that come
back. Both are architectures in which the language model is the orchestrator
and a separate, exact system supplies an answer the model cannot compute
itself — the shape a future AILM-as-tool composition would have to take,
and unbuilt and unmeasured here. That direction is not
symmetric with its reverse. This project already rejected the adjacent
composition in which an external probe's own prompt is seeded with AILM's
fitted prediction and the probe is then scored: the probe's resulting upper
bound is AILM itself, and the proposed leak guard for that design was found to
certify the circularity rather than catch it (`TODO.md`, "Rejected", *Supplying
AILM's fitted column to the probes*). A tool-use composition that let an LLM
call AILM would need an estimand that keeps AILM's output out of the context
the LLM is scored on, for exactly the reason that rejection gives; no such
estimand is specified here, and [@sec:future] is where one would have to be.

Whether an active inference reading of a language model's own computation is
the right frame at all is itself an open, and contested, question in the
active inference literature — not one this manuscript needs to settle,
because AILM does not run an LLM inside its own inference loop and makes no
claim about what one is doing internally. @pezzulo2024generating argue the
cautious side of that question directly: a system that only predicts, without
the closed action-perception loop and the risk of consequences active
inference agents are defined by, is not thereby an active inference agent
merely because it minimizes prediction error, and passive next-token
prediction stays short of the pragmatic, embodied sense of "meaning" their
account requires. AILM instantiates discrete filtering and policy-selection primitives, but a
passive filter with a single action does not establish an embodied
perception-action loop. Its generator selects symbols under stipulated lexical
preferences; no experiment here shows that loop acquiring the pragmatic,
world-coupled meaning discussed by @pezzulo2024generating. Named states and
executable policies therefore do not settle that philosophical or behavioral
question for AILM, and the comparison supplies no verdict about an LLM's internal
computation either.

One further connection is closer to home than a citation of technique: the
notation this manuscript's every level is declared in has never yet had its
own source named in this survey. Generalized Notation Notation
[@smekal2023gnn] is the Active Inference Institute's structured language for
generative models, and it is what `gnn/models/*.gnn.md` is written in and what
`ailm.gnn.assert_conforms` checks the running agents against
([@sec:architecture]). Its authorship overlaps this manuscript's, on the same
disclosed basis already stated for @friedman2024cerebrum and
@friedman2024ccd. GNN's `StateSpaceBlock` and `ActInfOntologyAnnotation`
blocks name a hidden-state factor and its role independently of how that
factor's observations are produced; nothing in the notation restricts an
observation modality to a hand-counted table. A GNN declaration that named an
LLM-backed signal as a factor's observation source — logged, versioned, and
conformance-checked the way every hand-built emission here already is,
rather than folded into a matrix by hand — is a proposal this manuscript does
not build or measure, and it is elaborated, as a proposal rather than a
result, in [@sec:future].

## Case Grammar, Morphological Case, and UD-Derived Role Labels

Two computational-linguistics traditions this manuscript relies on for one of
its six levels have gone uncited in this survey specifically, though each is
cited where the technical claim is made ([@sec:case_level]). Level 4's case
roles are Fillmore's case grammar [@fillmore1968case], later qualified by
Fillmore's own reopening of the question of a fixed case inventory
[@fillmore1977reopened] — a caution [@sec:case_level] already inherits rather
than overlooks. The heuristic built here maps a syntactic role, not
morphological case, and the CEREBRUM source this section later places
Level 4 relative to ([@friedman2024cerebrum]) is one heuristic reading of that
older grammar rather than a replacement for it. Level 4's observations —
part-of-speech tags without dependency labels — are drawn from Universal
Dependencies English EWT [@marneffe2021universal; @nivre2020ud], the
annotation standard whose tagset and treebank make a syntactic-role heuristic
checkable against real sentences at all; the gold dependency arcs UD also
carries are the deliberately unused information [@sec:case_level] draws its
withheld upper bound from. Placed beside the finite-state comparators above and the
stratificational ones below, the point is the same one this section makes
throughout: the level is built on named, citable structure from an older
field, not invented for this project, and citing that structure here is what
lets a reader distinguish what Level 4 borrows from what it claims on its own.

## Discourse Structure and the Limits of Document-Position States

Level 6's hidden state is a document-position tercile, and
[@sec:narrative_level] names it that rather than a narrative phase. The reason
the distinction is worth a subsection is that the term is already occupied by
constructs defined over events and relations rather than over position.
@chambers2008narrative learn narrative event chains — sets of events sharing a
protagonist, with a temporal order induced from a corpus — which is a structure
an ordinal partition of a document cannot express. Discourse structure has an
annotated standard of its own: the RST Discourse Treebank [@carlson2001rst]
labels rhetorical relations between spans, so that an opening is identified by
its relation to what follows and not by its index.

An ordinal partition is a legitimate variable, and it predicts sentence type out
of sample ([@sec:narrative_level]). It is neither of those constructs, and
nothing in the corpora used here annotates the structures that would let this
project check whether the tercile correlates with them. The discourse literature
is cited to mark what a narrative claim would cost: annotation, not more
sentences.

## Stratificational Linguistics as an Architectural Precedent

The architecture of this paper — several named strata, each with its own
alphabet and its own inference, related to its neighbors by a correspondence
rather than by a rule that rewrites one into another — is not new. It is the
shape of stratificational grammar, developed by Sydney Lamb from the early
1960s [@lamb1966outline] and set out across phonology [@lamb1966prolegomena]
and semantics [@lamb1964sememic], with a textbook [@lockwood1972introduction]
and a collected volume [@makkai1973readings] by the early 1970s.

The resemblance has to be stated narrowly or it says nothing. "Several levels,
each with its own alphabet, related by a correspondence" is equally true of the
hierarchical hidden Markov model and of a transducer cascade, both of which this
section has already compared against at the level of mechanism, so an identity
claim at that abstraction adds no information about which tradition AILM is in.
What is specifically stratificational is the *realization relation*: a unit at
one level is not rewritten into a unit at the next but stands in a declared
correspondence to it, and the theory's economy comes from having that one
relation do work a rule system spreads across several mechanisms. AILM has
exactly that wherever an upper level's observation alphabet *is* the lower
level's hidden state space ([@sec:architecture]). It does not have the relation
at the word and case seams: the word level reads letters over a
within-word alphabet of its own, and the historical case evaluation reads UPOS tags from treebank annotation.
Connected raw-text execution adds a train-fitted lexical UPOS encoder, rather
than making the case observation alphabet identical to the word state space.
These declared correspondences and explicit bridges bound the stratificational
analogy.

The disanalogy matters more than the resemblance, and it runs against this
project. Lamb's realization is a live, bidirectional path through a network: the
strata are coupled, and a signal arriving at one propagates. AILM's historical
standalone decoders exchanged no beliefs. The connected runtime now propagates
specified conditional messages, with a separately evaluated scope
([@sec:connected_gum]). The original evidence comprised *composition* — L2B
infers over a variable L1 represents — plus
seven
registered receiver interventions built one at a time —
five measured, the
case-word bridge measured as an exploratory influence sweep, and its
privileged-oracle diagnostic the one registered seam that records no
estimand; one measured arm is a
privileged oracle rather than a level output. On the stratificational
reading, the original AILM stage experiments built the strata and only selected
realization relations. Their seven registered interventions
were the initial independently evaluated attempts:
three help, one harms,
three move a prior without a delivered end-to-end effect estimate — one measured
as insensitive, the two case-word seams prior-only (the bridge measured as an
exploratory development-population sweep, the diagnostic unexecuted) — and
zero are mixed-direction. The corrected
punctuation-type-oracle→word measurement belongs to the helpful count while
still falsifying its
predicted harm — a poor
advertisement for assuming the couplings rather than measuring them. Collapsing
the prior-only result into the helpful count would erase the sensitivity
distinction the working hypothesis of [@sec:hierarchy] turns on.

One level is nearly absent from that accounting, and on this reading its
near-absence is the sharpest thing the comparison produces. The measured
interventions are L1→L2B, L3→L2B, L3→L1g, L6→L5 and a privileged
punctuation-type oracle→L3 arm. Level 4 is neither a model sender
nor a receiver in any of them; it enters only as the sender of the prior-only
case-word bridge, whose influence was measured as an exploratory sweep. In those historical experiments its observations come from annotation rather
than from the level below, and there is no isolated end-to-end effect estimate
in either direction. The connected model now includes L4 among its
six levels, but the GUM comparison disables named channels
together and therefore supplies no separate L4 effect estimate. That distinction
limits what the stratificational comparison can attribute to this level.

The program's arguments most directly bearing on the claims made here are
older than the framing this paper reached independently.

The first is Peter Reich's thesis that natural language is finite, and therefore
describable by finite-state means [@reich1969finiteness]. AILM's isolated component filters use first-order state transitions; their
scoped HMM equivalences let a plain-numpy oracle check the corresponding
`pymdp` paths ([@sec:verification]). The results here neither
support nor refute Reich: a finite model that captures
89.9% of the information available at its own
order says nothing about whether a *larger* finite model would suffice for
language, which is the claim at issue.

That thesis is contested, and citing it without its standard objection would be
using a live dispute as though it were settled in this paper's favor. The
objection is Chomsky's [@chomsky1956three]: English contains constructions whose
dependencies nest without bound, and no finite-state device generates them, so
the finiteness of any actual corpus does not establish the finiteness of the
grammar that produced it. Reich's reply and the objection are about different
objects — a body of utterances against a generative capacity — and this paper
takes no side, because nothing it measures could. What the finiteness literature
supplies is not a license but the reason not to treat "it is only finite-state"
as a settled objection either; the finite-state character of these levels is
disclosed in [@sec:verification] as a property, and it is neither defended by
Reich nor refuted by Chomsky here.

A closer argument concerns this paper's methodological core. Reich argued that the
transformational simplicity metric, which counted symbols, measured the notation
rather than the object: the algebraically shorter grammar could be the
structurally more complex one: on his account a symbol count measures skill at
algebraic manipulation rather than the complexity of what is described
[@reich1968symbols]. The wording is paraphrased rather than quoted, because the
source is an unpaginated technical report and this bibliography's standard is to
omit a locator it cannot give rather than quote without one.
His replacement counted nodes in the network, weighted by their connections, and
compared only grammars encoding the same structures. That is the argument of
[@sec:gate_design] one level up and in an earlier research tradition. A declared numeric
criterion can be an artifact of how it was written rather than a fact about the
system, and the remedy in both cases is the same: fix what is being counted to
something the notation cannot inflate, then check the metric against a trivial
baseline and an achievable ceiling. [@sec:complexity] counts operations and bytes
for the same reason Reich counted nodes.

Where this paper departs from the program is in what it will assert. The
stratificational literature argued its economy on the page; the units were
declared eliminable and the networks drawn. Here the analogous claims are gated:
the mis-wiring batteries, the oracle agreement, and the gate audit exist because
a formalism's parsimony is not evidence that the model it expresses is right.

What this comparison does not do is survey the program's reception. Every
source cited above is used for what it proposed, so the section is silent on how the field
answered it — including on why a framework this close in shape to the present
architecture did not persist. That silence is a bounded engagement rather than a
verdict, and it is disclosed here rather than left for a reader to infer from
the dates in the bibliography. It also bounds what the resemblance is worth: a
tradition's shape can be recovered independently, as it was here, without that
recovery being evidence for either the tradition or the recovery.

## CEREBRUM and Categorical Case Accounts as Hypothesis Sources

Level 4's case structure comes from two prior works by the author. CEREBRUM
[@friedman2024cerebrum] defines the 8 case roles with
structurally distinct free-energy functionals $F_{\text{case}}[q]$ and the
$\langle S,A,T,\Omega,O,F,\pi\rangle$ tuple. It does not print a pymdp
crosswalk: the main-text Active Inference Integration figure is caption-only,
and AILM's `T`-to-`B` and `O`-to-`A` correspondences are project heuristics.
What the built level infers over the inventory is a syntactic role rather than
grammatical case, for the reasons [@sec:case_level] gives.

The categorical companion [@friedman2024ccd] supplies the enriched-categorical
formalism, the sequential belief-update rule, and the case-typed DisCoCat
analysis in `FVect`/`FinHilb`, which in turn descends from the compositional
distributional program of @coecke2010mathematical. Its source identity is
also intentionally two-part: the Zenodo record title and the deposited PDF's
printed title differ, and the optional pinned companion repository identifies
the printed-title work at a later manuscript edition. The bibliography and
`data/reference_sources.json` retain those identities rather than silently
choosing one.

One adaptation is load-bearing and is this project's own. Both prior works describe
*comprehension*: how case-structured observations update beliefs. Neither provides a
generation mechanism. AILM's proposal is that generation is policy selection —
policies are next-symbol commitments scored by expected free energy. The historical generation experiment uses the independent L3→L1g seam: the word lexicon supplies L1g's `C`,
and the Brown bigram supplies its policy habit `E`. The L6→L5 model-output
intervention and the punctuation-type-oracle→L3 prior intervention were measured
separately and do not feed that experiment's generator. The connected runtime
also supports live next-word policy selection under its conditional schedule;
that implementation is distinct from the historical generation sweep.
AILM does not redirect that DisCoCat functor. Its implemented `A` and `B` arrays
are column-stochastic conditional kernels, and compatible kernels compose by
matrix multiplication. `FinStoch` is not compact closed, however, so those
products do not interpret the pregroup cups and caps and are not a DisCoCat
meaning functor. The no-embeddings invariant instead holds directly: no measured
path contains learned semantic vectors, and no categorical semantic functor is
implemented. `docs/design/no_embeddings.md` records the corrected boundary and
the proof obligations for a future bridge. The generation commitment is
exercised and measured ([@sec:generation]); no result reported here depends on a
categorical semantic mapping.

## Selection Bias, Reproducibility, Test Oracles, and Formal Proof

Several subjects that would sit here belong with the material they qualify, and
are placed there instead. The apparatus that decides whether a difference in this
manuscript is real is assembled out of four separate literatures, and every
choice in it has a named alternative that was declined rather than overlooked;
that survey is [@sec:statistics], beside the derivations it justifies. What a
fair comparison against a prompted transformer would actually require — and why
the probes reported here are a reference frame rather than a competitiveness
claim — is [@sec:external_benchmarks], beside the probes themselves.

What this project takes from the preregistration and reproducibility-checklist
literature [@nosek2018preregistration; @pineau2021improving] — which knowledge a
late-registered estimand cannot exclude, and which properties a machine-checked
gate audit enforces on every run rather than a checklist enforces once — is
[@sec:stage6_heldback_method] and [@sec:reproducibility] respectively, each
beside the mechanism that implements it. Two adjacent
literatures diagnose why that kind of practice fails rather than only prescribing
it, and this project's exposure to each is worth stating rather than assuming it
identical to theirs. @kapoor2023leakage catalog leakage — train-test
contamination in its several disguised forms, from duplicated near-neighbors to
preprocessing computed over the whole dataset — as the recurring mechanical cause
behind inflated results across machine-learning-based science. [@sec:phoneme_level]
already runs exactly that check on one of AILM's own splits:
82.5% of the Level 2 held-out words share a long
prefix with a training word, a near-duplicate exposure in their taxonomy, and the
manuscript states the bound and declines to call it contamination only because
morphological regularity is part of what the task measures rather than a
confound of it. What their taxonomy does not name is AILM's more basic exposure:
count-based estimation removes gradient optimization from the fitting procedure,
but it does not remove choices about alphabets, smoothing, message weights,
policy precision or evaluation populations. @cawley2010overfitting distinguish
fitting a model from overfitting the criterion used to select it; that distinction
applies to selecting a discrete architecture or channel weight as well as to
optimizing a neural model. A deterministic rerun reproduces a selected analysis,
without supplying fresh evidence that the selection generalizes. The documented sweeps and reused
development data expose AILM to analyst selection even when each fitted table is
deterministic. The token-resolved-from-ledger discipline of [@sec:reproducibility]
addresses a different failure: manually reporting a number inconsistent with its
recorded measurement. It does not repair selection or leakage.
@dodge2019show emphasize the search budget behind a reported score. That issue
remains relevant to the weight and precision sweeps here; their best tested
points are not independently validated optima. Recording baselines, comparators,
sweep grids and registration chronology makes the selection visible rather than
eliminating it.
@bouthillier2021accounting show how much of a claimed benchmark improvement
instead reflects unaccounted variance — across seeds, data order, and search
noise — once enough runs are actually collected to measure it. The core
recorders fix corpora, counts, fitted arrays and declared seeds; replay checks
that configuration's repeatability rather than estimating variance over model
selection or alternative fits.
For the deterministic core, sampling variation on the fixed evaluation
populations is addressed in [@sec:statistics]. Fresh externally served calls can
also vary across executions despite fixed requested settings; their transport,
provider and run-cohort provenance is retained separately. Neither deterministic
core execution nor cached replies establishes serving-weight immutability.

[@sec:verification]'s apparatus answers a question with its own name in software
testing: what checks a system's output when no independent ground truth is
available. @barr2015oracle survey that oracle problem and its workarounds, among
them the pseudo-oracle — an independently written alternate implementation whose
disagreement with the system under test is evidence of a fault in one of them —
and metamorphic testing, which checks relations between outputs on related
inputs when no absolute oracle exists at all. `src/ailm/verification/oracle.py`
is a pseudo-oracle in exactly that sense: a plain-NumPy forward-filtering,
forward–backward, and Viterbi implementation forbidden by test from importing
pymdp, so agreement between the two is evidence rather than circularity. It is
also, deliberately, not a test oracle for the whole system: it checks
decoder-equivalence at the tolerances [@sec:verification] declares, not whether
the state ontology those decoders operate over is the right one, and it caught
the transposed-emission wiring defect [@sec:verification] reports only because
that defect happened to move a decodable quantity — a limit the oracle-problem
literature would call structural rather than incidental. The manuscript's other
use of the word names a different thing: the **privileged oracle** of
[@sec:gate_design] and the derived label of [@sec:case_level] are a measured
comparator given information the model forgoes, not an independent
implementation of the system under test, and the two senses are kept apart here
because the oracle-problem literature's guarantees attach only to the first.

That verification apparatus is checked by tests running against the live
implementation; a further layer checks a narrower claim by a different method
entirely. [@sec:lean] machine-checks the gate taxonomy, the sweep-shape
predicates, and the multiplicity arithmetic in Lean 4 [@demoura2021lean] against
a ledger generated from the same recorded measurements the prose cites, so that
a theorem which fixes a threshold is evaluated by the kernel rather than trusted
by inspection. That is a narrower use of the tool than most: it proves no
property of pymdp, of the corpora, or of whether a measurement was taken at the
right position, only that the recorded values have the logical properties
[@sec:results] and the gate audit assert of them. Most Lean 4 verification work
outside this project builds on Mathlib's library of formalized mathematics; this
layer deliberately does not, declaring no dependency so that the build runs
offline from a clean checkout — possible here only because what needed proving
was a taxonomy and an arithmetic bound rather than a body of theory large enough
to need borrowing one.

## Contribution Relative to Established Finite-State Methods

AILM belongs to the tradition of explicit probabilistic finite-state language
models before it belongs to anything else, and its distinctive contribution
relative to that tradition is not the discrete hierarchy. Hierarchical hidden
Markov models already represented temporal structure at several scales at once,
and estimated it from unlabeled data rather than declaring it
[@fine1998hierarchical]; weighted finite-state transducers [@mohri2002weighted]
already composed lexicons, phonological mappings, and probabilistic transition
systems at scales this implementation cannot reach, using graph representations
rather than the dense matrices priced in [@sec:complexity]. A composed graph can
still grow to the product state space; the representation does not guarantee
a small composition. What is not
standard in that tradition is the rest of the apparatus, and it is four things:
state spaces and matrix slots with explicit semantics rather than post-hoc
labels; a specification an executable
conformance check holds the running agent to ([@sec:architecture]); an inference
oracle forbidden from importing the library it checks ([@sec:verification]); and
a gate methodology in which a threshold's trivial baseline and achievable
reference are both measured, so that a threshold nobody can fail is recorded as
a defect of the same severity as a wrong number ([@sec:gate_design]). The claim
is that those four together catch errors an accuracy number does not — including
several of this project's own.

AILM does not compete on perplexity, accuracy, or capability, and no comparison in
this manuscript is framed as if it did. Its declared comparators are deliberately
scoped context-free baselines chosen to make one claim falsifiable — that the
model uses context — rather than to establish superiority over the strongest
non-neural alternative. The stronger left-context lookup audit in [@sec:results]
shows why those are different claims.
