# Interpretation, External-Validity Limits, and Unmeasured Extensions {#sec:discussion}

The connected GUM comparison in [@sec:connected_gum] evaluates a fixed message
schedule on a locally specified population. Its character-loss result does not
remove closed-vocabulary word failure or identify the contribution of each
channel. The historical component gates answer a different question.

For those gates, [@fig:level_gate_summary] places every built level between its own trivial baseline and its own ceiling
comparator on a single normalized axis, so a reader who has followed six
level-specific results in six units can see in one place how far each shipped
model went and what kind of comparator bounds that placement.

![**What to see:** how far each built level's shipped model moved from its own
trivial baseline (0) toward its own ceiling comparator (1), on one shared axis.
Each row prints the raw baseline, model and ceiling values in the level's own
unit beside the normalized placement, names the ceiling's comparator type, and
carries the audited gate verdict. A placement beyond 1 is possible only against
an exceedable ceiling (a reference model or a privileged oracle) and is marked;
beating a model-class or analytic bound would falsify the bound instead. Every
value and verdict is read from the committed gate audit and each verdict is
re-derived from its own row before drawing, so an edited report refuses to
render rather than drawing a stale number. The normalization is a summary for
orientation, not a new estimand: the per-level results and their intervals in
[@sec:results] remain the evidence.](../../output/figures/level_gate_summary.png){#fig:level_gate_summary width=95%}

## Limitations that follow from deliberate architectural choices {#sec:limitations}

Several limitations follow from deliberate architectural choices; stating them
plainly is the price of those choices.

### Not a competitive language model, and not comparable to subword perplexities

AILM has no embeddings, no attention, no learned continuous representations, and
no capacity to generalize beyond discrete conditional tables. The perplexities in
[@tbl:letter_ladder] and [@tbl:phoneme_sequence] are order-1 numbers over small
inventories — not comparable to word-level or subword-level perplexities.
Its named arrays make parameter roles accessible to inspection
([@sec:related]); the recursion, encoder and message schedule still determine
how those roles combine into behavior. The experiments characterize this
architecture's limits, without estimating a general tradeoff between
interpretability and language capability.

### Historical component composition and current conditional coupling

The historical standalone levels compose by sharing an alphabet rather than
passing beliefs. The current connected runtime does pass explicitly scheduled
messages; its global posterior is not asserted to be exact, and its aggregate
GUM comparison does not estimate individual channel effects.
`update_empirical_prior` is within-level temporal recursion at every call site.
Inter-level coupling was built and measured (L1→L2B, `src/ailm/levels/coupling.py`):
accuracy fell monotonically as the mixing weight rose, with a significant *harmful*
margin of -0.2639 points at the best positive weight on the recorded
grid, alpha = 0.1 (paired McNemar
*p* = 1.547e-06). Because the decline is monotone, that figure is the
*smallest* harm the sweep contains rather than a summary of it.
The harm is not marginal under exact uncertainty: the uncoupled decoder wins
a 0.603291 share of the
547 discordant pairs, exact 95% interval
[0.560912, 0.644548]
([@tbl:multiplicity_family]).
The cause is structural: at position *t* the letter-to-phone level already observes letter *t*,
while the message is a function of letters 0 to *t-1* only.

### First-order limits on orthographic and syllabic structure

The letter and phoneme levels are both first-order Markov. The letter level cannot represent orthographic
constraints spanning >2 characters, and the phoneme level cannot represent syllable
structure. The 89.9% figure is a large fraction of
a small quantity.

### The length-matched subset is not English

Results are measured on the length-matched subset of CMUdict
(74.3% of pronunciations discarded), biased toward transparent orthography. No comparison with
published G2P systems is made. The thin margin follows from the subset's transparency
— where the letter nearly determines the phone, context adds little. Two external-validity
bounds follow. First, the family-disjoint arm of [@sec:phoneme_level] removes shared-prefix
transfer but not lexically or morphologically grouped leakage more broadly.
The existing held-out family-group analysis already measures heterogeneity
within those prefix-defined groups ([@sec:results]). The remaining
sensitivity check would assign train/test membership using independently defined
morphological families, rather than treating shared prefixes as the family definition.
Second, the task itself — equal-length position-aligned letter-to-phone decoding under a
pairing the evaluation imposes — stops short of many-to-many grapheme-to-phoneme conversion
over a full dictionary, the setting in which the classical G2P literature evaluates
(e.g. @jiampojamarn2007many); no result here transfers to that setting without the
alignment being learned rather than stipulated.

### Filtering rather than smoothing, and what whole-word methods gain by it

The model's online algorithm is forward filtering; its canonical full-set score
is computed by the NumPy implementation after the bounded live-Agent check.
Whole-word methods do substantially better on the
same parameters — 75.642% for forward-backward smoothing
and 75.306% for Viterbi, against 70.801%
for filtering. The gap is reported rather than hidden precisely because the
architecture forgoes it by construction.

### Part A's dictionary perplexity is in-sample, defended only by its parameter ratio

The dictionary perplexity in [@tbl:phoneme_sequence] is a descriptive in-sample
protocol: the same primary pronunciations supply the counts that fit the
transition matrix and the sufficient statistics that are scored. Its defense is
the parameter-to-data ratio — 39 × 39
parameters from 800,198 positions, with the recorded protocol's add-1
smoothing — not a holdout, so on its own it is not a generalization claim and
is not comparable with Level 1's held-out number as one.

The held-out word split converts that defense into a measurement. The Part A
unigram and bigram are re-fit — inside the analysis code, from the pinned
CMUdict snapshot, with the same recorded add-1 smoothing and per-word prior reset — on
the 100,841 dictionary words that remain when a
seedless stride holds out every 5th
sorted word, the same interleave construction the letter-to-phone split uses
one level down, and the identical closed-form path is scored on the disjoint
25,211 held-out words over
160,196 phone positions. The context-free bar
is re-fit on the same training words, so model and bar share one fit and one
denominator. Held out, the bigram agent scores
16.7665 against the bar's
27.6027 — a margin of
0.4985 nats per position, essentially the whole
in-sample advantage — at a generalization penalty of only
0.022 perplexity units above the in-sample
16.7445. The reading is bounded by a control on each side. A model fit on
training phones re-dealt across word boundaries — the corpus unigram and word
lengths preserved, every trace of order destroyed — does not beat the bar; its
gap is -0.0013 nats, so the held-out margin
is not available to a structure-free fit. And a model fit on the training words
with each word's phone order shuffled within the word still clears the bar, at
0.1443 nats against the honest margin's
0.4985 — which says that a substantial part of the
margin is available from which phones merely co-occur in a word, before any
adjacency is used. The in-sample table remains the descriptive
dictionary protocol; the held-out arm is the measurement behind the claim that
Part A's margin survives generalization.

### Historical fast score paths and current live inference have different evidence

At Level 2B, `infer_states` is one call per observation. Its full-set statistics
are computed by the NumPy oracle and attributed to pymdp through the bounded
decoder-equivalence result ([@eq:decoder_equivalence]). This is a constraint on
that evaluation path, not a license to attribute every later exact or NumPy score
to live inference. L3 uses an exact path identity, L4 a NumPy filter that the
live recursion reproduces at every held-out position, L5 a closed-form mixture
with a live posterior-agreement check that is not score equivalence, and L6
separates its registered observed-position score from a descriptive live
sequential score. The connected GUM evaluation instead executes live agents on
its full declared population using compiled calls. That later execution does
not retroactively extend the historical decoder-equivalence checks or equate
their estimands with the normalized-character result.

### Decoder equivalence is bounded, and licenses nothing outside its subset

Decoder equivalence is tested on a bounded subset (13,224 of
42,820 held-out positions; 0
disagreements on the checked positions); the recorder refuses to proceed if nonzero.
A finite test cannot establish that two implementations are the same function, so
the result licenses no statement about positions outside the subset, and full-set
attribution to pymdp remains an extrapolation from the checked ones. Proving
equality would take an algebraic derivation over the two call sequences or a
full-population comparison; neither is done ([@sec:verification]).

### The UD case oracle is a research claim, not ground truth

Only NOM, ACC and GEN carry morphological marking in English EWT —
3 of the 8 roles. The remaining
5 — DAT, INS, LOC, ABL and VOC — are derived
from dependency relations, a principled heuristic that this project has not
validated. Agreement with it is agreement, not accuracy.

### Unresolved component verdicts and the speech interface boundary

The clause level's **case-only ablation** is non-passing at bigram order over
punctuation-typed sentences (decision 83).
Its privileged-label reference has a negative point estimate, its paired
whole-document-bootstrap interval spans zero, and no equivalence margin was
declared. The canonical clause level, on the widened observation alphabet,
clears its registered bar. But what it clears is a development-session
specification with no held-back replication, so it does not close the
question that ablation leaves open. The document-position (tercile) level
— `narrative` in code — has a positive point estimate, in its preregistered direction, of
0.0051 nats per sentence, whose primary
document-clustered log-loss interval spans zero on the development population.
The same estimand, registered separately and carried to the held-back split,
gives 0.008288 nats with an
interval that excludes zero ([@sec:stage6_heldback_clustered]). So the level
is no longer a case of a positive point estimate with nothing behind it.
Three limits keep it from being more than that. The two populations disagree.
The development one is not retracted. At the family-corrected coverage the
held-back interval spans zero too. So the resolution is a claim at the level
the development analysis was read at, not at the stricter one. The held-back
population's own resolution, 0.005726
nats, is coarser than the development-sized effect. So what was demonstrated
is that a larger effect is present there, not that this effect size
generalizes. A null there would not establish absence of the
development-sized effect. The development-population sentence-sign statistic
remains secondary: it crosses neither the fixed Bonferroni threshold nor its Holm step-down threshold. Even reversed tercile
columns with worse mean loss can yield a nominally favorable sign
([@sec:multiplicity]). That is exactly why the held-back sign test was not
allowed to stand in for the interval. A separate live-Agent filter is
descriptive, not a replacement for that primary analysis. The independent
L6→L5 prior intervention moves the receiver's `D`. The case-only clause
arm's privileged-label reference has an unresolved gain and is not a hard
bound; that arm is the receiver the intervention was measured into. The
speech level is deliberately not a model at all: the boundary contract
(`docs/design/speech_boundary.md`) is tested, while everything acoustic
remains someone else's system.

### The familywise survivor sets are smaller than the nominal set

The declared significance family ([@sec:multiplicity]) has
16 members and 7 survivors at
the fixed Bonferroni threshold 0.000625,
7 under Holm's step-down correction at the same
familywise level. The four
headline paired results — Stage 2B letter-to-phone, Stage 4 derived, the lexical
channel, and the coupling harm — survive both. The secondary Stage 6 sentence-sign
test crosses neither the fixed Bonferroni threshold nor its Holm step-down threshold (`stat_stage6_familywise_confirmed` =
false, `stat_holm_stage6_rejected` =
false), and one survivor,
`subsample_prefix_mcnemar_p`, is significant only because it measures a
biased subsample ([@sec:gate_audit]). Multiplicity correction bounds the
survivor set; it does not rank the science. A further boundary none of these
corrections touches: every gate here is a placement audit against a threshold.
Each audited gate now declares a smallest effect size of interest, and the
audit reads the gate's margin against it (decision 123), so a hairline margin
is displayed as such rather than silently certified. The declared margins are
floors attached after the fact for the retrospective rows — they change no
recorded verdict — and the uncertainty reporting remains what carries the
finding where a recorded margin sits near its floor; a gate registered from
now on must declare its margin before measurement ([@sec:gate_design]).

Two conditions that bound every one of those verdicts are stated in
[@sec:multiplicity] rather than here, because they qualify the results before
the reader reaches them rather than afterwards: the paired tests are computed
at units nested inside the units their corpora sample, and the family was
assembled retrospectively on the population this project developed against.
This entry adds nothing to either. It records only the arithmetic consequence —
that the survivor set is smaller than the nominal one, and that neither set is
a set of confirmed effects.

### The calibration gate was a proxy, and the proxy was not conservative

Calibration was the one axis on which AILM and the external probes looked
directly commensurable ([@sec:external_benchmarks]), and it was originally
reported only
for a system whose answer-slot distribution was at least
0.5 observed rather than reconstructed. That
captured-mass floor stood in for the quantity that actually matters — whether
the completion rule rather than the model decides the reported number — and it
was declared on an argument rather than measured. It has since been measured,
offline and with no model call. Truncating
AILM's own letter columns to 96 target captured masses
and reconstructing each under both completion rules, the shipped min-bound rule
and observed-only renormalization, puts the smallest captured mass above which
neither expected calibration error nor the Brier score again disagrees by more
than 0.01 at 0.9913050462092039, which is
1.9826100924184078 times the declared floor. The declared floor is
permissive, not conservative: at 0.5009002481860143 captured
mass the two rules still report expected calibration errors
0.027279486311722168 apart, against AILM's whole recorded
value of 0.06525983583459523.

That located value depends on requiring a sustained crossing rather than a first
one. Each metric's divergence crosses zero exactly once on the way up, at
0.46163474055756576 and 0.4955158182362678
captured mass — essentially at the declared floor — while the two
reconstructions are still 0.8852931589225496 apart in total
variation. A locator that stopped at the first crossing would have confirmed the
declared floor on a coincidence of opposite biases.

The conservative threshold is also unreachable through the API these probes are
served by. Holding 0.9913050462092039 captured mass takes
19.205 of 30 candidates on
average against a server cap of 20, and
0.395 of positions need more than the cap.
It is out of reach in principle for any probe read through this endpoint, not an
unmet bar that more positions would clear.

The gate is therefore no longer the proxy. It is the direct check of the two
completion rules against each other on each probe's own recorded distribution
(decision 93): a system whose expected calibration error or Brier score moves
by more than 0.01 between rules is withheld. Under that
gate Qwen 2.5 3B reports and Gemma 3 4B
is withheld — its recorded divergences are
0.012740862022979305 in expected calibration error and
0.012133730087630523 in Brier score — although the
captured-mass floor had admitted it at 0.8610770659256101
captured mass; SmolLM 1.7B Base v0.2 is withheld by both rules. The
withheld figures remain in the ledger, and the values in
[@sec:external_benchmarks] are the gate-admitted ones.

The case the proxy could not see is the one the direct check exists for: a
probe whose reported distribution is as flat as AILM's own could clear the
declared floor and still sit where the two rules disagree materially, so that
its reported calibration would be a property of the completion rule rather than
of the model. The direct check flags exactly that, which is why the proxy was
retired from gating. The floor constant is retained as provenance for the
starved-fraction diagnostic and the floor-locating sweep above, not as an
admission rule.

### The gate-audit method generalizes only as a working hypothesis

[@sec:conclusion] expects the bracketing check to generalize past this project and
past active inference, and names the method among the most transferable results
here. That claim carries the *working hypothesis* label of [@sec:introduction]
and no stronger one: it organizes the design and is bounded by the evidence that
produced it. That evidence is every declared gate of a single repository
([@tbl:gate_audit], [@tbl:unaudited_gates]), declared and audited by the same
author. The recorded failure modes are varied enough — unreachable, vacuous,
both at once, mis-scoped, necessary-but-not-sufficient — to establish that the
defect class is real and cheap to detect. They are not a sample from which a
base rate follows, and nothing here tests whether the check catches defects in
gates whose author did not also write them.

### Corpus and language boundaries remain specific to each result

The letter and phoneme results use Brown and CMUdict statistics vendored as
frozen JavaScript Object Notation (JSON) artifacts. The EWT experiments,
exploratory Czech morphology arm and connected
GUM comparison add distinct populations, not blanket evidence for other
registers, languages or pronunciation dictionaries. The normalization choices — collapsing whitespace, keeping only the
primary pronunciation, discarding stress digits — each bias the counts in ways that
were chosen for defensible reasons and not independently validated. GUM's
multilayer corpus design provides a resource for examining linguistic structure
[@zeldes2017gum], but the present raw-text comparison does not evaluate its
discourse annotations or establish transfer across genres. Its document average
and interval describe the frozen scored population; they leave performance on
unexamined registers and independently collected documents open.

### Adaptive observation selection was measured, does not pay, and is not built

One extension this architecture invites is budgeted attention: given a budget of
observations for a word, choose *which* letters to look at by expected free
energy rather than on a fixed schedule. It was measured rather than assumed, and
it does not pay, so it is not built.

This arm is the exact complement of the generation regime in [@sec:generation],
and the two are best read together. There the information-gain term is
identically zero and the preference term carries the whole policy; here there is
no preference term and the information-gain term is the sole criterion, each
candidate position scored by the posterior entropy it is expected to leave. So
the two components of the expected free energy have each been measured in
isolation in this project, in the only two places its architecture admits them,
and the one that carries the work is the pragmatic one.

The bar is the exhaustively optimal fixed schedule per word length and budget,
fitted on the training split — for each shape, the observation set maximizing
training accuracy, searched over every subset rather than approximated. That bar
uses no inference at decode time and is strictly less data-dependent than the
policy it is bounding. Over the held-out letter-to-phone positions at a 50%
budget it scores 50.0381% against the adaptive arm's
48.9549%, a loss of 1.0832 points
for adaptivity. The comparison the proposal originally invited, against a
"uniform stride", is not a comparison at all: the stride's phase is an unstated
free parameter worth up to 12.3129 points across phases at one
word length, which
is larger than any margin at issue.

Two details of how that number was obtained belong here rather than in a
methods appendix, because both are ways the measurement could have come out
wrong in this paper's favor. A policy may only use what it has already
observed, so a candidate position is scored by the entropy it is *expected* to
leave, marginalizing over the letters it might reveal under the model's own
predictive distribution. Scoring it instead by the entropy left after
conditioning on the letter actually present returns
+3.90 points for adaptivity
— but that is not a policy, since it consumes the observation to decide whether
to pay for it, and the budget it operates under is fictional. That variant is
retained in the probe as a declared upper bound; it sits
4.987 points above the
policy that can actually be run.

### Zero epistemic value in the deterministic letter generator

Policy selection at the letter level ([@sec:generation]) is driven entirely by the pragmatic term of the expected free energy: the preference vector supplied top-down by the lexicon ([@sec:generation]), combined with the habit prior from the learned bigram. The information-gain term is 0 nats — identically zero rather than approximately zero. The cause is structural: the generator's transition is deterministic and controllable, so the policy-conditioned predictive distribution is a point mass and carries no residual uncertainty the next observation could resolve. A stochastic-transition control confirms the quantity can be nonzero; it reaches 3.401197 nats, the entropy of a uniform letter distribution. AILM's letter generator therefore cannot trade off exploration against exploitation: it selects letters because the lexicon wants them and habit expects them, not because any letter would say more about the hidden state than another. Whether that is a limitation depends on what the hidden state is. If it is the emitted prefix — which is the case in the shipped generator — then no letter is more informative than another and the zero is correct. The alternative construction — a word-identity hidden state over the lexicon — was built and measured ([@sec:generation]). Its information gain is 2.716363 nats at the empty prefix, yet because emitting a letter never changes which word is being spelled, every action induces the same predictive distribution: the per-action spread of $G$ is exactly 0 and the policy posterior stays on the habit to a maximum deviation of 1.11e-16. Uncertainty in the hidden state is necessary but not sufficient; the epistemic term bears on action only where actions induce different predictives, which over a fixed word is interrogation, not generation. The budgeted-attention arm above is that regime, and it does not pay.

## Stratificational alternatives beyond the widened arm {#sec:widened_caveat}

The widened arm produces the number that makes a choice informed; it does not
make the choice, and the choice it forces is the load-bearing one. A stream
containing verb and auxiliary symbols is no longer Level 4's hidden states read
one level down, unless Level 4's own state space is widened to match — which
would move every Stage 4 number and contradict the stratificational reading the
hierarchy rests on. The alternative, letting Level 5 observe Level
4's states plus material Level 4 does not represent, abandons the one-level-down
claim outright. Both branches remain live after this measurement, and nothing
here selects between them. Making the widened arm the canonical clause level
does not select between them either. It settles which arm the paper describes,
and it takes the second branch as a matter of description while leaving the
stratificational reading unrepaired, which is why the alphabet seam from Level 4
to Level 5 is stated as a containment rather than an identity in
[@sec:architecture] and why the case-only arm, whose seam is a true identity, is
retained rather than discarded. What the measurement establishes is the size of what
is at stake: the positions occupied by verbs and auxiliaries carry a type signal
that the case-only projection does not see, worth
54.95% of the available entropy against the
case-only arm's unresolved gain.

Two further limits. The arm shows *that* those positions carry the signal, not
*why*. Auxiliary fronting is the obvious account and is untested here; the
first-symbol-is-an-auxiliary comparator bounds how much a single fronting
indicator reproduces on its own, at 0.7543 against
the model's 0.8908, which is a bound on one rival explanation rather
than a test of the mechanism. And the arm is descriptive throughout. It was
specified before its artifact was regenerated and its predictions were not
revised afterwards, but the specification and the outcome share one development
session, so it carries the weight of a well-run exploratory result and not that
of a confirmed one.


## Scope of the held-back replication {#sec:confirmatory_discussion}

What the replication of [@sec:confirmatory] changes is narrow and worth stating
precisely. Two of the 16 family members rest on a population that was
fixed before the prediction was written and scored once under a registered
rule, and two more were put to that test and returned no discordance. That is
the difference between a result and a corrected-threshold survivor of a
retrospectively assembled family, and it is available for exactly 4 members
because only those had a held-back partition left: the eight lexicon members
were exhausted, since upstream CMUdict has not moved since the vendored pin,
and the four split diagnostics are re-analyses of a shipped
subsample with nothing to replicate on fresh data. The
twelve unrescored members are
unchanged, and calling the paper confirmed would misread this.

The two members that did cross also differ in what they buy. Stage 4's derived
comparison already survived the frozen correction on the development population;
its replication supplies held-back evidence for that survivor. Stage 6's
crossing adds evidence where the
development run did not cross. The registration predicted the crossing on
sample size alone, but the observed effect estimates also differ across splits.
The crossing corroborates the registered direction; it does not establish a
stable effect size or identify how much of the smaller p-value comes from the
larger population. The case-only clause arm's replicated
null is the most
useful result of the four for a reader deciding what to build on: an
observation alphabet that
fails on its development split and fails again, with zero discordance, on
several times the data, against a live instrument, is not an alphabet awaiting
more data. It is the one that was awaiting a different alphabet, and the widened
arm of [@sec:clause_widened] is what happened when it got one. That contrast is
also the sharpest available statement of what the promotion does and does not
buy: the widened result rests on a registered bar, the null it displaced rests
on a held-back partition, and no single arm here has both.

## Future Tests of Model Scope, Human Auditability, and Hybrid Composition {#sec:future}

Everything below is a proposal, not a result. Where an item is already partly
built, registered, or measured, the text says so; no item here licenses a claim
about what a future run would find. Where a design choice is already fixed by a principal
decision or a rejected alternative, that fact is stated so a reader does not
mistake an open question for a settled one.

### Richer State Dynamics, Wider Linguistic Coverage, and Cross-Framework Checks

A separate next test concerns the reader, rather than the model's predictive
score. The human-evaluation proposal in [@sec:related] would assess whether
named parameters and a trace of the recursion help readers diagnose a defect
or anticipate an intervention. Such a study would need its own participants,
comparison interface and prospective success criterion. Conformance tests and
a functioning dashboard do not supply that evidence
[@doshivelez2017rigorous].

The open Major items already carried in `TODO.md` are the nearest and most
concrete of these: equal opportunity to fit the task, the aligned-pronunciation
zero-shot arms, and the end-to-end top-down cascade's performance claim. None
of them is a fresh idea. Each has an unadopted design written down in
`docs/extension_protocols.md`, and each design is partially landed rather than
merely sketched. Protocol P1's offline data and identity foundation — the
streaming census objective, the diagnostic carve, the separate fitted-probe
registry, the stride-aliasing refutation — is built and fixture-tested, and its
warm-up compute gate has been measured and found not to clear the census
budget at the observed rate (decision 122); what remains is the fit pipeline
itself, gated on the external model infrastructure the protocol assigns to its
own implementation grant. Protocol P2's stipulated pair-structured artifact and
offline evaluator are built, its comparator arm is recorded offline, and its
zero-shot probe arms are recorded: both scoring arms fall far below the
analytic contract floor and the third probe is recorded as a contract refusal,
so the arm closes by refusal rather than ranking ([@sec:external_benchmarks]).
Protocol P3's label-construction and split account, its registry
entries, and its L3 inlet and executor are built, and its influence sweep is
recorded (decision 121, an exploratory prior-only sweep on the development
population); what remains, by the protocol's own three-layer separation, is a
performance claim under its own population, comparator, controls, and
uncertainty analysis — deliberately the last of the protocols to be attempted,
because it is the one whose cost is largest and whose value is least certain.

The connected schedule is now executed and evaluated on the locally frozen
GUM population in [@sec:connected_gum]. Its next empirical steps are independent
replication, coverage expansion and attribution of the aggregate result to
particular channels; those steps remain unmeasured. The channels this project has
already run disagree in sign on that question, which is the whole point of
running them rather than arguing for them: the L3→L2B lexical channel and
the independent oracle→L3 word-prior intervention both help their
receivers, the L1→L2B coupling channel harms its receiver, and the
case-word bridge has exploratory prior-only sweep evidence on the development
population, while its privileged-oracle diagnostic remains structural
([@sec:case_word_sweep]). Attributing the connected gain to a particular channel requires further
registered receiver interventions, each with its own zero-weight anchor,
negative control and accounting of whether the expected direction preceded
the measurement. The existing whole-configuration comparison is not that
attribution, and neither result would declare the hierarchy complete.

Learning of parameters within Active Inference has already been probed at its
boundary rather than left untried, and the next step should be read against
that boundary rather than around it. Observed-state conjugate counting under
`pymdp`'s own parameter-learning machinery reproduces the shipped counted
matrices exactly, which is a correctness anchor rather than a capability
(decision 22). Continuing those same updates online, on a stream whose
statistics are made to drift partway through, does recover a measured,
predicted-in-advance improvement over a matrix frozen before the drift — the
one regime in which online learning has been shown to earn its place over not
learning (decision 28). But in the one place a hidden state had to be inferred
before the parameter update could use it — the G2P emission, whose phone
states are never observed — the learned matrix was measured to be worse than
counting the same data, with an observed-state control confirming the learning
machinery itself was sound and the harm traceable to the latent-state
inference (decision 29). A future attempt at parameter learning within this
architecture should say in advance which of those two regimes it is in, rather
than repeating the optimism this project's own earlier documentation stated
before decision 22 first measured the boundary.

Larger lexica and other languages remain almost entirely untried. The one
non-English population this project has scored is the exploratory Czech case
arm, built by reusing the English case-level builders unmodified against UD
Czech-PDTC and reporting raw accuracy against morphological ground truth
rather than agreement with a derived oracle, because Czech is the one
population where that ground truth exists. That arm is also where the
boundary of the approach shows most plainly: Czech's own case inventory omits
a role English case grammar carries, and several of Czech's roles are marked
directly on the word in a way English morphology does not supply, so the arm
is not a template that generalizes by relabeling — it is a demonstration that
extending to a new language means building that language's own oracle,
vendoring and checksumming that language's own corpus, and stating that
language's own boundary, not rescaling the English numbers. The fixed default
vocabulary at the word level is a separate, already-declined axis of scaling:
a principal decision keeps it at its shipped size because the scaling ladder
already publishes every rung through a closed form exact at any vocabulary
size, so raising it buys no science, only a re-record. A larger lexicon or a
second language is future work exactly to the extent that it is willing to pay
for its own oracle and its own disclosed boundary, not to the extent that it
reuses this project's counts under a new label.

Finally, every shipped level's GNN declaration already carries the
ontology-annotated structure that a conformance checker reads — the same
declaration a hand-written builder and `ailm.spec_agent.build_agent_from_spec`
both construct an agent from, checked to match in shape, dtype, and bytes.
What this project has deliberately not done is run those same declarations
through the upstream GeneralizedNotationNotation project's own generation
pipeline, which independently renders a spec to `RxInfer` and
`ActiveInference.jl` in addition to `PyMDP` — AILM keeps its local
construction paths and does not adopt that pipeline. Rendering the shipped
specs through it and comparing the resulting agent's filtering output against
AILM's own, on the same held-out positions, would be a cross-implementation
check in the same spirit as the pymdp-versus-independent-oracle agreement
checks that already license every exact and NumPy score path in this paper
([@sec:verification]) — except across frameworks maintained by different
people, rather than within the one repository this project controls.

### Declared Parameters and Learned Representations as Distinct Evaluation Targets

AILM and a transformer-based large language model are not competing answers to
the same question, and stating the differences precisely is what makes that
claim more than a disclaimer.

AILM's probability and preference arrays have declared roles and axes checked
against the live agent by the GNN conformance guard; habits and policy precision
have separately named settings ([@sec:architecture]); a transformer's parameters are weights distributed
across attention and feed-forward blocks learned end to end by gradient
descent [@vaswani2017attention], with no declared role for any individual
weight. The consequence is asymmetric epistemic access. AILM's semantic
addressability is designed in: the ontology block names what a matrix means
before it is measured. A transformer's internal structure is instead the
subject of mechanistic interpretability, a field that exists precisely because
that structure is not declared and must be reverse-engineered from a trained
network after the fact, circuit by circuit
[@olah2020zoom; @elhage2021framework]. AILM's guard would have nothing to
check without the declaration; interpretability's methods would have nothing
to interpret without a network that hides its own organization.

Each component uses first-order transitions over its hidden state. That does
not limit a filtered belief to the preceding observed symbol: recursive beliefs
summarize earlier observations, the word posterior carries a within-word prefix,
and the connected schedule carries messages across its declared reset clocks.
These fixed finite-state summaries differ from learned attention across an input
context. Component HMM correspondences are scoped in [@sec:related]; no global
HMM equivalence is asserted for the connected runtime. A single-writer claims
ledger binds reported values to their runs ([@sec:reproducibility]). The external
probes do align bounded next-symbol units and positions, but their unequal
training opportunities, prompting contracts and contamination exposure prevent
those contrasts from establishing general competitiveness
([@sec:external_benchmarks]). Parameter-level auditability and broad language
capability are different evaluation targets. Recent theoretical work argues in the same direction from
the side of prediction and action: a system optimized to passively predict
the next token and a system whose inference is coupled to acting on the world
answer different questions about meaning, and the productive move is treating
them as complementary resources rather than substitutes for one another
[@pezzulo2024generating]. Nothing in this paper measures that argument; it is
cited here as the framing the compositions below take for granted.

### Proposed Tool and Observation Channels Between AILM and Language Models

Compositions in both directions are sketched below, plus the discipline
either one would have to satisfy to be worth building rather than merely
demonstrating. Neither direction has been built, and no result about either
is claimed.

**An LLM calling AILM as an auditable tool.** An agent built in the style of
self-taught tool use [@schick2023toolformer] or interleaved reasoning and
acting [@yao2023react] could delegate a bounded subtask — letter-to-phone
decoding, case-role inference, or a run of the gate-audit method itself
([@sec:gate_design]) — to AILM as a callable tool, and receive back not only
an answer but the posterior and the provenance chain the claims ledger already
attaches to every number in this paper ([@sec:reproducibility]). The point of
the composition, if it is worth building, is that the tool's output would be
checkable by the same mechanism a human reader of this paper already uses,
rather than trusted because a well-audited tool produced it. What would make
this falsifiable rather than a demonstration: a target the calling model
cannot see held out in advance, with the composed answer scored against that
held-out target and not against AILM's own prediction of it.

**AILM calling an LLM.** The designs below are proposals only. First, an
LLM's output could be declared as a new observation modality with its own
explicitly measured likelihood matrix inside a GNN spec — the same kind of
typed declaration every shipped level already carries and the conformance
guard already checks ([@sec:architecture]), fit and audited like any other
emission rather than assumed useful because the source is a large model.
Second, an LLM's output could be treated as a proposal distribution mixed into
a receiving level's prior or preference, registered and measured exactly as
the existing channels are: a weighted mixture with a bit-identical zero-weight
anchor, a scrambled or shuffled negative control, and a document-clustered
interval, reusing the apparatus `P3`'s case-word bridge already specifies
([Protocol P3 bridge account](ailm-supporting-docs/docs__extension_protocols.html#protocol-p3-bridge)) rather than inventing a new one. Read
against work that already translates natural language into a probabilistic
substrate for inference [@wong2023word], the second design is the same move
made in the other direction: instead of a language model translating language
into a program that a probabilistic system executes, a probabilistic system
here would be treating a language model's output as one more typed input
whose influence on the receiver is measured, not assumed.

Either direction inherits a discipline this project already learned the hard
way. It has already rejected one shape of this composition outright:
supplying AILM's own fitted column to an external probe as the probe's input,
then scoring the probe against a target AILM itself produced, so that AILM
becomes the probe's ceiling by construction and any reported margin certifies
the leak instead of catching it (`TODO.md`, "Rejected, with reasons"). Every
proposal above is written to avoid that shape by the same means Protocol P1
already draws the line for the fitted-probe design ([Protocol P1 fit/evaluation boundary](ailm-supporting-docs/docs__extension_protocols.html#protocol-p1-fit-evaluation)): the evaluation target is held out and independent of whichever system
sent the message, the effect is measured at the receiver rather than assumed
from the sender's identity, and no arm scores a system against its own
prediction. The channels already in this paper follow exactly that rule — the
sign of a message is something this project has measured at the receiver in
both directions, not something it has ever been willing to assume from the
sender ([@sec:hierarchy]) — and either an LLM-to-AILM or an AILM-to-LLM
channel would be required to earn its sign the same way.

**The gate-audit method applied to LLM evaluation claims.** This paper already
states, as a working hypothesis rather than a demonstrated one, that the
gate-audit method — measure what a trivial baseline scores and what an
achievable ceiling allows before accepting any threshold in between — should
generalize past this project and past active inference to any staged or gated
build whose progress is defined by a numeric criterion. A published LLM
benchmark claim is exactly that shape of object: an accuracy or a leaderboard
score judged against a threshold whose own trivial floor and achievable
ceiling are usually neither stated nor measured. The calibration gate in this
paper is the nearest worked example of what that audit looks like in
practice: a declared floor, defended by argument rather than measurement, was
found — once actually measured — to sit well short of where the two
completion rules it was meant to distinguish still disagreed materially
([@sec:limitations]). Applying the same discipline to a published LLM
evaluation claim would mean measuring, rather than accepting, where its own
threshold sits relative to a trivial baseline and an achievable ceiling before
treating a model's clearing of that threshold as informative — the same
question this project was willing to ask of its own gates, turned outward.

### Generative Artificial Intelligence Integration: Predictions That a Hybrid Must Earn

The next integration study should test a declared division of labor rather
than add an LLM and infer that the result is more cognitive. Generative
artificial intelligence (AI) would remain an external predictor or tool caller;
AILM would retain its explicit discrete state contract, GNN specification and
`pymdp` inference. Tool-use work [@schick2023toolformer; @yao2023react] and the
hybrid reading model of @donnarumma2025reading motivate this separation, but
do not validate an AILM hybrid.

**Our proposed tool-use prediction** is that a frozen LLM given access to AILM
will improve held-out correctness on bounded phoneme or role-inference tasks
relative to the same LLM without the tool, under a matched inference budget.
The comparison would fix the model version, prompts, decoding rules, tool
schema and independently labeled evaluation targets before measurement.
Matched-output sham tools and shuffled AILM messages would test whether any
gain comes from informative tool content rather than extra tokens or an
additional opportunity to answer. Tool refusals, invalid outputs and call costs
would be scored explicitly. A demonstration of fluent explanations would not
count as success, and self-reported confidence would not be substituted for
measured calibration.

**Our proposed receiver-side prediction** is that an independently fitted,
task-relevant LLM message will lower AILM's proper predictive loss relative to
the same receiver with that message disabled. The message would enter a named
observation or prior interface; observation evidence and preferences would
remain separate experimental arms. The fit split would determine the mapping
and mixing rule. A bit-identical zero-weight anchor, shuffled messages and a
same-budget non-LLM predictor would distinguish useful information from an
advantage supplied by additional fitting. The primary analysis would be paired
on independently held-out documents, with uncertainty reported at the document
level. No gain, or gain retained under shuffled messages, would reject the
claimed benefit of that interface. These predictions concern specified tasks;
they do not forecast general conversational competence or preserve the current
paper's gate verdicts for a modified system.

### Biometrics: Testing Linguistic Beliefs Against Human Measurements

Here biometrics means behavioral and physiological measurements during language
tasks, not a claim that the current system measures a person's cognitive state.
A proposed reading study would align model predictions with fixation durations,
regressions and pupil responses. Expectation-based comprehension motivates
testing surprisal against processing difficulty [@hale2001probabilistic;
@levy2008expectation], while pupillometry requires controls for illumination,
gaze and arousal as well as task demand [@mathot2018pupillometry].

**Our proposed behavioral prediction** is that greater pre-observation
surprisal will predict longer first-pass reading times on independently held-out
items. **Our proposed pupil prediction** is that larger model-predicted belief
revisions will accompany larger task-evoked pupil responses after the nuisance
controls are applied. These are hypotheses to register, not associations already
observed in AILM. Surprisal, posterior uncertainty and belief revision would be
specified as distinct predictors rather than selected interchangeably after
seeing the data. Each predictor would identify its originating model level and
conditioning history; the stimulus population, finite-score requirement and
treatment of out-of-vocabulary items would be fixed in advance. Lexical
frequency, word length, preceding-word effects,
luminance, gaze position and baseline pupil size would enter the predefined
comparison; time-shifted model traces would provide a negative control for
spurious temporal alignment. Participant and item dependence would be modeled,
with evaluation on held-out participants and items. The stronger cognitive
claim would require predictive value beyond both simple lexical baselines and
a frozen LLM predictor, not a correlation considered alone.

Electroencephalography (EEG) and event-related potentials (ERPs) offer a separate
possible extension. The N400 component is sensitive to meaning-related
processing [@kutas2011n400], but the current phoneme, syntactic-role and
document-position states are not validated semantic representations. A study
targeting that response would therefore first need an independently justified
meaning-sensitive task and an explicit extension of the state contract; it
cannot obtain a semantic neural explanation by relabeling an existing state.
Any such study would need its own consent, measurement and analysis protocol.
Failure to generalize beyond the baseline predictors would count against the
proposed model-to-measurement bridge, not invite a reinterpretation of the
biometric measure as proof of active inference.

### Cognitive Linguistics: Interventions on Role, Construction and Context

Case grammar and constructionist accounts offer different hypotheses about
how linguistic form relates to participant roles and learned generalizations
[@fillmore1968case; @goldberg2006constructions]. AILM should test those
distinctions rather than treat its present case inventory as a settled theory.
A candidate extension would declare which role or construction distinctions
are represented, then evaluate independently annotated minimal contrasts in
argument structure, word order and contextual disambiguation. Grammatical case,
semantic participant role and CEREBRUM model-functional role would remain
separate annotation and state inventories.

**Our proposed structural prediction** is that a correctly aligned role- or
construction-sensitive representation will improve held-out disambiguation over
an otherwise identical receiver whose incoming observation-to-role or
message-to-role correspondence is scrambled after fitting, while its learned
arrays and evaluation labels remain fixed. This breaks the informative mapping;
a coordinated permutation of all state axes would merely rename the states
and is not a negative control. The comparator would retain the same lexical information and parameter budget,
so an advantage cannot be credited simply to a larger state space. Independent
annotations and human interpretation judgments would provide targets outside
the features the model observes; derived-label agreement alone would not test
the semantic hypothesis. Cross-linguistic work would use each language's own
state contract and overt morphology where available, with constructions held
out to distinguish generalization from lexical memorization. A null effect
with live controls would reject the chosen inventory or channel claim. It would
not warrant replacing the target after development.

The proposed sequence is to validate the external interfaces and offline hybrid
comparisons before undertaking human measurement studies. Each project would
freeze its population, observation access, primary outcome, direction of the
prediction, comparison family and failure criteria before its first evaluation
data touch [@nosek2018preregistration]. Model or protocol changes after that
point would start a new exploratory record. This agenda specifies what would
make the next claims worth believing; it supplies no new empirical result,
human validation, paid-model acquisition or commitment to clinical deployment.
