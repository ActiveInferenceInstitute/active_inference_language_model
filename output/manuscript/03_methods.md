# Model Construction, Conditional Coupling, and Falsifiable Verification {#sec:methods}

This section specifies the components and their connected execution. It states the level architecture, each level's GNN declaration and the conformance check that binds it to the live agent, the gate-audit taxonomy every threshold in this paper is classified by, the construction of each of the six modeled levels and the generation path, the cross-level interventions and the hypothesis they test, and the verification apparatus that binds every reported number to a recorded run. Outcomes are reported in [@sec:results].

## Implemented levels and interface boundaries {#sec:architecture}

[@sec:background] fixed the notation; this section assigns it to the levels the
rest of the paper measures. It names the stack, states the rule that keeps
sequence dynamics in the transition matrix, and distinguishes historical
standalone score paths from the connected conditional schedule in [@sec:hierarchy].

### Named factors and matrix slots

AILM is a stack of `pymdp` agents, one per linguistic level from letters up to
document-position terciles. Each level owns its own $\langle \mathbf{A}, \mathbf{B},
\mathbf{C}, \mathbf{D} \rangle$ from [@eq:pomdp], each level's state space is
enumerable and named, and no matrix is shared between levels. 6 levels are
built this way; the speech row of [@tbl:level_stack] is an interface contract
rather than an agent, and is marked as one wherever it appears.
[@fig:hierarchy] shows the stack together with the emission of each level, which
is the fastest way to see where the learned statistics sit.

![The AILM level stack, drawn from the shipped code's own alphabet constants.
Each solid row is a built `pymdp` agent with its hidden factor, observation
modality, and the declared home of its `A` and `B`; the badge marks the
emission's kind — identity, learned, structural (deterministic, read off the
artifact), or outside `pymdp` entirely (L7, dashed: an interface whose tested
contract is inventory identity and fail-closed rejection, not a model). The
Stage column carries one audited gate per level, and where a stage audits more
than one it names which: Stage 2 draws its grapheme-to-phoneme (G2P) accuracy
gate rather than its phoneme-perplexity gate, and Stage 5 draws the clause gate on the widened
observation alphabet. This table is one row per level, so the retained
case-only ablation is not here; the gate audit records it under
`stage5_clause_case_only` and [@fig:gate_taxonomy] draws it at its measured
position, non-passing, with its privileged-label gain unresolved. A row shown
solid whose gate audit no longer passes, a status contradicting
`scripts/audit_gates.py`, a badge drawn for a stage that audits more than one
gate without naming which, or a symbol count disagreeing with the code
constants would each falsify this figure, and the counts are imported rather
than typed so the last cannot drift silently.](../../output/figures/hierarchy.png){#fig:hierarchy width=90%}

The case-level observation is a universal part-of-speech (UPOS) tag from UD
annotation [@marneffe2021universal]. The speech interface handles ARPAbet
phonetic symbols; digital signal processing (DSP) is outside the model stack.

| Level | Hidden factors and observations | Likelihood and transition | Preferences and initial prior |
|----------------|----------------------------------|----------------------------|----------------------|
| L1 letter | **Hidden:** current letter; **Observed:** current letter | **A:** identity; **B:** Brown letter bigram | **C:** zero; **D:** uniform |
| L2A phone sequence | **Hidden:** current phone; **Observed:** current phone | **A:** identity; **B:** CMUdict phone bigram | **C:** zero; **D:** learned phone unigram |
| L2B letter-to-phone | **Hidden:** current phone; **Observed:** current letter | **A:** learned $p(\text{letter} \mid \text{phone})$; **B:** train-only phone bigram | **C:** zero; **D:** learned first-phone distribution |
| L3 word | **Hidden:** word identity × position within the word; **Observed:** letter or END | **A:** deterministic spelling table; **B:** identity over word × forward position shift | **C:** zero; **D:** word unigram × opening-position point mass |
| L4 case role | **Hidden:** case role (syntactic role, not grammatical case; UD-derived heuristic from the 8 CEREBRUM-named roles of [@tbl:case_roles]); **Observed:** UPOS tag | **A:** learned $p(\text{UPOS} \mid \text{case})$; **B:** case-role bigram | **C:** zero; **D:** learned first-case distribution |
| L5 clause | **Hidden:** sentence type × widened symbol, jointly; **Observed:** widened symbol: a case role, or `VERB`, or `AUX` | **A:** deterministic symbol-component read-off; **B:** type-persistent blocks of learned widened-symbol chains | **C:** zero; **D:** type prior × first-symbol-given-type |
| L5 clause, retained case-only ablation | **Hidden:** sentence type × case role, jointly; **Observed:** case role | **A:** deterministic case-component read-off; **B:** type-persistent blocks of learned case chains | **C:** zero; **D:** type prior × first-case-given-type |
| L6 document position | **Hidden:** document-position tercile (`phase` in code); **Observed:** sentence type | **A:** learned $p(\text{type} \mid \text{tercile})$; **B:** learned left-to-right tercile transition | **C:** zero; **D:** opening-tercile point mass |
| L7 speech | **Hidden:** — outside pymdp by design; **Observed:** ARPAbet at the boundary | **A:** —; **B:** — | **C:** —; **D:** — |

: The AILM level stack as implemented. Earlier drafts of the design merged clause and case into one level; the shipped stack splits them — case is L4, clause structure is L5 above it — and the speech row is deliberately an interface, not a model: DSP sits outside the level stack (`docs/design/speech_boundary.md`). {#tbl:level_stack}

Every modeled row has a live `pymdp.Agent` builder. [@tbl:level_score_paths]
separately states whether the recorded headline score calls that Agent's
application programming interface (API) for inference; a
builder and conforming matrices do not turn an exact or NumPy scorer into live
pymdp inference.

| Level | Recorded score path and status |
|--------------------|--------------------------------------------------------------------------------|
| L1 letter | live pymdp filtering; bracketed bound |
| L2A phone sequence | exact full-dictionary sufficient-statistic score; bounded agreement between live pymdp and NumPy; bracketed bound |
| L2B letter-to-phone | full-set NumPy forward filter; bounded live check with 0 decision disagreements; bracketed reference |
| L3 word | exact path identity; sampled live-pymdp check; bracketed reference ([@sec:word_level]) |
| L4 case role | NumPy forward filter; exhaustive live-pymdp posterior comparison over the held-out set; agreement reported ([@sec:case_level]) |
| L5 clause | exact closed-form mixture, equal to oracle forward filtering on the joint block-diagonal chain to numerical tolerance, with a spec-build bit-identity check against a running agent; gate registered before the artifact existed and passing, with no held-back replication available ([@sec:clause_widened]) |
| L5 clause, retained case-only ablation | exact closed-form mixture with a live-pymdp posterior-agreement check (argmax and joint-posterior tolerance) on the full held-out set, which is not score equivalence for the headline estimand; gate non-passing and privileged-label gain unresolved ([@sec:clause_level]) |
| L6 document position | registered observed-position score uses the direct conditional table; a separate descriptive score runs live sequential filtering; document-clustered primary interval spans zero on the development population and excludes zero on the held-back one ([@sec:narrative_level], [@sec:stage6_heldback_method]) |
| L7 speech | serialization and parsing contract only ([@sec:limitations], [@sec:gate_design]) |

: Recorded score paths and status for the historical components of [@tbl:level_stack]. Agent construction and the recorded scoring route are distinct: live, exact, and NumPy paths are identified per component. {#tbl:level_score_paths}

Reading each hidden factor alongside its observation shows what each level
is for. At Levels 1 and 2A the emission is the identity, so the observation *is*
the state and the level's content is entirely its next-symbol prediction. At
Level 2B and above, the observation and the hidden variable are drawn from
different alphabets and the posterior is an inference: the co-indexed phone from
a letter ([@sec:phoneme_level]), word identity from a letter prefix
([@sec:word_level]),
case roles from tags ([@sec:case_level]), sentence type from a case-role prefix
([@sec:clause_level]), document-position tercile from sentence types
([@sec:narrative_level]). Those are latent-variable semantics, not a claim that
every recorded score was produced by `Agent.infer_states`.

### Declared and measured parameter counts

Every entry of every matrix in [@tbl:level_stack] arrives one of two ways.
A *declared* entry is fixed by the design and is the same number whatever corpus
is read: the identity emissions at Levels 1 and 2A, the word level's spelling
table and its two structural transitions, the clause level's symbol read-off and
the zero mass it places on every cross-type transition, the flat $\mathbf{C}$
every level carries while it is comprehending rather than generating. A
*counted* entry is a smoothed relative frequency estimated on a training
split — a count of a named event over a count of its condition — installed as a
fixed array before inference begins.

Each level therefore has one place where its corpus knowledge lives, and it is
worth naming per level because it moves. Levels 1 and 2A keep it in the
transition, as the Brown letter bigram and the CMUdict phone bigram, with the
phone level's start prior counted as well. Level 2B keeps it in both slots at
once: a letter-given-phone emission and a phone bigram, both estimated on the
training half of the split. Level 3 keeps it in the *prior*. Its emission is
deterministic and both of its transitions are structural constants, so the
lexicon with its frequencies is the level's entire learned content, and a wrong
$\mathbf{D}$ there is what a wrong $\mathbf{B}$ is elsewhere
([@sec:word_level]). Level 4 counts all three of emission, transition and start
distribution. Level 5 counts its transition and its start over the joint
type-by-symbol space while its emission stays deterministic, and Level 6 counts
its type-given-phase emission and its phase transition, whose left-to-right
support is structural even though its entries are counts.
[@sec:verification] reports the one place a matrix is estimated inside `pymdp`
instead of counted outside it, and that the two procedures are the same
arithmetic.

### Cross-level channels and shared alphabets

*Hierarchical* here names a relation between representations. Each level's
hidden state space is a different description of the same text, running from the
symbol to the document — letters, phones, words, case roles, sentence types,
document-position terciles — and the upper level's observation alphabet *is*
the lower level's hidden state space where letters enter the letter-to-phone
model and sentence types enter the narrative level. The word, case and clause
seams are looser, and saying so costs nothing. The word level reads letters over
a within-word alphabet of its own, the lowercase letters plus an end-of-word
symbol, and the historical case-level evaluation reads UPOS tags from treebank annotation.
The connected raw-text mode instead derives a soft UPOS observation through a
train-fitted lexical encoder, whose boundary is stated in [@sec:hierarchy]. The clause seam was an
identity while the clause level observed case roles alone. The canonical clause
level observes the case roles *plus* `VERB` and `AUX`, so its alphabet strictly
contains Level 4's hidden state space rather than equaling it, and material
Level 4 does not represent enters the level above it. That containment is not a
detail of bookkeeping: it is the open stratificational question
[@sec:widened_caveat] states and does not resolve, and it is why the retained
case-only ablation, whose seam is a true identity, is kept rather than
discarded.

That relation is representational, and it is what the shipped stack is. Whether
the stack also earns the word in the stronger sense — a level above making the
level below better — is a separate and measurable question, and it is the one
[@sec:hierarchy] answers channel by channel. One thing the representational
relation does not carry is worth fixing here: it does not make the levels stages
of one pass over one index. In the historical component experiments, each runs or is scored on its own
index against its own artifact, and
[@fig:belief_dynamics] draws the displayed inference paths each on its own index for exactly that
reason.

In these historical standalone constructions, levels compose through their
alphabets rather than through an active channel. The connected schedule of
[@sec:hierarchy] adds explicit messages to those component models.
Level 2's letter-to-phone model observes the
30-symbol letter alphabet that is Level 1's
hidden state space, and infers over the 39-phone inventory, so
its emission $\mathbf{A}_{lp} = p(\text{letter } l \mid \text{phone } p)$ is a
rectangular matrix linking two different alphabets of two different sizes. It
relates a letter to the phone that shares its index in an equal-length word,
which is a stipulated pairing rather than a recovered grapheme-to-phoneme
alignment ([@sec:phoneme_level]). That alphabet identity is fixed when the
matrices are built and is not a value computed at run time: two agents indexing
the same symbol set share a vocabulary the way two functions can share an
argument type without ever calling each other. Level 2B is where that first pays,
being the lowest level whose posterior is hidden-state inference rather than a
read-off of the observation — inference inside one agent. This component-level inference is distinct from
the later coordination of agents by conditional messages.

[@fig:gnn_pipeline] traces one level from its GNN specification to the live agent
and back through the conformance check that binds them. The loop is closed in
both directions: beyond being checked against the live agent, every
specification also *constructs* one. `ailm.spec_agent.build_agent_from_spec`
reads the dimensions, counts, slot roles, and parameterization forms from the
spec and assembles the `pymdp` agent, resolving each
`Learned(generator=..., artifact=...)` declaration through an explicit
default-deny registry backed by the required corpus artifacts, rather than
executing whatever path the spec names — a
registered provenance annotated on the wrong slot is refused at construction,
which turns the anti-informative wiring of [@fig:wiring_comparison] from a
conformance violation into a model that cannot be built at all. For every
shipped level the spec-built agent is bit-identical — shapes, dtypes, and bytes
across every matrix — to the agent the builder functions construct, so the
specification and the code provably describe one model rather than two sources
of truth that happen to agree (decision 20). That equality is asserted per level
in `tests/test_spec_agent.py`, together with the controls that keep it
falsifiable: a planted perturbation, a dtype change, and a missing slot must
each make the comparator fail. [@fig:composition] puts
the Part-A models side by side with the alphabet identity that joins L2B to L1
— the composition seam the rest of the stack repeats. The full
6-level stack is drawn in [@fig:hierarchy], and [@fig:channel_comparison]
plots L1→L2B and L3→L2B, the measured channels that share a receiver;
[@sec:hierarchy] reports all seven registered interventions. `docs/gnn_composition.md`
works through representational, structural, and planned composition at length.

![One level from GNN specification to running agent. The spec's `StateSpaceBlock`, `InitialParameterization`, and `ActInfOntologyAnnotation` fix the dimensions and the role of every slot; the builder functions construct the matrices; `pymdp` receives them; and `ailm.gnn.assert_conforms` closes the loop by checking the live agent's shapes and role assignments back against the declaration. The spec does not generate the builders — both are written by hand, as the panel's own footer states. Two things bind them rather than one: the closing conformance arrow drawn here, and `ailm.spec_agent.build_agent_from_spec`, which constructs every shipped level from its spec and registered artifacts and agrees with the builder bit-identically (decision 20, described in the text but not drawn on this panel). Dimensions shown are read from `gnn/models/L2b_grapheme_to_phoneme.gnn.md` when the figure is generated. A declared dimension on the panel that disagreed with that file, or a live agent whose shapes or slot roles contradicted the declaration while the closing arrow still reported conformance, would each falsify this figure.](../../output/figures/gnn_pipeline.png){#fig:gnn_pipeline width=95%}

![The Part-A models in detail. The alphabet identity joins L2B to L1 as a plain connector along which nothing flows; the full 6-level stack and each gate's audited verdict are drawn in [@fig:hierarchy]. 7 receiver interventions are registered — 5 measured: harmful L1→L2B coupling, helpful L3→L2B predictive lexical messaging, the L3→L1g generation preference, the L6→L5 clause-`D` intervention, and a helpful privileged punctuation-type-oracle→L3 word-`D` intervention that falsified its predicted harm, the last not an L5-output channel — and two further case-word seams, of which the bridge — the case level's own role posterior marginalized into the L3 word prior — is now measured as an exploratory prior-only influence sweep on the development population (decision 121, reported in [@sec:case_word_sweep]), while its privileged-oracle diagnostic remains a registered structural inlet whose evaluation records nothing. [@sec:hierarchy] reports all seven registered interventions. The independently measured upper experiments are not a cascade. [@fig:channel_comparison] plots the channels that share the L2B receiver. These historical standalone score paths exchange no further beliefs; the
connected runtime is evaluated separately in [@sec:connected_gum]. This caption is falsified by a channel effect that disagrees with its recorded value, by a belief exchange where none is drawn, by attribution of the oracle arm to L5 output, or by evidence that the independent experiments were executed as one path.](../../output/figures/composition.png){#fig:composition width=95%}

`update_empirical_prior` is within-level machinery. Every call site in `src/`,
outside the `pymdp` API smoke probes, pushes a level's own posterior through
that same level's own $\mathbf{B}$ to form the prior for its own next symbol —
the second half of [@eq:filtering], staying inside a single agent. The probes
in `src/ailm/generative/smoke.py` are an exception: they call it on
placeholder arrays with no level behind them, only to establish which call
signature the installed `pymdp` exposes. That temporal API is not itself an inter-level channel. The historical
standalone score paths run one level at a time; the connected runtime adds
separate, explicit message transformations rather than redefining
`update_empirical_prior` as a coupling operation. This boundary is tested by
five inter-level experiments, each measured separately. A genuine Level 1 to Level 2 message was
implemented and measured, and it made the system significantly worse;
[@sec:limitations] reports the result and its controls, while [@sec:hierarchy]
shows a lexical channel into that same level helps substantially when it
carries information the level lacks. The L3→L1g generation seam runs top-down
instead: [@sec:generation] gives the letter level a controllable transition
and sets its $\mathbf{C}$ from the word level's lexicon, selecting actions
under [@eq:efe]. Separately again, the narrative level's phase prediction
moves the clause level's joint $\mathbf{D}$ prior, with an exact zero-weight
anchor and a scrambled control ([@sec:narrative_level]); this intervention
does not use the generator's $\mathbf{C}$ mechanism. The privileged
punctuation-type-oracle→L3 word-prior intervention is separate again, and it
does not consume an L5 posterior. None of these five experiments passes its
output into the next. One further caution applies to two of them: the
privileged-label arm of the retained case-only clause ablation is evidence of
low receiver sensitivity, not a hard bound on the L6→L5 intervention, and both
that arm and the intervention are measured against the case-only clause model
rather than against the widened one that [@sec:clause_widened] promotes.

### Worked filtering step: prior to posterior

[@fig:inference_cycle] traces a single symbol through a filtering path. The prior over
the current state is the previous posterior pushed through $\mathbf{B}$; the
observation is scored against that prior through $\mathbf{A}$, which is what
[@eq:perplexity] measures; the posterior is the normalized product; and the
posterior is then pushed through $\mathbf{B}$ again to become the next prior. The
whole cycle is the two halves of [@eq:filtering] and nothing else. L1 executes
its headline path through live pymdp. L2A's canonical full-dictionary score collapses
the same recursion to exact initial-phone and bigram sufficient statistics after
a bounded live-pymdp/NumPy agreement check; L2B's full-set scorer implements it
in NumPy after a bounded live check with 0 decision disagreements. L4's
full scorer implements the cycle in NumPy, and the live-pymdp recursion is
compared with it at every held-out position ([@sec:case_level]). L3's exact
path score, L5's closed-form mixture, and L6's registered observed-position
score do not execute this live-agent loop; L5's live recursion is separately
checked against the NumPy joint filter on the full held-out set at the posterior
level ([@sec:clause_level]).
L6's separately reported descriptive filter does execute it, but measures
sequential prediction from the Agent's inferred prior rather than the registered
known-position conditional.

![One symbol, one inference cycle at a single AILM level. The emission `A` enters twice — once to score the observation against the prior and once in the predictive read-off where per-symbol perplexity is accumulated — while the transition `B` enters once, to push the posterior forward into the next position's prior. A missing emission contribution, a transition applied before rather than after the update, or a predictive score that disagreed with the displayed recurrence would falsify this panel.](../../output/figures/inference_cycle.png){#fig:inference_cycle width=95%}

Locating $\mathbf{A}$ and $\mathbf{B}$ in that cycle explains why their placement is not
interchangeable, and why getting it wrong produces a model that runs.

[@fig:belief_dynamics] makes the cycle concrete across the displayed levels: one
held-out sentence, traced through the model-specific explanatory paths — the belief
state drawn as a heatmap column per position at L1 through L4, and as a scalar
type-posterior trace at L5. This visualization does not imply that all panels are
recorded by live pymdp inference. Each panel runs on its own index, since the levels
do not share one: letters at L1, the letters of a single word at L2B and at L3,
tokens at L4, and the case-role prefix at L5. L6 is absent by construction
rather than by omission — its observation stream is the sentence-type sequence
across a document, so one sentence gives it a single observation and nothing to
condition on ([@sec:narrative_level]).

![One UD EWT dev sentence traced through the shipped inference stack, one belief
column per position. Top: the pymdp letter agent's predictive prior (the
posterior is omitted because the identity emission makes it one-hot). Middle:
the letter-to-phone posterior for one sentence word, computed by the same pymdp
filtering loop the decoder argmaxes over and cross-checked at generation time
against the float64 oracle — the build refuses to draw if they disagree beyond
float32 tolerance; the trie word posterior narrowing letter by letter; and the
per-token case-role posterior (cross-checked against the oracle to float64
exactness, both sides being numpy). Bottom: the clause-type posterior over the
decoded case prefix relative to the static type prior; this panel draws the
retained case-only ablation, whose population-level point result is non-passing;
and each level's belief entropy per position. The paired gain interval,
reported separately, does not resolve how much type signal the case sequences
carry. The
sentence is verified at build time against the Stage 3 vocabulary and Stage 4
tag alphabet; a sentence that leaves either fails the build rather than being
silently substituted.](../../output/figures/belief_dynamics.png){#fig:belief_dynamics width=98%}

### Transition statistics and emission semantics

\begin{proposition}[Sequence statistics belong in \(\mathbf{B}\)]
Let \(M\) be a learned matrix with \(M_{ij} = p(x_{t+1} = i \mid x_t = j)\). Placing
\(M\) in the transition slot, \(\mathbf{B} = M\), with \(\mathbf{A} = I\), yields a filtering
recursion whose predictive prior is \(\rho_{t+1} = M q_t\), so the model predicts
the next symbol from the current one.
Placing \(M\) in the emission slot instead, \(\mathbf{A} = M\) with \(\mathbf{B} = I\), yields a
posterior \(q_t(j) \propto M_{o_t j}\), which is a distribution over the symbol
that \emph{preceded} \(o_t\); an identity transition then carries that
backward-looking belief forward as the prediction for \(o_{t+1}\). Both
placements satisfy column-stochasticity, so both are accepted by
\texttt{pymdp}'s validator, and only the first is informative.
\end{proposition}

The wrong placement is quiet rather than loud, which is why it is kept in the
suite as a live negative control rather than merely described: the mis-wired agent
constructs, validates, performs inference, and returns a finite perplexity
throughout. Measured on held-out prose it scores
27.2499 against 11.9215 for
the correct wiring — informative enough to clear a threshold badly placed at the
uniform baseline of 30, and anti-informative enough to
fall well short of the context-free unigram bar at
19.9558, which is the comparator the negative control is
actually held to. [@fig:wiring_comparison]
puts the transition and emission placements side by side; `docs/decisions.md` records the measurement
and the named mis-wirings it generalizes to.

![The same learned bigram matrix in the transition slot (left, the shipped agent) and the emission slot (right, the mis-wiring). Both are column-stochastic and both pass `pymdp` validation; the right placement produces a posterior over the *preceding* symbol and measures 27.2499 perplexity, the left measures 11.9215. A bar chart beneath ranks both wirings against the uniform comparator, the context-free unigram and the empirical order-1 optimum. A transpose, non-stochastic matrix, or placement that made the two panels indistinguishable would falsify the comparison.](../../output/figures/wiring_comparison.png){#fig:wiring_comparison width=90%}

Two consequences shape the rest of the system. First, orientation must be
asserted by index, not inferred from a passing validator. The gate tests pin
that a lopsided count lands in `B[0, next, current, 0]` and not its transpose,
and they assert normalization along the specific axis that indexes the
produced variable — because a transposed matrix still sums to one along *some*
axis. Second, and more general, structural checks cannot catch this class of
defect at all. What caught it instead was numerical disagreement with an
independently written closed-form chain. That is why `src/ailm/verification/oracle.py`
exists, and why it is forbidden from importing `pymdp`. [@sec:verification]
develops the point.

### Tracing a prediction through named matrix entries

The claim that a prediction is answered by naming a column and reading it is
worth demonstrating once, on real numbers. Take the letter level at the moment
it has just observed `q`. Three steps close the question. *Name the state:* the
emission is the identity, so the posterior over the current state is `q` and
nothing else. *Name the column:* `q` sits at index 20 of
the sorted 30-symbol alphabet, so by
[@eq:filtering] the prediction for the next symbol is
`B[0, :, 20, 0]`. *Read it:* the three largest entries
are *u* at 0.9758,
*space* at 0.0095 and *period*
at 0.0051, and the column sums to 1, so
nothing has been renormalized for display. Each of those entries is a
Laplace-smoothed relative frequency of the corresponding Brown bigram: a count
of how often `q` was followed by that symbol, over a count of how often `q`
occurred. The model predicts *u* because the number in
that cell is 0.9758, and a reader who knows how `q` behaves in
English can judge whether that parameter is the belief it ought to be. That last
point is the one worth keeping. The audit is available to a reader, not only to
an author with the training run in hand.

Near-determinism is not typical, and quoting only `q` would imply it is. The
column for `t` is the ordinary case: *h* at
0.3067, then *space* at 0.2031
and *i* at 0.0988 — a real distribution over
successors, and a correspondingly weaker prediction. What is identical in the two
cases is the procedure: the same three steps, the same named column, read
directly off the matrix that a reader can open.
`scripts/record_worked_example_claims.py` recomputes both columns from the
vendored counts, so every number in these two paragraphs is a recorded
measurement rather than an illustration.

### Level factorization and conditioning variables

The alternative to a stack is one agent whose hidden state is the joint of every
level's factor — a single instance of [@eq:generative] over the product space. Its
emission would have $\lvert O \rvert \prod_f \lvert S_f \rvert$ entries, the product running
over every hidden factor, so the joint letter-phoneme-word emission grows in
proportion to the size of the vocabulary. A flat factor over every base word in
CMUdict is not merely slow to work with; the transition matrix for that factor
is quadratic in the vocabulary and exceeds any reasonable storage budget by
orders of magnitude, and the joint emission with the phoneme and letter factors
multiplies that figure again. The scale table in
`docs/design/architecture.md` records the arithmetic for each level.

Two mechanisms avoid it. Between levels, each level is a separate agent over a
separate hidden-factor set, so its matrices are sized by that set alone rather than
by the product of every factor beneath it. The interface between Levels 1 and 2 is
an alphabet of 30 symbols and the 30 × 39 emission defined over
it — not a joint letter-phoneme state space, which is the object whose size the
stack exists to avoid ever forming. Within a level, `A_dependencies` declares which hidden factors
each observation modality actually depends on, so `pymdp` builds one
low-dimensional array per modality instead of a single array indexed by every
factor jointly. A modality that reads only its own factor gets a
three-dimensional array of batch, observation, and state, regardless of how many
other factors the level carries. `tests/test_pymdp_smoke.py::TestADependenciesShape`
asserts the resulting shapes, so the factorization is a tested property of the
build rather than an assumption about the library.

The word level is where this stops being an optimization. The design document
specified a word as a path through a letter-phoneme trie: the state space
would be trie positions, and the transition would be the trie's own branching
structure. That alternative was measured rather than argued away. A dense
`pymdp` transition over those trie nodes is 24.6 GB at the
length-matched Level 2B vocabulary, and roughly 0.37
TB at full CMUdict. No sparse path is reachable through `pymdp`'s public
`Agent` API, so this is a representation that does not fit, not a
representation that is merely slow. The boundary sits at the API rather than
in the mathematics: `pymdp`'s JAX numerical backend [@jax2018] contains sparse contraction
machinery, and the transition contraction this level needs is already
sparse-capable, but `Agent` rejects sparse arrays at construction and does not
preserve sparsity across the filtering loop ([@sec:complexity]). The shipped
level factorizes instead. Its hidden state is word identity × position within
the word, and both transitions are structural constants. The trie is not
discarded; it is *induced*: after a prefix, the posterior over the word factor
is supported on exactly the vocabulary words carrying that prefix, and that
support set is the trie node. This factorized level is the one that is built,
and its gate is verified bracketed — see [@sec:word_level].

### Parameter counts and structural constraints {#sec:parameter_count}

The factorization argument above is about what a flat model would cost. This is
what the shipped one does cost, counted from the built agents rather than from
the design: `ailm.parameters` constructs every level through its own shipped
builder and counts three things per array, because any one of them alone points
in the wrong direction.

**Stored** is the dense entry count, what `pymdp` holds and what
[@sec:complexity] prices in bytes. **Nonzero** is the entries the model can
place mass on; the rest are structural zeros fixed by the lexicon and the trie,
never estimated from anything. **Free** is simplex capacity on the live support: every
distribution loses one degree of freedom to the simplex constraint, so a column
with $k$ live entries carries $k - 1$, and a deterministic column carries none.
A one-hot emission is a lookup table written as a matrix — it has entries and it
has no freedom. This per-array rule does not distinguish fitted distributions
from builder-fixed distributions with the same support: Level 1's uniform
start prior contributes simplex capacity although the corpus does not fit it.
It also does not infer constraints shared across arrays.

| Level | Declared in | Stored | Nonzero | Free |
|:------|:------------|-------:|--------:|-----:|
| L1 letters | `gnn/models/L1_letter.gnn.md` | 1830 | 960 | 899 |
| L2A phonemes | `gnn/models/L2a_phoneme_sequence.gnn.md` | 3081 | 1599 | 1520 |
| L2B letter-to-phone | `gnn/models/L2b_grapheme_to_phoneme.gnn.md` | 2730 | 2730 | 2651 |
| L3 words | `gnn/models/L3_word.gnn.md` | 5136462 | 46022 | 1999 |
| L4 case roles | `gnn/models/L4_case.gnn.md` | 168 | 168 | 151 |
| L5 widened clause types | `gnn/models/L5w_clause_widened.gnn.md` | 620 | 240 | 199 |
| L6 document-position terciles | `gnn/models/L6_narrative.gnn.md` | 18 | 12 | 5 |
| **Total** | — | **5,144,909** | **51,731** | **7424** |

: Parameters of the historical filtering components, counted from their built agents. The rows are the 7 filtering-agent variants across the six levels of [@sec:architecture]: Level 2 contributes two, a phoneme sequence model and a letter-to-phone model, and they are listed separately because they are separate agents with separate matrices. Preference vectors are excluded from every column and total 150 entries: `C` does not normalize, nothing estimates it from a corpus, and folding a preference into a parameter count would inflate the model by an amount carrying no information about the data. Each array's orientation is asserted before it is counted — a count summed over the wrong axis is wrong without looking wrong, which is the recurring defect [@sec:verification] guards elsewhere. Every level is declared as a GNN file, and the `Declared in` column is that file. For these historical component builders, the declarations and builders are independently authored, so `ailm.parameters` checks each built array against the size its spec declares; the counted components conform exactly. The connected runtime instead constructs its six filtering agents from GNN declarations. This census does not count the complete connected runtime: its additional channel and lexical UPOS-encoder arrays are outside these totals. The generation variant of Level 1 has its own spec, `gnn/models/L1g_letter_generation.gnn.md`, and is not a row here because it is not part of the filtering stack. The design is falsified if any total disagrees with the sum of its levels, if `free` ever exceeds `nonzero` or `nonzero` exceeds `stored`, if any array fails to normalize on the axis its kind declares, or if any built array's size differs from its declaration. {#tbl:parameters}

Three things in [@tbl:parameters] are worth reading directly. First, these counted
filtering components together have **7424 free parameters**, and that is the number
this architecture is built to make small enough to inspect by hand. Of the
three columns, `free` describes the capacity remaining after support and
normalization constraints. It is not a count of parameters trained by the
shipped pipeline, and it is not a directly comparable neural-model parameter
count. The reason is that this model *declares* what a network would have to
learn: the 30-symbol letter alphabet, the
39-phone inventory, the 8 case roles
borrowed from CEREBRUM, the deterministic spelling table that *is* the
lexicon, the structural zeros, and the builder code that decides which entries
exist at all. These counts do not price the information carried by the chosen
alphabets, lexicon, role meanings, or builder code. The deterministic spelling
table and structural zeros do contribute to the stored and non-zero columns,
but they contribute no live-support freedom. The free count also includes the
capacity of the fixed uniform start prior, so it does not state how much a
corpus decides. Second, only 1.0055% of the stored
entries are non-zero, so most of the dense allocation holds structural zeros.
Third, the largest dense array belongs to a level with few free values: Level
3 stores 5,136,462 entries, of which 1999 independent
probability values are carried by its word unigram prior. Its word transition
is an identity padded out to the square of the vocabulary, and its emission
gives one letter per word-position pair. The corpus also determines the
retained vocabulary and its spelling and length structure, but those
contributions are not a count of fitted probabilities.

The factorization makes these independence and deterministic assumptions
explicit. An unrestricted learned joint transition over word and position
would admit additional degrees of freedom. A dense joint matrix can also
encode the same stipulated structure, however, so dense storage alone does not
imply that every entry must be estimated. The table counts the shipped design;
it does not establish factorization as the only representation of that design.

## Gate taxonomy, threshold placement, and controls {#sec:gate_design}

A staged build is only as good as its gates. The premise of the construction in
[@sec:architecture] — that Stage $N+1$ may not begin until Stage $N$ passes a
falsifiable test — buys nothing if the test cannot be failed, and buys less than
nothing if it cannot be passed, because then the honest response to a correct
implementation is indistinguishable from the response to a broken one.

This project shipped both failure modes. They are opposites, they were introduced
by the same kind of inattention, and neither was caught by any test, review, or
validator. That symmetry is the substance of the audit in [@sec:gate_audit]: the
two defects are not two mistakes but two halves of one missing check.

### Vacuous, unreachable, and discriminating thresholds

A gate is a threshold together with a direction. Write $x \succeq t$ for "the
measurement $x$ satisfies the threshold $t$ as declared", and $x \succ t$ for
the strict form the audit applies:

$$
x \succeq t \;\Longleftrightarrow\;
\begin{cases}
x \le t, & \text{lower is better (perplexity, error),}\\
x \ge t, & \text{higher is better (accuracy, agreement),}
\end{cases}
\qquad
x \succ t \;\Longleftrightarrow\; x \succeq t \ \text{and}\ x \ne t .
$$ {#eq:gate_satisfies}

The inclusive relation is what every declared threshold's own wording —
`>=2 nats above uniform`,
`>=60%` — means by "pass". The strict relation is
what the audit uses, and it is exactly the semantics `gate_passes` in
`src/ailm/gates.py` implements: equality fails in both directions, so a
baseline sitting exactly on the bar is never certified vacuous and a typed
comparator sitting exactly on the bar never certifies reachability. The two relations
disagree only on the boundary $x = t$ — and that boundary is not an edge case
here, it is where every audited row sits. `scripts/audit_gates.py` sets each
row's threshold *to* its measured baseline, so $b_P = t$ exactly, by
construction. Under $\succeq$ every such baseline would "pass" its own bar and
every audited gate would be reported vacuous; under $\succ$ none does. The
strict relation is what makes the audit report anything at all, and the price is
stated below: with the bar on the baseline, the vacuity
half of a bracketed verdict is true by construction and the verdict carries
exactly one measured comparison.

Two further quantities are needed, and the whole argument is that they must be
*measured* rather than assumed. The **trivial baseline** $b$ is what a model that
does none of the work the stage is supposed to demonstrate already scores — for a
sequence model, the context-free unigram distribution; for a classification task,
a per-item lookup table. The **typed comparator** is a pair $c = (c_{\text{value}},
c_{\text{type}})$: a measured or derived comparator together with a declaration of
what kind of quantity it is, because the verdicts below differ in what they
license and the license comes entirely from the type. The four types this
project uses are those of [@tbl:gate_audit]: a **model-class bound** (a
theorem-backed upper bound on every model of the class, on the scored
population), an **analytic bound** (a closed-form bound needing no measurement,
such as a total-variation distance of zero), a **privileged oracle** (a measured
comparator given information the model forgoes), and a **reference model** (a
measured system with no privileged access). The first two are *bound* types;
the last two are *exceedable* types, and the Stage 4 derived row — where the
shipped agent scores above its own smoothing reference — is the recorded proof
that exceedable comparators are exceeded in practice.

\begin{definition}[Threshold placement]
Let \(t\) be a threshold whose direction fixes the relations \(\succeq\) and \(\succ\)
above, let \(b\) be the measured score of the trivial baseline, and let
\(c = (c_{\text{value}}, c_{\text{type}})\) be the typed comparator. Then the gate is
\emph{vacuous} if \(b \succ t\); otherwise,
if \(c_{\text{value}} \nsucc t\), it is
\emph{unreachable} when \(c_{\text{type}}\) is a bound and
\emph{above-reference} when \(c_{\text{type}}\) is exceedable;
and if \(c_{\text{value}} \succ t\), it is
\emph{bracketed-by-bound} or \emph{bracketed-by-reference} by the same split.
A bracketed gate of either strength is one the trivial baseline strictly fails
and the comparator strictly clears, so that passing it is evidence about the
model rather than about the threshold; only the bound-typed verdicts assert
anything about models other than the ones that were measured.
\end{definition}

The typing is the load-bearing repair in this definition, and it was forced by
an external audit of an earlier version in which a single untyped $c$ licensed
`UNREACHABLE` whenever it failed to clear $t$. That inference is not valid for
an exceedable comparator:
$$
c_{\text{reference}} \nsucc t
\quad\not\Rightarrow\quad
\sup_{m \in \mathcal{M}} \operatorname{score}(m) \nsucc t,
$$ {#eq:reference_no_impossibility}
because a reference is one member of the admissible class, not its supremum.
Only a genuine bound — information-theoretic, analytic, or exhaustive — turns
"the comparator fails the bar" into "no implementation can pass". The
classifier therefore refuses an untyped comparator outright
(`classify_gate` in `src/ailm/gates.py` fails closed), and the licensing rule
is machine-checked: `unreachable_requires_bound` in
`formal/AILM/GateVerdict.lean` proves that `UNREACHABLE` is derivable *only*
from a bound-typed comparator, whatever the numbers are, and its dual proves a
bound is never softened into a reference statement.

The verdict is deliberately named for what it checks. *Bracketed* is a
placement audit and nothing more: it says the measured baseline strictly fails
the threshold, and the typed comparator strictly clears it, on a stated
population. It does not say the metric is the right one. It does not say the
comparator is the strongest available. It does not say the sample is
representative. It does not say the effect is large enough to matter. It does
not say the threshold was fixed before the run. This paper found several of
those conditions violated somewhere in its own work: a
defensible-but-not-strongest comparator ([@sec:comparator_strength]), an
unrepresentative subsample ([@sec:gate_audit]), and a retrospectively re-scoped
threshold ([@tbl:gate_audit]). That is exactly why *bracketed* is not called
*sound*. A gate worth calling scientifically valid needs seven things: a valid
construct, a relevant metric, the strongest available comparator, a declared
minimum effect, adequate power, a representative test population, and a
prospective freeze. The taxonomy here audits one of those seven. The
*strongest available comparator* clause also has a methodological reading this
project does not yet meet. That reading would make the comparator search part
of the formal method itself: declare the admissible information class, fit its
strongest member on training data only, freeze it, and gate against that
frozen artifact. Doing so would remove the residual discretion in which
comparator gets declared — [@sec:comparator_strength] shows what that
discretion cost at Stage 2B. That change belongs to the analysis code rather
than to this paper, and it is recorded here as the standard the next gate
revision should meet.

Equivalently, as the classification `scripts/audit_gates.py` computes, with the
verdict indexed by the evaluation population $P$:

$$
\operatorname{verdict}(t; P) =
\begin{cases}
\text{VACUOUS} & \text{if } b_P \succ t,\\
\text{UNAUDITED} & \text{if no ceiling } c_P \text{ is measured,}\\
\text{UNREACHABLE} & \text{if } c_{P} \nsucc t \text{ and } c_{\text{type}} \text{ is a bound,}\\
\text{ABOVE\_REFERENCE} & \text{if } c_{P} \nsucc t \text{ and } c_{\text{type}} \text{ is exceedable,}\\
\text{BRACKETED\_BOUND} & \text{if } c_{P} \succ t \text{ and } c_{\text{type}} \text{ is a bound,}\\
\text{BRACKETED\_REFERENCE} & \text{otherwise,}
\end{cases}
$$ {#eq:gate_verdict}

where $b_P$ and $c_P$ are the trivial baseline and typed comparator
*measured on $P$*. The comparator's type is load-bearing, not annotation: a
bound type that fails the bar makes the gate `UNREACHABLE` and licenses an
impossibility claim, while an exceedable type that fails it makes the gate
`ABOVE_REFERENCE` and licenses none; a bound type that clears the bar brackets
the model as `BRACKETED_BOUND`, and an exceedable one that clears it is the
ordinary `BRACKETED_REFERENCE` comparison. The case analysis the audit applies
to the declared thresholds is set out in [@sec:audit_trails]. Step 4 of the
pre-build checklist ([@sec:audit_trails]) makes the population index a requirement rather
than a notation, and the Stage 1 row is where it bit: its comparator was the
Brown *training* conditional entropy while its baseline and model were scored on
held-out prose, so the single comparison a baseline-relative verdict contains
straddled two populations. `scripts/audit_gates.py` now uses that passage's own
empirical order-1 optimum, 5.9928 (the perplexity of the all-position bound of [@eq:heldout_bound]), which is the same
quantity measured where the gate is scored. The population index is not decoration; it is this section's
own Stage 4 lesson stated in its own notation:
$\operatorname{verdict}(t; P_{\text{gold}}) = \text{VACUOUS}$ while
$\operatorname{verdict}(t; P_{\text{derived}}) = \text{ABOVE\_REFERENCE}$ at the
same $t$ ([@sec:case_gate]) — above-reference and not unreachable, because
the derived population's comparators are a smoothing decode and a test-fitted
lookup, both exceedable, so no impossibility claim is licensed there. A verdict is a property of the pair $(t, P)$, never
of $t$ alone — which is why `audit_stage4()` in `scripts/audit_gates.py` takes
`population` as a parameter instead of baking one in.

The order of the cases in [@eq:gate_verdict] is not arbitrary. Under the strict
relation, `VACUOUS` and the failing-comparator verdicts cannot co-fire whenever
$c_P \succeq b_P$: if $b_P \succ t$ and $c_P \succeq b_P$, then $c_P \succ t$, so
neither failing case can trigger. Reporting `VACUOUS` first therefore loses no
information and keeps the more actionable diagnosis in front. The `UNAUDITED`
case sits second for the same reason at one remove: the classifier tests the
baseline, then whether a ceiling was measured at all, and only then the
comparator-relative cases, so a gate with no measured comparator is classified
before any comparison against one can apply.

The condition $c_P \succeq b_P$ is *measured on every audited row*, not assumed.
It cannot be justified "by construction, since the comparator bounds every model
in the class including the trivial one", because these values are typed
comparators — a smoothing oracle, a conditional entropy, a per-item argmax —
and only some of them bound the model class. The
Stage 4 derived row is the counterexample in this project's own ledger, where the
shipped agent scores *above* its recorded smoothing reference. What the case order
needs is the weaker fact about the *baseline*, which holds on every recorded row
except the retained case-only clause ablation, where the type oracle scores worse than the type-free baseline (and where, with $t = b_P$, `VACUOUS` cannot fire anyway), and is checked there; nothing here needs, or gets, a bound over every model.

One degenerate case is worth naming because the argument above is silent
on it: when $b_P = c_P$ the bracketed region is empty, and no threshold on that
population can earn either bracketed verdict — the data admits no gate at all, which is itself
a measurement worth having before building the level. This is not a hypothetical
shape. The Stage 4 gold population realizes it exactly: its trivial baseline is
63.8%, its smoothing reference 63.8% and
the shipped model 63.8% — one number three times
([@tbl:case_both_defects]). No threshold scored against that population could
discriminate anything, whatever value were chosen, which is why no gate is
stated there. The general statement is machine-checked as `degenerate_no_gate`
in `formal/AILM/GateVerdict.lean` rather than argued in prose.

A consequence of the audit's own convention deserves stating plainly, because
a bracketed verdict otherwise reads as more than it is. Setting $t := b_P$ makes
the first case of [@eq:gate_verdict] never fire: $b_P \succ b_P$ is false for every
population and either direction, so the vacuity test is discharged by the
strictness of $\succ$ rather than by anything measured. On those rows
a bracketed verdict is therefore *equivalent* to $c_P \succ b_P$ — the comparison that
the typed comparator genuinely beats the trivial baseline. That is a real and
falsifiable claim about the data, and it is the whole of what every row in the
canonical audit inventory asserts; a re-measurement can move a comparator relative to its baseline, but a
baseline can never cross a threshold defined to equal it. The equivalence is
machine-checked as `bracketed_at_baseline_bar_iff` in `formal/AILM/GateVerdict.lean`.

Because a reader meets the verdicts in the audit's output before meeting this
paragraph, the property is named in the code as well: `gate_is_baseline_relative`
in `src/ailm/gates.py` reports whether a row's threshold equals its baseline, and
`scripts/audit_gates.py` prints that beside every affected verdict and counts the
affected rows in its summary. The predicate is deliberately *not* an extra
verdict. Whether a verdict was predetermined is a fact about how the gate was
constructed, not about how the classifier behaves, and the six-way
classification is fixed by theorems in the Lean development that a new case would
invalidate. What carries evidential weight on a baseline-relative row is
therefore not the verdict but the two measurements beside it — that the comparator
beats the baseline, and that the shipped model does, the latter reported as the
audit's `OUTCOME` line and, where the margin is small, backed by a paired test
rather than by a bare number.

![**What to see: every gate in this project is placed on one number line between a trivial baseline and a typed comparator, so a vacuous, unreachable, or otherwise mis-specified threshold shows up as a defect at a visible position rather than as a pass/fail label alone.** The gate taxonomy on a number line: the trivial baseline and the typed comparator bracket the region in which a threshold is bracketed. A threshold on the baseline side is vacuous (Stage 2B's row); one beyond the comparator is unreachable when the comparator is a model-class bound (Stage 1's row) and above-reference when it is an exceedable reference; both defects in this project are labeled at their measured positions, and the retained case-only clause ablation is drawn as a third measured row at its own. That row is the figure's only non-passing one and its only one whose typed comparator loses to its own bar: the type oracle scores worse than the type-free baseline, so no threshold could discriminate between them, the bracketed region does not exist on that row, and the figure draws neither it nor a width for it. Its gate is its baseline rather than a retired declaration, so the two coincide on one marker rather than being drawn twice. The bracketed zone on each row that has one states its own endpoints and width, because a shaded region says a band exists and only a number says how much room a threshold ever had inside it. Each measured row also states the population it was scored on with its unit, the corpus it was fitted on, and the exact test with its p where the row's verdict rests on one, or that no paired per-item test attaches to that measure where it does not. A threshold classified as bracketed while lying outside that region, either defect marker plotted on the wrong side of its comparator, a passing row's shipped model outside the band the panel draws it inside, a paired test covering more pairs than its row has units, a bracketed band drawn on a row whose comparator does not beat its baseline, or a recorded outcome that disagrees with the model-against-bar arithmetic would falsify this panel.](../../output/figures/gate_taxonomy.png){#fig:gate_taxonomy width=85%}

[@fig:gate_taxonomy] is the same statement drawn once per gate. The useful property
of the picture is that it makes the two defects visibly symmetric — one threshold
falls on the wrong side of $b$, the other on the wrong side of $c$ — which is not
obvious from either defect considered alone.

The audit is also executable rather than narrative. `scripts/audit_gates.py`
re-derives the classification of [@eq:gate_verdict] for every implemented stage
on every run, fixes its inventory in `AUDITED_GATE_IDS`, and exits non-zero when
a verdict changes, when the model lands on the wrong side of its own bar, or
when a gate stops being audited. The same experience distilled a pre-build
checklist — metric and direction fixed in code before any number exists, the
trivial baseline and a typed comparator measured on the scored population, the
negative control written before the model — that no gate in this project was
built under and that is recorded as what the next gate revision should satisfy.
The script's exit conditions, its restatement caveat, and every checklist step
are set out in [@sec:audit_trails].

## Level 1: a letter bigram fitted on Brown against a restated context-free bar {#sec:letter_level}

### Letter states and Brown corpus counts

This subsection fixes the letter alphabet Level 1 predicts over and names the
corpus its transition counts come from.

**L1 is declared as a GNN file.** `gnn/models/L1_letter.gnn.md` fixes the letter transition, the identity emission and the flat start prior; every dimension and slot role in this section is read from it, and [@tbl:parameters] checks the built agent against that declaration. The spec does not generate the builder — both are written by hand, which is why the check is worth running.


Level 1 models a single character at a time. The alphabet in
`src/ailm/levels/letter.py` is lowercase a through z plus space, period,
apostrophe, and hyphen, sorted so that indexing is deterministic and every
matrix row and column has a fixed meaning across runs. A uniform distribution
over that alphabet has perplexity 30, which is the
number every subsequent comparison is read against. `normalize_text` lowercases
input and maps every character outside the alphabet to a space, so the model
never sees a symbol it has no state for.

The transition is trained on the Brown corpus [@francis1979brown] and reaches
805 distinct bigram types over
5,856,736 bigram tokens. Normalization is documented in
`docs/provenance/sources.md` and includes two choices that matter: whitespace
runs collapse to a single space, and spaces immediately preceding a period or
apostrophe are stripped. Brown's tokenizer emits punctuation as separate tokens,
so without those steps the space-space pair would dominate every other bigram
and no letter would ever transition directly into a period, which would distort
exactly the end-of-word statistics the level is supposed to learn. Counts are
vendored as `vendor/brown_counts.json`
(14 kB, sha256 `778e2b31740b`) and
verified byte-identical on regeneration, so the level's tests need no network
access and the training data is pinned rather than described.

### Bigram transition matrix

`build_letter_transition` accumulates the counts into a square matrix with default add-1
smoothing, then normalizes each column so that
$\mathbf{B}_{ij} = p(\text{next} = i \mid \text{current} = j)$. Smoothing is not
cosmetic: without it, the matrix would assign zero probability to any legal
English pair absent from Brown, driving [@eq:perplexity] to infinity on held-out
text.
`build_letter_agent` then places that matrix in the transition slot and sets the
emission to the identity, since observing a letter reveals the letter state
exactly. [@fig:letter_transition] shows the result.

![The Level 1 transition `B`, a square matrix over the letter alphabet with entries $p(\text{next} \mid \text{current})$ on a log color scale, estimated from 5,856,736 Brown corpus bigram tokens. Columns index the current letter and rows the next letter; the visible structure — the vowel band, the `q`-to-`u` near-determinism, the space column's broad spread — is the entire learned content of the level. A row/column reversal, a non-column-stochastic matrix, or a matrix whose entries did not match the trained transition would falsify the panel.](../../output/figures/letter_transition.png){#fig:letter_transition width=80%}

Every entry of that matrix is a named conditional probability, and the whole of
the level's behavior is a lookup in it. There is nothing else in the level to
interpret.

## Level 2: a phoneme chain and a position-aligned letter-to-phone decoder {#sec:phoneme_level}

Level 2 holds a phone-sequence model and a letter-to-phone model, and the distinction between them is the
point of the level. Part A is the letter level one rung down: a sequence model
over phones, emission the identity, dynamics in the transition. Part B is the
first level whose emission is not the identity — letters are observed, phones
are hidden, and `A` is a learned letter-given-phone matrix. That makes Part B's
posterior an inference rather than a read-off of the observation. It does not
make it a message from one agent to another; [@sec:architecture] keeps those two
senses of *hierarchical* apart, and only the first is claimed here. Both live in
`src/ailm/levels/phoneme.py`.

**L2 is declared as a GNN file.** `gnn/models/L2a_phoneme_sequence.gnn.md` fixes the phoneme sequence model; the letter-to-phone model beside it is declared in `gnn/models/L2b_grapheme_to_phoneme.gnn.md`; every dimension and slot role in this section is read from them, and [@tbl:parameters] checks each built agent against its declaration. The spec does not generate the builder — both are written by hand, which is why the check is worth running.


The inventory is the 39 canonical ARPAbet phones with stress
digits stripped, sourced from `CANONICAL_ARPABET` in
`src/ailm/corpus/cmudict.py`. The source contributes
135,166 pronunciation entries for
126,052 base words. The inventory count is an enforced
invariant rather than a constant: naive whitespace splitting of CMUdict
[@cmudict] yields 54 distinct tokens, including
15 comment-only phantoms after `#`; stripping
comments recovers the canonical 39. A negative control asserts
that the naive path fails the count. All Stage 2 inputs are derived into
`vendor/cmudict_counts.json` (978 kB, sha256
`9f0f20746086`), verified byte-identical on regeneration
and checked against the file on disk by `tests/test_provenance_checksums.py` — a
documented checksum that has drifted from its artifact is worse than no checksum.

### Part A: phoneme bigram chain

`build_phone_transition` estimates
$\mathbf{B}_{ij} = p(\text{next phone} = i \mid \text{current phone} = j)$ from the
primary pronunciation of every CMUdict base word, and `build_phone_unigram`
supplies $\mathbf{D}$, the distribution over word-initial phones. The prior resets to
$\mathbf{D}$ at each word boundary. That is a modeling decision with a measured
justification. The alternative treats the dictionary as one continuous phone
stream, predicting each word's first phone from the previous word's last phone.
It measures worse than paying the unigram price at position zero, because there
is no dependency across a word boundary to exploit. Position zero has no
predecessor and is charged for accordingly.

Part A is computed in closed form over all
800,198 positions of the dictionary, and never over the
length-matched subset that Part B requires. The two are measurements of different
populations. That subset selects words whose letter and phone counts coincide,
which biases it toward short words with transparent orthography and, as Part B's
alignment subsection sets out, pairs even those words by index rather than by
correspondence. So no whole-dictionary claim can be read off it in either
direction, and the size of the discrepancy is not predictable from the selection
rule. [@sec:gate_audit] quantifies the gap.

### Part B: position-aligned letter-to-phone decoding

Part B observes letters and infers phones, on words whose letter count equals
their phone count, pairing the two sequences by index. The name of the task is
that sentence: **equal-length position-aligned letter-to-phone decoding**. It is
not grapheme-to-phoneme conversion, and the matrix it learns is not a
grapheme-to-phoneme alignment. The next subsection says exactly what the index
pairing does and does not buy, and the label is used consistently from here on
because phoneme prediction and position-aligned letter-to-phone decoding have
different targets, not merely different difficulty.

The shipped builders, the GNN specification file and every recorded claim label
at this level nevertheless carry the string `g2p`. Those names are the
implementation's, fixed before the task was named correctly, and the construct
they denote is the one this subsection names. They are left unchanged because a
claim label is an identifier that the ledger, the tests and the specification
registry resolve, not a description of a construct; renaming them would break
the single-writer discipline that makes every number here traceable, in exchange
for a word. Level 6 carries the same split for the same reason
([@sec:narrative_level]), and the rule is the same in both places: read the
identifier as a key, and the surrounding sentence as the claim.

`build_g2p_emission` estimates
$\mathbf{A}_{lp} = p(\text{letter } l \mid \text{phone } p)$ over co-indexed
pairs, column-stochastic over letters for each phone; `build_g2p_transition`
estimates the phone bigram on the training split only; `build_g2p_start`
supplies the distribution over first phones. `decode_phonemes` then runs
[@eq:filtering] one letter at a time and takes the posterior mode at each
position. This is the first level whose emission carries information, and
therefore the first level where the inference does real work rather than reading
the state off the observation. [@fig:phoneme_matrices] shows both learned
matrices side by side; their differing shapes are the visible form of the point
[@sec:architecture] argues structurally.

![Level 2's learned emission and transition, computed at generation time through the same
`build_g2p_emission` and `build_phone_transition` calls the shipped agent uses.
Left: the letter-to-phone emission $\mathbf{A}$,
30 letter observations over
39 phones, counted over letters and phones that share an index
in a length-matched word. Its bright ridge is that co-occurrence and is read as
nothing more: where a phone's conventional spelling is one letter the ridge cell
also happens to be a grapheme-phoneme correspondence, and where the phone is
spelled by a digraph, or the pairing hands its letter to a neighboring phone,
the ridge cell is an artifact of the pairing rule. The panel is therefore not
evidence of a learned phonographic relationship. Right: the phone-bigram
transition $\mathbf{B}$ on the same log color scale. The panels' different
shapes are the point: an emission and a transition here cannot be confused by
slot alone, which is exactly the mis-wiring class Part A could not detect
structurally. A rebuilt emission whose columns were near-uniform — each phone's
letter mass spread flat across the alphabet, so that the co-index statistic
carried no information at all — would falsify the caption's reading of the left
panel, and the peak cells printed beneath the panels are re-derived from the
matrices on every build, so the figure refuses to draw numbers the model does
not contain.](../../output/figures/phoneme_matrices.png){#fig:phoneme_matrices width=95%}

#### Alignment restriction and excluded corpus coverage

Learning $p(\text{letter} \mid \text{phone})$ requires knowing which letter
emitted which phone, and CMUdict does not say. Equal letter and phone counts
make the *index pairing* unique — zip the equal-length sequences and every letter has
exactly one partner, with no insertion or deletion left to place. They do not
make those pairs grapheme-to-phoneme correspondences. English spells a phone
with a digraph, spells multiple phones with a single letter, and leaves letters silent,
so a word can carry as many letters as phones and still require a many-to-many
alignment. `use` = `Y UW S` has three of each, and the index pairing counts
`u`→`Y`, `s`→`UW`, `e`→`S`; all three are wrong, because `u` spells `Y UW`
together, `s` spells `S`, and `e` is silent. `one` = `W AH N` counts `o`→`W`,
`n`→`AH`, `e`→`N`, and is wrong the same way. Both words sit in the training
half of the split, so those counts are in the shipped matrix.

The consequence runs deeper than a biased sample. The learned emission can
encode systematically wrong local correspondences on words the selection rule
admits, which fixes what $\mathbf{A}$ is: a statistical relation between
co-indexed symbols, not a learned grapheme-to-phoneme alignment. The G2P
literature treats alignment as a latent many-to-many problem for exactly this
reason, and joint-sequence models over graphone units are its classical
probabilistic answer [@bisani2008joint]. AILM does not answer it. It declines to
guess an alignment on unequal-length words, because a guessed alignment is
fabricated training data, and it declines to run expectation maximization,
which is a research contribution with its own convergence diagnostics and
held-out likelihood check and cannot be smuggled into a level gate as a
preprocessing step. What is left is the index pairing on the length-matched
subset — a stipulation rather than a discovery, and the level is scoped to it.

The rest of the cost is stated rather than absorbed. The length-matched subset
is 25.7% of the dictionary
(25,936 training and 6484 held-out
pairs). It is held out by a stride over the sorted pair list: every
fifth index is test and the rest are train, so the
split is reproducible from the artifact with no seed to record. The stride
matters and is worth stating explicitly rather than leaving to the word
*sorted*. The list is alphabetical, so a contiguous cut would put whole
initial-letter neighborhoods on one side, while an interleave distributes them.
It does: the initial-letter distributions of the two sides differ by
0.00085 in total variation. What the interleave does
*not* remove is morphological neighbors: 82.5%
of held-out words share a 4-character prefix
with some training word. On a task whose whole subject is letter-to-phone
regularity, that overlap is partly the phenomenon rather than contamination.
Still, it bounds how independent the held-out score is, and it is recorded here
rather than left for a reader to infer. 74.3% of the dictionary
is discarded, and not at random: length-matched words skew short, skew
transparent, and skew away from silent letters, digraphs, and vowel clusters.
`through` and `knight` are excluded; `cat`, `stop`, and `plant` are not. Every
accuracy figure below is therefore a claim about position-aligned
letter-to-phone decoding on that subset. It is not a claim about
grapheme-to-phoneme conversion, and it is not a claim about English.
Reconstructing the same subset would still not make it comparable with a
published G2P system, because such a system predicts a word's pronunciation
while Part B predicts the phone co-indexed with each letter under a pairing the
evaluation itself imposes. [@sec:limitations] carries this forward as the
sharpest restriction on the level's generality.

## Level 3: a word level factored into identity and position {#sec:word_level}

Level 3 is where the design document's specification stopped being
implementable. Words were to be paths through a letter–phoneme trie, with the
transition encoding trie-constrained letter steps. Measured over the
length-matched letter-to-phone vocabulary, that trie has
78,398 nodes and 78,397 edges, so a dense
`pymdp` transition over trie nodes is 24.6 GB at float32 and
0.0013% occupied; extrapolated to full CMUdict it is roughly
0.37 TB. No sparse path is reachable through
`pymdp.agent.Agent`'s public API. That is a statement about the API and not
about the mathematics, and the difference is worth stating because the shorter
claim — that `pymdp` has no sparse path — is false: its JAX backend contains
sparse contraction machinery, and the transition contraction this level uses is
already sparse-capable. What `Agent` does is reject sparse arrays at
construction and lose sparsity across the filtering loop. Under two patches at
those two sites a sparse-transition agent reproduces the dense agent's
posteriors exactly, which is what makes the restriction an interface boundary
rather than a limit of the method. This is not a performance problem to optimize
but a representation that does not fit, and the distinction matters: the response to
the first is tuning, and the response to the second is a design decision that has
to be recorded with its rejected alternatives.

**L3 is declared as a GNN file.** `gnn/models/L3_word.gnn.md` fixes both hidden factors, word identity and position within the word, and the emission over letters; every dimension and slot role in this section is read from it, and [@tbl:parameters] checks the built agent against that declaration. The spec does not generate the builder — both are written by hand, which is why the check is worth running.

### Word identity and position factorization

The shipped level factorizes the hidden state into two `pymdp` factors — word
identity over a bounded vocabulary of 2000 types, and position
within the word — with a single letter modality depending on both through
`A_dependencies = [[0, 1]]`. The emission is deterministic: `A[k, w, p]` is one
exactly when letter `p` of word `w` is symbol `k`, or when `k` is the end-of-word
symbol and `p` has passed the word's last letter. `B` over word identity is the
identity matrix — a word does not become a different word part-way through being
spelled — and `B` over position is the `+1` shift.

The trie is not discarded, it is *induced*. After observing a prefix, the
posterior over the word factor is supported on exactly the vocabulary words
carrying that prefix, which is the trie node. That correspondence is checked
against a live agent rather than argued: `test_posterior_support_is_the_trie_node`
compares the posterior's support against the set of vocabulary words carrying
each prefix, and a companion test confirms the support genuinely narrows as the
prefix lengthens, so the equality cannot pass by returning everything.

What those tests establish is *prefix-support equivalence*, and the claim is
stated at that strength deliberately. The map from a factored state $(w, t)$ to
the trie node for the prefix $w_{1:t}$ is many-to-one — every word sharing a
prefix lands on one node — so it is a projection rather than a change of
coordinates, and the factored level retains a word identity the node has already
aggregated away. Promoting the correspondence to equivalence of the two
stochastic processes would require showing the projection is a lumping: that the
prefix partition is a probabilistic bisimulation, with each node's mass equal to
the summed prior mass of the words it aggregates and the trie's transition equal
to the pushforward of the factored one. This project neither proves that nor
relies on it; every measurement below is stated over the factored model.
[@fig:word_matrices] shows the level's matrices and [@fig:trie_narrowing] the narrowing they produce.

![The level's parameters, read directly off the live builders. Top: the
deterministic emission path `A[·, w, ·]` for two vocabulary words, one short and
one long, chosen at draw time as the most frequent of their lengths — each column of a panel is a
one-hot on the word's letter at that position, then on END. Bottom left:
`B` over position, the +1 shift absorbing at the final slot. Bottom right: the
head of the 2000-word prior `D` — the level's *only* learned
quantity. Every cell is a conditional probability a reader can check against
the word's spelling; a top-row panel with two marks in one column, a shift off
the sub-diagonal, or a prior head disagreeing with Brown frequency order would
each falsify the corresponding structural
claim.](../../output/figures/word_matrices.png){#fig:word_matrices width=94%}

![The induced trie, measured rather than asserted: how many of the
2000 vocabulary words remain consistent with a prefix, per
prefix length, over all 78,241 held-out tokens (mean and
median), with one worked example annotated prefix by prefix. Support *growing*
with prefix length anywhere would falsify the induced-trie correspondence, and
the data function refuses to draw rather than show it. Support collapsing over
the short prefixes displayed is the mechanism behind the lexical channel of
[@sec:hierarchy].](../../output/figures/trie_narrowing.png){#fig:trie_narrowing width=92%}

The factorization is also cheaper by an order of magnitude for the same
vocabulary and the same coverage, because it spends states on words rather than
on prefixes. It puts word identity in the model explicitly, which is what a word
level ought to have — under the trie encoding the word is recoverable only by
asking which terminal node was reached.

Because every factor except the word prior is deterministic, the probability the
model assigns to a word's full emission path collapses exactly to that word's
unigram probability. That identity is the level's load-bearing invariant, and
[@sec:word_gate] is where it earns that description.

### Vocabulary scope and closed-form extension

The vocabulary bound is a scoping decision, not a tuning parameter.
76.4% of held-out alphabetic Brown tokens are in vocabulary;
the model assigns the remainder **no** probability at all, not a small one, and
every number in this section describes that sublanguage. The functions refuse to
score an out-of-vocabulary token rather than falling back to a smoothed value,
because a smoothed number would report a result for an input the model is
undefined on.

The corpus population is partitioned before frequency fitting by a
category-stratified, revision-bound hash of whole-document identities. Of
500 authenticated Brown source documents,
448 train and 52 are held out,
with 0 document overlap. Both derived artifacts retain
the ordered file identities, their length-framed digest, the exhaustive split,
and a document index for every evaluation token; the loaders reject a duplicate,
cross-arm alias, category relabeling, count drift, or token mapped outside the
held-out arm. All 15 Brown categories contribute:
15 are represented in training and
15 in evaluation, so the corpus's genre-blocked file
order cannot collapse evaluation onto its last genres. Thus
the word-frequency prior and its evaluation population are source-document
disjoint and genre-stratified, not merely token-position disjoint.

The refusal to score out-of-vocabulary tokens is correct and it has a cost, and until the complete held-out
stream was vendored the cost could not be measured. `vendor/brown_word_counts.json`
carries the evaluation stream already filtered to the lexicon, so the tokens the
level excludes were absent from the repository entirely and the coverage figure
above was a definition rather than a measurement.
`vendor/brown_heldout_stream.json` is the unfiltered denominator —
102,404 tokens over 12,930 types, of which
24,163 (23.595757978203977%) fall outside the
lexicon — and filtering it to the lexicon reproduces the shipped evaluation
stream element-for-element, which is asserted as a test rather than assumed.

What it shows is that coverage selects an easier population, and the *comparator*
is what shows it. The letter-bigram bar is defined on any alphabetic string, so
it can be scored where the word level cannot, and scoring one model on three
populations isolates the population from the system. It reaches
9.054720916239274 on the covered sublanguage,
10.330468611764502 on the whole held-out stream, and
13.27373940157792 on the excluded remainder alone.

The obvious mechanism is length, and it is not the mechanism. Covered tokens
average 3.839508697485973 letters against
7.2394984066548025 for excluded ones. That difference invites
a simple story: a frequency-ranked lexicon selects short words, and short words
are easier for a letter model. Two marginal means cannot support that story,
and stratifying by length refutes it. Split the data into
14 length buckets, each holding at least
30 tokens on both sides: the excluded tokens are
harder in 12 of those buckets. Next,
standardize both per-emission scores to the covered length distribution, weighting
each length-$L$ bucket by its covered token count times $L+1$ emissions. That
gives 9.054720916220827 against
14.80835728699474. Matching on length does not shrink
the gap; it *widens* it, from 0.38250133238192774 nats to
0.4919054330492272, a change of
28.60227962763262%. So on this per-emission estimand,
length was masking part of the effect rather than producing it.

What the effect actually is remains unmeasured, and saying so is the honest
stopping point. The plausible account is that in-vocabulary tokens are the
frequent, orthographically typical ones whose letter transitions dominate the
token-weighted counts the comparator is estimated from, so the covered
sublanguage is easy for the bar in the same way it is easy for the level. That
is a hypothesis this section does not test, and it is recorded as one.

This does not make the Stage 3 comparison wrong. It is a paired comparison on
identical tokens, and both systems meet the same easier population. What it does
is fix what the comparison is *about*: the level's reported gain describes
76.4% of running text, and the 23.595757978203977%
it says nothing about is the harder part. The word level is not scored on the
excluded tokens under any smoothing rule, here or anywhere, because that would
be the smoothed number this section refuses above.

Stage 3 is measured on Brown rather than the Penn Treebank (PTB) the original gate
named. The full treebank is licensed through the Linguistic Data Consortium
(LDC), and `nltk` ships only a partial sample
under research terms, so a number reported against "PTB" would have named a
corpus this project does not have. Brown is already vendored, checksummed and
byte-reproducible, and is the corpus Level 1 trains on. All Stage 3 inputs are
derived into `vendor/brown_word_counts.json`, verified byte-identical on
regeneration and checked against the file on disk.

## Level 4: case roles inferred over a UD-derived label, scored on both populations {#sec:case_level}

Level 4 assigns each token of a sentence one of the 8 case
roles of the CEREBRUM inventory [@friedman2024cerebrum]. The construct it
infers is a syntactic role, not grammatical case. Universal Dependencies
English EWT marks case morphologically as Nominative, Accusative and Genitive
only, and marks it on pronouns only [@marneffe2021universal; @nivre2020ud; @silveira2014gold]: in the
split this project vendors, every token carrying `feats[Case]` is a pronoun.
The other 5 roles — dative, instrumental,
locative, ablative and vocative — have no morphological ground truth in this
corpus in principle, and every label they carry comes from an explicit
heuristic over `deprel` and adposition lemmas. So does every label carried by
a non-pronoun in the other 3.

**L4 is declared as a GNN file.** `gnn/models/L4_case.gnn.md` fixes the case-role transition and the learned emission over UPOS tags; every dimension and slot role in this section is read from it, and [@tbl:parameters] checks the built agent against that declaration. The spec does not generate the builder — both are written by hand, which is why the check is worth running.


That most of the inventory has no morphological realization in English is a
property of the construct, not a defect of the corpus. The point dates to
Fillmore's case grammar [@fillmore1968case]. Fillmore argued that case
relationships belong in the base as primitives, and that "subject" and "direct
object" are proper only to the surface structure of some languages. His own
inventory (agentive, instrumental, dative, factitive, locative, objective,
later benefactive) is deliberately semantic: he did not expect English
morphology to mark it. Dative, instrumental, and locative are deep cases in
Fillmore's own inventory, and they are among the
5 roles this level carries without
morphological ground truth. Ablative is not one of Fillmore's cases. Vocative
cannot be one by his definition, since it is not a relation between a nominal
and the predicate. The lineage is therefore partial. The part that transfers is
the general point rather than the list: a case inventory can be semantic and
still have no morphological realization in a particular language. Naming
Fillmore does not license the heuristic used here; it identifies what the
heuristic is a heuristic *for*. That is why the scope of the partial fidelity
check below matters rather than being a formality. Fillmore himself later
reopened the question of whether a fixed inventory is the right object at all
[@fillmore1977reopened], and this section inherits that caution along with the
frame.

The honest description of the level is therefore heuristic syntactic case-role
inference over an inventory of 8 roles,
3 of them partly anchored in UD morphology and
5 of them supplied entirely by the
heuristic. Naming it that way is what makes the rest of this section
measurable rather than merely disclosed. A level claiming grammatical case
would have to treat its labels as ground truth and report accuracy against
them; a level claiming a labeling scheme can report that scheme's fidelity
against the morphology that does exist, and can state its gate over the
population where the scheme actually applies. Both appear below.

The hidden factor is the case role, `A` is a learned emission
$p(\text{UPOS} \mid \text{case})$, and `B` is the case-role transition along
the sentence. Structurally it is the letter-to-phone model one level up:
two different alphabets, observations of one and inference over the other. It
is implemented in `src/ailm/levels/case.py` and evaluated over
2001 sentences and 25,148 tokens from
318 authenticated EWT `# newdoc id` blocks. Those raw
document identities are reconstructed before either label population is
filtered. A deterministic stride assigns 254
documents to training and 64 to test, with
0 documents in both arms. Gold and derived populations inherit that one partition;
filtering can remove a document from a population, but can never move one
across the split.

### Case-level observation alphabet

The model observes the token's universal part-of-speech tag and nothing else.
Neither `deprel` nor `feats[Case]` reaches it, and the restriction is recorded in
the vendored artifact's own provenance rather than left to the implementation.

The reason is that the evaluation labels are *derived from* those fields. A model
observing `deprel` would be reading its own answer key: agreement would approach
perfect agreement and would measure nothing about case-role structure at all.
This is the sharpest trap in the project so far, and it is worth being precise
about why. Every earlier mis-wiring made a model *worse* — anti-informative,
collapsed, upside down — and a suspicious number is something a reviewer can pull
on. This one makes the model *excellent*. There is no numeric signature to notice,
no degraded score to investigate, and a gate stated purely as a threshold would
pass comfortably while certifying a circularity. The only possible guard is
structural, so the observation alphabet is checked for what it is not allowed to
contain. [@fig:case_matrices] shows the emission, transition, and start-prior arrays
with their cells annotated.

![The entire Level 4 generative model — every parameter, on one canvas, built
by the same shipped calls the live agent uses over the derived training
population of [@sec:case_level]. Left: the emission `P(UPOS | case)`; center:
the transition $P(\mathrm{case}_t \mid \mathrm{case}_{t-1})$; right: the start prior. The
readable structure IS the claim: prepositions load on the oblique roles (DAT,
INS), pronouns on NOM, and sentences overwhelmingly open in NOM. Any cell
disagreeing with a hand count of the training pairs — or a column failing to
sum to one — would falsify the corresponding builder, and the annotated values
make that check possible by
eye.](../../output/figures/case_matrices.png){#fig:case_matrices width=98%}

### Gold and derived case populations

The `gold` population is the tokens carrying UD `feats[Case]`:
3 roles, real English morphology, every one of them
a pronoun. The `derived` population is every token the syntactic heuristic can
label, and it is the population Stage 4's gate is stated over. The two are
reported separately because a merged figure would be part measurement and part
heuristic with no way to tell which part is which.

The heuristic is a research claim, not ground truth, and its fidelity is
measurable: where both labelings exist, over
1752 tokens, the syntactic rules agree with real
morphology 89.5% of the time.

That figure is narrower than it looks, and the narrowness is not a caveat but the
whole of what it measures. Both labelings exist only on tokens carrying
`feats[Case]`, which in English EWT means pronouns, in the
3 morphologically marked roles. The derived test
population is 1319 tokens spanning all
8. So the agreement figure checks the heuristic
exactly where the corpus can check it, and says nothing about the
5 roles for which this corpus has no
morphological ground truth in principle. It is not a bound on the fidelity of the
derived labeling as a whole, and reading it as one is the mistake the wording
here is built to prevent. The scoring function is named `case_agreement` rather
than `case_accuracy` so that overstating it takes deliberate effort; the same
discipline applies to the fidelity number itself.

The sufficient counts make that boundary executable. Across the full parsed EWT
development shard, the oracle labels 7288
tokens: 1798 through morphology,
3898 through the dependency-relation map,
and 1592 through the token-level
adposition route. The heuristic-versus-gold overlap contains
1752 dependency-route tokens and
0 token-level
adposition-route tokens. The reported agreement therefore supplies no fidelity
estimate for that adposition route.

That route also exposes a construct duplication rather than a parser accident.
It assigns the ADP token from the ADP's own lemma; it does not look up or relabel
the nominal head. None of its 1592
tokens carries gold `Case` morphology, while
1516 have a head that the oracle
separately labels. The ADP-token and head roles disagree in
888 of those pairs
(58.58%). This audit preserves
the measured labels and gates; it shows why neither the overlap agreement nor an
English morphological comparison can decide whether the ADP-token construct is
valid. Relabeling the nominal head would define a new research population and
would require a new recorded run, not an editorial correction.

Computing that fidelity number correctly required one correction worth recording.
The natural implementation calls the project's own case oracle on each token and
compares the result against the gold morphology — but the oracle consults
`feats[Case]` *first*, so on exactly the tokens being compared it returns the
gold label itself. The measurement reads as perfect agreement, green by
construction, and confirms nothing. The syntactic rules have to be applied
directly, bypassing the morphological shortcut, which is the comparison the
number was always supposed to make.

## Level 5: clause type, built on a case-only ablation and a canonical widened alphabet {#sec:clause_level}

This section describes the **case-only** clause model, which is the retained
ablation rather than the canonical Level 5. The canonical arm's alphabet,
population and comparators are specified in [@sec:clause_widened_method]. The
`stage5_` ledger prefix stays with the case-only arm. Decision 118 updates
[@tbl:parameters] to count the canonical widened agent; the retained ablation is
counted separately as 400 stored,
160 nonzero and 127 free
entries, excluded from the canonical total.

Its joint hidden state contains **sentence type** — declarative or
interrogative, labeled by the sentence-final punctuation token — together with
case role. Its observation alphabet is the same case-role inventory that Level
4 uses as a hidden state, which is the exact identity the canonical arm's
widened alphabet gives up ([@sec:widened_caveat]). The scored stream is
supplied directly by the
UD-derived artifact, however; it is not a posterior emitted by a running Level
4 Agent.

**This arm is declared as a GNN file.** `gnn/models/L5_clause.gnn.md` fixes the joint type-by-case state space and its transition. The canonical arm's `gnn/models/L5w_clause_widened.gnn.md` declares the wider alphabet and the agent counted in [@tbl:parameters]. Both specs are checked against their own built agents, with mutation controls. The specs and builders are independently written. The parameter recorder requires each arm to conform to its own declaration; separate crossed-spec tests require each arm to fail against the other's declaration.


The construct is that and no more. Sentence type here is a binary latent
defined by a single character. The level models no constituent structure, no
main-versus-subordinate distinction, no coordination or complementation, and
no mood beyond what the final mark encodes; none of those is present in the
observation stream, and none could be recovered from it. Calling this a clause
level would import four constructs into a measurement that contains none of
them. What the measurement does contain — what a punctuation-defined binary is
worth to case-role prediction — is a smaller question, and the one this design
can answer.

The model is a 2-component mixture of case-role Markov chains, embedded as a
single-factor `pymdp` agent over the joint (type × case) space:
16 states, a block-diagonal transition with zero cross-type mass (a sentence does
not change type mid-way, as a word does not drift mid-spelling), a
deterministic emission reading off the case component, and a prior
$P(\text{type}) \otimes P(\text{first case} \mid \text{type})$. Filtering a
case sequence accumulates evidence about the type; the type posterior is the
block marginal. The reported full-set score computes the corresponding mixture
evidence in closed form. The live Agent stores the same arrays, and its inference API is equated with
the same model on the full held-out set. Over all
178 held-out sentences (1072
positions), the Agent's `infer_states` / `update_empirical_prior` recursion
produces the sentence-type posterior argmax with
0 disagreements against the NumPy joint
forward filter over the identical matrices, and its full joint (type × case)
posterior deviates from that filter by at most
1.67e-07 at any position. The argmax count is not
the load-bearing statistic: on this population the filter's type argmax is the
same type at every position, so a transition with no case dynamics at all also
scores 0 disagreements while its joint
posterior departs from the filter by a wide margin. The posterior tolerance is
the check that separates the two, and the recorder refuses to write if either
statistic moves. This is posterior agreement between the live inference path and the
independent oracle that audits it. It is not score equivalence for the headline
mixture estimand, which remains closed-form.

The disjointness discipline of [@sec:case_level] applies one level up, and
holds structurally: the type label's source is the punctuation token's form,
the observations are syntactic case roles derived from UD dependency
structure, and PUNCT
tokens carry no case role — so the label source *cannot* appear in the
observation stream. The artifact generator asserts this at build time, and
the observation alphabet is checked to be exactly the 8 case roles.
Sentences ending "!" (68) and unpunctuated web
fragments (184) carry no label and are excluded,
stated rather than smoothed over; every Stage 5 number describes the typed
sublanguage of 846 training and
178 held-out sentences. The split is made before those
filters: the generator reconstructs all 318 raw EWT
`# newdoc id` blocks, assigns 254 whole documents
to training and 64 to test, asserts
0 overlapping documents,
and only then retains typed sentences. The retained training sequences come
from 196 contributing documents and the test sequences
from 53. Thus neither a second sentence from a held-out
document nor a filtered sentence from it can leak that document's identity
into model fitting.

### Canonical widened observation alphabet {#sec:clause_widened_method}

This is the canonical Level 5's specification. The clause level admits a second
observation alphabet, and decision 114 made that one the arm the paper
describes. The case-only stream of [@sec:clause_level] is
the derived case-role sequence, which is Level 4's hidden states read exactly
one level down, and that stream cannot represent the two parts of speech that
occupy a position in English surface order while carrying no case. The widened
arm adds one distinguished symbol for each verb and each auxiliary, taking the
alphabet to 10 symbols and the joint state space to
20 states in the same type-major, block-diagonal
arrangement. Nothing else about the receiver changes: the recognition matrix
remains the deterministic read-off of the symbol component, the transition
retains exactly zero cross-type mass, and the prior remains the type prior
times each type's first-symbol distribution. The estimators are the shipped
ones, called at the larger alphabet rather than transcribed a second time.

Widening an observation stream toward the label is the failure this level was
built to avoid, so the disjointness assertion is strengthened rather than
relaxed. Sentence-final punctuation remains the type label's source and
punctuation emits nothing; a verb or auxiliary contributes its part-of-speech
tag alone, never its lemma, its morphological features, or its surface form.
Both properties are asserted per sentence when the artifact is built, by
recomputation rather than by inspection: every punctuation token is offered to
the extractor alone and must yield nothing, and the recorded sequence must
equal what the extractor produces from a view of the same tokens in which
lemma, features and form are replaced by a single shared sentinel. An
extraction that consults any withheld field disagrees with itself under that
substitution and the build stops. A planted lemma leak and a planted
punctuation emitter are both rejected by that assertion in the test suite.

The arm inherits the authenticated document partition before exclusions, so
its arms are document-disjoint and its intervals resample whole source
documents. One rule genuinely differs and is stated rather than inherited in
silence: a sentence is usable here when it contributes at least the minimum
number of *widened symbols*, where the shipped artifact requires that many
*case-bearing tokens*. The two populations are therefore nested rather than
equal, and per-position perplexity is never compared between the case-only arm
and the widened one. The
comparison that is valid across them is a per-sentence statistic about the
same binary variable on the same sentences, and that is the one reported.

The arm was specified before its artifact was regenerated. The alphabet, the
partition, the receiver, the comparators, the gate, the controls, the
predictions and the falsifiers were fixed in advance, and the outcome was not
re-selected afterwards. The specification was written within the development
session and has no independently timestamped prior artifact, so every quantity
it produces is recorded as descriptive and none of its *p*-values joins the
frozen multiplicity family.

## Level 6: document-position terciles, registered before the held-out score {#sec:narrative_level}

One step up from the sentence sits the document. Level 6's observation alphabet
is the sentence-type component named inside Level 5, and its hidden state is the
**document-position tercile**: the opening, middle or closing third of a document
by sentence index. Its scored sequence comes from punctuation-defined corpus
labels, not a running Level 5 Agent's posterior. The shipped code,
the figure panels and the recorded claim labels name this variable `phase`;
that name is the implementation's, and the construct it denotes is the
tercile.

**L6 is declared as a GNN file.** `gnn/models/L6_narrative.gnn.md` fixes the
three document-position terciles and the emission over sentence types; every dimension and slot role in this section is read from it, and [@tbl:parameters] checks the built agent against that declaration. The spec does not generate the builder — both are written by hand, which is why the check is worth running.


The tercile is a position, not a narrative phase. Nothing in this corpus
annotates discourse structure, so no measurement here can distinguish an
opening third from an exposition, a setup or a lede: the construct is the
ordinal partition, not whatever that partition might correlate with. Naming
it that way costs the level its most interesting-sounding claim and keeps the
one it can test — whether sentence-type probabilities differ by where a sentence
sits in its document.

The operationalization is deliberately austere. A tercile is a function of
position, so every parameter is a count; there is no expectation-maximization
(EM) or fitted latent
segmentation. That austerity does not make the evidence confirmatory: the
development population and analysis family were reused.
The tercile transition is left-to-right by construction, and its strictly
upper triangle is exactly zero (0): in
$B[\text{next}, \text{current}]$ orientation a non-decreasing position puts
mass *on and below* the diagonal. This project's own first draft stated the
invariant with the wrong triangle — decision 1's orientation trap arriving in
prose rather than in a matrix. The error surfaced only because the number was
recorded rather than asserted (`docs/decisions.md` entry 26).

The directional recognition prediction and the prior-intervention anchors are
recorded in decision 25. This development-session
protocol does not establish independent confirmation: the development population
was reused, and uncertainty must respect the document sampling unit.

## Generation by expected-free-energy policy selection {#sec:generation}

Every level described so far is a recognition model: observations arrive and
beliefs are updated. Generation is the other direction, and until this section it
was apparatus without exercise — the expected free energy of [@eq:efe] was
specified, `C` was zero at every level, and each level had a single trivial
action, so there was nothing to select between.

That is not a missing feature so much as a half-specified model. A generative
model that cannot generate has only been used in one direction.

### Next-letter actions and policy selection

The generating agent is the letter level with a **controllable** transition. It
has one action per symbol, and $\mathbf{B}[\text{next}, \text{current}, k] =
\delta(\text{next} = k)$: action $k$ emits letter $k$. Choosing an action *is*
choosing the next symbol, which is what makes policy selection generation rather
than an analogy.

Two quantities then compete. The policy prior $\mathbf{E}$ is the learned Brown
bigram column for the current letter — the agent's **habit**, what it would do
absent any goal. The preference vector $\mathbf{C}$ is supplied **top-down** by
the word level: given the prefix emitted so far, the lexicon posterior is
supported on the vocabulary words carrying that prefix, and marginalizing it over
each word's next letter gives a distribution over what should come next. pymdp
combines them as

$$
q(\pi) \;=\; \sigma\!\left(\gamma\, \mathrm{neg\text{-}EFE} + \ln \mathbf{E}\right),
$$ {#eq:policy_posterior}

with $\gamma$ the policy precision that [@tbl:generation_sweep] sweeps.

This is the first place in AILM where one level sets another's `C`, and the first
non-flat `C` anywhere in the project. It is also the second inter-level channel
attempted. The first ran from letters into phonemes and measured *harmful*
([@sec:limitations]); the reason was structural, that it carried strictly less
information about the observed symbol than the observation itself. This channel
is not in that position — it carries the lexicon, which the letter level does not
possess in any form.

Under a controllable transition the corpus statistics change role rather than
place. The bigram table is identical to Level 1's; it is the policy prior here
and the transition there, because the transition has become a structural
consequence of the chosen action. A specification annotating it as the transition
in this configuration would describe a model that cannot choose what to emit,
which is why the generative agent carries its own GNN model rather than a flag on
the recognition one.

### Why membership cannot discriminate generation quality

The obvious measurement is the fraction of generated tokens that are real words.
It is **green by construction**. The preference floors any letter that would
leave the trie, so at any usable precision the generator essentially cannot emit
a non-word: membership is 100% at the operating
point, and also 100% at a precision where
the model emits 1 distinct word forever.

A metric that cannot fail is not a gate, however good the number looks. Had
membership been the gate, the degenerate generator would have passed it
perfectly. This is the fourth occasion in this project on which the first metric
anyone reaches for turned out to constrain nothing, and the pattern is now
familiar enough to be worth stating as a rule: before adopting a metric, ask what
configuration would fail it, and go and build that configuration.

The gate is instead the **total-variation distance between the generated word
distribution and the lexicon's**, against the exact word distribution induced by
ancestral sampling from the letter bigram — the strongest generator with the
same letter statistics and no top-down channel. Producing a real word is easy
once the channel exists; producing *the* in the right proportion is the actual
claim.

### Generation controls and gate falsifiability

Three, and the first is an exact identity rather than a comparison. With
$\mathbf{C} = 0$ the policy posterior reduces to the habit, so the generator must
reproduce ancestral sampling **character-for-character** under the same seed. Not
approximately — the two are the same stochastic process, and the recorder refuses
to write if they differ. That anchor is what licenses reading the whole sweep as
an effect of the top-down channel rather than of anything else in the pipeline.

The anchor is paired with its own non-vacuity check, since a bug making the
preference always zero would satisfy it while proving the opposite. And the third
control replaces the preference with its letter axis reversed — a fixed
involutive permutation, so it keeps its entropy and loses only its meaning.
Membership falls to
0%. Replacing the preference with a
uniform vector instead would have confounded content with entropy, which is why
the reversal is the stronger control — the same argument that governs the
inter-level coupling null.

Finally, the closed-form policy posterior that drives generation is checked
against pymdp's own `infer_policies` primitive: the historical fast generator never calls
`infer_policies` itself, and the recorder measures the worst absolute deviation
between the two posteriors over 120
(letter, prefix) states, refusing to write above its tolerance; the recorded
value is 1.16e-07. An unproven fast
path is a substitution rather than an optimization.

With the standalone generation mechanism characterized and its reduction measured,
what remains is the question the stack has been deferring: whether anything
passes between the levels. [@sec:hierarchy] states the hypothesis and its test,
and [@sec:results] reports what each channel delivered.

## Cross-level interventions and channel contracts {#sec:hierarchy}

AILM is a stack of levels ordered by descriptive scale, and whether that stack
is a hierarchy is the question this section measures rather than assumes. The
vocabulary is kept separate throughout because the distinctions are separately
checkable: a **stack** is levels arranged by scale, **composition** is one
level's alphabet or output structurally entering another's, **coupling** is one
level's probability object actually changing another's computation, and
**hierarchy** is the stronger claim that cross-scale variables participate in a
repeatedly interacting inferential architecture. AILM ships a stack with
composition, and seven registered receiver interventions, five of them measured and the case-word bridge's influence measured as an exploratory development-population sweep ([@sec:case_word_sweep]). It does
not establish the fourth from those measurements. The connected development
runtime described below is a separate engineering extension. By source kind,
five have a model-output source and two have a privileged-oracle source.
6 levels are built, and the interventions
have opposite outcomes: a letter-to-phoneme
message that made the system significantly *worse*, an L3→L1g lexical preference
that makes generation work, a lexicon-to-decoder channel that helps, an L6→L5
intervention on the receiver's joint `D` prior, and a privileged
punctuation-type-oracle→L3 word-prior intervention whose predicted harm did not
reproduce (measured in [@sec:clause_word_channel]). The oracle arm does not use an L5 posterior. The
last three are independently measured experiments, not stages in one cascade.

**Connected development runtime.** `ailm.levels.connected.ConnectedRuntime`
executes the GNN-built levels on an explicit causal schedule. Document-position
predictions set clause priors; clause predictions affect case priors; case beliefs
set word priors before spelling arrives; and prefix-conditioned word beliefs
change phone and letter predictions. Completed-token evidence updates the clause
agent, and completed-sentence type beliefs update document position for subsequent
sentences. Raw input contains normalized American Standard Code for Information
Interchange (ASCII) lexical runs and sentence
boundaries. Structured input may supply UPOS before spelling, but has no slot for
case labels, dependency relations, morphology, phonemes or sentence-type labels.
In raw-text mode, a fixed train-fitted lexical encoder supplies a soft UPOS
observation after the current word has been processed. This encoder is retained
in both connected and named-channel-disabled arms; because its input is the word
belief, its outputs can differ between connected and named-channel-disabled execution.
Bridge kernels are fitted from training inputs only. Each call resets document
state, and the composition contract specifies the subordinate reset clocks.
The resulting next-word prior can also drive bounded lexical continuation through
live L1g `infer_policies` and `sample_action` calls.

This composition is a schedule of conditional updates, not an assertion of an
exact posterior for a single global model. Local conditional reference checks,
channel ablations, scrambled messages and reset controls exercise the mechanism;
they do not by themselves establish a beneficial corpus-level effect. The
separate, locally specified GUM comparison in [@sec:connected_gum] evaluates the
fixed schedule against disabling its named channels. Its channel strength is
fixed before that evaluation rather than selected on its outcomes. The historical
stage and channel measurements below are not reinterpreted as results for the
connected runtime. The executable contract is `docs/design/connected_runtime.md`.

One of the two case-word seams carries a measured estimand as well. The
L4→L3 case-word bridge's influence sweep was registered before it ran
(`docs/decisions.md` entry 121): the estimand is the
per-token log-gain of the L3 word prior under the case-word inlet against the
bit-identical zero-weight anchor, over the case-word artifact's test
documents, restricted to lowercase exact-string matches against the Stage 3
receiver vocabulary. The scored population is
3001 tokens across
64 documents, expanding to
14,449 word-level emissions under the
whole-document clustered bootstrap; a token whose UPOS tag has no row in the
case model's observation alphabet stays in the stream under the no-message
policy (`D_α = D_0` exactly, 561 such
tokens of 4 tags). The message is the
model-sender marginalization of the case level's own per-token role
posterior; the privileged-oracle arm is a separately labeled diagnostic and
remains unexecuted. The controls are the shipped ones: the bit-identical
zero-weight anchor (the recorder records the bit-identity as
1 and the anchor's non-vacuity as
1; 1 = verified), the scrambled-word negative
control, and the declared positive control — the evaluation stream's own
pooled in-vocabulary unigram — which must lead the
scrambled control by the pre-registered per-token margin of
0.177528463 nats at every positive
weight; the recorder refuses without writing anything otherwise. The
receiver arithmetic is the L3 word level's own per-token scoring with the
per-token UPOS observation in place of the clause type, and the sweep reuses
the shipped mixing grid, `mix_prior`, and clustered summaries unchanged.
Scope, stated as the registration states it: development evidence under a
fresh registration, never confirmatory — the scored documents are the
already-scored EWT dev population in a new representation — with
cross-corpus transfer (an EWT-estimated message applied to the Stage 3
Brown-receiver vocabulary) part of the estimand and named as such, and the
recorded receiver sensitivities of the other arms describing their own
interventions and bounding nothing here.

Results with opposite signs are an anecdote unless something explains them all.
This section states the explanation, derives a prediction from it, and reports a
test of that prediction that was specified before it ran.

### Information-value hypothesis for cross-level messages

**Working hypothesis (information value).** This hypothesis separates what a
message could deliver from what it actually delivers, holding the receiver and
the loss function fixed. Fix the receiving level and score it under a *proper
predictive loss* [@gneiting2007strictly] — a loss that a forecaster minimizes
exactly by reporting its true beliefs. Under that setup, an inter-level
message has positive *potential* value to the extent that it carries
conditional information about the receiver's target beyond what the receiver
already has. Under logarithmic loss, that potential value equals exactly
$I(Y; M \mid X)$ — the conditional mutual information between the target $Y$
and the message $M$, given what the receiver already observes, $X$ — but only
for a *Bayes-optimal* receiver, one whose belief update is exact
[@jiao2015logarithmic]. *Realized* value is different: it is a property of the
receiving side rather than of the sender. It depends on how the receiver is
misspecified, on how well the message is calibrated, on where and by what rule
the message is inserted, and on how sensitive the receiver's likelihood is to
the latent variable the message moves.

Realized value is **not** bounded above by potential value when the receiver is
misspecified, and AILM's receivers are finite and misspecified by construction.
Write $q_0$ for the receiver before the message and $q_1$ for the receiver
after it. [@eq:message_log_loss] decomposes the log-loss reduction into three terms:

$$
\Delta_{\log} = I(Y; M \mid X)
  + \mathbb{E}_X\, D_{\mathrm{KL}}\!\left(p(Y \mid X) \,\|\, q_0(Y \mid X)\right)
  - \mathbb{E}_{X,M}\, D_{\mathrm{KL}}\!\left(p(Y \mid X, M) \,\|\, q_1(Y \mid X, M)\right).
$$ {#eq:message_log_loss}

The first term is the conditional mutual information above; the second and
third measure how far the receiver's beliefs sit from the truth before and
after the message arrives. Because of these extra terms, a message that moves
a badly misspecified receiver toward the correct distribution can deliver more
than its conditional mutual information alone would predict. Conversely, a
poorly inserted message can destroy performance even while that information is
strictly positive. All three terms are separately measurable, which makes the
empirical program sharper rather than weaker: source information, receiver
sensitivity, and *mismatch correction* are three distinct mechanisms, and the
channels below are read against all three.

The coupling sweep and its harm were already known when the hypothesis was
formulated. The lexical sweep's sign and *interior optimum* shape were then
registered before that sweep ran, and the sign held while the shape did not
survive the correction of decision 110; the earlier
coupling shape supplied the
motivating contrast, not an independent prospective confirmation. The
hypothesis's support is its measured channels, not a proof of universality.

Novelty — conditional information the receiver lacks — is the first term. The
lexical and clause-channel predictions were prospectively specified against
that term (decision 27 in a prior commit; decisions 18--19 within one
development session); the earlier coupling result was already in hand and
motivated the criterion.

The hypothesis is deliberately not a biconditional. "A message helps exactly
when it is novel to the receiver" is the sharper statement, and it fails in both
directions under the machinery this paper actually runs. One of the two failures
is measured inside this section.

**Novel and still harmful.** Messages here enter by linear mixing at a weight,
and linear mixing is not the combination rule a correctly structured receiver
would use. Novelty is a property of the message; the weight is not. The
lexical channel was the demonstration of this until its backoff was corrected
(decision 110). Under the defective lookup, the same
message into the same receiver helped at one weight and cost accuracy at the
highest. Under the documented rule, it helps at every tested weight instead: by
3.333 points at weight 0.7, and by
3.258 at weight 0.9. The
structural point survives the correction even though its example did not:
conditional information is identical at every weight, so a weight-dependent
sign, where one occurs, cannot come from the message's content. Finite-sample
estimation is a second route to the same outcome. The message is a statistic
estimated on a training split, so what is delivered on a held-out item can
point the wrong way even when the underlying quantity is informative.

**Redundant and still helpful.** A receiver whose own model is misspecified can
be moved in the right direction by a message that would add nothing to a
correctly specified predictor, because the message redistributes mass the
receiver had mis-assigned. That is regularization rather than information, and
it is indistinguishable from information if the only instrument is the sign of
an end-to-end effect. The punctuation-type-oracle→word-prior intervention of [@sec:clause_word_channel]
is the measured case this paper cannot rule that reading out on.

Both failure modes are about the receiver and the delivery, which is why the
hypothesis separates potential value from realized value rather than hedging the
biconditional.

The hypothesis accounts for the existing results structurally, without appeal to
their magnitudes.

The letter-to-phoneme channel carries no potential value because at position $t$
the letter-to-phone level already observes letter $t$, while the letter
level's predictive is a function of letters $0 \ldots t-1$. The message is a
strictly weaker statement about the very symbol the receiver can see, so mixing
it in substitutes weak evidence for a certainty. Accuracy fell *monotonically*
in the mixing weight, which is what a message with nothing to add looks like:
there is no weight at which saying it more loudly helps.

The lexicon-to-letter channel carries potential value because a bigram cannot
represent "only these letter sequences are words" at any order below the length
of the longest word. The lexicon is information the letter level does not
possess in any form.

### Prospective lexical-sweep prediction

A hypothesis that only ever explains results already in hand explains nothing.
The test holds the *receiving level fixed* and changes the sender: a lexical
message into the same letter-to-phone level the first channel harmed.

It does not hold everything else fixed, and the claim of control is exactly this
narrow. The two messages enter the receiver at different places — the letter
predictive folds into the observation likelihood over phones, the lexical
message mixes into the phone predictive — so message source is not the only
variable that differs between the sweeps. The factorial version of this
experiment crosses letter versus lexical source with evidence versus predictive
insertion, holding receiver, mixing rule, weight grid, and evaluation positions
fixed across every combination. The sweeps below occupy the letter-at-evidence
and lexicon-at-predictive combinations; the crossed combinations are reported in
[@sec:channel_factorial], which is where the
attribution this paragraph could not make is made.

The message is $p(\text{phone}_t \mid \text{letter prefix } 0 \ldots t)$,
estimated on the training split and applied to held-out pairs, mixed linearly
into the phone predictive. It is conditioned on the *prefix* rather than on word
identity for a reason that [@sec:case_level] makes concrete: the lexicon maps
word to pronunciation, so conditioning on identity would turn the task into a
dictionary lookup and any improvement would be circular. A prefix statistic
evaluated on held-out words can be wrong, which is the property that makes it a
measurement.

The prediction recorded before the experiment was that the channel would help,
and would do so **non-monotonically** — an interior optimum being the signature
the hypothesis expected from a message the receiver lacks.

### Fixed and varying factors in channel comparisons

This table lines up the measured channels side by side so the one
variable they do not hold fixed — where the message enters the receiver — is
visible rather than buried in the numbers.

```{=latex}
\vspace{0.5\baselineskip}
```

| | Letter → phoneme | Lexicon → phoneme |
|---|---|---|
| Message | letter predictive over $0 \ldots t-1$ | lexical prefix statistic |
| Novel to the receiver? | **no** — the level observes letter $t$ | **yes** — the level has no lexicon |
| Enters the receiver at | the observation likelihood over phones | the phone predictive |
| Shape in the mixing weight | monotone decline | **rise to a plateau** (interior argmax, not distinguished from the top weight) |
| Sign at the best recorded weight | -0.2639 points at weight 0.1 | +3.333 points at weight 0.7 |

Both figures in the last row are recorded at the weight that scored best on the
grid, so they are read the same way; for the coupling that is the *least
harmful* weight, because the decline is monotone and every larger weight costs
more. Direction and receiving level are matched, and both messages are linear
mixtures whose zero-weight end is the uncoupled decode exactly. Rows two and
three of the table are the two variables that differ together: the pair varies
novelty and insertion point at once, so it cannot attribute the opposite signs
to either alone. The coupling result was already known; it motivated the
potential-value account and the later registered lexical prediction. The lexical
result supports that prediction, while an insertion-point account of the two
signs remains available.

### Registered proper-loss estimand {#sec:proper_loss_axis}

Every channel number above is an argmax accuracy. The working-hypothesis box at
the head of this section is not stated on that axis: it is stated under *a
proper predictive loss* [@gneiting2007strictly], and accuracy is not one.
Accuracy is a threshold statistic on the receiver's posterior, so a perturbation
that moves mass without moving the argmax is invisible to it, and one that flips
a few near-ties can move it in either direction while the receiver's actual
belief about the target gets worse. The two axes can therefore disagree in both
sign and shape, and until they have been put side by side on the same positions,
a monotone decline over the whole grid is a property of the statistic as much as
of the channel. For the letter-at-evidence cell of the channel factorial the two
axes have since been put side by side on the same positions
([@sec:channel_factorial]): they agree in sign and in shape class, so for that
cell the decline is a property of the channel.

## Verification by Mutation Controls and Independent Decoder Agreement {#sec:verification}

The verification apparatus is not scaffolding around the result — it is a
deliverable of the same standing as the levels themselves. Both defects that
reached a shipped state in this project ([@sec:letter_level], [@sec:phoneme_level])
were invisible to every structural check available: pymdp validated the matrices,
the suite was green, and the reported numbers cleared the declared thresholds.
What surfaced them was a control that could fail and an instrument that did not
share code with the thing it measured. This section describes those two mechanisms. It describes the
decoder-equivalence result that lets a closed-form oracle stand in for pymdp on
full-corpus statistics, and the statistical procedures the paired claims are
judged by and what each of them assumes. It describes the two design facts —
clustered sampling units and a retrospectively assembled test family — that
bound what any of those procedures can be read to establish, the convention
under which an interval-spanning result is reported as a bound rather than as
an absence, and the ledger that binds every number in this manuscript to a
recorded run. Both
bounding facts are stated here, before [@sec:results] reports the numbers they
qualify, because a caveat that arrives after the claim it conditions has already
been read as unconditional. Each mechanism is stated with
the failure it forecloses and, where one exists, the failure it does not: a
verification layer described only by what it checks is indistinguishable from one
that checks nothing.

### Positive controls and required mutation failures

Every gate carries a positive and a negative control.

`src/ailm/control.py` defines `ControlHarness` and `GateVacuousError`, and it
offers two services that are easy to conflate and do different work. The first,
`assert_gate_is_live(good_value, bad_value, direction)`, compares a good model
against a deliberately degraded one *directly*, using the degraded score as the
bar. It raises `GateVacuousError` when the two scores are identical — a metric
that returns the same number for a working and a broken model is measuring
nothing — and it fails an assertion when the good model is merely not better.
It says nothing at all about where the declared threshold sits. The second,
`verify_negative_control_fails(metric_fn, broken_input, threshold, direction)`,
is the one the threshold enters: it re-runs the metric function on broken input
and raises `GateVacuousError` if the broken result *strictly passes* the bar.
Both inherit the strict comparison of `gate_passes` in `src/ailm/gates.py`
([@eq:gate_satisfies]), so a broken model landing exactly on the bar is treated
as having failed it. The boundary is resolved against the gate rather than
against the model, which is the conservative direction for a vacuity check.

The distinction that makes this worth the machinery: a passing positive control
establishes only that the number is achievable. A failing negative control establishes that the gate detects that planted defect
under the tested conditions; it does not establish that every faulty model fails. A threshold no broken model
can fail, or no working model can pass, is not a weak test, it is not a test — and the two defects in
[@sec:gate_audit] are both instances of exactly that, reached from opposite
directions.

What the harness does not do is worth stating with equal precision, because a
control that is trusted beyond its scope is the next defect. Neither method
measures a trivial baseline or an achievable ceiling, so neither can classify a
threshold's placement; that is the audit of [@sec:gate_design], and the two are
independent. A stage can hold a live negative control and still carry a misplaced
gate, vacuous or unreachable, and both shipped defects did.

### Independent NumPy reference decoder

`src/ailm/verification/oracle.py` is a closed-form hidden-Markov implementation in
plain numpy: `forward_filter`, `forward_backward`, `viterbi`,
`sequence_perplexity`, `markov_chain_perplexity`. Its independence is
implementation-scoped and no wider: it consumes the same fitted matrices, the
same preprocessing, and the same corpus artifacts as the shipped decoder, so it
audits the implementation of inference, not the estimation pipeline or the
modeling assumptions. It computes [@eq:filtering] in
its own array layout,

$$
\alpha_t \;\propto\; \operatorname{diag}\!\big(\mathbf{A}_{o_t,:}\big)\, \mathbf{B} \, \alpha_{t-1},
\qquad \alpha_0 \;\propto\; \operatorname{diag}\!\big(\mathbf{A}_{o_0,:}\big)\, \mathbf{D},
$$ {#eq:forward_recursion}

renormalizing at every step and accumulating log-evidence from the normalizers,
so a long sequence does not underflow; Viterbi runs entirely in log space. The
renormalization is not only numerical bookkeeping: the normalized $\alpha_t$ is
exactly the filtered posterior $q_t$ of [@eq:filtering], and each normalizer is
the predictive probability $\sum_j \mathbf{A}_{o_t j}\, \rho_t(j)$ that
[@eq:perplexity] consumes. The module's docstring pins the conventions
explicitly, because a transpose anywhere in [@eq:forward_recursion] silently
invalidates every cross-check the module is used for. Two fix the array layout:
$\texttt{emission}[i, j] = p(o = i \mid s = j)$ and
$\texttt{transition}[i, j] = p(s' = i \mid s = j)$, both column-stochastic. A
third fixes the initialization: $\texttt{start}$ is the distribution over $s_0$
*before* any observation, so the transition is not applied first. Those
conventions match pymdp's once the leading batch dimension is stripped, and the
match is asserted rather than assumed: `tests/test_phoneme.py` requires the two
implementations to agree numerically on perplexity, and the decoder-equivalence
result below requires them to agree position-for-position on argmax decoding.
Both are the same assertion at different resolutions, and either would break
under a transposed matrix — which is precisely how the Stage 1 wiring defect was
found.

The module carries one prohibition: **it must never import pymdp or jax.** Its
entire value is being an independent implementation. An auditor that shares code
with the system it audits inherits the flaw it is auditing for — a numpy oracle
built on the same array-layout helper that transposed the matrix would agree with
pymdp perfectly and prove nothing. A cross-check of this shape is what caught the
Stage 1 wiring defect, and having it as a tested module with pinned conventions
is what makes it available to every level rather than to whoever happens to write
the script again.

The prohibition is machine-checked rather than reviewed.
`tests/test_oracle_independence.py` imports the module in a fresh, isolated
interpreter and then inspects that interpreter's loaded-module table for
`pymdp`, `jax`, or `jaxlib`. Reading the module table rather than the source text
is what makes the check transitive: an import three packages deep, added by a
dependency rather than by this file, is caught on the same run. The probe carries
its own positive control. A second test plants a counterfeit `jax` module in the
child interpreter before the import and requires the probe to name it, so a probe
that had stopped looking would fail instead of reporting a clean boundary.

Two properties of the numerics bound what agreement with this module can mean,
and both are consequences of choices made against underflow rather than accidents.
Probabilities are floored before any logarithm, so an event the model calls
impossible is charged a large finite penalty instead of $-\infty$: the oracle
never returns an infinite perplexity, and a cross-check that expected an
infinity to expose a defect will not get one. And when every state assigns zero
likelihood to an observation, `forward_filter` logs a warning, charges the floor
to the log-evidence, and carries the predictive belief forward unchanged rather
than propagating a NaN through the rest of the sequence. That keeps later
timesteps interpretable, at the price of making the degenerate step quiet in the
returned value — the warning is the only signal it happened, so a long run is
read with logging enabled or not read at all.

### Bounded decoder equivalence and refusal on disagreement

Tested decoder equivalence is what licenses the oracle-computed statistics.

pymdp's per-position `infer_states` does not scale to full-corpus evaluation. The
letter-to-phone held-out set alone is 42,820 positions,
and the phoneme sequence model spans 800,198; a naive
full-corpus pymdp pass is impractical at Level 2 and gets worse with depth.
The tempting response — shrink the evaluation set until pymdp can finish — is the
subsample trap that [@sec:gate_audit] treats at length.

The response taken instead is an equivalence result. `scripts/record_stage2_claims.py`
decodes a bounded, evenly spaced stride of the held-out pairs — 13,224 of the 42,820 held-out positions — with both the pymdp agent and the oracle, and requires

$$
\sum_{t} \mathbf{1}\!\left[\hat{p}^{\,\text{pymdp}}_{t} \neq \hat{p}^{\,\text{oracle}}_{t}\right] = 0
$$ {#eq:decoder_equivalence}

before any full-set figure is recorded. The measured disagreement count is
0. The script **refuses to record** if that count
is ever nonzero, which is the load-bearing half of the arrangement: the full-set
statistics are reported as properties of the pymdp agent, and that reading is
licensed only while the two decoders remain tested-equivalent on the positions
actually checked. A cross-check that logs a warning and proceeds would convert
the license into an assumption.

Three things bear on whether the two decoders agree everywhere, and this
is the weakest of them. An algebraic derivation showing that the pymdp call
sequence and [@eq:forward_recursion] compute the same argmax for every input
would settle it by proof; running every held-out position through both
implementations would settle it by exhaustion over the population in question.
What is done instead is a finite test on a subset, and a finite test cannot
establish that two implementations are the same function — a disagreement
triggered only by inputs the subset omits is consistent with every observation
here. The result is therefore reported throughout as *tested* equivalence, and
the full-set attribution as an extrapolation from it ([@sec:limitations]);
neither is reported as proven equality anywhere in this manuscript.

The population the equivalence is demonstrated on is not the population it
licenses, and the gap is the assumption. Both decoders run over a bounded,
evenly spaced stride of the held-out pairs — bounded because pymdp is the
constraint the arrangement exists to route around, evenly spaced because a
prefix of this alphabetically sorted list would check only early-alphabet
words. The conclusion is then applied to every held-out position. What makes
that extrapolation reasonable rather than hopeful is its object: the two
implementations are being compared as *functions of the same matrices*, not as
estimators of a shared population quantity, so a disagreement would have to come
from a systematic layout or tie-breaking difference, and such a difference would
show on any word rather than on rare ones. What would defeat it is a disagreement
whose trigger is confined to inputs the stride happens to skip — an unusual
letter pair, a word length outside the sampled range — and the check as written
cannot see that. The residual risk is stated rather than closed. Additional
guards sit alongside: the subsample is resolved once, by the same `_iter_pairs`
call both decoders use, so the comparison cannot silently pair a stride against
a prefix; and the script raises if the two decoders yield different position
counts, because a length mismatch would otherwise be absorbed as a
disagreement-free result over a truncated overlap.

The later levels do not inherit that license automatically. Their score paths
are reported separately:

- **L3** uses an exact path identity for the full score. A sampled live-pymdp
  word walk agrees with that identity; the full scorer itself does not call
  `infer_states`.
- **L4** uses a NumPy forward filter for the full score. Every one of the
  1319 held-out positions is compared with live
  `infer_states` and `update_empirical_prior`: 0
  argmax disagreements and a largest posterior deviation of
  1.25e-07. The comparison is exhaustive over the
  scored population; the headline path is still the NumPy filter.
- **L5** uses exact closed-form mixture evidence for the headline score. Beside
  it, the live Agent's `infer_states` / `update_empirical_prior` recursion is
  compared with the NumPy joint filter over the full held-out set: the
  sentence-type argmax agrees at every position and the joint posterior agrees
  within 1.67e-07. The argmax count alone is not
  informative here, since the filter's argmax is a single type throughout the
  set; the posterior tolerance is the statistic that carries the check. This is
  posterior agreement, not score equivalence for the headline estimand.
- **L6** registers the observed-position conditional-table score. Its primary
  inference is document-clustered: the interval spans zero on the development
  population and excludes it on the held-back one
  ([@sec:stage6_heldback_method]). A separate descriptive
  live-filter run reports 0.3687 nats, an improvement of
  0.0043 nats; it is a different
  call path and does not replace the registered estimand.

This inventory prevents the existence of a live Agent from being used as a
synonym for a score computed by live Agent inference.

### Statistical methods and supplementary derivations

Three instruments carry every comparison in this paper, and each is stated here
with its verdict and derived in full in [@sec:statistics].

**The test is paired and exact.** Every stage comparison scores two predictors on
the *same* positions and tests the discordant pairs, because an aggregate
difference between two accuracy figures discards the pairing the design already
has. The exact conditional test is used rather than its asymptotic form, and it
is conservative rather than anti-conservative where the two disagree. One
assumption is strained everywhere and is not repaired by exactness: the scored
units nest inside larger sampling units — positions inside words, tokens inside
sentences, sentences inside documents — so `ailm.clustering` recomputes each
effect with the cluster as the resampling unit, using the
document/sentence-cluster percentile bootstrap of [@cameron2008bootstrap], and reports both.

**The family is corrected twice.** All 16 declared significance
tests form one family at familywise $\alpha =$ 0.01.
7 cross the fixed Bonferroni per-test threshold
[@dunn1961multiple] of
0.000625; 7 cross their Holm
step-down threshold [@holm1979] at the same level and under the same dependence assumptions.
The fixed threshold is a constant and is what the automated gate asserts against;
Holm depends on the whole ordered family and is the sharper scientific reading.
The Stage 6 sentence-sign test crosses neither the fixed Bonferroni threshold nor its Holm step-down threshold and remains
secondary: a reversed-phase model with worse mean log loss can still obtain a
nominally favorable sign. Neither correction
repairs the larger problem, which is that the family was assembled
retrospectively on the population every gate was tuned against.

**A null is reported as a bound.** Where a pre-registered test failed, the paper
prints the interval the data support rather than the word *none*. An interval
containing the null bounds how large the effect can be; it never establishes that
the effect is absent, and turning one into an equivalence claim needs a declared
smallest effect of interest [@lakens2017equivalence; @lakens2018tutorial] and
enough discordant pairs to resolve it. Most nulls
here do not have enough pairs, and [@sec:statistics] names the one arm that does.

### Claims ledger and recorded-run provenance

`src/ailm/claims.py` implements the ledger: every quantitative claim is tagged
`claim:<label>` in source prose and backed by an entry in `data/claims.json`
recorded from a run that actually produced the value.
`ClaimsLedger.scan_prose(*paths)` returns the labels with no backing entry.

`tests/test_claims_discipline.py` points that scanner at the real tree —
every root Markdown file, all Markdown under `docs/`, `gnn/` and `formal/`,
and every Python file under `src/` — because a gate that runs only on synthetic input
constrains nothing about the shipped artifact. A ledger built and discarded inside
a `tmp_path` fixture is a demonstration of the mechanism, not an audit of the
repository, and for an entire stage that distinction let an unbacked claim sit in
this project's own `README.md` while the suite stayed green.

The scan's two vacuity risks are guarded separately, because they are
different failures. The first risk is an empty or shrinking scan:
`assert unbacked == []` also passes when the glob silently returns nothing. To
guard against this, the discovered file set is asserted non-empty, asserted to
contain the reader-facing files by name, and asserted to contain
`formal/README.md` by *path*. The check uses path rather than name because a
name-only check on `README.md` is satisfied by the root file, while the Lean
layer's companion file would then quietly drop out of scope. The second risk
is an insensitive scanner, and this is where the honest statement differs from
the tempting one. The sensitivity control does not plant a tag in a tracked
file. Instead, it writes a fabricated `claim:` tag into a temporary file and
requires `scan_prose` to flag it, paired with a second control that writes a
*backed* tag into the same kind of file and requires the scanner to stay
silent. Together those controls say the instrument discriminates. They do
not say the repository was scanned — that is what the file-set assertions are
for, and neither guard substitutes for the other. Both are instances of the
general rule stated in [@sec:gate_design]: a check is accepted only once it has
been made to fail.

### Single-writer protection for claim recording

The ledger is read whole at the start of a recorder run and written whole at the
end, so two overlapping recorders are a lost update: the second to finish
discards everything the first wrote, with no error on either side and a green
suite afterwards. `src/ailm/ledger_lock.py` closes that race with a
single-writer lock, and the mechanism is unremarkable — an atomically created
lock file carrying the holder's process id, reclaimed if that process no longer
exists so a crashed run cannot wedge the repository, and released only by the
process that recorded itself as the holder so a writer that already lost the lock
cannot delete the lock of whoever holds it now.

The interesting part is the coverage, because for most of this project's life the
lock was taken by a small minority of the recorders and every other one wrote the
same file unguarded. A lock that one of two contending writers takes is not
mutual exclusion; it is a comment. Every claim recorder now takes it, and the
rule is enforced as a rule rather than as an inventory:
`tests/test_claims_single_writer.py` and `tests/test_benchmarks.py` each
discover the recorder set from the filesystem and require the lock in every
member, so a recorder added tomorrow is covered the moment it exists rather than
when someone remembers to list it. An enumeration would be strictly weaker,
because the edit that admits a new recorder to the list is exactly the moment
the lock gets forgotten. Both scans assert their discovered set is large enough
to be meaningful before asserting anything about it, since "no unlocked
recorders" is also true of an empty scan.

### Source Validation and Exact Hydration of Ledger-Bound Manuscript Values

This manuscript is bound to the same ledger by a different mechanism, described in
[@sec:reproducibility]: prose here carries double-brace token markers resolved from
`data/claims.json` at render time by `src/ailm/manuscript_variables/__init__.py`, and a
token with no ledger entry fails the render instead of appearing as a plausible
default. [@tbl:verification_layers] collects the whole apparatus, each layer named
by the artifact that implements it and by the specific failure it forecloses,
including the spec-conformance layer described in [@sec:architecture], which
checks a live agent's slot roles and shapes back against the declaration that
named them. The column that matters is the third one: a verification layer whose
stated purpose cannot be written as something it makes impossible is decoration,
and one whose stated purpose is broader than what it checks is worse than
decoration.

| Layer | Artifact | What it makes impossible |
|---|---|---|
| Control harness | `src/ailm/control.py` | Accepting a threshold no broken model can fail |
| Independent oracle | `src/ailm/verification/oracle.py` | Confirming a mis-wired matrix by re-deriving it with the same code |
| Oracle independence probe | `tests/test_oracle_independence.py` | The oracle acquiring a transitive dependency on what it audits |
| Spec conformance | `src/ailm/gnn/`, `gnn/models/` | A live agent whose slot roles or shapes contradict its declaration |
| Stage 2 decoder equivalence | `scripts/record_stage2_claims.py` | Attributing Stage 2 oracle-computed statistics to pymdp without checked subset evidence |
| Later-level scorer audit | `tests/test_stage3_gate.py`, `tests/test_pymdp_inference_equivalence.py`, level-specific scorers | Treating L3-L6 Agent construction as proof that each headline score called live inference |
| Paired significance | `decode_positions` / `baseline_positions` | Resting a sub-point margin on two aggregate accuracies |
| Declared multiplicity family | `src/ailm/statistics.py` | Judging a p-value without counting the tests around it |
| Derived interval convention | `paired_difference_bound_from_counts`, `TestDerivedPairedBound` | Reporting a null as an absence when the data only bound it |
| Claims ledger | `src/ailm/claims.py`, `data/claims.json` | A number in prose with no recorded run behind it |
| Single-writer lock | `src/ailm/ledger_lock.py` | Two overlapping recorders silently losing one result |
| Prose scan on the real tree | `tests/test_claims_discipline.py` | A ledger gate that passes by examining nothing |
| Token resolution | `src/ailm/manuscript_variables/__init__.py` | A hand-typed number in this manuscript |
| Prior committed learning prediction | `ailm.learning`, decisions 22–23 | A prediction first introduced with the versioned learning outcome; earlier uncommitted execution is not timestamped by Git |

: The verification layers, each named by the artifact that implements it and the specific failure it forecloses. {#tbl:verification_layers}

Two failures in this project have no row, and saying so is the honest reading of
the table. No artifact here makes it impossible to compute a paired test at a
unit finer than the one the corpus samples, and no artifact here makes it
impossible to declare a test after seeing what it will say. Both are properties
of the process rather than of any file, which is why neither can be foreclosed
by one; [@sec:multiplicity] states each and bounds what the reported p-values
can be read to mean. A table of foreclosed failures is itself a claim about
coverage, and coverage stated without its complement is the defect the third
column exists to prevent.


## Replication protocol registered before held-back evaluation {#sec:confirmatory_method}

Every threshold crossing in [@sec:results] is inference on the population the
gates were revised against ([@sec:multiplicity]). One partition of the vendored
corpora had never been scored by anything in this repository: the test split of
the UD English EWT treebank, `en_ewt-ud-test.conllu`, from which the case,
clause and narrative artifacts draw nothing. Of the frozen 16-member significance family,
4 members are computed on EWT populations, and decision 111 of the
decisions log registers, before any of them was scored, a single run of exactly
those four against that split. The outcome is reported in [@sec:confirmatory].

The protocol has five parts. The population is the whole split, vendored through
the identical generators, oracles, exclusion rules and minimum-length filters
that built the development artifacts, with no stride applied because nothing is
fitted on it; its document identities are checked disjoint from every
development document identity. Every model parameter is read unchanged from the
development train fits, and the recorder refuses to run unless the arrays it
builds equal the development artifacts' arrays element for element. Each member
keeps its frozen endpoint, test, sidedness and threshold, the per-test level
0.000625 that [@sec:multiplicity] fixes, and Holm's
step-down verdict is reported beside it over the same frozen family. Both controls run on the population before any member is scored: a planted-effect
positive control fitted on the population's own labels, which must fire, and a
label-shuffled negative control, which must not. The run happens once. The
recorder writes the sentinel claim `confirmatory_run_completed_ud_ewt_test`
before its first score and refuses on that sentinel afterwards, so the
population is spent by the act of scoring it.

The registration states a direction and a threshold prediction for each member
and names each member's falsifier, so that a result contrary to prediction is
recorded and reported as a failure and no member is re-selected, re-weighted or
dropped. The other 12 members of the family have no held-back partition
left and were not rescored: upstream CMUdict is byte-identical to the vendored
pin, which retired the lexicon members by exhaustion (decision
91), and the split diagnostics are re-analyses of
a shipped subsample with nothing to replicate on fresh data. The confirmatory
status of every label the run records follows the registration-provenance rule:
a label is confirmatory only when its registration
commit strictly precedes its outcome commit.

### Held-back scoring of the document-clustered Level 6 estimand {#sec:stage6_heldback_method}

The 4 members above are the family's *tests*. Level 6's primary estimand is
not one of them: the family member is the sentence-level sign test, while the
level's registered primary quantity is the mean per-sentence log-loss
improvement with the document as the resampling unit. The development population
illustrates the distinction: its sentence-sign statistic is nominally favorable
but crosses neither corrected threshold, while its document-clustered log-loss
interval spans zero. A held-back sign
test therefore licenses nothing about the held-back interval, and decision
115 registers that interval separately.

The registration fixes the population as the held-back documents already scored
above, the estimand as the same observation-weighted mean the development
analysis uses, the procedure as the same whole-document percentile bootstrap at
the same resamples, seed and coverage, and a prediction with the arithmetic
behind it, all in a commit that precedes the outcome commit. The models are the
frozen development fits and the recorder asserts them equal array for array, as
the protocol above requires.

Two things about it are deliberately weaker than the protocol above, and both are
recorded rather than argued around. The status is **descriptive, not
confirmatory**, even though the commit ordering would permit the stronger tag: the
population had already been spent by the run above, so an estimand registered
afterwards is registered by someone who knows that population's other results,
which is the knowledge preregistration exists to exclude
[@nosek2018preregistration]. And the recorder is
idempotent rather than one-run, because it re-analyzes a spent population through
a function of frozen inputs rather than scoring a held-back one.

Its controls are stated as a resolution rather than as a single planted size.
The smallest constant per-sentence shift the arm can resolve is computed from
the centered differences and verified from both sides: one tolerance is
planted just above it, which must be detected, and one just below it, which
must not. Whether an effect the size of the development population's own is
resolvable is itself recorded as a measurement. That measurement does not gate
the run; it only decides whether a *null* would be interpretable, and it says
nothing about a detection. A whole-document sign-flip control under a recorded
seed must not be detected. The first specification of these
controls made the development-sized plant a gate, which would have refused a
detection because a smaller hypothetical effect was unresolvable. The
amendment that corrected this is recorded in the same decision entry, before
any result of this arm was written to the ledger.
