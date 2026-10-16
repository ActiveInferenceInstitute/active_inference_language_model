# Generative Models of Linguistic Belief and Action {#sec:background}

Every matrix named in this section appears under the same name in the source:
`A`, `B`, `C`, and `D` are the arguments to the `pymdp` `Agent` constructor in
every level module under `src/ailm/levels/`. [@tbl:notation] is the symbol
registry: each symbol in it has exactly one meaning, fixed by an equation below.
The registry covers the formalism, not every letter in the paper. Among the
symbols outside it, $\alpha$ and $P$ have context-dependent meanings. $\alpha$ is a channel mixing weight in
the sweeps, a significance level in the multiplicity accounting, and, subscripted,
the forward-algorithm message $\alpha_t$ ([@sec:verification]) and the case
level's likelihood weight $\alpha_c$. $P$ is a probability wherever it takes an
argument, as in the clause level's type prior $P(\text{type})$
([@sec:clause_level]); it is the evaluation population indexing a gate verdict
in [@eq:gate_verdict], where the baseline and ceiling measured on that
population are written $b_P$ and $c_P$; and it is the size of the word level's
position factor in the memory accounting of [@sec:complexity]. Each use is
fixed by its subscript, its argument, or the sentence that introduces it; no
registry symbol is reused.

| Symbol | Meaning | Defined in |
|---|---|---|
| $\pi$ | policy (sequence of actions) | [@eq:efe], [@eq:generative] |
| $\rho_t$ | one-step predictive prior over states | [@eq:filtering] |
| $q_t$ | filtered posterior after observing $o_t$ | [@eq:filtering] |
| $\tilde{q}$ | policy-conditioned predictive over $(o_\tau, s_\tau)$ | [@eq:efe] |
| $\tilde{p}$ | preference distribution, $\mathbf{C}_i = \log\tilde{p}(o=i)$ | [@eq:pomdp] |
| $\mathbf{A},\mathbf{B},\mathbf{C},\mathbf{D}$ | the likelihood, transition, preference, and prior matrices; $\mathbf{A}$ and $\mathbf{B}$ are column-stochastic in the conditioning index | [@eq:pomdp] |
| $\mathbf{E}$ | prior over policies, the agent's *habit*: what it would do under no preference and no informative epistemic value at all | [@sec:generation] |

: Symbol registry for the formalism. Each registered symbol has exactly one meaning throughout the paper; the predictive prior is $\rho_t$, not $\pi$, precisely so that $\pi$ can mean policy unambiguously. Symbols outside this table are not covered by that guarantee — $\alpha$ and $P$ in particular are reused, and the text above says where. {#tbl:notation}

## Perceptual belief updating and variational free energy

An active inference agent holds a generative model $p(o, s)$ over observations
$o$ and hidden states $s$, and an approximate posterior $q(s)$. Perception
minimizes variational free energy, which decomposes into a complexity term and
an accuracy term and bounds the negative log evidence
[@friston2010free; @parr2022active; @smith2022stepbystep]. The complexity term
uses Kullback--Leibler (KL) divergence:

$$
F[q] \;=\; \underbrace{D_{\mathrm{KL}}\!\left[\, q(s) \,\big\|\, p(s) \,\right]}_{\text{complexity}}
\;-\; \underbrace{\mathbb{E}_{q(s)}\!\left[\log p(o \mid s)\right]}_{\text{accuracy}}
\;\geq\; -\log p(o) .
$$ {#eq:vfe}

When the family over which $q$ ranges contains the exact posterior, minimizing
[@eq:vfe] returns it, and the update is closed-form rather than iterative. The isolated filtering components meet that condition under their stated
observations and priors, though not all in the same way.
Levels 1, 2, 4, 5 and 6 carry a single hidden factor, so an unrestricted categorical
$q$ trivially contains the posterior. Level 3 is genuinely two-factor and meets
the condition for a different reason: its position factor is degenerate at every
step — $\mathbf{D}[1]$ is a point mass and $\mathbf{B}[1]$ the deterministic
shift ([@sec:word_level]) — so the exact joint posterior is already a product and
lies inside the factorized family. This exactness claim concerns each declared conditional update. It does not
establish a globally exact posterior for the connected message schedule. The
exactness is a property of those conditional updates, not of mean-field active inference in general, where a restricted $q$ need not
contain the posterior and minimizing [@eq:vfe] returns the closest member of the
family instead.

## Action selection: information seeking and preferred outcomes

Action selection minimizes expected free energy (EFE), $G(\pi)$, over policies $\pi$.
Under the standard decomposition it splits into an epistemic term rewarding
information gain about hidden states and a pragmatic term rewarding observations
the agent prefers [@friston2015epistemic; @dacosta2020active; @parr2022active]:

$$
G(\pi) \;=\; -\underbrace{\mathbb{E}_{\tilde{q}}\!\left[\,D_{\mathrm{KL}}\!\left[\, q(s_\tau \mid o_\tau, \pi) \,\big\|\, q(s_\tau \mid \pi) \,\right]\right]}_{\text{epistemic value}}
\;-\; \underbrace{\mathbb{E}_{\tilde{q}}\!\left[\log \tilde{p}(o_\tau)\right]}_{\text{pragmatic value}} ,
$$ {#eq:efe}

where $\tilde{p}(o_\tau)$ is the preferred-observation distribution encoded by
the log-preference vector $\mathbf{C}$, and
$\tilde{q}(o_\tau, s_\tau \mid \pi) = p(o_\tau \mid s_\tau)\, q(s_\tau \mid \pi)$
is the policy-conditioned predictive over outcomes and states, the measure under
which both expectations in [@eq:efe] are taken. AILM's generation mechanism is
policy selection under [@eq:efe] with $\mathbf{C}$ supplied top-down by the level
above; the decoding paths do not exercise it, because during comprehension each
level has a single trivial action and $\mathbf{C}$ is zero. Generation is built
and measured in [@sec:generation], and the document-position (tercile) level,
`narrative` in code, reaches the clause level only through a blended `D` prior,
not through policy selection ([@sec:narrative_level]).

Policies are scored by $G(\pi)$, but they are selected against a prior over
policies as well, $\mathbf{E}$ — informally the agent's *habit*: what it would
do under no preference and no informative epistemic value at all. The two
combine as a softmax over expected free energy shifted by a log-habit term
[@friston2017process; @dacosta2020active], so that where $G(\pi)$ carries no
information — a flat $\mathbf{C}$ and an epistemic term that vanishes — the
policy posterior collapses onto the habit alone. AILM's one generative level
exercises exactly that combination, and [@sec:generation] gives the closed
form it reduces to there.

## Representing linguistic states with explicit probability arrays

AILM's levels are discrete partially observable Markov processes, in the sense
formalized by @friston2017process and @dacosta2020active. Observations
are conditionally independent given the current state, and states evolve as a
first-order Markov chain under the selected policy:

$$
p(o_{1:T}, s_{1:T}, \pi) \;=\; p(\pi)\, p(s_1) \prod_{t=1}^{T} p(o_t \mid s_t) \prod_{t=2}^{T} p(s_t \mid s_{t-1}, \pi) .
$$ {#eq:generative}

\begin{definition}[The AILM matrix tuple]
A level of AILM is the tuple \(\langle \mathbf{A}, \mathbf{B}, \mathbf{C}, \mathbf{D} \rangle\), where \(\mathbf{A}\) and every action slice \(\mathbf{B}^{(u)}\) are column-stochastic: their
columns, indexed by the conditioning variable, sum to one.
\end{definition}

The entries of the tuple are defined in [@eq:matrix_entries]:

$$
\mathbf{A}_{ij} = p(o = i \mid s = j), \quad
\mathbf{B}^{(u)}_{ij} = p(s_{t+1} = i \mid s_t = j,\, u), \quad
\mathbf{C}_i = \log \tilde{p}(o = i), \quad
\mathbf{D}_j = p(s_1 = j),
$$ {#eq:matrix_entries}

$$
\mathcal{M} \;=\; \langle \mathbf{A}, \mathbf{B}, \mathbf{C}, \mathbf{D} \rangle
$$ {#eq:pomdp}

`pymdp` checks normalization, dependency dimensions and indices, control-factor
consistency, and restrictions imposed by the selected inference algorithm.
Those structural checks still cannot infer a compatible matrix's semantic role:
the same square learned matrix can satisfy the relevant shape and normalization
contracts in either the $\mathbf{A}$ or the $\mathbf{B}$ slot, but placing it as
an emission rather than a transition expresses a different model
([@sec:architecture]).

The factored word and joint clause levels generalize [@eq:generative] without
leaving it. Level 3
factorizes the hidden state into word identity and within-word position, with a
single letter modality depending on both factors, so the definition applies
factor-wise and each factor carries its own transition slice
([@sec:word_level]). Level 5 runs the construction in the other direction,
embedding a two-component mixture of widened-symbol chains (case role, `VERB`,
`AUX`) as one factor over a joint type-by-symbol state space with a
block-diagonal transition
([@sec:clause_level]). Level 5 is the single-factor object in different
coordinates; Level 3 is not, and stays exact for the degeneracy reason given
above rather than by reducing to one factor. [@eq:filtering] and
[@eq:perplexity] below are stated for one factor and applied to the factored
levels through their joint state space.
Nothing in this section assumes more structure than [@eq:generative] declares.

## Sequential inference: expectations corrected by observations

Every built level consumes one symbol at a time. Given the posterior $q_{t-1}$
over the previous state, the predictive prior $\rho_t$ is pushed through
$\mathbf{B}$ and then corrected by the observation through $\mathbf{A}$
[@dacosta2020active], the same predict-then-correct recursion given as the
forward algorithm for a hidden Markov model by [@rabiner1989tutorial]:

$$
\rho_t(j) \;=\; \sum_k \mathbf{B}_{jk}\, q_{t-1}(k), \qquad
q_t(j) \;=\; \frac{\mathbf{A}_{o_t j}\, \rho_t(j)}{\sum_k \mathbf{A}_{o_t k}\, \rho_t(k)} .
$$ {#eq:filtering}

In `pymdp` the two halves of [@eq:filtering] are `update_empirical_prior` and
`infer_states` respectively. Filtering conditions only on observations up to
$t$; it never revisits them. That is a modeling commitment rather than a
convenience: the L3→L2B lexical channel consumes exactly such a
streaming quantity — the L3→L2B message is conditioned on the letter prefix
rather than the whole word ([@sec:hierarchy]). Filtering also costs accuracy
relative to whole-sequence smoothing; [@sec:phoneme_level] quantifies that cost.

## Predictive evaluation and the limits of first-order context

The predictive probability of the observation at step $t$ is the denominator of
[@eq:filtering], so the per-symbol perplexity of a sequence follows directly from
its cross-entropy [@shannon1948], in the language-model convention of
[@jelinek1997statistical; @brown1992entropy]:

$$
\mathrm{PPL} \;=\; \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T} \log \sum_j \mathbf{A}_{o_t j}\, \rho_t(j)\right) .
$$ {#eq:perplexity}

The relevant reference for a first-order model is not zero entropy. It is the
conditional entropy of the training distribution — the smallest expected log
loss any first-order conditional achieves *on that distribution*:

$$
H_{\text{train}}(X_t \mid X_{t-1}) \;=\; -\sum_{i,j} p_{\text{train}}(i, j) \log p_{\text{train}}(i \mid j), \qquad
\mathrm{PPL}_{\min} \;=\; \exp H_{\text{train}}(X_t \mid X_{t-1}) .
$$ {#eq:conditional_entropy}

The name matters as much as the subscript. This quantity is the *empirical*
first-order optimum on a specific finite sample, not a distribution-free lower
bound. Calling it a floor requires specifying the distribution on which it is
an optimum; an evaluation-sample reference and a training-sample reference are
different quantities.

The subscript on $H_{\text{train}}$ states exactly what
[@eq:conditional_entropy] licenses and what it does not. It
*is* the optimum of the order-1 model class on the empirical training
distribution: the empirical conditional $p_{\text{train}}(i \mid j)$ minimizes
expected log loss among all first-order conditionals when the expectation is
taken under $p_{\text{train}}$, so no order-1 model scores below it there. It is
*not* a lower bound on held-out cross-entropy. The held-out sample is a
different empirical distribution whose own first-order optimum can sit on either
side of this one: a held-out passage can be more or less predictable than the
corpus a model was fitted to. A held-out score below $\mathrm{PPL}_{\min}$
is therefore unexpected rather than impossible, and no claim in this paper rests
on ruling it out.

What the quantity supports instead is a statement about a declared threshold,
which is what [@sec:gate_design] needs from it. Measuring
[@eq:conditional_entropy] costs one pass over the bigram counts and answers a
question that no run can answer: how much first-order information the training
distribution contains at all. The Stage 1 target of {{STAGE1_DECLARED_GATE_NATS}} nats of gain over
uniform demands more than that entire first-order budget by a factor of
{{STAGE1_CEILING_SHORTFALL_RATIO}}, so the bar was calibrated for a model class
the level does not implement — a miscalibration finding about the threshold, not
a theorem about the evaluation population. [@sec:gate_design] turns that into
the verdict `UNREACHABLE`, which [@eq:gate_verdict] licenses only against a
comparator typed as a *bound* — for Stage 1 the audited ceiling is the evaluation passage's own order-1
optimum ([@eq:heldout_bound]), measured on the same held-out prose as the
model, while the training-distribution quantity above is the miscalibration
diagnostic; the two populations are named wherever the comparison is made.

## Linguistic case and cognitive roles: the CEREBRUM hypothesis

Level 4 of the stack assigns a case-role label — a heuristic syntactic role
class derived from UD annotations, not grammatical case ([@sec:case_level]) —
following the CEREBRUM formalism [@friedman2024cerebrum] and its categorical
companion [@friedman2024ccd]. CEREBRUM is this author's own framework, so it is a *hypothesis source* for
this paper and not independent validation of it. Where the prose says
CEREBRUM defines or predicts something, that is a statement about an engineered
ontology with an archival record. Where it says a case distinction is
established, the support is the linguistic literature the case system comes from
[@fillmore1968case; @blake2001case], which is independent of both this paper and
CEREBRUM. Level 4's gate is scored against Universal Dependencies annotation
([@sec:case_level]), also independent of both. CEREBRUM applies its case
vocabulary to noun phrases, where a case is the grammatical relation a phrase
bears to its predicate, and to models, where a case is the functional role a model plays
among interacting agents.

The archival identity itself has a version conflict that this paper preserves
rather than normalizes away. The exact Zenodo record calls the deposited work
v{{CEREBRUM_ZENODO_VERSION}} and dates it {{CEREBRUM_ZENODO_DATE}}; the title
page of that deposited Portable Document Format (PDF) file prints
v{{CEREBRUM_PRINTED_VERSION}} and
{{CEREBRUM_PRINTED_DATE}}. `data/reference_sources.json` records both identities,
the exact filename and digest, and every PDF locator used below.

AILM keeps linguistic labels and model-function labels as distinct enumerations
in `src/ailm/cases.py`; [@tbl:case_roles] lists them as separate inventories,
not rowwise equivalents. The source's core Genitive model role is retained as
`CORE_GENITIVE`. A different novel *Generative* role appears in [Supplement
{{CEREBRUM_GENERATIVE_SUPPLEMENT}}]({{CEREBRUM_GENERATIVE_SUPPLEMENT_URL}}). CEREBRUM labels both `[GEN]`; AILM therefore
uses `CORE_GENITIVE` and `NOVEL_GENERATIVE` with scoped serialized values while
preserving the shared source abbreviation as provenance.

| Enumeration | Member | Source label | Meaning represented in AILM |
|---|---|---|---|
| `CaseRole` | `NOM` | `NOM` | Nominative linguistic role |
| `CaseRole` | `ACC` | `ACC` | Accusative linguistic role |
| `CaseRole` | `GEN` | `GEN` | Genitive linguistic role |
| `CaseRole` | `DAT` | `DAT` | Dative linguistic role |
| `CaseRole` | `INS` | `INS` | Instrumental linguistic role |
| `CaseRole` | `LOC` | `LOC` | Locative linguistic role |
| `CaseRole` | `ABL` | `ABL` | Ablative linguistic role |
| `CaseRole` | `VOC` | `VOC` | Vocative linguistic role |
| `CerebrumFunctionRole` | `CORE_NOMINATIVE` | `NOM` | Active predictor |
| `CerebrumFunctionRole` | `CORE_ACCUSATIVE` | `ACC` | Update target |
| `CerebrumFunctionRole` | `CORE_GENITIVE` | `GEN` | Relational or parameter source |
| `CerebrumFunctionRole` | `CORE_DATIVE` | `DAT` | Data recipient |
| `CerebrumFunctionRole` | `CORE_INSTRUMENTAL` | `INS` | Policy implementor |
| `CerebrumFunctionRole` | `CORE_LOCATIVE` | `LOC` | Context provider |
| `CerebrumFunctionRole` | `CORE_ABLATIVE` | `ABL` | Causal origin |
| `CerebrumFunctionRole` | `CORE_VOCATIVE` | `VOC` | Addressable entity |
| `CerebrumFunctionRole` | `NOVEL_GENERATIVE` | `GEN` | Output generator |

: The independent `CaseRole` and `CerebrumFunctionRole` inventories. The former contains {{STAGE0_CASE_ROLES}} linguistic labels; the latter contains {{STAGE0_FUNCTION_ROLES}} scoped model roles because it retains both core Genitive and novel Generative despite their shared source abbreviation. Row order carries no equivalence claim. {#tbl:case_roles}

CEREBRUM [Supplement {{CEREBRUM_CASE_SUPPLEMENT}}]({{CEREBRUM_CASE_SUPPLEMENT_URL}}) does not give every case a
pair of scalar weights. It gives a likelihood weight only to Nominative and a
complexity weight only to Accusative. Genitive, Dative, Instrumental, Locative,
Ablative, and Vocative instead augment the latent state with, respectively,
relationship, goal, action, environment, origin, and communication variables,
then state case-specific precision relations. `CEREBRUM_CASE_FORMULATIONS`
preserves those distinctions. `case_conditioned_free_energy` is deliberately
narrower: it implements only the scalar Nominative/Accusative approximation and
fails closed for every augmented-latent case. Its numerical weights in
`AILM_SCALAR_CASE_WEIGHTS` are project-owned defaults satisfying the source
inequalities, not values reported by CEREBRUM.

Unlike [@eq:efe], which [@sec:generation] exercises, this scalar scaffold is
called by nothing outside its unit tests. Level 4 is built ([@sec:case_level])
but does not evaluate it — the level's gate rests on case agreement, not on a
free-energy quantity, and wiring the scaffold in without a falsifiable
measurement would be decoration.

$$
F_{\text{AILM-scalar}}[q;c] \;=\; \beta_c \, D_{\mathrm{KL}}\!\left[\, q(s) \,\big\|\, p(s) \,\right] \;-\; \alpha_c \, \mathbb{E}_{q(s)}\!\left[\log p(o \mid s)\right],
\qquad c \in \{\mathrm{NOM},\mathrm{ACC}\} .
$$ {#eq:case_free_energy}

Comparing [@eq:case_free_energy] with [@eq:vfe] locates the implemented
approximation exactly: Nominative reweights the likelihood term and Accusative
reweights the complexity term. The equation is not asserted for the other core
cases.

Cautions carry forward. CEREBRUM [Supplement {{CEREBRUM_POMDP_SUPPLEMENT}}]({{CEREBRUM_POMDP_SUPPLEMENT_URL}})
defines the tuple $\langle S, A, T, \Omega, O, F, \pi \rangle$: notably, its
$A$ is an action space, $O$ is the likelihood, $F$ is variational free energy,
and $\pi$ denotes precision parameters. [Figure
{{CEREBRUM_ACTIVE_INFERENCE_FIGURE}}]({{CEREBRUM_ACTIVE_INFERENCE_FIGURE_URL}}) is only a caption in the deposited PDF and
does not supply a crosswalk. `AILM_PYMDP_HEURISTIC_CROSSWALK` is therefore an
explicit project heuristic: source $T$ corresponds structurally to pymdp `B`
and source $O$ to pymdp `A`, while pymdp `C`, `D`, and `E` have no direct source
tuple analogue. In particular, source $\pi$ is not pymdp `E`, and source $F$ is
not pymdp `G`. The decision and alternatives are recorded in
`docs/design/cerebrum_mapping.md`. And the case
oracle built from Universal Dependencies deprel labels is a research claim, not
ground truth: English marks only Nominative, Accusative, and Genitive
morphologically, so `UD_DEPREL_TO_CASE` derives the remaining roles from
syntactic annotation. Agreement with that oracle is agreement, not accuracy, and
[@sec:gate_audit] treats the Stage 4 gate description accordingly — the
original threshold was vacuous on gold morphology and above every measured,
exceedable comparator on the derived oracle. Decision 16 records the historical
`UNREACHABLE` wording; decision 75 corrects it to `ABOVE_REFERENCE`.
