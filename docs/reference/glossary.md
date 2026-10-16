# Glossary

Every project-specific term a newcomer meets, alphabetically, each pointing to
the file or source that defines it. Definitions are derived from the code and
from the canonical docs — [`hierarchy.md`](../hierarchy.md),
[`gate_design.md`](../gate_design.md), [`syntax_guide.md`](../syntax_guide.md),
[`method/release_checklist.md`](../method/release_checklist.md), and
[`notation.md`](notation.md) — never invented. It carries no result numbers; for
those, follow each entry's link.

If a term isn't here, check [`notation.md`](notation.md) for symbols and
equation-table entries, or [`../decisions.md`](../decisions.md) for the
reasoning behind a specific choice.

---

### `ABOVE_REFERENCE`

A [gate verdict](#gate-verdict): the trivial baseline fails the threshold, and
the measured ceiling is an *exceedable* comparator ([`REFERENCE_MODEL`](#reference_model)
or [`PRIVILEGED_ORACLE`](#privileged_oracle)) that also fails it. The bar sits
above that comparator, so no impossibility claim follows — a different model in
the admissible class could still clear it. Defined in
[`../../src/ailm/gates.py`](../../src/ailm/gates.py) (`classify_gate`); see the
taxonomy table in [`gate_design.md`](../gate_design.md#the-typed-gate-verdict-taxonomy).

### `ANALYTIC_BOUND`

A [comparator type](#comparator--comparatortype): a closed-form bound that
needs no measurement, such as a total-variation distance of zero that no
generator can score below. Licenses an [`UNREACHABLE`](#unreachable) or
[`BRACKETED_BOUND`](#bracketed_bound) verdict. Defined in
[`../../src/ailm/gates.py`](../../src/ailm/gates.py) (`ComparatorType`).

### Baseline

The trivial comparator a gate is scored against: the strongest model that
*lacks* the specific mechanism the level claims to add (a context-free unigram
for a sequence level, a context-free per-observation lookup table for a level
that infers a hidden alphabet). One of the two measurements — with the
[ceiling](#ceiling--comparator) — that a [gate](#gate) needs before its
[verdict](#gate-verdict) means anything. See
[`gate_design.md` § Uniform is a crippled comparator](../gate_design.md#uniform-is-a-crippled-comparator).

### `BRACKETED_BOUND`

A [gate verdict](#gate-verdict): the trivial baseline fails the threshold and a
bound-typed ceiling ([`MODEL_CLASS_BOUND`](#model_class_bound) or
[`ANALYTIC_BOUND`](#analytic_bound)) clears it. The gate discriminates, and the
passing region is capped by a genuine bound. Defined in
[`../../src/ailm/gates.py`](../../src/ailm/gates.py) (`classify_gate`).

### `BRACKETED_REFERENCE`

A [gate verdict](#gate-verdict): the trivial baseline fails the threshold and
an exceedable ceiling ([`REFERENCE_MODEL`](#reference_model) or
[`PRIVILEGED_ORACLE`](#privileged_oracle)) clears it. The gate discriminates,
but the comparator itself could in principle be exceeded by a different model
in the class. Defined in [`../../src/ailm/gates.py`](../../src/ailm/gates.py)
(`classify_gate`).

### Ceiling / comparator

The achievable-performance measurement a [gate](#gate) is scored against: a
conditional-entropy floor for a sequence metric, or a strictly more powerful
decode for a classification metric. Every ceiling carries a declared
[`ComparatorType`](#comparator--comparatortype) — the classification is fail-closed, so a
measured ceiling with an undeclared type is refused rather than silently
treated as a bound. See [`gate_design.md` § How to compute an information
ceiling](../gate_design.md#how-to-compute-an-information-ceiling) and
[`../../src/ailm/gates.py`](../../src/ailm/gates.py).

### Channel

A source-to-receiver message between two levels (or between a privileged
oracle and a level) that was actually built and measured, as opposed to a
proposed architecture. AILM's shipped channels are named individually — the
L1→L2B coupling (harmful), the L3→L2B lexical channel (helps), and the
case-word bridge into L3's word prior — and are explicit typed experiments
with their own populations. The connected runtime separately executes named
channels on a causal multirate schedule; an intervention there is not the same
estimand as an isolated historical sweep. See
[`hierarchy.md` § Two senses of "channel" in this
codebase](../hierarchy.md#two-senses-of-channel-in-this-codebase).

### Claim

One recorded numeric fact in the [claims ledger](#claims-ledger), keyed by a
label and carrying the run that produced it plus an optional epistemic
[status](#confirmatory-vs-exploratory-vs-descriptive-vs-analytic). Referenced
in prose with the [`claim:` tag](#claim-tag) or, in the manuscript, with a
[`{{TOKEN}}`](#token-hydration). Defined in
[`../../src/ailm/claims.py`](../../src/ailm/claims.py).

### `claim:` tag

The in-prose reference form `claim:<label>`, used in `docs/**/*.md`,
`README.md`, `AGENTS.md`, `CHANGELOG.md`, and Python docstrings. The tag itself
is not substituted; it is a verifiable pointer that a test checks against the
[claims ledger](#claims-ledger), so the source stays auditable while a
hydrated, human-readable mirror is written separately. Contrast with
[`{{TOKEN}}`](#token-hydration), the manuscript's substituted form. See
[`syntax_guide.md` § The two mechanisms](../syntax_guide.md#the-two-mechanisms).

### Claims ledger

The single source of numeric truth for the project, `data/claims.json`: every
quantitative claim in any README, docstring, or manuscript section must have a
backing entry here, recorded from the run that produced it, so a hand-typed
number cannot drift away from a measured one. Implemented by
[`../../src/ailm/claims.py`](../../src/ailm/claims.py) (`ClaimsLedger`,
`load_ledger`); see [`../README.md`](../README.md) for the schema and status
field.

### Comparator / `ComparatorType`

See [Ceiling / comparator](#ceiling--comparator). The type enum itself —
[`MODEL_CLASS_BOUND`](#model_class_bound), [`ANALYTIC_BOUND`](#analytic_bound),
[`PRIVILEGED_ORACLE`](#privileged_oracle), [`REFERENCE_MODEL`](#reference_model) —
is defined in [`../../src/ailm/gates.py`](../../src/ailm/gates.py).

### Confirmatory vs. exploratory vs. descriptive vs. analytic

The four values of a claim's epistemic `status` field in the
[claims ledger](#claims-ledger) (`CLAIM_STATUSES` in
[`../../src/ailm/claims.py`](../../src/ailm/claims.py)):

- **confirmatory** — pre-registered before its outcome existed and scored
  under its registered rule; the commit order is machine-checked against Git
  history, not merely asserted in prose.
- **exploratory** — a test-selected sweep or subsample arm whose headline was
  chosen on the evidence it is reported on.
- **descriptive** — a measured quantity carrying no inferential claim (a
  count, a size, a version, an un-registered measurement); the default when
  ambiguous.
- **analytic** — a closed-form or derived quantity (an entropy floor, a
  uniform baseline, complexity arithmetic) involving no sampling.

### Conformance

The property [`ailm.gnn.assert_conforms`](../../src/ailm/gnn/conformance.py)
(and its non-raising sibling `check_conformance`) checks: that a level's live
`pymdp` `Agent` — its array shapes and the semantic role each array actually
plays — matches its [GNN](#gnn) spec's declaration exactly. A [mutation
spec](#mutation-spec) must fail this check; every canonical spec must pass it.
This is the third of AILM's three independent guards against a matrix landing
in the wrong slot — after the closed-form oracle cross-check and a measured
ceiling comparison, neither of which alone is sufficient. See
[`../../gnn/README.md`](../../gnn/README.md) and
[`tests/test_gnn_conformance.py`](../../tests/test_gnn_conformance.py).

### Connected runtime

`ConnectedRuntime` composes live pymdp agents through explicit character, token
and sentence events. Its trace separates prediction before an observation from
the posterior after it. Exact conditional-update checks do not establish exact
inference in a global joint model. See [the contract](../design/connected_runtime.md)
and [the runnable guide](../guides/quickstart.md#1b-run-the-connected-model).

### DSP

Digital signal processing — the proposed external integration around the
Stage 7 boundary (Whisper speech-to-text on the way in, a TTS engine on the
way out). Those acoustic adapters are not implemented by the symbol contract. AILM
draws an explicit interface contract at this boundary rather than modeling
speech itself: DSP sits outside the level stack. See
[`../design/speech_boundary.md`](../design/speech_boundary.md) and
[`../architecture.md`](../architecture.md).

### Gate

A falsifiable numeric claim about one level of the hierarchy, stated before
that level is built, with three required parts: a metric (what is measured,
over which evaluation set, in which direction), a [comparator](#ceiling--comparator)
(what a model lacking the claimed mechanism would score), and a threshold (the
value separating pass from fail). Missing any part makes the claim
unfalsifiable. See [`gate_design.md` § What a gate
is](../gate_design.md#what-a-gate-is).

### Gate verdict

The typed classification of where a gate's threshold sits relative to its
[baseline](#baseline) and [ceiling](#ceiling--comparator) — one of `VACUOUS`,
`UNREACHABLE`, `ABOVE_REFERENCE`, `UNAUDITED`, `BRACKETED_BOUND`, or
`BRACKETED_REFERENCE` (each defined at its own alphabetical entry in this
glossary). The single classifier is `classify_gate` in
[`../../src/ailm/gates.py`](../../src/ailm/gates.py); gate *placement* and
model *outcome* (whether the shipped model passes) are reported separately and
must not be conflated. See
[`gate_design.md` § The typed gate-verdict taxonomy](../gate_design.md#the-typed-gate-verdict-taxonomy).

### `GateVacuousError`

Raised when a gate's negative control *passes* — a deliberately broken model
was not caught by the assertion meant to catch it. It exists so that a test
which cannot fail is treated as no test at all, rather than being weakened
quietly until it stops raising. Defined in
[`../../src/ailm/control.py`](../../src/ailm/control.py); see
[`testing_philosophy.md` § GateVacuousError](../testing_philosophy.md#gatevacuouserror).

### GNN

Generalized Notation Notation — the Active Inference Institute's structured,
section-ordered Markdown notation for declaring a generative model: its state
space, its connections, its parameterization, its time structure, and, the
part AILM depends on most, an [ontology annotation](#ontology-annotation)
naming the semantic role of every array. Every shipped AILM level is declared
as a `.gnn.md` spec under [`../../gnn/models/`](../../gnn/models/), each bound
to its live `pymdp` `Agent` by [conformance](#conformance) checking. AILM also
constructs live agents locally through `ailm.spec_agent` and explicitly
allowlisted parameter resolvers. Upstream ships a pipeline
that renders a spec into runnable `PyMDP`, `RxInfer`, and `ActiveInference.jl`
code; AILM’s local spec construction and conformance do not use that upstream
renderer, whose extension to AILM's own specs is tracked in
[`guides/roadmap.md`](../guides/roadmap.md#gnn-interoperability-from-parses-like-to-renders-like)
as proposed, not shipped. See [`../../gnn/README.md`](../../gnn/README.md) and
[`../gnn_composition.md`](../gnn_composition.md).

### Held-back partition

A population that has not yet been scored by anything in the repository, kept
in reserve so a [confirmatory](#confirmatory-vs-exploratory-vs-descriptive-vs-analytic)
result can be replicated on genuinely new data rather than re-reading the
population it was designed against. Not every corpus has one: for some members every
partition was already used during development, which is recorded
explicitly rather than assumed. See
[`../decisions.md` § 79](../decisions.md#79--pre-registration-the-confirmatory-replication-protocol-with-an-honest-census-of-what-can-still-be-held-back).

### Intervention

A measured change to one level's generative-model arrays caused by another
level or by a privileged reference — for example a narrative-level phase
distribution blended into a clause-level prior, or a privileged oracle's typed
word distribution blended into a word-level prior. Historical interventions
were measured individually; the connected runtime executes a separately
specified schedule. Neither existence nor type correctness of a route establishes
that it improves predictive loss. See
[`hierarchy.md` § Top-down intervention inventory](../hierarchy.md#two-senses-of-channel-in-this-codebase).

### L1 — Letters

The first level: hidden state and observation are both the current letter,
transitioning through a Brown-corpus letter bigram. See the level table in
[`hierarchy.md`](../hierarchy.md#what-is-shipped-level-by-level).

### L2A — Phonemes

The phoneme-sequence level: hidden state and observation are both the current
phone, transitioning through a CMUdict phone bigram. See
[`hierarchy.md`](../hierarchy.md#what-is-shipped-level-by-level).

### L2B — G2P

The letter-to-phone (grapheme-to-phoneme) level: the hidden factor is drawn
from the phone inventory while the observation is the current letter, making
its emission matrix an instance of [a shared alphabet](#shared-alphabet) with
L1. Also called Stage 2B in the stage-gate table. See
[`hierarchy.md` § L2B ← L1](../hierarchy.md#l2b--l1-cross-representation-inference-inside-one-agent).

### L3 — Words

The word level: the hidden factor is word identity crossed with within-word
position, observing each letter or an end-of-word symbol through a
deterministic spelling emission. See
[`hierarchy.md`](../hierarchy.md#what-is-shipped-level-by-level).

### L4 — Cases

The syntactic case level: the hidden factor is a case role, observed only
through Universal Dependencies part-of-speech (`UPOS`) tags — never the
dependency relation or the morphological case feature, both of which are
withheld from the model and used only by the evaluation oracle. See
[`hierarchy.md`](../hierarchy.md#what-is-shipped-level-by-level) and
[`notation.md` § Special observations](notation.md#special-observations).

### L5 — Clauses

The clause level: the hidden factor is a joint sentence type crossed with a
widened symbol (a case role, or a verb/auxiliary marker), observed through
that widened symbol. See
[`hierarchy.md`](../hierarchy.md#what-is-shipped-level-by-level).

### L6 — Narrative

The narrative level: the hidden factor is a document-position tercile
("phase" in code), observed through sentence type, and intervening on the
clause level's prior. See
[`hierarchy.md`](../hierarchy.md#what-is-shipped-level-by-level).

### Margin

The signed distance between a metric and its threshold, positive exactly when
the gate passes (`gate_margin` in
[`../../src/ailm/gates.py`](../../src/ailm/gates.py)). Compared against a
gate's declared [SESOI](#sesoi-smallest-effect-size-of-interest) so a hairline
positive margin cannot silently read as a clean finding.

### `MODEL_CLASS_BOUND`

A [comparator type](#comparator--comparatortype): a theorem-backed bound on
every model of the class, measured on the same population the gate scores
(for example, a held-out passage's own conditional entropy lower-bounding any
order-1 model's log loss there). Licenses an [`UNREACHABLE`](#unreachable) or
[`BRACKETED_BOUND`](#bracketed_bound) verdict. Defined in
[`../../src/ailm/gates.py`](../../src/ailm/gates.py) (`ComparatorType`).

### Mutation spec

A [GNN](#gnn) file under [`../../gnn/mutations/`](../../gnn/mutations/), named
`<base>@<defect>.gnn.md`, that injects exactly one deliberate structural defect
into a canonical spec — a swapped [ontology annotation](#ontology-annotation)
role, a frozen transition, an uninformed prior, and similar — and that must be
rejected by [conformance](#conformance) checking. It is the GNN layer's own
[negative control](#negative-control): a mutation spec the checker fails to
catch would mean the checker is not actually reading the declared roles back
off the live agent. See [`../../gnn/README.md`](../../gnn/README.md) and
[`../../tests/test_gnn_conformance.py`](../../tests/test_gnn_conformance.py).

### Negative control

A test run against a deliberately broken model or input, required to show
that a gate's assertion can actually *fail* — without one, a metric that
cannot tell good from bad is unfalsifiable in the direction that matters most.
See [`testing_philosophy.md` § Every gate needs a positive AND a negative
control](../testing_philosophy.md#every-gate-needs-a-positive-and-a-negative-control)
and [`GateVacuousError`](#gatevacuouserror).

### Ontology annotation

The `ActInfOntologyAnnotation` block of a [GNN](#gnn) spec: a `Variable=Term`
mapping (`A=RecognitionMatrix`, `B=TransitionMatrix`, `C=LogPreferenceVector`,
`D=Prior`) in bijection with the spec's declared variables — nothing declared
may go unannotated, and nothing annotated may be undeclared. It is the block
[conformance](#conformance) checking exists to enforce: without a declared
role to pin a matrix against, a learned table dropped into the wrong slot
passes every normalization check and looks like a slightly different number
rather than a named violation. See
[`../../gnn/SPEC.md`](../../gnn/SPEC.md) and
[`../../gnn/models/L1_letter.gnn.md`](../../gnn/models/L1_letter.gnn.md).

### Out-of-vocabulary (OOV)

A normalized token absent from the fixed word lexicon. The connected runtime
retains it in its trace and full-stream denominator. Its word probability is
zero, so full word log loss is infinite; a finite in-vocabulary diagnostic
answers a narrower question. The JSON CLI reports this explicitly with
`word_loss_status` rather than emitting nonstandard JSON infinity.

### Positive control

A test proving that a detector or gate *can* fire at all — for example, a
planted fake `claim:` tag that a scanner must flag. Paired with a
[negative control](#negative-control) so both directions of the check are
exercised. See [`testing_philosophy.md`](../testing_philosophy.md).

### `PRIVILEGED_ORACLE`

A [comparator type](#comparator--comparatortype): a measured comparator given
information the model itself forgoes, such as a fitted-on-test distribution or
a true-label oracle. Strong, but not a bound — the Bayes-optimal use of less
information can in principle beat a naive use of more. Defined in
[`../../src/ailm/gates.py`](../../src/ailm/gates.py) (`ComparatorType`).

### Receipt (network claim binding)

The artifact under `cache/network_claim_bindings/` that hashes a recorder's
own code, its local script-import closure, the whole `src/ailm` tree,
`pyproject.toml`, `uv.lock`, and the vendored JSON inventory, binding a
recorder's claims to the exact code that produced them. Any change to a
shared manifest input stales every receipt; a recorder-specific change stales
only its own. See
[`method/release_checklist.md` § 2. Re-bind the network
recorders](../method/release_checklist.md#the-sequence).

### Recorder

A `scripts/record_stage*_claims.py`-style script that measures a quantity and
writes it into the [claims ledger](#claims-ledger) under its own namespace,
carrying the run identity and, since schema v2, an epistemic status. A
recorder's claims are bound to the code that produced them via its
[receipt](#receipt-network-claim-binding); re-running one after an edit stales
its binding until it is re-bound. See
[`method/release_checklist.md`](../method/release_checklist.md) and
[`../../src/ailm/claims.py`](../../src/ailm/claims.py).

### `REFERENCE_MODEL`

A [comparator type](#comparator--comparatortype): a measured system with no
privileged access, such as a smoothing decode of the same model or a stronger
lookup table. Explicitly exceedable by the shipped model. Defined in
[`../../src/ailm/gates.py`](../../src/ailm/gates.py) (`ComparatorType`).

### Run scope

The namespace the external, synthetic, and in-context recorders give their
fresh-call cache, keyed by a UTC run id fixed at recorder start, so successive re-runs append new records rather than
overwriting an existing key. Run scopes are part of the digested
[receipt](#receipt-network-claim-binding) configuration and are checked
exactly by the binding validator. The log-probability drift recorder uses a
different contract — sampled stored calls paired with fresh uncached replies —
and has no run scope. See
[`method/release_checklist.md` § 2. Re-bind the network
recorders](../method/release_checklist.md#the-sequence).

### SESOI (smallest effect size of interest)

A gate's declared floor on its own reporting scale — a margin must strictly
exceed it before it is read as more than fitting noise, rather than a
hairline positive number certifying a gate silently. Declared per gate id in
`SMALLEST_EFFECT_SIZES`
([`../../src/ailm/gates.py`](../../src/ailm/gates.py)); an undeclared gate id
fails closed in `smallest_effect_size`. Compare [Margin](#margin).

### Shared alphabet

One representational sense of "hierarchical" in AILM: a level's observation alphabet
*is* the state alphabet of the level below it, so one level's inferred hidden
variable is exactly what the level beneath it emits. This is a type identity
fixed when the matrices are built, not a value computed at run time: no belief,
distribution, or message travels along it. L2B is the shipped example — its
hidden factor is drawn from the phone inventory and its observation modality is
drawn from the letter alphabet, so its emission matrix links two alphabets of
different sizes inside one agent. Defined in
[`gnn_composition.md` § 4.1](../gnn_composition.md#41-representational-composition--a-shared-alphabet);
see also [`hierarchy.md` § L2B ← L1](../hierarchy.md#l2b--l1-cross-representation-inference-inside-one-agent).

### Stage N

The build-order label for one step of AILM's staged construction (Stage 0
through Stage 7), matching the level structure with two exceptions: Stage 0 is
an infrastructure smoke test rather than a linguistic level, and Stage 7 is
the [DSP](#dsp)/pymdp speech-boundary interface contract rather than a scored
model. Stage 2 covers both L2A (phonemes) and L2B (G2P, sometimes called
"Stage 2B"). See the stage-gate table in
[`../../AGENTS.md`](../../AGENTS.md#stage-gates).

### `{{TOKEN}}` hydration

The manuscript's number-substitution mechanism: `docs/manuscript/*.md` prose
contains `{{TOKEN}}` placeholders (the [claims ledger](#claims-ledger) label,
uppercased) that are resolved from `data/claims.json` at render time, so the
rendered number is read out of the ledger on every build rather than typed by
hand. A token naming no ledger entry fails the render. Contrast with the
[`claim:` tag](#claim-tag), used everywhere outside the manuscript. See
[`syntax_guide.md` § Rule 1: token = ledger label,
uppercased](../syntax_guide.md#rule-1-token--ledger-label-uppercased).

### `UNAUDITED`

A [gate verdict](#gate-verdict): the trivial baseline fails the threshold, but
no ceiling has been measured at all, so reachability is unknown. Missing
evidence is not treated as a favorable verdict. Defined in
[`../../src/ailm/gates.py`](../../src/ailm/gates.py) (`classify_gate`).

### `UNREACHABLE`

A [gate verdict](#gate-verdict): a bound-typed ceiling
([`MODEL_CLASS_BOUND`](#model_class_bound) or
[`ANALYTIC_BOUND`](#analytic_bound)) does not clear the threshold, so no
correct implementation can pass on the population the bound was measured on.
Only a bound-type comparator can license this verdict. Defined in
[`../../src/ailm/gates.py`](../../src/ailm/gates.py) (`classify_gate`).

### `VACUOUS`

A [gate verdict](#gate-verdict): the trivial [baseline](#baseline) already
satisfies the threshold, so a model doing nothing hierarchical passes and the
gate constrains nothing. Defined in
[`../../src/ailm/gates.py`](../../src/ailm/gates.py) (`classify_gate`).

### Warm store

A pre-populated cache of external model calls (served via `ollama serve` or a
hosted transport) that lets a recorder replay stored calls instead of making
fresh ones wherever the recording protocol permits. The release checklist
uses `scripts/warm_call_cache.py --check` to inspect required local cache
coverage and serving prerequisites. A warm store does not replace fresh calls
required by a canonical publication recorder or authenticate its current claim
receipt. Historical records from another endpoint, loader or context setting
cannot fill the selected configuration’s audit strata. See
[`method/release_checklist.md` § Preconditions](../method/release_checklist.md#preconditions).
