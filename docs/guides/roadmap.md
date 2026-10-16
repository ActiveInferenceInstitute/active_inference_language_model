# Roadmap

What AILM does today is stated in [`../../README.md`](../../README.md); what it
does not yet do, and might do next, is stated here. This page is forward-looking
by design and carries no result. Every status claim below is a pointer into
[`../../TODO.md`](../../TODO.md), [`../extension_protocols.md`](../extension_protocols.md),
or [`../decisions.md`](../decisions.md) — where this page and any of those three
disagree, they govern and this page is a defect to be fixed. Nothing here is a
`claim:` tag, and nothing here should ever become one without first landing a
recorder, a gate, and a control, the same way every shipped level did.

- [Near-term: the open protocols](#near-term-the-open-protocols)
- [GNN interoperability](#gnn-interoperability-from-parses-like-to-renders-like)
- [Relationships with LLMs and transformers](#relationships-with-llms-and-transformers)
- [How to contribute to each direction](#how-to-contribute-to-each-direction)

---

## Near-term: the open protocols

[`../extension_protocols.md`](../extension_protocols.md) answers the open
Major items in [`../../TODO.md`](../../TODO.md) with protocol designs P1–P3.
Read the current status at the top of that protocol document separately from
its preserved proposal sections. The fitted probe remains unmeasured; its data
foundation and compute pilot are distinct from a trained scored arm. Aligned
pronunciation and case-word measurements have been recorded. Their source-bound
receipts and ledger entries govern completion, not the age of a proposal table.
The connected runtime and GUM comparison are also implemented, as described in
[the hierarchy](../hierarchy.md); they do not retroactively turn isolated seam
experiments into end-to-end evidence.

| Protocol | Open TODO item | What is landed | What remains, and its gate |
|---|---|---|---|
| **P1** — a fitted external probe arm ([extension_protocols.md §2](../extension_protocols.md#2-protocol-p1--a-fitted-external-probe-arm)) | "Equal opportunity to fit the task" — today's external-probe comparison gives AILM's own decoders a fitted comparator while the served probes answer zero-shot, which the paper's own fairness table records as failing in every arm | The offline data/identity foundation: streaming exact-census objective, the diagnostic carve behind its declared seam, write-once receipts, the separate fitted-probe registry, the stride-aliasing refutation, and the warm-up compute-gate measurement | The fit pipeline itself — the LoRA fit and its declared controls, fuse/convert/serve, and the fitted recorder and its claims — gated by scoring the fitted arm under the same proper-loss discipline every other channel in this repository already uses, once the external model infrastructure (checkpoint downloads, local serving toolchain, and a license decision) that protocol assigns to its own grant is in place |
| **P2** — stipulated aligned spelling-to-pronunciation, design B ([extension_protocols.md §3](../extension_protocols.md#3-protocol-p2--stipulated-aligned-spelling-to-pronunciation-design-b)) | Extend the stipulated-artifact design (already shipped for the phone chain) to aligned pronunciation | The pair-structured stipulated artifact with digest-bound provenance, the native comparator (the shipped letter-to-phone decoder refit on the stipulated train pairs), and the offline evaluator with its English-transfer, broken-correspondence, and uniform-floor controls; the comparator arm and the zero-shot probe arms are implemented and registered | Recorded probe evidence must pass its current receipt and semantic replay checks. Any new demonstration arm requires a prospective protocol; the artifact is never called contamination-free |
| **P3** — the L4→L3 `P(word \| case role)` bridge ([extension_protocols.md §4](../extension_protocols.md#4-protocol-p3--the-l4l3-pword--case-role-bridge-account)) | The end-to-end top-down cascade was blocked on data provenance: no label/split/observation account existed for a case-role-to-word message | The account itself — the label-construction/split/observation artifact, the registry entries, the L3 inlet/executor, the fixed model-sender estimand, and an exploratory influence sweep on the development population | Any *performance* claim: it still needs its own specified population, comparator, controls, and uncertainty analysis before it is more than the exploratory sweep already on record. Cost is large and the value is uncertain by design — the uncertainty is the reason this remains open rather than closed |

None of the rows above authorizes rebuilding anything
[`../../TODO.md`](../../TODO.md) has already rejected. Read its **Rejected, with
reasons** section before proposing a variant of the morphemic level, a GNN
composition ontology ahead of its executor, a raised Stage 3 vocabulary, a
Mathlib-backed Lean layer, or a probe design that scores AILM against its own
fitted column — the last of these is load-bearing for the LLM-composition
proposals below, and its reasoning is repeated there rather than only summarized
here.

---

## GNN interoperability: from "parses like" to "renders like"

Every shipped level is declared in [Generalized Notation
Notation](../../gnn/README.md) (GNN), the Active Inference Institute's
structured notation for generative models, and every declaration is
conformance-checked against its live `pymdp` agent
([`../gnn_composition.md`](../gnn_composition.md)). What that checks today is
internal: AILM's own parser reads a spec, AILM's own conformance checker
compares the spec's `StateSpaceBlock`, `InitialParameterization`, and
`ActInfOntologyAnnotation` block against the array shapes and role bindings of
an agent AILM itself built by hand.

**Already landed, and honestly scoped.** `tests/test_gnn_conformance.py`'s
`TestUpstreamInteroperability` class parses the same shipped specs with the
*real* GeneralizedNotationNotation Markdown parser from an upstream checkout,
when one is present (`AILM_GNN_UPSTREAM`, behind the declared `gnn-interop`
extra required by the `full` release profile). It skips cleanly rather than
claiming success when no checkout is available, because AILM must not take a
multi-step external pipeline as a hard test dependency. Where it does run, it
checks that the two readers agree on every declared variable, with the one
known upstream identifier normalization recorded by name so it cannot become a
hiding place for a real disagreement. This is parser-level agreement — "the
same notation reads the same way" — not pipeline-level agreement.

**Proposed next step.** The upstream project describes its own pipeline as
rendering a GNN spec into runnable code for `PyMDP`, `RxInfer`, and
`ActiveInference.jl`. AILM's specs have never been run through that renderer.
The natural extension of the check already built is a third leg on the same
triangle the conformance checker already closes between spec, hand-written
builder, and live agent (the pipeline figure in
[`../manuscript/03_methods.md`](../manuscript/03_methods.md)): spec, AILM's
builder, and an *independently rendered* agent from the upstream pipeline, cross-checked
for structural (and, where the target framework supports it, numerical)
agreement. A disagreement there would be evidence about the notation, not about
AILM's own reading of it, which is a stronger check than the parser-level one
this repository currently runs. This has not been designed as a protocol yet —
it has no artifact schema, no declared comparator, and no decision on what
"agreement" must mean for a `RxInfer` or `ActiveInference.jl` target, whose
type systems differ from `pymdp`'s. Following
[`forking_guide.md`'s Step 0](forking_guide.md#step-0--audit-the-gate-before-building)
discipline, that decision belongs before any implementation, not after.

**A second, cheaper extension of the same idea:** the shipped mutation specs
under [`../../gnn/mutations/`](../../gnn/mutations/) are deliberate defects that
must fail AILM's own conformance checker. Running them through the upstream
parser as well would ask whether an independent implementation of the notation
also treats a mis-wired ontology annotation as a violation, which is a different
and stronger claim than "an independent parser agrees on well-formed specs."
Nobody has run this. Any future run needs its own falsifier — what counts as
"the upstream reader also flagged it," given that its error taxonomy is not
AILM's — stated before the run, not read off its output afterward.

**What stays out of scope on purpose.**
[`../../gnn/compositions/`](../../gnn/README.md) is a reserved, currently empty
directory. The connected executor now exists, but AILM’s GNN parser and
conformance checker still describe individual agents. The event schedule and
interventions are enforced by the [connected contract](../design/connected_runtime.md)
and its executable controls. A future composition notation needs its own parser
and validation semantics; adding declarations alone would not certify clocks,
resets or channel execution.

---

## Relationships with LLMs and transformers

**What is measured today.** External LLM probes — served locally or hosted —
appear in this repository as a non-comparative reference frame on the same
next-symbol tasks AILM's own decoders run, never as part of AILM's model stack
and never folded into a leaderboard-style ranking
([`../manuscript/98_supplement_external_benchmarks.md`](../manuscript/98_supplement_external_benchmarks.md)).
The comparison's own fairness table records that neither system has the same
opportunity to fit the task in any arm as currently run — which is exactly what
protocol P1 above exists to change, and only there.

**The standing structural differences**, restated because they bound every
proposal below: AILM has no transformer, no neural network, and no learned
embedding anywhere in it — every level is a named, factored `A`/`B`/`C`/`D`
array, declared in GNN, and conformance-checked against the agent that uses it,
so a matrix in the wrong slot is a caught, named violation rather than a
silently different number. A transformer's weights carry no such per-parameter
declared role; reading meaning back out of them is itself an open research
problem this project takes no position on beyond citing it
([`../manuscript/06_related_work.md`](../manuscript/06_related_work.md)).
AILM retains a closed word vocabulary. Its connected runtime runs live agents
on distinct character, token and sentence clocks and can select a bounded next
word through L1g policy inference. Its historical per-level scorers and generation
sweeps remain separate evidence. This is narrower than general-purpose fluent
text generation; see [the connected contract](../design/connected_runtime.md)
and [the generation design](../design/generation_problem.md). Neither of these is a ranking; they are different projects answering
different questions, which is the whole reason the external probes are framed
as a reference rather than a comparison.

Two compositions have been *named* as directions worth scoping. Neither is
designed, gated, or built. Both are constrained by the same rejected proposal,
covered last.

### Proposed direction: LLM calls AILM, as an auditable tool

[`../design/speech_boundary.md`](../design/speech_boundary.md) already states
the shape of this contract for one boundary: "a caller can pass parsed phone
symbols to the independently built Level 2 phone model," with the caller left
unspecified as external orchestration. The same shape generalizes to a caller
that is itself an LLM or an LLM-driven agent: rather than answering a language
question from its own weights, it invokes a shipped AILM decoder as a tool call
and receives back not just an answer but the complete per-position posterior and
the named matrix that produced it — the same conformance-checked, ledger-linked
trace this repository already produces for its own reported numbers. What
transformer-only tool use cannot offer by construction, a call into AILM could:
an answer whose every step is a declared probability over a named state space
rather than an activation pattern.

This is a description of a shape, not a protocol. Before it is more than that,
it needs a specified LLM-facing calling contract around the existing
`ConnectedRuntime` API or source-checkout CLI, an operational
definition of what "the caller used AILM's trace" means, and a receiver-side
measurement of whether that use changes the caller's own scored output — the
same "declared sender, receiver's own proper loss" method
[`../hierarchy.md`](../hierarchy.md#two-senses-of-channel-in-this-codebase)
already applies to every internal channel, not a new one invented for this
case.

### Proposed direction: AILM calls LLM, with a declared, measured likelihood

The historical cross-level experiments — L1→L2B coupling,
the L3→L2B lexical channel, the punctuation-type-oracle→L3 word-prior arm,
L6→L5, and the two case-word seams — is a *declared* sender whose distribution
is blended into a receiver's prior, likelihood, or preference at a named slot,
at a swept mixing weight, and then scored at the receiver under the receiver's
own proper loss against the unmixed anchor — never assumed to help, and
sometimes measured harmful
([`../hierarchy.md`](../hierarchy.md#two-senses-of-channel-in-this-codebase)).
The connected runtime separately fixes its named channel schedule and compares
it against channel interventions; those results are not a retuned sweep.
The proposed extension is the same design with an external LLM standing in as
the sender: for example, a served model's predicted next-word or predicted
case-role distribution blended into an AILM receiver's `D`-prior or
`C`-preference at a named slot, scored the same way the case-word bridge
(protocol P3, and decision 121) and the punctuation-oracle arm (decisions 27
and 88) already are.

What would make this a real arm rather than a restatement of a rejected one is
exactly the receiver's task definition, which is the subject of the next
section.

### The rejected circularity trap, and why it constrains both directions

`../../TODO.md`'s Rejected section already closed "supplying AILM's fitted
column to the probes," and the reasoning generalizes past its original target.
The trap is not "an external model touches AILM's output" in general — it is
scoring a receiver on a task whose answer is the very sender it is being
compared against, so the sender's own output becomes the receiver's ceiling by
construction, and the win looks *impressive* rather than degraded, which is
what makes it the more dangerous shape of the trap rather than the safer one.
A leak guard added after the fact cannot fix this: it *certifies* the
circularity instead of catching it, because the ceiling is definitional, not a
bug the guard could observe.

Both directions above must be designed to avoid the same shape:

- **LLM calls AILM** is safe from this trap only as long as the caller's own
  scored task is not "predict what AILM would say" — if it ever became that,
  AILM's own output would again be the ceiling of a task scored against it, the
  identical trap in the reverse orientation.
- **AILM calls LLM** is safe from this trap only as long as the AILM receiver's
  scored task is the receiver's own independent task (next-letter, next-phone,
  or whatever that level already scores) and not "predict what the sender
  said." Every existing internal channel already satisfies this — a receiver's
  gate has never been "match the sender" — and any LLM-sender extension must
  keep that invariant, not relax it for convenience.

Any future proposal in either direction should restate this constraint
explicitly and show, before a single call is scored, that its receiver's task
does not reduce to predicting its sender. That is the falsifier this roadmap
asks a future contributor to write down first.

---

## How to contribute to each direction

Read [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) first — this is a research
repository with a hard evidentiary standard, and the claims-ledger and
positive/negative-control disciplines it describes apply to every direction on
this page exactly as they apply to the six shipped levels.

- **The open protocols (P1–P3).** Read
  [`../extension_protocols.md`](../extension_protocols.md) in full, including
  its [ownership-grant boundaries](../extension_protocols.md#5-ownership-grant-boundaries-and-what-this-milestone-did-not-do),
  before picking up a partition; each protocol's remaining work is
  execution-sequenced, not independent tasks to grab in any order. Check
  [`../decisions.md`](../decisions.md) and `../../TODO.md`'s Rejected section
  before re-proposing a design that history already refuted.
- **GNN interoperability.** Start at [`../../gnn/AGENTS.md`](../../gnn/AGENTS.md)
  and [`../../gnn/SPEC.md`](../../gnn/SPEC.md) for the notation's contract, then
  `tests/test_gnn_conformance.py`'s `TestUpstreamInteroperability` class for how
  the existing parser-level check is wired. Running it locally needs the
  `gnn-interop` extra (`uv sync --locked --extra gnn-interop`) and
  `AILM_GNN_UPSTREAM` pointed at a real
  [upstream GNN checkout](../../gnn/README.md); without both it skips cleanly rather than failing or claiming
  success.
- **New levels in general**, including anything that would extend the
  hierarchy rather than compose across it, follow
  [`forking_guide.md`](forking_guide.md) — a checklist, not prose, built around
  auditing the gate before the level exists.
- **LLM↔AILM composition proposals.** These are named directions, not open
  tickets. Before drafting a design, read `../../TODO.md`'s Rejected section
  and [`../hierarchy.md`'s "Two senses of channel"](../hierarchy.md#two-senses-of-channel-in-this-codebase)
  in full, and write down the receiver's task and comparator — and confirm the
  task does not reduce to predicting the sender — before proposing any code.
  A design that skips this step will be asked to restate it, not built around
  it.

Status governs from `../../TODO.md`, `../extension_protocols.md`, and
`../decisions.md`; this page points at them and does not duplicate or override
what they say.
