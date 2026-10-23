# docs/ — AILM Documentation Index

Everything written down about the Active Inference Language Model, by category.
Start with [`quickstart.md`](guides/quickstart.md) if you want a running checkout; start
with [`../README.md`](../README.md) if you want to know what the project claims.

Each section heading below names who it's for — newcomer, contributor, auditor,
or researcher — so you can find the file for your question without reading the
whole index.

## Current entry points

| Need | Start with | Scope |
| --- | --- | --- |
| Run connected text inference or bounded next-word generation | [Connected runtime API and CLI](design/connected_runtime.md#active-generation-and-invocation) | Explicit causal messages through GNN-built agents; conditional-update checks, not a global joint posterior |
| Interpret the connected evaluation | [GUM result and limits](manuscript/04_results.md#sec:connected_gum), [input provenance](provenance/sources.md#ud-english-gum-connected-evaluation-input) | Locally frozen raw-text comparison; normalized character loss, coverage and infinite full-stream word loss reported separately |
| Inspect individual model components in a browser | [Browser interface inventory](../src/ailm/dashboard/README.md#one-dashboard-url-with-an-optional-server-free-export) | Component inspection and the separate L3→L1g demonstration; these browser surfaces do not run `ConnectedRuntime` |
| Reproduce historical gates and interventions | [Governing hierarchy](hierarchy.md), [decision chronology](decisions.md) | Original populations, scorers and retained ablations remain attached to their own results |
| Assess speech or formal guarantees | [Speech boundary](design/speech_boundary.md), [Lean scope](formal.md) | Boundary serialization and selected mathematical proofs; neither certifies an end-to-end speech system or the connected model's global correctness |

## Start here

Three short paths through this index, by audience. Each step is a file this
index already lists in full below.

**Newcomer** (want a running checkout):
1. [`../README.md`](../README.md) — what the project claims and does not claim.
2. [`guides/quickstart.md`](guides/quickstart.md) — clone, install, run the gates.
3. [`reference/glossary.md`](reference/glossary.md) — look up any unfamiliar term as you go.
4. [`guides/roadmap.md`](guides/roadmap.md) — where the project is headed: the open
   protocols, GNN interoperability, and its stance on composing with LLMs.

**Reviewer / auditor** (checking whether a claim holds up):
1. [`hierarchy.md`](hierarchy.md) — the governing level-by-level description.
2. [`gate_design.md`](gate_design.md) — how a gate's placement is classified, and what an
   impossibility claim requires.
3. [`../data/claims.json`](../data/claims.json) with
   [`reference/formalism_map.md`](reference/formalism_map.md) — trace a number back to
   the code and test that produced it.

**Contributor** (about to change code or prose):
1. [`guides/agent_instructions.md`](guides/agent_instructions.md) — hard behavioral rules
   and the command reference.
2. [`architecture.md`](architecture.md) — package map and import contract.
3. [`syntax_guide.md`](syntax_guide.md) — the `{{TOKEN}}` and `claim:<label>` protocols
   for recording a new number.

Two documents share the name `architecture.md` and they are deliberately not the
same document:

| | File | Answers |
|---|---|---|
| **Code** | [`architecture.md`](architecture.md) | Where does a given kind of logic live, what may import what, which module owns which invariant |
| **Science** | [`hierarchy.md`](hierarchy.md), then [`design/architecture.md`](design/architecture.md) | The governing shipped hierarchy and scorer paths first; then the design rationale, hidden factors, A/B/C/D, and factorization |

If you are about to move a function, read the first. If you are about to change
what a matrix means, read the second.

## Guides (newcomer/contributor) — `guides/`

Task-oriented, in the order you'd actually read them.

| File | What it is for |
|---|---|
| [`quickstart.md`](guides/quickstart.md) | Clone → interpreter → gates → figures → render, with the exact commands and the exact interpreter |
| [`agent_instructions.md`](guides/agent_instructions.md) | Hard behavioral rules to read before modifying anything; the canonical command reference |
| [`forking_guide.md`](guides/forking_guide.md) | Reusing this layout for another staged-build project |
| [`roadmap.md`](guides/roadmap.md) | Forward-looking only, no results: the open protocols P1–P3 and their gates, extending GNN interoperability from parser-level agreement to a rendered upstream cross-check, and the two named (undesigned) LLM↔AILM composition directions bounded by the rejected circularity trap |
| [`faq.md`](guides/faq.md) | Recurring questions |
| [`troubleshooting.md`](troubleshooting.md) | Symptom → cause → fix |
| [`AGENTS.md`](AGENTS.md) | How agents work *inside* `docs/`: ownership, which gates read these files, verification commands |

## Governing references (contributor/auditor) — stay at `docs/` root

Root-level because each is the file its own topic's other members point back to;
splitting the "wins on disagreement" file from its supporting reference would make
the tree harder to navigate, not easier.

| File | What it is for |
|---|---|
| [`architecture.md`](architecture.md) | The code architecture: package map, module ownership, dependency direction, forbidden patterns |
| [`hierarchy.md`](hierarchy.md) | The governing description of the level structure — where any other description disagrees, this file wins |
| [`publication_derivative.md`](publication_derivative.md) | Exact two-source publication attestation, original measurement evidence reuse, isolated semantic replay and separate live local cache audit |
| [`gate_design.md`](gate_design.md) | The comparator-typed taxonomy: VACUOUS / UNREACHABLE / ABOVE_REFERENCE / UNAUDITED / BRACKETED_BOUND / BRACKETED_REFERENCE. Only bound comparators license impossibility claims |
| [`testing_philosophy.md`](testing_philosophy.md) | What a green suite is allowed to mean. Positive and negative controls, `GateVacuousError`, the ways this project's own tests have failed to constrain anything |
| [`syntax_guide.md`](syntax_guide.md) | The two numeric mechanisms — `{{TOKEN}}` in the manuscript, `claim:<label>` in prose — and the cross-reference conventions |
| [`decisions.md`](decisions.md) | The single record of the decisions behind AILM's numbers. Append-only chronology |
| [`development_history.md`](development_history.md) | Historical changelog preserved from `CHANGELOG.md`; dated descriptions are not current API or release-readiness assertions |
| [`research_queue.md`](research_queue.md) | Historical, exploratory research proposals for Levels 3-6 and the hierarchy, each with its build, comparator, falsifier and evidence. Proposer-stage: read its provenance note before treating any figure in it as a result |

## Reference (contributor/auditor) — `reference/`

Consult-as-needed material: symbol tables, equation maps, artifact inventories,
the Lean layer's documentation-side account, and what each corpus generalizes to.

| File | What it is for |
|---|---|
| [`glossary.md`](reference/glossary.md) | Alphabetical definitions of every project-specific term (level names, gate verdicts, comparator types, the claims-ledger vocabulary), each linked to the file or source that defines it |
| [`notation.md`](reference/notation.md) | Consolidated symbol-to-meaning reference for the manuscript, decision records, and code |
| [`formalism_map.md`](reference/formalism_map.md) | Every labeled equation, definition, and proposition mapped to the code that computes it and the test that exercises it |
| [`formal.md`](formal.md) | The Lean 4 layer under `formal/`: what is machine-checked, the Lean ↔ equation ↔ Python ↔ test isomorphism, the generated ledger bridge, and what is deliberately not formalized |
| [`output_conventions.md`](reference/output_conventions.md) | `output/` is entirely regenerable; stable publication surfaces are tracked while transient render products are ignored; the regeneration order; why hand-editing it to pass a gate is the worst available move |
| [`output_inventory.md`](reference/output_inventory.md) | Concrete artifact list: every generated figure with its label and generator, the token map, the injected manuscript, PDF/slides/web, the render diagnostics |
| [`configuration.md`](reference/configuration.md) | Every knob the shipped code reads: the `LevelConfig` hyperparameters and each `AILM_*` environment variable with its reader, meaning and unset behavior, grouped by library runtime, recorders, dashboard, release verifier and formal layer |
| [`data_statement.md`](reference/data_statement.md) | What each corpus **generalizes to**, in the sense of Bender and Friedman: curation rationale, variety, era, producers, what was done to it, and who is absent. [`provenance/sources.md`](provenance/sources.md) answers whether a number can be rebuilt; this answers what population it can be about |
| [`gnn_composition.md`](gnn_composition.md) | GNN composition declarations and conformance; `hierarchy.md` governs claims about which score paths actually execute live inference |

## Method (contributor) — `method/`

The mechanics of writing prose, generating numbers, and cutting a release.

| File | What it is for |
|---|---|
| [`style_guide.md`](method/style_guide.md) | Prose and code conventions |
| [`rendering_pipeline.md`](method/rendering_pipeline.md) | The token flow — claims ledger → `{{TOKEN}}` map → injected manuscript → Pandoc/XeLaTeX — plus the `TEMPLATE_REPO_ROOT` requirement that a symlinked sidecar creates |
| [`release_checklist.md`](method/release_checklist.md) | The release-day sequence: gates, authenticated recorder re-execution, Lean bridge, token map, figures, version bump, full-profile acceptance, tag, deposit; what blocks a release and who clears it |

## Proposed protocols (researcher/contributor, mixed status)

| File | What it is for |
|---|---|
| [`extension_protocols.md`](extension_protocols.md) | Three protocol proposals responding to the open Major TODO items: a fitted external probe arm (equal fit opportunity), a stipulated aligned spelling-to-pronunciation artifact, and the L4→L3 `P(word | case role)` label/split/observation account. Their status is mixed and stated per protocol: P3's case-word estimand is pre-registered and scored (decision 121), P2's stipulated artifact, offline comparator arms, and zero-shot probe arms are recorded (the probe run `synthetic_aligned_g2p_probe_v1`), and P1's warm-up compute gate is measured (decision 122) with the fit pipeline itself still pending external model infrastructure; [`design/`](design/) holds only adopted decision records |

## Science and design decisions (contributor)

One file per decision that could reasonably have gone the other way. Each states
the alternative and why it was rejected.

| File | Decision |
|---|---|
| [`design/README.md`](design/README.md) | The index of this subdirectory, with the decision each file records |
| [`design/AGENTS.md`](design/AGENTS.md) | How to write and verify a decision record: what must be named, what is gated, what governs on disagreement |
| [`design/connected_runtime.md`](design/connected_runtime.md) | Connected runtime: observation access, clocks, train-only bridges, resets, conditional references, locally frozen GUM evaluation and next-word generation limits |
| [`design/architecture.md`](design/architecture.md) | The level hierarchy, per-level A/B/C/D, the scale arithmetic |
| [`design/stage2_g2p.md`](design/stage2_g2p.md) | The 1:1 letter↔phone alignment restriction and what it biases |
| [`design/stage3_word.md`](design/stage3_word.md) | Why the word level factorizes into word identity × position instead of trie nodes, what the closed vocabulary costs, and why three of five mis-wirings beat the bar |
| [`design/stage4_case.md`](design/stage4_case.md) | Why the case level observes only UPOS (the circularity trap), why one declared threshold is vacuous on gold and above-reference on derived labels, and why the gold population's null is exact rather than small-sample |
| [`design/cerebrum_mapping.md`](design/cerebrum_mapping.md) | The exact CEREBRUM source tuple, Figure 13's caption-only boundary, and AILM's explicitly project-owned pymdp crosswalk |
| [`design/no_embeddings.md`](design/no_embeddings.md) | Why kernel composition is not a DisCoCat functor, and what a valid future bridge would require |
| [`design/generation_problem.md`](design/generation_problem.md) | The measured L3→L1g policy generator and the separate connected next-word API, with their distinct evidence |
| [`design/speech_boundary.md`](design/speech_boundary.md) | Where DSP ends and pymdp begins |
| [`design/power_and_ci.md`](design/power_and_ci.md) | Power analysis and exact confidence intervals for the paired tests |

## Provenance (auditor)

| File | What it is for |
|---|---|
| [`provenance/README.md`](provenance/README.md) | Quick reference: the section map of `sources.md` and the digest-check commands |
| [`provenance/AGENTS.md`](provenance/AGENTS.md) | The ownership boundary and the rules for adding or regenerating an artifact |
| [`provenance/sources.md`](provenance/sources.md) | Every corpus and paper: URL, license, vendored artifact, sha256, and the derivation choices that bias each dataset |

See also [`data_statement.md`](reference/data_statement.md) (in Reference, above) for
what each corpus **generalizes to** (Bender and Friedman) — `sources.md` answers
whether a number can be rebuilt, `data_statement.md` answers what population it
can be about.

## Interfaces (auditor — read before citing)

| File | What it is for |
|---|---|
| [`../src/ailm/dashboard/README.md`](../src/ailm/dashboard/README.md) | The interactive dashboard: how to run it, every configuration key, the four endpoints, the speech lifecycle, and — the part worth reading before citing anything it shows — what the artifact is and is not. It is a demonstrator, backs no manuscript number, has no `claim:` tag, and runs Level 5 on *inferred* syntactic case roles where the probe uses gold (UD-derived) ones |

## Outside `docs/`

| Path | What it is |
|---|---|
| [`../README.md`](../README.md) | Project overview, honest scope statement, stage roadmap |
| [`../AGENTS.md`](../AGENTS.md) | Invariants, stage gates and their restatements, test commands |
| [`../CHANGELOG.md`](../CHANGELOG.md) | Concise release record; the full pre-1.0.0 history is in `docs/development_history.md` |
| [`../data/claims.json`](../data/claims.json) | The claims ledger — the single source of numeric truth. Since schema `ailm-claims-v2` every entry carries an epistemic `status` — `confirmatory` (a schema label for pre-registered and scored under its registered rule, not a claim of independent-population confirmation), `exploratory` (test-selected sweeps and subsample arms), `descriptive` (measured, no inferential claim; the default when ambiguous), or `analytic` (closed-form) — assigned mechanically by recorder of origin in [`../scripts/assign_claim_status.py`](../scripts/assign_claim_status.py) and documented in [`../data/README.md`](../data/README.md) |
| [`../docs/manuscript/config.yaml`](../docs/manuscript/config.yaml) | Paper metadata consumed by the renderer and by the `META_*` tokens |
| [`../pyproject.toml`](../pyproject.toml) | Dependency pins, pytest options, the coverage floor |
| [`../src/ailm/`](../src/ailm/) | All logic. Per-package `README.md` and `AGENTS.md` live beside the code |
| [`../scripts/`](../scripts/) | Orchestrators and evidence writers; new reusable computation belongs in `src/ailm/`, while audited recorder-local estimands remain explicit migration debt. [`scripts/README.md`](../scripts/README.md) gives the running order |
| [`../scripts/verify_release.py`](../scripts/verify_release.py) | The release-profile runner, invoked only as `.venv/bin/python -I -S scripts/verify_release.py` so neither caller nor venv Python startup code can precede the release boundary. `quick` is the developer loop — environment, format, lint, typing, every test except recorder replay and raw-corpus regeneration, and a bounded gate audit. `full` proves locked package builds, authenticated raw-corpus regeneration, source/configuration/model/call bindings for every network-excluded recorder, live external-call cache replay, full coverage, gate/formal checks, source-generated prose and figures, and the validated final PDF. It refuses unsafe Git index flags and dirty inputs unless the development-only dirty override is explicit; that override can never yield `publication_acceptance`. Sidecar Python steps use a controlled no-`site` import bootstrap, subprocesses use a fixed executable path and explicit environment allowlist, and the receipt binds command evidence plus Python-startup, external-tool, Git, cache, corpus, package, and template identities. Guarded by `tests/test_release_orchestration.py` |
| [`../tests/`](../tests/) | The gates |
| [`../formal/`](../formal/) | The Lean 4 proofs and their generated ledger bridge — see [`formal.md`](formal.md) |

## Manuscript source files

| Document | Contents |
| --- | --- |
| [`manuscript/00_abstract.md`](manuscript/00_abstract.md) | Manuscript source: Abstract {#sec:abstract} |
| [`manuscript/01_introduction.md`](manuscript/01_introduction.md) | Manuscript source: Introduction {#sec:introduction} |
| [`manuscript/02_background.md`](manuscript/02_background.md) | Manuscript source: Background {#sec:background} |
| [`manuscript/03_methods.md`](manuscript/03_methods.md) | Manuscript source: Methods {#sec:methods} |
| [`manuscript/04_results.md`](manuscript/04_results.md) | Manuscript source: Results {#sec:results} |
| [`manuscript/05_discussion.md`](manuscript/05_discussion.md) | Manuscript source: Discussion {#sec:discussion} |
| [`manuscript/06_related_work.md`](manuscript/06_related_work.md) | Manuscript source: Related Work {#sec:related} |
| [`manuscript/07_reproducibility.md`](manuscript/07_reproducibility.md) | Manuscript source: Reproducibility and Availability {#sec:reproducibility} |
| [`manuscript/08_conclusion.md`](manuscript/08_conclusion.md) | Manuscript source: Conclusion {#sec:conclusion} |
| [`manuscript/98_supplement_audit_trails.md`](manuscript/98_supplement_audit_trails.md) | Manuscript source: Supplement A: The Gate-Audit Trails, the Mis-Wiring Catalogs, and the Confirmatory Replication {#sec:audit_trails} |
| [`manuscript/98_supplement_channel_factorial.md`](manuscript/98_supplement_channel_factorial.md) | Manuscript source: Supplement B: The Channel Factorial and the Proper-Loss Re-Scoring {#sec:channel_factorial} |
| [`manuscript/98_supplement_complexity.md`](manuscript/98_supplement_complexity.md) | Manuscript source: Supplement C: What Each Level Costs to Run, in Exact Operations and Bytes {#sec:complexity} |
| [`manuscript/98_supplement_external_benchmarks.md`](manuscript/98_supplement_external_benchmarks.md) | Manuscript source: Supplement D: Non-Comparative Reference Probes on the Same Next-Symbol Tasks {#sec:external_benchmarks} |
| [`manuscript/98_supplement_lean.md`](manuscript/98_supplement_lean.md) | Manuscript source: Supplement E: The Machine-Checked Lean Layer {#sec:lean} |
| [`manuscript/98_supplement_statistics.md`](manuscript/98_supplement_statistics.md) | Manuscript source: Supplement F: The Paired Test, the Correction Family, and the Bounds for the Nulls {#sec:statistics} |
| [`manuscript/99_references.md`](manuscript/99_references.md) | Manuscript source: References {#sec:references} |
| [`manuscript/AGENTS.md`](manuscript/AGENTS.md) | Manuscript source: Manuscript agent contract — AILM |
| [`manuscript/MANUSCRIPT_STATUS.md`](manuscript/MANUSCRIPT_STATUS.md) | Manuscript source: Manuscript Status |
| [`manuscript/README.md`](manuscript/README.md) | Manuscript source: `docs/manuscript/` — the AILM paper |
| [`manuscript/SYNTAX.md`](manuscript/SYNTAX.md) | Manuscript source: Manuscript Syntax Reference — AILM |
| [`manuscript/preamble.md`](manuscript/preamble.md) | Manuscript source: LaTeX Preamble |
