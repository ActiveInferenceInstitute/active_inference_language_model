# Documentation index — every long-lived document

The flat list of this repository's long-lived documentation, one row per file,
sorted by path. [`README.md`](README.md) is the hub: it groups the same
documents by the question a reader is asking. Use this page when you know the
path or the filename and want to know what it is for; use the hub when you know
the question.

Generated artifacts are listed in [`_generated/README.md`](_generated/README.md)
with the script that refreshes each one. A document that exists and is not
listed here is a defect: `tests/test_documentation.py` fails on it.

| Path | Purpose |
| --- | --- |
| [`README.md`](README.md) | Documentation hub: entry points by audience, grouped by question |
| [`AGENTS.md`](AGENTS.md) | How to work inside `docs/`: what is gated, who owns what, how to verify a change |
| [`architecture.md`](architecture.md) | Code architecture: package map, module ownership, import contract |
| [`decisions.md`](decisions.md) | Append-only chronology of the decisions behind every recorded number |
| [`development_history.md`](development_history.md) | Pre-1.0.0 changelog history; historical, not a current API statement |
| [`extension_protocols.md`](extension_protocols.md) | The three proposed extension protocols and their per-protocol status |
| [`formal.md`](formal.md) | The Lean 4 layer: what is machine-checked and what is deliberately not |
| [`gate_design.md`](gate_design.md) | Comparator-typed gate taxonomy and what an impossibility claim requires |
| [`gnn_composition.md`](gnn_composition.md) | GNN composition declarations and conformance |
| [`hierarchy.md`](hierarchy.md) | The governing level structure; wins on any disagreement |
| [`publication_derivative.md`](publication_derivative.md) | Two-source publication attestation and its isolated replay |
| [`research_queue.md`](research_queue.md) | Exploratory proposals with their builds, comparators and falsifiers |
| [`syntax_guide.md`](syntax_guide.md) | The `{{TOKEN}}` and `claim:<label>` numeric protocols |
| [`testing_philosophy.md`](testing_philosophy.md) | What a green suite is allowed to mean here |
| [`troubleshooting.md`](troubleshooting.md) | Symptom → cause → fix |
| [`_generated/README.md`](_generated/README.md) | Policy for generated documents and the script that refreshes each |
| [`_generated/gate_arms.md`](_generated/gate_arms.md) | Measured wall time per gate arm, rendered from release receipts |
| [`design/README.md`](design/README.md) | Index of the decision records, with the decision each one records |
| [`design/AGENTS.md`](design/AGENTS.md) | How to write and verify a decision record |
| [`design/architecture.md`](design/architecture.md) | The scientific architecture: levels, A/B/C/D, scale arithmetic |
| [`design/cerebrum_mapping.md`](design/cerebrum_mapping.md) | The CEREBRUM source tuple and the project-owned pymdp crosswalk |
| [`design/connected_runtime.md`](design/connected_runtime.md) | Connected runtime: observations, clocks, bridges, resets, GUM evaluation |
| [`design/generation_problem.md`](design/generation_problem.md) | The measured L3→L1g generator and the separate next-word API |
| [`design/no_embeddings.md`](design/no_embeddings.md) | Why kernel composition is not a DisCoCat functor |
| [`design/power_and_ci.md`](design/power_and_ci.md) | Power analysis and exact intervals for the paired tests |
| [`design/speech_boundary.md`](design/speech_boundary.md) | Where DSP ends and pymdp begins |
| [`design/stage2_g2p.md`](design/stage2_g2p.md) | The 1:1 letter↔phone alignment restriction and its bias |
| [`design/stage3_word.md`](design/stage3_word.md) | Why the word level factorizes into identity × position |
| [`design/stage4_case.md`](design/stage4_case.md) | Why the case level observes only UPOS, and the threshold's two failures |
| [`guides/agent_instructions.md`](guides/agent_instructions.md) | Hard behavioral rules and the canonical command reference |
| [`guides/faq.md`](guides/faq.md) | Recurring questions |
| [`guides/forking_guide.md`](guides/forking_guide.md) | Reusing this layout for another staged-build project |
| [`guides/quickstart.md`](guides/quickstart.md) | Clone → interpreter → gates → figures → render |
| [`guides/roadmap.md`](guides/roadmap.md) | Forward-looking only: open protocols and composition directions |
| [`manuscript/README.md`](manuscript/README.md) | The manuscript directory: assembly order and render contract |
| [`manuscript/AGENTS.md`](manuscript/AGENTS.md) | Manuscript agent contract |
| [`manuscript/SYNTAX.md`](manuscript/SYNTAX.md) | Manuscript syntax reference |
| [`manuscript/MANUSCRIPT_STATUS.md`](manuscript/MANUSCRIPT_STATUS.md) | Manuscript status |
| [`manuscript/preamble.md`](manuscript/preamble.md) | LaTeX preamble |
| [`manuscript/config.yaml`](manuscript/config.yaml) | Paper metadata consumed by the renderer and the `META_*` tokens |
| [`manuscript/00_abstract.md`](manuscript/00_abstract.md) | Abstract |
| [`manuscript/01_introduction.md`](manuscript/01_introduction.md) | Introduction |
| [`manuscript/02_background.md`](manuscript/02_background.md) | Background |
| [`manuscript/03_methods.md`](manuscript/03_methods.md) | Methods |
| [`manuscript/04_results.md`](manuscript/04_results.md) | Results |
| [`manuscript/05_discussion.md`](manuscript/05_discussion.md) | Discussion |
| [`manuscript/06_related_work.md`](manuscript/06_related_work.md) | Related work |
| [`manuscript/07_reproducibility.md`](manuscript/07_reproducibility.md) | Reproducibility and availability |
| [`manuscript/08_conclusion.md`](manuscript/08_conclusion.md) | Conclusion |
| [`manuscript/98_supplement_audit_trails.md`](manuscript/98_supplement_audit_trails.md) | Supplement A: gate-audit trails and the replication |
| [`manuscript/98_supplement_channel_factorial.md`](manuscript/98_supplement_channel_factorial.md) | Supplement B: the channel factorial and proper-loss re-scoring |
| [`manuscript/98_supplement_complexity.md`](manuscript/98_supplement_complexity.md) | Supplement C: what each level costs in operations and bytes |
| [`manuscript/98_supplement_external_benchmarks.md`](manuscript/98_supplement_external_benchmarks.md) | Supplement D: non-comparative reference probes |
| [`manuscript/98_supplement_lean.md`](manuscript/98_supplement_lean.md) | Supplement E: the machine-checked Lean layer |
| [`manuscript/98_supplement_statistics.md`](manuscript/98_supplement_statistics.md) | Supplement F: the paired test, the correction family, the null bounds |
| [`manuscript/99_references.md`](manuscript/99_references.md) | References |
| [`method/release_checklist.md`](method/release_checklist.md) | The release-day sequence and what blocks a release |
| [`method/rendering_pipeline.md`](method/rendering_pipeline.md) | The token flow from ledger to injected manuscript to PDF |
| [`method/style_guide.md`](method/style_guide.md) | Prose and code conventions |
| [`provenance/README.md`](provenance/README.md) | Section map of `sources.md` and the digest-check commands |
| [`provenance/AGENTS.md`](provenance/AGENTS.md) | Ownership boundary and rules for regenerating an artifact |
| [`provenance/sources.md`](provenance/sources.md) | Every corpus and paper: URL, license, artifact, checksum, bias |
| [`reference/configuration.md`](reference/configuration.md) | Every knob the shipped code reads, by reader and unset behavior |
| [`reference/data_statement.md`](reference/data_statement.md) | What each corpus generalizes to |
| [`reference/formalism_map.md`](reference/formalism_map.md) | Equations, definitions and propositions mapped to code and tests |
| [`reference/glossary.md`](reference/glossary.md) | Alphabetical definitions of every project-specific term |
| [`reference/hosted_ci.md`](reference/hosted_ci.md) | Hosted CI: jobs, pinned actions, the GNN pin, and the local gate |
| [`reference/notation.md`](reference/notation.md) | Consolidated symbol-to-meaning reference |
| [`reference/output_conventions.md`](reference/output_conventions.md) | `output/` is regenerable; what is tracked and the regeneration order |
| [`reference/output_inventory.md`](reference/output_inventory.md) | Every generated figure, token map, and render product |
