# Output Inventory

Every project-owned current artifact class under `output/`, what writes it, and
what consumes it. Current outputs are regenerable; a curated set of publication
surfaces is tracked and the transient remainder is ignored. A named legacy-residue
section keeps old ignored cache directories visible without pretending a current
writer owns them. See
[`output_conventions.md`](output_conventions.md) for the exact tracking rule and
regeneration order.

Paths below are relative to the project root. A fresh clone contains the curated
figures, resolved manuscript, combined PDF, publication reports, token map, and
two stable web surfaces, but not transient slides, LaTeX intermediates, caches,
or hydrated documentation.

## Figures — `output/figures/`

Written by `scripts/generate_figures.py`, which discovers every public `generate_*`
function in every non-underscore module of `ailm.figures` rather than consulting a
hardcoded registry. List them without writing anything:

```bash
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py --list
```

| File | Cross-reference label | Generator module | What it argues |
|---|---|---|---|
| `hierarchy.png` | `{#fig:hierarchy}` | `ailm.figures.architecture` | The level stack — one row per level, naming the hidden factor, the observation modality, what `A` is, what `B` carries, and the stage that builds it. Identity-versus-learned `A` is encoded three ways (badge, tint, legend) because L2B is the first model whose `A` is a non-identity emission |
| `inference_cycle.png` | `{#fig:inference_cycle}` | `ailm.figures.architecture` | The four pymdp calls in one symbol step, with `A` and `B` labeled at the exact call each enters. This is what makes the orientation argument legible: `A` is consulted at `infer_states`, `B` at `update_empirical_prior`, so swapping them is visibly a claim about which variable the posterior is over |
| `letter_transition.png` | `{#fig:letter_transition}` | `ailm.figures.matrices` | The Brown letter bigram matrix that parameterizes `B` at Level 1, built through the same `build_letter_transition` call the agent uses — so the figure cannot show a matrix the model does not have. High-probability cells are marked because that structure is exactly what vanishes when the matrix is put in the wrong slot |
| `wiring_comparison.png` | `{#fig:wiring_comparison}` | `ailm.figures.matrices` | The load-bearing figure. Correct wiring (`A` identity, `B` bigram) beside the anti-informative wiring (`A` bigram, `B` identity), with each measured perplexity. Both are column-stochastic, both pass pymdp's `_validate()`, and both panels look equally plausible — which is the entire problem |
| `perplexity_ladder.png` | `{#fig:perplexity_ladder}` | `ailm.figures.results` | Uniform, context-free unigram, model, and order-1 information floor on one axis, for both levels. Separates two very different claims: "beats uniform" and "beats unigram" |
| `information_ceiling.png` | `{#fig:information_ceiling}` | `ailm.figures.results` | Stage 1's unreachable original target, in nats. Shades the achievable band and places the demanded gain outside it |
| `g2p_accuracy.png` | `{#fig:g2p_accuracy}` | `ailm.figures.results` | Panel A: the two decoders at full scale, nearly indistinguishable. Panel B: a labeled truncated axis carrying the paired McNemar statistic, which is what actually establishes the difference. Also marks the smoothing ceiling as headroom the online architecture does not claim |
| `gate_taxonomy.png` | `{#fig:gate_taxonomy}` | `ailm.figures.gates` | The complete typed enum: VACUOUS / UNREACHABLE / ABOVE_REFERENCE / UNAUDITED / BRACKETED_BOUND / BRACKETED_REFERENCE. A threshold is only meaningful relative to the trivial baseline and the declared kind of its strongest comparator |
| `subsample_bias.png` | `{#fig:subsample_bias}` | `ailm.figures.gates` | Prefix versus stride versus full held-out set. A reproducible subsample is not a representative one, and the fix was not to shrink the eval |
| `complexity_ops.png` | `{#fig:complexity_ops}` | `ailm.figures.complexity` | See its caption in `docs/manuscript/98_supplement_complexity.md` |
| `complexity_memory.png` | `{#fig:complexity_memory}` | `ailm.figures.complexity` | See its caption in `docs/manuscript/98_supplement_complexity.md` |
| `gnn_pipeline.png` | `{#fig:gnn_pipeline}` | `ailm.figures.composition` | See its caption in `docs/manuscript/03_methods.md` |
| `composition.png` | `{#fig:composition}` | `ailm.figures.composition` | See its caption in `docs/manuscript/03_methods.md` |
| `connected_core.png` | `{#fig:connected_core}` | `ailm.figures.connected` | Runtime-derived causal schedule and report-bound equal-document GUM character-loss comparison; shared observation encoder and OOV scope remain explicit |
| `dashboard_observation.png` | `{#fig:dashboard_observation}` | `ailm.figures.dashboard_observation` | Warm local dashboard capture separates input analysis from the L3-to-L1g lexical policy reply; capture provenance and scope are recorded in the figure caption |
| `word_miswirings.png` | `{#fig:word_miswirings}` | `ailm.figures.composition_results` | See its caption in `docs/manuscript/04_results.md` |
| `case_both_defects.png` | `{#fig:case_both_defects}` | `ailm.figures.composition_results` | See its caption in `docs/manuscript/04_results.md` |
| `generation_sweep.png` | `{#fig:generation_sweep}` | `ailm.figures.composition_results` | See its caption in `docs/manuscript/04_results.md` |
| `channel_comparison.png` | `{#fig:channel_comparison}` | `ailm.figures.composition_results` | See its caption in `docs/manuscript/04_results.md` |
| `case_matrices.png` | `{#fig:case_matrices}` | `ailm.figures.interpretability` | See its caption in `docs/manuscript/03_methods.md` |
| `word_matrices.png` | `{#fig:word_matrices}` | `ailm.figures.interpretability` | See its caption in `docs/manuscript/03_methods.md` |
| `trie_narrowing.png` | `{#fig:trie_narrowing}` | `ailm.figures.interpretability` | See its caption in `docs/manuscript/03_methods.md` |
| `level_gate_summary.png` | `{#fig:level_gate_summary}` | `ailm.figures.level_gate_summary` | See its caption in `docs/manuscript/05_discussion.md` |
| `learning_agreement.png` | `{#fig:learning_agreement}` | `ailm.figures.learning` | See its caption in `docs/manuscript/04_results.md` |
| `clause_null.png` | `{#fig:clause_null}` | `ailm.figures.levels56` | See its caption in `docs/manuscript/04_results.md` |
| `inference_evidence.png` | `{#fig:inference_evidence}` | `ailm.figures.inference_evidence` | See its caption in `docs/manuscript/04_results.md` |
| `narrative_channel.png` | `{#fig:narrative_channel}` | `ailm.figures.levels56` | See its caption in `docs/manuscript/04_results.md` |
| `generation_samples.png` | `{#fig:generation_samples}` | `ailm.figures.samples` | See its caption in `docs/manuscript/04_results.md` |
| `phoneme_matrices.png` | `{#fig:phoneme_matrices}` | `ailm.figures.matrices` | See its caption in `docs/manuscript/03_methods.md` |
| `belief_dynamics.png` | `{#fig:belief_dynamics}` | `ailm.figures.belief_dynamics` | See its caption in `docs/manuscript/03_methods.md` |
| `confirmatory_replication.png` | `{#fig:confirmatory_replication}` | `ailm.figures.confirmatory` | See its caption in `docs/manuscript/04_results.md` |
| `multiplicity_forest.png` | `{#fig:multiplicity_forest}` | `ailm.figures.multiplicity` | See its caption in `docs/manuscript/98_supplement_statistics.md` |
| `external_benchmarks.png` | `{#fig:external_benchmarks}` | `ailm.figures.benchmarks` | See its caption in `docs/manuscript/98_supplement_external_benchmarks.md` |
| `calibration_floor.png` | `{#fig:calibration_floor}` | `ailm.figures.calibration_floor` | See its caption in `docs/manuscript/98_supplement_external_benchmarks.md` |
| `incontext_sweep.png` | `{#fig:incontext_sweep}` | `ailm.figures.incontext_sweep` | See its caption in `docs/manuscript/98_supplement_external_benchmarks.md` |
| `aligned_g2p.png` | `{#fig:aligned_g2p_arms}` | `ailm.figures.aligned_g2p` | See its caption in `docs/manuscript/98_supplement_external_benchmarks.md` |
| `case_word_sweep.png` | `{#fig:case_word_sweep}` | `ailm.figures.case_word_sweep` | See its caption in `docs/manuscript/04_results.md` |
| `null_bounds.png` | `{#fig:null_bounds}` | `ailm.figures.null_bounds` | See its caption in `docs/manuscript/98_supplement_statistics.md` |
| `scaling_ladder.png` | `{#fig:scaling_ladder}` | `ailm.figures.scaling_ladder` | See its caption in `docs/manuscript/04_results.md` |

The table is a reader-facing projection of
`output/figures/figure_registry.json`, not an independent registry.
`tests/test_documentation.py` compares the filename, label, and generator-module
triples exactly, with planted missing-row and wrong-generator controls. Update
the registry through the figure pipeline, then update this projection from that
artifact; never reconstruct it from memory.

Inputs to every figure: `output/data/manuscript_variables.json` for measured numbers,
the vendored corpus artifacts for matrix content, and the code's own constants for
structural figures. No generator contains a typed-in measurement.

Two figure modules deserve a note on internal consistency:

- `ailm.figures.gates._classify` is a thin wrapper around
  `ailm.gates.classify_gate`, and `scripts/audit_gates.py` reaches the same function
  through `ailm.gates.evaluate_gate`. There is one classifier, not two copies kept
  in step, so the figure and the re-runnable audit cannot disagree about a verdict
  — drift is structurally impossible rather than merely unlikely.
- Each function in `ailm.figures.results` re-derives an identity that must hold
  between separately recorded claims and raises if it does not, so a figure cannot
  quietly outlive the ledger it was drawn from.

## Manuscript token map — `output/data/`

| File | Written by | Consumed by |
|---|---|---|
| `manuscript_variables.json` | `scripts/z_generate_manuscript_variables.py` (logic in `src/ailm/manuscript_variables/`) | `infrastructure.rendering.manuscript_injection` for prose substitution; `ailm.figures._common.load_manuscript_variables` for figures |

Sorted keys, one flat `{"TOKEN": "value"}` mapping. Token names are ledger labels
uppercased, plus `_RUN` / `_SOURCE` provenance companions, plus derived quantities,
plus `META_*` build metadata. Full rules in
[`rendering_pipeline.md`](../method/rendering_pipeline.md).

## Injected manuscript — `output/manuscript/`

| File pattern | Written by | Note |
|---|---|---|
| `*.md` | `infrastructure.rendering.manuscript_injection.write_resolved_manuscript_tree` | Token-substituted copies of `docs/manuscript/*.md`, one per authored section |
| `config.yaml` | refreshed from `docs/manuscript/config.yaml` by `resolve_manuscript_dir` | Unless the injected copy owns generated ordering |
| `preamble.md` | refreshed from `docs/manuscript/preamble.md` | LaTeX preamble injected at compile time |
| `*.bib` | refreshed from `docs/manuscript/` | If present |

**The renderer reads this directory, not `docs/manuscript/`.** That is the single most
consequential fact in this inventory; see the hand-editing trap in
[`output_conventions.md`](output_conventions.md).

## PDF and LaTeX intermediates — `output/pdf/`

| File | Written by | Note |
|---|---|---|
| `_combined_manuscript.md` | `scripts/pipeline/stage_03_render.py` | The assembled single-document markdown Pandoc is handed |
| `active_inference_language_model_combined.pdf` | same | The combined publication PDF. **Absent when any section fails LaTeX** — check `output/reports/diagnostics.json` |
| `_combined_manuscript.tex`, `.aux`, `.log` | same | XeLaTeX intermediates. The `.log` is where an undefined citation or control sequence is named |

## Per-section slides — `output/slides/`

One Beamer PDF per manuscript section, plus its intermediates: `.tex`, `.aux`,
`.log`, `.nav`, `.snm`, `.toc`, `.vrb`. Also `_slides_math_header.tex`.

Slide rendering does **not** invoke `mmdc`, which is why "slides render but the
combined PDF fails" is diagnostic of a missing `chrome-headless-shell` rather than of
a manuscript problem. A slide deck that fails LaTeX while its PDF still exists is a
*partial* failure — the PDF may be truncated — and the renderer reports it as such.

## Per-section web — `output/web/`

One HTML file per section, named `manuscript__<section>.html`, plus
`_combined_manuscript.md` and `favicon.ico`.

Two further files there are **not** render output. Every quick/full release
profile builds them, before pytest, so their interface checks execute rather
than skip in a clean checkout. They carry the model itself rather than a
description of it:

| File | Written by | Contains |
|---|---|---|
| `ailm_model.json` | `scripts/export_web_model.py` | The ignored local letter transition, lexicon, held-out passage, and displayed claims used to build and test the standalone demo. It is generated in a clean acceptance run but is not a tracked distribution surface |
| `ailm.html` | `scripts/build_web_ailm.py` | An ignored self-contained local demo with the model inlined. It runs the forward filter, top-down lexicon preference, and policy posterior in the browser, but is not tracked or distributed |

`ailm.html` opens with no server and no network. Its first act is a self-check
that recomputes `stage1_letter_perplexity` from the inlined transition and
compares it against the recorded claim; on disagreement it renders an error
instead of the demo, because a demo that has drifted from the paper is worse than
no demo — it looks like corroboration. Rebuild both after any re-record, in that
order, or the page will show stale numbers and say so.

The dashboard (`.venv/bin/python -m ailm.dashboard`) is a separate interactive
surface with per-level views and browser speech controls. Those controls do not
establish an evaluated acoustic model or make its views the connected inference
runtime. Use `scripts/run_connected_ailm.py` for the explicit multirate trace.
`ailm.html` is a generated local subset that opens from `file://`; it is neither
the tracked web publication nor a distributed package asset. The comparison table lives
in `src/ailm/dashboard/README.md`, and `tests/test_interface_inventory.py` pins
both to the same builders and refuses an undeclared third interface.

## Render diagnostics — `output/reports/`

| File | Written by | Contains |
|---|---|---|
| `diagnostics.json` | `infrastructure.core.logging.diagnostic.DiagnosticReporter`, via the render pipeline | Per-file severity, category, message, log path, log tail, recent LaTeX errors, and detected issues. Cleared at the start of every render |
| `artifact_manifest.json` | template Stage 04 | The authenticated stable publication-artifact inventory |
| `evidence_registry.json` | template validation | Evidence surfaces and their bound provenance |
| `gate_audit.json` | `scripts/audit_gates.py --json` | Typed gate verdicts reconstructed from their registered comparators |
| `gate_audit_ablation.json` | `scripts/audit_gates.py --json` | Separate retained case-only clause verdict; required by ablation figure consumers and included in the publication allowlist |
| `manuscript_composition.json` | template render | Ordered manuscript composition and section inputs |
| `rendered_provenance.json` | template render/validation | Source, resolved, evidence, and rendered digests for every consumed manuscript file |
| `validation_report.json` | template Stage 04 | Machine-readable publication validation; `validation_report.md` is an ignored human-readable projection when emitted |
| `connected_gum.json` | `scripts/record_connected_gum_claims.py` | Frozen-population paired document results, OOV coverage, uncertainty and complete canonical execution binding; the connected figure authenticates it against the ledger and current bound sources |
| `release-receipt.json` | `scripts/verify_release.py` | The quick/full transaction, repository/toolchain snapshots, steps, test/coverage evidence, final-state equality, and publication-acceptance verdict |
| `logs/` | `scripts/verify_release.py` | Ignored stdout/stderr per release step under `logs/<run_id>/`; each execution retains separate evidence |

This is the first thing to open when a render fails. It localizes the failure to a
named section with its recent LaTeX errors, instead of leaving you to grep the logs.

## External-model working receipts — `output/bench_raw/`

`scripts/record_external_benchmark_claims.py` writes ignored per-model result
sidecars plus `cache_receipt.json` here as progress/debugging receipts. They never
re-enter a measurement and are not the provenance accepted for a network-authored
claim. Canonical replay lives in `cache/bench_calls.jsonl`; deterministic
`cache/network_claim_bindings/*.json` receipts bind the ledger namespaces to the
exact trusted calls.

## Curated architecture export — `output/architecture-map/`

When Groma is installed, `groma export output/architecture-map` renders the
CLI-owned architecture as a static site with embedded source snapshots. It is
an ignored inspection aid, not model execution evidence or a publication
receipt. Follow `groma agent-instructions` before scanning or curating; preserve
curated ownership and authenticate embedded source bytes after source changes.
Read source/model responsibilities from [architecture.md](../architecture.md).

## Generated upstream candidate — `output/upstream_pymdp_pr/`

`scripts/generate_pymdp_sparse_patch.py` owns the candidate patch and
`PR_DESCRIPTION.md`. `--check` regenerates both in memory, verifies that the patch
applies to a copied installed pymdp tree, runs the fresh-process dense/sparse
equivalence probe, and compares the bundle bytes. The directory is ignored and is
not evidence that a pull request was submitted.

## Legacy release scratch — `output/.release-cache/`

Older release runs placed disposable caches here. The current
`scripts/verify_release.py` writes its uv and Matplotlib caches to
`.venv/release-cache/`, outside the publication and source inventories. A dependency-cache symlink inside
`output/` fails the template's artifact inventory even below a hidden directory;
an ignored cache is not automatically an admissible output. No current release
writer uses this old path. Preserve any existing cache outside `output/` before
validation if it contains such entries.

## Legacy ignored residue — `output/bench_cache/`

No current code writes or reads this path. Older checkouts may retain a v1
`calls.jsonl` from before the canonical store moved to
`cache/bench_calls.jsonl`. It is inventory-only residue, cannot satisfy replay or
network-claim bindings, and is not part of the regeneration contract. Its possible
presence is documented so “ignored” is not mistaken for “owned by a current
writer.”

## Not under `output/`

Runtime vendor artifacts live outside `output/`, on purpose. Their complete
authoritative inventory is `ailm.resources.RUNTIME_VENDOR_ARTIFACTS`;
`tests/test_wheel_resources.py` and `tests/test_corpus_source_integrity.py` bind
that inventory to the source checkout, wheel package data, loaders, checksums, and
provenance. This document deliberately does not duplicate the filenames: the
registry is the one list that must change when a runtime artifact is added.

`data/claims.json` also lives outside `output/`. It is the tracked evidence
ledger, written by the claim recorders under a single-writer lock; every entry
carries a `run_id`, `source`, value, and epistemic status.

## Cross-references

- [`output_conventions.md`](output_conventions.md) — regeneration order, determinism, the hand-editing trap
- [`rendering_pipeline.md`](../method/rendering_pipeline.md) — how the token map and the injected manuscript are produced
- [`syntax_guide.md`](../syntax_guide.md) — how to reference a figure label from the manuscript
- [`architecture.md`](../architecture.md) — which module owns which generator

## `output/docs/` — the hydrated documentation mirror

`scripts/z_render_docs.py` writes every markdown source under `docs/`, plus
`README.md`, `AGENTS.md` and `CHANGELOG.md`, with each `claim:<label>` replaced
by its recorded value. This transient mirror is regenerable and ignored, and
`--check` fails when it is stale, because a hydrated document
carrying numbers from a previous ledger is the exact failure the mechanism
exists to prevent.

The public path is always a real directory. The writer builds and synchronizes a
complete hidden sibling first, then atomically renames it into an absent public
path or uses native directory exchange on Darwin and Linux filesystems that
provide it. The portable replacement fallback uses descriptor-anchored renames
with rollback and prints an explicit warning about its reader-visible interval.
Old real directories are retained under hidden sibling names so an in-flight
reader does not lose the tree it already opened. A bounded relative pointer made
by the former writer is accepted only for a one-time regeneration migration;
`--check` rejects every symlink publication, and migration removes the retired
pointer rather than leaving a hidden symlink artifact. A failed parent-directory
`fsync` leaves the committed real directory intact but returns nonzero because
durability was not established.

Read `output/docs/` to see the numbers. Edit `docs/` to change the words.

With `--supporting-protocols`, the same generator also writes and validates the
portable `ailm-supporting-docs/` HTML/context bundle beside both PDF and web
consumers. It includes the complete protocol page and data statement plus the
source-hydrated context inventory. The companion index must travel with the
PDF. The current HTML parser checks local targets and anchors, duplicate URI
attributes and IDs, and forbidden rebasing. Generate and then verify both
consumers with the selected template provider:

```bash
TEMPLATE_REPO_ROOT="$AILM_TEMPLATE_ROOT" .venv/bin/python scripts/z_render_docs.py --supporting-protocols
TEMPLATE_REPO_ROOT="$AILM_TEMPLATE_ROOT" .venv/bin/python scripts/z_render_docs.py --check --supporting-protocols
```

A removed authored document is not silently dropped: regeneration refuses a
retired source without its explicit retirement disposition. Retained sibling
generations are historical artifacts, not current pages.
