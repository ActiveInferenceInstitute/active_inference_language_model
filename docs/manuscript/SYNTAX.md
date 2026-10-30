# Manuscript Syntax Reference — AILM

The project's cross-reference registry. Every table below was derived from the
files actually on disk, not from a plan: section labels from the `# ` headings,
equation/figure/table labels from their `{#…}` markers, figures from
`output/figures/`, and tokens from the live claims ledger. Re-derive it after any
structural change rather than editing counts by hand:

```bash
rg -n "^# " docs/manuscript/*.md                     # section labels
rg -no "\{#eq:[a-z_0-9]*\}"  docs/manuscript/*.md      # equation labels
rg -no "\{#fig:[a-z_0-9]*\}" docs/manuscript/*.md      # figure labels
rg -no "\{#tbl:[a-z_0-9]*\}" docs/manuscript/*.md      # table labels
ls output/figures/                                # figures that exist
.venv/bin/python -c "import json,sys; sys.path.insert(0,'src'); \
  from pathlib import Path; from ailm.manuscript_variables import generate_manuscript_variables; \
  print('\n'.join(sorted(generate_manuscript_variables(Path('.')))))"   # tokens
```

This file, `AGENTS.md`, `MANUSCRIPT_STATUS.md`, and `README.md` are excluded from token substitution by
`infrastructure.rendering.manuscript_injection.EXCLUDED_DOC_FILENAMES`, so literal
token examples here are safe and are never copied into `output/manuscript/`.

## Assembly order

Files assemble lexicographically from `output/manuscript/` (the token-resolved copy
of this directory), so the numeric prefix *is* the section order. `99_` keeps the
bibliography last.

| File | Section H1 | Label |
|---|---|---|
| `00_abstract.md` | Abstract | `{#sec:abstract}` |
| `01_introduction.md` | Language and Cognition Through Explicit Active Inference | `{#sec:introduction}` |
| `02_background.md` | Generative Models of Linguistic Belief and Action | `{#sec:background}` |
| `03_methods.md` | Model Construction, Conditional Coupling, and Falsifiable Verification | `{#sec:methods}` |
| `04_results.md` | Connected and Component Results Under Their Registered Comparators | `{#sec:results}` |
| `05_discussion.md` | Interpretation, External-Validity Limits, and Unmeasured Extensions | `{#sec:discussion}` |
| `06_related_work.md` | Related Work: Active Inference, Finite-State Language, and Evaluation | `{#sec:related}` |
| `07_reproducibility.md` | Source-Bound Reproduction, Deterministic Builds, and Availability | `{#sec:reproducibility}` |
| `08_conclusion.md` | Established Findings and the Next Independent Tests | `{#sec:conclusion}` |
| `98_supplement_audit_trails.md` | Supplement A: Gate Audits, Mutation Controls, and Held-Back Replication | `{#sec:audit_trails}` |
| `98_supplement_channel_factorial.md` | Supplement B: Channel Factorial and Proper-Loss Re-Scoring | `{#sec:channel_factorial}` |
| `98_supplement_complexity.md` | Supplement C: Exact Computational and Memory Costs | `{#sec:complexity}` |
| `98_supplement_external_benchmarks.md` | Supplement D: Non-Comparative Reference Probes on the Same Next-Symbol Tasks | `{#sec:external_benchmarks}` |
| `98_supplement_lean.md` | Supplement E: Machine-Checked Lean Proofs and Their Scope | `{#sec:lean}` |
| `98_supplement_statistics.md` | Supplement F: Paired Tests, Multiplicity, and Null Bounds | `{#sec:statistics}` |
| `99_references.md` | References | `{#sec:references}` |

`preamble.md` is not a section — it is copied verbatim and parsed by
`infrastructure/rendering/latex_utils.py`. `config.yaml` and `references.bib` are
copied verbatim too.

Every H1 carries a `{#sec:…}` label so cross-section references survive
reordering. Reference sections as `[@sec:results]`.

## Connected evaluation additions

The connected comparison opens `04_results.md`, before the historical component
results. Its source-derived labels are
`sec:connected_gum` and `tbl:connected_gum` in `04_results.md`. Its measured tokens
are the uppercased `connected_gum_` ledger namespace; derived outcome, whole-word
loss and abbreviated digest tokens are produced by
`src/ailm/manuscript_variables/connected.py`. The render refuses missing claims
or inconsistent counts, uncertainty direction, protocol fields or OOV disclosure.

## Dashboard observation

`fig:dashboard_observation` and `sec:dashboard` are defined in
`07_reproducibility.md`. The figure frames an authentic browser raster from
`data/dashboard_capture/`, with byte, decoded-pixel, and served-source bindings.
It is a warm inspection demonstration, separate from scientific evaluation.

The aligned spelling-to-pronunciation methods use
`MODEL_DEFAULT_EMISSION_SMOOTHING` and `MODEL_DEFAULT_SMOOTHING`, produced
directly from `LevelConfig` defaults. The source scanner also rejects literal
additive-smoothing coefficients in prose.

## Equation label registry

Every display equation uses the Pandoc-crossref form and is referenced `[@eq:label]`.
Numbering is assigned automatically by the renderer. Keep display math outside
raw theorem environments so the same expression appears in PDF and HTML.
Unlabeled displays and raw display environments fail source validation.
**Never** write raw LaTeX `\ref` or `\eqref` in Markdown — including inside a
`\begin{definition}` block, where it would render in the PDF path and vanish
everywhere else.

```markdown
$$
H_2 \;=\; \log_2 \mathrm{PPL}
$$ {#eq:bits_per_char}

Referenced in prose as [@eq:bits_per_char].
```

| Label | Content | Defined in |
|---|---|---|
| `{#eq:vfe}` | Variational free energy, complexity + accuracy decomposition | `02_background.md` |
| `{#eq:efe}` | Expected free energy, epistemic + pragmatic value | `02_background.md` |
| `{#eq:generative}` | The discrete POMDP joint | `02_background.md` |
| `{#eq:matrix_entries}` | Entries of the matrix tuple | `02_background.md` |
| `{#eq:message_log_loss}` | Information and receiver-mismatch decomposition | `03_methods.md` |
| `{#eq:transition_cross_entropy}` | Transition entropy and conditional KL decomposition | `98_supplement_audit_trails.md` |
| `{#eq:agent_memory}` | Exact parameter-array storage from factor shapes | `98_supplement_complexity.md` |
| `{#eq:pomdp}` | The four matrices $\langle A, B, C, D\rangle$ | `02_background.md` |
| `{#eq:filtering}` | One-step predictive prior $\rho_t$ and posterior correction $q_t$ | `02_background.md` |
| `{#eq:perplexity}` | Per-symbol perplexity from the predictive $\rho_t$ — the single canonical definition | `02_background.md` |
| `{#eq:conditional_entropy}` | Order-1 training-set conditional entropy $H_{\text{train}}$ and the perplexity floor | `02_background.md` |
| `{#eq:case_free_energy}` | Case-conditioned free energy $F_{\text{case}}[q]$ | `02_background.md` |
| `{#eq:forward_recursion}` | The oracle's array-layout form of `eq:filtering` | `03_methods.md` |
| `{#eq:decoder_equivalence}` | Zero-disagreement criterion between pymdp and the oracle | `03_methods.md` |
| `{#eq:mcnemar}` | McNemar's exact test — two-sided exact binomial on the discordant pairs | `98_supplement_statistics.md` |
| `{#eq:step_cost}` | Per-step arithmetic cost of the filtering recursion | `98_supplement_complexity.md` |
| `{#eq:bits_per_char}` | Perplexity in bits per symbol | `04_results.md` |
| `{#eq:signal_captured}` | Fraction of available order-1 signal captured | `04_results.md` |
| `{#eq:gate_satisfies}` | Declared (inclusive) $x \succeq t$ and audit (strict) $x \succ t$ satisfies relations | `03_methods.md` |
| `{#eq:gate_verdict}` | Population-indexed, comparator-typed VACUOUS / UNREACHABLE / ABOVE_REFERENCE / UNAUDITED / BRACKETED classification $\operatorname{verdict}(t;P)$ | `03_methods.md` |
| `{#eq:reference_no_impossibility}` | Why a failing exceedable comparator licenses no impossibility claim | `03_methods.md` |
| `{#eq:policy_posterior}` | The policy posterior, expected free energy weighted against the habit prior | `03_methods.md` |
| `{#eq:heldout_bound}` | Exact all-position order-1 lower bound on the Stage 1 evaluation passage | `98_supplement_audit_trails.md` |
| `{#eq:policy_reduction}` | The policy posterior under a deterministic transition and an identity emission | `04_results.md` |
| `{#eq:paired_difference}` | Discordant win share mapped back to a paired difference $\delta$ | `98_supplement_statistics.md` |

Perplexity is defined **once**, in `02_background.md`. Do not restate it; reference
`[@eq:perplexity]`. Two definitions of the same quantity is how the two numbers
diverge.

## Figure registry

```markdown
![Self-contained caption.](../../output/figures/name.png){#fig:name width=85%}

Referenced in prose as [@fig:name].
```

Paths are relative to the manuscript directory, using `../../output/figures/`.
Always set `width=` to keep floats placeable. Captions must be self-contained. Separate short accessibility descriptions live
in `data/figure_alt_text.yaml`; the registry carries both caption and alt text.

Every figure is embedded **exactly once**. A second `![…]{#fig:x}` for the same
label makes pandoc-crossref renumber silently.

Dense benchmark plots also have PDF-only enlarged detail views following their
canonical overview. These are raw `{=latex}` blocks using graphicx
`viewport` and `clip` on the **same PNG**, with no new figure label or caption.
The overview remains the single registry-owned Markdown embedding and the
complete figure for HTML and other output formats. Detail views carry authored
`alt` text and repeat the treatment/control mapping where the shared legend is
outside the viewport. External-probe groups use `pdflscape`, loaded in the fenced
preamble, to preserve titles that extend into the adjacent column gutter.

Keep each complete article-only block inside a TeX group and a braced
class guard: `\begingroup`, then
`\csname @ifclassloaded\endcsname{beamer}{}{` before its contents, followed by
`}` and `\endgroup` before the closing raw fence. Beamer therefore consumes
an empty branch and never executes article landscape or page-break commands.
The explicit group also keeps the template's dense-frame splitter from placing
a frame boundary inside the conditional argument. Do not open the guard in one
raw block and close it in another. Use `{=latex}`, not `{=tex}`, so the web
image-conversion filter leaves these article-only views out of HTML. Verify the
actual Beamer renderer as well as the article proof after any layout change.

Viewport coordinates are layout values in big points, measured from the PNG's
lower-left corner using its recorded DPI; they are not scientific measurements.
Do not redraw the data, create cropped image files, or add a duplicate `#fig:`
label. After the source PNG layout changes, recheck every viewport against the
actual image and render a bounded LaTeX proof before the full paper. Inspect all
axes, legends, intervals, titles, and crop boundaries; use `pdfimages -list` to
check effective resolution and compute printed font size from the source DPI
and font role. A successful compile alone does not establish readability.

| Label | PNG | Generator in `src/ailm/figures/` | Embedded in |
|---|---|---|---|
| `{#fig:case_both_defects}` | `case_both_defects.png` | `composition_results.generate_case_both_defects` | `04_results.md` |
| `{#fig:case_word_sweep}` | `case_word_sweep.png` | `case_word_sweep.generate_case_word_sweep` | `04_results.md` |
| `{#fig:case_matrices}` | `case_matrices.png` | `interpretability.generate_case_matrices` | `03_methods.md` |
| `{#fig:channel_comparison}` | `channel_comparison.png` | `composition_results.generate_channel_comparison` | `04_results.md` |
| `{#fig:clause_null}` | `clause_null.png` | `levels56.generate_clause_null` | `04_results.md` |
| `{#fig:complexity_memory}` | `complexity_memory.png` | `complexity.generate_complexity_memory` | `98_supplement_complexity.md` |
| `{#fig:complexity_ops}` | `complexity_ops.png` | `complexity.generate_complexity_ops` | `98_supplement_complexity.md` |
| `{#fig:dashboard_observation}` | `dashboard_observation.png` | `dashboard_observation.generate_dashboard_observation_figure` | `07_reproducibility.md` |
| `{#fig:connected_core}` | `connected_core.png` | `connected.generate_connected_core_figure` | `04_results.md` |
| `{#fig:composition}` | `composition.png` | `composition.generate_composition_figure` | `03_methods.md` |
| `{#fig:confirmatory_replication}` | `confirmatory_replication.png` | `confirmatory.generate_confirmatory_replication` | `98_supplement_audit_trails.md` |
| `{#fig:g2p_accuracy}` | `g2p_accuracy.png` | `results.generate_g2p_accuracy` | `04_results.md` |
| `{#fig:gate_taxonomy}` | `gate_taxonomy.png` | `gates.generate_gate_taxonomy` | `03_methods.md` |
| `{#fig:generation_samples}` | `generation_samples.png` | `samples.generate_generation_samples` | `04_results.md` |
| `{#fig:generation_sweep}` | `generation_sweep.png` | `composition_results.generate_generation_sweep` | `04_results.md` |
| `{#fig:belief_dynamics}` | `belief_dynamics.png` | `belief_dynamics.build_belief_dynamics_figure` | `03_methods.md` |
| `{#fig:calibration_floor}` | `calibration_floor.png` | `calibration_floor.generate_calibration_floor` | `98_supplement_external_benchmarks.md` |
| `{#fig:external_benchmarks}` | `external_benchmarks.png` | `benchmarks.generate_external_benchmarks` | `98_supplement_external_benchmarks.md` |
| `{#fig:incontext_sweep}` | `incontext_sweep.png` | `incontext_sweep.generate_incontext_sweep` | `98_supplement_external_benchmarks.md` |
| `{#fig:gnn_pipeline}` | `gnn_pipeline.png` | `composition.generate_gnn_pipeline_figure` | `03_methods.md` |
| `{#fig:hierarchy}` | `hierarchy.png` | `architecture.generate_hierarchy_figure` | `03_methods.md` |
| `{#fig:inference_cycle}` | `inference_cycle.png` | `architecture.generate_inference_cycle_figure` | `03_methods.md` |
| `{#fig:information_ceiling}` | `information_ceiling.png` | `results.generate_information_ceiling` | `04_results.md` |
| `{#fig:inference_evidence}` | `inference_evidence.png` | `inference_evidence.generate_inference_evidence` | `04_results.md` |
| `{#fig:learning_agreement}` | `learning_agreement.png` | `learning.generate_learning_agreement` | `04_results.md` |
| `{#fig:letter_transition}` | `letter_transition.png` | `matrices.generate_letter_transition_figure` | `03_methods.md` |
| `{#fig:multiplicity_forest}` | `multiplicity_forest.png` | `multiplicity.generate_multiplicity_forest` | `98_supplement_statistics.md` |
| `{#fig:narrative_channel}` | `narrative_channel.png` | `levels56.generate_narrative_channel` | `04_results.md` |
| `{#fig:null_bounds}` | `null_bounds.png` | `null_bounds.generate_null_bounds` | `98_supplement_statistics.md` |
| `{#fig:perplexity_ladder}` | `perplexity_ladder.png` | `results.generate_perplexity_ladder` | `04_results.md` |
| `{#fig:scaling_ladder}` | `scaling_ladder.png` | `scaling_ladder.generate_scaling_ladder` | `98_supplement_audit_trails.md` |
| `{#fig:phoneme_matrices}` | `phoneme_matrices.png` | `matrices.generate_phoneme_matrices_figure` | `03_methods.md` |
| `{#fig:subsample_bias}` | `subsample_bias.png` | `gates.generate_subsample_bias` | `98_supplement_audit_trails.md` |
| `{#fig:trie_narrowing}` | `trie_narrowing.png` | `interpretability.generate_trie_narrowing` | `03_methods.md` |
| `{#fig:wiring_comparison}` | `wiring_comparison.png` | `matrices.generate_wiring_comparison_figure` | `03_methods.md` |
| `{#fig:word_matrices}` | `word_matrices.png` | `interpretability.generate_word_matrices` | `03_methods.md` |
| `{#fig:level_gate_summary}` | `level_gate_summary.png` | `level_gate_summary.generate_level_gate_summary` | `05_discussion.md` |
| `{#fig:word_miswirings}` | `word_miswirings.png` | `composition_results.generate_word_miswirings` | `98_supplement_audit_trails.md` |

Every PNG in the table exists under `output/figures/` and every generator is discovered
dynamically by `scripts/generate_figures.py` — there is no hardcoded registry in
the generator script, so adding a `generate_*` function to any non-underscore
module of `ailm.figures` is sufficient to have it built.

## Table registry

A table caption is the line immediately after the table, starting with `: ` and
ending with the label. Referenced `[@tbl:label]`.

```markdown
| Comparator | Perplexity |
|---|---|
| Uniform | 30 |

: Self-contained caption. {#tbl:example}
```

| Label | Caption summary | Defined in |
|---|---|---|
| `{#tbl:notation}` | Symbol registry — one meaning per symbol; $\rho_t$ is the predictive prior, $\pi$ the policy | `02_background.md` |
| `{#tbl:case_roles}` | The linguistic case roles and the CEREBRUM functional roles | `02_background.md` |
| `{#tbl:level_stack}` | The level stack, built and specified | `03_methods.md` |
| `{#tbl:level_score_paths}` | Recorded scoring routes and status of the historical components | `03_methods.md` |
| `{#tbl:letter_comparators}` | Level 1 held-out perplexity against every comparator | `04_results.md` |
| `{#tbl:phoneme_comparators}` | Part A perplexity over all dictionary phone positions | `04_results.md` |
| `{#tbl:g2p_comparators}` | Part B G2P position accuracy and its ceilings | `04_results.md` |
| `{#tbl:generation_sweep}` | Generation sweep operating points | `04_results.md` |
| `{#tbl:lexical_channel}` | The lexical channel measured end to end | `04_results.md` |
| `{#tbl:complexity}` | Per-level state, observation, and cost accounting | `98_supplement_complexity.md` |
| `{#tbl:miswiring_signatures}` | Three mis-wirings; which the baseline signature catches | `98_supplement_audit_trails.md` |
| `{#tbl:verification_layers}` | Each verification layer and the failure it forecloses | `03_methods.md` |
| `{#tbl:multiplicity_family}` | The declared significance family and its Bonferroni correction | `98_supplement_statistics.md` |
| `{#tbl:letter_ladder}` | Consolidated Level 1 perplexity ladder | `04_results.md` |
| `{#tbl:letter_information}` | Level 1 information budget in nats | `04_results.md` |
| `{#tbl:phoneme_sequence}` | Consolidated Level 2 Part A perplexity | `04_results.md` |
| `{#tbl:g2p_accuracy}` | Consolidated Level 2 Part B accuracy | `04_results.md` |
| `{#tbl:gate_audit}` | Both shipped gate defects classified by `eq:gate_verdict`, baseline and ceiling indexed by population | `98_supplement_audit_trails.md` |
| `{#tbl:unaudited_gates}` | Every declared Stage 3–7 gate and its audit outcome | `98_supplement_audit_trails.md` |
| `{#tbl:word_comparators}` | Stage 3 per-emission perplexity against every comparator | `98_supplement_audit_trails.md` |
| `{#tbl:word_miswirings}` | Level 3 mis-wirings against the letter-bigram bar | `98_supplement_audit_trails.md` |
| `{#tbl:case_both_defects}` | One Stage 4 threshold, two populations, opposite verdicts | `98_supplement_audit_trails.md` |
| `{#tbl:artifacts}` | Vendored corpus artifacts with checksums | `07_reproducibility.md` |
| `{#tbl:commands}` | The complete reproduction path | `07_reproducibility.md` |
| `{#tbl:case_oracle_paths}` | The derived gate partitioned by oracle path | `04_results.md` |
| `{#tbl:case_path_information}` | Observation content within each oracle-path stratum | `04_results.md` |
| `{#tbl:channel_factorial}` | The channel factorial against the uncoupled anchor | `98_supplement_channel_factorial.md` |
| `{#tbl:channel_proper_loss}` | The channel cells on the receiver's proper loss | `98_supplement_channel_factorial.md` |
| `{#tbl:cluster}` | Cluster-robust reanalysis at the outer sampling unit | `98_supplement_statistics.md` |
| `{#tbl:comparator_strength}` | The strongest member of the comparator class | `98_supplement_audit_trails.md` |
| `{#tbl:confirmatory}` | The four EWT members of the frozen family scored on the held-back split | `98_supplement_audit_trails.md` |
| `{#tbl:derived_bounds}` | The derived paired bounds | `98_supplement_statistics.md` |
| `{#tbl:ewt_document_correction}` | Exact before/after consequences of the authenticated source-document split | `98_supplement_statistics.md` |
| `{#tbl:parameters}` | Historical filtering-component census; connected channel and encoder arrays excluded | `03_methods.md` |
| `{#tbl:wild_cluster}` | Wild cluster bootstrap sensitivity intervals beside the primary cluster intervals | `98_supplement_statistics.md` |

## Citation syntax

```markdown
[@shannon1951]                    parenthetical
@shannon1951 measures…            narrative
[@shannon1951; @brown1992entropy] multiple
[@friedman2024cerebrum, p. 112]   with locator
```

Every key must exist in [`references.bib`](references.bib). Pandoc runs with
`--natbib`, which rewrites `[@key]` to the appropriate LaTeX cite command;
**never** write `\cite{}` in Markdown. `references.bib` omits any field that could
not be confirmed rather than filling it with a plausible value — see the policy
comment at the top of that file.

## Theorem-like blocks

`preamble.md` loads `amsthm` and declares `definition`, `theorem`, and
`proposition`, each numbered within its section. Use the raw environment and let
LaTeX number it:

```markdown
\begin{definition}[Gate soundness]
Let \(t\) be a threshold …
\end{definition}
```

Never hand-number a definition or theorem. Markdown inside such a block is *not*
processed — write plain LaTeX in there, and keep labeled numbered equations
outside the environment so pandoc-crossref can see them.

**Inside these blocks, use the LaTeX parenthetical form for inline math and an
unnumbered `equation*` environment for display math; never use dollar
delimiters.** Pandoc passes the whole environment through verbatim, so the
renderer’s prose Unicode remap is the next thing to read it, and its math-region
detector recognizes the parenthetical form and equation environments — not
dollar delimiters. Dollar-delimited math in a raw
theorem block is therefore treated as prose, and every text-mode math command in
it is rewritten: `\pi` silently became `pi` and `\mid` became `|` in the rendered
PDF while every other check stayed green. Outside these blocks `$…$` is fine,
because pandoc converts it to `\(…\)` before the remap ever runs.

## Token protocol

Every quantitative statement in a numbered section is a double-brace token
resolved from `data/claims.json`. The token is the ledger label uppercased:

```text
ledger label  stage1_letter_perplexity
token         STAGE1_LETTER_PERPLEXITY   (wrapped in double braces in prose)
```

There is no second mapping table, so the ledger cannot drift from the token set.
`src/ailm/manuscript_variables/__init__.py` builds the map, raises `MissingClaimError` for
any required claim the ledger lacks, and generates `<TOKEN>_RUN` and
`<TOKEN>_SOURCE` provenance companions alongside every value.

### Decision locators

The `DECISION_*` literals that resolve decision numbers in prose are the only
decision locators under contract: they live in `data/manuscript_literals.yaml`
and are checked at render time, so a locator the prose relies on mechanically —
as part of a token, a caption sentence bound to a recorded value, or any other
text the checks parse — must use the registered literal, never a hand-typed
numeral. Typed locators in plain prose (`decision 14`, `decisions 22–23`) are
accepted for entries whose numbers are frozen history: the decisions log is
append-only, so those numbers will not move, and the contract machinery does not
need to police them. Keep the register authoritative: when in doubt about what a
number must be, add it to `data/manuscript_literals.yaml` rather than typing it.

### Tokens available

Ledger-derived (one per claim, plus `_RUN` / `_SOURCE` for each):

| Group | Tokens |
|---|---|
| Stage 0 | `PHONEME_COUNT_39`, `STAGE0_CASE_ROLES`, `STAGE0_FUNCTION_ROLES` |
| Stage 1 perplexity | `STAGE1_LETTER_PERPLEXITY`, `STAGE1_UNIGRAM_BASELINE`, `STAGE1_UNIFORM_BASELINE`, `STAGE1_ANTI_INFORMATIVE_PERPLEXITY`, `STAGE1_BIGRAM_FLOOR_PERPLEXITY` |
| Stage 1 information | `STAGE1_LETTER_NATS`, `STAGE1_LETTER_BITS`, `STAGE1_BIGRAM_FLOOR_NATS`, `STAGE1_GAIN_OVER_UNIFORM_NATS`, `STAGE1_MAX_BIGRAM_GAIN_NATS`, `STAGE1_BIGRAM_SIGNAL_CAPTURED_PCT` |
| Stage 1 corpus | `STAGE1_BROWN_BIGRAM_TYPES`, `STAGE1_BROWN_BIGRAM_TOKENS` |
| Stage 2 Part A | `STAGE2_PHONEME_PERPLEXITY`, `STAGE2_PHONEME_UNIGRAM_BASELINE`, `STAGE2_PHONEME_UNIFORM_BASELINE`, `STAGE2_PHONEME_ALL_POSITION_FLOOR_PERPLEXITY`, `STAGE2_PHONEME_ALL_POSITION_MAX_GAIN_NATS`, `STAGE2_PHONEME_FLOOR_PERPLEXITY`, `STAGE2_PHONEME_MAX_GAIN_NATS`, `STAGE2_PHONEME_EVAL_POSITIONS`, `STAGE2_PHONEME_SUBSET_PERPLEXITY` |
| Stage 2 Part B | `STAGE2_G2P_ACCURACY`, `STAGE2_G2P_BASELINE_ACCURACY`, `STAGE2_G2P_MARGIN_PTS`, `STAGE2_G2P_SMOOTHING_CEILING`, `STAGE2_G2P_VITERBI_CEILING`, `STAGE2_G2P_LENGTH_MATCHED_PCT`, `STAGE2_G2P_TRAIN_PAIRS`, `STAGE2_G2P_TEST_PAIRS`, `STAGE2_G2P_EVAL_POSITIONS` |
| Stage 6 registered observed-position score | `STAGE6_PHASE_LOGLOSS`, `STAGE6_UNIGRAM_LOGLOSS`, `STAGE6_LOGLOSS_IMPROVEMENT_NATS`, `STAGE6_SIGN_TEST_POSITIVE`, `STAGE6_SIGN_TEST_NEGATIVE`, `STAGE6_SIGN_TEST_P` |
| Stage 6 document-clustered primary analysis | `CLUSTER_STAGE6_LOGLOSS_EFFECT`, `CLUSTER_STAGE6_LOGLOSS_CI_LOW`, `CLUSTER_STAGE6_LOGLOSS_CI_HIGH`, `CLUSTER_STAGE6_LOGLOSS_P` |
| Stage 6 descriptive live-pymdp filter | `STAGE6_LIVE_FILTER_LOGLOSS`, `STAGE6_LIVE_FILTER_IMPROVEMENT_NATS`, `STAGE6_LIVE_FILTER_SIGN_POSITIVE`, `STAGE6_LIVE_FILTER_SIGN_NEGATIVE`, `STAGE6_LIVE_FILTER_SIGN_P` |
| Formal layer | `FORMAL_THEOREM_COUNT`, `FORMAL_EXAMPLE_COUNT`, `FORMAL_SORRY_FREE` (renders `true`/`false` — write the sentence so it reads correctly with that word substituted), `FORMAL_AXIOMS_BEYOND_CORE`. Used in `07_reproducibility.md`. Also recorded but not currently in prose: `FORMAL_AXIOM_REPORT_VERDICTS`, `FORMAL_NATIVE_DECIDE_USES`, and `FORMAL_AXIOM_NAMES_BEYOND_CORE` (renders as a Python list — name the axioms in backticks in prose instead, as with `STAT_FAMILY_SURVIVORS`) |
| Repository statistics | `STAT_FAMILYWISE_ALPHA`, `STAT_FAMILY_SIZE`, `STAT_BONFERRONI_ALPHA`, `STAT_FAMILY_SURVIVOR_COUNT`, `STAT_FAMILY_SURVIVORS` (renders as a Python list — name survivor labels in backticks in prose instead), `STAT_STAGE6_FAMILYWISE_CONFIRMED` (renders `true`/`false`), plus per-effect `STAT_<EFFECT>_DISCORDANT_N`, `STAT_<EFFECT>_FAVORED_RATIO`, `STAT_<EFFECT>_SHARE_CI_LOW`, `STAT_<EFFECT>_SHARE_CI_HIGH`, `STAT_<EFFECT>_POWER_AT_REFERENCE` for `<EFFECT>` in `STAGE2`, `STAGE4`, `LEXICAL`, `COUPLING_HARM`. The ledger also carries `STAT_<EFFECT>_POWER_CORRECTED`, which is power at the *observed* share; it is recorded for provenance and is **not usable in the manuscript**, because power at the observed share is a monotone transform of the *p*-value. `tests/test_observed_power_is_not_reported.py` enforces this. |

Derived in `manuscript_variables.py` from ledger values (not themselves claims).
A derived token is computed at render time from recorded claims or from a closed
form; it never carries a `_RUN` or `_SOURCE` companion, because no run produced
it. `tests/test_benchmarks.py::test_no_derived_token_shadows_a_recorded_claim`
holds every name below apart from the ledger's own labels, so a derived token can
never quietly overwrite a measured one.

| Group | Tokens |
|---|---|
| Margins and ratios | `STAGE1_UNIGRAM_MARGIN`, `STAGE1_CEILING_SHORTFALL_RATIO`, `STAGE2_PHONEME_UNIGRAM_MARGIN`, `STAGE2_MCNEMAR_TOTAL_DISCORDANT` |
| Detectability for the arms the recorder left without power | `BENCH_LETTER_AZ_POWER_<MODEL>`, `BENCH_LETTER_PROMPT_POWER_<MODEL>`, `BENCH_LETTER_CTX_POWER_<MODEL>` for `<MODEL>` in `GEMMA3`, `QWEN25` |
| Interval convention | `BOUND_CONFIDENCE_PCT`, `BOUND_CORRECTED_CONFIDENCE_PCT`, `EQUIVALENCE_REFERENCE_SHARE`, `EQUIVALENCE_REGION_LOW`, `EQUIVALENCE_REGION_HIGH`, `EQUIVALENCE_MIN_DISCORDANT_PAIRS`, `EQUIVALENCE_MIN_DISCORDANT_PAIRS_CORRECTED` |
| Receiver-intervention registry | `CHANNEL_COUNT`, `CHANNEL_COUNT_WORD`, `CHANNEL_OUTCOME_SUMMARY`, `MODEL_OUTPUT_INTERVENTION_COUNT`, `MODEL_OUTPUT_INTERVENTION_COUNT_WORD`, `PRIVILEGED_ORACLE_INTERVENTION_COUNT`, `PRIVILEGED_ORACLE_INTERVENTION_COUNT_WORD`, `CHANNEL_SOURCE_SUMMARY`, `CHANNEL_SEPARATELY_MEASURED_COUNT`, `CHANNEL_SEPARATELY_MEASURED_COUNT_WORD`, `CHANNEL_CASE_WORD_SEAM_COUNT`, `CHANNEL_CASE_WORD_SEAM_COUNT_WORD` |
| Confirmatory family composition | `CONFIRMATORY_MEMBERS_SCORED`, `CONFIRMATORY_MEMBERS_SCORED_WORD`, `CONFIRMATORY_MEMBERS_UNRESCORED`, `CONFIRMATORY_MEMBERS_UNRESCORED_WORD`, `CONFIRMATORY_MEMBERS_LEXICON_COUNT`, `CONFIRMATORY_MEMBERS_LEXICON_COUNT_WORD`, `CONFIRMATORY_MEMBERS_SPLIT_DIAGNOSTIC_COUNT`, `CONFIRMATORY_MEMBERS_SPLIT_DIAGNOSTIC_COUNT_WORD` (the lexicon and split-diagnostic counts are cross-checked at derivation time to sum with the scored count to `confirmatory_frozen_family_size`) |
| Stage 5 paired gain | `STAGE5_ORACLE_GAIN_NATS_PER_SENTENCE`, `STAGE5_ORACLE_GAIN_SENTENCE_CI_LOW`, `STAGE5_ORACLE_GAIN_SENTENCE_CI_HIGH`, `STAGE5_ORACLE_GAIN_NATS_PER_POSITION`, `STAGE5_ORACLE_GAIN_POSITION_CI_LOW`, `STAGE5_ORACLE_GAIN_POSITION_CI_HIGH`, `STAGE5_ORACLE_GAIN_CI_CONFIDENCE_PCT`, `STAGE5_ORACLE_GAIN_BOOTSTRAP_RESAMPLES`, `STAGE5_ORACLE_GAIN_BOOTSTRAP_SEED`, `STAGE5_ORACLE_GAIN_SENTENCE_INTERVAL_STATUS`, `STAGE5_ORACLE_GAIN_POSITION_INTERVAL_STATUS`, `STAGE5_ORACLE_GAIN_RESOLUTION` |
| Paired classification and sign bounds | `STAGE5_CLASSIFICATION_DELTA_PTS`, `STAGE5_CLASSIFICATION_DELTA_LOW_PTS`, `STAGE5_CLASSIFICATION_DELTA_HIGH_PTS`, `STAGE5_CLASSIFICATION_DISCORDANT`, `STAGE6_SIGN_NET_SHARE`, `STAGE6_SIGN_NET_SHARE_LOW`, `STAGE6_SIGN_NET_SHARE_HIGH`, `STAGE6_SIGN_DISCORDANT`, `STAGE2_G2P_DISCORDANT_NET_SHARE`, `STAGE2_G2P_DISCORDANT_NET_SHARE_LOW`, `STAGE2_G2P_DISCORDANT_NET_SHARE_HIGH`, `LEXICAL_HIGH_ALPHA_DELTA_LOW_PTS`, `LEXICAL_HIGH_ALPHA_DELTA_HIGH_PTS`, `LEXICAL_HIGH_ALPHA_DISCORDANT`, `LEXICAL_HIGH_ALPHA_EQUIVALENT` (renders `1`/`0`) |
| Context-window bounds | `BENCH_LETTER_CTX_DELTA_<MODEL>`, `BENCH_LETTER_CTX_DELTA_LOW_<MODEL>`, `BENCH_LETTER_CTX_DELTA_HIGH_<MODEL>`, `BENCH_LETTER_CTX_DISCORDANT_<MODEL>` |

Scales are not uniform across the bound tokens and must not be assumed: a token
is written on the scale of the recorded quantity it sits beside in prose, so the
Stage 5 and Stage 3 bounds are in percentage points while the Stage 2, Stage 6,
and context-window bounds are fractions. `fig:null_bounds`
declares the scale per arm rather than inferring it, and refuses to draw if a
point estimate recomputed from the arm's own counts contradicts its token.

Metadata and provenance: `META_TITLE`, `META_VERSION`, `META_AUTHORS`,
`META_AUTHOR_COUNT`, `META_KEYWORDS`, `META_CLAIM_COUNT`, `META_BUILD_TIMESTAMP`,
`META_BROWN_COUNTS_KB`, `META_BROWN_COUNTS_SHA256`,
`META_BROWN_COUNTS_SHA256_SHORT`, `META_CMUDICT_COUNTS_KB`,
`META_CMUDICT_COUNTS_SHA256`, `META_CMUDICT_COUNTS_SHA256_SHORT`.

Publication typography is read from `config.yaml`'s `rendering`
fields and validated by `src/ailm/manuscript_variables/literature.py`:
`META_BODY_FONT_PT`, `META_BODY_BASELINE_PT`, `META_TOC_FONT_PT`,
`META_TOC_BASELINE_PT`, `META_MARGIN_TOP_IN`, `META_MARGIN_BOTTOM_IN`,
`META_MARGIN_LEFT_IN`, and `META_MARGIN_RIGHT_IN`. The preamble consumes
an exact generated `article_typography.sty` auxiliary derived from these tokens,
rather than duplicating the settings. The manuscript contract checks that
auxiliary after the template refreshes the authored preamble.

External CEREBRUM locators use paired label and URL producers from
`data/reference_sources.json`: `CEREBRUM_POMDP_SUPPLEMENT_URL`,
`CEREBRUM_GENERATIVE_SUPPLEMENT_URL`, `CEREBRUM_CASE_SUPPLEMENT_URL`, and
`CEREBRUM_ACTIVE_INFERENCE_FIGURE_URL`. The validated deposited-file URL and
physical-page locator determine each target; the displayed supplement or figure
label comes from the same source record. Internal references use their stable
section labels, such as `[@sec:external_benchmarks]`.

### The one exception, and it is narrow

A **quoted declared threshold** is not a measurement and has no ledger entry — it
is a string that was once written down and is now being examined. Those appear in
backticks, verbatim, e.g. `` `>=2 nats above uniform` `` and
`` `perplexity < 500 on PTB-1k` ``. The backticks mark them as quoted
specifications rather than results. Nothing else in a numbered section may contain
a hand-typed numeral.

## Prose conventions

- No "In summary" or "In conclusion" at a section end.
- Active voice for methods.
- Name files explicitly: `src/ailm/levels/letter.py`, never "the letter module".
- One idea per paragraph.
- Report a limitation where it applies, and again in [@sec:limitations] if it
  bounds the result.

## See also

- [`AGENTS.md`](AGENTS.md) — the agent contract for editing this directory
- [`README.md`](README.md) — what each section is and how to render
- [`preamble.md`](preamble.md) — the active LaTeX preamble (keep it fenced)
- `../docs/provenance/sources.md` — corpus provenance and artifact checksums

## The `_ceiling` label suffix is historical

Several recorded labels end in `_ceiling` — `stage2_g2p_smoothing_ceiling` among
them — and the name is older than the distinction the manuscript now draws.
Prose says **reference** or **comparator** where the quantity is a
richer-information model rather than a proven upper bound, and reserves
*ceiling* for a bound that holds under the same information the candidate has.
A smoother sees the whole word; a streaming filter does not, so the smoothing
figure is a reference and not a ceiling on what a same-information model could
reach.

Renaming the labels would rewrite ledger history for a wording change, so they
stay. Read the suffix as "the comparator recorded at this site", and follow the
prose for what kind of object it is.

The hosted stipulated-letter/phone and in-context views use
`BENCH_HOSTED_COHORT_<KEY>_DISPLAY` for retained run keys, not the
`BENCH_MODEL_<KEY>_DISPLAY` local-weight names. Recorded
`BENCH_SYNTH_OPENROUTER_MODEL` / `BENCH_SYNTH_OPENROUTER_PROVIDER` and
`BENCH_CTX_OPENROUTER_MODEL` / `BENCH_CTX_OPENROUTER_PROVIDER` identify the requested
hosted model and provider route. `BENCH_SYNTHPHONE_CONTEXT_RESULT` derives the
phone context outcome from each recorded family-corrected contrast.
