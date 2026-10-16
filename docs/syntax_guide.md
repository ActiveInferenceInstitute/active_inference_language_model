# Syntax Guide

The `{{TOKEN}}` protocol, the `claim:<label>` protocol, and the cross-reference
conventions. Two different mechanisms carry numbers in this project and they are
not interchangeable — using the wrong one in the wrong file is a test failure, not
a style problem.

- [The two mechanisms](#the-two-mechanisms)
- [Rule 1: token = ledger label, uppercased](#rule-1-token--ledger-label-uppercased)
- [Rule 2: a missing claim fails the render](#rule-2-a-missing-claim-fails-the-render)
- [Token reference](#token-reference)
- [Adding a number](#adding-a-number)
- [Detecting unresolved tokens](#detecting-unresolved-tokens)
- [Cross-references (pandoc-crossref)](#cross-references-pandoc-crossref)
- [Figure label registry](#figure-label-registry)
- [Tables, equations, sections, citations](#tables-equations-sections-citations)
- [Code blocks and links](#code-blocks-and-links)

---

## The two mechanisms

| Where | Mechanism | Why |
|---|---|---|
| `docs/manuscript/*.md` | `{{TOKEN}}` | The manuscript is token-substituted at render time, so the rendered number is read out of the ledger on every build |
| `docs/**/*.md`, `README.md`, `AGENTS.md`, `CHANGELOG.md`, `src/**/*.py` docstrings | `claim:<label>` | The sources are not substituted in place; the tag is a verifiable pointer, checked by a test. `scripts/z_render_docs.py` writes a **hydrated mirror** to `output/docs/` where every tag has become its value, so the source stays auditable and the reader still gets numbers |

Both resolve to the same source of truth,
[`../data/claims.json`](../data/claims.json), recorded by the
`scripts/record_stage*_claims.py` runs.

Writing a bare measured numeral in either place is the defect both mechanisms
exist to prevent. In `docs/manuscript/` a literal silently stops tracking the
measurement. In current documentation and Python docstrings,
[`../tests/test_claims_discipline.py`](../tests/test_claims_discipline.py) both
checks that visible tags are backed and rejects high-signal untagged forms such
as percentages, measured units, and statistical assignments. The latter is a
conservative grammar, not a digit ban: naked scalars and spelled-out quantities
still require review.

If you need a number that has no ledger entry: **do not invent one and do not
write a literal.** Record it from the run that produced it, or report it as a
missing claim.

---


## Typing a number instead of tagging it

Tag validity checks the forward direction: a tag naming no ledger entry fails.
The bare-quantity scan and hydration checks add complementary guards against
typing the number without a tag. They are conservative detectors, not proof that
every untagged phrase is a measured claim or that all prose has been reviewed.

That mistake does not look like a mistake. Decision 13's vocabulary table was
typed markdown, and by the time anyone rebuilt it none of its four node counts
survived (against the shipped claim:stage3_vocab_size word vocabulary,
claim:stage3_trie_nodes_v2000 now regenerates as 5,643; the handoff receipt
records the historically measured 5,693 against that table's printed 5,700)
and two of its four
coverage figures did not round from the ledger values recorded in the same rows.
Every `claim:` tag in that file resolved perfectly throughout.

So `tests/test_docs_hydration.py` closes the reverse direction:

| Where | Rule |
|---|---|
| `docs/**`, `README.md`, `AGENTS.md` | A **distinctive** numeral — three or more decimals, or thousands-separated — that exactly equals a recorded claim must be a `claim:` reference |
| `docs/manuscript/**` | Stricter: prose carries no measured decimal at all. Every quantity is a `{{TOKEN}}` |

The documentation rule is narrow on purpose. A ledger of over a thousand claims
contains small values that collide with ordinary prose, and a noisy check gets
suppressed wholesale — which is worse than no check. The cost is that a typed
`0.13` is invisible; the benefit is that every report is real.

Exemptions live in `ailm.docs_hydration.LITERAL_EXEMPTIONS` (whole files) and
`COINCIDENTAL_MATCHES` (single occurrences), and **each states its reason**. A
per-occurrence exemption is checked against the literal on that line, so it
cannot drift onto a different number when the file is edited.

Two files are exempt as documents rather than as oversights.
`docs/research_queue.md` quotes what a probe measured, and the whole value of
that record is that the figure can be compared against the ledger later —
substituting would make every entry agree by construction and delete the
comparison. `CHANGELOG.md` is a dated record, and several entries quote a
superseded value beside the current one deliberately.

## Rule 1: token = ledger label, uppercased

The mapping is mechanical, so there is no second table to drift out of sync with
the ledger:

```
ledger label   stage1_letter_perplexity
token          {{STAGE1_LETTER_PERPLEXITY}}
docs tag       claim:stage1_letter_perplexity
```

`generate_manuscript_variables()` in
[`../src/ailm/manuscript_variables/`](../src/ailm/manuscript_variables/__init__.py)
walks every ledger entry and emits `label.upper()`. It also emits two provenance
companions per label so the manuscript can cite where a number came from without
a second lookup:

| Token | Content |
|---|---|
| `{{<LABEL>}}` | the measured value, formatted for prose |
| `{{<LABEL>_RUN}}` | the `run_id` that produced it |
| `{{<LABEL>_SOURCE}}` | the file or test that produced it |

Value formatting is handled by `_format_value`: integers stay integral and gain
thousands separators above 10,000 (a position count must never render as
`42820.0`), while floats use the shortest representation that round-trips to the
recorded value. This avoids the former six-significant-digit `:g` collapse, under
which distinct ledger values could print identically; small values retain
scientific notation naturally.

The tokens that are *not* ledger labels are derived quantities, generated table
blocks, build metadata, validated decision/search metadata declared in
`data/manuscript_literals.yaml`, bounded search metadata read from
`data/literature_search.json`, deposited-source fields read directly from
`data/reference_sources.json`, and frozen protocol values read from their
executable registries. All enter a duplicate-refusing producer map.

---

## Rule 2: a missing claim fails the render

`REQUIRED_CLAIMS` in
[`../src/ailm/manuscript_variables/`](../src/ailm/manuscript_variables/__init__.py) lists
the labels the manuscript's central claims rest on. If any is absent,
`MissingClaimError` is raised and the render stops.

This is deliberate. The alternatives are worse:

- Emitting a literal `{{TOKEN}}` into the PDF — an unresolved token in a rendered
  document reads as a fabricated result.
- Substituting a plausible default — a fabricated result with no marker at all.

`REQUIRED_CLAIMS` must also list every label the *derived* block reads, not just
what prose references directly. Otherwise a ledger can satisfy `REQUIRED_CLAIMS`
and still fail with a bare `KeyError` deep inside the generator — which is
exactly what [`../tests/test_manuscript_variables.py`](../tests/test_manuscript_variables.py)
caught. `num()` raises `MissingClaimError` naming the omission rather than
`KeyError`, so the failure is diagnosable.

The `docs/` counterpart is
`tests/test_claims_discipline.py::test_no_unbacked_prose_claims`, which scans
`README.md`, `AGENTS.md`, `CHANGELOG.md`, `docs/**/*.md`, and `src/**/*.py` and
fails on any `claim:<label>` with no ledger entry. **Fix it by recording the
measurement, never by deleting the tag.**

---

## Token reference

The authoritative list is the return value of
`generate_manuscript_variables()`. The tables below name every token group; they
carry no values, because a value written here would be a second copy that drifts.
To read the live values:

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root
.venv/bin/python -m json.tool output/data/manuscript_variables.json
```

### Stage 0 — inventory

| Token | Ledger label | Measures |
|---|---|---|
| `{{PHONEME_COUNT_39}}` | `phoneme_count_39` | size of the canonical ARPAbet inventory after comment stripping and stress removal |
| `{{STAGE0_CASE_ROLES}}` | `stage0_case_roles` | linguistic `CaseRole` values |
| `{{STAGE0_FUNCTION_ROLES}}` | `stage0_function_roles` | CEREBRUM `CerebrumFunctionRole` values |

### Stage 1 — letters

| Token | Ledger label | Measures |
|---|---|---|
| `{{STAGE1_LETTER_PERPLEXITY}}` | `stage1_letter_perplexity` | held-out per-letter perplexity of the bigram agent |
| `{{STAGE1_LETTER_NATS}}` | `stage1_letter_nats` | the same, in nats per letter |
| `{{STAGE1_LETTER_BITS}}` | `stage1_letter_bits` | the same, in bits per character |
| `{{STAGE1_UNIGRAM_BASELINE}}` | `stage1_unigram_baseline` | context-free unigram letter-frequency perplexity — **the gate bar** |
| `{{STAGE1_UNIFORM_BASELINE}}` | `stage1_uniform_baseline` | uniform over the alphabet — a scale marker, not a bar |
| `{{STAGE1_BIGRAM_FLOOR_PERPLEXITY}}` | `stage1_bigram_floor_perplexity` | order-1 conditional-entropy floor, as perplexity |
| `{{STAGE1_BIGRAM_FLOOR_NATS}}` | `stage1_bigram_floor_nats` | the same floor in nats |
| `{{STAGE1_MAX_BIGRAM_GAIN_NATS}}` | `stage1_max_bigram_gain_nats` | total gain over uniform available at order 1 — why the target of claim:stage1_declared_gate_nats nats was UNREACHABLE |
| `{{STAGE1_GAIN_OVER_UNIFORM_NATS}}` | `stage1_gain_over_uniform_nats` | gain the agent realizes |
| `{{STAGE1_BIGRAM_SIGNAL_CAPTURED_PCT}}` | `stage1_bigram_signal_captured_pct` | realized gain as a fraction of the available gain |
| `{{STAGE1_ANTI_INFORMATIVE_PERPLEXITY}}` | `stage1_anti_informative_perplexity` | the mis-wired variant (bigram in `A`, identity `B`) |
| `{{STAGE1_BROWN_BIGRAM_TYPES}}` | `stage1_brown_bigram_types` | distinct bigrams in the vendored Brown counts |
| `{{STAGE1_BROWN_BIGRAM_TOKENS}}` | `stage1_brown_bigram_tokens` | bigram tokens behind them |

### Stage 2A — phoneme sequence

| Token | Ledger label | Measures |
|---|---|---|
| `{{STAGE2_PHONEME_PERPLEXITY}}` | `stage2_phoneme_perplexity` | all-positions perplexity, prior reset at each word boundary |
| `{{STAGE2_PHONEME_UNIGRAM_BASELINE}}` | `stage2_phoneme_unigram_baseline` | context-free unigram phone frequencies — **the gate bar** |
| `{{STAGE2_PHONEME_UNIFORM_BASELINE}}` | `stage2_phoneme_uniform_baseline` | uniform over the inventory — scale marker |
| `{{STAGE2_PHONEME_ALL_POSITION_FLOOR_PERPLEXITY}}` | `stage2_phoneme_all_position_floor_perplexity` | exact unsmoothed start-plus-transition MLE bound over the full dictionary — **the gate bound** |
| `{{STAGE2_PHONEME_ALL_POSITION_MAX_GAIN_NATS}}` | `stage2_phoneme_all_position_max_gain_nats` | exact all-position gain over uniform available at this order |
| `{{STAGE2_PHONEME_FLOOR_PERPLEXITY}}` | `stage2_phoneme_floor_perplexity` | transitions-only diagnostic, not the all-position gate bound |
| `{{STAGE2_PHONEME_MAX_GAIN_NATS}}` | `stage2_phoneme_max_gain_nats` | optimistic transitions-only gain over uniform |
| `{{STAGE2_PHONEME_EVAL_POSITIONS}}` | `stage2_phoneme_eval_positions` | positions in the full-dictionary closed-form measurement |
| `{{STAGE2_PHONEME_SUBSET_PERPLEXITY}}` | `stage2_phoneme_subset_perplexity` | the same metric on the length-matched subset — the subsample-bias evidence |

### Stage 2B — grapheme-to-phoneme

| Token | Ledger label | Measures |
|---|---|---|
| `{{STAGE2_G2P_ACCURACY}}` | `stage2_g2p_accuracy` | full-set NumPy forward-filtering position accuracy, licensed by bounded zero disagreement with the pymdp Agent |
| `{{STAGE2_G2P_BASELINE_ACCURACY}}` | `stage2_g2p_baseline_accuracy` | context-free per-letter `argmax` — **the gate bar** |
| `{{STAGE2_G2P_MARGIN_PTS}}` | `stage2_g2p_margin_pts` | the margin, in percentage points |
| `{{STAGE2_G2P_EVAL_POSITIONS}}` | `stage2_g2p_eval_positions` | positions in the full held-out set |
| `{{STAGE2_G2P_MCNEMAR_P}}` | `stage2_g2p_mcnemar_p` | paired McNemar p-value on the full set |
| `{{STAGE2_G2P_MCNEMAR_B}}` | `stage2_g2p_mcnemar_b` | positions where only the agent is right |
| `{{STAGE2_G2P_MCNEMAR_C}}` | `stage2_g2p_mcnemar_c` | positions where only the baseline is right |
| `{{STAGE2_G2P_SUBSAMPLE_POSITIONS}}` | `stage2_g2p_subsample_positions` | positions in the representative stride |
| `{{STAGE2_G2P_SUBSAMPLE_MCNEMAR_P}}` | `stage2_g2p_subsample_mcnemar_p` | the same test on that stride — the underpowered-subsample evidence |
| `{{STAGE2_G2P_SMOOTHING_CEILING}}` | `stage2_g2p_smoothing_ceiling` | forward–backward ceiling (whole-word context) |
| `{{STAGE2_G2P_VITERBI_CEILING}}` | `stage2_g2p_viterbi_ceiling` | Viterbi MAP-path ceiling |
| `{{STAGE2_G2P_TRAIN_PAIRS}}` | `stage2_g2p_train_pairs` | length-matched training pairs |
| `{{STAGE2_G2P_TEST_PAIRS}}` | `stage2_g2p_test_pairs` | length-matched held-out pairs |
| `{{STAGE2_G2P_LENGTH_MATCHED_PCT}}` | `stage2_g2p_length_matched_pct` | share of CMUdict that survives the 1:1 restriction |
| `{{STAGE2_DECODER_DISAGREEMENTS}}` | `stage2_decoder_disagreements` | pymdp-vs-oracle position disagreements — must be zero for the full-set statistics to describe the agent |

### Derived tokens

Computed in `generate_manuscript_variables()` from ledger entries, because the
ledger stores the parts rather than the difference. **They have no `claim:`
form.** Writing one — a lowercased tag for a derived token — puts an unbacked
label in `docs/` prose and turns `test_no_unbacked_prose_claims` red, because no
such entry exists in the ledger. In `docs/` cite the two component claims and
name the derived token as a token.

| Token | Derivation |
|---|---|
| `{{STAGE1_UNIGRAM_MARGIN}}` | `stage1_unigram_baseline − stage1_letter_perplexity` |
| `{{STAGE2_PHONEME_UNIGRAM_MARGIN}}` | `stage2_phoneme_unigram_baseline − stage2_phoneme_perplexity` |
| `{{STAGE1_CEILING_SHORTFALL_RATIO}}` | `2.0 / stage1_max_bigram_gain_nats` — how far the original target of claim:stage1_declared_gate_nats nats sat beyond the ceiling |
| `{{STAGE2_MCNEMAR_TOTAL_DISCORDANT}}` | `stage2_g2p_mcnemar_b + stage2_g2p_mcnemar_c` |

### Build and provenance metadata

| Token | Content |
|---|---|
| `{{META_TITLE}}`, `{{META_VERSION}}` | from `docs/manuscript/config.yaml` `paper:` |
| `{{META_AUTHORS}}`, `{{META_AUTHOR_COUNT}}` | from `config.yaml` `authors:` |
| `{{META_KEYWORDS}}` | from `config.yaml` `keywords:` |
| `{{META_BUILD_TIMESTAMP}}` | UTC build stamp, honouring explicit `SOURCE_DATE_EPOCH` and otherwise using the epoch of the last audited commit that touched an input outside `output/` |
| `{{META_CLAIM_COUNT}}` | number of entries in the ledger |
| `{{META_BROWN_COUNTS_SHA256}}` / `_SHA256_SHORT` / `_KB` | `vendor/brown_counts.json` |
| `{{META_CMUDICT_COUNTS_SHA256}}` / `_SHA256_SHORT` / `_KB` | `vendor/cmudict_counts.json` |

The checksum tokens exist so a reader can verify they have the same inputs,
rather than trusting that "the Brown corpus" and "CMUdict" mean the same bytes
everywhere. They are enforced against the artifacts on disk by
[`../tests/test_provenance_checksums.py`](../tests/test_provenance_checksums.py).

---

### Connected results and nonfinite values

Connected measurements use the `connected_gum_*` ledger namespace and its
source-bound report. A manuscript token is not independent evidence that this
report is current: the connected figure reader checks its complete canonical
execution binding and exact frozen population. Preserve the difference between
character loss, finite in-vocabulary word diagnostics and full-stream word loss.
Use the recorded OOV/loss-status fields; do not manufacture a finite replacement
for an infinite closed-vocabulary result. The [data statement](reference/data_statement.md)
explains the population and normalization.

## Adding a number

1. **Measure it in a recorder script.** Extend the relevant
   `scripts/record_stage*_claims.py`; never compute a prose number by hand.

   ```python
   from ailm.claims import load_ledger

   ledger = load_ledger(PROJECT_ROOT)
   ledger.record(
       "stage3_word_perplexity",
       measured_value,
       run_id="stage3_word_gate",
       source="scripts/record_stage3_claims.py",
   )
   ```

   Both `run_id` and `source` are mandatory —
   `test_every_ledger_entry_has_provenance` fails on an entry that carries a
   value with no way to re-derive it.

2. **Run the recorder**, then regenerate the token map:

   ```bash
   .venv/bin/python scripts/record_stage3_claims.py
   .venv/bin/python scripts/z_generate_manuscript_variables.py --print-only
   ```

3. **Add it to `REQUIRED_CLAIMS`** if the manuscript's central claims depend on
   it, or if the derived block reads it.

4. **Reference it.** `{{STAGE3_WORD_PERPLEXITY}}` in `docs/manuscript/`, and the same
   label lowercased behind a `claim:` prefix in `docs/`.

   Note that a document *about* the protocol cannot spell an example tag out in
   full: `scan_prose` cannot tell an illustration from an assertion, so an
   invented label written as a complete tag turns
   `test_no_unbacked_prose_claims` red. Write the prefix and the label
   separately, or use the unresolvable `claim:` + `<label>` placeholder form.

5. **Label naming.** `stage<N>_<subject>_<what>`, lowercase with underscores, with
   a unit suffix where the unit is not obvious: `_nats`, `_bits`, `_pts`, `_pct`,
   `_positions`, `_perplexity`, `_accuracy`. A label whose unit is ambiguous
   produces prose whose unit is ambiguous.

---

## Detecting unresolved tokens

An unresolved `{{TOKEN}}` in a rendered document is a fabricated result. Check
before publishing anything:

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root
! rg -n -F '{{' output/manuscript/
```

Check the `docs/` side with the real gate rather than by eye:

```bash
.venv/bin/python -m pytest tests/test_claims_discipline.py -q
```

See [`troubleshooting.md`](troubleshooting.md#a-literal-token-appears-in-the-rendered-output)
when either check fires.

---

## Cross-references (pandoc-crossref)

Inside `docs/manuscript/`, use pandoc-crossref bracket syntax. **Never** write raw
LaTeX `\ref{}` / `\eqref{}` in Markdown source: it renders literally in the HTML
and slide outputs and bypasses auto-numbering. Never hardcode a figure, table, or
equation number.

- **BAD** — `See Figure 3.`
- **BAD** — `See Figure \ref{fig:wiring_comparison}.`
- **GOOD** — `See [@fig:wiring_comparison].`
- **GOOD** (narrative) — `@fig:wiring_comparison shows the two orientations.`

---

## Figure label registry

This table is the reader-facing projection of
[`../output/figures/figure_registry.json`](../output/figures/figure_registry.json),
generated by [`../src/ailm/figures/`](../src/ailm/figures/) and orchestrated by
[`../scripts/generate_figures.py`](../scripts/generate_figures.py).
`tests/test_documentation.py` checks the filename, label, and generator module
against that registry exactly. Use these labels; do not invent new ones for an
existing figure.

| Anchor | PNG in `output/figures/` | Builder | Shows |
|---|---|---|---|
| `{#fig:aligned_g2p_arms}` | `aligned_g2p.png` | `ailm.figures.aligned_g2p` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:belief_dynamics}` | `belief_dynamics.png` | `ailm.figures.belief_dynamics` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:calibration_floor}` | `calibration_floor.png` | `ailm.figures.calibration_floor` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:case_both_defects}` | `case_both_defects.png` | `ailm.figures.composition_results` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:case_matrices}` | `case_matrices.png` | `ailm.figures.interpretability` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:case_word_sweep}` | `case_word_sweep.png` | `ailm.figures.case_word_sweep` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:channel_comparison}` | `channel_comparison.png` | `ailm.figures.composition_results` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:clause_null}` | `clause_null.png` | `ailm.figures.levels56` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:complexity_memory}` | `complexity_memory.png` | `ailm.figures.complexity` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:complexity_ops}` | `complexity_ops.png` | `ailm.figures.complexity` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:composition}` | `composition.png` | `ailm.figures.composition` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:connected_core}` | `connected_core.png` | `ailm.figures.connected` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:dashboard_observation}` | `dashboard_observation.png` | `ailm.figures.dashboard_observation` | Warm local dashboard capture separates input analysis from the L3-to-L1g lexical policy reply; capture provenance and scope are recorded in the figure caption |
| `{#fig:external_benchmarks}` | `external_benchmarks.png` | `ailm.figures.benchmarks` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:g2p_accuracy}` | `g2p_accuracy.png` | `ailm.figures.results` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:gate_taxonomy}` | `gate_taxonomy.png` | `ailm.figures.gates` | VACUOUS / UNREACHABLE / ABOVE_REFERENCE / UNAUDITED / BRACKETED_BOUND / BRACKETED_REFERENCE |
| `{#fig:generation_samples}` | `generation_samples.png` | `ailm.figures.samples` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:generation_sweep}` | `generation_sweep.png` | `ailm.figures.composition_results` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:gnn_pipeline}` | `gnn_pipeline.png` | `ailm.figures.composition` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:hierarchy}` | `hierarchy.png` | `ailm.figures.architecture` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:incontext_sweep}` | `incontext_sweep.png` | `ailm.figures.incontext_sweep` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:inference_cycle}` | `inference_cycle.png` | `ailm.figures.architecture` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:inference_evidence}` | `inference_evidence.png` | `ailm.figures.inference_evidence` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:information_ceiling}` | `information_ceiling.png` | `ailm.figures.results` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:learning_agreement}` | `learning_agreement.png` | `ailm.figures.learning` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:letter_transition}` | `letter_transition.png` | `ailm.figures.matrices` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:confirmatory_replication}` | `confirmatory_replication.png` | `ailm.figures.confirmatory` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:multiplicity_forest}` | `multiplicity_forest.png` | `ailm.figures.multiplicity` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:narrative_channel}` | `narrative_channel.png` | `ailm.figures.levels56` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:null_bounds}` | `null_bounds.png` | `ailm.figures.null_bounds` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:perplexity_ladder}` | `perplexity_ladder.png` | `ailm.figures.results` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:phoneme_matrices}` | `phoneme_matrices.png` | `ailm.figures.matrices` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:scaling_ladder}` | `scaling_ladder.png` | `ailm.figures.scaling_ladder` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:subsample_bias}` | `subsample_bias.png` | `ailm.figures.gates` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:trie_narrowing}` | `trie_narrowing.png` | `ailm.figures.interpretability` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:wiring_comparison}` | `wiring_comparison.png` | `ailm.figures.matrices` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:word_matrices}` | `word_matrices.png` | `ailm.figures.interpretability` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:level_gate_summary}` | `level_gate_summary.png` | `ailm.figures.level_gate_summary` | Caption, alt text, and falsifier are published in the figure registry |
| `{#fig:word_miswirings}` | `word_miswirings.png` | `ailm.figures.composition_results` | Caption, alt text, and falsifier are published in the figure registry |

Image syntax in the manuscript, with the anchor attached to the image and a
width to prevent float-too-large warnings:

```markdown
![Correct wiring places the learned bigram matrix in the transition; the
anti-informative variant places it in the emission.](../../output/figures/wiring_comparison.png){#fig:wiring_comparison width=90%}
```

Captions must be self-contained: they appear in the PDF and beneath the image
in HTML. They are **not** the image alternative. Every figure also needs a
one-sentence entry in `data/figure_alt_text.yaml`, keyed by label, which the
registry publishes and the renderer sets as `alt`. It must be free of
`{{TOKEN}}`, citation keys and math, and `tests/test_figure_alt_text.py`
enforces both that and the requirement that every referenced figure has one.
A caption that says "the results" is not a caption.

To list or regenerate:

```bash
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py --list
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py --only wiring_comparison
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py
```

---

## Tables, equations, sections, citations

**Tables** — caption below the table, anchor inside the caption line:

```markdown
| Comparator | Perplexity |
|---|---|
| Context-free unigram | {{STAGE1_UNIGRAM_BASELINE}} |
| Bigram agent | {{STAGE1_LETTER_PERPLEXITY}} |

: Letter-level comparator ladder. {#tbl:stage1_ladder}
```

Reference with `[@tbl:stage1_ladder]`. Do not write a `Table:` prefix — pandoc
infers the type from placement.

**Equations** — attach the anchor after the closing `$$`. Never use a raw
`\begin{equation}...\label{...}` block in Markdown; pandoc-crossref will not pick
up the LaTeX label and the equation becomes unreferenceable from prose.

```markdown
$$
H(X_t \mid X_{t-1}) = -\sum_j P(x_{t-1}{=}j) \sum_i P(x_t{=}i \mid x_{t-1}{=}j) \log P(x_t{=}i \mid x_{t-1}{=}j)
$$ {#eq:conditional_entropy}
```

Reference with `[@eq:conditional_entropy]`.

**Sections** — give every manuscript H1 a `{#sec:<name>}` label so cross-section
references survive reordering: `[@sec:methods]`.

**Citations** — `[@key]`, multiple as `[@key_one; @key_two]`, narrative as
`@key showed that…`, with a locator as `[@key, pp. 42-45]`. Every key must exist
in the manuscript bibliography; the keys above are placeholders, not live ones.
Never write raw `\cite{}`.

Both papers are the author's own and both are described in
[`provenance/sources.md`](provenance/sources.md) — including the note that the
truncated 38-page CEREBRUM `full_text.md` export omits Supplements 8–15 and must
not be cited for engineering content.

---

## Code blocks and links

Tag every code block with its language — pandoc syntax highlighting depends on
it:

````markdown
```python
from ailm.levels.phoneme import build_g2p_agent
```

```bash
.venv/bin/python scripts/audit_gates.py
```
````

Link text must be informative, and paths must be real:

- **BAD** — "See [this file](../src/ailm/levels/phoneme.py) for details."
- **GOOD** — "The emission is built by [build_g2p_emission](../src/ailm/levels/phoneme.py)."

(In real prose the function name inside the link text carries backticks; it is
written plainly above so this example does not nest code spans.)

Links are relative to the Markdown file containing them; a link that does not resolve is a
broken claim about the repository's shape, so check it rather than assuming it.
Markdown, never HTML, for anything markdown supports.

---

## Cross-references

- [`style_guide.md`](method/style_guide.md) — the prose and honesty rules that sit on top of this syntax
- [`gate_design.md`](gate_design.md) — where the recorded numbers come from
- [`testing_philosophy.md`](testing_philosophy.md) — the tests that enforce both protocols
- [`../src/ailm/manuscript_variables/`](../src/ailm/manuscript_variables/__init__.py) — the authoritative token generator
- [`../src/ailm/claims.py`](../src/ailm/claims.py) — `ClaimsLedger`, `scan_prose`
