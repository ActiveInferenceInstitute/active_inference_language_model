# tests/ — the AILM test suite

Real pymdp, real corpora, real matplotlib, real files. No mocks anywhere. The default
suite is offline; the few tests that deliberately query a user's local Ollama process
require explicit opt-in. The suite's job is not only to check that the code works —
it is to make sure each *gate* is capable of failing, because this project has shipped
a gate that could not.

## Layout

The discovered `test_*.py` files are the authoritative inventory; no copied
file count is kept here. The tables below name every stage gate and the files a
newcomer needs to find first.

### Stage gates (the meta-tests)

| File | Gates |
| --- | --- |
| `test_stage0_gate.py` | is Stage 0 done? ledger operational, harness live, pymdp probes pass, 39-phone invariant, `cases` self-consistent |
| `test_stage1_gate.py` | letter agent vs uniform (declared), vs the context-free unigram bar (load-bearing), plus a live negative control against the anti-informative wiring |
| `test_stage2_gate.py` | Part A phoneme perplexity vs the unigram bar; Part B G2P vs the per-letter argmax bar with a paired McNemar test; independent re-derivation of the emission |
| `test_stage3_gate.py` | word level vs a letter-bigram model over the identical alphabet, tokens, and positions |
| `test_stage4_gate.py` | the case level, and why the declared "≥60% vs the UD 8-case oracle" fails in both directions depending on the population scored |
| `test_stage5_gate.py` | the retained case-only clause ablation's non-passing gate, fixed-seed paired whole-document cluster-bootstrap gain intervals, positive/negative interval controls, and failure paths; the separate punctuation-type-oracle→L3 word-prior intervention has its own test file and does not use an L5 posterior |
| `test_stage5_widened.py` | the canonical clause gate on the widened observation alphabet, its resolved paired gain, its decoder-equivalence and swapped-blocks negative controls, and the cross-arm guard that the case-only ablation's record survives the promotion unchanged |
| `test_stage6_gate.py` | the positive registered point estimate, non-confirmatory document-clustered primary analysis, separate descriptive live-Agent score, and independent L6→L5 `D`-prior intervention |
| `test_speech_boundary.py` | the interface contract at the speech boundary, which is what this project owns rather than Whisper's or a TTS engine's accuracy |

### Unit and property suites

| File | Subject |
| --- | --- |
| `test_letter.py` | `ailm.levels.letter` — matrix construction, indexing, perplexity |
| `test_phoneme.py` | `ailm.levels.phoneme` — both agents, orientation, oracle agreement |
| `test_phoneme_full_corpus.py` | the closed-form full-dictionary Part A identity |
| `test_oracle.py` | `ailm.verification.oracle` — every expectation derived on paper |
| `test_oracle_independence.py` | the oracle is independent of the agent it audits, checked by import |
| `test_cmudict.py`, `test_cmudict_load.py` | phoneme inventory, comment stripping, file I/O |
| `test_ud.py` | CoNLL-U parsing and the derived case oracle |
| `test_cases.py` | `CaseRole` / `CerebrumFunctionRole`, and that no symbol denotes both |
| `test_cross_level_alphabets.py` | collected adjacent-level alphabet contracts, including exact canonical ARPAbet equality |
| `test_gates.py` | the shared gate contract — `gate_passes`, `gate_margin`, `classify_gate`, `evaluate_gate` — that `audit_gates.py`, the figures, and the Lean layer all bind |
| `test_coupling.py`, `test_lexical_channel.py`, `test_clause_word_channel.py`, `test_generation.py` | the four cross-level channels bearing on the novelty criterion |
| `test_learning.py`, `test_online_learning.py`, `test_g2p_learning.py` | parameter learning against the shipped counting path, non-stationary online adaptation, and G2P learning |
| `test_statistics.py`, `test_complexity.py` | exact statistical contracts and the multiplicity family; exact operation- and byte-counts against executed counts |
| `test_benchmarks.py`, `test_synthetic_benchmark.py` | the external benchmark harness and the stipulated-language exact-artifact-overlap probe |
| `test_aligned_g2p_recorder.py` | the P2 comparator-arm recorder — live A1/A2/A3 refusal gates on the frozen artifact and a happy path that writes exactly its own namespace |
| `test_aligned_g2p_probe_recorder.py` | the P2 zero-shot probe-arm recorder, replayed through the shipped parser/scorer over real captured probe responses (no mocks, no live server) |
| `test_case_word_channel.py` | the case-word channel — model-sender message math, the L3 inlet, and reset semantics, each control numerically identical to its Bayes expression |
| `test_case_word_influence_recorder.py` | the decision-121 influence-sweep recorder's gate chain over small fixture channels or the frozen artifacts |
| `test_corpus_case_word.py` | the case-word bridge artifact loader — every structural refusal the account protocol P3 froze |
| `test_fitted_probe_foundation.py` | the fitted-probe data/identity foundation (identity, census, dataset builders, receipts) and its CLI as a subprocess |
| `test_config.py` | every knob in `ailm.config` is consumed, not merely accepted |
| `test_pymdp_smoke.py` | the pymdp API surface probes |
| `test_control_harness.py` | the harness itself — if it cannot detect a broken model, every gate is vacuous |
| `test_control_meta_validation.py` | each registered negative control contains its load-bearing construction and assertion, executes through pytest without skip, and rejects a planted no-op control |

### Discipline suites

| File | What it enforces |
| --- | --- |
| `test_claims_ledger.py` | `ClaimsLedger` mechanics against temp ledgers |
| `test_claims_discipline.py` | **every `claim:<label>` in real repo prose is backed** |
| `test_claims_single_writer.py` | one writer per label — two code paths recording the same label from different measurements make the stored value a race |
| `test_claim_recorder_orchestration.py` | the dynamically discovered recorder inventory is partitioned exactly once across core/network/formal/aggregate phases, network validation runs last, and a real failing subprocess stops later writers |
| `test_checkout_completeness.py` | a partial checkout fails loudly instead of skipping its gates into a green suite |
| `test_manuscript_variables.py` | the `{{TOKEN}}` map, including the missing-claim failure path |
| `test_manuscript_tokens.py` | the production token grammar, malformed/unknown/empty controls, hard-coded variable controls, exact source-to-resolved-tree contract, and the rule that no unresolved `{{TOKEN}}` may ship |
| `test_manuscript_crossrefs.py` | every `[@sec:x]`/`[@fig:x]`/`[@tbl:x]` resolves, and every float is referenced |
| `test_notation_registry.py` | one meaning per registered symbol — the predictive prior is `\rho_t` so `\pi` can mean policy and nothing else |
| `test_caption_standard.py` | every caption states what would falsify its panel, as a ratchet |
| `test_provenance_checksums.py` | documented sha256s match the artifacts, and the generators are deterministic |
| `test_figures_architecture.py`, `test_figures_results.py`, `test_figures_composition.py`, `test_figures_multiplicity.py` | figures exist, are 200 dpi PNGs, are byte-identical on regeneration, are legible, and agree with the token map |
| `test_figure_layout.py` | no overlapping text, nothing clipped — the part `assert_legible` cannot see |
| `test_figure_provenance.py` | the v3 receipt binds the pinned real numeric stack, canonical token map, complete AILM code inventory, bounded data/vendor/GNN inputs, shared renderer/writer, and PNG bytes; real FIFO, descriptor-swap, dependency-mutation, stale-input, symlink, mutated-PNG, and generator-failure controls prove selective preflight and receipt-last publication fail closed |
| `test_gnn_conformance.py`, `test_gnn_registry.py`, `test_spec_agent.py` | each level's GNN spec declares the semantic role of every matrix, the registry is a grammar rather than a hand-kept index, and a spec-built agent is bit-identical to the builder-built one |
| `test_formal_isomorphism.py` | the generated Lean ledger regenerates byte-identically and matches the recorded values |
| `test_worked_example.py` | the interpretability walk-through in the architecture section names the transition column the model actually read |
| `test_no_self_offset_strict_zip.py` | `strict=True` is never applied to a self-offset zip, where it raises on every non-empty input |
| `test_release_orchestration.py` | `verify_release.py`'s profiles, deterministic build environment, complete Git-state fingerprint, clean-input policy, and read-only list mode |
| `test_replay_provenance.py` | deterministic bindings for every network-excluded recorder, with real current-schema cache records and planted stale-source, wrong-configuration, wrong-model, missing-call, legacy-only, namespace-drift, and path-tampering failures |
| `test_reproduction_documentation.py` | complete recorder/status/generator ordering before every manuscript consumer, correct negation of Lean's must-fail control, clean-suite prerequisites, dynamic skip guidance, independent-unit split guidance, and planted status-masking/order failures |

`conftest.py` provides offline corpus and ledger fixtures, real pymdp agents,
a loopback HTTP server, and shared `gate_audit_rows`. The audit fixture requires
both committed gate reports and rejects missing, empty, partial, duplicate, or
unregistered rows before a cover, hierarchy, or finding check consumes them.
Regenerate those reports with
`.venv/bin/python scripts/audit_gates.py --full --json`.

## Running

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root

# Everything, with the coverage gate
.venv/bin/python scripts/z_generate_manuscript_variables.py --print-only
.venv/bin/python scripts/export_web_model.py
.venv/bin/python scripts/build_web_ailm.py
.venv/bin/python -m pytest tests/ --cov=ailm --cov-report=term-missing

# Fast, no coverage
.venv/bin/python -m pytest tests/ -q --no-cov

# One file, one test
.venv/bin/python -m pytest tests/test_stage2_gate.py -q --no-cov
.venv/bin/python -m pytest tests/test_letter.py::TestComputePerplexity -v --no-cov

# Deliberately reach the user's local Ollama process. Availability is checked
# inside the opted-in test body, never during collection. Recorder subprocesses
# put their call stores beside their tmp_path ledger rather than in cache/.
AILM_RUN_LIVE_OLLAMA_TESTS=1 \
  .venv/bin/python -m pytest tests/test_benchmarks.py tests/test_synthetic_benchmark.py -v
```

**Always `.venv/bin/python -m pytest`.** A bare `pytest` or `uv run pytest` in this
sidecar can resolve to a system interpreter with a different Python and a different
pymdp, which silently changes the numbers. `pyproject.toml` supplies
`pythonpath = ["src"]` and a 300 s per-test timeout.

Full-hierarchy parameter checks carry a scoped startup allowance, and figure
layout checks share a bounded construction fixture. Array controls and other
fast checks retain their own smaller budgets. Import or filesystem stalls need
process-level diagnosis alongside the pytest output; a live process without a
terminal test summary is pending, and a timeout is a failure rather than a skip.
The documentation round-trip test runs the real renderer CLI with `-I -S` so
every publication and refusal operation also checks its standard-library boundary.

### Parallel runs

`pytest-xdist` is in the dev group, so the development loop can shard the suite:

```bash
# Use -n 4 instead of -n auto when other pytest fleets share the machine or
# drive (a 1-minute load of 5-13 is normal here).
.venv/bin/python -m pytest tests/ -n auto -q --no-cov
```

The default stays serial. The `quick` profile in `scripts/verify_release.py`
runs its test step on up to four workers, half the host's cores (no coverage is
taken there), while the
`full` profile is serial by design — its receipt parser expects exactly one
coverage TOTAL row. Marker exclusions are unchanged under `-n`
(`-m "not recorder_replay and not raw_corpus_regeneration and not requires_cuda12 and not publication_context and not template_context"`),
and with `--cov` the workers combine coverage automatically, with the pyproject
floor still applied.

## The three invariants a newcomer would otherwise violate

### 1. No mocks — and here that is a correctness requirement, not a style rule

No `MagicMock`, no `unittest.mock`, no `mocker.patch`, no monkeypatched dependency.
Use the real thing: real pymdp agents, the vendored corpus JSON, `tmp_path` for files,
inline CoNLL-U strings and the `tiny_cmudict` fixture for corpora. Every test is
offline.

`test_pymdp_smoke.py` shows why the rule bites: a mock has *every* attribute, so
`hasattr(agent, "update_empirical_prior")` would pass against a pymdp that lacks it,
and the absence check `not hasattr(agent, "update_A")` could never be demonstrated at
all.

### 2. Every gate carries a negative control, and the control must be live

A test that asserts only "the good model beats the threshold" proves nothing about the
threshold. Each gate additionally runs the *broken* model and requires it to fail.
`ailm.control.ControlHarness` formalizes this; `GateVacuousError` means the metric
cannot discriminate and the stage is **blocked**.

The controls that are load-bearing right now, and must not be deleted:

- `test_cmudict.py::TestNegativeControl` — three tests proving the 39-phone gate can
  fail: `test_comment_token_not_in_canonical`,
  `test_build_inventory_detects_comment_tokens` (feeds `;;;` in as a phone token and
  requires it to appear in the inventory), and
  `test_assert_full_inventory_catches_comment_token`. Without comment stripping the
  real inventory reads 54.
- `test_stage1_gate.py` — a uniform model must fail the bar, *and* the
  anti-informative wiring must be caught. The declared `< 28.0` threshold admits the
  mis-wiring at claim:stage1_anti_informative_perplexity, so uniform alone is not
  enough.
- `test_figures_architecture.py::TestAssertLegible::test_tiny_font_figure_is_rejected`
  — `assert_legible` runs on every figure the project produces, so if it could not
  fail, every other legibility assertion in the suite would be vacuously true.
- `test_figure_layout.py` — positive controls for the overlap and clipping checks. A
  geometric assertion that never fires is worse than none, because it licenses the
  belief that layout was verified. Plus
  `test_discovery_covers_every_shipped_figure`, which binds the discovered builder
  count to the PNGs on disk: the prefix heuristic once omitted `generate_` and
  silently swept only five of nine figures while reading as complete.
- `test_oracle.py` — the transposition control. An identity transition is
  column-stochastic either way, so an identity-only test cannot detect a transpose.
- `test_worked_example.py` — the manuscript's one concrete demonstration of its
  central interpretability claim, re-derived from the vendored counts. Beyond
  checking the recorded values it asserts the example still makes its *point*:
  the `q` column must stay near-deterministic and the `t` column must not. Both
  can drift while every printed number remains correct, which is the failure a
  value-only check cannot see.
- `test_manuscript_crossrefs.py` — every cross-reference resolves, every table and
  figure is pointed at by a sentence, and no reference carries a hand-written
  `Figure` prefix that pandoc-crossref would double. It found four orphaned
  tables and one doubled prefix when it was written.
- `test_cover_honesty.py` — cover structure, gate qualifiers, layout, and published
  freshness. Real missing and tampered files prove that `generate_cover.py --check`
  fails without silently repairing its input. Regenerate the cover before rendering;
  run template Stages 03 then 04 before the rendered-provenance cross-reference test.
- `test_notation_registry.py` — `$\pi$` is the policy and never the predictive
  prior. `[@tbl:notation]` promised exactly this and one proposition broke it.

### 3. `test_claims_discipline.py` scans **real** prose, and must keep its positive control

`ClaimsLedger.scan_prose` existed from Stage 0 and was exercised only against
`tmp_path` fixtures. Every test passed while `claim:phoneme_count_39` sat unbacked in
README.md, AGENTS.md, and `docs/provenance/sources.md` for an entire stage. **A gate
that runs only on synthetic input constrains nothing about the shipped artifact.**

That file points the scanner at the real tree — all root Markdown, every
`docs/**/*.md`, GNN and formal Markdown, and every `src/**/*.py`. Tag validity is
checked once per source file with a shared read-only ledger. Each file keeps the
ordinary per-test timeout; a slow aggregate filesystem walk cannot consume one
test's budget for all sources, and a failed node names its source. The complete
inventory remains covered, with the controls that keep the scan non-vacuous:

- `test_prose_file_set_is_non_empty` — an empty glob would pass every check below it.
- `test_prose_set_includes_the_top_level_docs` — the reader-facing files specifically.
- `test_scanner_detects_an_injected_unbacked_claim` — the **positive control**. Plants
  a fake tag and requires the scanner to flag it. Delete this and
  `assert unbacked == []` also passes when `_prose_files()` returns nothing or the
  regex stops matching.
- `test_scanner_accepts_a_backed_claim` — the negative control.

`tests/` is deliberately **excluded** from the scanned set: test files carry
illustrative `claim:` tags (`alpha_claim` and friends) that are fixtures, not assertions about
the project.

When this gate fails, fix it by recording the measurement with the matching
`scripts/record_stage*_claims.py`. **Never by deleting the tag from prose.**

## Numbers in tests

Every numeric assertion is either an exact closed form or a value re-derived inside
the test from the same inputs. Nothing is a literal copied from a previous run — the
unigram comparator is rebuilt from `phone_unigrams` in-test rather than pinned, and
the trained bigram figure is never asserted against the transitions-only unsmoothed
floor claim:stage2_phoneme_floor_perplexity, because comparing an all-positions
metric to that transitions-only diagnostic is invalid. The gate instead consumes
claim:stage2_phoneme_all_position_floor_perplexity over the identical denominator;
the add-1-smoothed model must sit at or above that unsmoothed optimum.

Float32 is JAX's default, so exact anchors land at ~2e-7 relative. Compare with a
tolerance, never for equality.

## More information

[`AGENTS.md`](AGENTS.md) for what an agent may change here.

The three whole-figure geometry sweeps share a module-scoped parametrized
fixture in `test_figure_layout.py`: each figure stem is constructed once for
layout, legend, and rule-crossing assertions, then closed. The assertions
remain independent. Discovery resolves every save-writing `generate_*` wrapper
to its in-memory `build_*_figure` sibling before anything is called, so the
sweep never runs a publication writer, and the completeness check compares the
shipped PNG stems against the discovered stems as sets — a missing figure
cannot hide behind extra or duplicate builders the way it could under the old
count comparison. Run verification in a fresh Python subprocess after edits;
reusing `pytest.main()` in a persistent interpreter can execute an earlier
imported module while displaying a plausible but stale result — the remedy is
the fresh process, not a second run inside the stale one.

The stock quick profile runs the portable checkout surface. `publication_context`
tests require retained private Git history, raw corpus sources or original-source
proofs; `template_context` tests require the configured renderer or staged render
auxiliaries. Quick deselects these explicit prerequisites. Full selects both
groups and requires their assertions to pass; missing inputs remain failures.
