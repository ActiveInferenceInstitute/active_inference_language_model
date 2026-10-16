# AGENTS.md — src/ailm/

Rules for the package root — the flat modules that sit beside the subpackages —
and what holds across all of them. Subpackage-specific rules are in the nested
`AGENTS.md` files: [`levels/`](levels/AGENTS.md), [`corpus/`](corpus/AGENTS.md),
[`verification/`](verification/AGENTS.md), [`figures/`](figures/AGENTS.md),
[`generative/`](generative/AGENTS.md), [`gnn/`](gnn/AGENTS.md),
[`benchmarks/`](benchmarks/AGENTS.md), and [`dashboard/`](dashboard/AGENTS.md).

The module-by-module inventory is owned by one file,
[`../../docs/architecture.md`](../../docs/architecture.md), which a repository test
checks against every shipped Python module. This file does not duplicate it.

## Ownership boundary

This directory owns the project's **bookkeeping primitives** — the modules that
other code is measured against rather than merely using:

- `claims.py` is the source of numeric truth. Nothing else may be.
- `control.py` decides whether a gate is allowed to certify anything.
- `gates.py` holds the one gate contract that `scripts/audit_gates.py`, the figures,
  and the Lean layer all bind to.
- `viz_config.py` is the only place a color, a font size, or a canvas size is
  decided.
- `required_claims.py` names the labels a render may not proceed without.

`cases.py` owns the case ontology; `manuscript_variables.py` and
`manuscript_contract.py` own the ledger → manuscript bridge and its rejection rules;
`statistics.py` and `complexity.py` own the shared statistical and accounting
computations; `resources.py` owns the runtime vendored-artifact manifest;
`call_cache.py`, `replay_provenance.py`, and `ledger_lock.py` own the replay and
single-writer machinery. Model code belongs in `levels/`, not here.

## May change freely

- Add a flat module for a genuinely cross-cutting concern, with tests.
- Add a `CaseRole` deprel mapping or an adposition entry in `cases.py`, provided
  `tests/test_ud.py` pins the new behavior and the change is recorded as a revision
  of a research claim, not as a bug fix.
- Extend `REQUIRED_CLAIMS` in `required_claims.py` when a new number becomes
  load-bearing for the manuscript. Adding a label there makes a render fail loudly
  when the measurement is missing, which is the intended direction of failure.
- Add a palette key or a font role to `viz_config.py`, keeping every size at or above
  `MIN_FONT_PT`.

## Must not change without an explicit decision

- **`CaseRole.GEN` is Genitive, the linguistic case.** CEREBRUM also defines a
  core Genitive model function and a distinct novel Generative function, reusing
  source abbreviation `[GEN]`. They are `CerebrumFunctionRole.CORE_GENITIVE` and
  `CerebrumFunctionRole.NOVEL_GENERATIVE`; their scoped serialized values must
  remain distinct. Merging either changes the represented source ontology.
- **`MIN_FONT_PT` and the palette semantics.** Palette keys name *meanings*
  (`MODEL`, `UNIGRAM`, `UNIFORM`, `FLOOR`, `ANTI`), not appearances. Renaming one is
  a claim that the meaning changed. Lowering `MIN_FONT_PT` to make a figure fit
  inverts the invariant it exists to enforce.
- **`ClaimsLedger` on-disk schema** (`ailm-claims-v2`) and the
  `claim:([A-Za-z0-9_\-]+)` prose pattern. Widening the pattern changes what counts
  as a tagged claim across the whole repository.
- **Never hand-edit `data/claims.json`.** Write through `ledger.record(...)` from a
  recorder script, so `run_id` and `source` are real.

## Local gotchas

**The `{{TOKEN}}` name is mechanical.** Token = ledger label uppercased:
`stage1_letter_perplexity` → `{{STAGE1_LETTER_PERPLEXITY}}`. There is deliberately
no second mapping table. Do not add one; a second table is a second thing to drift.

**Some tokens are derived, not recorded.** `manuscript_variables.py` also emits
composites such as `STAGE1_UNIGRAM_MARGIN`, `STAGE1_CEILING_SHORTFALL_RATIO`,
`STAGE2_MCNEMAR_TOTAL_DISCORDANT`, `STAGE2_PHONEME_UNIGRAM_MARGIN`, and the
`META_*` provenance block. Those exist as manuscript tokens but have **no ledger
entry**, so a `claim:` tag naming one of those labels in prose is *unbacked* and will fail
`tests/test_claims_discipline.py`. In `.md` prose use only labels that are actually
in `data/claims.json`; in the manuscript use the `{{TOKEN}}` form, which resolves
either kind.

**`ControlHarness` is not decoration.** A gate that only asserts the good value is
below a threshold proves nothing about the threshold. `assert_gate_is_live` plus
`verify_negative_control_fails` is the contract; `GateVacuousError` means the metric
cannot discriminate and the stage is blocked, not that the harness is too strict.

**`cases.py` has a research-claim boundary inside it.** `UD_DEPREL_TO_CASE` and
`ADPOSITION_TO_CASE` derive DAT/INS/LOC/ABL/VOC that English does not mark
morphologically. Agreement with them is *agreement*, never *accuracy*. Any Stage 4
number resting on those tables must say so in the same sentence.

**`manuscript_variables.py` must stay `infrastructure`-free.** It reads
`SOURCE_DATE_EPOCH` from the environment itself rather than importing a template
helper. That is not an oversight —
`tests/test_figure_registry.py::test_nothing_under_src_imports_infrastructure`
enforces it; the template's drift gate carries the same check but only warns for
this project.

## Verification for this directory

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root

# The bookkeeping primitives and the ontology
.venv/bin/python -m pytest \
  tests/test_claims_ledger.py tests/test_claims_discipline.py \
  tests/test_control_harness.py tests/test_cases.py \
  tests/test_manuscript_variables.py -q --no-cov

# Package-wide coverage gate
.venv/bin/python -m pytest tests/ --cov=ailm --cov-report=term-missing
```
