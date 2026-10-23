# Output Conventions

`output/` holds the regenerable publication and working outputs described below.
Packaged corpus artifacts and canonical external-call evidence live elsewhere. Two rules govern it, and the second one
is the one that matters.

1. **Generated current artifacts in `output/` are regenerable.** The explicit
   retained-input exception is `output/reports/connected_gum.json`: its original
   bytes are authenticated against the claims ledger and complete execution
   binding. It is distributed for offline validation and must not be regenerated
   merely to change provenance hashes or satisfy a check. A
   documented command sequence recreates the current output contract, and none
   of it is a source of truth.
   This was asserted for a long time before anyone tested it. Deleting the tree
   found two things in it that were **not** regenerable, and both are now fixed
   or documented:

   - The candidate pymdp patch bundle was hand-built and untracked, so clearing
     `output/` destroyed it and only a scratch copy survived.
     `scripts/generate_pymdp_sparse_patch.py` now derives it from the installed
     source and verifies `git apply --check` before writing.
   - The memoized external-model calls could not be regenerated without a live
     Ollama server serving the declared probes, and the `fail_under` coverage
     release gate silently depended on them: with a cold store the benchmark
     modules lose enough exercised branches for the gate to fall below
     claim:stage0_coverage_floor_pct%, with no failing test pointing at the cause.
     **That store has moved to
     `cache/bench_calls.jsonl`**, outside this directory, because an artifact
     that cannot be rebuilt is an input whatever directory it sits in. Warming
     it is now a documented prerequisite of the `full` release profile rather
     than an exception to the rule above. The dependency is stated in
     [`../scripts/verify_release.py`](../../scripts/verify_release.py).
     Network-authored claims also leave deterministic receipts under
     `cache/network_claim_bindings/`. Those ignored receipts are inputs, not
     generated publication surfaces: each binds one recorder's complete ledger
     namespace to the source/vendor/lock manifest, canonical configuration,
     local model content identities or hosted serving-route identities, and exact
     validated current-schema call keys. Hosted slugs/provider receipts do not
     authenticate immutable hosted weights.
     Their unkeyed digests are integrity/self-consistency checks under the
     trusted-local writer boundary, not hostile-writer authentication.
     An isolated warming ledger never overwrites them, and older-schema cache
     rows are explicitly ineligible to satisfy them.

   The lesson generalizes past these two, and both fixes take the same shape:
   give the artifact a generator, or move it out. An artifact that lives in the
   regenerable tree and has neither is protected by nothing at all, and the only
   way to find one is to reconstruct the owned inventory from its generators.
   Nothing in the **current owned inventory** now falls into that category.
2. **Nothing in `output/` is ever hand-edited to make something pass.** Not a
   figure, not a token map, not an injected manuscript section, not a diagnostic
   report. Editing a generated artifact to satisfy a check converts a caught
   problem into an uncaught one and leaves no trace in any source file.

## Ignored by default, with a curated publication allowlist

The release runner keeps disposable dependency and Matplotlib caches in
`.venv/release-cache/`, outside `output/` and within the environment tree excluded
from source-provenance scans. These are runtime caches, distinct from the
network measurement inputs in `cache/`. Dependency caches may
contain symlinks; the template's output inventory rejects such entries even
inside hidden output directories. They may also contain `.git` sentinel files
that are not repositories. Do not direct `UV_CACHE_DIR` into `output/` or the
project's source inventory.

`.gitignore` ignores `output/` and then re-includes this exact publication set:
`output/data/manuscript_variables.json`; every file under `output/figures/` and
`output/manuscript/`; `output/pdf/active_inference_language_model_combined.pdf`;
the eight report files `artifact_manifest.json`, `connected_gum.json`, `evidence_registry.json`,
`gate_audit.json`, `gate_audit_ablation.json`, `manuscript_composition.json`,
`rendered_provenance.json`, and `validation_report.json`; and both `output/web/_combined_manuscript.md` and
`output/web/index.html`. Everything else stays ignored — hydrated docs, slides,
`bench_raw`, diagnostics/logs, and every LaTeX intermediate. The memoized model
calls and their network-claim binding receipts live in `cache/`, which is an
input. The curated tracked set is materially smaller than the complete output
tree.

**Why any of it is tracked.** The render pipeline's `stage_04_validate.py` binds
a validation report to a manifest of the output tree, and
`collect_current_artifact_manifest` drops every git-ignored file on purpose: the
manifest is evidence a fresh clone must be able to reproduce, and an entry no
clone can produce reads as drift. With `output/` ignored outright the manifest
came back empty and the check could not run at all. The split above is the
template's own stable/transient convention, adopted rather than invented.

**Why the rest is not.** Ignored render intermediates, documentation mirrors,
and diagnostic logs can be regenerated and are not the distributed publication
set. They can still be checked locally: for example, documentation check mode
validates `output/docs/`, and a release receipt points to its step logs. Retain
those logs when preserving a local validation receipt. The canonical and
case-only gate reports are both publication inputs to figure consumers, so both
belong in the allowlist.

Two consequences worth internalising:

- **A fresh clone has the publication surfaces and nothing else.** The figure
  generators, the renderer and the token map still regenerate everything from
  source; a clone does not need to render before it can read the paper, but it
  does need the documented prerequisites before regenerating the complete
  publication. Offline model inference and supported measurements do not require
  first rendering a PDF.
- **Tracked is not the same as authoritative.** A reviewer still verifies a claim
  by re-running the producing command. The tracked artifacts are what the
  pipeline hashes to prove a report describes the run it says it does; they are
  not a substitute for reproducing the number. The authoritative evidence remains
  `data/claims.json`, with a `run_id` and `source` per entry, and the checksummed
  inputs under `vendor/`.

Rule 2 above is unchanged and matters more now, not less: a tracked artifact is
still never hand-edited to make something pass.

Distinct from that: `vendor/brown_counts.json` and `vendor/cmudict_counts.json` are
generated **and committed**, deliberately, so tests need no network. They are not in
`output/` and they are not disposable in the same sense — their sha256 values are
recorded in [`provenance/sources.md`](../provenance/sources.md) and enforced by
`tests/test_provenance_checksums.py`, because a documented checksum that has drifted
from the artifact is worse than no checksum.

## Sources of truth

Every artifact traces back to one of these, and only these:

| Source of truth | What it determines |
|---|---|
| `data/claims.json` | Every number in the manuscript, in the figures, and in the token map |
| The committed files named by `ailm.resources.RUNTIME_VENDOR_ARTIFACTS` | Every packaged corpus statistic the models are built from; `tests/test_provenance_checksums.py` binds the manifest to the documented checksums and installed wheel |
| `docs/manuscript/*.md` | The authored prose, with `{{TOKEN}}` markers where numbers go |
| `docs/manuscript/config.yaml` | Paper metadata and render policy |
| `src/ailm/**` | Models, figures and reusable computations |
| `scripts/`, `gnn/models/`, `formal/` | Orchestration, bounded recorder estimands, model declarations and formal contracts |
| `data/` protocol, registration, reference and literal records | Frozen measurement definitions, source identities and presentation producers |
| Pinned raw corpus inputs and `data/confirmatory/` | Original source/replay and separately registered retained evaluation evidence |
| `cache/bench_calls.jsonl` and `cache/network_claim_bindings/` | Append-only call evidence and original measurement bindings; inputs, not outputs |

If an artifact appears to disagree with one of these, the artifact is stale. Fix the
source and regenerate; never patch the artifact.

## Regeneration order

Select the [release evidence route](../method/release_checklist.md#select-the-evidence-route-before-running-commands)
first. The sequence below is the measurement-changing route: steps 2–4 include
evidence writers and are not safe docs-only validation. For a frozen publication
derivative, preserve measured inputs and run current output regeneration from
step 5 under its frozen source epoch, followed by its approved full recipe. Use
read-only ledger/Lean checks when those inputs are unchanged; do not re-record
or relabel historical measurements.

The order is not arbitrary — each output step consumes the previous step’s output.

Set `AILM_TEMPLATE_ROOT` to the selected template checkout whose qualified project
route resolves to this AILM root. Export that provider choice before local
hydration, supporting HTML, or figure-registry commands.

```bash
export TEMPLATE_REPO_ROOT="$AILM_TEMPLATE_ROOT"
# 1. Only when a corpus transform changed: run its owning generator from
#    vendor/README.md, then prove the documented bytes are current.
.venv/bin/python -m pytest tests/test_provenance_checksums.py -v

# 2. Re-run the complete affected recorder phases. Dynamic discovery prevents a
#    newly added recorder from disappearing from a hand-maintained command list.
.venv/bin/python scripts/orchestrate_claim_recorders.py core
.venv/bin/python scripts/orchestrate_claim_recorders.py network
.venv/bin/python scripts/orchestrate_claim_recorders.py aggregate

# 3. Bind the measured ledger into Lean and compile the formal layer.
.venv/bin/python scripts/generate_lean_ledger.py
(cd formal && lake build && ./Scripts/axiom_report.sh)
.venv/bin/python -m pytest tests/test_formal_isomorphism.py::test_the_negative_control_does_not_compile -v --no-cov
.venv/bin/python scripts/orchestrate_claim_recorders.py formal
.venv/bin/python scripts/generate_lean_ledger.py --check

# 4. Complete the ledger schema and audit every implemented gate.
.venv/bin/python scripts/assign_claim_status.py
.venv/bin/python scripts/audit_gates.py --json

# 5. Token map and resolved manuscript → output/data and output/manuscript
.venv/bin/python scripts/z_generate_manuscript_variables.py

# 6. Resolved documentation mirror → output/docs
.venv/bin/python scripts/z_render_docs.py --supporting-protocols
.venv/bin/python scripts/z_render_docs.py --check

# 7. Figures → output/figures (read their numbers from step 5)
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py
MPLBACKEND=Agg .venv/bin/python scripts/generate_cover.py
MPLBACKEND=Agg .venv/bin/python scripts/generate_cover.py --check

# 8. Bind the generated images to manuscript captions and accessibility text.
TEMPLATE_REPO_ROOT="$AILM_TEMPLATE_ROOT" .venv/bin/python scripts/generate_figure_registry.py
TEMPLATE_REPO_ROOT="$AILM_TEMPLATE_ROOT" .venv/bin/python scripts/generate_figure_registry.py --check

# 9. Render → resolve individual HTML references → check → validate.
#    Rendering and final validation run in the template; HTML repair runs here.
AILM_PROJECT_ROOT="$(pwd -P)"
cd "$AILM_TEMPLATE_ROOT"
uv run --locked python scripts/pipeline/stage_03_render.py \
  --project working/active_inference_language_model
cd "$AILM_PROJECT_ROOT"
.venv/bin/python scripts/z_render_docs.py --publication-references
.venv/bin/python scripts/z_render_docs.py --publication-references --check
.venv/bin/python scripts/z_render_docs.py --check --supporting-protocols
cd "$AILM_TEMPLATE_ROOT"
uv run --locked python scripts/pipeline/stage_04_validate.py \
  --project working/active_inference_language_model
```

The post-render HTML phase reads the combined index's actual reference links,
numbers and anchors, then replaces unresolved reference spans in the individual
exports with links into that index. It preserves literal notation inside code,
preformatted examples, scripts and styles. Missing or conflicting targets and
malformed references fail before the write plan is published. Check mode writes
nothing and rejects unresolved references or display numbers that differ from
the combined index. Final template validation runs afterward so its manifest and
provenance describe the repaired outputs rather than the earlier render.

Steps 5 and 7 must run in that order. The figure modules read
`output/data/manuscript_variables.json`, so generating figures against a stale token
map produces panels that disagree with the prose beside them. The figure modules
defend against this — each results figure re-derives an identity that must hold
between separately recorded claims (the perplexity ladder must be monotone,
`log(uniform perplexity)` must equal the model's cross-entropy plus its recorded
gain, the recorded G2P margin must equal model minus baseline) and raises rather than
drawing a story the ledger no longer supports. `tests/test_figures_results.py` feeds
each check a deliberately inconsistent token map, so none of them is decoration.

The figure receipt closes the remaining time-of-check/time-of-publication gap.
`figure_provenance.json` schema `ailm-figure-provenance-v3` binds the canonical token
map, every AILM Python source, shared orchestration/writer/style code, the bounded
data/vendor/GNN input inventory, and every PNG by portable path plus SHA-256, in
addition to the numeric environment. Strict JSON is parsed and hashed from one
non-following, nonblocking, regular-file-authenticated descriptor. A selective build
authenticates the whole prior receipt before it creates a stage. All selected figures
render in a temporary sibling; inputs and sources are rechecked, PNGs are fsynced and
individually atomically replaced, and the receipt is published last. A failed
generator therefore changes no PNG or receipt, and stdout contains final published
paths only. The receipt is a fail-detect commit marker, not a multi-file transaction:
an interrupted mixed set cannot validate against the retained old receipt. Missing,
mutated, or unreceipted PNGs make a selective build fail closed rather than silently
inheriting them.

## Determinism

Regeneration is byte-stable, and that is a checkable property rather than a hope:

- **Figures.** No RNG, sorted iteration, no timestamp drawn into the image, and the
  PNG `Software` text key overridden with a constant — matplotlib stamps its own
  version there by default, which would make two byte-comparisons of the same plot
  disagree across environments. The v3 receipt records and revalidates each PNG's
  SHA-256 and is itself byte-stable: it contains neither absolute checkout paths nor
  elapsed wall times.
- **Token map.** Sorted keys. `{{META_BUILD_TIMESTAMP}}` honours an explicit
  `SOURCE_DATE_EPOCH`; when it is absent, the manuscript orchestrator and the
  release runner both derive the epoch of the last audited commit that touched
  an input outside `output/`, so direct and release regeneration produce the
  same reproducible stamp. Deriving it from that commit rather than from `HEAD`
  is what makes regeneration converge: commit the input change first, regenerate,
  then commit the refreshed `output/` tree; that second commit touches no input,
  so it does not move the stamp it carries.
- **Corpus artifacts.** Every owning generator fixes its ordering and split,
  writes no timestamp, and records any seed it genuinely uses. The exact
  documented inventory and digests are enforced by
  `tests/test_provenance_checksums.py`.

Checking determinism:

```bash
AILM_FIGS_BEFORE="$(mktemp)"
AILM_FIGS_AFTER="$(mktemp)"
shasum -a 256 output/figures/*.png > "$AILM_FIGS_BEFORE"
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py
shasum -a 256 output/figures/*.png > "$AILM_FIGS_AFTER"
diff "$AILM_FIGS_BEFORE" "$AILM_FIGS_AFTER"
```

## What each subdirectory is

| Path | Written by | Contains |
|---|---|---|
| `output/data/` | `scripts/z_generate_manuscript_variables.py` | `manuscript_variables.json` — the resolved `{{TOKEN}}` map |
| `output/docs/` | `scripts/z_render_docs.py` | Claim-tag-resolved mirror of the authored documentation tree; transient and ignored |
| `output/figures/` | `scripts/generate_figures.py` and `scripts/generate_figure_registry.py` | The manuscript PNGs, deterministic generation receipt, and validated caption/alt-text registry |
| `output/manuscript/` | `scripts/z_generate_manuscript_variables.py` through `infrastructure.rendering.manuscript_injection` | Token-substituted copies of `docs/manuscript/*.md`, plus refreshed `config.yaml` and `preamble.md`. **The renderer reads these, not `docs/manuscript/`** |
| `output/pdf/` | `scripts/pipeline/stage_03_render.py` | The assembled markdown, the combined PDF, and LaTeX intermediates |
| `output/slides/` | same | Per-section Beamer PDFs and their `.tex` / `.aux` / `.log` / `.nav` / `.snm` / `.toc` / `.vrb` intermediates |
| `output/web/` | same | Per-section HTML and the assembled markdown |
| `output/reports/` | release, gate-audit, and template render/validation orchestrators | Evidence receipts, gate audit, diagnostics, artifact manifests, and render validation/provenance reports |

## The hand-editing trap, concretely

`output/manuscript/*.md` is the most dangerous file set in the project, because it
is both **generated** and **what the renderer actually reads**. `resolve_manuscript_dir`
prefers `output/manuscript/` whenever it exists and contains `*.md`. So an edit
there changes the PDF, is explained by no source file, and is silently destroyed by
the next hydration.

| Tempting edit | What it actually does | The real fix |
|---|---|---|
| Fix a sentence in `output/manuscript/03_methods.md` | Changes this PDF only; reverted on next hydration | Edit `docs/manuscript/03_methods.md` |
| Type a number over an unresolved `{{TOKEN}}` | Fabricates a result and removes the signal that a claim is missing | Record the claim, re-run hydration |
| Hand-adjust `output/data/manuscript_variables.json` | Decouples prose *and* figures from the ledger in one move | Fix `data/claims.json` through a recorder run |
| Redraw or crop a figure by hand | Breaks byte-determinism; the next regeneration silently reverts it | Change the generator in `src/ailm/figures/` |
| Delete a `diagnostics.json` entry | Hides a real render failure | Fix the section that failed |

## Adding a new artifact

1. Put reusable computation in `src/ailm/` and orchestration/artifact writing in
   its owning `scripts/` command. Bounded recorder-specific estimands follow the
   documented ownership contract in [architecture.md](../architecture.md).
   Read measured values from authenticated inputs; do not type them in.
2. Give it a fixed, predictable filename under the right `output/` subdirectory.
3. Add a test. For a figure, the legibility and canvas-overflow assertions in
   `ailm.figures._common` apply automatically through `save_figure`; there is no
   clamp, so an overflowing label fails the build rather than shrinking to
   illegibility.
4. Register it in [`output_inventory.md`](output_inventory.md) with its label and
   generator.
5. Reference it from the manuscript using the cross-reference conventions in
   [`syntax_guide.md`](../syntax_guide.md).

## Cross-references

- [`output_inventory.md`](output_inventory.md) — the concrete artifact list
- [`rendering_pipeline.md`](../method/rendering_pipeline.md) — how the token map is built and why a missing claim aborts the render
- [`provenance/sources.md`](../provenance/sources.md) — the committed, checksummed corpus artifacts
- [`architecture.md`](../architecture.md) — which module may write which artifact
