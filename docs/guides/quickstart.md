# Quick Start

Start with the offline model, then choose the test or publication workflow you
need. Commands run from the AILM repository root unless a block explicitly
changes to the template checkout. These are instructions, not a current release
receipt: a successful demo, a passing development check, and publication
acceptance establish different things. Preserve the logs and source identities
from your own run; [the release checklist](../method/release_checklist.md) defines
the final review and publication sequence.

Unfamiliar term along the way (a level name, a gate verdict, `{{TOKEN}}`,
`claim:<label>`)? Check the [glossary](../reference/glossary.md).

## Prerequisites

| Requirement | Pin | Checked with |
|---|---|---|
| Python | `3.12` — the pin lives in `.python-version` | `.venv/bin/python --version` |
| [`uv`](https://github.com/astral-sh/uv) | any recent release | `uv sync --dry-run` |
| Pandoc | needed only for the render step | `pandoc --version` |
| XeLaTeX (TeX Live) | needed only for the render step | `xelatex --version` |
| `mmdc` (mermaid-cli) | needed only if a manuscript section contains a ` ```mermaid ` block | `mmdc --version` |
| Lean/Lake | needed by the full profile's formal build and mutation control | `(cd formal && lake --version)` |
| `qpdf` | needed by the full profile after template rendering | `qpdf --version` |
| `rg` (ripgrep) | needed by the formal axiom report (`formal/Scripts/axiom_report.sh`) and by the vacuity control in `tests/test_formal_isomorphism.py`; hosted CI installs it | `rg --version` |
| Ollama, every declared probe, and a warmed call store | needed by the full profile's live cache audit and coverage run | see [Before running the `full` release profile](#before-running-the-full-release-profile) |
| Hosted acquisition credential | `OPENROUTER_API_KEY` is needed by the synthetic and in-context producers before clean release verification; keep it in a protected operator context, never committed | `test -n "${OPENROUTER_API_KEY:-}"` checks presence without printing the key |
| A `docxology/template` checkout | needed only for the render step — it provides the `infrastructure` package | see [`rendering_pipeline.md`](../method/rendering_pipeline.md) |

Start with the offline demos below. Rendering needs the template toolchain;
full-suite and publication acceptance additionally need the declared corpus,
formal, upstream-parser and cache prerequisites. A working demo is not evidence
that every acceptance dependency is present.

## 1. Clone and install

```bash
# [not re-run] — this checkout already existed. Clone anywhere; the render step
# later symlinks this directory into a docxology/template checkout as
# projects/working/active_inference_language_model.
git clone --branch v1.0.0 https://github.com/ActiveInferenceInstitute/Active_Inference_Language_Model.git active_inference_language_model
cd active_inference_language_model

# Prepare this checkout from its committed lock.
uv sync --locked
```

`uv sync` resolves [`../pyproject.toml`](../../pyproject.toml) against `uv.lock` and
writes `.venv/`. Dependency selection:

- The `dev` dependency group (the exact Hatchling build backend, pytest,
  pytest-cov, pytest-timeout, mypy, and Ruff) is what the release gates need,
  and it is included by default. Keeping Hatchling in the lock is what permits
  the wheel and source distribution to build offline without an isolated
  environment resolving a new backend.
- `gnn-interop` is an **optional** extra. It exists only so the GNN specification
  files can additionally be parsed by the upstream GeneralizedNotationNotation
  tooling; `src/ailm/gnn/` itself is dependency-free and its conformance checks
  run without the extra. Install it with `uv sync --extra gnn-interop` if you want
  the interoperability test to do more than skip.
- Because that extra is opt-in, a plain `uv sync` **removes** `psutil` from an
  existing `.venv` that has it; `uv sync --dry-run` shows the removal before it
  happens. That is correct behavior, not damage: the interoperability test then
  skips instead of running. Re-add it with `uv sync --extra gnn-interop`.

## 1a. Sanity-check the install with the offline demo

Before touching gates, figures, or the render pipeline, confirm the install
works at all with the offline newcomer demo — read-only and offline. The first live call may compile JAX code:

```bash
.venv/bin/python -m ailm            # or: ailm-demo
.venv/bin/python examples/quickstart_demo.py   # identical calls, no CLI
```

It prints the committed Stage 1-6 gate-verdict table from
`output/reports/gate_audit.json`, then a live Stage 2B G2P decode of one word.
See [`README.md`](../../README.md#usage) for what each call does and
[`architecture.md`](../architecture.md#package-map) for `ailm.demo`'s module row.

## 1b. Run the connected model

The historical demo above is separate from the live multirate hierarchy:

```bash
.venv/bin/python scripts/run_connected_ailm.py --text "the man sees a dog."
.venv/bin/python scripts/run_connected_ailm.py --text "the man." --generate
.venv/bin/python scripts/run_connected_ailm.py --text "the man." --trace
```

For explicit observations, save a JSON array of sentences such as
`[[{"form":"the","upos":"DET"},{"form":"man","upos":"NOUN"}]]` and run:

```bash
.venv/bin/python scripts/run_connected_ailm.py --structured "$AILM_INPUT_JSON"
```

`form` must be lowercase ASCII letters. UPOS is optional; when supplied it must
be information available before that token’s spelling, not an evaluation label
inferred from future context. Raw mode splits ASCII lexical runs, treats `.?!`
as boundaries, and does not observe their sentence-type labels. See
[the input and clock contract](../design/connected_runtime.md) before comparing
raw and structured results.

JSON includes letter loss, token coverage/OOV flags, upper-level posteriors and
an event count. `--trace` adds the prior, prediction, observation and posterior
for each event; it can produce a large output. `--generate` adds a bounded
next-word policy trace. Full word loss is `null` with `word_loss_status` set to
`infinite_oov` when any token is outside the lexicon; it is not a missing result
or a finite score on a silently filtered population. This CLI does not write the
claims ledger. The installed package exposes `ConnectedRuntime` and
`load_connected_model`; the script is the source-checkout command.

## 2. Use the project's interpreter, not `uv run`

This is the single most important operational rule in the repository, and it is
not stylistic.

```bash
# CORRECT
.venv/bin/python scripts/export_web_model.py
.venv/bin/python scripts/build_web_ailm.py
.venv/bin/python -m pytest tests/ -v

# WRONG — may resolve to a shim on PATH with the wrong Python and the wrong pymdp
uv run pytest tests/
pytest tests/
```

### While you are iterating, deselect the recorder replay

```bash
# Development loop over the same files, with the recorder replay deselected
.venv/bin/python -m pytest tests/ -m "not recorder_replay"
```

One test, `test_rerunning_the_recorders_is_a_no_op`, launches every claims
recorder as its own subprocess against an isolated ledger. Each pays full JAX and
`pymdp` import cost before doing any work, and one of them re-measures Stage 2.
In a dated local profiling receipt that single test dominated wall clock;
deselecting it shortened the same run materially. No duration is promised:
import caches, hardware, and the recorder inventory all change the timing.

It cannot be parallelized without destroying what it proves. The recorders write
one ledger under a single-writer lock, and the property being checked is that no
recorder overwrites another's label; give each its own copy and the test can no
longer see the conflict it exists to catch.

So it stays serial and stays slow, and you skip it while iterating. Run it before
you trust a result: `scripts/verify_release.py` already excludes it from the quick
profile and includes it in the full one, which is the same split expressed in the
release gate.

### Parallel development loop

```bash
# Parallel run over the whole suite with pytest-xdist. Use -n 4 instead of
# -n auto when other pytest fleets share the machine or drive (a 1-minute load
# of 5-13 is normal here).
.venv/bin/python -m pytest tests/ -n auto -q --no-cov

# The quick-profile marker exclusions work identically under -n
.venv/bin/python -m pytest tests/ -n auto -q --no-cov \
  -m "not recorder_replay and not raw_corpus_regeneration and not requires_cuda12"
```

The suite default stays serial. The `quick` profile in `scripts/verify_release.py`
runs its test step on up to four workers, half the host's cores, because that
step takes no coverage; the
`full` profile is intentionally serial, since its receipt parser expects exactly
one coverage TOTAL row and its recorder replay shares one ledger lock. With
`--cov`, the workers combine coverage automatically and the pyproject floor
still applies.

### Inspect every skip; do not trust a fixed census

The skip inventory changes when optional extras, external services, and the test
surface change. Run the suite with `-rs` and review every reported marker and
reason against the current test source; a remembered skip count is not release
evidence. The authoritative `full` release profile prepares its required inputs
and rejects any skip or expected failure in its publication pytest steps. Because
it rejects skips, that profile also sets `AILM_RELEASE_MANDATORY_ENVIRONMENT=1`
for its full pytest step: tests that would otherwise skip on an environment gap
(the upstream GNN checkout and its `gnn-interop` extra, an unserved Ollama probe)
fail instead and name what is missing. The CUDA-only tests under the
`requires_cuda12` marker are deselected by both profiles; run them with
`.venv/bin/python -m pytest -m requires_cuda12` on a CUDA host.

Tests that deliberately query the user's Ollama process are disabled unless
`AILM_RUN_LIVE_OLLAMA_TESTS=1` is set. Availability is checked inside each opted-in
test body, never in a collection-time marker, so merely collecting or running the
default suite does not probe the configured local endpoint. The live synthetic-recorder test
also supplies an absolute `AILM_CALL_CACHE_PATH` beneath its `tmp_path`; it cannot
append to the repository's call store.

AILM is a *sidecar* project: a standalone checkout that is symlinked into a
`docxology/template` checkout as `projects/working/active_inference_language_model`. It is not a workspace member of
that repository, so `uv run` invoked from the wrong directory can bind a different
virtualenv, and a bare `pytest` can pick up a `pyenv` shim. Both failure modes are
silent: the suite runs, and it runs against a different numpy or a different pymdp
than the one this project pins. Address the interpreter explicitly and the question
does not arise.

`uv run python` **is** correct inside the template checkout, because there it is
the repository's own declared toolchain. The rule is about which repository you
are standing in.

The release orchestrator enforces this boundary rather than merely documenting
it: invoke `.venv/bin/python -I -S scripts/verify_release.py`. The exact project
interpreter is required. `-I -S` prevents inherited `PYTHONPATH`, `.pth`,
`sitecustomize`, or `usercustomize` code from running before the orchestrator can
audit its environment. A site-enabled, non-isolated, or different interpreter is
refused before an inventory or release step runs.
Template Stage 03 and Stage 04 remain `uv run --locked` commands because their
working directory and dependency contract belong to the template repository.
The release runner also replaces any inherited `SOURCE_DATE_EPOCH` with the
audited source epoch recorded in the receipt; caller state cannot silently change its build
metadata.

## 3. Run the gates

The stage gates are the point of the project — each stage has a falsifiable
threshold with a positive and a negative control, and Stage N+1 may not start
until Stage N's gate passes.

```bash
# The registered stage gates on their own — the fastest meaningful signal.
.venv/bin/python -m pytest tests/test_stage0_gate.py tests/test_stage1_gate.py \
  tests/test_stage2_gate.py tests/test_stage3_gate.py tests/test_stage4_gate.py \
  tests/test_stage5_gate.py tests/test_stage6_gate.py -v

# Or by glob, so a gate added later is picked up without editing this page.
.venv/bin/python -m pytest tests/test_stage*_gate.py -v

# A clean whole-suite run first materializes the ignored web prerequisites.
.venv/bin/python scripts/export_web_model.py
.venv/bin/python scripts/build_web_ailm.py

# The whole suite.
MPLBACKEND=Agg .venv/bin/python -m pytest tests/ -v
```

For the coverage-floor command and the rest of the canonical command set
(gate audit, claims/manuscript tokens, corpus artifacts, figures), see
[`agent_instructions.md#rule-3-the-command-set`](agent_instructions.md#rule-3-the-command-set)
— that file is the canonical reference; this page only walks the first-run
order.

The coverage floor is claim:stage0_coverage_floor_pct% branch coverage, set once as
`fail_under` in [`../pyproject.toml`](../../pyproject.toml); pytest-cov applies it
whenever `--cov` is passed, and the release runner reads the same key, so no command
repeats the number. Note what that gate does and does not
prove: a green exit means the floor was cleared, not that any particular behavior
is covered. See [`testing_philosophy.md`](../testing_philosophy.md) for what a
passing suite here is allowed to mean.

## 4. Audit the gates themselves

A passing gate is worthless if the threshold could not have failed. This project
has shipped one gate that no implementation could pass and one that a lookup table
already passed — in opposite directions — so the classification is a re-runnable
script rather than something someone happens to do twice.

For each implemented stage `scripts/audit_gates.py` prints the trivial baseline,
the shipped model, the typed comparator, and one exact `GateVerdict`: `VACUOUS`,
`UNREACHABLE`, `ABOVE_REFERENCE`, `UNAUDITED`, `BRACKETED_BOUND`, or
`BRACKETED_REFERENCE`. The reasoning and the bound-versus-reference distinction
are in [`gate_design.md`](../gate_design.md); the exact command (and the faster,
noisier subsample variant) is in
[`agent_instructions.md#rule-3-the-command-set`](agent_instructions.md#rule-3-the-command-set).

## 5. Hydrate the manuscript variables

```bash
# Write output/data/manuscript_variables.json without touching docs/manuscript/.
.venv/bin/python scripts/z_generate_manuscript_variables.py --print-only

# Write the token map AND the injected manuscript tree. Needs TEMPLATE_REPO_ROOT
# when run by hand from the sidecar; the renderer sets it automatically.
TEMPLATE_REPO_ROOT="$AILM_TEMPLATE_ROOT" \
  .venv/bin/python scripts/z_generate_manuscript_variables.py
```

Every measured statement in the manuscript is a `{{TOKEN}}` resolved from
[`../data/claims.json`](../../data/claims.json); declared non-measurement variables
come from the validated v2 registry in `data/manuscript_literals.yaml`, the
search cutoff and priority policy come from `data/literature_search.json`,
deposited source identity and locators come directly from
`data/reference_sources.json`, and executable protocol values come from their
code registries. Before writing,
the production contract rejects malformed, unknown, empty, or hard-coded
result/configuration variables. After injection it requires an exact, token-free
source/output tree. Any violation exits non-zero before PDF work begins. The full
flow and the `TEMPLATE_REPO_ROOT` explanation are in
[`rendering_pipeline.md`](../method/rendering_pipeline.md).

## 6. Generate the figures

```bash
# List every discovered generator without writing anything.
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py --list

# Write all of them to output/figures/.
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py

# One at a time
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py --only perplexity_ladder
```

Figures read their numbers from `output/data/manuscript_variables.json`, so the
map generated in step 5 is a load-bearing prerequisite. A missing map makes the
generator raise; a stale map can remain structurally valid, which is why the
ordering itself matters. The full artifact list is in
[`output_inventory.md`](../reference/output_inventory.md).

Regeneration is byte-deterministic by construction (no RNG, no timestamps, a pinned
PNG `Software` field); `tests/test_figures_results.py` regenerates each figure and
compares the bytes.

## 7. Render

Set `AILM_TEMPLATE_ROOT` to the absolute path of the selected template checkout,
or declare that path once in a gitignored `.template-root` file at this
repository's root, which every provider-dependent command then reads when
`TEMPLATE_REPO_ROOT` is unset ([configuration reference](../reference/configuration.md)).
The checkout's local-only `projects/working/active_inference_language_model`
symlink must resolve to this AILM checkout. Prepare companions from this AILM root, then render
from the template and repair individual HTML references back here before final
validation:

```bash
AILM_PROJECT_ROOT="$(pwd -P)"
export TEMPLATE_REPO_ROOT="$AILM_TEMPLATE_ROOT"
.venv/bin/python scripts/z_render_docs.py --supporting-protocols
.venv/bin/python scripts/z_render_docs.py --check
cd "$AILM_TEMPLATE_ROOT"   # the explicitly selected docxology/template checkout

# [not re-run] — rewrites the tracked publication surfaces under output/
uv run --locked python scripts/pipeline/stage_03_render.py \
  --project working/active_inference_language_model

cd "$AILM_PROJECT_ROOT"
.venv/bin/python scripts/z_render_docs.py --publication-references
.venv/bin/python scripts/z_render_docs.py --publication-references --check
.venv/bin/python scripts/z_render_docs.py --check --supporting-protocols
cd "$AILM_TEMPLATE_ROOT"

# [not re-run] — validates the completed publication tree and PDF
uv run --locked python scripts/pipeline/stage_04_validate.py \
  --project working/active_inference_language_model
```

The pre-render check verifies the hydrated Markdown mirror. Complete HTML
acceptance follows rendering and reference repair: individual pages link to the
combined index's actual numbered targets, and strict checks reject unresolved,
malformed or inconsistent references. Stage 04 then records the repaired output
tree's validation and provenance.

Use the qualified route. A bare project name can select an existing
`projects/active/` entry before the `projects/working/` route; it is equivalent
only when both actually resolve to this checkout.

Produces, under the sidecar's `output/`: per-section HTML in `web/`, per-section
Beamer PDFs in `slides/`, the assembled markdown and combined PDF in `pdf/`, and a
machine-readable failure record in `reports/diagnostics.json`. A section whose
LaTeX fails is reported there per file — read it before theorizing.

For frozen publication-only repairs, select the independently reviewed
[publication-derivative route](../method/release_checklist.md#select-the-evidence-route-before-running-commands)
before any recorder command. It preserves original measurement inputs and
receipts while binding exact new source/output bytes. The ordinary strict route
below applies when current measurement source must match its original bindings.

For publication acceptance, prefer
`.venv/bin/python -I -S scripts/verify_release.py full` from the sidecar. It runs Stage
03 and then Stage 04 from the exact template checkout recorded in the receipt,
including individual HTML reference repair and strict supporting checks between them,
with a repeated locked offline wheel/source-distribution build (including
filenames, byte sizes, and SHA-256 digests in the receipt) and required Lean,
`qpdf`, and structural PDF-integrity gates between source acceptance and final
validation. An explicit `TEMPLATE_REPO_ROOT` is
authoritative: an invalid path fails closed rather than falling back to another
checkout. Both the sidecar and template Git states must be clean unless the
development-only dirty override is explicitly selected.

What the full profile checks, in order: the lock file, the environment, format,
lint, and types; a repeated locked offline wheel and source-distribution build
with recorded artifact digests; the full suite under the `pyproject.toml`
coverage floor; the stage-gate audit; the generated Lean bridge; the resolved
manuscript and documentation trees; the figures and the figure registry; the
benchmark call-store provenance bindings plus a live audit of the store (no
cache stratum is documented as an expected failure: the profile validates the
on-disk store and then re-queries its registered live sample on every run);
then template Stage 03 rendering, `qpdf` and structural PDF checks, and template
Stage 04 validation. The exact template checkout and both Git-state fingerprints
are recorded, and the ignored project route is validated again when the run
finishes. A development-override run may report successful checks, but its
separate `publication_acceptance` field is always false.

Git index entries marked `assume-unchanged` or `skip-worktree` are always
refused; the dirty development override does not waive that visibility failure.
Snapshot commands force real file-mode comparison and neutralize a local
`core.excludesFile`, so those settings cannot conceal a changed executable bit or
untracked input. Receipt and log destinations are checked lexically before any
symlink resolution, including scratch names from the writer's actual grammar.

The runner does not inherit arbitrary process configuration. Its top-level
interpreter and every sidecar Python step use `-I -S`. A stdlib-only bootstrap
adds the audited source and virtual-environment package directories explicitly,
without importing `site` or processing `.pth` files; unexpected executable or
unowned `.pth` entries and venv `sitecustomize`/`usercustomize` modules are hard
failures. It constructs a small allowlisted environment, replaces caller `PATH`
with fixed system and account tool locations, fixes the hash seed and timezone,
and removes pytest, coverage, proxy, Python-path and uv settings that could change
what a command executes. Git, uv, Lake, and qpdf are independently resolved and
their real paths, byte digests, availability, and version results are bound into
the receipt. Its parser then requires a nonzero pytest execution count from every
pytest step and exactly one coverage total from the coverage step. Consequently
`PYTEST_ADDOPTS=--collect-only` cannot create a green release receipt, even when
inherited by the shell that starts the runner.

The receipt separates `success` (the requested checks completed without changing
their recorded inputs) from `publication_acceptance`. A full run using
`--development-allow-dirty-full` can be useful while regenerating tracked
artifacts, but it is always marked ineligible for publication even if its checks
pass. The ignored template project route is validated again at finish because a
Git fingerprint cannot detect that route being retargeted.

Hosted GitHub CI intentionally runs only the `quick` profile. A real `full`
acceptance consumes the ignored machine-local external-call cache, requires a
reachable Ollama server serving every declared probe to recheck a stratified
sample, and uses an exact, clean template checkout routed to this sidecar, in
addition to the local Lean, TeX, Pandoc, and `qpdf` toolchain. A hosted runner
without those inputs could only fail or skip, so it must not publish a misleading
"full" receipt. Run the full command above from a prepared release workstation;
a green hosted quick receipt is necessary evidence, not publication acceptance.

## 8. Regenerating the inputs

These rewrite tracked data. Run them only when the underlying measurement or
corpus has genuinely changed.

```bash
# [not re-run] — rewrites the vendored corpus artifacts; a generate_*_counts.py
# precedes every recorder that reads its artifact.
.venv/bin/python scripts/generate_counts.py
.venv/bin/python scripts/generate_phoneme_counts.py

# [not re-run] — rewrites data/claims.json from measured runs. The phase
# inventory is discovered from every record_*_claims.py writer. The network
# phase's synthetic and in-context hosted producers require the credential
# and approved budget configured in the private acquisition environment.
# Do not put a literal key in a shell history or committed file.
test -n "${OPENROUTER_API_KEY:-}"
.venv/bin/python scripts/orchestrate_claim_recorders.py core
.venv/bin/python scripts/orchestrate_claim_recorders.py network
.venv/bin/python scripts/orchestrate_claim_recorders.py aggregate
```

The orchestrator partitions the exact live recorder inventory; `--list` shows a
phase without executing it. The **order matters between some producers**:
a `generate_*_counts.py` must precede the recorder that reads its artifact, the
Lean ledger is generated before the `formal` phase reads the build, aggregate
claims follow their measurement phases, statuses follow every recorder, and the
token map is generated only after all of them. Follow the running
order in [`../scripts/README.md`](../../scripts/README.md#running-order), which is
authoritative; re-recording out of order leaves the manuscript describing a
previous run.

The runtime vendor inventory is owned by
`ailm.resources.RUNTIME_VENDOR_ARTIFACTS`; do not copy its count into prose.
Every artifact in that registry is deterministic, documented in
[`provenance/sources.md`](../provenance/sources.md), and enforced by
`tests/test_provenance_checksums.py` — a documented checksum that has drifted from
the artifact is worse than no checksum, so it is checked rather than trusted.

## Where to go next

| You want to | Read |
|---|---|
| Know where a function belongs | [`architecture.md`](../architecture.md) |
| Understand the number pipeline | [`rendering_pipeline.md`](../method/rendering_pipeline.md) |
| Understand what the gates mean | [`gate_design.md`](../gate_design.md), [`testing_philosophy.md`](../testing_philosophy.md) |
| Understand the model itself | [`design/architecture.md`](../design/architecture.md) |
| Fix a failure | [`troubleshooting.md`](../troubleshooting.md) |
| Add a number to prose | [`syntax_guide.md`](../syntax_guide.md) |

## Capped hosted-model runs

For an explicitly authorized capped run, configure `AILM_OPENROUTER_BUDGET_FILE`
to the run's prepared budget ledger and `AILM_OPENROUTER_BUDGET_LIMIT_USD` to its
approved cap. `ailm.benchmarks.openrouter_budget` pins the declared provider and
pricing, reserves a worst-case charge before every HTTP attempt, and reconciles
only an explicit returned usage cost. Missing or invalid charge evidence retains
the reservation. The ledger is shared across recorder processes; a budget refusal
stops further requests. This controls spending and does not establish scientific
measurement or publication readiness. A protected operator-only secret file may
supply the acquisition process environment; neither keys nor private launchers
belong in the repository. After acquisition, use the
[clean release invocation](../method/release_checklist.md#the-sequence), which
explicitly unsets the key and budget variables.

## Before running the `full` release profile

Select the [evidence route](../method/release_checklist.md#select-the-evidence-route-before-running-commands)
first. Preparation that runs recorders is acquisition, not a routine docs check.
The frozen derivative route verifies the retained original source and performs
its declared local live audit; it does not rerun hosted measurement producers.

The release profiles check exact dependency sets without modifying `.venv`.
`quick` requires the base set from `uv sync --locked`; `full` also requires the
already-declared upstream-parser extra:

```bash
uv sync --locked --extra gnn-interop
```

Direct upstream interoperability tests accept `AILM_GNN_UPSTREAM`. The full
release’s controlled child environment does not inherit it, so the intended
checkout must resolve through `_UPSTREAM_CANDIDATES` in
`tests/test_gnn_conformance.py`; its first candidate follows the shared checkout
hierarchy. Confirm the actual selected parser path as described in the
[release prerequisites](../method/release_checklist.md#preconditions). A missing
checkout or extra fails the full profile instead of skipping. Returning to
`quick` requires `uv sync --locked` to restore the base dependency set.

`scripts/verify_release.py full` enforces the `pyproject.toml` coverage floor.
Some benchmark branches execute only when the required memoized external-model
calls are available. On a cold checkout the total falls
below claim:stage0_coverage_floor_pct% with no failing test to say why.

Warming the store is therefore a prerequisite, not an optimization, and it has
its own runbook:

```bash
ollama serve                                          # in another shell
.venv/bin/python scripts/warm_call_cache.py --check   # preflight only
.venv/bin/python scripts/warm_call_cache.py           # preflight, then warm
.venv/bin/python scripts/warm_call_cache.py --audit-strata-only
                                                    # repair only missing audit cells
```

For an already running project-owned Ollama server, select its loopback origin
before every warming, recording, replay and release command. For example:

```bash
export AILM_OLLAMA_ENDPOINT=http://127.0.0.1:15434
.venv/bin/python scripts/warm_call_cache.py --check
```

An absent variable retains `http://localhost:11434`; an explicitly empty or invalid
value fails instead of falling back. The selection is fixed when the process
imports the benchmark configuration. The full release preserves this approved
variable in its controlled child environment. Receipts and cache keys bind the
selected endpoint, and old-server records do not count toward new-server audit
coverage. A historical receipt must be verified with its recorded endpoint
selected, or the current-configuration check refuses it. Server lifecycle and
other users' workloads remain operator responsibilities: setting this variable
does not start, stop or reconfigure a server. Keep the recorded model weights and
serving template options identical when isolating the endpoint.

The preflight refuses to start unless the server answers and serves every
declared local probe, so a missing model is discovered before a long sweep. It also prints how much of the
store is already warm.

**Isolated warming resumes through eligible cached calls.** The store is
content-addressed and retains completed replies after interruption. This does
not apply to canonical publication scoring that requires fresh replies:
synthetic and in-context runs start a new scope, disable scored-call replay and
refuse cache hits. Preserve an interrupted scope as evidence; do not relabel
cached cells as a completed fresh run. Runtime and cost depend on the current
protocol and serving environment.

The run writes to an isolated copy of the ledger, never to `data/claims.json`.
A warming run that wrote the real ledger would overwrite recorded claims with a
fresh measurement; whether a value moved is the question worth asking, and the
script reports the delta at the end rather than resolving it for you.

Warming only supplies trusted calls. Publication also requires each network
recorder to be run canonically against the tracked ledger after its source,
configuration, or claim namespace changes. The two hosted scored-call
recorders (the stipulated synthetic and in-context probes) make their live
calls through the OpenRouter transport and require the exported
`OPENROUTER_API_KEY` and the approved budget for their fresh acquisition runs;
the external, aligned-G2P and drift producers use the local endpoint. Completed
receipts are then checked without requiring a hosted credential. That run writes a deterministic
receipt under `cache/network_claim_bindings/`; an isolated warming run
deliberately leaves those receipts untouched. Check the complete set directly
before starting the longer release:

```bash
.venv/bin/python scripts/verify_network_claim_bindings.py
```

Immediately after one producer finishes, checkpoint only that dynamically
discovered recorder before allowing a dependent recorder to make calls:

```bash
.venv/bin/python scripts/verify_network_claim_bindings.py \
  --recorder scripts/record_external_benchmark_claims.py
```

Without `--recorder`, the validator still requires the complete discovered set.
It refuses missing receipts, stale source/vendor/lock manifests, noncanonical
configurations, changed ledger values or run ids, wrong model identities,
missing or extra call keys, absent semantic evidence or validators, and any
claim value that cannot be rederived from the exact stored replies.

For an isolated, nonpublication recorder run, set `AILM_CALL_CACHE_PATH` to an
absolute JSONL path. An explicit `enable_call_cache(path=...)` argument takes
precedence over that environment value. Canonical recorders targeting the tracked
ledger refuse every nondefault selection: their receipts are validated against
`cache/bench_calls.jsonl`, so measuring against another store would make the
recorder and validator disagree about their provenance input.

`--audit-strata-only` is a bounded recovery path for a partially populated
store: it makes a small real call sample for every registered-model × request-kind
cell that `--verify-cache` requires. It does not reproduce the benchmark and its
reduced isolated-ledger values are never compared with or promoted to the real
ledger. The recorder verifies whatever cells already exist before this bootstrap,
then requires all declared cells after measurement before it can write even the
isolated ledger.

The `quick` profile has no such prerequisite.
