# AGENTS.md — scripts/

## Thin-orchestrator contract

A file in this directory is orchestration only: argument parsing, path
bootstrap (the shared `TEMPLATE_REPO_ROOT` resolution), logging, and delegated
calls into `src/ailm/` entrypoints. A script may:

- parse `argv`, read files, write `vendor/` or `output/` or `data/claims.json`
- print paths to stdout, diagnostics to stderr, and return a meaningful exit code
- import from `src/ailm/` and call it

New reusable matrix builders, metrics, decoders, and statistical computations go
in `src/ailm/`, where the coverage gate and unit suites can exercise them directly.
Bounded recorder-specific estimands may remain beside their sole artifact writers only
when they are pure and directly tested; the first second consumer moves them into
`src/ailm/` rather than duplicating them. Claiming that no script computes would
still be false.

`scripts/` does own two things nothing else may write:

- **`vendor/*.json`** — the derived corpus artifacts (`generate_counts.py`,
  `generate_phoneme_counts.py`)
- **`data/claims.json`** — every measurement (`record_stage*_claims.py`; a handful of
  specialized recorders own the remaining namespaces; tests verify and never write
  canonical claims)

## `infrastructure` imports — the sanctioned boundary

`infrastructure` (the template repository's render-time package) is importable
only from `scripts/`, never from `src/ailm/`, and only by the **two** sanctioned
importers, each after resolving the template root:

- `z_generate_manuscript_variables.py` imports
  `infrastructure.rendering.manuscript_injection` — wiring manuscript tokens is
  its entire reason to exist.
- `generate_figure_registry.py` imports `infrastructure.documentation`
  (`FigureRegistryError`, `build_generated_figure_registry`,
  `write_generated_figure_registry`) so the registry schema has exactly one
  definition; that script's docstring explains why the builder is imported
  rather than reimplemented.

This is a deliberate, documented deviation from any one-importer rule —
`docs/architecture.md` ("`src/` must never import `infrastructure`") is the
authoritative statement, and
`tests/test_figure_registry.py::test_nothing_under_src_imports_infrastructure`
enforces the `src/ailm/` side of the line. Adding a third importer requires a
decision entry and a matching edit here.

## Inventory

101 Python files (2026-10-23 audit). `scripts/README.md` carries the
per-script table — script | purpose | delegates to | run command. By family:

| Pattern | Count | Role |
| --- | --- | --- |
| `generate_*` | 23 | `vendor/` corpus artifacts, figures, cover, Lean ledger bridge, figure registry, pymdp patch |
| `record_*` | 47 | the only writers of `data/claims.json`; each owns one claim namespace — the 46 `record_*_claims.py` recorders plus `record_confirmatory_replication.py` (decision 111) |
| `evaluate_*` | 2 | structural/evaluation reports that print and record nothing |
| `probe_*` | 5 | standalone measurements that record nothing |
| `verify_*` | 4 | release profiles, network-claim bindings, reference sources, the publication derivative |
| `z_*` | 2 | last-pipeline-step renderers: manuscript tokens, docs hydration |
| other | 18 | `audit_gates.py`, `orchestrate_claim_recorders.py`, `assign_claim_status.py`, `sweep_smoothing.py`, `warm_call_cache.py`, `build_web_ailm.py`, `export_web_model.py`, `capture_dashboard.py`, `run_connected_ailm.py`, `precheck_confirmatory_replication.py`, `list_raw_corpus_inputs.py`, `fillmore_subject_rerun.py`, the shared `_artifact_io.py` atomic writer, and the five `_publication_*.py` workers behind `verify_publication_derivative.py` |

## May change freely

The cover carries qualitative claims and is a release input.
`generate_cover.py` validates layout before atomically replacing
`docs/manuscript/cover.png`; `--check` compares a fresh render with the published
bytes without writing. The full release plan requires both steps before Stage 03.
Regenerate after changing cover text, architecture rows, verdicts, or shared style.

The documentation renderer reports Markdown absent from its current source
inventory as stale, including obsolete directories containing such documents.
Check mode exits 1 and leaves them intact; write mode still refuses unmapped
content. Reconcile the reported paths explicitly before publishing a new tree.

- Add a `record_stageN_claims.py` for a new stage, writing through
  `ledger.record(label, value, run_id=..., source=...)`.
- Add a diagnostic flag or a clearer failure message.
- Add an `audit_*` function to `audit_gates.py` and move that stage out of
  `UNAUDITED_STAGES`. **Do this before building the stage, not after.**

## Must not change without an explicit decision

**Do not hand-edit `data/claims.json`.** Run the recorder. An entry without a real
`run_id` and `source` cannot be re-derived, and
`test_every_ledger_entry_has_provenance` fails on it.

**Do not record a number the script did not just compute.** `record_stage0_claims.py`
is the model: it derives the value from `len(CANONICAL_ARPABET)`, cross-checks against
`EXPECTED_PHONEME_COUNT`, and *refuses to record* on mismatch rather than persisting a
stale value. Copying a figure from a previous run's stdout into a recorder converts the
ledger from evidence into transcription.

**Do not remove `verify_decoder_equivalence` from `record_stage2_claims.py`, and do not
let it pass with a nonzero count.** The full-set G2P statistics are computed through
the numpy oracle *because* it is proven identical to the pymdp decoder on
claim:stage2_decoder_disagreements positions. Without that proof the recorded numbers
describe the oracle, not the agent, and every Stage 2 claim is void.

**Do not shrink an eval set to make a recorder fast.** Any subsample of the G2P eval is
underpowered — the stride reaches only
p = claim:stage2_g2p_subsample_mcnemar_p while the full set reaches
claim:stage2_g2p_mcnemar_p. And never use a *prefix*: `g2p_pairs` is alphabetically
sorted, so `test[:1200]` is a/b/c words only and reports +claim:subsample_prefix_margin_pts pts against a true
claim:stage2_g2p_margin_pts pts — a claim:subsample_prefix_overstatement_factor× overstatement that reads as a comfortable
pass.

**Do not make the artifact generators non-deterministic.** Keys sorted, no RNG, no
timestamps. `tests/test_provenance_checksums.py` regenerates each artifact and requires
byte-identical output, and it cross-checks the sha256 recorded in
`docs/provenance/sources.md`. Adding a key changes the hash — update the document in
the same change. A documented checksum that has drifted is worse than none: it tells a
reader their correct file is corrupt. (This has already happened once here, when
`phone_initials` was added.)

**Do not remove the non-canonical phone guard** in `generate_phoneme_counts.py`. It
raises rather than vendoring an inventory containing a phone outside
`CANONICAL_ARPABET`.

**Do not make `generate_counts.py --force-fallback` canonical.** It is an
explicitly noncanonical diagnostic: a different filename, a rejected schema,
and no claim-recording path. Canonical writers authenticate the pinned raw
archive and fail closed. The shared normalization steps — collapsing space
runs and stripping spaces before `.` and `'` — remain load-bearing.

**Do not reimplement the verdict inside `audit_gates.py`.** It calls
`ailm.gates.evaluate_gate`, which delegates to `ailm.gates.classify_gate`, and
`_classify` in `ailm/figures/gates.py` is a thin wrapper over that same function —
the shared call is why the audit and the published figure cannot disagree. Restoring a
local copy of the classification to either caller reintroduces the drift hazard the delegation removed. A
change to how a threshold is classified belongs in `src/ailm/gates.py`, which
`tests/test_gates.py` pins and the Lean layer binds through
`GateVerdict.classify` (see `docs/formal.md`).

**Do not delete an `UNAUDITED_STAGES` row to make the output look clean.** The one
remaining row is Stage 7's "Whisper → phoneme → TTS round trip", which measures
Whisper/TTS rather than this project's model and is re-scoped to the boundary
interface contract (`docs/design/speech_boundary.md`). Listing it is the point.

**Do not add computation to `z_generate_manuscript_variables.py`.** It wires
`src/ailm/manuscript_variables` to `infrastructure.rendering.manuscript_injection` and
nothing else. It must keep failing loudly when `infrastructure` cannot be located,
rather than emitting a manuscript full of unresolved `{{TOKEN}}` markers.

## Local gotchas

**`TEMPLATE_REPO_ROOT` is required, not optional.** AILM is a sidecar: a
standalone checkout symlinked into `template/projects/working/`, so a relative
walk-up from the real path lands in the sidecar's parent directory, which has no
`infrastructure` package — or worse, silently finds an unrelated one. The renderer sets the variable; the walk-up and
sibling-checkout fallbacks exist only for manual invocation.
An explicitly configured root is authoritative: manuscript hydration and figure
registry publication both refuse an invalid value instead of selecting another
checkout. Fallback discovery applies only when the variable is absent or blank.

**Hydrated docs preserve an unchanged publication.** `z_render_docs.py` checks
the existing tree before reading and substituting source contents. Invalid
publication paths fail before hydration, and a byte-identical rerun preserves
the public directory instead of retaining another identical generation. Changed
content still goes through staging, atomic publication, and post-write checks.

`--prune-retired --dry-run` lists the sibling generations without hydration or
deletion. Applying `--prune-retired` keeps the newest `--keep` eligible trees
(default 3), plus every tree younger than `--min-age` (default 3600 seconds).
The active publication is protected. Symlinks, non-directories and names outside
the publisher's token grammar are reported and retained. Already-open files
remain readable after pruning; later path-based opens into a pruned generation
can fail. The age grace is a precaution for staging trees, not a reader lease.

**The `z_` prefix is a human ordering hint, not enforcement.** The canonical pipeline
orders by stage definition. A new script that must run between analysis and token
hydration should be named `y_*`.

**`generate_figures.py` writes the manifest to stdout and everything else to stderr.**
One absolute path per line on stdout, nothing else. `--list` prints to stderr and exits
0. Do not add chatter to stdout.

**Every Stage 1 consumer imports the authenticated production fixture.**
`ailm.evaluation.STAGE1_EVALUATION_TEXT` is the single owner of the gate
population. Its version, SHA-256, byte extent, source, license, and repetition
count are validated before use. Recorders and `audit_gates.py` must import it
directly; importing `tests/test_stage1_gate.py` reintroduces a second protocol
owner and is rejected by `tests/test_evaluation_fixture.py`.

**`--limit` in `audit_gates.py` is an evenly spaced stride, never a prefix** —
`_iter_pairs` strides across the whole sorted list, and the `--help` string says
so. An alphabetical prefix would inflate the G2P margin claim:subsample_prefix_overstatement_factor× (decision 4).

**Never mask a generator or recorder failure.** No `2>/dev/null || true`. Suppressing
stderr *and* swallowing the exit code turns a broken regenerate step into a silent
no-op, and a stale `vendor/` artifact then propagates into every recorded claim.

**macOS has no `timeout`.** Do not wrap any of these invocations in it — every entry
returns 127 and a smoke loop reads as a pass.

**Set `MPLBACKEND=Agg`** for `generate_figures.py`. Run the command directly:
piping combined output through a filter makes the filter's status easy to mistake
for the generator's status. Font-cache chatter is preferable to masked failure.

**Always `.venv/bin/python`.** Never `uv run python` in this sidecar.

## Verification for this directory

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root

# Artifacts must regenerate byte-identically and match their documented checksums
.venv/bin/python -m pytest tests/test_provenance_checksums.py -q --no-cov

# Gate soundness — non-zero exit on any non-bracketed typed verdict / model on the
# wrong side; add --json to also export output/reports/gate_audit.json
.venv/bin/python scripts/audit_gates.py

# The recorders → the ledger → the tokens, then the prose gate
.venv/bin/python scripts/record_stage0_claims.py
.venv/bin/python scripts/record_stage1_claims.py
.venv/bin/python scripts/record_stage2_claims.py
.venv/bin/python scripts/z_generate_manuscript_variables.py
.venv/bin/python -m pytest \
  tests/test_claims_discipline.py tests/test_manuscript_variables.py -q --no-cov

# Figures, discovered and written
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py
```
