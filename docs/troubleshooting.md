# Troubleshooting

Symptom-driven recipes for failures that have actually occurred in this project.
Every command below is written for the sidecar working directory:

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root
```

---

## `uv` fails with "Operation not permitted" on its cache

**Symptom.** Any `uv` invocation — `uv sync`, `uv run …` — aborts while reading or
writing `~/.cache/uv`, or reports a permission error unrelated to the project
files.

**Cause.** A sandboxed shell (agent harness, restricted CI runner) that permits
writes under the project tree but not under `~/.cache`. `uv` needs its cache
directory for every operation, including ones that appear to be pure reads.

**Fix.** Skip `uv` entirely. The project's `.venv` is already provisioned, and its
interpreter needs no cache:

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/audit_gates.py
```

Use `uv sync` only from an unsandboxed shell, when dependencies actually changed.
Nothing in the day-to-day loop requires it.

---

## Bare `uv run pytest` resolves to the wrong Python and a different pymdp

**Symptom.** Tests fail with import errors, missing `pymdp.agent.Agent`
signatures, or assertion failures on matrix shapes that pass under the documented
command. Or: pymdp reports version 0.0.7.1 rather than the pinned 1.0.4.

**Cause.** `uv run pytest` asks uv to run *a program named pytest*, which can
resolve to a system or shim `pytest` outside the project environment — a different
Python and a different, much older pymdp. pymdp 1.0.x is a JAX rewrite; against
0.0.7.1 essentially every matrix convention in this project is wrong.

**Fix.** Always invoke the interpreter, never a bare console script:

```bash
.venv/bin/python -m pytest tests/ -q          # canonical in this sidecar
```

Confirm what you are actually running before debugging anything else:

```bash
.venv/bin/python -c "import sys; print(sys.version)"
.venv/bin/python -c "import pymdp, inspect; print(inspect.getfile(pymdp))"
```

[`../AGENTS.md`](../AGENTS.md) records the same rule. `python -m pytest` is what
makes the difference; `.venv/bin/python` removes the remaining ambiguity.

---

## Both `cpython-312` and `cpython-313` `.pyc` files under `__pycache__`

**Symptom.** `src/ailm/**/__pycache__/` contains bytecode for two Python minor
versions. Sometimes accompanied by behavior that does not match the source you
just edited.

**Cause.** The tree has been run under more than one interpreter. Stale bytecode
does not normally execute across versions, but an mtime collision under a single
version can make Python reuse a `.pyc` for source that has since changed —
which presents as edited code that has no effect.

**Fix.** First verify the executing interpreter and imported module path. A
mixed bytecode inventory alone is not evidence that stale code executed. If a
same-version cache is implicated, reproduce in a fresh process with an isolated
bytecode-cache prefix rather than deleting caches across a shared tree:

```bash
AILM_PYCACHE_DIR="$(mktemp -d)"
.venv/bin/python -X pycache_prefix="$AILM_PYCACHE_DIR" -c \
  "import ailm, sys; print(sys.executable); print(ailm.__file__)"
```

Keep the reproduction’s paths and output with the failure evidence.

---

## `timeout: command not found`

**Symptom.** Every command in a loop returns 127, and the loop reads as a clean
pass because nothing raised.

**Cause.** Stock macOS has no `timeout` binary. A smoke loop wrapping each step in
`timeout` returns 127 for every entry — and a check that only looks for the
absence of an exception treats that as success.

**Fix.** Do not wrap anything in `timeout`. Per-test time bounds are already
configured (`timeout = 300` in
[`../pyproject.toml`](../pyproject.toml), via `pytest-timeout`), and a
long-running step should be bounded by making it cheaper, not by killing it. If
you need a wall-clock bound for an ad-hoc command, run it in the background and
poll.

More generally: assert that a tool exists *and runs* before building a loop on
it. `command -v X` passes for a binary that is present and broken.

---

## `UserWarning: A JAX array is being set as static!`

**Symptom.** One such warning per `Agent()` construction:

```
A JAX array is being set as static! This can result in unexpected behavior
and is usually a mistake to do.
```

**Cause.** pymdp 1.0.3's internals, on its own arrays. It was emitted for every
agent this project builds and is not caused by AILM code. Verified against the
installed 1.0.4 (2026-08-29): `Agent(A=..., B=..., C=..., D=...)` with real jnp
arrays no longer raises the warning (Policies now stores a hashable nested tuple
statically and materializes `policy_arr` lazily).

**Status.** Resolved upstream in 1.0.4. On 1.0.3 it was filtered where it would drown useful
output — see the `warnings.filterwarnings` call at the top of
[`../scripts/audit_gates.py`](../scripts/audit_gates.py) and
[`../scripts/record_stage2_claims.py`](../scripts/record_stage2_claims.py):

```python
warnings.filterwarnings("ignore", message=".*JAX array is being set as static.*")
```

**Do not broaden that filter.** Suppressing `UserWarning` wholesale would hide
pymdp's normalization complaints, which are the warnings you want.

---

## A literal `{{TOKEN}}` appears in the rendered output

**Symptom.** The rendered manuscript contains `{{STAGE2_G2P_ACCURACY}}` rather
than a number.

**Cause.** One of three: the variable generator was not re-run after the
recorders; the token name does not match a ledger label uppercased; or injection
was skipped because the template root could not be resolved (next section).

**Fix.**

```bash
# 1. Regenerate the token map and inspect it
.venv/bin/python scripts/z_generate_manuscript_variables.py --print-only
.venv/bin/python -m json.tool output/data/manuscript_variables.json

# 2. Confirm the token exists (token = ledger label, uppercased)
.venv/bin/python -c "
import json
d = json.load(open('output/data/manuscript_variables.json'))
print('STAGE2_G2P_ACCURACY' in d)"

# 3. Sweep the rendered tree (the generator already enforces this; this is inspection)
! rg -n -F '{{' output/manuscript/
```

An unresolved or malformed token in a published document reads as a fabricated
result. The project contract now makes it a nonzero generator exit even though
the shared template writer itself only warns. See
[`syntax_guide.md`](syntax_guide.md#detecting-unresolved-tokens).

---

## `MissingClaimError: the manuscript requires N claim(s) the ledger does not have`

**Symptom.** `scripts/z_generate_manuscript_variables.py` exits 1 with a list of
missing labels.

**Cause.** `REQUIRED_CLAIMS` in
[`../src/ailm/manuscript_variables/`](../src/ailm/manuscript_variables/__init__.py) names
a label that [`../data/claims.json`](../data/claims.json) does not contain — the
recorders have not been run, or a new prose number was added without recording
it.

**Fix.** Run the recorders for the stages involved, then regenerate:

```bash
.venv/bin/python scripts/record_stage0_claims.py
.venv/bin/python scripts/record_stage1_claims.py
.venv/bin/python scripts/record_stage2_claims.py
.venv/bin/python scripts/z_generate_manuscript_variables.py --print-only
```

**Never** resolve this by hand-writing the number into prose or by deleting the
label from `REQUIRED_CLAIMS`. Both convert a loud failure into a silent one.

If the failure names a label the derived-quantity block reads rather than one
prose references, the fix is to add it to `REQUIRED_CLAIMS` so the guard reports
it up front instead of failing later with a bare `KeyError`.

---

## `ERROR: could not locate the template repo root` — `TEMPLATE_REPO_ROOT` unset

**Symptom.** `scripts/z_generate_manuscript_variables.py` writes
`output/data/manuscript_variables.json` and then exits 1, refusing to inject.

**Cause.** Injection needs `infrastructure.rendering.manuscript_injection` from
the template repo. AILM is a **sidecar**: a standalone checkout symlinked into
`template/projects/working/`. A relative walk-up from the real path lands in
the sidecar's parent directory, which has no `infrastructure` package — so the
resolution must be told where to look. The renderer sets `TEMPLATE_REPO_ROOT`;
manual invocation does not.

**Fix.**

```bash
TEMPLATE_REPO_ROOT="$AILM_TEMPLATE_ROOT" \
  .venv/bin/python scripts/z_generate_manuscript_variables.py
```

Or skip injection when you only want to inspect tokens:

```bash
.venv/bin/python scripts/z_generate_manuscript_variables.py --print-only
```

The script fails loudly rather than emitting an un-injected manuscript, which is
correct: a silent skip produces a document full of literal `{{TOKEN}}` markers.

---

## `test_no_unbacked_prose_claims` fails

**Symptom.**

```
Unbacked prose claims: ['stage3_word_perplexity']
```

**Cause.** Prose in `README.md`, `AGENTS.md`, `CHANGELOG.md`, `docs/**/*.md`, or a
`src/**/*.py` docstring carries a `claim:<label>` tag with no ledger entry.

**Fix.** Record the measurement from the run that produced it. If the tag was a
typo, fix the tag. If the number is a *derived* token
(`{{STAGE1_UNIGRAM_MARGIN}}`, `{{STAGE1_CEILING_SHORTFALL_RATIO}}` and the other
two listed in [`syntax_guide.md`](syntax_guide.md#derived-tokens)), it has no
lowercase tag form at all — cite the component claims instead.

**Never** delete the tag to make the gate green. That is the exact failure this
gate exists to catch: it was vacuous for an entire stage because it only ever ran
against `tmp_path` fixtures while a real label sat unbacked in the real README.

```bash
.venv/bin/python -m pytest tests/test_claims_discipline.py -q
```

---

## A pymdp evaluation takes forever, or trips the 300-second test timeout

**Symptom.** A test or script hangs on a full-corpus pass. A naive full-corpus
pymdp G2P evaluation can exceed the configured per-test budget by a wide margin.

**Cause.** An eager per-position loop pays Python/backend dispatch costs. First
separate import, compilation and steady execution. The connected runtime compiles
its actual pymdp methods and preserves the same event trace; its first call may
therefore have a different cost from a warmed call.

**Wrong fixes.** Raising the timeout (the cost is real, not a configuration
artifact). Shrinking the evaluation set — which reintroduces the subsample-bias
defect and, at these margins, an underpowered test.

**Use the protocol’s existing execution path.** For connected evaluation, keep
the compiled live-agent path and its eager/compiled controls. The historical
G2P recorder instead uses a bounded pymdp/NumPy decoded-position agreement check,
then the fast oracle for full-set statistics, refusing to record if that check
fails. That check does not prove equality for every input or predictive loss.
`verify_decoder_equivalence` in
[`../scripts/record_stage2_claims.py`](../scripts/record_stage2_claims.py) is the
implementation; the reasoning is in
[`gate_design.md`](gate_design.md#when-the-full-set-is-too-slow-oracle-equivalence).

For sequence perplexity there is a second route: replace the walk with a closed
form over the count table (`full_corpus_perplexity`), tested against both
analytic anchors and a literal step-by-step walk on hand-built corpora.

For a quick classification pass while iterating, bound the audit instead of the
gate:

```bash
.venv/bin/python scripts/audit_gates.py --limit 300
```

and remember that its subsample numbers classify the gate but are **not**
quotable results.

---

## A figure fails to build: clipped, overlapping, or illegible text

**Symptom.** One of:

- `IllegibleFigureError` — type below `ailm.viz_config.MIN_FONT_PT`
- `FigureOverflowError` — an artist outside the canvas
- an assertion from
  [`../tests/test_figure_layout.py`](../tests/test_figure_layout.py) reporting
  overlapping label pairs or clipped labels with their overhang in points

**Cause.** Legibility is a **build-time invariant** here. `save_figure` refuses to
write a figure that violates the font floor, and there is deliberately **no
clamp** — a clamp is not a floor, it just makes the defect quiet.

**Fix.** Give the figure more room or shorten the text. Do **not** shrink type,
raise `OVERLAP_TOLERANCE`, or raise `CLIP_TOLERANCE_PT` to get past the check;
those constants are calibrated against rendered glyph padding, and moving them
disables the detector for every other figure too.

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_figure_layout.py -q
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py --only wiring_comparison
```

Note the completeness check: `test_discovery_covers_every_shipped_figure` binds
the number of discovered builders to the PNGs on disk. A builder named outside
the recognized prefixes, or one that takes required arguments, silently escapes
the sweep — that omission once excluded four of nine figures, two with confirmed
defects, while the sweep read as clean.

---

## `MissingVariableError` from a figure generator

**Symptom.** A figure build fails naming a token rather than a layout problem.

**Cause.** Figure generators read measured values from
`output/data/manuscript_variables.json` — no generator hardcodes a number. If the
token map is absent or stale, the generator fails instead of drawing a plausible
placeholder.

**Fix.** Regenerate the token map first, then the figures:

```bash
.venv/bin/python scripts/z_generate_manuscript_variables.py --print-only
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py
```

---

## `scripts/audit_gates.py` exits nonzero

**Symptom.** The audit prints a full report but the exit status is 1.

**Cause.** A registered outcome or control failed its declared contract. The
canonical widened Stage 5 row is checked separately from its retained case-only
ablation, whose non-passing outcome is intentional. Read both `GATE` (the scalar
comparison) and `OUTCOME` (the registered outcome/control check), plus the
population notes. The script’s exit status is not simply “every scalar score wins.”

**Fix.** Read the verdict line:

- `VACUOUS` — the trivial baseline already clears the threshold. Restate the gate
  against the baseline.
- `UNREACHABLE` — a bound-typed comparator cannot clear the threshold. Restate
  the gate, and retain the bound that licenses the impossibility claim.
- `ABOVE_REFERENCE` — an exceedable reference or privileged oracle cannot clear
  the threshold. Restate the gate without calling it impossible.
- `UNAUDITED` — no typed ceiling was supplied, so reachability is unknown.
- `BRACKETED_BOUND` or `BRACKETED_REFERENCE` with `GATE … FAIL` — the gate is
  well-formed and the model is failing it. This is the case where the outcome,
  rather than threshold placement, is the problem.

See [`gate_design.md`](gate_design.md) for both restatements this project has
already had to make.

---

## A float assertion fails by ~1e-7

**Symptom.** An exact-equality assertion fails on a value that looks right:
`38.999998 != 39.0`.

**Cause.** JAX defaults to float32, so analytic anchors land within roughly 1e-7
relative no matter how exact the algebra is.

**Fix.** Compare with a tolerance. `FLOAT32_REL = 1e-6` in
[`../tests/test_phoneme_full_corpus.py`](../tests/test_phoneme_full_corpus.py) is
the documented bound; asserting tighter than the arithmetic can deliver produces a
test that fails on correct code.

Related trap: a floor computed from the **unsmoothed** MLE table differs in the
third decimal from the add-1-smoothed version, and an all-positions perplexity is
legitimately above a transitions-only floor. A failing assertion may be comparing
two different quantities rather than reporting a numerical error — check which
pair you are comparing before widening a tolerance. See
[`gate_design.md`](gate_design.md#how-to-compute-an-information-ceiling).

---

## `FileNotFoundError`: a vendored counts artifact is missing

**Symptom.** `Brown counts not found at …/vendor/brown_counts.json` or
`Phoneme counts not found at …/vendor/cmudict_counts.json`.

**Fix.** Regenerate. Both are deterministic and byte-identical on regeneration,
so the checksums in [`provenance/sources.md`](provenance/sources.md) must still
match afterwards:

```bash
.venv/bin/python scripts/generate_counts.py
.venv/bin/python scripts/generate_phoneme_counts.py
.venv/bin/python -m pytest tests/test_provenance_checksums.py -q
```

A `ValueError` about `_schema` or a `phones` mismatch instead means the artifact
on disk was produced by a different generator version — regenerate rather than
editing the JSON, since a hand-edited artifact would silently misindex every
matrix built from it.

---

## `CacheCorruptionError: divergent reply for existing key`

**Symptom.** A network recorder stops mid-run with a `CacheCorruptionError` from
`cache/bench_calls.jsonl` naming one key. No claims are written.

**Cause.** The store refuses to replace an existing record with a different
reply. A key authenticates the request, serving configuration and recorded
identity; it does not prove the service is a pure function. Even temperature-zero
responses can depend on server history. Preserve the stored record and the
observed conflicting reply, then investigate request options, serving identity,
shared run scopes and server history. Never overwrite a response to make replay
appear deterministic.

Most calls replay from the store rather than re-asking. The exceptions are
deliberate: the external benchmark's two reliability trials per position, and
every scored call of the stipulated synthetic and in-context probes, all of
which would be no measurement at all if a memo could answer them. Those cache
namespaces are scoped by a UTC run id fixed once at recorder start plus the
probe tag and each scored call's own position slot and arm, so an ordinary
re-run writes its own keys and does not reach this refusal. If you are
seeing it on a key whose protocol ends in a run id, two runs shared a scope,
which should not be reachable through a recorder: check that the scope is
generated per run rather than pinned.

On any other key, the refusal is telling you something real: the server no
longer answers a stored question the same way. Confirm it directly, then decide
whether the store or the server is the thing that moved.

```bash
.venv/bin/python scripts/record_external_benchmark_claims.py --verify-cache 12
.venv/bin/python scripts/warm_call_cache.py --check
```

**Fix.** Preserve the offending record and failed audit. Check the selected
endpoint, model identity, serving options and producer receipt before deciding
which evidence needs a new measurement. Historical rows from another serving
configuration are ineligible, not corrupt merely because they differ. Follow
[the release checklist](method/release_checklist.md) to remeasure affected
producers and dependents under a new scope. Do not delete records, relabel a
scope, or rewrite old receipt hashes to get past the refusal.

---

## `benchmark refused: source or vendored benchmark input changed during measurement`

**Symptom.** A network recorder aborts partway through, naming the model it was
measuring. No sidecar and no claims are written.

**Cause.** Binding receipts hash the recorder, `pyproject.toml`, and the whole
`src/ailm` tree, and the recorder re-checks that manifest between models. A file
changing mid-run would leave claims bound to a source snapshot that no longer
exists, so the recorder stops instead. In a multi-agent session this usually
means another lane edited `src/` or `scripts/` while the phase was running.

**Fix.** Let the tree settle, confirm it has, then re-run the phase:

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from pathlib import Path
from ailm.replay_provenance import source_manifest
print(source_manifest(Path.cwd(), 'scripts/record_external_benchmark_claims.py')['manifest_sha256'])
"
```

Compare snapshots and coordinate a source freeze with every writer. Equal
samples show equality at those instants, not that later edits cannot occur.
Re-run only after the required inputs, endpoint and spending configuration are
fixed; the recorder’s own checks must still remain active throughout.

---

## Tests pass but nothing ran

**Symptom.** A green exit code with a suspiciously fast run.

**Cause.** A selection that collected zero tests, a wrong path, or an interpreter
without pytest installed. Ordinary pytest reports no tests collected with a
non-success status, but collection-only mode can exit successfully without
executing tests. Require the executed count, not merely collected names.

**Fix.** Always read the collected count, and check coverage separately:

```bash
.venv/bin/python -m pytest tests/ -q                      # read "N passed"
.venv/bin/python -m pytest tests/ --collect-only -q
.venv/bin/python -m pytest tests/ --cov=ailm
```

---

## Connected JSON reports `infinite_oov`

This is an explicit closed-vocabulary result. The CLI serializes the nonfinite
word loss as `null` and reports `word_loss_status: "infinite_oov"`; each token
still carries its OOV flag. Letter and phone inference continue. Do not replace
the value with zero, drop those tokens, or present an in-vocabulary diagnostic as
full-stream word performance. See [the input contract](design/connected_runtime.md)
and [the data statement](reference/data_statement.md).

## A moved checkout imports provider code from its old location

A copied virtualenv is not automatically relocatable: editable-install metadata,
console launchers and local-only project symlinks can retain old absolute paths.
Set the intended repository root explicitly, inspect `sys.executable`,
`sys.prefix` and imported module `__file__` values, then prepare or reanchor the
environment under its own locked dependency contract. Verify the template route
and `AILM_GNN_UPSTREAM` separately; source-copy equality alone does not verify
execution paths.

After a storage error such as `Device not configured` or `No space left on
device`, preserve the failed logs, cache, budget and partial scope. Verify the
selected volume and free space before restarting a writer. Use a task-scoped
`TMPDIR` on the selected storage when needed; keep it out of source and
publication inventories. Canonical fresh-call scoring starts a new authenticated
scope after recovery. An existing cache or a copied receipt does not make an
interrupted run complete.

## JAX version changes float results or breaks pymdp

**Symptom.** A previously green test fails after a dependency update, with a
traceback inside pymdp (e.g. `compute_expected_state` → `factor_dot(B_f[...,u_f])`,
an advanced-indexing call) rather than in AILM code. Or: the same suite passes on
one machine and errors on another.

**Cause.** pymdp 1.0.3's and 1.0.4's `update_empirical_prior` vmaps over `B` and indexes it
with an array slice (`B_f[..., u_f]`). Later JAX releases changed behavior on
that path. This repository's environment of record is therefore the exact
`jax==0.4.38` / `jaxlib==0.4.38` pair declared in `pyproject.toml`, alongside
`inferactively-pymdp==1.0.4`; a compatible version range is not the contract.

**Fix.** Do not let `uv sync` upgrade JAX past the pin. `pyproject.toml` declares the
combined-lock extra `cuda12 = ["jax[cuda12]==0.4.38"]`, so on a CUDA machine (linux x86_64)
one command installs the plugin set from the project's own metadata:

```bash
uv sync --locked --extra cuda12
```

then **do not** run a plain `uv sync` afterwards without `--extra cuda12` — it would restore
the CPU jaxlib. Verify the backend before running the suite:

```bash
.venv/bin/python -c "import jax; print(jax.default_backend())"   # gpu or cpu
```

A first GPU run can be slow per live-pymdp test while XLA compiles; the
project's live-walk tests carry `@pytest.mark.timeout(300)` for exactly this.
`JAX_PERSISTENT_CACHE_DIR` is set inside those tests so subsequent runs reuse the
compiled HLO.

## Apple Silicon (Metal) — measured 2026-08-04, works but do not use for receipts

`jax-metal` 0.1.1 initializes cleanly against the pinned `jax==0.4.38` on an
M4 Pro (macOS 15 / Darwin 25) with the PJRT compatibility shim:

```bash
uv pip install jax-metal          # in a THROWAWAY venv, never the project .venv
ENABLE_PJRT_COMPATIBILITY=1 JAX_PLATFORMS=METAL python -c "import jax; print(jax.devices())"
# [METAL(id=0)]   — note JAX_PLATFORMS wants the uppercase METAL
```

Measured on that machine, same commit, same corpus. The qualitative verdict is
what this page keeps; the numbers behind it are in the dated
[development-history](development_history.md) entry for that measurement (*Verified — full release receipt + rendered
artifact (2026-08-04)*), which is the record, not this page:

| Workload | Verdict |
|---|---|
| letter-level `compute_perplexity` on a fixed Brown prefix | **slower on Metal** than on CPU |
| perplexity agreement, CPU vs Metal | agree to float32 precision |
| synthetic square f32 matmul | faster on Metal — the only workload that was |
| `test_letter.py` and `test_phoneme.py` on Metal | every collected test passed |

Reading: the Metal device is real and numerically consistent at float32, but the
probe exercised only L1 (claim:stage0_letter_alphabet_size letter states) and L2
(claim:phoneme_count_39 phoneme states), both small enough that
per-step dispatch overhead dominates and inference is *slower* on the GPU. The
float64 numpy oracle is the project's reference and Metal has no float64 at all.
CPU with the locked `jax==0.4.38` therefore remains the only environment of
record; treat Metal as a curiosity that happens to pass the two tested levels,
not as a supported profile. L3's word factor already has
claim:stage3_vocab_size states × claim:complexity_l3_position_states positions
and was not run on Metal, nor were L4–L6, so the dispatch-overhead reading has
not been tested where the state spaces are large; re-run this measurement on L3
before reconsidering.

---

## See also

- [`gate_design.md`](gate_design.md) — gate verdicts, ceilings, subsample bias
- [`testing_philosophy.md`](testing_philosophy.md) — controls, oracle cross-checks, running the suite
- [`syntax_guide.md`](syntax_guide.md) — token and claim protocols
- [`agent_instructions.md`](guides/agent_instructions.md) — the command set and invariants
- [`faq.md`](guides/faq.md) — design questions rather than failures
