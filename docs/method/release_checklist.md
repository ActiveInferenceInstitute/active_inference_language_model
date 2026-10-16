# Release checklist: cutting the first public artifact

How a tree that passes its gates becomes a tagged, citable release. This file
is the release-day sequence; the mechanics of each step live where they are
maintained ([`quickstart.md`](../guides/quickstart.md) § 7 for rendering,
[`output_conventions.md`](../reference/output_conventions.md) § "Regeneration order" for the
regeneration chain, [`rendering_pipeline.md`](rendering_pipeline.md) for the
token flow), and this file only orders them and names what blocks. Every
command is run from the sidecar root with the project interpreter unless the
step says otherwise.

## What a release is, here

A local release candidate binds a frozen source commit, its subsequently
regenerated output checkpoint, exact tested distributions and a clean `full`
receipt with `publication_acceptance: true`. Independent exact-product QA and
final review complete local acceptance. A public release additionally requires
the authorized tag, push and verified archive; local acceptance does not perform
those actions. The version string appears in exactly four
files and one lock (`pyproject.toml`, `src/ailm/__init__.py`, `CITATION.cff`,
`docs/manuscript/config.yaml`, and the `ailm` entry in `uv.lock`);
`tests/test_release_orchestration.py::test_the_citation_file_stays_in_step_with_the_paper`
refuses drift among the first four and `uv lock --check` refuses a
stale lock. Nothing else in the tree carries the version.

## Select the evidence route before running commands

For a new measurement or changed numerical/model/protocol input, use the
acquisition sequence below with an explicitly approved service and spending
boundary. Recorder, aggregation, ledger-status and formal-recording commands
write evidence; they are not documentation validation commands.

For a frozen publication-only repair, preserve the original claims, call store,
protocols and binding receipts. Follow the exact independently reviewed
[publication-derivative contract](../publication_derivative.md), freeze its named
source/support delta, regenerate current outputs, and run the clean full profile
against that external contract. Do not run the acquisition or ledger-writing
steps below for this route. Review the current command inventory without running
it, then execute only the approved recipe:

```bash
.venv/bin/python -I -S scripts/verify_release.py full --list \
  --publication-derivative "$AILM_PUBLICATION_CONTRACT"
.venv/bin/python -I -S -B scripts/verify_release.py full \
  --publication-derivative "$AILM_PUBLICATION_CONTRACT" \
  --receipt "$AILM_RELEASE_RECEIPT"
```

Set `AILM_PUBLICATION_CONTRACT` to the reviewed external contract and
`AILM_RELEASE_RECEIPT` to a unique external receipt destination. Unset hosted
credentials and budget overrides. The derivative performs its declared live
local reproducibility audit, makes no fresh hosted acquisition, and retains
separate original measurement, current source and output identities. An old
receipt never certifies new bytes. Retain the distributions actually tested by
the package step before temporary rotation; a later build is not their substitute.
Independent exact-product PDF/docs QA and one final gate must accept the current
candidate before delivery. This does not authorize a tag, push or deposition.

## Preconditions

- **Quiet tree.** No other session is editing this checkout. Check
  `git status --short` (only ignored operator files may be dirty) and
  `git log -5` for commits you did not expect.
- **Clean template checkout.** The renderer runs from the sibling
  `docxology/template` checkout, and the `full` profile refuses a dirty one
  unless the development-only override is set, which can never yield
  acceptance. Use a clean checkout of the required snapshot; preserve unrelated
  work rather than resetting another owner’s tree.
- **Local verification server and warm store.** The explicitly selected
  loopback Ollama endpoint is answering, every declared probe is served, and `scripts/warm_call_cache.py
  --check` reports the required local cache coverage. Fresh acquisition in step
  2 needs a credential for the synthetic and in-context hosted producers, while
  the external, aligned-G2P and logprob-drift producers use the configured local
  endpoint. The full release verifies their completed stored evidence and makes
  its declared local cache-audit calls; it does not rerun hosted acquisition. Select `AILM_OLLAMA_ENDPOINT` before
  importing benchmark configuration and keep it unchanged across producer,
  replay, audit and release processes. Loader/context choices and endpoint are
  part of current cache eligibility, not labels that can be changed afterward.
- **Approved spending boundary.** A capped hosted run uses the prepared
  `AILM_OPENROUTER_BUDGET_FILE` and approved `AILM_OPENROUTER_BUDGET_LIMIT_USD`
  throughout fresh hosted acquisition. Supply the API key through the process
  environment or a protected operator-only secret file loaded by the acquisition
  launcher; never commit it or print it. Use the clean release invocation below
  after acquisition, with the key and budget variables unset. The current
  verifier allowlist can forward an inherited key but does not forward budget
  variables, so do not rely on it to create this separation automatically.
  A spend guard is not a scientific success criterion.
- **Stable storage and paths.** Confirm the execution checkout, canonical cache,
  template project route and environments resolve to the selected copies.
  Preserve failed/interrupted runs and their budgets; reanchor an execution copy
  before taking its new receipts. Never splice a partial fresh-call run into a
  later canonical scope.
- **Toolchain.** `lake`, `pandoc`, `xelatex`, `qpdf`, and `mmdc` on `PATH`
  (the quickstart's prerequisite table).
- **Python environment.** `quick` checks the exact base environment prepared by
  `uv sync --locked`; `full` checks the base environment plus the existing
  `gnn-interop` extra, prepared by `uv sync --locked --extra gnn-interop`.
  Switching profiles requires the corresponding sync: returning to `quick`
  removes the optional parser dependency. The release runner checks this state
  without changing it.
- **Upstream GNN checkout.** The full profile requires the real parser and the
  `gnn-interop` extra. Its controlled environment does not forward
  `AILM_GNN_UPSTREAM`; make the intended checkout discoverable through
  `_UPSTREAM_CANDIDATES` in `tests/test_gnn_conformance.py`. The first candidate
  is checkout-relative under the shared `Username` hierarchy at
  `ActiveInferenceInstitute/Public/Generalized_Notation_Notation`. Direct test
  invocations can use `AILM_GNN_UPSTREAM`, but exporting it alone does not select
  the full-profile checkout. Verify the actual resolved parser path.

## The sequence

The steps in this section are the measurement-changing acquisition route.
Select the frozen derivative route above for publication-only repairs.

1. **Prepare the final source/version before acquisition.** Finish the reviewed
   implementation and version changes in `pyproject.toml`, `src/ailm/__init__.py`,
   `CITATION.cff` and `docs/manuscript/config.yaml`; refresh `uv.lock` when
   required and verify version parity. Preserve the current truthful availability
   disclosure: a private source repository or old DOI is not proof of a public
   release. Coordinate a source freeze across owners before running fresh
   network measurements. Lint, format, typing, then the quick profile:

   ```bash
   uv sync --locked
   .venv/bin/python -m ruff check --no-cache src scripts tests
   .venv/bin/python -m ruff format --check --no-cache src scripts tests
   .venv/bin/python -m mypy
   .venv/bin/python -I -S scripts/verify_release.py quick
   ```

2. **Re-bind the network recorders.** Every binding receipt under
   `cache/network_claim_bindings/` hashes its recorder and local script-import
   closure, the whole `src/ailm` Python tree, `pyproject.toml`, `uv.lock`, and
   the vendored JSON inventory. Changes to shared manifest inputs stale every affected
   receipt; changes specific to one recorder stale its binding. This is by design: a recorder's claims are
   bound to the code that produced them. Re-run the phase, which reuses stored
   calls where the protocol permits and makes fresh calls where required,
   then run the aggregate. Each recorder replaces its own ledger namespace
   under the single-writer lock:

   ```bash
   .venv/bin/python scripts/verify_network_claim_bindings.py   # expect: stale
   .venv/bin/python scripts/orchestrate_claim_recorders.py network
   .venv/bin/python scripts/orchestrate_claim_recorders.py aggregate
   .venv/bin/python scripts/verify_network_claim_bindings.py   # expect: clean
   ```

   Run `core` before `network` only when a measurement's inputs changed; the
   core recorders re-measure and are not a no-op in wall clock.

   Canonical fresh-call requirements include the external
   benchmark's repeat-agreement trials, every scored call of the stipulated
   synthetic and in-context probes, and the log-probability drift sample.
   Synthetic and in-context scoring refuse any replayed scored call. Their
   hosted cohort labels identify repeated protocol cohorts, not distinct local
   model weights; the configured hosted model identity is disclosed separately.
   Fresh
   replies can differ from the preceding run, so rebinding is a new measurement
   and does not guarantee unchanged claim values.

   The external, synthetic, and in-context recorders scope their fresh-call
   cache protocols with a UTC run id fixed at recorder start. Their re-runs
   append new records while preserving the store's refusal to overwrite an
   existing key. These run scopes are part of the digested receipt configuration;
   the validator checks the exact declared records. The in-context recorder
   reads the synthetic producer's scope from its receipt to authenticate the
   producer's stored replies before making its own scored calls.

   The drift recorder uses a different contract: it samples authenticated stored
   log-probability calls and pairs them with fresh uncached replies. Its binding
   receipt contains those observation pairs and the digests of the sampled cache
   records and eligible universe. It does not use the other recorders' UTC cache
   namespaces. Its validator re-derives the claims from the bound pairs.

   The re-bind does need a source tree that holds still while it runs. Every
   receipt hashes the manifest described above, and the recorder re-checks
   that manifest during measurement and before publication: an edit
   landing mid-run aborts it with "source or vendored benchmark input changed
   during measurement", before a sidecar or a claim is written. Run the phase
   when nothing else is writing to `src/`, `scripts/`, or `vendor/`.
   `orchestrate_claim_recorders.py network --list` inspects the discovered phase
   without calls. The executing phase validates each producer before a dependent
   recorder starts. Rebinding means running the required measurement and
   validation; editing hashes or relabeling historical calls is not a substitute.

3. **Bind the ledger into Lean and re-record the formal layer.**

   ```bash
   .venv/bin/python scripts/generate_lean_ledger.py
   (cd formal && lake build && ./Scripts/axiom_report.sh)
   .venv/bin/python -m pytest tests/test_formal_isomorphism.py -q --no-cov
   .venv/bin/python scripts/orchestrate_claim_recorders.py formal
   .venv/bin/python scripts/generate_lean_ledger.py --check
   ```

4. **Ledger schema, gate audit, tokens, docs mirror, figures, cover, registry.**

   ```bash
   .venv/bin/python scripts/assign_claim_status.py
   .venv/bin/python scripts/audit_gates.py --json
   .venv/bin/python scripts/z_generate_manuscript_variables.py
   .venv/bin/python scripts/z_render_docs.py && .venv/bin/python scripts/z_render_docs.py --check
   MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py
   MPLBACKEND=Agg .venv/bin/python scripts/generate_cover.py
   MPLBACKEND=Agg .venv/bin/python scripts/generate_cover.py --check
   TEMPLATE_REPO_ROOT="$AILM_TEMPLATE_ROOT" .venv/bin/python scripts/generate_figure_registry.py
   TEMPLATE_REPO_ROOT="$AILM_TEMPLATE_ROOT" .venv/bin/python scripts/generate_figure_registry.py --check
   ```

5. **Review the measured and generated changes before acceptance.** Read every
   affected ledger namespace, producer receipt and report against the frozen
   protocol. Preserve null, harmful and OOV outcomes; do not select a run because
   its result looks better. Verify the connected GUM report’s complete canonical
   execution binding separately from network receipts. A changed connected
   implementation/input binding requires its actual deterministic scoring replay,
   not a new digest on old rows.

   For this acquisition route, a version or source change after step 2 invalidates affected evidence: return
   to the relevant measurement, then regenerate aggregate/status, formal,
   token, figure and manuscript dependencies. This is why version preparation
   precedes acquisition. Freeze the final inputs again before the acceptance run.

6. **Render, validate, and accept.** The template’s Stage 03 and Stage 04 commands
   run from its checkout (quickstart § 7). The full profile drives both from the
   AILM root; run this clean verification command from AILM:

   ```bash
   uv sync --locked --extra gnn-interop
   env -u OPENROUTER_API_KEY \
        -u AILM_OPENROUTER_BUDGET_FILE -u AILM_OPENROUTER_BUDGET_LIMIT_USD \
     .venv/bin/python -I -S scripts/verify_release.py full
   ```

   Read `output/reports/release-receipt.json`: `success` says the checks
   completed without changing their inputs; `publication_acceptance` is the
   verdict. A receipt with the dirty override set is never acceptance.

7. **Commit the regenerated publication tree** by explicit path, in the
   established form `output: regenerate the publication tree against <sha>`,
   where `<sha>` is the commit the tree was rendered from. Confirm the final
   tracked tree and receipt still agree, and have the principal review the
   rendered PDF at its intended reading size: title/availability, main results,
   tables, figures, references and appendices. A successful compiler and
   structural PDF check do not establish visual legibility. Only after the
   reviewed release identity is fixed and tagging is authorized, set
   `AILM_RELEASE_VERSION` to that version and tag:

   ```bash
   git tag -a "v$AILM_RELEASE_VERSION" -m "AILM $AILM_RELEASE_VERSION"
   ```

8. **Verify publication identity before updating availability.** The current
   source remote is the private `docxology/active_inference_language_model`
   repository. The intended Institute repository is empty. Keep the manuscript,
   `CITATION.cff`, and `[project.urls]` aligned with that observed state until
   publication actually succeeds. The existing DOI does not establish that the
   current tree is archived. After an authorized publication, verify the public
   repository's source commit and the archive's contents before changing those
   claims and rendering the final PDF. `tests/test_availability_disclosure.py`
   checks the current disclosure and its negative controls offline; it is not a
   live availability probe. `commit` and `date-released` in `CITATION.cff` stay
   absent until the corresponding tagged release has been verified.

   The public push is one squashed (orphan) commit of the reviewed tree: every
   tracked file ships, including `docs/decisions.md` and
   `docs/development_history.md`; only the private commit history is left
   behind, retained in the private development repository. Commit SHAs cited
   in `docs/decisions.md` refer to that private history.

9. **Push, deposit, and close the citation.** After explicit publication
   authorization, push the reviewed public commit and tag. Verify the intended
   archival deposition in the authenticated Zenodo account before uploading the
   tagged archive and combined PDF. The formerly configured identifier has no
   verified public record; do not assume a reservation or private draft exists.
   Publish the deposition only with the principal's authorization, verify the
   public record and exact deposited artifacts, then set the same verified DOI
   in `CITATION.cff`, `publication.doi`, and package URLs. Publishing is
   irreversible and is the principal's call. In a follow-up commit, fill
   `commit` and `date-released` in `CITATION.cff` and turn the orchestration
   test's absence assertion for them into a presence assertion.

## What blocks a release

| Blocker | Where it shows | Who clears it |
|---|---|---|
| Stale strict network-claim receipts without an accepted finite derivative | `verify_network_claim_bindings.py` refuses; the `full` profile stops at its bindings step | step 2, after the last source edit |
| Dirty template checkout | the `full` profile refuses without the development override | whoever owns the template's working tree |
| Cold call store or unserved probe | `warm_call_cache.py --check` refuses; coverage falls under the floor with no failing test | `warm_call_cache.py`, with `ollama serve` running |
| Version drift or stale lock | `test_the_citation_file_stays_in_step_with_the_paper`; the `environment` release step | step 5 |
| Open principal decisions | `TODO.md` § "Principal decisions" | the principal; a release may proceed with them recorded as open, but not with them silently dropped |

## What stays the principal's

The main-text/supplement split and the funding and competing-interest
declarations are already resolved — see
[`docs/manuscript/MANUSCRIPT_STATUS.md`](../manuscript/MANUSCRIPT_STATUS.md) and
the availability-and-licensing subsection of
[`docs/manuscript/07_reproducibility.md`](../manuscript/07_reproducibility.md) —
and the `ud_ewt_test` replication was decided and executed under decision 111,
without establishing a current public archive. Outstanding decisions are recorded
in `TODO.md` §
"Principal decisions". Resolve or explicitly defer them with their scope intact;
a technical check does not decide authorship, disclosures or publication consent.


Each release execution records a unique `run_id`; step logs live beneath
`logs/<run_id>/` beside the requested receipt. Multiple receipts in the same
folder, or rerunning a receipt filename, retain their distinct log files. Keep a
copy of an earlier receipt too if rerunning the same receipt path: that JSON path
still names the latest run.
