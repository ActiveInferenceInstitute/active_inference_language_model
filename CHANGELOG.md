# Changelog

All notable changes to AILM are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Semantic versioning.

The full decision-by-decision development record predating this release lives
in [`docs/development_history.md`](docs/development_history.md); the numbered
rationale behind each decision is in [`docs/decisions.md`](docs/decisions.md).

---

## [Unreleased]

### Added

- A released upstream GNN pin in `gnn/upstream.json` and offline source-custody
  verification before parser interoperability. The full profile records and
  forwards `AILM_GNN_UPSTREAM`, checks its release identity, and refuses modified
  source even when Git index flags hide changes. This integration does not add
  a runtime dependency on the upstream pipeline or on FEP_Formal.

- A configuration reference (`docs/reference/configuration.md`): the
  `LevelConfig` hyperparameters and every `AILM_*` environment variable the
  shipped code reads, each with its reader, meaning and unset behavior,
  grouped by library runtime, recorders, dashboard, release verifier and
  formal layer.
- The aligned-G2P zero-shot probe arms are reported in the external-benchmark
  supplement: each scoring arm's words scored, position and exact-word
  accuracy, and contract compliance, read against the analytic contract
  floor, plus the smollm arm's recorded contract refusal. The labels join
  `REQUIRED_CLAIMS` and render through `BENCH_SYNTHG2P_PROBE_*_PCT` tokens.
  This integrates the 2026-09-23 to 2026-09-25 release-checkout branch
  (`salvage/2026-09-30/home-ailm-release-main`); its ledger, outputs and
  decision text were superseded on `main` and were not taken.

- The dashboard's chat transcript can be exported as a local JSON download:
  every turn's text and, for each reply, what the analysis read the preceding
  user turn as, the turn index and the rollout seed. The export is built from
  the page's own store and sends nothing to the server, which a test holds it
  to. The manuscript's dashboard figure was recaptured against the changed
  page with `scripts/capture_dashboard.py`, a new scripted procedure: one
  headless-Chrome chat turn against the live dashboard, the raster, and the
  receipt binding it to every served asset, assembled by the stdlib-only
  `ailm.dashboard.capture` module that the figure generator now shares, so the
  bindings a reader checks are the ones the tool wrote. The capture was
  previously a manual procedure with no tool in the tree.
- The location of the `docxology/template` checkout is configurable per
  machine: a gitignored `.template-root` file at the repository root, read by
  `ailm.template_root` for manuscript hydration and the figure registry and
  restated in the release verifier, stands in for exporting
  `TEMPLATE_REPO_ROOT` before every provider-dependent command. The
  environment value still comes first, and both are final rather than
  fallbacks.

### Changed

- Every conformance suite constructs the typed `LiveConformanceInputs`
  record directly instead of splatting keyword arguments into
  `check_conformance`. The keyword adapter itself stays only because the
  connected report's execution inventory binds `gnn/conformance.py` byte for
  byte; its removal is queued in `TODO.md` for the next re-bind.
- The upstream GeneralizedNotationNotation parser that the interoperability
  tests use is discovered at the checkout's sibling namespace, then beside the
  checkout, or through `AILM_GNN_UPSTREAM`. A home-directory checkout and a
  scratch probe, two machine-specific locations from the 2026-09 development
  history, are no longer searched; nothing under a temporary root ever is.
- `ailm.benchmarks.fitted` exports `validate_current_registry`, which reads the
  live registered probe keys and tags, and a test now exercises it; the two
  stream helpers nothing called are removed.
- Documentation describes the tree as it is: the output inventory names the
  release cache's home under `.venv/release-cache/` instead of two old
  `output/` cache paths no writer uses, the caption-standard guide states that
  the allowlist reached empty and was removed rather than that it may only
  shrink, and the benchmark constants name the Stage 1 passage binding as the
  benchmark's own alias rather than a compatibility layer.

- The dashboard's reply line counts turns from one, as its header does, and
  discloses how many part-of-speech tags the lexicon defaulted; its Level 4
  tag names the panel's eight UD-derived case roles (CEREBRUM names) instead
  of a Sanskrit case ontology, matching the project's statement that these
  are syntactic case roles rather than grammatical case; and the panels' bar
  colors are stylesheet classes rather than style text injected by the script.
- The confirmatory recorder's fixture-mode root is `AILM_CONFIRMATORY_ROOT`;
  it shared the name `AILM_PROJECT_ROOT` with the figure generators' project
  root, two unrelated meanings under one variable, each validated
  differently.
- Source, scripts and tests use the Python 3.11 standard annotations and
  imports (`list[...]`, `collections.abc`, `functools.cache`) that pyupgrade
  prescribes, except the seven modules and two reviewed test fixtures whose
  bytes the retained connected-GUM report and the publication derivative bind;
  those, and the lint gate's rule set (`pyproject.toml` is a non-URL
  configuration input of the same report), wait for that report's next
  scoring replay.
- The `quick` release profile runs its test step on `pytest-xdist` workers,
  half the host's cores and at most four, cutting that step from about forty
  minutes to about eleven on the recording workstation with no change in
  selection or outcome; a three-core hosted runner stays serial, because four
  workers there made the negative-control subprocess exceed its timeout.
  `full` stays serial because its coverage receipt needs one TOTAL row and its
  recorder replay shares one ledger lock.
- The README's verification section leads with the portable `quick` profile
  that any clone can run before the workstation-only `full` profile.
- Hosted CI runs the quick profile on an Apple-silicon macOS runner, the
  architecture the numeric environment was recorded on, so the suite's
  machine-architecture provenance check and the published-cover byte check are
  exercised there instead of failing by design on an `x86_64` Linux runner.
  The job also installs ripgrep, which the formal axiom report and its
  vacuity control require, and pins the receipt-upload action to its current
  release.

### Removed

- The dashboard's unused `build_unigram_a` re-export and `MAX_REQUEST_BYTES`
  alias. Active callers use `build_letter_transition` and `DashboardConfig`.
  Measured-source adapters stay bound to their original evidence until their
  required scoring replay; historical decisions and controls remain intact.

- `scripts/generate_case_counts_test.py`, the decision-79 preparation twin of
  the Stage 4 case generator. Decision 111's confirmatory run read the whole
  UD English EWT test split through `generate_confirmatory_ewt_counts.py`, so
  the twin had nothing left to prepare; its row said "retired-pending" and
  nothing imported or ran it. The decision record keeps the account of what it
  did. `precheck_confirmatory_replication.py` stays: it exits 1 by design and
  still refuses a second pass over the spent population, which its tests hold
  it to.

### Fixed

- The dashboard CLI restores Python's interrupt handler when a launcher has
  inherited an ignored `SIGINT`. A real exec-based control reproduced the
  server ignoring the interrupt before the fix. Worker completion and interrupted
  startup also share an idempotent slot release: the former double release
  replaced `KeyboardInterrupt` with a semaphore error that the HTTP server
  caught, leaving it running. A synchronized real-thread/signal control catches
  that race. Embedded server callers keep their own signal policy.

- Availability identifies the Zenodo DOI as reserved for an unpublished
  deposition, separately from the public GitHub source. The abstract names
  the existing release repository directly. Older acceptance receipts remain
  tied to their original source and do not certify the later mainline.

- The protocol companion delivers the HTML provider's own assets (scripts,
  styles, fonts and icons) beside the pages that reference them instead of
  refusing them as unregistered context sources, so a `docxology/template`
  improvement such as its shared web-font loader cannot fail an AILM release
  on its own. Each referenced asset must still be an ordinary file inside the
  package, escaping or decorated paths keep the context rules, and context
  documents remain registry-bound, with controls for each.
- The poisoned-PATH trusted-Git control keeps the loader variables
  (`DYLD_INSERT_LIBRARIES`, `LD_PRELOAD`) out of its baseline run, as it
  already did for the child under test: hosted macOS runners abort a process
  whose insert library is missing, which failed the control's positive half
  on the first Apple-silicon CI run for a reason unrelated to the poison.
- The trusted-Git lookup accepts the merged-`/usr` directory alias
  (`/bin` as a symlink to `usr/bin`) that Debian-family hosts ship and that
  Python's default path lists first; it previously refused every such host
  before any object read. An alias resolving outside the default path and a
  symlinked `git` entry remain refused, with controls for each.
- The derived-versus-recorded paired-power rendering test and the published
  cover check assert byte identity only on the recording architecture and
  report a cross-architecture last-bit difference as such, instead of as a
  formatter defect or a stale cover.
- The held-out-stream recorder's namespace-ownership control carries the
  raw-corpus replay marker, because the recorder reproduces the vendored stream
  from the pinned Brown archive before writing; the full profile's raw-corpus
  step now runs that file.
- The virtual-environment startup control reads the standard-library
  `sitecustomize` that Debian-family interpreters ship ahead of site-packages
  instead of expecting a shadowed planted module to execute.
- The release-orchestration fixture repositories set `core.excludesFile` to
  the null device, so a developer's user-level Git ignore file cannot keep the
  planted interpreter out of the fixture commit that the runner, which reads
  Git without user configuration, then refuses as dirty.
- The held-out-stream artifact reproduction control skips with the shared
  offline reason when the pinned Brown archive is absent, like the other
  raw-corpus controls, instead of failing with a raw traceback.
- `AGENTS.md`, this changelog and the quickstart prerequisites state that
  `handoff/` is retained in the private development repository and that
  ripgrep is a toolchain prerequisite.
- `ollama_available` allows a cold server's model-directory scan to finish
  before its first `/api/tags` answer, instead of reading a healthy endpoint
  as absent.
- Float protocol fields bind to the executable registry within a tolerance
  on the probability scale (decision 131). The macOS 27.0.1 Accelerate update
  moved the two target-start TVDs in their last bits and stopped
  manuscript-variable generation; integer, boolean and string fields stay
  bit-exact.
- The aligned-G2P control test snapshots the ledger digest at import instead
  of pinning an absolute digest that broke on every ledger commit.
- A release-candidate review of the hydrated manuscript, in independent
  section reviews, corrected internal contradictions, stale status
  statements, the level-stack table's shifted L4 row, the abstract's
  lexical-gain wording, wrong decision and count references, and wording
  defects across the body and supplements. The findings, the applied and
  deferred log, and the decisions left to the maintainer are archived in
  `handoff/reports/rc_review_2026-09-30/` of the private development
  repository; the public source distribution ships without `handoff/`.
- The inference-evidence figure labels its L5 panel as the case-only
  ablation's type-free bar instead of a closed-form mixture score.
- The wheel-resource test pins its build interpreter, so a user-level uv
  Python pin cannot redirect the locked offline source-distribution rebuild.
- `docs/troubleshooting.md` links `AGENTS.md` inside the repository.
- Documentation brought in line with the tree: five network recorders, the
  configured per-test timeout, run-scoped cache namespaces, the
  `fig:case_word_sweep` row of the figure registry, a token instead of a
  spelled count in the methods' channel sentence, and the P2 probe-arm status
  in `TODO.md`, `docs/README.md` and `docs/extension_protocols.md`.

## [1.0.0] — 2026-09-27

Prepared private version; no public release or archive is established by this
entry. Current availability is stated in [README.md](README.md#citation).
AILM is a research suite of six discrete
probabilistic language levels — letter, phoneme, word, case, clause, and
document position — each a real `pymdp` generative model built from named
conditional probability tables, with no transformer, neural network, or
learned embedding anywhere in the stack. Every reported number resolves from
a single-writer claims ledger, and every stage gate is audited against a
trivial baseline, a typed comparator, and deliberately broken variants.

### Added

- The L1–L6 modeled level stack, plus a Stage 7 speech boundary interface
  contract, each with a recorded gate outcome; see the stage table in
  [`README.md`](README.md#stage-roadmap) and [`AGENTS.md`](AGENTS.md) for the
  per-stage verdicts.
- A gate-audit method: `ailm.control.ControlHarness` requires a live positive
  and negative control before any gate counts as passing.
- A claims-ledger discipline: `ailm.claims.ClaimsLedger` verifies every
  `claim:<label>` in project prose against a recorded, sourced measurement
  and flags high-signal bare numbers that bypass it.
- A rendered manuscript (`docs/manuscript/`, built to `output/pdf/`) whose
  quantitative claims are hydrated from the same ledger at render time.
- A GNN (Generalized Notation Notation) specification for every modeled
  level under `gnn/models/`, machine-checked against the code that builds
  each level's generative model.
- Measured cross-level receiver interventions (letter→phone, word→phone,
  word→letter generation, document-position→clause, and a privileged
  punctuation-type oracle→word arm), each a separate recorded experiment
  rather than an assumed message-passing channel.
- A confirmatory replication protocol: a frozen family of pre-registered
  comparisons re-scored exactly once against a held-back split, recorded in
  [`docs/decisions.md`](docs/decisions.md).

### Changed

- The clause level (L5) moved to a widened observation alphabet as its
  canonical form; the case-only model it displaced is retained as a
  documented ablation rather than removed.
- The external-probe comparison arm moved to a hosted transport, with the
  comparator switch and its bounds recorded as a decision rather than a
  silent re-record.

### Fixed

- Numerous restated stage gates, corrected populations, and re-scoped
  declarations are recorded as corrections in
  [`docs/decisions.md`](docs/decisions.md) rather than retired silently; the
  displaced value is never deleted, only superseded in place.

See [`docs/development_history.md`](docs/development_history.md) for the
complete pre-release history this section summarizes.
