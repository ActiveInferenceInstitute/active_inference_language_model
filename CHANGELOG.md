# Changelog

All notable changes to AILM are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Semantic versioning.

The full decision-by-decision development record predating this release lives
in [`docs/development_history.md`](docs/development_history.md); the numbered
rationale behind each decision is in [`docs/decisions.md`](docs/decisions.md).

---

## [Unreleased]

### Added

- The aligned-G2P zero-shot probe arms are reported in the external-benchmark
  supplement: each scoring arm's words scored, position and exact-word
  accuracy, and contract compliance, read against the analytic contract
  floor, plus the smollm arm's recorded contract refusal. The labels join
  `REQUIRED_CLAIMS` and render through `BENCH_SYNTHG2P_PROBE_*_PCT` tokens.
  This integrates the 2026-09-23 to 2026-09-25 release-checkout branch
  (`salvage/2026-09-30/home-ailm-release-main`); its ledger, outputs and
  decision text were superseded on `main` and were not taken.

### Fixed

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
  `handoff/reports/rc_review_2026-09-30/`.
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
