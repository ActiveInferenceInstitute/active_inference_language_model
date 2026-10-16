# AGENTS.md — Active Inference Language Model

Operational reference for AI agents and contributors working in this repository.
See `README.md` for the project overview, `docs/hierarchy.md` for the governing
level structure, and `docs/design/architecture.md` for the design rationale.

---

## Repository Identity

- **Project:** `active_inference_language_model` (AILM)
- **Repository root:** the checkout root. A maintainer may route a `docxology/template` checkout to this standalone repository through the local-only symlink `projects/working/active_inference_language_model/`; every command in this tree runs from the repository root unless it says otherwise
- **Publication repository:** `https://github.com/ActiveInferenceInstitute/Active_Inference_Language_Model` hosts the versioned publication source. The separate `docxology/active_inference_language_model` remote is retained for private development. Keep commits in the selected standalone checkout, never `docxology/template`; publishing or changing remotes requires explicit authorization.
- **Picking this project up:** start with the governing [`docs/hierarchy.md`](docs/hierarchy.md) and the chronological [`docs/decisions.md`](docs/decisions.md). `HANDOFF.md` and `ISA.md` are historical planning records, not current status sources
- **Current stage:** Stage 6 (Stages 0–6 implemented; the six linguistic levels are L1–L6. Stage 5's canonical gate is the widened-observation clause arm of decision 112, promoted by decision 114: it passes as a development-session specification and has no held-back replication, while the case-only arm it displaced is a retained ablation whose gate stays non-passing with its paired privileged-label gain unresolved. Stage 6's clustered primary spans zero on the development population and excludes zero on the held-back split under decision 115, at the development coverage and not at the family-corrected one; four EWT members of the frozen significance family were rescored once on the held-back test split under decision 111 with two crossing and two yielding no discordant pair, and the Stage 7 speech boundary contract is enforced.)

---

## Invariants (Never Violate)

| Invariant | Why |
|-----------|-----|
| **No mocks** (`MagicMock`, `mocker.patch`, `unittest.mock`) | All tests use real pymdp, real corpora |
| **No transformers / no neural networks** | AILM is entirely a pymdp generative model |
| **No learned embeddings** | Implemented semantics uses named discrete pymdp states and kernels; no DisCoCat meaning functor is claimed |
| **Every claim tagged** `claim:<label>` | Backed by `data/claims.json` entry from a measured run |
| **Every gate has positive AND negative control** | `GateVacuousError` if negative control passes |
| **claim:stage0_coverage_floor_pct% test coverage minimum** | Enforced by `pyproject.toml` |
| **39 canonical ARPAbet phonemes** | Strict; comment stripping required — see `cmudict.py` |
| **`CaseRole.GEN` is linguistic Genitive** | The source also has core model-role `CORE_GENITIVE` and a distinct novel `NOVEL_GENERATIVE`; both source labels are `[GEN]`, so scoped serialization is mandatory |
| **`A_dependencies` explicit** | Each modality must name the factors it observes; at L3 the spelling likelihood depends on both word and position, while the two `B` transitions remain separate |
| **`update_empirical_prior` is within-level temporal recursion** | NOT a channel between levels — `docs/hierarchy.md` governs. Learning is `infer_parameters`, never `update_A` |
| **`formal/AILM/LedgerData.lean` is generated, never hand-edited** | `scripts/generate_lean_ledger.py` is its only writer; `tests/test_formal_isomorphism.py` requires byte-identical regeneration. See `docs/formal.md` |

**Machine-checked layer:** the gate taxonomy, sweep-shape predicates, filtering
correction, and Bonferroni arithmetic are proved in Lean 4 under `formal/` —
[`docs/formal.md`](docs/formal.md) for what is and is not formalized, the
Lean ↔ equation ↔ Python ↔ test isomorphism, and the regenerate/re-record
commands. Changing a recorded value that the Lean reads means
`scripts/generate_lean_ledger.py` **then** `scripts/record_formal_claims.py`.

---

## Stage Gates

| Stage | Name | Gate Condition | Status |
|-------|------|----------------|--------|
| **0** | Infrastructure | All smoke tests pass; claims ledger works; control harness live | ✅ PASSED |
| **1** | Letters | Brown bigram **transition**; beats the context-free unigram baseline (claim:stage1_letter_perplexity vs claim:stage1_unigram_baseline) | ✅ PASSED — RETROACTIVE RESTATEMENT |
| **2** | Phonemes | (A) phoneme bigram perplexity beats the context-free unigram bar; (B) position-aligned letter-to-phone decoding (`g2p` in code) beats the per-letter `argmax` bar with a significant paired McNemar test | ✅ PASSED — RETROACTIVE RESTATEMENT |
| **3** | Words | Trie-constrained word paths; letter-bigram bar on identical positions (restated 2026-07-30, see decision 14) | ✅ PASSED — RETROACTIVE RESTATEMENT |
| **4** | Cases | 8-case assignment on UD EWT dev; gold morphology oracle vs derived oracle (restated 2026-07-30, see decision 16) | ✅ PASSED — RETROACTIVE RESTATEMENT |
| **5** | Clauses | Canonical arm (decisions 112, 114): joint sentence-type × widened-symbol model over case roles plus `VERB` and `AUX`; exact closed-form mixture score; gate registered before the artifact existed and passing, paired privileged-label gain resolved away from zero, detectability control and both negative controls live. Retained case-only ablation (decisions 83, 85): gate non-passing, paired privileged-label gain unresolved, detectability control live; replicated non-passing with zero discordant pairs on the held-back UD EWT test split (decision 111) | 🟨 PASSING — DEVELOPMENT-SESSION GATE, NO HELD-BACK REPLICATION |
| **6** | Narrative | Document-position (tercile) temporal structure — `narrative`/`phase` in code — intervenes on clause priors; preregistered positive point estimate, with a document-clustered primary interval that spans zero on the development population (decisions 25-26 and their correction); the sentence-level observed-position sign test crosses the frozen family threshold on the held-back UD EWT test split (decision 111), and the document-level primary estimand, registered separately under decision 115, excludes zero there while spanning zero at the family-corrected coverage | 🟨 POSITIVE POINT ESTIMATE — PRIMARY REPLICATED ON THE HELD-BACK SPLIT, NOT AT THE CORRECTED COVERAGE |
| **7** | Speech | Whisper STT → AILM phoneme input; pymdp phoneme output → TTS (boundary interface contract only) | 🔷 INTERFACE BOUNDARY |

**Prospective rule:** Stage N+1 cannot start until Stage N gate passes. The
historical build did not consistently follow this rule.

**Note on retroactive validation:** The original Stage 1-4 gates were restated
after those levels had been built or measured because the original thresholds
were vacuous, unreachable, or named the wrong population. Later stage
declarations were also re-scoped during audit. The repaired gates and controls
are more rigorous, but they are retroactive re-certification, not evidence that
the historical build proceeded sequentially. Stage 6 predictions were committed
before their held-out measurement, yet the development population and analysis
family were reused; its document-clustered primary result on that population is
therefore reported as spanning zero, and it is not retracted by the held-back
result that followed. The one partition nothing had scored, the UD EWT test
split, was spent under decision 111 on the four family members computed on
EWT populations, and touched once more under decision 115 for the separately
registered Stage 6 document-level estimand; the other twelve members remain development-population
inference. See `docs/decisions.md` for the full chronology. This is a
methodological defect, not a design choice.

### Stage 1 gate restatement (2026-07-29)

The Stage 1 gate was originally specified as "beats uniform by more than
claim:stage1_declared_gate_nats nats." That
target is **information-theoretically unreachable for any letter bigram model** —
the conditional entropy of the Brown bigram distribution caps the achievable
gain at claim:stage1_max_bigram_gain_nats nats, and the agent realizes
claim:stage1_gain_over_uniform_nats of it
(claim:stage1_bigram_signal_captured_pct% of what the order allows). The gate is
therefore stated against the **context-free unigram baseline**, which is both
reachable and diagnostic: a model can beat uniform while using no context at
all, but it cannot beat unigram letter frequencies without using the previous
letter. A gain of at least claim:stage1_declared_gate_nats nats is a
*trigram-or-higher* target — do not carry the old
wording into Stage 2's phoneme gate.

### Stage 2 gate restatement (2026-07-29)

Stage 2 was originally specified with a fixed percentage target on the English
EWT oracle. Wrong twice:

1. **UD EWT has no phonemic annotation.** Its CoNLL-U fields are
   `lemma`/`upos`/`feats`/`deprel` — the line was copied from Stage 4's case gate.
   A phoneme task cannot be scored on a syntactic treebank.
2. **The fixed target is vacuous.** A context-free per-letter lookup table scores
   claim:stage2_g2p_baseline_accuracy on held-out CMUdict — no transition matrix,
   no pymdp, no inference. The stated gate would pass a model doing nothing
   hierarchical.

Note this is the **inverse** of the Stage 1 defect: Stage 1's target was
unreachable, Stage 2's was too easy. Same detection method — measure the ceiling
*and* the trivial baseline before accepting a threshold.

### Stage 3 gate restatement (2026-07-30)

"Perplexity < 500 on PTB-1k" was **VACUOUS**: the letter-bigram bar itself is
claim:stage3_bigram_baseline, clearing 500 by claim:stage3_declared_gate_slack
times. It also named a corpus this project does not have — the full PTB is
LDC-licensed and `nltk` ships only a small sample. Stage 3 is measured on Brown, already
vendored and checksummed. The gate is now the letter-bigram bar on identical
positions; see [`docs/decisions.md`](docs/decisions.md) entry 14.

**The Stage 3 lesson, and it sharpens the Stage 2 one.** Five mis-wirings were
built and scored; *three of five* clear the restated bar, including one that
inverts the frequency ranking so the rarest words are predicted most confidently.
Of the two the bar does catch, one — end-of-word emitted a position early —
scores infinite perplexity and is therefore trivially caught, so the bar's real
discrimination is thinner than the raw count suggests. When a level's
mechanism is very informative, a baseline comparison stops discriminating
*precisely because the level is good*. What pins Level 3 is a closed-form path
identity, not the bar.

### Stage 4 gate restatement (2026-07-30)

The declared claim:stage4_declared_gate% threshold against the UD case oracle
fails in **both directions at once**, and which
one you observe depends only on the population it is scored against: VACUOUS on
gold morphology (a constant majority predictor scores
claim:stage4_gold_majority%) and ABOVE_REFERENCE on the derived oracle (every
measured, exceedable comparator falls below it). A threshold without a
named population is not a threshold. See entry 16.

**Stage 4 warning, still in force:** the 8-case UD oracle is a *research claim,
not ground truth* — only NOM/ACC/GEN are morphological in English;
claim:stage4_cases_without_morphology of the eight are derived from `deprel`.
Agreement with it is agreement, not accuracy. The heuristic itself agrees with
real morphology only claim:stage4_heuristic_gold_agreement_pct% of the time on
the morphologically marked overlap. That overlap contains
claim:stage4_heuristic_gold_adposition_token_tokens token-level adposition-route
examples and therefore does not bound the full derived population (decision 84).

**The Stage 4 trap, which generalizes.** The case level must never observe
`deprel` or `feats[Case]`, because the evaluation labels are derived from them. A
model reading those fields scores near-perfectly while measuring nothing — and
unlike every mis-wiring below it, it looks *excellent* rather than broken, so
there is no numeric signature and the guard must be structural. Wherever labels
are derived from features rather than annotated independently, check the
disjointness in code before trusting a good number.

---

## Claims Ledger Protocol

Every quantitative claim in README, docstrings, or manuscript must:
1. Be tagged inline: `claim:<label>` (e.g. `"claim:phoneme_count_39 phonemes"`)
2. Have an entry in `data/claims.json` recorded from an actual measured run
3. Be scannable: `ClaimsLedger.scan_prose(*paths)` returns unbacked labels

When writing prose with numbers:
```python
ledger = load_ledger(BASE_DIR)
count = build_phoneme_inventory(cmudict)
ledger.record("phoneme_count_39", len(count), run_id="stage0_cmudict", source=__file__)
```
Then in prose: `"The canonical ARPAbet has claim:phoneme_count_39 phonemes."`

---

## Control Harness Protocol

Every metric gate needs:
```python
harness = ControlHarness("my_gate")
# 1. Verify gate is live (positive and negative controls differ)
harness.assert_gate_is_live(good_metric, bad_metric, "lower_is_better")
# 2. Verify broken input actually fails
harness.verify_negative_control_fails(metric_fn, broken_input, threshold, direction)
```
If `GateVacuousError` is raised, the gate cannot discriminate and Stage N is **blocked**.

---

## Test Commands

IMPORTANT: Address the project's own interpreter explicitly — `.venv/bin/python -m
pytest`. Never `uv run pytest`, and never `uv run` at all inside this sidecar. AILM
is symlinked into a `docxology/template` checkout and is not a workspace member of
it, so `uv run` can bind a different virtualenv and a bare `pytest` can pick up a
shim: the suite still runs, against a different Python and a different pymdp than
the one this project pins. Both failure modes are silent. The rationale is worked
through in [`docs/guides/quickstart.md`](docs/guides/quickstart.md) § 2.

```bash
# Generated, ignored interface artifacts are mandatory for the full suite.
.venv/bin/python scripts/export_web_model.py
.venv/bin/python scripts/build_web_ailm.py

# All tests
.venv/bin/python -m pytest tests/ -v --tb=short

# Stage 0 gate only
.venv/bin/python -m pytest tests/test_stage0_gate.py -v

# With coverage (the pyproject floor, claim:stage0_coverage_floor_pct%, applies whenever --cov is passed)
.venv/bin/python -m pytest tests/ --cov=ailm --cov-report=term-missing

# Specific module
.venv/bin/python -m pytest tests/test_cmudict.py -v
.venv/bin/python -m pytest tests/test_pymdp_smoke.py -v

# Single test
.venv/bin/python -m pytest tests/test_stage0_gate.py::TestStage0PymdpSmoke::test_all_pymdp_smoke_tests_pass -v
```

Parallel dev loop (`pytest-xdist`); the suite default and the release profiles in
`scripts/verify_release.py` stay serial — the release receipt parser expects exactly
one coverage TOTAL row, so parallelism is a development-loop convenience, not a
release mode.

```bash
# Parallel development loop. Use -n 4 instead of -n auto when other pytest fleets
# share the machine or drive (a 1-minute load of 5-13 is normal here).
.venv/bin/python -m pytest tests/ -n auto -q --no-cov

# The marker exclusions work identically under -n
.venv/bin/python -m pytest tests/ -n auto -q --no-cov \
  -m "not recorder_replay and not raw_corpus_regeneration and not requires_cuda12"
```

With `--cov`, xdist workers combine coverage automatically and the pyproject floor
still applies.

---

## Source Layout

Top-level shape only. **The module inventory is owned by one file** —
[`docs/architecture.md`](docs/architecture.md)'s package map states what each
module owns, its principal dependencies, and which test exercises it; its import
contract separately states the enforced dependency boundaries. Duplicating a
tree here is how it goes stale, so this section deliberately stops at the
directory level.

```
src/ailm/          reusable logic: levels/, verification/, figures/, corpus/,
                   generative/, gnn/, benchmarks/, dashboard/, plus generation.py,
                   learning.py, gates.py, statistics.py, complexity.py,
                   spec_agent.py, manuscript_variables.py, and the rest
scripts/           orchestration and artifact writers, plus bounded
                   recorder-specific estimands: record_*_claims.py, generate_*.py,
                   audit_gates.py, verify_release.py
tests/             the gates (test_stage0..6_gate.py) and the unit tests
data/              claims ledger plus bounded registration, literature-search,
                   manuscript-literal, figure-alt-text, learning-agreement, and
                   reference-source evidence records
vendor/            committed, checksummed corpus artifacts — no test hits the
                   network; plus the one held-back raw EWT test partition
gnn/               models/ (one spec per shipped model), mutations/ (deliberate
                   defects that must FAIL), SPEC.md, and this directory's own docs
formal/            the Lean 4 layer and its generated ledger bridge
docs/manuscript/        the authored sections
docs/              this project's documentation; docs/README.md indexes it
handoff/           ledger snapshots taken before recorded passes, and
                   handoff/reports/ for dated lane reports and review snapshots
                   (provenance only; outside the prose gates, never a status source)
cache/             the memoized external-model call store and its claim bindings.
                   An INPUT, not an output: gitignored and machine-specific, but
                   the release profile's coverage floor depends on it
output/            entirely regenerable; stable publication surfaces tracked,
                   transient render products ignored
```

Two directories exist on disk and are outside the project tree proper: `refs/` holds
external repositories cloned for reference (gitignored, each with its own remote),
and `scratchpad/` holds untracked run logs and one-off working files. Nothing in
either is an input to a recorded result.

Per-package `README.md` and `AGENTS.md` live beside the code — read those before
editing inside a package.

---

## Dependency Stack

Regenerate this table from `[project].dependencies` and the `dev` group in
`pyproject.toml`; those pins are authoritative and this is a reading of them.

| Library | Role | Version |
|---------|------|---------|
| `inferactively-pymdp` | The ONLY generative model | ==1.0.4 |
| `jax`, `jaxlib` | pymdp's array backend; the pymdp-verified line is pinned exactly | ==0.4.38 |
| `numpy` | Array math | ≥1.24,<2.4 |
| `scipy` | `scipy.stats.binomtest` — the exact binomial behind every McNemar test | ≥1.10 |
| `requests` | CMUdict / UD EWT download | ≥2.28 |
| `nltk` | Brown corpus access for the vendored count generators | ≥3.8 |
| `pyyaml` | Reads `docs/manuscript/config.yaml` for the `META_*` tokens | ≥6.0 |
| `matplotlib` | Figure generation; declared rather than inherited from pymdp | ≥3.7 |
| `cycler` | Figure palette cycling | ≥0.11 |
| `hatchling` | Exact, locked wheel and sdist build backend | ==1.32.0 |
| `pytest` | Test runner | ==9.1.1 |
| `pytest-cov` | Coverage | ==7.1.0 |
| `pytest-timeout` | The per-test 300 s bound in `pyproject.toml` | ==2.4.0 |
| `pytest-xdist` | Parallel test workers for the dev loop; the suite default and the release profiles stay serial | ==3.8.0 |
| `mypy` | Typing gate over all of `src/ailm`, figures included (no package is excluded) | ==2.2.0 |
| `ruff` | Lint and format gate | ==0.15.20 |

Optional extras: `cuda12` (the CUDA JAX plugin) and `gnn-interop` (`psutil`; it
exists only to enable the upstream GNN parser interoperability test, which skips
cleanly when the GNN checkout is absent). The `full` release profile requires
`gnn-interop`: `scripts/verify_release.py` verifies the environment with
`uv sync --check --extra gnn-interop` rather than installing it, so install the
extra beforehand; `quick` checks the base set.

Install: `uv sync` (uses `pyproject.toml`)

---

## What NOT to Do

- **Never add a transformer, neural net, or learned embedding** — hard invariant
- **Never use `update_A` or `update_B`** — use `infer_parameters`
- **Never hardcode phoneme counts without running the inventory** — use `build_phoneme_inventory`
- **Never write a number in prose without a `claim:` tag** — the ledger will catch it
- **Never gate on a single value** — always prove the gate is live with positive and negative controls
- **Never skip Stage N gate to start Stage N+1** — even if it seems obvious

---

## Paper References

| Paper | Role in AILM |
|-------|-------------|
| CEREBRUM (Friedman 2025; concept DOI `10.5281/zenodo.15170907`) | Zenodo record v1.4 / deposited title-page v1.2; core and novel case ontology, structurally distinct case functionals, Supplement 8.1 POMDP tuple |
| Compositional Approaches to Linguistic Case for Cognitive Modeling (Friedman 2026; concept DOI `10.5281/zenodo.19695259`) | Zenodo-record identity for a deposited PDF titled *Cognitive Diagrams: Reviewing Categorical Accounts of Linguistic Case*; enriched category formalism, belief update equation, DisCoCat structure |

**CEREBRUM critical note:** The complete deposited PDF
(`CEREBRUM_1-4_April-16-2025.pdf`) is the engineering source; the shorter
`full_text.md` export omits later supplements. Figure 13 is caption-only and is
not evidence for a component crosswalk. Exact source identities, digests, and
physical-page locators are pinned in `data/reference_sources.json`.

*The two blocks below are optional local tooling integrations (Groma and
Backlog.md); both are inert if their CLIs are not installed.*

<!-- groma:start -->
## Groma

This project uses Groma. Before you scan, inspect, or curate architecture, or change files for a Backlog task, run `groma agent-instructions` and read the guide it names for that job. When it reports a first scan, ask the user whether they want you to curate the architecture. Do not edit Groma-owned architecture files directly.
<!-- groma:end -->

<!-- BACKLOG.MD GUIDELINES START -->
<!-- backlog.md-instructions-version: 1.53.0 -->
<CRITICAL_INSTRUCTION>

## Backlog.md Workflow

This project uses Backlog.md for task and project management.

**At the beginning of each conversation in this project, run `backlog instructions overview` before answering or taking action. Re-read it only if you have not read it yet in the current conversation.**

Use the overview to decide whether to search, read, create, or update Backlog tasks.

Before task lifecycle actions, read the matching detailed guide:
- `backlog instructions task-creation` before creating or splitting tasks
- `backlog instructions task-execution` before planning, changing status or assignee, adding a plan or implementation notes, or implementing task work
- `backlog instructions task-finalization` before checking acceptance criteria, writing final summaries, or moving tasks to terminal statuses

Use `backlog <command> --help` before running unfamiliar commands. Help shows options, fields, and examples.

Do not edit Backlog task, draft, document, decision, or milestone markdown files directly. Use the `backlog` CLI so metadata, relationships, and history stay consistent.

</CRITICAL_INSTRUCTION>
<!-- BACKLOG.MD GUIDELINES END -->
