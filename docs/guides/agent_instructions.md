# Agent Instructions

The operational contract for anyone — human or agent — modifying this repository.
[`../AGENTS.md`](../../AGENTS.md) is the short reference at the repository root; this
file is the working procedure and the read order behind it.

Read this before touching anything else.

---

## Rule 1: read order

Each document gates a category of action. Skipping one has a specific, observed
cost.

| Read this | Before | Skip consequence |
|---|---|---|
| **This file** | any modification | every risk below |
| [Governing hierarchy](../hierarchy.md) and [decisions](../decisions.md) | changing model scope, channels, or a reported outcome | confusing historical isolated experiments with the connected runtime |
| [`gate_design.md`](../gate_design.md) | specifying or changing a gate; starting a stage | shipping a fourth vacuous or unreachable gate |
| [`testing_philosophy.md`](../testing_philosophy.md) | writing or changing a test | a green suite that constrains nothing |
| [`syntax_guide.md`](../syntax_guide.md) | writing any number in prose or the manuscript | a hand-typed number, or an unbacked `claim:` tag turning the suite red |
| [`style_guide.md`](../method/style_guide.md) | writing prose or source | overstated claims, ambiguous units, wrong import layer |
| [`design/architecture.md`](../design/architecture.md) | touching a level's matrices | a wiring error pymdp cannot detect |
| [`design/stage2_g2p.md`](../design/stage2_g2p.md) | touching the G2P level | reintroducing the subsample bias |
| [`provenance/sources.md`](../provenance/sources.md) | touching a corpus artifact | a checksum that drifts from the bytes it documents |
| [`troubleshooting.md`](../troubleshooting.md) | debugging | hours on the wrong interpreter |

---

## Rule 2: invariants that are never negotiable

| Invariant | Why |
|---|---|
| **No mocks** — no `MagicMock`, `unittest.mock`, `mocker.patch` | Every test runs real pymdp against real vendored corpora. A mocked `Agent` makes the project's central claim false while the suite stays green |
| **No transformers, no neural networks, no learned embeddings** | The generative engine is pymdp, entirely; stochastic-kernel composition is implemented, but no DisCoCat meaning functor is claimed |
| **`infer_parameters` for learning** — never `update_A` / `update_B` | pymdp 1.0.4's API |
| **`update_empirical_prior` is within-level temporal recursion only** | It is the second half of filtering inside one agent, never a channel between levels. The connected runtime transports messages on explicit event clocks; historical isolated channel experiments retain their own populations and outcomes. `docs/hierarchy.md` governs |
| **Sequence dynamics in the transition `B`; emission in `A`** | The inverse is anti-informative and pymdp validates it anyway |
| **`A_dependencies` factorization** | Without it the word-level `A` is physically impossible to store |
| **The canonical ARPAbet inventory is claim:phoneme_count_39 phones** | Naive parsing yields claim:phoneme_count_naive_unstripped by capturing `#` comment tokens |
| **`CaseRole.GEN` is linguistic Genitive** | CEREBRUM separately uses `[GEN]` for the core model-role `CORE_GENITIVE` and novel output-generating `NOVEL_GENERATIVE`; scoped serialization preserves both |
| **Every quantitative claim is tagged and backed** | `claim:<label>` in prose, `{{TOKEN}}` in the manuscript, both from `data/claims.json` |
| **Every gate has a positive AND a negative control** | `GateVacuousError` blocks the stage if the negative control passes |
| **`src/ailm/verification/oracle.py` imports neither pymdp nor jax** | An auditor sharing code with its subject inherits the flaw it audits for |
| **`src/` imports no `infrastructure`** | `src/` stays standalone and copy-pasteable |
| **Coverage floor** — `fail_under` in `pyproject.toml`, published as claim:stage0_coverage_floor_pct% | Enforced; the ledger reads the value from `pyproject.toml`, so a floor change is followed by `scripts/record_stage0_claims.py` |
| **Stage N+1 does not start until Stage N's gate passes** | The Stage 1 wiring bug would otherwise have become the phoneme level's foundation |
| **Commits go to `docxology/active_inference_language_model`** | **Never** to `docxology/template` |

---

## Rule 3: the command set

The canonical interpreter is the project's own `.venv`. Everything below assumes:

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root
```

### Tests

```bash
# The full suite requires the generated, ignored web artifacts; build them first.
.venv/bin/python scripts/export_web_model.py
.venv/bin/python scripts/build_web_ailm.py

.venv/bin/python -m pytest tests/ -q                                     # full suite
.venv/bin/python -m pytest tests/ --cov=ailm --cov-report=term-missing
.venv/bin/python -m pytest tests/test_stage2_gate.py -v                  # one gate
.venv/bin/python -m pytest tests/test_claims_discipline.py -q            # the prose gate
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_figure_layout.py -q # figure layout
```

Use `.venv/bin/python` for every AILM execution. `uv run` inside this
sidecar can select another checkout’s environment, independently of whether
the cache is writable. Use `uv sync --locked` only to prepare the intended
environment; the template checkout has its own interpreter contract. See
[`troubleshooting.md`](../troubleshooting.md#bare-uv-run-pytest-resolves-to-the-wrong-python-and-a-different-pymdp).

### Gate audit

```bash
.venv/bin/python scripts/audit_gates.py                # full registered outcome and control checks
.venv/bin/python scripts/audit_gates.py --limit 300    # faster, classification only
```

Its subsample figures classify gates; they are **not** quotable results.

### Claims and manuscript tokens

```bash
# Every recorder, discovered live and run in producer→consumer order, one phase at a time.
.venv/bin/python scripts/orchestrate_claim_recorders.py core
.venv/bin/python scripts/orchestrate_claim_recorders.py network
.venv/bin/python scripts/orchestrate_claim_recorders.py aggregate
.venv/bin/python scripts/orchestrate_claim_recorders.py formal
# One recorder on its own, when only its namespace changed.
.venv/bin/python scripts/record_stage2_claims.py
.venv/bin/python scripts/z_generate_manuscript_variables.py --print-only
TEMPLATE_REPO_ROOT="$AILM_TEMPLATE_ROOT" \
  .venv/bin/python scripts/z_generate_manuscript_variables.py     # with injection
```

### Corpus artifacts

```bash
.venv/bin/python scripts/generate_counts.py                # vendor/brown_counts.json
.venv/bin/python scripts/generate_phoneme_counts.py         # vendor/cmudict_counts.json
.venv/bin/python -m pytest tests/test_provenance_checksums.py -q
```

Both artifacts are byte-identical on regeneration; if a checksum moves, something
is wrong with the generator, not with the checksum.

### Figures

```bash
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py --list
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py --only perplexity_ladder
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py
```

### Environment sanity

```bash
.venv/bin/python -c "import sys; print(sys.version)"           # expect 3.12.x
.venv/bin/python -c "import pymdp, inspect; print(inspect.getfile(pymdp))"
```

The per-test default is owned by `[tool.pytest.ini_options]` in
`pyproject.toml`; individual integration tests may declare larger bounds. Read
the current setting before diagnosing a timeout. The connected runtime uses
cached compilation of real pymdp methods; its eager mode is a numerical control.

---

## Rule 4: what never to do

- **Never write a number in prose without a `claim:` tag** (or a `{{TOKEN}}` in
  the manuscript). If no ledger entry exists, record it or report it as a missing
  claim — do not invent one, do not write a literal.
- **Never delete a `claim:` tag to make the scanner green.** Record the
  measurement.
- **Never lower a threshold to make a failing gate pass.** Measure the baseline
  and the ceiling, then restate the gate *as a correction*, in
  [`../CHANGELOG.md`](../../CHANGELOG.md), with the original wording and why it was
  wrong.
- **Never weaken an assertion to silence `GateVacuousError`.** It means the
  negative control passed. Fix the metric or the threshold; the stage is blocked
  until it fails properly.
- **Never gate on a single value.** A comparison against a baseline is necessary
  and not sufficient — one of Level 2's three mis-wirings clears the
  context-free bar while being wrong.
- **Never trust a subsample without checking what the data is sorted by.** A
  deterministic prefix over alphabetically sorted pairs is reproducible and
  badly biased.
- **Never quote a subsample number as a result.** Ledger entries are the record.
- **Never edit a vendored artifact by hand.** Regenerate it.
- **Never hardcode a measured number in a figure generator.** Read it from the
  token map.
- **Never raise a legibility or layout tolerance to get a figure past the check.**
  Give the figure more room.
- **Never suppress a generator, recorder, or validator failure** with
  `2>/dev/null || true`. A swallowed exit code turns a broken step into a silent
  no-op.
- **Never skip a stage gate**, even when the next step seems obvious.
- **Report each arm with its population and limits.** The canonical widened
  Stage 5 arm, retained case-only ablation, Stage 6 development and held-back
  analyses, and connected GUM comparison have distinct evidence. Use
  [the hierarchy](../hierarchy.md) rather than carrying a historical verdict
  forward as the status of every arm. Stage 7 is a boundary contract.
- **Never report agreement with a research-claim oracle as accuracy.**
- **Never commit to `docxology/template`.**

---

## Rule 5: verification checklist

Choose checks for the changed surface and preserve their commands, exit codes,
and artifacts. The commands below are a release-level inventory, not a reason
to launch another full suite during a shared run. Assign a single owner for
heavy tests, recorders, and rendering; a documentation edit needs the lightweight
link/claim checks in [docs/AGENTS.md](../AGENTS.md), while the release owner
runs the complete [release checklist](../method/release_checklist.md).

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root

# 1. Suite plus coverage floor, after the ignored web artifacts the suite requires.
#    Read the collected count — exit 0 with zero collected is not a pass.
.venv/bin/python scripts/export_web_model.py
.venv/bin/python scripts/build_web_ailm.py
.venv/bin/python -m pytest tests/ --cov=ailm -q

# 2. No mocks anywhere. Shell negation makes any match fail the checklist.
! rg -n "unittest\.mock|MagicMock|mocker\.patch|@patch|create_autospec" \
  src/ tests/ scripts/ --glob '*.py'

# 3. The oracle is independent of what it audits.
! rg -n "^(from|import) (pymdp|jax)" src/ailm/verification/oracle.py

# 4. src/ stays standalone.
! rg -n "^(from|import) infrastructure" src/

# 5. Every claim tag in real prose is backed.
.venv/bin/python -m pytest tests/test_claims_discipline.py -q

# 6. Every registered outcome and control is internally consistent, including retained null/negative arms.
.venv/bin/python scripts/audit_gates.py

# 7. Documented checksums match the artifacts on disk.
.venv/bin/python -m pytest tests/test_provenance_checksums.py -q
```

If a change touched the manuscript or figures, add:

```bash
.venv/bin/python scripts/z_generate_manuscript_variables.py --print-only
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_figure_layout.py -q
```

---

## Rule 6: when you find a defect in something already shipped

The project’s history contains repeated findings of this kind; their record matters
as much as how they were fixed.

1. **Reproduce it as a number**, not as an argument. The Stage 1 wiring bug became
   real when the implementation-independent reference decoder (the oracle)
   disagreed materially with pymdp's perplexity.
2. **Add the control that would have caught it**, and confirm it fails on the old
   behavior. A fix without that control leaves the class of error open.
3. **Record the finding in [`../CHANGELOG.md`](../../CHANGELOG.md) as a correction** —
   the original wording, why it was wrong, what replaced it. Do not rewrite it into
   a decision that was made deliberately.
4. **Ask whether the same class exists elsewhere.** The Stage 1 gate defect
   (unreachable) and the Stage 2 gate defect (vacuous) are opposites found by the
   same procedure; Stage 2's Part A subsample bias was the Part B bias recurring.
   A defect found once is a hypothesis about the rest of the tree.
5. **Update the affected docs in the same change.** A correction that lives only
   in the changelog leaves the wrong statement in the document a reader opens.

---

## Cross-references

- [`../AGENTS.md`](../../AGENTS.md) — repository-root reference: stage gate table, gate restatements
- [`../README.md`](../../README.md) — scope statement and roadmap
- [`gate_design.md`](../gate_design.md) — the centerpiece; read before any gate work
- [`forking_guide.md`](forking_guide.md) — the procedure for adding a level
- [`testing_philosophy.md`](../testing_philosophy.md) · [`style_guide.md`](../method/style_guide.md) · [`syntax_guide.md`](../syntax_guide.md) · [`troubleshooting.md`](../troubleshooting.md) · [`faq.md`](faq.md)
