# `docs/manuscript/` — the AILM paper

The Markdown sources for the AILM paper (title, version, and authors come from
[`config.yaml`](config.yaml)). These files are templates: they carry
double-brace token markers resolved by a strict registered map at render time;
measured inputs come from the claims ledger,
so the files here are not the document a reader sees — `output/manuscript/*.md` is.

Agents editing this directory must read [`AGENTS.md`](AGENTS.md) first.
[`SYNTAX.md`](SYNTAX.md) is the cross-reference registry: every section, equation,
figure, table, and token label, with the commands to re-derive the lists.

## Sections

The renderer assembles sections in filename order; the numeric prefix therefore
owns publication order, and `99_` keeps the bibliography last. The table below
is a thematic inventory for readers, not a second declaration of assembly order.

| File | What it is |
|---|---|
| `00_abstract.md` | The model, connected GUM comparison, historical component evidence and gate-audit contribution |
| `01_introduction.md` | What AILM is, what it is not, and why a fully explicit language system is worth building |
| `02_background.md` | Active inference in discrete state spaces; the four matrices; filtering; perplexity and the empirical order-1 optimum; CEREBRUM cases. Defines the equations the rest of the paper references |
| `03_methods.md` | The level stack; the typed-comparator VACUOUS / UNREACHABLE / ABOVE_REFERENCE / UNAUDITED / BRACKETED_BOUND / BRACKETED_REFERENCE taxonomy; the construction of each modeled level and of the generation path; the cross-level interventions and their hypothesis; the verification apparatus. The re-runnable audit script and the pre-build checklist moved to the audit-trails supplement, which the gate-design section points to |
| `04_results.md` | The connected GUM comparison first, followed by the historical recorded outcomes: each level's measurements, the generation sweep, the channel results, the learning and verification outcomes, the information budget, and what the numbers do not establish. The gate-audit defect findings, the subsample finding, the Level 3 mis-wiring catalog, and the per-member held-back replication detail are stated verdict-level here and carried in full in the audit-trails supplement; the exploratory channel-factorial and proper-loss re-scoring extension is summarized here in a few sentences and carried in full in its own supplement |
| `05_discussion.md` | The limitations, stated without hedging; the stratificational choice the widened clause arm sizes but does not make; and what the held-back replication changes |
| `06_related_work.md` | Active inference and pymdp, HMMs, n-gram and letter-entropy work, G2P, mechanistic interpretability, CEREBRUM |
| `07_reproducibility.md` | The exact reproduction path, vendored artifacts with checksums, and the token mechanism; the Lean layer is summarized in a paragraph here and carried in full in its supplement |
| `08_conclusion.md` | What was established, what the gate methodology generalizes to, and the next step |
| `98_supplement_audit_trails.md` | Supplement. The gate-audit trails: the re-runnable audit script, the pre-build checklist, the defect findings worked through stage by stage, the Level 3 mis-wiring catalog and scaling ladder, and the per-member confirmatory replication detail. Moved out of `03_methods.md` and `04_results.md`, which state the verdicts and point here |
| `98_supplement_channel_factorial.md` | Supplement. The channel factorial crossing message source with insertion point, the combination-rule control, the fitted variant, and the re-scoring of every cell on the receiver's proper predictive loss. Moved out of `04_results.md`, which keeps a short summary and points here |
| `98_supplement_complexity.md` | Supplement. Exact operation and byte counts per level |
| `98_supplement_external_benchmarks.md` | Supplement. The external reference-frame probes and their protocol conditions, plus the probe-versus-comparator verdict carried in from the results close. Rendered after the body: the `98` prefix is what puts it there, and no primary result depends on it |
| `98_supplement_lean.md` | Supplement. The machine-checked Lean layer and which parts of the argument it proves. Moved out of `07_reproducibility.md`, which keeps a one-paragraph summary and points here |
| `98_supplement_statistics.md` | Supplement. The paired test's derivation and assumption audit, the multiplicity family under both corrections, and the bounds argument for the nulls. Moved out of the verification apparatus in `03_methods.md`, which states the verdicts in a page and points here for the reasoning |
| `99_references.md` | Labeled heading only; Pandoc inserts the bibliography here |

Supporting files, none of which is a section:

| File | Role |
|---|---|
| `config.yaml` | Paper metadata — title, version, authors, keywords. Source of the `META_*` tokens; also points at the cover image |
| `cover.png` | The generated cover image `config.yaml` references. Written by `scripts/generate_cover.py --out docs/manuscript/cover.png`; never hand-edited |
| `preamble.md` | LaTeX packages and theorem environments. **The LaTeX must stay inside its ```` ```latex ```` fence** or it is silently dropped |
| `references.bib` | BibTeX bibliography. Unconfirmed fields are omitted, never guessed |
| `layer_contract.yaml` | Declares which project modules may import the template `infrastructure` package. Empty by design |
| `MANUSCRIPT_STATUS.md` | Publication-readiness of this directory's content, and the principal decisions still open (not rendered) |
| `SYNTAX.md` | Cross-reference and token registry (not rendered) |
| `AGENTS.md` | Agent editing contract (not rendered) |
| `README.md` | This file (not rendered) |

## Rendering

From the AILM project root, hydrate tokens and confirm none are unresolved:

```bash
.venv/bin/python scripts/z_generate_manuscript_variables.py
! rg -n -F '{{' output/manuscript/
```

Figures must exist before rendering — every PNG named by
`output/figures/figure_registry.json` must be present, and a missing one is a
broken image in the PDF:

```bash
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py
```

Then render from the template repository root, with the project addressed by its
qualified sidecar name:

```bash
cd "$TEMPLATE_REPO_ROOT"
uv run --locked python scripts/pipeline/stage_03_render.py --project working/active_inference_language_model
```

The renderer invokes `scripts/z_generate_manuscript_variables.py` itself before
assembling, using the project's own `.venv` and passing `TEMPLATE_REPO_ROOT`, so the
hydration step above is a check rather than a prerequisite. It reads
`output/manuscript/` when that directory contains Markdown, which is why editing
`output/manuscript/*.md` directly is always wrong — the next hydration overwrites it.

Read the resulting PDF before believing it. A single pass can leave `??` in place of
cross-references, and mangled math inside a theorem environment does not show up in
extracted text — look at the page.

## Bibliography rendering

The bibliography source is `references.bib`. The numbered references section
contains only its heading; Pandoc inserts the bibliography there using `--natbib`.
Write citations as `[@key]` in authored sections. Keep renderer instructions in
this non-rendered guide so they do not appear in the paper.

## Adding a section

1. Pick a numeric prefix that places it correctly. Renumbering an existing file
   changes the section order for everything after it.
2. Give the H1 a `{#sec:…}` label. Every H1 in this manuscript has one so that
   cross-references survive reordering.
3. Write numbers as tokens only. If the number is not in `data/claims.json`, record
   it from a measured run first — see [`AGENTS.md`](AGENTS.md).
4. Add any new equation, figure, or table labels to [`SYNTAX.md`](SYNTAX.md), using
   the re-derivation commands at the top of that file.
5. Add citations to `references.bib` only if every field can be confirmed; omit the
   fields that cannot.
6. Render and read the PDF.

## What ledger consistency checks establish

Measured inputs live in `data/claims.json`, each recorded by its owning run.
Validated producers also calculate derived display values and read declared
protocol, metadata and reference fields. `src/ailm/manuscript_variables/` turns that
ledger into the token map by uppercasing each label, and raises `MissingClaimError`
rather than emitting a default when a required claim is absent. Decision-log
metadata come from the validated v2 registry in `data/manuscript_literals.yaml`;
the search cutoff and priority policy come from the bounded
`data/literature_search.json` record; deposited-source fields and executable
protocol values come directly from their canonical JSON/code registries.
`src/ailm/manuscript_contract.py` then rejects
hard-coded result/configuration values and proves that the resolved tree is the
exact token-free substitution. These contracts catch the value forms they recognize; they do not prove that
arbitrary unsupported prose or a wrongly selected token is impossible.

That guarantee is narrow and worth stating precisely: it makes a number in the
manuscript consistent with the ledger. It does not make the ledger's number right.
That is what the controls, the oracle, and the gate audit in
the verification apparatus and the gate-audit taxonomy in `03_methods.md` are
for.
